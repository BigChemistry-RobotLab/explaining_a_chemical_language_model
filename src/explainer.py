
""" adapted from ref: https://arxiv.org/abs/2103.15679 """

import torch
import torch.nn as nn
from rdkit.Chem import Draw
from rdkit import Chem

from rdkit import rdBase
from rdkit.Chem import AllChem
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


class MolecularSelfAttentionViz():
    """ apply self-attention update rule only """

    def __init__(self, n_layers=6, save_heatmap=False):
        self.n_layers = n_layers
        self.save_heatmap = save_heatmap

    def avg_heads(self, attn, grad):
        """ identical, increased readability """
        attn = grad.permute(0, 2, 1) * attn.permute(0, 2, 1)
        return attn.clamp(min=0).mean(dim=0)

    def agg_relevance(self, attn, grad, ml, token=None):
        # init relevancy matrix as identity
        rel = torch.eye(ml, ml)

        # cast to float32 for torch.clamp
        attn, grad = attn.float(), grad.float()
        
        # loop through each layer
        for layer in range(self.n_layers):
            # calculate avg over heads using attn & gradient
            attn_map = self.avg_heads(
                attn[layer, :, :ml, :ml], 
                grad[layer, :, :ml, :ml]
            ).cpu().detach()
            # apply update rule
            a_bar = torch.matmul(attn_map, rel) 
            rel = rel + a_bar

            # optionally save heatmaps
            if self.save_heatmap:
                save_heat(a_bar, ml, token, prefix='a_bar_l{layer}')
        if self.save_heatmap:
            save_heat(rel - torch.eye(ml), ml, token, f'full_rel')
        return rel

    def get_weights(self, rel, ml):
        """ extract weights from <REG> token importance """
        # apply mask and remove I diagonal
        rel = rel[:ml, :ml] - torch.eye(ml, ml)
        # extract column, drop <REG> token itself
        return np.array(rel[1:, 0])

    def __call__(self, attn, grad, mask, token=None):
        # get mask length
        ml = sum(mask)
        # aggregate relevance matrix R
        rel = self.agg_relevance(attn, grad, ml, token)
        # extract token importance as relevance weights to <REG> token
        return self.get_weights(rel, ml)

class ColorMapper():
    def __init__(self, cmap=None, vmin=None, vmax=None):
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.atoms = ['C', 'c', 'O', 'o', 'N', 'B', 'Br', 'F', 'S', 'Cl', 'P',
            '[P]', 'I', 'n', '[n]', 's', '[s]', '[S]',  '[P+]', '[B]', '[N+]',
            '[O-]', '[#6]', '[#7]', '[C@H]', '[C@]', '[C@@]', '[C@@H]', '[nH]',
            '[NH]', '[NH0]', '[SH0]', '[H]', '[N]',
            # logP specific below
            '[13C]', '[CH2]', '[CH2+]', '[3H]', '[13C@@H]', '[13CH2]', '[I-]',
            '[Cl+]', '[CH+]', '[CH]', '[2H]', '[11CH]', '[35Cl]', '[P+]',
            '[NH3+]', '[NH2+]', '[CH2-]', '[C-]', '[S+]', '[13CH3]', '[IH+]',
            '[OH+]', '[S-]', '[PH3+]', '[O]', '[OH2+]', '[C]', '[C+]', '[CH-]',
            '[N-]', '[P-]', '[PH+]', '[13CH]', '[18F]', '[n+]'] 
        self.nonatoms = ['-', '=', '#', '@', '[', ']', '(', ')', ':', '/',
            '\\', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '|']

        self.cmap = sns.light_palette("green", reverse=False, as_cmap=True)
        # TODO validate vmin, vmax scaling works (_autoscale gets called)
        self.norm = Normalize(self.vmin, self.vmax)

    def filter_atoms(self, weight, token):
        ''' filter out non-atom tokens '''
        return [weight[i] for i, t in enumerate(token) if t in self.atoms]

    def __call__(self, weight, token):
        ''' filter relevance weights for atom-only tokens
            and scale by its own min/max'''
        self.norm = Normalize(self.vmin, self.vmax)
        return self.norm(self.filter_atoms(weight, token))
   
    def to_rdkit_cmap(self, weight):
        '''helper to map to shape required by RDkit to visualize '''
        return {i: [tuple(self.cmap(w))] for i, w in enumerate(weight)}

def make_legend():
    mapper = ColorMapper()
    sm = ScalarMappable(cmap=mapper.cmap, norm=None)
    sm.set_array([])  # create a scalar mappable without any data

    # Create an empty figure and add the colorbar to it
    fig, ax = plt.subplots(figsize=(1, 9))
    fig.subplots_adjust(bottom=0.2)

    cbar = fig.colorbar(sm, cax=ax, orientation='vertical',
        ticks=list(np.linspace(0, 1, 9)), label='Relative importance (a.u.)')

    # Save the colorbar as an image file
    plt.savefig(f'/workspace/results/aqueous-solu/colorbar_au.png', 
        dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def save_heat(rel, ml, token, prefix=""):
    rel = torch.flipud(rel[:ml, :ml])
    token = ['<REG>'] + token
    cmap = sns.cubehelix_palette(
        start=0.5, rot=-0.75, dark=0.15, light=0.95, reverse=True, as_cmap=True
    )
    rel *= 100 / 0.8 #scale for visibility
    ax = sns.heatmap(
        rel,
        square=True,
        xticklabels = np.array(token), 
        yticklabels = np.array(list(reversed(token))),
        cmap = cmap,
        vmin = 0., vmax = 1.
    )
    plt.tight_layout()
    plt.savefig(f'/workspace/results/aqueous-solu/{prefix}_Aqueous_heatmap.png')
    plt.clf() 
    
    ax = sns.heatmap(rel, 
        cmap = cmap, 
        cbar=False, 
        square=True,
        xticklabels = False,
        yticklabels = False, 
        vmin = 0., vmax = 1.) 
    plt.savefig(f'/workspace/results/aqueous-solu/{prefix}_Aqueous_heatmap_raw.png',
        bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_weighted_molecule(atom_colors, smiles, token, logS, pred, prefix="", savedir=""):
    atom_colors = atom_colors
    bond_colors = {}
    h_rads = {} #?
    h_lw_mult = {} #?

    label = f'Exp logS: {logS:.2f}, predicted: {pred:.2f}\n{smiles}'

    mol = Chem.MolFromSmiles(smiles)
    mol = Draw.PrepareMolForDrawing(mol)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(700, 700)
    d.drawOptions().padding = 0.0 

    # some plotting issues for 'C@@H' and 'C@H' tokens since 
    # another H atom is rendered explicitly. 
    # Might break for ultra long SMILES using |c:1:| notation
    # vocab = ft_model.cmapper.atoms + ft_model.cmapper.nonatoms
    # if int(mol.GetNumAtoms()) != len(atom_colors.keys()):
    #     print(f"Warning: {int(mol.GetNumAtoms()) - len(atom_colors.keys())}")
    #     print(f"count mismatch for {smiles}:\
    #          {[t for t in token if t  not in vocab]}")
    #     print(f'{token}')

    d.DrawMoleculeWithHighlights(
        mol, label, atom_colors, bond_colors, h_rads, h_lw_mult, -1
    )
    # todo legend
    d.FinishDrawing()
    
    with open(file=f'{savedir}/{prefix}_MolViz.png',
        mode = 'wb') as f:
        f.write(d.GetDrawingText())


# TODO
# draw full molecule in the corner
# draw ecfp4 fragments for whole dataset 

# draw aggregated ECFP4 fragments, aggregated & weighted by
# regression head weights for each (?top-n) fragments
# https://github.com/rdkit/rdkit/blob/0e7871dc5ed7a690cae9607de0ce866e49a886b4/rdkit/Chem/Draw/__init__.py#L736
#submol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, atomsToUse=atomsToUse))

# TODO OURS
# split regression head backprop into [512] features,
# obtain individual attribution weight

# show sum(per-feature-attrib) == total-feature-attrib


