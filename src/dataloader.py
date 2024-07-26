import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import List
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
)

class PropertyDataset(Dataset):
    def __init__(self, subset, file_path, smilesname, propname,
                 split, split_frac, n_splits=5, data_seed=42,
                 scale=False, augment=False):
        self.subset = subset
        self.split = split
        self.split_frac = split_frac
        self.n_splits = n_splits
        self.smilesname = smilesname
        self.propname = propname
        self.data_seed = data_seed
        self.scale = scale
        self.augment = augment
        print(smilesname, propname)
        print(split, n_splits, 'splits')

        # split data into accurate test set according to SolProp
        df = pd.read_csv(file_path)

        print('n unique', len(np.unique(df[self.smilesname])))
        print('len df', len(df))
        uni, cnt = np.unique(df[self.smilesname], return_counts=True)
        # uncount = list(zip(uni, cnt))
        # print(sorted(uncount, key=lambda x: x[1], reverse=True)[:50])

        df = self.custom_preprocess(df)
        self.min = df[propname].min()
        self.max = df[propname].max()
        print('min', self.min, 'max', self.max)

        splitter = ShuffleSplit(
            n_splits=5,
            test_size=1.-self.split_frac,
            random_state=self.data_seed
        )
        if split == 'scaffold':
            test_splitter = MurckoScaffoldSplitter(
                smiles=df[smilesname].to_list(),
                n_splits=5,
                test_size=1.-self.split_frac,
                seed=self.data_seed
            )
        else:
            test_splitter = splitter

        # split into train+val/test,
        if split == 'accurate' or split == 'custom':
            test_idx = self.custom_split(df)
            _sanity = None
        else:
            _sanity, test_idx = next(
                test_splitter.split(df[smilesname].to_list()))
            print(min(test_idx), max(test_idx))
            print(min(_sanity), max(_sanity))
            print(len(df))

        if _sanity is not None:
            assert len(_sanity)+len(test_idx) == len(df)
            sane_train = df.iloc[_sanity, :].reset_index(drop=True)
            sane_test = df.iloc[test_idx].reset_index(drop=True)

        # set aside test set
        test_df = df.iloc[test_idx].reset_index(drop=True)
        tr_va_df = df.drop(test_idx).reset_index(drop=True)

        if self.scale:
            self.scaler = RobustScaler(quantile_range=[10, 90]).fit(
                np.expand_dims(np.array(tr_va_df[self.propname]), -1))

            print('scaler params')
            paramdict = self.scaler.get_params()
            print(paramdict)
            print(self.scaler.center_)
            print(self.scaler.scale_)
            # json.dumps(paramdict)
            # self.scaler = MinMaxScaler().fit(
            #     np.expand_dims(np.array(tr_va_df[self.propname]), -1))

            # self.scaler = QuantileTransformer(
            #     output_distribution='uniform'  # 'normal'
            # ).fit(
            #     np.expand_dims(np.array(tr_va_df[self.propname]), -1))

            tr_va_df[self.propname] = self.scaler.transform(
                np.expand_dims(np.array(tr_va_df[self.propname]), -1))
            test_df[self.propname] = self.scaler.transform(
                np.expand_dims(np.array(test_df[self.propname]), -1))

        self.tr_va_df = tr_va_df

        # sanity check: assert train/test non-overlapping
        if _sanity is not None:
            assert tr_va_df.shape == sane_train.shape
            print(len(set(df[smilesname].to_list())))
            print(len(set(sane_train[smilesname].to_list())))
            print(len(set(sane_test[smilesname].to_list())))
            assert len(set(df[smilesname].to_list()) ^
                   set(sane_train[smilesname].to_list()) ^
                   set(sane_test[smilesname].to_list())) == 0
            assert len(set(tr_va_df[smilesname].to_list()) ^
                       set(sane_train[smilesname].to_list())) == 0
            assert len(set(test_df[smilesname].to_list()) ^
                       set(sane_test[smilesname].to_list())) == 0

        if self.subset == 'test':
            self.data = DataSplit(
                    smiles=test_df[smilesname].to_list(),
                    labels=torch.tensor(
                        test_df[propname].to_list(),
                        dtype=torch.float32),
                    subset=self.subset
                )
        else:
            # split remaining df into train/val
            self.tr_va_splitter = splitter.split(
                tr_va_df[smilesname].to_list())

            self.data = []
            for fold in range(self.n_splits):
                train_idx, val_idx = next(self.tr_va_splitter)
                assert len(train_idx) > len(val_idx)
                if self.subset == 'train':
                    df = self.tr_va_df.iloc[train_idx]
                elif self.subset == 'valid':
                    df = self.tr_va_df.iloc[val_idx]

                self.data.append(
                    DataSplit(
                        smiles=df[smilesname].to_list(),
                        labels=torch.tensor(
                            df[propname].to_list(),
                            dtype=torch.float16),
                        subset=self.subset)
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.subset == 'test':
            return self.data
        else:
            return self.data[idx]

    def custom_preprocess(self, df):
        raise NotImplementedError
        # return df

    def custom_split(self, df):
        raise NotImplementedError


class DataSplit(Dataset):
    def __init__(self, smiles: List[str], labels: List[float], subset: str):
        self.smiles = smiles
        self.labels = labels
        self.subset = subset

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        data = self.smiles[idx]
        # if self.subset == 'train' and self.augment:
        #     smiles_data, _ = aug_smiles(smiles_data)
        labels = self.labels[idx]
        return data, labels


class ECFPDataSplit(DataSplit):
    def __init__(self, ds, nbits=512):
        self.smiles = ds.smiles
        self.labels = ds.labels
        self.subset = ds.subset
        self.nbits = nbits
        self.make_ecfp()

    def make_ecfp(self):
        # try:
        #     print([Chem.MolFromSmiles(smi) for smi in self.smiles])
        # except:
        #     print(smi)
        ecfp = [AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(smi), radius=2, nBits=self.nbits
               ) for smi in self.smiles]
        self.ecfp = torch.tensor(ecfp, dtype=torch.float32)

    def __getitem__(self, idx):
        data = self.ecfp[idx]
        labels = self.labels[idx]
        return data, labels


class MurckoScaffoldSplitter():
    # 10544 798 461 with k=3, seed=42
    # 7287 3590 921 with k=2, seed=42
    def __init__(self, smiles, n_splits=1, test_size=0.1, seed=42, top_k=2):
        self.n_splits = n_splits
        self.test_size = test_size,
        self.seed = seed
        self.top_k = top_k
        self.topscf = self.get_top_scaffolds(
            [self.get_murcko_scaffolds(smi) for smi in smiles]
        )

    def get_murcko_scaffolds(self, smi):
        ''' get murcko-* scaffolds from smi '''
        mol = Chem.MolFromSmiles(smi)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold, canonical=True)

    def get_top_scaffolds(self, scaffolds):
        ''' get list of most frequent murcko-* scaffolds '''
        vals, cnts = np.unique(scaffolds, return_counts=True)
        print(sorted(cnts, reverse=True)[:10])
        print(sorted(list(zip(vals, cnts)), key=lambda x: x[1],
                     reverse=True)[:10])
        # filter out scaffolds with less than 5 occurances
        return [scf for scf, cnt in list(zip(vals, cnts)) if cnt >= self.top_k]

    def split(self, smiles):
        ''' split dataset using frequent murcko-* scaffolds '''
        scaffolds = [self.get_murcko_scaffolds(smi) for smi in smiles]
        scaffolds = [s if s in self.topscf else 'rare' for s in scaffolds]
        # print('nunique', np.unique(scaffolds, return_counts=True))
        # splitter = GroupShuffleSplit(n_splits=self.n_splits,
        #                              test_size=0.1,
        #                              random_state=self.seed)
        # if self.top_k == -1:
        #     PredefinedSplit(n_splits=1,
        #                     test_size=0.1,
        #                     random_state=self.seed)
        #     scaffolds = [1 if scf == 'rare' else 0 for scf in scaffolds]
        # else:
        splitter = GroupShuffleSplit(n_splits=self.n_splits,
                                     test_size=0.1,
                                     random_state=self.seed)
        return splitter.split(smiles, groups=scaffolds)


class AqSolDataset(PropertyDataset):
    def __init__(self, subset, file_path, smilesname, propname,
                 split, split_frac, n_splits=5, data_seed=42,
                 scale=False, augment=False):
        super().__init__(subset, file_path, smilesname, propname,
                         split, split_frac, n_splits, data_seed,
                         scale, augment)
        # self.smilesname = 'smiles solute'
        # self.propname = 'logS_aq_avg'

    def custom_preprocess(self, df):
        # drop one extreme outlier (logS 6.4)
        return df[df[self.propname] < 2.05].reset_index(drop=True)

    def custom_split(self, df):
        # predefined accurate split: >2 measuments with low std
        test_idx = np.where(
            (df['count'] > 1) & (df['logS_aq_std'] < 0.2))[0]
        return test_idx


class CombiSoluDataset(Dataset):
    def __init__(self, file_path, subset, temp_test, split, augment=False,
                 data_seed=42, scale_logS=True):
        self.subset = subset
        self.temp_test = temp_test
        self.augment = augment
        self.data_seed = data_seed
        self.scale_logS = scale_logS

        df = pd.read_csv(file_path)
        df = df.dropna(axis=0, subset=[
            'temperature', 'solvent_density [kg/m3]', 'experimental_logS [mol/L]'
            ])

        test_idx = round(df['temperature']) == 298
        # set aside random test set
        if not temp_test:
            ntest = round(len(df) * (1-split))
            test_idx[:ntest] = True
            test_idx[ntest:] = False
            np.random.seed(self.data_seed)
            test_idx = np.random.permutation(test_idx)

        test_df = df[test_idx]
        df = df[~test_idx]
        total = df.shape[0]
        split_index = int(total * split)

        self.min = df['experimental_logS [mol/L]'].min()
        self.max = df['experimental_logS [mol/L]'].max()
        print(self.min, self.max)

        if self.subset == 'train':
            self.solu_smi = df['solute_smiles'][:split_index].to_list()
            self.solv_smi = df['solvent_smiles'][:split_index].to_list()

            self.temperature = torch.tensor(
                df['temperature'][:split_index].to_list(),
                dtype=torch.float16)
            self.density = torch.tensor(
                df['solvent_density [kg/m3]'][:split_index].to_list(),
                dtype=torch.float16)
            self.logS = torch.tensor(
                df['experimental_logS [mol/L]'][:split_index].to_list(),
                dtype=torch.float16)
            if self.scale_logS:
                self.logS = self.scale(self.logS)

        elif self.subset == 'valid':
            self.solu_smi = df['solute_smiles'][split_index:].to_list()
            self.solv_smi = df['solvent_smiles'][split_index:].to_list()

            self.temperature = torch.tensor(
                df['temperature'][split_index:].to_list(),
                dtype=torch.float16)
            self.density = torch.tensor(
                df['solvent_density [kg/m3]'][split_index:].to_list(),
                dtype=torch.float16)
            self.logS = torch.tensor(
                df['experimental_logS [mol/L]'][split_index:].to_list(),
                dtype=torch.float16)
            if self.scale_logS:
                self.logS = self.scale(self.logS)

        elif self.subset == 'test':
            self.solu_smi = test_df['solute_smiles'].to_list()
            self.solv_smi = test_df['solvent_smiles'].to_list()

            self.temperature = torch.tensor(
                test_df['temperature'].to_list(),
                dtype=torch.float16)
            self.density = torch.tensor(
                test_df['solvent_density [kg/m3]'].to_list(),
                dtype=torch.float16)
            self.logS = torch.tensor(
                test_df['experimental_logS [mol/L]'].to_list(),
                dtype=torch.float16)
            if self.scale_logS:
                self.logS = self.scale(self.logS)

    def scale(self, logS):
        ''' scale logS to [0, 1] '''
        return (logS - self.min) / (self.max - self.min)

    def unscale(self, logS):
        ''' scale logS back to original range '''
        return logS * (self.max - self.min) + self.min

    def __len__(self):
        return len(self.solu_smi)

    def __getitem__(self, idx):
        solu_smi = self.solu_smi[idx]
        solv_smi = self.solv_smi[idx]
        logS = self.logS[idx]
        temperature = self.temperature[idx]
        return (solu_smi, solv_smi, temperature), logS


# from nemo_chem.data.augment import MoleculeEnumeration
def aug_smiles(smiles: str, augment_data: bool = True, canonicalize_input: bool = True):
    """Regularize SMILES by coverting to RDKit mol objects and back

    Args:
        smiles (str): Input SMILES from dataset
        canonicalize_input (bool, optional): Canonicalize by default. Defaults to False.
        smiles_augmenter: Function to augment/randomize SMILES. Defaults to None
    """
    mol = Chem.MolFromSmiles(smiles)
    canon_smiles = Chem.MolToSmiles(mol, canonical=True) if canonicalize_input else smiles

    if augment_data:
        # aug_mol = self.aug(mol)
        atom_order = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_order)
        aug_mol = Chem.RenumberAtoms(mol, atom_order) # TODO how to use PySMILESutils for this

        # There is a very rare possibility that RDKit will not be able to generate
        # the SMILES for the augmented mol. In this case we just use the canonical
        # mol to generate the SMILES
        try:
            aug_smiles = Chem.MolToSmiles(aug_mol, canonical=False)
        except RuntimeError:
            # logging.info(f'Could not generate smiles for {smiles} after augmenting. Forcing canonicalization')
            aug_smiles = canon_smiles if canonicalize_input else Chem.MolToSmiles(mol, canonical=True)
    else:
        aug_smiles = Chem.MolToSmiles(mol, canonical=False)

    assert len(aug_smiles) > 0, AssertionError('Augmented SMILES string is empty')
    assert len(canon_smiles) > 0, AssertionError('Canonical SMILES string is empty')
    return aug_smiles, canon_smiles


if __name__ == '__main__':
    import pytorch_lightning as pl
    import json
    from src.dataloader import AqSolDataset

    with open('/workspace/scripts/aqueous_config.json', 'r') as f:
        cfg = json.load(f)
    pl.seed_everything(cfg['seed'])

    cfg['split'] = 'scaffold'
    print('_scaffold train val test')
    train_scaffold = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train',
        cfg['split'], cfg['split_frac'], data_seed=cfg['seed'], augment=False)
    valid_scaffold = AqSolDataset('/workspace/data/AqueousSolu.csv', 'valid',
        cfg['split'], cfg['split_frac'], data_seed=cfg['seed'], augment=False)
    test_scaffold = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test',
        cfg['split'], cfg['split_frac'], data_seed=cfg['seed'], augment=False)
    train_scaffold.next_split()
    valid_scaffold.next_split()
    print(len(train_scaffold), len(valid_scaffold), len(test_scaffold))
    for i in range(4):
        train_scaffold.next_split()
        valid_scaffold.next_split()
        print(len(train_scaffold), len(valid_scaffold), len(test_scaffold))

    cfg['split'] = 'accurate'
    print('train_accurate')
    train_accurate = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train',
        cfg['split'], cfg['split_frac'], data_seed=cfg['seed'], augment=False)

    cfg['split'] = 'random'
    print('train_random')
    train_random = AqSolDataset('/workspace/data/AqueousSolu.csv', 'valid',
        cfg['split'], cfg['split_frac'], data_seed=cfg['seed'], augment=False)

    cfg['split'] = 'scaffold'
    sanity = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train',
        cfg['split'], cfg['split_frac'], data_seed=cfg['seed'], augment=False)

    assert len(sanity.smiles) == len(train_scaffold.smiles)
    assert set(sanity.smiles) == set(train_scaffold.smiles)
    assert set(sanity.smiles) ^ set(train_scaffold.smiles) == set()

    ecfp_scaffold = AqSolECFP('/workspace/data/AqueousSolu.csv', 'train',
        'scaffold', cfg['split_frac'], data_seed=cfg['seed'], nbits=512)
    ecfp_random = AqSolECFP('/workspace/data/AqueousSolu.csv', 'valid',
        'random', cfg['split_frac'], data_seed=cfg['seed'], nbits=512)
    ecfp_accurate = AqSolECFP('/workspace/data/AqueousSolu.csv', 'test',
        'accurate', cfg['split_frac'], data_seed=cfg['seed'], nbits=512)

    train_scaffold.next_split()
    valid_scaffold.next_split()
    sanity.next_split()
    ecfp_scaffold.next_split()
    ecfp_random.next_split()

    assert train_scaffold.smiles[42] == sanity.smiles[42]
    assert ecfp_scaffold.smiles[42] == train_scaffold.smiles[42]
