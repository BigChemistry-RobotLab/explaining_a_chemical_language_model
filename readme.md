# What can Attribution Methods show us about Chemical Language Models
This repository accompanies the paper (What can Attribution Methods show us about Chemical Language Models)[https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00084f] with code to reproduce the results and apply the technique to other self-attention encoder architectures.

The repository uses is split into the following structure and uses DVC + Hydra to configure runs.
```
dvc.yaml                - dvc pipeline stages to reproduce results

src/
    model.py            - AqueousRegModel <REG> tokenizer
    dataloader.py       - AqueousSolu Dataloaders (SolProp) & splitting
    explainer.py        - Explainability code to attribute atom relevance

scripts/
    split_data.py       - script to split AqueousSolu-Exp dataset according to conf/split/*
    train_model.py	- training script for all pytorch models (mmb, mmb-avg, ecfp-lin, ecfp-hier)
    train_sklearn.py	- training script for all sklearn-based models (svr, rf)
    predict_model.py    - use best model checkpoint to predict test set & parity plot
    explain_mmb.py 	- MMB + XAI: explainability script + plots + visualization
    explain_shap.py 	- MMB + SHAP: explainability script + plots + visualization
    explain_ecfp.py 	- ECFP (lin,hier,svr,rf) explainability script to attrib + save figs
    plot_{}.py          - various plotting script (model, similarity, datasplit, pca, ...)

nemo_src/
    transformer_attn.py - MegaMolBART source augmented with code to extract attention + grads
    infer.py            - MegaMolBART source to load model
    regex_tokenizer.py  - MegaMolBART source for the tokenzier

final/{modelname}/
    models/             - model checkpoint file produced by training script
    viz/                - visualizations for test set produced by explain script
```

# Setup / Installation:
### Clone repository, build MegaMolBART docker container
```
git clone https://github.com/BigChemistry-RobotLab/explaining_a_chemical_language_model.git
cd explaining_a_chemical_language_model
# pull nvidia megamolbart:v0.2 docker container, mount repo (current directory) into /workspace
docker run \
    --gpus all \
    --name mmb \
    -p 8888:8888 \
    --detach \
    --volume $PWD:/workspace \
    nvcr.io/nvidia/clara/megamolbart_v0.2:0.2.0
```

### Download the SolProp datasets (Version v1.2, Jul 1, 2022)
```
download SolProp datasets from https://zenodo.org/record/5970538
extract AqueousSolu.csv into /data
```

### Run & attach to docker container
```
# attach shell to container
docker exec -it mmb bash
# change to /workspace directory (which mounts the local repo into container)
cd /workspace
# install requirements
pip install -r requirements.txt
```
Alternatively, install requirements with pip directly. Other dependencies should be installed with the MegaMolBART docker container.
```
pip install dagshub==0.3.8 mlflow==2.7.1 dvc==3.24.0 matplotlib==3.5.2 rdkit==2023.3.3 scikit-learn==0.24.2 numpy==1.22.3 pandas==1.4.3 seaborn==0.13.0 hydra-core==1.3.2 omegaconf==2.3.0 wandb==0.13.1 pyopenssl==22.1.0 shap==0.43.0
```

### Add code to extract attention scores + gradients
```
# Option 1: overwrite transformer.py with nemo_src/transformer_attn.py
cp /workspace/nemo_src/transformer_attn.py /opt/conda/lib/python3.8/site-packages/nemo/collections/nlp/modules/common/megatron/transformer.py
```
```
# Option 2: Do it yourself (eg. for your own model):
# ADD to class CoreAttention(MegatronModule):
    def save_attn(self, attn):
        self.attn = attn

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn(self):
        return self.attn

    def get_attn_gradients(self):
        return self.attn_gradients

# ADD AFTER SOFTMAX, BEFORE ATTN_DROPOUT
    if not self.training:
        self.save_attn(attention_probs)
        attention_probs.register_hook(self.save_attn_gradients)
```


# Reproduciblity with DVC + Hydra
This project uses DVC + Hydra to set default configurations in './conf/{task,split,model,head}' and execute them from the command line.
All scripts read the 'params.yaml' file, which is modified by DVC before execution. 
Settings can be selected using -S 'split=random', or overridden with  specific options such as -S 'split.n_splits = 3'
Note that the model & xai config need to match. The --queue option enables queueing experiments for sequential execution by specifing multiple config options.
```
dvc config hydra.enabled=True 
dvc exp run --queue -S

# run single experiment 
dvc exp run -S 'task=aq' -S 'model=mmb-ft' -S 'head=lin' -S 'split=accurate' -S 'model.n_epochs=30' -S 'model.n_batch=48' -S 'split.n_splits=5' -S 'xai=ours'

# using DVC queue (dvc queue start) 
dvc exp run --queue -S 'task=aq' -S 'model=mmb,mmb-ft' -S 'head=lin,hier' -S 'split=accurate' -S 'model.n_epochs=30' -S 'model.n_batch=48' -S 'split.n_splits=5' -S 'xai=ours'
dvc exp run --queue -S 'task=aq' -S 'model=mmb-avg,mmb-ft-avg' -S 'head=lin,hier' -S 'split=accurate' -S 'model.n_epochs=30' -S 'model.n_batch=48' -S 'split.n_splits=5' -S 'xai=shap'
dvc exp run --queue -S 'task=aq' -S 'model=ecfp,ecfp2k -S 'head=lin,hier,svr,rf' -S 'split=accurate' -S 'model.n_epochs=30' -S 'model.n_batch=48' -S 'split.n_splits=5' -S 'xai=ecfp'
```

### or run train + explain scripts individually
```
# edit params.yaml first
python scripts/split_data.py
python scripts/train_model.py
python scripts/predict_model.py
python scripts/explain_model.py
```

### Acknowledgements 
We acknowledge funding from the National Growth Fund project 'Big Chemistry' (1420578), funded by the Ministry of Education, Culture and Science. This project has received funding from the European Union’s Horizon 2020 research and innovation programmes under Grant Agreement No. 833466 (ERC Adv. Grant Life-Inspired). This project has received funding from a Spinoza Grant of the Netherlands Organisation for Scientific Research (NWO).

## Citation
If you found this repository useful, please cite our paper:
```
@article{Hoedl2024,
  title = {What can Attribution Methods show us about Chemical Language Models?},
  author = {H{\"o}dl, Stefan and Kachman, Tal and Bachrach, Yoram and Huck, Wilhelm and Robinson, William},
  journal = {Digital Discovery},
  year = {2024},
  publisher = {Royal Society of Chemistry},
  doi = {https://doi.org/10.1039/D4DD00084F}
}
```
