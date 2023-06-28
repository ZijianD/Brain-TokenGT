# Brain-TokenGT
This is the official repository for "Beyond the Snapshot: Brain Tokenized Graph Transformer for Longitudinal Brain Functional Connectome Embedding" (MICCAI 2023)
[model illustration figure](model.png)


## Dependencies

The framework needs the following dependencies:

```
numpy==1.24.2
optuna==3.1.0
PyYAML==6.0
scikit_learn==1.2.2
scipy==1.9.1
torch==2.0.0
torch_geometric==2.2.0
tqdm==4.64.1
```

## Datasets

- ADNI: https://adni.loni.usc.edu/
- OASIS: https://www.oasis-brains.org/

We used brain FC metrics derived from ADNI and OASIS-3 resting state fMRI datasets, with preprocessing pipelines following:

Kong, R., Li, J., Orban, C., Sabuncu, M.R., Liu, H., Schaefer, A., Sun, N., Zuo,
X.N., Holmes, A.J., Eickhoff, S.B., et al.: Spatial topography of individual-specific
cortical networks predicts human cognition, personality, and emotion. Cerebral
cortex 29(6), 2533–2551 (2019)

Li, J., Kong, R., Li´egeois, R., Orban, C., Tan, Y., Sun, N., Holmes, A.J., Sabuncu,
M.R., Ge, T., Yeo, B.T.: Global signal regression strengthens association between
resting-state functional connectivity and behavior. NeuroImage 196, 126–
141 (2019)


## Installation

1. Clone the repository: `git clone https://github.com/ZijianD/Brain-TokenGT.git`
2. Change to the project directory: `cd Brain-TokenGT`
3. Install the dependencies: `pip install -r requirements.txt`


## Usage 

``` python
python main_optuna.py # you may modify config.py to change the hyperparameter setup
```

## References
Our implementation uses code from the following repositories:
- [EvolveGCN](https://github.com/IBM/EvolveGCN.git) 
- [tokengt](https://github.com/jw9730/tokengt.git)
