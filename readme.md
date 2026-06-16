# DMKGC

Source code for WWW2026 paper: [_Conditional Diffusion Guided Knowledge Transfer for Multi-Domain Knowledge Graph Completion_](https://dl.acm.org/doi/10.1145/3774904.3792252).

## Overview

Multi-domain knowledge graph completion (MKGC) predicts missing triples in a target knowledge graph (KG) by transferring knowledge from other support KGs. Many existing methods rely on strong consistency constraints over aligned entities, which can suppress domain-specific context and hurt performance, especially in low-resource settings.

To address this, we pioneer a generation-based paradigm for MKGC and propose DMKGC, a conditional diffusion-guided knowledge transfer framework. Our key insight is to treat each KG as a partial view of the entity entire information, and generate informative domain-general entity embeddings through diffusion models conditioned on support KGs. Particularly, we first initialize domainagnostic entity embeddings as prior entity embeddings, and then encode them within individual KGs. Afterward, we fuse equivalent entities from support KGs as the conditional diffusion generation guidance. We leverage the prior entity embeddings as the proxy generation objective, which ensures this conditional generation to be unbiased towards any conditioned KGs. Simultaneously, we also train the generated embeddings to be predictive across KGs, thus preserving domain-specific information. Extensive experiments on 14 KGs in 3 benchmarks demonstrate a 4.3% average MRR improvement in tail entity prediction over state-of-the-art methods, with sustained gains in low-resource data settings.


## Requirements

```
python>=3.9
torch==2.3.0+cu118
torch-geometric==2.5.3
torch-scatter==2.1.2+pt23cu118
transformers
numpy==1.26.3
pandas==2.2.2
tqdm==4.66.4
```


## Datasets

We evaluate on three benchmarks:

| Benchmark | `--dataset` flag | Domains |
|-----------|------------------|---------|
| DBP-5L | `dbp5l` | el, en, es, fr, ja |
| E-PKG | `depkg` | de, es, fr, it, jp, uk |
| DWY | `dwy-L` | db, wk, yg |

Data should be placed under `dataset<name>/` (e.g., `datasetdbp5l/`), which is resolved from `--data_path dataset` and `--dataset <name>`.

Dataset sources:

- **DBP-5L**: [LSMGA-MKGC](https://github.com/RongchuanTang/LSMGA-MKGC)
- **E-PKG**: [ss-aga-kgc](https://github.com/amzn/ss-aga-kgc)
- **DWY**: adapted from [BootEA](https://github.com/nju-websoft/BootEA); see also [IMKGC](https://github.com/JiaweiSheng/IMKGC/tree/main)

On first run, the code builds and caches subgraph files under each dataset directory. Reusing existing caches significantly speeds up subsequent training.

Pre-trained text embeddings are **not** used in our experiments.


## Quick Start

Train and evaluate with `run_model.py`. Most hyperparameters are already set to the paper defaults; only the dataset flag and a few dataset-specific overrides are required.

**DBP-5L**

```bash
CUDA_VISIBLE_DEVICES=0 python run_model.py --dataset dbp5l --v exp_dbp5l
```

**E-PKG**

```bash
CUDA_VISIBLE_DEVICES=0 python run_model.py --dataset depkg --round 50 --epoch_each 2 --s_strength 1 --v exp_depkg
```

**DWY**

```bash
CUDA_VISIBLE_DEVICES=0 python run_model.py --dataset dwy-L --v exp_dwy-L
```

Logs and the best checkpoint are saved under `<dataset>/trained_model/`.

### Default hyperparameters

The following defaults are shared across DBP-5L and DWY (`run_model.py`):

- `model=dmkgc`, `lr=0.001`, `margin=0.5`, `batch_size=300`
- `round=30`, `epoch_each=1`, `scheduler=linear`, `warmup=1`
- `n_steps=50`, `n_sampling_step=50`, `beta_sche=exp`
- `p_uncond=0.1`, `s_strength=2`, `w_recon=0.01`, `w_reg=0.001`


## Project structure

```
DMKGC/
├── run_model.py          # training and evaluation entry point
├── run.sh                # example commands for three benchmarks
└── src/
    ├── dmkgc.py          # DMKGC model (GNN + diffusion + TransE)
    ├── dm.py             # diffusion process and denoiser
    ├── gnn.py            # graph encoder
    ├── modules.py        # cross-domain attention fusion
    ├── data_loader.py    # dataset loading
    ├── validate.py       # filtered evaluation
    └── utils.py          # subgraph construction and utilities
```

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{sheng2026:DMKGC,
  title={Conditional Diffusion Guided Knowledge Transfer for Multi-Domain Knowledge Graph Completion},
  author={Sheng, Jiawei and Su, Taoyu and Lin, Xixun and Li, Xiaodong and Liu, Tingwen},
  booktitle={Proceedings of the ACM Web Conference},
  pages={3744--3754},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [LSMGA-MKGC](https://github.com/RongchuanTang/LSMGA-MKGC), [ss-aga-kgc](https://github.com/amzn/ss-aga-kgc), and [OpenKE-PyTorch](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch) for their open-source contributions.
