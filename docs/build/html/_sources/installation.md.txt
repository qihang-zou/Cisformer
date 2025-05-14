# Installation Guide

## Miniconda3
We recommend using [Miniconda3](https://www.anaconda.com/docs/getting-started/miniconda/main) or [Anaconda](https://www.anaconda.com/) as the environment manager. Make sure `conda` is installed.

## Creating the Cisformer Environment
To get started, copy [requirement.sh](https://raw.githubusercontent.com/qihang-zou/Cisformer/refs/heads/main/requirement.sh) to your local server and run:
```bash
conda create -n cisformer python=3.10
conda activate cisformer
bash ./requirement.sh
```
Alternatively, you can install dependencies manually:
```bash
conda create -n cisformer python=3.10
conda activate cisformer
conda install numpy=1.23
conda install pytorch=2.2.1 torchvision=0.17.1 torchaudio=2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge accelerate==0.22.0
conda install -c conda-forge scanpy python-igraph leidenalg
pip install ninja
pip install flash-attn --no-build-isolation
pip install torcheval
conda install tensorboard
conda install pybedtools
```

## Install from PyPI
```bash
pip install cisformer
```