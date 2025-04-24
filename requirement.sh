#!/bin/bash

conda install numpy=1.23
conda install pytorch=2.2.1 torchvision=0.17.1 torchaudio=2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge accelerate==0.22.0
conda install -c conda-forge scanpy python-igraph leidenalg
pip install ninja
pip install flash-attn --no-build-isolation
conda install tensorboard