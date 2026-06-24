# Installation

## Install the Command-Line Package

Install Cisformer from PyPI:

```bash
pip install cisformer
```

This installs the `cisformer` command-line entry point.

## Recommended Runtime Environment

Cisformer training and prediction rely on PyTorch, CUDA, Hugging Face
Accelerate, Scanpy, pybedtools, flash-attn, torcheval, and tensorboard. We
recommend using conda to isolate these dependencies:

```bash
conda create -n cisformer python=3.10
conda activate cisformer
bash ./requirement.sh
```

You can also install the dependencies manually:

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

## Verify the Installation

After installation, check the CLI:

```bash
cisformer -h
cisformer generate_default_config -h
```

## GPU and Distributed Training

Cisformer uses Hugging Face Accelerate for distributed training. After running
`cisformer generate_default_config`, edit `cisformer_config/accelerate_config.yaml`
to match your machine:

- `gpu_ids`: comma-separated GPU IDs.
- `num_processes`: number of GPU processes.
- `main_process_port`: use a free port, especially when running multiple jobs.

## Bedtools Requirement

Preprocessing and link inference use genomic interval operations through
`pybedtools`. Make sure the system `bedtools` binary is available in your
environment if pybedtools reports backend errors.
