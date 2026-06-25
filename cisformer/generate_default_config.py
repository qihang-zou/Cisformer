import os
import argparse
from importlib.resources import files as rfiles
from cisformer.resource_utils import SUPPORTED_SPECIES, download_annotation, validate_species

def _total_gene(species):
    validate_species(species)
    gene_file = rfiles("cisformer.resource") / f"{species}_genes.tsv"
    with open(gene_file, "r") as f:
        return sum(1 for _ in f)

def main(species="human", download_genome_annotation=True):
    total_gene = _total_gene(species)
    if download_genome_annotation:
        download_annotation(species)
    os.makedirs("cisformer_config", exist_ok=True)
    with open(os.path.join("cisformer_config","atac2rna_config.yaml"), "w") as f:
        f.write(f"""
datapreprocess:
  enc_max_len: 10000 # "whole" for whole sequence
  dec_max_len: 3000 # "whole" for whole sequence
  multiple: 1

model:
  total_gene: {total_gene} # length of cisformer gene vocab
  max_express: 7
  dim: 280 # should be devided by 7
  dec_depth: 4
  dec_heads: 7
  dec_ff_mult: 4
  dec_dim_head: 140
  dec_emb_dropout: 0.1
  dec_ff_dropout: 0.1
  dec_attn_dropout: 0.1
  
training:
  SEED: 2023
  batch_size: 96
  num_workers: 2
  epoch: 100
  lr: 5e-4
  gamma_step: 4
  gamma: 0.6
  patience: 2 # patience for early stop
            """
        )
        
    with open(os.path.join("cisformer_config","rna2atac_config.yaml"), "w") as f:
        f.write(f"""
datapreprocess:
  enc_max_len: 2048 # "whole" for whole sequence
  dec_max_len: 2048 # "whole" for whole sequence
  multiple: 2 # 40

model:
  total_gene: {total_gene} # length of cisformer gene vocab
  max_express: 64
  dim: 210 # should be devided by 7
  dec_depth: 6
  dec_heads: 6
  dec_ff_mult: 4
  dec_dim_head: 128
  dec_emb_dropout: 0.1
  dec_ff_dropout: 0.1
  dec_attn_dropout: 0.1
  
training:
  SEED: 0
  batch_size: 16
  num_workers: 2
  epoch: 500
  lr: 1e-3
  gamma_step: 5
  gamma: 0.9
  patience: 5  # patience for early stop
            """
        )
            
    with open(os.path.join("cisformer_config","accelerate_config.yaml"), "w") as f:
        f.write("""
{
  "compute_environment": "LOCAL_MACHINE",
  "debug": false,
  "distributed_type": "MULTI_GPU",
  "downcast_bf16": True,
  "gpu_ids": "1,2",
  "machine_rank": 0,
  "main_training_function": "main",
  "mixed_precision": "fp16",
  "num_machines": 1,
  "num_processes": 2,
  "rdzv_backend": "static",
  "same_network": true,
  "tpu_use_cluster": false,
  "tpu_use_sudo": false,
  "use_cpu": false,
  "main_process_port": 29934
}
            """
        )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=SUPPORTED_SPECIES, default="human", help="human or mouse")
    parser.add_argument("--skip_annotation_download", action="store_true", help="Only generate config files; do not download genome annotation")
    args = parser.parse_args()
    main(species=args.species, download_genome_annotation=not args.skip_annotation_download)
