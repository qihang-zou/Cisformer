import scanpy as sc
import random
import argparse

parser = argparse.ArgumentParser(description='Split datasets')
parser.add_argument('--RNA', type=str, help='RNA h5ad file')
parser.add_argument('--ATAC', type=str, help='ATAC h5ad file')
parser.add_argument('--train_pct', type=float, help='Training dataset percentage')
parser.add_argument('--valid_pct', type=float, help='Validating dataset percentage')

args = parser.parse_args()
rna_file = args.RNA
atac_file = args.ATAC
train_pct = args.train_pct
valid_pct = args.valid_pct

rna = sc.read_h5ad(rna_file)
atac = sc.read_h5ad(atac_file)

random.seed(0)
idx = list(range(rna.n_obs))
random.shuffle(idx)
train_idx = idx[:int(len(idx)*train_pct)]
val_idx = idx[int(len(idx)*train_pct):(int(len(idx)*train_pct)+int(len(idx)*valid_pct))]
test_idx = idx[(int(len(idx)*train_pct)+int(len(idx)*valid_pct)):]

rna[train_idx, :].write(rna_file.replace('.h5ad', '')+'_train.h5ad')
rna[val_idx, :].write(rna_file.replace('.h5ad', '')+'_val.h5ad')
rna[test_idx, :].write(rna_file.replace('.h5ad', '')+'_test.h5ad')
atac[train_idx, :].write(atac_file.replace('.h5ad', '')+'_train.h5ad')
atac[val_idx, :].write(atac_file.replace('.h5ad', '')+'_val.h5ad')
atac[test_idx, :].write(atac_file.replace('.h5ad', '')+'_test.h5ad')
