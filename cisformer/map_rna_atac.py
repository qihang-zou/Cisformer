import warnings
warnings.filterwarnings("ignore")
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import pybedtools
import numpy as np
from tqdm import tqdm
import os
import random
from importlib.resources import files as rfiles

def is_ens(gene_iter):
    """Give a gene iterator, and return True if most of them are named by ENSEMBLE id

    Args:
        gene_iter (iter): Gene iterator. A list like object containing gene names

    Returns:
        bool
    """
    bool_list = [each.startswith("ENSG") for each in gene_iter]
    ens_percent = sum(bool_list) / len(bool_list)
    if ens_percent > 0.95:
        return True
    else:
        return False

def main(rna_file, atac_file, save_dir=None, test_percent=0.2, seed=2023, divide=False, log1p=False):
    """map rna and atac and divide them into train and test set

    Args:
        rna_file (_type_): _description_
        atac_file (_type_): _description_
        save_dir (_type_, optional): _description_. Defaults to None. By default it will use the path of atac file.
        test_percent (float, optional): _description_. Defaults to 0.2.
        seed (int, optional): _description_. Defaults to 2023.
        divide (bool, optional): Whether to split the dataset into train and test dataset. Defaults to True.
        log1p (bool, optional): Whether to normalize rna dataset by log1p.
    """
    random.seed(seed)
    
    ## rna
    rna = sc.read_h5ad(rna_file)  # 12012 × 36601
    if log1p:
        sc.pp.log1p(rna)
    genes = pd.read_table(rfiles('cisformer.resource')/'human_genes.tsv', names=['gene_ids', 'gene_name'])
    if len(rna.var_names) == len(genes['gene_name']) and all(rna.var_names == genes['gene_name']):
        print("Previous mapped rna detected, skip mapping.")
        rna_new = rna
    else:
        rna.var_names_make_unique()
        rna.X = rna.X.toarray()
        rna_exp = pd.DataFrame(rna.X.T, index=rna.var.index)
        rna_exp.index = rna.var.index
        if is_ens(rna.var_names):
            genes.index = genes['gene_ids'].values
            X_new = pd.merge(genes, rna_exp, how='left', left_index=True, right_index=True).iloc[:, genes.shape[1]:].T
        else:
            genes.index = genes['gene_name'].values
            X_new = pd.merge(genes, rna_exp, how='left', left_index=True, right_index=True).iloc[:, genes.shape[1]:].T
        X_new.fillna(value=0, inplace=True)
        # rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
        # X_new = pd.merge(genes, rna_exp, how='left', left_on='gene_ids').iloc[:, 5:].T
        # X_new.fillna(value=0, inplace=True)
        rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'gene_name': genes['gene_name'], 'feature_types': 'Gene Expression'}))
        rna_new.var.index = genes['gene_name'].values
        rna_new.X = csr_matrix(rna_new.X)   # 12012 × 38244
    rna_obs = set(rna_new.obs_names)

    ## atac
    atac =  sc.read_h5ad(atac_file) # 12012 × 94993
    cCREs = pd.read_table(rfiles('cisformer.resource')/'human_cCREs.bed', names=['chr', 'start', 'end'])
    cCREs['idx'] = range(cCREs.shape[0])
    ccre = cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str)
    if len(atac.var_names) == len(ccre) and all(atac.var_names == ccre):
        print("Previous mapped atac detected, skip mapping.")
        atac_new = atac
    else:
        atac.X[atac.X>0] = 1  # binarization
        atac.X = atac.X.toarray()
        
        peaks = pd.DataFrame({'id': atac.var_names})
        peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
        peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
        peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
        peaks.drop(columns='id', inplace=True)
        peaks['idx'] = range(peaks.shape[0])

        cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
        peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
        idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
        idx_map.columns = ['peaks_idx', 'cCREs_idx']

        m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
        for i in tqdm(range(atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
            m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1

        atac_new = sc.AnnData(m, obs=atac.obs, var=pd.DataFrame({'cCREs': ccre, 'feature_types': 'Peaks'}))
        atac_new.var.index = atac_new.var['cCREs'].values
        atac_new.X = csr_matrix(atac_new.X)    # 12012 × 1033239
    atac_obs = set(atac_new.obs_names)
    
    # map, divide and save
    same_cells = list(rna_obs.intersection(atac_obs))
    rna_new = rna_new[same_cells, :]
    atac_new = atac_new[same_cells, :]
    train_set = random.sample(same_cells, int(len(same_cells) * (1-test_percent)))
    test_set = [each for each in same_cells if each not in train_set]
    
    # save
    if save_dir:
        _, rna_name = os.path.split(rna_file)
        _, atac_name = os.path.split(atac_file)
    else:
        _, rna_name = os.path.split(rna_file)
        save_dir, atac_name = os.path.split(atac_file)
    rna_new.write(os.path.join(save_dir, "mapped_" + rna_name))
    atac_new.write(os.path.join(save_dir, "mapped_" + atac_name))
    output = {'rna_new': rna_new, 'atac_new': atac_new}
    if divide:
        train_rna = rna_new[train_set, :]
        test_rna = rna_new[test_set, :]
        train_atac = atac_new[train_set, :]
        test_atac = atac_new[test_set, :]
        train_rna.write(os.path.join(save_dir, "train_" + rna_name))
        test_rna.write(os.path.join(save_dir, "test_" + rna_name))
        train_atac.write(os.path.join(save_dir, "train_" + atac_name))
        test_atac.write(os.path.join(save_dir, "test_" + atac_name))
        output.update({'train_rna': train_rna, 'test_rna': test_rna, 'train_atac': train_atac, 'test_atac': test_atac})
    
    return output

if __name__ == '__main__':
    rna_file = input("rna path: ")
    atac_file = input("atac path: ")
    save_dir = input("save path: ")
    main(rna_file, atac_file, save_dir)