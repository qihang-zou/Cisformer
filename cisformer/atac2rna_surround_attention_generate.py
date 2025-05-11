import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import pickle as pkl
import torch
import yaml
import tqdm
import random
import time
# import argparse
from importlib.resources import files as rfiles

import pybedtools
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz
import anndata

# import sys
# sys.path.append("M2Mmodel")
# import M2Mmodel
from cisformer.M2Mmodel.utils import *
from cisformer.M2Mmodel.M2M import M2M_atac2rna
from cisformer.compute_surround_enhancer import main as cse

random.seed(2024)

# %% [markdown]
# # 函数

# %%
def Info(*args):
    print(f"[{time.strftime('%H:%M:%S')}]" , *args) 

def split_grange(gene_coordinate):
    """split genome range string

    Args:
        gene_coordinate (str): genome range information like: "chr16:28931939-28939346"

    Returns:
        str: chr
        int: start_loc
        int: end_loc
    """
    gene_coordinate = gene_coordinate.strip()
    chr = gene_coordinate.split(":")[0]
    start_loc = int(gene_coordinate.split(":")[1].split("-")[0])
    end_loc = int(gene_coordinate.split(":")[1].split("-")[1])
    return chr, start_loc, end_loc

def granges2bed(granges, output, no_info=False):
    """covert granges into bed format file
    Args:
        granges (iterable object): format like this: ['chr14:103715039-103715244', 'chr19:10518814-10518968']
        output (str): name of output file
    """
    bed = [split_grange(each) for each in granges]
    bed = pd.DataFrame(bed)
    bed.to_csv(output, sep="\t", header=None, index=None)
    if not no_info:
        Info(f"Output saved at: {output}")
        
def combine_grange(chr, start, end):
    return str(chr) + ":" + str(start) + "-" + str(end)

def read_result(result_file):
    with open(result_file, "rb") as f:
        result = pkl.load(f)
    data = pd.DataFrame({
        'Category': result['top_ks'],
        'cisFormer': result['cisformer']['score'],
        'Scarlink': result['scarlink']['score'],
        'ArchR': result['archr']['score']
    })
    return data

def exists(val):
    return val is not None

def grange_intersect(a,b):
    a_cols = a.columns
    b_cols = b.columns
    a = pybedtools.BedTool.from_dataframe(a)
    b = pybedtools.BedTool.from_dataframe(b)
    o_a = a.intersect(b).to_dataframe(names = a_cols)
    o_b = b.intersect(a).to_dataframe(names = b_cols)
    o_a = o_a.drop_duplicates()   
    return o_a, o_b

def comp(ground_truth, output_links):
    a,b = grange_intersect(ground_truth, output_links)
    output = []
    for x, y in b.iterrows():
        (Chr, start, end, gene) = y[:4]
        tmp_a = a[(a['chrom'] == Chr) &
                (a['chromStart'] <= end) &
                (a['chromEnd'] >= start) & 
                (a['measuredGeneSymbol'] == gene)]
        if exists(tmp_a):
            output.append(tmp_a)
    output = pd.concat(output)
    return output['Regulated'].sum() / len(output)

def main(output_dir, data_path, celltype_info, model_parameters, num_of_cells, config, distance):
    # parser = argparse.ArgumentParser(description='pretrain script of M2M')

    # parser.add_argument('-d', '--data_path', type=str, help="Preprocessed file end with '.pt'. You can specify the directory or filename of certain file")
    # parser.add_argument('-c', '--celltype_info', help="Celltype information of your input data")
    # parser.add_argument('-l', '--model_parameters', type=str, help="Pre-trained model parameters")
    # parser.add_argument('-o', '--output_dir', type=str, help="Output directory")
    # parser.add_argument('-n', '--num_of_cells', type=int, default=None, help="How many random cell you want for attention generation of each cell type. By default use all the input cell. This parameter should be multiples of 20")
    # parser.add_argument('--config', type = str, default="config/atac2rna_config.yaml", help = "Config file") 
    # parser.add_argument('--distance', type=int, default=250000, help="Consider enhancers located within a certain number of base pairs upstream or downstream of the gene.")

    # args = parser.parse_args()

    save_dir = output_dir
    # output_dir = args.output_dir
    data_path = data_path
    celltype_info = celltype_info
    model_parameters = model_parameters
    num_of_cells = num_of_cells
    config = config
    extend = int(distance/1e3) #kbp
    gne = rfiles("cisformer.resource")/f"gene_surround_enhancers_{extend}kbp_idx.pkl"
    if not os.path.exists(gne):
        print(gne)
        print("Generating gene surrounded enhancers dictionary for the first time. This will take a while...")
        cse(distance)
    else:
        print("Previous generated gene surrounded enhancers dictionary found!")

    if num_of_cells:
        num_of_cells = num_of_cells // 20
        if num_of_cells == 0:
            num_of_cells = 1
    else:
        num_of_cells = float('inf')

    # processed data
    data = torch.load(data_path)
    save_dir = os.path.join(save_dir, 'cisformer_link')
    os.makedirs(save_dir, exist_ok=True)

    # parameters
    batch_size = 20

    # %%
    # ref
    peak_list = rfiles("cisformer.resource")/"human_cCREs.bed"
    gene_list = rfiles("cisformer.resource")/"human_genes.tsv"

    peak_list = pd.read_csv(peak_list, sep="\t", header=None)
    gene_list = pd.read_csv(gene_list, sep="\t", header=None)
    gene_list = gene_list[1].tolist()
    peak_list_str = list(peak_list[0]+":"+peak_list[1].map(str)+"-"+peak_list[2].map(str))

    gene_ref = pd.read_csv(rfiles("cisformer.resource")/"hg38.refGene.gtf.gz", sep="\t", header=None)
    gene_ref[9] = gene_ref.iloc[:,8].map(lambda x: x.split(";")[-2].split('"')[-2])
    gene_ref = gene_ref[gene_ref[2]=="transcript"]

    # celltype_info
    celltype_info = pd.read_csv(celltype_info, sep="\t", header=None)
    celltype_info.columns = ['barcode', 'celltype']
    celltypes = celltype_info['celltype'].unique()
    cell_info = pd.read_csv(os.path.join(os.path.dirname(data_path), "cell_info.tsv"), sep="\t", index_col=0)
    celltype_info = pd.merge(celltype_info,cell_info,how='inner',left_on='barcode',right_index=True)
    celltype_info.index = range(len(celltype_info))

    # %%
    celltype_data = {}
    cell_nums = {}
    for celltype in celltypes:
        temp_celltype_info = celltype_info[celltype_info['celltype']==celltype]
        temp_indices = temp_celltype_info.index.tolist()
        temp_data = [data[i][temp_indices] for i in range(len(data))]
        celltype_data[celltype] = temp_data
        print(celltype, len(temp_indices))
        cell_nums[celltype] = len(temp_indices)
    with open(os.path.join(save_dir, 'cell_num.pkl'), "wb") as f:
        pkl.dump(cell_nums, f)

    # %%
    os.makedirs(save_dir, exist_ok=True)
    with open(gne, "rb") as f:
        gene_near_enhancers = pkl.load(f)

    # %% [markdown]
    # Model initial

    # %%
    # from config
    with open(config, "r") as f:
        config = yaml.safe_load(f)
        
    model = M2M_atac2rna(
        dim = config["model"].get("dim"),
        
        enc_depth = 0,
        enc_heads = config["model"].get("dec_heads"),
        enc_ff_mult = config["model"].get("dec_ff_mult"),
        enc_dim_head = config["model"].get("dec_dim_head"),
        enc_emb_dropout = config["model"].get("dec_emb_dropout"),
        enc_ff_dropout = config["model"].get("dec_ff_dropout"),
        enc_attn_dropout = config["model"].get("dec_attn_dropout"),
        
        dec_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
        dec_num_value_tokens = config["model"].get('max_express') + 1, # +1 for 0
        dec_depth = config["model"].get("dec_depth"),
        dec_heads = config["model"].get("dec_heads"),
        dec_ff_mult = config["model"].get("dec_ff_mult"),
        dec_dim_head = config["model"].get("dec_dim_head"),
        dec_emb_dropout = config["model"].get("dec_emb_dropout"),
        dec_ff_dropout = config["model"].get("dec_ff_dropout"),
        dec_attn_dropout = config["model"].get("dec_attn_dropout")
    )

    model.load_state_dict(torch.load(model_parameters))
    model = model.half()

    # %% [markdown]
    # ## Computing loop
    # 采用横轴纵轴排rank的方式（参考[scGPT](https://www.nature.com/articles/s41592-024-02201-0)）并且通过min-max的方式计算出enhancer-gene的score，是gene based的

    # %%
    # generate attention score matrix
    device = select_least_used_gpu()
    # device = "cuda:6"
    model.to(device)
    dataloader_kwargs = {'batch_size': batch_size, 'shuffle': True}

    with torch.no_grad():
        for celltype in celltype_data.keys():
            Info(f"Processing {celltype} ...")
            tmp_celltype = celltype.replace(" ", "-")
            tmp_celltype = tmp_celltype.replace("/", "-or-")
            output_name = os.path.join(save_dir, f"{tmp_celltype}_{extend}kbp_correlation_matrix.h5ad")
            if os.path.exists(output_name):
                Info(f"{celltype} exists, skip ...")
                continue
            dataset = PreDataset(celltype_data[celltype])
            loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
            output_attn = lil_matrix((len(gene_list), len(peak_list)))
            i = 0
            for inputs in tqdm.tqdm(loader, ncols=80):
                if i >= num_of_cells:
                    break
                i += 1
                rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
                attn = model.generate_attn_weight(atac_sequence, rna_sequence, which = "cross", enc_mask = enc_pad_mask, dec_mask = (rna_value != 0)) # row: gene col: peak
                atac_sequence_new = atac_sequence.cpu().numpy()
                atac_sequence_new = np.apply_along_axis(list_digit_back, -1, atac_sequence_new) - 1 # 减掉pad index
                atac_sequence_new = torch.Tensor(atac_sequence_new).to(device)
                rna_sequence_new = rna_sequence - 1 # 减掉pad index
                    
                for j in range(rna_sequence.shape[0]):
                    attn_cell = torch.Tensor(attn[j]).to(device)
                    atac_sequence_cell = atac_sequence_new[j, atac_sequence_new[j] >= 0]
                    rna_sequence_cell = rna_sequence_new[j, rna_sequence_new[j] >= 0]
                    # 对两个维度都做rank normalize
                    # Rank normalization by row
                    order = torch.argsort(attn_cell, dim=0)
                    attn_cell = torch.argsort(order, dim=0)
                    # Rank normalization by column
                    order = torch.argsort(attn_cell, dim=1)
                    attn_cell = torch.argsort(order, dim=1)
                    # min-max normal
                    a_max = attn_cell.max()
                    a_min = attn_cell.min()
                    attn_cell = (attn_cell - a_min) / (a_max - a_min)
                    
                    for x in range(attn_cell.shape[0]):
                        gene = int(rna_sequence_cell[x].item())
                        try:
                            related_enhancers = gene_near_enhancers[gene]
                        except KeyError:
                            continue
                        related_enhancers = gene_near_enhancers[gene]
                        related_enhancers = [each for each in related_enhancers if each in atac_sequence_cell]
                        related_enhancers_attn_idx = [atac_sequence_cell.tolist().index(each) for each in related_enhancers]
                        if related_enhancers_attn_idx:
                            output_attn[gene, related_enhancers] += attn_cell[x, related_enhancers_attn_idx].cpu().numpy()
            
            # 使用csr稀疏矩阵格式保存
            output_attn = output_attn / len(dataset)
            output_attn = output_attn.tocsr()
            adata_output = anndata.AnnData(X=output_attn)
            adata_output.obs_names = gene_list
            adata_output.var_names = peak_list_str
            adata_output.write(output_name)

if __name__ == "__main__":
    main()