# %% [markdown]
# Attention! You should install [liftover](https://genome.ucsc.edu/FAQ/FAQdownloads.html#liftOver) and [bedtools v2.31.0](https://bedtools.readthedocs.io/en/latest/) to use this manuscript

# %%
import pandas as pd
import os
import pickle as pkl
import torch
import yaml
import tqdm
import random
import argparse

from bio_tools import split_grange, granges2bed
from py_tools import Info
import pybedtools
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse import save_npz

# import sys
# sys.path.append("M2Mmodel")
import M2Mmodel
from M2Mmodel.utils import *
from M2Mmodel.M2M import M2M_atac2rna

random.seed(2024)

# %% [markdown]
# # 函数

# %%
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

# %% [markdown]
# 传参
parser = argparse.ArgumentParser(description='pretrain script of M2M')

parser.add_argument('-d', '--data_path', type=str, help="Preprocessed file end with '.pt'. You can specify the directory or filename of certain file")
parser.add_argument('-c', '--celltype_info', help="Celltype information of your input data")
parser.add_argument('-l', '--model_parameters', type=str, help="Pre-trained model parameters")
parser.add_argument('-o', '--output_dir', type=str, help="Output directory")
parser.add_argument('-n', '--num_of_cells', type=int, default=None, help="How many random cell you want for attention generation of each cell type. By default use all the input cell. This parameter should be multiples of 20")
# parser.add_argument('-o', '--output_dir', type=str, default="output", help="Output directory")

args = parser.parse_args()

save_dir = args.output_dir
# output_dir = args.output_dir
data_path = args.data_path
celltype_info = args.celltype_info
model_parameters = args.model_parameters
num_of_cells = args.num_of_cells

if num_of_cells:
    num_of_cells = num_of_cells // 20
    if num_of_cells == 0:
        num_of_cells = 1
else:
    num_of_cells = float('inf')

# %%
# output_dir = "output"
# dataset_name = "Tumor_B"
# dataset_name = "PanCancer"
# dataset_name = "PC_GBM"
# dataset_name = "PC_Normal"
# dataset_name = "Brain"
# dataset_name = "PC_UCEC"
# dataset_name = "BMMC"
# dataset_name = "PC_BRCA"
# dataset_name = "PC_PDAC"
# dataset_name = "PC_CRC"
# dataset_name = "PC_HNSCC"
# dataset_name = "PC_SKCM"
# dataset_name = "PC_CRC2"
# dataset_name = "PC_CRC3"
# dataset_name = "K562"
# save_dir = os.path.join(output_dir, dataset_name)
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/Tumor_B/cisformer_test_dataset/10000_3000_0.pt"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/PBMC_10k_cell_sorting/cisformer_test_dataset/10000_3000_0.pt"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/pan-cancer/ML1652M1-Ty1/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/pan-cancer/ML1652M1-Ty1/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2024-11-09_tumor_B_arg3/epoch10/pytorch_model.bin"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-01-06_pbmc_cs_arg3/epoch4/pytorch_model.bin"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-19_pan-cancer_arg3/epoch5/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/GBM/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/GBM/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-20_pc_GBM_arg3/epoch12/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/Normal/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/Normal/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-21_pc_normal_arg3/epoch9/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/Brain/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/Brain/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-23_Brain_arg3/epoch15/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/UCEC/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/UCEC/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-25_pc_UCEC_arg3/epoch5/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/BMMC/cisformer_test_dataset_downscale10/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/BMMC/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-25_bmmc_arg3/epoch5/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/BRCA/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/BRCA/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-25_pc_BRCA_arg3/epoch8/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/PDAC/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/PDAC/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-25_pc_PDAC/epoch14/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/HNSCC/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/HNSCC/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-25_pc_HNSCC/epoch6/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/SKCM/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/SKCM/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-25_pc_SKCM/epoch9/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/CRC2/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/CRC2/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-26_pc_crc2_arg3/epoch61/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/CRC3/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/PanCancer/CRC3/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2025-02-26_pc_crc3_arg3/epoch7/pytorch_model.bin"
# data_path = "/data/home/zouqihang/desktop/project/M2M/dataset/K562/cisformer_test_dataset/10000_3000_0.pt"
# celltype_info = "/data/home/zouqihang/desktop/project/M2M/dataset/K562/celltype_info.tsv"
# model_parameters = "/data/home/zouqihang/desktop/project/M2M/version1.0.0/save/2024-11-09_k562_arg3/epoch5/pytorch_model.bin"

# processed data
data = torch.load(data_path)
config = "args/arg3_noenc.yaml"
gne = "output/gene_near_enhancers_250_idx.pkl"
extend = 250 #kbp
save_dir = os.path.join(save_dir, f'{extend}kbp_attention_weight')
os.makedirs(save_dir, exist_ok=True)

# parameters
batch_size = 20

# %%
# ref
peak_list = "human_cCREs.bed"
gene_list = "human_genes.tsv"

peak_list = pd.read_csv(peak_list, sep="\t", header=None)
gene_list = pd.read_csv(gene_list, sep="\t", header=None)
gene_list = gene_list[1].tolist()
peak_list_str = list(peak_list[0]+":"+peak_list[1].map(str)+"-"+peak_list[2].map(str))

gene_ref = pd.read_csv("/data/home/zouqihang/desktop/project/M2M/dataset/refGene/hg38.refGene.gtf.gz", sep="\t", header=None)
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
    
    enc_depth = config["model"].get("enc_depth"),
    enc_heads = config["model"].get("enc_heads"),
    enc_ff_mult = config["model"].get("enc_ff_mult"),
    enc_dim_head = config["model"].get("enc_dim_head"),
    enc_emb_dropout = config["model"].get("enc_emb_dropout"),
    enc_ff_dropout = config["model"].get("enc_ff_dropout"),
    enc_attn_dropout = config["model"].get("enc_attn_dropout"),
    
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
        output_name = os.path.join(save_dir, f"{extend}kbp_{tmp_celltype}_attention_weight.npz")
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
        save_npz(output_name, output_attn)


