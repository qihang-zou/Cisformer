import pandas as pd
import os
import tqdm
import pickle as pkl

total_enhancers = "human_cCREs.bed"
total_genes = "human_genes.tsv"
extend = 250000

total_enhancers = pd.read_csv(total_enhancers, sep="\t", header=None)
total_genes = pd.read_csv(total_genes, sep="\t", header=None)
total_genes = total_genes[1].tolist()

gene_ref = pd.read_csv("/data/home/zouqihang/desktop/project/M2M/dataset/refGene/hg38.refGene.gtf.gz", sep="\t", header=None)
gene_ref[9] = gene_ref.iloc[:,8].map(lambda x: x.split(";")[-2].split('"')[-2])
gene_ref = gene_ref[gene_ref[2]=="transcript"]

gene_near_enhancers = {}
gene_near_enhancers_idx = {}
for gene in tqdm.tqdm(total_genes, ncols=80):
    if gene in gene_ref[9].to_list():
        tmp_df = gene_ref[gene_ref[9] == gene]
        gchr = tmp_df.iloc[0,0]
        gstart = tmp_df.iloc[0,3]
        gend = tmp_df.iloc[0,4]
        near_enhancers = total_enhancers[
            (total_enhancers[0] == gchr) &
            (total_enhancers[1] <= gend + extend) &
            (total_enhancers[2] >= gstart - extend)
        ]
        if len(near_enhancers) > 0 :
            gene_near_enhancers[gene] = near_enhancers[0]+':'+near_enhancers[1].map(str)+'-'+near_enhancers[2].map(str)
            gene_near_enhancers_idx[total_genes.index(gene)] = near_enhancers.index.tolist()
    # break
with open(os.path.join(f"output/gene_near_enhancers_{int(extend/1e3)}.pkl"), "wb") as f:
    pkl.dump(gene_near_enhancers, f)
with open(os.path.join(f"output/gene_near_enhancers_{int(extend/1e3)}_idx.pkl"), "wb") as f:
    pkl.dump(gene_near_enhancers_idx, f)