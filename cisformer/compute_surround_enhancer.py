import pandas as pd
import os
import tqdm
import pickle as pkl
from importlib.resources import files as rfiles
from cisformer.resource_utils import gene_surround_path, require_annotation, validate_species

def main(extend = 250000, species = "human"):
    validate_species(species)
    total_enhancers = rfiles("cisformer.resource")/f"{species}_cCREs.bed"
    total_genes = rfiles("cisformer.resource")/f"{species}_genes.tsv"
    # extend = 250000

    total_enhancers = pd.read_csv(total_enhancers, sep="\t", header=None)
    total_genes = pd.read_csv(total_genes, sep="\t", header=None)
    total_genes = total_genes[1].tolist()

    gene_ref = pd.read_csv(
        require_annotation(species),
        sep="\t",
        header=None,
        comment="#",
        low_memory=False,
    )

    gene_ref = gene_ref[gene_ref[2] == "transcript"].copy()

    attr = gene_ref[8].astype(str)
    gene_name = attr.str.extract(r'gene_name "([^"]+)"', expand=False)
    gene_id = attr.str.extract(r'gene_id "([^"]+)"', expand=False)
    transcript_id = attr.str.extract(r'transcript_id "([^"]+)"', expand=False)

    gene_ref[9] = gene_name.fillna(gene_id).fillna(transcript_id)

    # gene_near_enhancers = {}
    # gene_near_enhancers_idx = {}
    # for gene in tqdm.tqdm(total_genes, ncols=80):
    #     if gene in gene_ref[9].to_list():
    #         tmp_df = gene_ref[gene_ref[9] == gene]
    #         gchr = tmp_df.iloc[0,0]
    #         gstart = tmp_df.iloc[0,3]
    #         gend = tmp_df.iloc[0,4]
    #         near_enhancers = total_enhancers[
    #             (total_enhancers[0] == gchr) &
    #             (total_enhancers[1] <= gend + extend) &
    #             (total_enhancers[2] >= gstart - extend)
    #         ]
    #         if len(near_enhancers) > 0 :
    #             gene_near_enhancers[gene] = near_enhancers[0]+':'+near_enhancers[1].map(str)+'-'+near_enhancers[2].map(str)
    #             gene_near_enhancers_idx[total_genes.index(gene)] = near_enhancers.index.tolist()
    
    # GPT加速方案
    # Remove records without gene name
    gene_ref = gene_ref.dropna(subset=[9]).copy()

    # Make sure coordinates are numeric
    gene_ref[3] = pd.to_numeric(gene_ref[3], errors="coerce")
    gene_ref[4] = pd.to_numeric(gene_ref[4], errors="coerce")
    gene_ref = gene_ref.dropna(subset=[3, 4]).copy()

    # Use gene-level span across all transcripts
    gene_span = (
        gene_ref
        .groupby(9, sort=False)
        .agg({
            0: "first",   # chromosome
            3: "min",     # gene start from all transcripts
            4: "max",     # gene end from all transcripts
        })
    )

    # Split enhancers by chromosome to speed up filtering
    enhancers_by_chr = {
        chrom: df
        for chrom, df in total_enhancers.groupby(0, sort=False)
    }

    gene_near_enhancers = {}
    gene_near_enhancers_idx = {}

    for gene_idx, gene in tqdm.tqdm(enumerate(total_genes), total=len(total_genes), ncols=80):
        if gene not in gene_span.index:
            continue

        gchr = gene_span.loc[gene, 0]
        gstart = int(gene_span.loc[gene, 3])
        gend = int(gene_span.loc[gene, 4])

        if gchr not in enhancers_by_chr:
            continue

        chr_enhancers = enhancers_by_chr[gchr]

        near_enhancers = chr_enhancers[
            (chr_enhancers[1] <= gend + extend) &
            (chr_enhancers[2] >= gstart - extend)
        ]

        if len(near_enhancers) > 0:
            gene_near_enhancers[gene] = (
                near_enhancers[0]
                + ":"
                + near_enhancers[1].map(str)
                + "-"
                + near_enhancers[2].map(str)
            )
            gene_near_enhancers_idx[gene_idx] = near_enhancers.index.tolist()
        # break
        
    print("Total genes in gene list:", len(total_genes))
    print("Total genes in GTF:", gene_ref[9].nunique())

    overlap = set(total_genes).intersection(set(gene_ref[9]))
    print("Gene name overlap:", len(overlap))
    print("Example overlap:", list(overlap)[:20])
    print("Example total_genes:", total_genes[:20])
    print("Example GTF gene names:", gene_ref[9].head(20).tolist())
    extend_kbp = int(extend/1e3)
    with open(gene_surround_path(species, extend_kbp), "wb") as f:
        pkl.dump(gene_near_enhancers, f)
    with open(gene_surround_path(species, extend_kbp, idx=True), "wb") as f:
        pkl.dump(gene_near_enhancers_idx, f)

if __name__ == "__main__":
    main()
