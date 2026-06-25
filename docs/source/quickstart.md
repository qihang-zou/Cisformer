# Quick Start

This page shows the shortest complete workflows for Cisformer. For all examples,
replace `human` with `mouse` if you are using the mouse reference.

## 1. Generate Config Files

```bash
cisformer generate_default_config --species human
```

This creates:

- `cisformer_config/accelerate_config.yaml`
- `cisformer_config/atac2rna_config.yaml`
- `cisformer_config/rna2atac_config.yaml`
- `cisformer_config/resource/<species> Gencode annotation .gtf.gz`

For mouse:

```bash
cisformer generate_default_config --species mouse
```

The generated config uses the correct species-specific `model.total_gene`
automatically.

Large Gencode annotation files are downloaded during config generation instead
of being bundled in the PyPI package. Human uses Gencode v49 and mouse uses
Gencode M39.

## 2. Prepare Input Data

Cisformer expects paired RNA and ATAC `.h5ad` files:

- RNA: cells by genes.
- ATAC: cells by peaks.
- ATAC peak names should use `chr:start-end`.
- Cell barcodes must overlap between RNA and ATAC.

## RNA-to-ATAC

### Preprocess

```bash
cisformer data_preprocess \
  -r test_data/rna.h5ad \
  -a test_data/atac.h5ad \
  -s preprocessed_dataset \
  --species human
```

### Train

```bash
cisformer rna2atac_train \
  -t preprocessed_dataset/cisformer_rna2atac_train_dataset \
  -v preprocessed_dataset/cisformer_rna2atac_val_dataset \
  -n rna2atac_test
```

### Predict

```bash
cisformer rna2atac_predict \
  -r preprocessed_dataset/test_rna.h5ad \
  -m save/2025-05-12_rna2atac_test/epoch34/pytorch_model.bin \
  --species human
```

The default output is `output/cisformer_predicted_atac.h5ad`.

## ATAC-to-RNA

### Preprocess

```bash
cisformer data_preprocess \
  -r test_data/rna.h5ad \
  -a test_data/atac.h5ad \
  -s preprocessed_dataset \
  --atac2rna \
  --species human
```

For ATAC-to-RNA preprocessing, Cisformer checks whether the RNA matrix appears to
already be normalized. If the maximum RNA value is greater than 10, `log1p` is
skipped. Otherwise, `log1p` is applied and the decision is printed.

### Train

```bash
cisformer atac2rna_train \
  -d preprocessed_dataset/cisformer_atac2rna_train_dataset \
  -n atac2rna_test
```

### Predict

```bash
cisformer atac2rna_predict \
  -d preprocessed_dataset/cisformer_atac2rna_test_dataset/atac2rna_0.pt \
  -m save/2025-05-12_atac2rna_test/epoch30/pytorch_model.bin \
  --species human
```

The default output is `output/cisformer_predicted_rna.h5ad`.

## Link cCREs to Genes

Create a two-column, header-free cell type file:

```text
GTACCGGGTATACTGG-1	CD14 Mono
ACTGAATGTCACCAAA-1	cDC2
AACCTTGCAAACTGTT-1	CD14 Mono
```

Then run:

```bash
cisformer atac2rna_link \
  -d preprocessed_dataset/cisformer_atac2rna_test_dataset/atac2rna_0.pt \
  -m save/2025-05-12_atac2rna_test/epoch30/pytorch_model.bin \
  -c test_data/celltype_info.tsv \
  --species human
```

Outputs are written to `output/cisformer_link/`. Each `.h5ad` file is a sparse
gene-by-cCRE matrix for one cell type. Non-zero values are rank-normalized link
scores derived from valid attention scores.

## Next Steps

- See [Usage](usage.md) for all command options.
- See [Concept](conception.md) for how Cisformer link matrices should be
  interpreted.
- See [Release notes](release.md) for changes in v1.1.0.
