# Usage

Cisformer provides seven CLI subcommands:

```bash
cisformer \
  {generate_default_config,data_preprocess,atac2rna_train,atac2rna_predict,atac2rna_link,rna2atac_train,rna2atac_predict}
```

## Species

Version 1.1.0 supports human and mouse references.

| Species | Gene vocabulary | cCRE vocabulary |
| ------- | --------------- | --------------- |
| `human` | 38,244          | 1,033,239       |
| `mouse` | 23,234          | 262,853         |

Use the same `--species` value for config generation, preprocessing,
prediction, and link inference. A model trained with one reference vocabulary
should be used with the matching config and species.

## generate_default_config

Generate default configuration files:

```bash
cisformer generate_default_config [--species {human,mouse}]
```

Arguments:

- `--species`: Reference species. Default: `human`.

Outputs:

- `cisformer_config/accelerate_config.yaml`
- `cisformer_config/atac2rna_config.yaml`
- `cisformer_config/rna2atac_config.yaml`

`model.total_gene` is written from the selected species reference. For human it
is `38244`; for mouse it is `23234`.

### accelerate_config.yaml

Cisformer uses Hugging Face Accelerate for distributed training. Edit these
fields before training:

- `gpu_ids`: GPU IDs to use.
- `num_processes`: number of distributed processes.
- `main_process_port`: a free port for distributed initialization.

### atac2rna_config.yaml

Main parameters:

| Parameter | Meaning |
| --------- | ------- |
| `enc_max_len` | Maximum number of input cCRE tokens sampled per cell. |
| `dec_max_len` | Maximum number of RNA gene tokens predicted per cell. |
| `multiple` | Number of resampled training examples per cell. |
| `total_gene` | Species-specific gene vocabulary size. |
| `max_express` | Number of expression bins used by the decoder. |
| `batch_size` | Training batch size. |
| `patience` | Early stopping patience. |

### rna2atac_config.yaml

Main parameters:

| Parameter | Meaning |
| --------- | ------- |
| `enc_max_len` | Maximum number of input RNA gene tokens sampled per cell. |
| `dec_max_len` | Maximum number of ATAC cCRE tokens sampled per cell. |
| `multiple` | Resampling factor. Larger values expose more loci during training. |
| `total_gene` | Species-specific gene vocabulary size. |
| `max_express` | Maximum RNA count/bin value used by the encoder. |
| `batch_size` | Training batch size. |

For RNA-to-ATAC training, increasing `multiple` often improves coverage of the
large cCRE vocabulary, at the cost of more preprocessing and training time.

## data_preprocess

Preprocess paired RNA and ATAC `.h5ad` files:

```bash
cisformer data_preprocess \
  -r RNA.h5ad \
  -a ATAC.h5ad \
  -s preprocessed_dataset \
  --species {human,mouse} \
  [--atac2rna] \
  [--config CONFIG] \
  [--cnt CNT] \
  [--batch_size BATCH_SIZE] \
  [--num_workers NUM_WORKERS] \
  [--manually] \
  [--shuffle] \
  [--dec_whole_length]
```

Required arguments:

- `-r`, `--rna`: RNA `.h5ad` file.
- `-a`, `--atac`: ATAC `.h5ad` file.
- `--species`: `human` or `mouse`.

Common optional arguments:

- `-s`, `--save_dir`: output directory. Default: current directory.
- `-c`, `--config`: config file. If omitted, Cisformer selects
  `rna2atac_config.yaml` or `atac2rna_config.yaml` from `cisformer_config/`.
- `--atac2rna`: preprocess for ATAC-to-RNA. Without this flag, preprocessing is
  for RNA-to-ATAC.
- `--cnt`: maximum number of cells per output `.pt` file.
- `--manually`: do not split into train/test automatically.
- `--shuffle`: shuffle cells. Do not use this for test sets that need
  `cell_info.tsv`.

Outputs:

- RNA-to-ATAC:
  - `cisformer_rna2atac_train_dataset/`
  - `cisformer_rna2atac_val_dataset/`
- ATAC-to-RNA:
  - `cisformer_atac2rna_train_dataset/`
  - `cisformer_atac2rna_test_dataset/`

When `--shuffle` is not used, Cisformer writes `cell_info.tsv`, which is required
by prediction and link commands that preserve barcode order.

### ATAC-to-RNA log1p Handling

For ATAC-to-RNA preprocessing, Cisformer automatically checks the loaded RNA
matrix:

- If `rna.X.max() <= 10`, Cisformer applies `scanpy.pp.log1p`.
- If `rna.X.max() > 10`, Cisformer assumes the RNA matrix is already normalized
  and skips `log1p`.

The decision is printed to the terminal.

## atac2rna_train

Train an ATAC-to-RNA model:

```bash
cisformer atac2rna_train \
  -d preprocessed_dataset/cisformer_atac2rna_train_dataset \
  -n PROJECT_NAME \
  [-s SAVE_DIR] \
  [-c cisformer_config/atac2rna_config.yaml] \
  [-m MODEL_PARAMETERS]
```

Arguments:

- `-d`, `--data_dir`: preprocessed ATAC-to-RNA training dataset directory.
- `-n`, `--name`: project name used in the checkpoint directory.
- `-s`, `--save`: output checkpoint root. Default: `save`.
- `-c`, `--config_file`: model config.
- `-m`, `--model_parameters`: optional checkpoint to resume from.

## atac2rna_predict

Predict RNA expression from a preprocessed ATAC `.pt` file or directory:

```bash
cisformer atac2rna_predict \
  -d preprocessed_dataset/cisformer_atac2rna_test_dataset/atac2rna_0.pt \
  -m save/PROJECT/epoch30/pytorch_model.bin \
  --species {human,mouse} \
  [-o output] \
  [-n cisformer_predicted_rna] \
  [-c cisformer_config/atac2rna_config.yaml]
```

Required arguments:

- `-d`, `--data`: preprocessed `.pt` file or directory of `.pt` files.
- `-m`, `--model_parameters`: trained ATAC-to-RNA checkpoint.
- `--species`: reference species used for gene names.

Output:

- Default: `output/cisformer_predicted_rna.h5ad`.
- Rows are cells.
- Columns are species-specific genes from `{species}_genes.tsv`.

## atac2rna_link

Infer cell-type-specific cCRE-gene link matrices from a trained ATAC-to-RNA
model:

```bash
cisformer atac2rna_link \
  -d preprocessed_dataset/cisformer_atac2rna_test_dataset/atac2rna_0.pt \
  -m save/PROJECT/epoch30/pytorch_model.bin \
  -c celltype_info.tsv \
  --species {human,mouse} \
  [-o output] \
  [-n NUM_OF_CELLS] \
  [--config cisformer_config/atac2rna_config.yaml] \
  [--distance 250000]
```

Required arguments:

- `-d`, `--data_path`: one preprocessed ATAC-to-RNA `.pt` file. The same
  directory must contain `cell_info.tsv`.
- `-c`, `--celltype_info`: two-column, header-free TSV with barcode and cell
  type.
- `-m`, `--model_parameters`: trained ATAC-to-RNA checkpoint.
- `--species`: `human` or `mouse`.

Optional arguments:

- `-o`, `--output_dir`: output directory. Default: `output`.
- `-n`, `--num_of_cells`: maximum number of cells sampled per cell type.
- `--config`: ATAC-to-RNA config.
- `--distance`: maximum genomic distance in base pairs between a gene and
  candidate cCRE. Default: `250000`.

Example `celltype_info.tsv`:

```text
GTACCGGGTATACTGG-1	CD14 Mono
ACTGAATGTCACCAAA-1	cDC2
AACCTTGCAAACTGTT-1	CD14 Mono
```

Outputs:

- `output/cisformer_link/cell_num.pkl`
- `output/cisformer_link/<celltype>_<distance>kbp_correlation_matrix.h5ad`

The `.h5ad` matrix has genes as rows and cCREs as columns. The filename keeps the
historical `correlation_matrix` suffix, but the stored values are **not raw
correlations**. Cisformer ranks valid attention-derived cCRE-gene scores and
writes the rank-normalized values. Treat `X` as a sparse matrix of relative link
scores for candidate cCRE-gene pairs within the selected distance window.

Example:

```python
import anndata

links = anndata.read_h5ad(
    "output/cisformer_link/CD14-Mono_250kbp_correlation_matrix.h5ad"
)
print(links.shape)
print(links.obs_names[:5])  # genes
print(links.var_names[:5])  # cCRE genomic coordinates
```

For efficiency, use `-n 100` or `-n 200` when each cell type has many cells.

## rna2atac_train

Train an RNA-to-ATAC model:

```bash
cisformer rna2atac_train \
  -t preprocessed_dataset/cisformer_rna2atac_train_dataset \
  -v preprocessed_dataset/cisformer_rna2atac_val_dataset \
  -n PROJECT_NAME \
  [-s SAVE_DIR] \
  [-c cisformer_config/rna2atac_config.yaml] \
  [-m MODEL_PARAMETERS]
```

Arguments:

- `-t`, `--train_data_dir`: RNA-to-ATAC training dataset directory.
- `-v`, `--val_data_dir`: RNA-to-ATAC validation dataset directory.
- `-n`, `--name`: project name.
- `-s`, `--save`: checkpoint root. Default: `save`.
- `-c`, `--config_file`: RNA-to-ATAC config.
- `-m`, `--model_parameters`: optional checkpoint to resume from.

## rna2atac_predict

Predict ATAC accessibility from RNA:

```bash
cisformer rna2atac_predict \
  -r RNA.h5ad \
  -m save/PROJECT/epoch34/pytorch_model.bin \
  --species {human,mouse} \
  [-o output] \
  [-n cisformer_predicted_atac] \
  [-c cisformer_config/rna2atac_config.yaml] \
  [--rna_len 3600] \
  [--batch_size 2] \
  [--num_workers 2]
```

Required arguments:

- `-r`, `--rna_file`: RNA `.h5ad` file.
- `-m`, `--model_parameters`: trained RNA-to-ATAC checkpoint.
- `--species`: reference species used for RNA mapping and cCRE output names.

Optional arguments:

- `--rna_len`: maximum number of expressed genes used per cell. Increasing this
  can improve coverage but uses more memory.
- `--batch_size`: prediction batch size. Keep this small for large cCRE
  vocabularies.
- `--num_workers`: DataLoader workers.

Output:

- Default: `output/cisformer_predicted_atac.h5ad`.
- Rows are cells.
- Columns are species-specific cCRE coordinates from `{species}_cCREs.bed`.

## Recommended File Organization

```text
project/
  cisformer_config/
  preprocessed_dataset/
  save/
  output/
  celltype_info.tsv
```

Keep config files with trained checkpoints. The model architecture depends on
`model.total_gene`, so a checkpoint should always be reused with the matching
species-specific config.
