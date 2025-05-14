# Usage

Cisformer is primarily designed to perform two tasks:

1. **Cross-modal prediction**: Accurately and efficiently predict chromatin accessibility (ATAC) from gene expression (RNA), and vice versa.
2. **Cis-regulatory inference**: Use the relatively simple ATAC ➝ RNA prediction task to identify meaningful links between cis-regulatory elements (cCREs) and their target genes.

To support these functionalities, Cisformer provides six main commands:

```bash
usage: cisformer [-h]
       {generate_default_config,data_preprocess,atac2rna_train,atac2rna_predict,atac2rna_link,rna2atac_train,rna2atac_predict}
       ...
```

**Available Subcommands:**

| Subcommand                | Description                              |
| ------------------------- | ---------------------------------------- |
| `generate_default_config` | Generate default config files            |
| `data_preprocess`         | Preprocess RNA/ATAC data                 |
| `atac2rna_train`          | Train ATAC ➝ RNA prediction model        |
| `atac2rna_predict`        | Predict gene expression from ATAC        |
| `atac2rna_link`           | Extract cCRE–gene regulatory links       |
| `rna2atac_train`          | Train RNA ➝ ATAC prediction model        |
| `rna2atac_predict`        | Predict chromatin accessibility from RNA |

Detailed usages are listed as follows:

---

## generate_default_config

This command generates default configuration files required for later steps.

```bash
usage: cisformer generate_default_config [-h]
```

**Arguments:** None

After execution, a new directory named `cisformer_config` will be created in the current working directory. It contains the following three configuration files:

* `accelerate_config.yaml`
* `atac2rna_config.yaml`
* `rna2atac_config.yaml`

### accelerate_config.yaml

This is the configuration file for [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index), which Cisformer uses for distributed training. Example content:

```yaml
{
  "compute_environment": "LOCAL_MACHINE",
  "debug": false,
  "distributed_type": "MULTI_GPU",
  "downcast_bf16": true,
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
```

You typically only need to modify:

* `"gpu_ids"`: GPU device IDs to use.
* `"main_process_port"`: Use a unique port if running multiple distributed jobs.

See the [Accelerate launch guide](https://huggingface.co/docs/accelerate/basic_tutorials/launch) for further instructions.

---

### atac2rna_config.yaml

This file defines parameters for the ATAC ➝ RNA prediction task.

```yaml
datapreprocess:
  enc_max_len: 10000
  dec_max_len: 3000
  multiple: 1

model:
  total_gene: 38244
  max_express: 7
  dim: 280
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
  patience: 2
```

**Recommended Parameters**

| Parameter     | Description                                                                                                           |
| ------------- | --------------------------------------------------------------------------------------------------------------------- |
| `enc_max_len` | Max number of expressed cCREs to consider. Higher values improve performance but consume more GPU memory.             |
| `dec_max_len` | Max number of predicted genes. Larger values improve accuracy (especially during inference) but increase GPU usage.   |
| `multiple`    | Number of times to resample input–output pairs per cell. This increases training diversity without using more memory. |
| `max_express` | The top-N gene expression ranks to predict. Larger values produce more detailed results and attention scores.         |
| `batch_size`  | Number of cells per training batch. Adjust based on memory constraints.                                               |
| `patience`    | Early stopping patience: training stops if validation loss doesn't improve after this many epochs.                    |

---

### rna2atac_config.yaml

This file defines parameters for the RNA ➝ ATAC prediction task.

```yaml
datapreprocess:
  enc_max_len: 2048
  dec_max_len: 2048
  multiple: 40 # Recommend: ≥40

model:
  total_gene: 38244
  max_express: 64
  dim: 210
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
  patience: 5
```

**Recommended Parameters**

| Parameter     | Description                                                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `enc_max_len` | Max number of input genes to consider. Larger values improve performance but use more memory.                                |
| `dec_max_len` | Max number of predicted cCREs during training. High values increase memory usage.                                            |
| `multiple`    | Number of times to sample gene/cCRE pairs per cell. Helps cover more loci without raising memory load. **Recommended ≥ 40**. |
| `max_express` | Maximum expression value range considered. Higher values give better results.                                                |
| `batch_size`  | Number of cells per training batch. Tune based on available memory.                                                          |
| `patience`    | Early stopping patience. Training ends early if no improvement on validation loss.                                           |

---

## data_preprocess
This command is used to preprocess paired RNA and ATAC data for model training or inference.

```
usage: cisformer data_preprocess [-h] -r RNA -a ATAC [-c CONFIG] [-s SAVE_DIR] [--cnt CNT] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                                 [--atac2rna] [--manually] [--shuffle]

options:
  -h, --help            show this help message and exit
  -r RNA, --rna RNA     Should be .h5ad format
  -a ATAC, --atac ATAC  Should be .h5ad format
  -c CONFIG, --config CONFIG
                        Config file
  -s SAVE_DIR, --save_dir SAVE_DIR
                        Save directory
  --cnt CNT             Number of cells per output file
  --batch_size BATCH_SIZE
                        Batch size
  --num_workers NUM_WORKERS
                        Number of workers
  --atac2rna            Process ATAC to RNA
  --manually            Manual mode
  --shuffle             Shuffle data
```

**Required arguments**

* `-r`: Input RNA expression matrix in `.h5ad` format (AnnData object).
* `-a`: Input binary ATAC accessibility matrix in `.h5ad` format.

> ⚠️ Note: Cell barcodes in RNA and ATAC files must match, though they can be in different orders.

**Optional arguments**

* `-c`: Path to the config file. Defaults to `cisformer_config/rna2atac_config.yaml` or `atac2rna_config.yaml` depending on direction.
* `-s`: Output directory to save preprocessed files. Defaults to the current working directory.
* `--cnt`: Maximum number of cells per output `.pt` file. A high number may increase memory usage and reduce file I/O efficiency. Default is 10000 (applies to training sets only).
* `--batch_size`: Number of cells processed per iteration. Default is 10. Adjust according to available memory.
* `--num_workers`: Number of parallel workers used for processing. Default is 10.
* `--atac2rna`: By default, preprocessing is done in the RNA-to-ATAC direction. Use this flag to switch to ATAC-to-RNA.
* `--manually`: Disables automatic train/val/test splitting (default 8:2) and prevents automatic setting of `multiple=1` in config for val/test sets.
* `--shuffle`: If set, cell order in the dataset will be randomized (with identical shuffling for ATAC and RNA to preserve pairing). This improves model training performance but is **not recommended for test sets** as it breaks barcode alignment required for downstream analysis.

> 💡 If `--shuffle` is not set, a `cell_info.tsv` file will be generated, which is required for downstream steps.

**Recommended usage**

```bash
cisformer data_preprocess -r -a -s [--atac2rna]
```

---

## atac2rna_train

```
usage: cisformer atac2rna_train [-h] -d DATA_DIR -n NAME [-s SAVE] [-c CONFIG_FILE] [-m MODEL_PARAMETERS]

options:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        Data directory
  -n NAME, --name NAME  Project name
  -s SAVE, --save SAVE  Save directory
  -c CONFIG_FILE, --config_file CONFIG_FILE
                        Config file
  -m MODEL_PARAMETERS, --model_parameters MODEL_PARAMETERS
                        Load previous model
```

**Function overview**

This command trains the ATAC-to-RNA prediction model using the preprocessed dataset.

**Required arguments**

* `-d`: Path to the directory containing the preprocessed training `.pt` files (generated by `data_preprocess`).
* `-n`: Project name. The final model directory will be named using this project name and training timestamp.

**Optional arguments**

* `-s`: Output directory to save trained model weights. Defaults to `./save`.
* `-c`: Model config file. Defaults to `cisformer_config/atac2rna_config.yaml`.
* `-m`: Path to previously trained model parameters to resume training. Configs must match the current run.

**Recommended usage**

```bash
cisformer atac2rna_train -d -n -s
```

> 💡 Model weights are saved at the end of every epoch. The best-performing model is usually from the last epoch.

---

## atac2rna_predict
This command uses a trained ATAC-to-RNA model to perform RNA expression prediction on preprocessed ATAC test data.

```
usage: cisformer atac2rna_predict [-h] -d DATA -m MODEL_PARAMETERS [-o OUTPUT_DIR] [-n NAME] [-c CONFIG_FILE]

options:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Data file
  -m MODEL_PARAMETERS, --model_parameters MODEL_PARAMETERS
                        Previous trained model parameters
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory
  -n NAME, --name NAME  Output name
  -c CONFIG_FILE, --config_file CONFIG_FILE
                        Config file
```

**Required arguments**

* `-d`: Preprocessed ATAC test set (`.pt` file, generated by `data_preprocess`). The same directory must contain a `cell_info.tsv` file (only available if `--shuffle` was **not** set).
* `-m`: Trained model parameters from a previous training run.

**Optional arguments**

* `-o`: Output directory. Defaults to `./output`.
* `-n`: Output file name. Defaults to `cisformer_predicted_rna`.
* `-c`: Model config file. Defaults to `cisformer_config/atac2rna_config.yaml`.

**Recommended usage**

```bash
cisformer atac2rna_predict -d -m -o
```

> ⚠️ Only a fixed number of high-expression genes (defined by `dec_max_len` in the config file) will be predicted. If this value is too small, clustering and downstream analyses may be suboptimal.

**Output**

The prediction result is saved in `.h5ad` format and can be directly loaded using Scanpy:

```python
import scanpy as sc
predict_rna = sc.read_h5ad("output/cisformer_predicted_rna.h5ad")
print(predicted_rna)
```

Example output:

```
AnnData object with n_obs × n_vars = 20 × 38244
    obs: 'cell_anno_rna', 'n_genes_rna', 'cell_anno_atac', 'n_genes_atac'
```

---

## atac2rna_link

The `atac2rna_link` module generates cell-type-specific `AnnData` files that represent the associations between cis-regulatory elements (cCREs) and genes within each cell type.

**Command-Line Usage**

```
usage: cisformer atac2rna_link [-h] -d DATA_PATH -c CELLTYPE_INFO -m MODEL_PARAMETERS [-o OUTPUT_DIR] [-n NUM_OF_CELLS] [--config CONFIG]
                               [--distance DISTANCE]

options:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
                        Data path
  -c CELLTYPE_INFO, --celltype_info CELLTYPE_INFO
                        Cell type info
  -m MODEL_PARAMETERS, --model_parameters MODEL_PARAMETERS
                        Previous trained model parameters
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory
  -n NUM_OF_CELLS, --num_of_cells NUM_OF_CELLS
                        Number of cells
  --config CONFIG       Config file
  --distance DISTANCE   Distance threshold
```

**Required arguments**

* `-d`, `--data_path`: Path to the preprocessed `.pt` test dataset (generated by `data_preprocess`). This must be a file (not a directory), and a corresponding `cell-info.tsv` file must exist in the same directory (generated automatically by `data_preprocess` when the `--shuffle` flag is not set).
* `-c`, `--celltype_info`: A TSV file specifying the cell types. The first column should contain cell barcodes, and the second column should specify the corresponding cell type. No header row should be included.
* `-m`, `--model_parameters`: Path to a previously trained model's parameters. The configuration used for training must match the current configuration.

**Optional arguments**

* `-o`, `--output_dir`: Directory to store the output files. Default is `./output`.
* `-n`, `--num_of_cells`: Maximum number of cells to consider per cell type.
* `--config`: Path to the model configuration YAML file. Default is `cisformer_config/atac2rna_config.yaml`.
* `--distance`: Maximum genomic distance (in base pairs) from each gene within which to consider potential enhancer regions. Default is 250,000 bp.

**Recommended Argument Set**

```bash
cisformer atac2rna_link -d -c -m -o
```

> ⚠️ If the dataset contains too few cells, it is possible that not all cell types will be present in the test set. In such cases, you should manually split the dataset and run `data_preprocess` manually:

```bash
cisformer data_preprocess -r -a -c cisformer_config/atac2rna_config.yaml --cnt 1e10 --atac2rna --manually
```

Since the model parameters are fixed after training, increasing the number of cells in each type does **not** significantly affect the linkage results—as long as each gene and cCRE in a given cell type is represented. For optimal efficiency, we recommend using **100–200 cells per cell type**, e.g., `-n 100` or `-n 200`.

**Output Files**

Within the specified output directory, a `cisformer_link` folder will be created, containing two main types of files:

* `<celltype>_<distance>_correlation_matrix.h5ad`:
  The core output file representing the correlation (association strength) between cCREs and genes for each cell type. Rows represent genes, columns represent cCREs.

  Example usage in Python:

  ```python
  import anndata
  corr_mtx = anndata.read_h5ad("output/cisformer_link/CD4-intermediate_250kbp_correlation_matrix.h5ad")
  print(corr_mtx)
  ```

  Output:

  ```
  AnnData object with n_obs × n_vars = 38244 × 1033239
  ```

  * `corr_mtx.obs_names`: List of gene names.

  * `corr_mtx.var_names`: Genomic coordinates of cCREs.

  * `corr_mtx.X`: Sparse matrix of correlation values.

  > The linkage strengths between different cell types in the same dataset are directly comparable.

* `cell_num.pkl`:
  A Python pickle file storing the number of cells used for each cell type during inference. This can be loaded using the `pickle` module.

---

## rna2atac_train

Trains a model to predict chromatin accessibility from RNA expression.

**Command-Line Usage**

```
usage: cisformer rna2atac_train [-h] -t TRAIN_DATA_DIR -v VAL_DATA_DIR -n NAME [-s SAVE] [-c CONFIG_FILE] [-m MODEL_PARAMETERS]

options:
  -h, --help            show this help message and exit
  -t TRAIN_DATA_DIR, --train_data_dir TRAIN_DATA_DIR
                        Training data directory
  -v VAL_DATA_DIR, --val_data_dir VAL_DATA_DIR
                        Validation data directory
  -n NAME, --name NAME  Project name
  -s SAVE, --save SAVE  Save directory
  -c CONFIG_FILE, --config_file CONFIG_FILE
                        Config file
  -m MODEL_PARAMETERS, --model_parameters MODEL_PARAMETERS
                        Load previous model
```

**Required arguments**

* `-t`, `--train_data_dir`: Directory containing the preprocessed training data (generated using `data_preprocess`).
* `-v`, `--val_data_dir`: Directory containing the preprocessed validation data.
* `-n`, `--name`: Project name. This name will be used (along with the training timestamp) to name the folder where model checkpoints are saved.

**Optional arguments**

* `-s`, `--save`: Directory in which to save training checkpoints. Default is `./save`.
* `-c`, `--config_file`: Model configuration file. Default is `cisformer_config/rna2atac_config.yaml`.
* `-m`, `--model_parameters`: Load a previously trained model for continued training or fine-tuning. The configuration must match the current one.

**Recommended Argument Set**

```bash
cisformer rna2atac_train -t -v -s -n
```

> 💡 The model parameters from every training epoch will be saved. Typically, the checkpoint from the **last epoch** provides the best performance.

---

## rna2atac_predict

Predicts ATAC-seq (chromatin accessibility) profiles from scRNA-seq input using a pretrained model.

**Command-Line Usage**

```
usage: cisformer rna2atac_predict [-h] -r RNA_FILE -m MODEL_PARAMETERS [-o OUTPUT_DIR] [-n NAME] [-c CONFIG_FILE] [--rna_len RNA_LEN]
                                  [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]

options:
  -h, --help            show this help message and exit
  -r RNA_FILE, --rna_file RNA_FILE
                        Path of rna adata
  -m MODEL_PARAMETERS, --model_parameters MODEL_PARAMETERS
                        Previous trained model parameters
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Load model
  -n NAME, --name NAME  Load model
  -c CONFIG_FILE, --config_file CONFIG_FILE
                        Config file
  --rna_len RNA_LEN     Number of non-zero expressed gene in used
  --batch_size BATCH_SIZE
                        Batch size
  --num_workers NUM_WORKERS
                        Number of workers
```

**Required arguments**

* `-r`, `--rna_file`: Path to an RNA expression matrix in `.h5ad` format (processed using `scanpy`).
* `-m`, `--model_parameters`: Path to a pretrained model's parameters (must match the current configuration).

**Optional arguments**

* `-o`, `--output_dir`: Output directory. Default is `./output`.
* `-n`, `--name`: Filename for the predicted ATAC output. Default is `cisformer_predicted_atac`.
* `-c`, `--config_file`: Configuration file path. Default is `cisformer_config/rna2atac_config.yaml`.
* `--rna_len`: Maximum number of expressed genes to use per cell when predicting ATAC profiles. Default is `3600`.

  > Note: Increasing this value can dramatically increase memory usage.
* `--batch_size`: Number of cells to process per batch. Due to the large number of prediction targets (up to 1 million peaks), keep this small to avoid excessive memory usage. Default is `2`.
* `--num_workers`: Number of worker processes for parallel processing. Also recommended to keep this low. Default is `2`.

**Recommended Argument Set**

```bash
cisformer rna2atac_predict -r -m -o --rna_len
```

**Output**

The output is an `.h5ad` file that can be read using `scanpy`:

```python
import scanpy as sc
predicted_atac = sc.read_h5ad("output/cisformer_predicted_atac.h5ad")
print(predicted_atac)
```

Example output:

```
AnnData object with n_obs × n_vars = 20 × 1033239
    obs: 'cell_anno', 'n_genes'
```