# Quick Start

To quickly get started and better understand each step of the Cisformer workflow, we recommend downloading the test files from the [Cisformer GitHub repository](https://github.com/qihang-zou/Cisformer/tree/main/test_data) and following the examples below.

## Generate Default Config Files

```bash
cisformer generate_default_config
```

This command creates a `cisformer_config` folder in the current directory, containing:

* `accelerate_config.yaml`
* `atac2rna_config.yaml`
* `rna2atac_config.yaml`

Cisformer uses [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index) for distributed training. You may need to modify `accelerate_config.yaml` based on your environment. See the [Accelerate launch guide](https://huggingface.co/docs/accelerate/basic_tutorials/launch) for more details.

---

## RNA ➝ ATAC

### 1. Configure Parameters (Optional)

Edit the RNA2ATAC configuration:

```text
cisformer_config/rna2atac_config.yaml
```
**It is recommanded to modify the parameter `multiple` to at least 40 to get a good performance.**
Refer to the [Usage](usage.md#rna2atac_configyaml) page for detailed explanations of each parameter.

### 2. Preprocess Data

Cisformer expects input scRNA-seq and scATAC-seq data in [Scanpy `.h5ad` format](https://scanpy.readthedocs.io/en/stable/tutorials/index.html).

```bash
cisformer data_preprocess -r test_data/rna.h5ad -a test_data/atac.h5ad -s preprocessed_dataset
```

* `-r`: RNA `.h5ad` file
* `-a`: ATAC `.h5ad` file
* `-s`: output directory

See the [Usage](usage.md#data_preprocess) page for additional options and output details.

### 3. Train the Model

```bash
cisformer rna2atac_train -t preprocessed_dataset/cisformer_rna2atac_train_dataset -v preprocessed_dataset/cisformer_rna2atac_val_dataset -n rna2atac_test
```

* `-t`: training dataset path
* `-v`: validation dataset path
* `-n`: project name

A `save` directory will be created (can be customized with `-s`). Inside, a subdirectory like `2025-05-12_rna2atac_test` will contain the trained models. The model from the final epoch is typically the best.

Refer to the [Usage](usage.md#rna2atac_train) page for more training options and output details.

### 4. Run Prediction

```bash
cisformer rna2atac_predict -r preprocessed_dataset/test_rna.h5ad -m save/2025-05-12_rna2atac_test/epoch34/pytorch_model.bin
```

* `-r`: RNA `.h5ad` file for prediction
* `-m`: trained model checkpoint

The predicted ATAC matrix will be saved to `output/cisformer_predicted_atac.h5ad`.

Refer to the [Usage](usage.md#rna2atac_predict) page for more training options and output details.

---

## ATAC ➝ RNA

### 1. Configure Parameters (Optional)

Edit the ATAC2RNA configuration:

```text
cisformer_config/atac2rna_config.yaml
```

Refer to the [Usage](usage.md#atac2rna_configyaml) page for parameter details.

### 2. Preprocess Data

```bash
cisformer data_preprocess -r test_data/rna.h5ad -a test_data/atac.h5ad -s preprocessed_dataset --atac2rna
```

* `--atac2rna`: specifies the ATAC2RNA direction
* Other arguments are the same as in RNA2ATAC

See the [Usage](usage.md#data_preprocess) page for additional options and output details.

### 3. Train the Model

```bash
cisformer atac2rna_train -d preprocessed_dataset/cisformer_atac2rna_train_dataset -n atac2rna_test
```

* `-d`: training dataset path
* `-n`: project name

A subdirectory like `2025-05-12_atac2rna_test` will be created under `save`, containing the trained model.

Refer to the [Usage](usage.md#atac2rna_train) page for more training options and output details.

### 4. Run Prediction (Optional)

```bash
cisformer atac2rna_predict -d preprocessed_dataset/cisformer_atac2rna_test_dataset/atac2rna_0.pt -m save/2025-05-12_atac2rna_test/epoch30/pytorch_model.bin
```

* `-d`: `.pt` test dataset
* `-m`: trained model checkpoint

The predicted RNA matrix will be saved to `output/cisformer_predicted_rna.h5ad`.

Refer to the [Usage](usage.md#atac2rna_predict) page for more training options and output details.

### 5. Link Cis-Regulatory Elements to Genes

```bash
cisformer atac2rna_link -d preprocessed_dataset/cisformer_atac2rna_test_dataset/atac2rna_0.pt -m save/2025-05-12_atac2rna_test/epoch30/pytorch_model.bin -c test_data/celltype_info.tsv
```

* `-d`: test dataset `.pt` file (must be accompanied by `cell_info.tsv` in the same folder)
* `-m`: trained model
* `-c`: TSV file mapping cell barcodes to cell types (no header)

Example `celltype_info.tsv`:

```
GTACCGGGTATACTGG-1	CD14 Mono
ACTGAATGTCACCAAA-1	cDC2
AACCTTGCAAACTGTT-1	CD14 Mono
...
```

Linked cis-regulatory and gene information will be saved in `output/cisformer_link/` as `.h5ad` files, organized by cell type.

Refer to the [Usage](usage.md#atac2rna_link) page for more training options and output details.