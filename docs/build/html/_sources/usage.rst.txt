Quick Start
===========

Generating Configuration Files
------------------------------

.. code-block:: bash

    cisformer generate_default_config

This creates a `cisformer_config/` directory with three config files:
- accelerate_config.yaml
- rna2atac_config.yaml
- atac2rna_config.yaml

Refer to `accelerate` documentation for distributed setup if needed.

RNA ➝ ATAC Mode
---------------

1. Preprocessing:

.. code-block:: bash

    cisformer data_preprocess -r test_data/rna.h5ad -a test_data/atac.h5ad -s preprocessed_dataset

2. Training:

.. code-block:: bash

    cisformer rna2atac_train -t preprocessed_dataset/cisformer_rna2atac_train_dataset -v preprocessed_dataset/cisformer_rna2atac_val_dataset -n rna2atac_test

3. Prediction:

.. code-block:: bash

    cisformer rna2atac_predict -r preprocessed_dataset/test_rna.h5ad -m save/2025-05-12_rna2atac_test/epoch34/pytorch_model.bin

ATAC ➝ RNA Mode
---------------

1. Preprocessing:

.. code-block:: bash

    cisformer data_preprocess -r test_data/rna.h5ad -a test_data/atac.h5ad -s preprocessed_dataset --atac2rna

2. Training:

.. code-block:: bash

    cisformer atac2rna_train -d preprocessed_dataset/cisformer_atac2rna_train_dataset -n atac2rna_test

3. Prediction:

.. code-block:: bash

    cisformer atac2rna_predict -d preprocessed_dataset/cisformer_atac2rna_test_dataset/atac2rna_0.pt -m save/2025-05-12_atac2rna_test/epoch30/pytorch_model.bin

4. Linking Cis-Regulators and Genes:

.. code-block:: bash

    cisformer atac2rna_link -d preprocessed_dataset/cisformer_atac2rna_test_dataset/atac2rna_0.pt -m save/2025-05-12_atac2rna_test/epoch30/pytorch_model.bin -c test_data/celltype_info.tsv
