# Release Notes

## Version 1.1.0

### Added

- Added mouse reference support across config generation, preprocessing,
  prediction, and ATAC-to-RNA link inference.
- Added `--species {human,mouse}` to species-sensitive commands.
- Added species-aware config generation. Human configs use `total_gene: 38244`;
  mouse configs use `total_gene: 23234`.

### Changed

- ATAC-to-RNA preprocessing now performs automatic log1p detection. Cisformer
  checks the loaded RNA matrix and prints whether `scanpy.pp.log1p` is applied.
- Improved preprocessing output buffering.
- Improved `atac2rna_link` performance by reducing repeated Python-level peak
  lookup and sparse matrix writes.
- Documentation now clarifies that `atac2rna_link` writes rank-normalized link
  scores rather than raw correlation coefficients.

### Upgrade Notes

- Regenerate configs with `cisformer generate_default_config --species human` or
  `--species mouse` before training new v1.1.0 models.
- Use the same species value for preprocessing, prediction, and link generation.
- Existing human models trained with the old 38,244-gene vocabulary should use
  human configs.

## Version 1.0.1 - 2025-05-14

- Initial PyPI release of Cisformer.
