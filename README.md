# MODIS_to_Sentinel

Research prototype for generating 10 m land-cover maps from Sentinel-2 imagery using MODIS-derived low-resolution labels.

## Overview

The project implements a MODIS-to-Sentinel mapping workflow:

1. use MODIS land-cover maps as coarse labels;
2. extract reliable training samples from these labels;
3. stack Sentinel-2 spectral features;
4. train a Random Forest classifier;
5. generate a higher-resolution thematic map;
6. optionally validate the result against ESA WorldCover.

The research target is to transfer labels from 230 m MODIS-based maps to 10 m Sentinel-2 data. The main tested area is Samara Oblast for 2020:

![Generated land cover map of Samara Oblast for 2020 in 6 classes](docs/modis_transition.gif)

## Repository structure

```text
MODIS_to_Sentinel/
├── data/              # Input and output data
├── external/          # External helper files
├── researches/        # Research notebooks and experiments
├── scripts/           # CLI pipelines and utilities
├── src/               # Core source modules
├── .gitignore
├── environment.yml
├── main.py
├── requirements.txt
└── README.md
```

## Core scripts and modules

### Scripts

| File | Purpose |
|---|---|
| `scripts/M2S_create.py` | Main CLI pipeline: sampling, training, prediction, validation. |
| `scripts/M2S_create_fullstudy.py` | Experimental pipeline for larger cross-tile studies. |
| `scripts/M2S_create_secure_downgrade.py` | Experimental downgrade-based pipeline. |
| `scripts/M2S_utils.py` | Utilities for class reassignment, tile merging, and validation. |

### Modules

| File | Purpose |
|---|---|
| `src/prepare_ds.py` | Prepares rasters and training samples. |
| `src/train_ml.py` | Trains the machine-learning model. |
| `src/create_map.py` | Predicts and saves the output map. |
| `src/validation.py` | Computes validation reports and error maps. |
| `src/utils.py` | Shared raster, path, and class utilities. |
| `src/visualisation.py` | Visualization helpers. |

## Input data

The pipeline expects GeoTIFF rasters.

### Sentinel-2 features

Recommended channels:

```text
b2,b3,b4,b8,b11
```

Meaning:

| Channel | Meaning |
|---|---|
| `b2` | Blue |
| `b3` | Green |
| `b4` | Red |
| `b8` | Near infrared, NIR |
| `b11` | Short-wave infrared, SWIR1 |

The research workflow used 90-day cloud-free median composites for spring, summer, and autumn.

### MODIS labels

MODIS-based land-cover maps are used as low-resolution labels. The research work uses maps with 230 m spatial resolution and remaps the original classes into 6 target classes.

### Validation maps

ESA WorldCover 10 m maps can be used as external reference data. In the research work, WorldCover was used for 2020 validation.

## Basic usage

Run the main pipeline:

```bash
python scripts/M2S_create.py \
  --src path/to/sentinel2_data \
  --labels path/to/modis_labels \
  --year 2020 \
  --out data/output \
  --channels b2,b3,b4,b8,b11 \
  --method secure \
  --chunk 300 \
  --verbose
```

Run with validation:

```bash
python scripts/M2S_create.py \
  --src path/to/sentinel2_data \
  --labels path/to/modis_labels \
  --validate_by path/to/worldcover_maps \
  --year 2020 \
  --out data/output \
  --channels b2,b3,b4,b8,b11 \
  --method secure \
  --chunk 300 \
  --verbose
```

Force recomputation:

```bash
python scripts/M2S_create.py \
  --src path/to/sentinel2_data \
  --labels path/to/modis_labels \
  --year 2020 \
  --out data/output \
  --force
```

## CLI arguments

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--src` | yes | — | Path to Sentinel-2 GeoTIFF data. |
| `--labels` | yes | — | Path to MODIS label maps. |
| `--year` | yes | — | Year or year range, for example `2020` or `2018-2020`. |
| `--out` | no | current directory | Output directory. |
| `--chunk` | no | `300` | Prediction chunk size in MB. |
| `--method` | no | `secure` | Sampling method: `secure` or `median`. |
| `--channels` | no | `b2,b3,b4,b8,b11` | Sentinel-2 channels used as features. |
| `--validate_by` | no | `None` | Path to reference maps. |
| `--verbose` | no | `False` | Print detailed logs. |
| `--force` | no | `False` | Overwrite existing outputs. |
| `--ignore_error` | no | `False` | Continue after tile-level errors. |

## Output

Typical output structure:

```text
data/output/
├── processed/
├── models/
├── cache/
├── <tile>/
│   └── <year>/
│       ├── M2S_<tile>_<year>.tif
│       └── etalon_<name>.tif
├── train_classification_report.csv
└── etalon_classification_report.csv
```

Main output:

```text
M2S_<tile>_<year>.tif
```

This file is a generated land-cover map at Sentinel-2 spatial resolution.

Reported research baseline for Samara Oblast, 2020, using the reliable sampling method:

| Metric group | Precision | Recall | F1 |
|---|---:|---:|---:|
| Macro avg | 0.66 | 0.73 | 0.67 |
| Weighted avg | 0.90 | 0.86 | 0.88 |

These values describe the tested research setup, not a guaranteed result for other regions or years.

## Class palette

| Class ID | Class | Color |
|---:|---|---|
| 0 | No data / background | `#000000` |
| 1 | Forest land | `#018f33` |
| 2 | Cultivated land | `#fc9b01` |
| 3 | Grassland / pastures | `#fef901` |
| 4 | Wetlands / water | `#1f05fe` |
| 5 | Settlements | `#dc1010` |
| 6 | Other land | `#01f7fe` |

## Utility scripts

Reassign MODIS classes:

```bash
python scripts/M2S_utils.py \
  --src path/to/modis_maps \
  --out path/to/reassigned_maps \
  --assign_cache_classes \
  --force
```

Merge GeoTIFF tiles:

```bash
python scripts/M2S_utils.py \
  --src "path/to/maps/*.tif" \
  --out path/to/merged.tif \
  --stick
```

Validate a generated map:

```bash
python scripts/M2S_utils.py \
  --src path/to/result_map.tif \
  --out path/to/validation_output/ \
  --validate path/to/reference_map.tif
```

## Recommended workflow

1. Prepare Sentinel-2 GeoTIFF bands.
2. Prepare MODIS label maps.
3. Reassign source classes to the 6 target classes.
4. Run one tile for one year.
5. Check the generated map visually.
6. Check `train_classification_report.csv`.
7. Validate against WorldCover if available.
8. Scale to several tiles or years.
9. Merge output tiles if a regional map is required.

The recommended default method is `secure`. It samples reliable pixels away from class transition boundaries and was selected as the most stable method in the research work.

## License

No license file is currently provided. Add a license before public reuse or distribution.