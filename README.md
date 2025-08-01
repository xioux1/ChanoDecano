# Kaggle Notebook - ChanoDecano

This repository contains the Kaggle notebook `chanodecano.ipynb`. The notebook trains a LightGBM ranking model for the **AeroClub RecSys 2025** competition. All preprocessing helpers and the training pipeline live directly inside the notebook because Kaggle kernels do not allow referencing local modules easily.
Debido a que los kernels de Kaggle no permiten leer módulos externos fácilmente, todas las funciones están dentro del notebook; no se pueden crear scripts aparte.

## Notebook Overview

The notebook defines several utilities for data preparation and model training:

- `plot_feature_importances` – utility for visualising the top features.
- `add_is_min_segments` – flag itineraries that have the fewest segments per query (defined within the notebook).
- `in_missing_segments` – cuenta cuántas columnas de segmentos de vuelta faltan.
- Functions such as `_normalise_utc_offset`, `smart_fill_numeric`, `unify_nan_strategy`, and `reduce_mem_usage` to clean and optimise the data.
- `calculate_hit_rate_at_3` and `lgb_hit_rate_at_3` – custom HitRate@3 metric used during cross-validation.
- `price_vs_carrier_mean` – relación del precio con el promedio por aerolínea de marketing.
- `sex_baggage_total` – cruza el sexo del pasajero con la cantidad total de equipaje incluido.
- `load_data`, `preprocess_dataframe`, `prepare_matrices`, `encode_categoricals`, and `train_model` build the full LightGBM ranking pipeline. The dataset is read from the Kaggle input directory (e.g. `/kaggle/input/aeroclub-recsys-2025/train.parquet`).
- A `main()` routine orchestrates the workflow and saves `submission.csv` and `submission.parquet`.

Important functions are defined in-place so the notebook runs standalone in Kaggle. Calling code outside the notebook is not supported.

## Running

1. Upload `chanodecano.ipynb` to Kaggle and open it in a new kernel.
2. Run all cells from top to bottom.
3. The notebook will load the competition data, perform feature engineering, train a LightGBM ranker with `GroupKFold`, and produce the `submission.csv` for upload.

A helper `readme()` function inside the notebook returns a short description of these utilities:

```python
print(readme())
```

## Outputs

- `submission.csv` / `submission.parquet` – predictions for the public leaderboard.
- `feature_importances.csv` – list of features ranked by average gain.

All code stays within the notebook due to Kaggle's environment restrictions on external files.

## Experiments

`chanodecano.py` now has all duration-related features commented out. This allows
testing model performance without the dominant `duration` signal. Search for
`duration-related` in the script for details.
