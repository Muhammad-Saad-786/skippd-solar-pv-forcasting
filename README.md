# SKIPP'D Solar Forecasting

This project studies short-term photovoltaic (PV) power forecasting using the
**SKIPP'D (SKy Images and Photovoltaic Power Generation Dataset)** from Stanford.
The repository contains notebooks for data download, exploratory analysis, and
PV-only and sky-image-based deep-learning forecasts.

## Dataset

The project uses the processed 2017–2019 SKIPP'D benchmark dataset:

- Dataset repository: <https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset>
- Dataset paper: <https://doi.org/10.1016/j.solener.2023.03.043>
- Stanford benchmark record: <https://purl.stanford.edu/dj417rh1007>
- Dataset license: Creative Commons Attribution 4.0 International

After downloading and extracting the processed data, the following files are
expected:

```text
trainval_images_log.npy
trainval_pv_log.npy
test_images_log.npy
test_pv_log.npy
times_trainval.npy
times_test.npy
```

The image arrays contain sky-camera observations and the PV arrays contain the
corresponding generation values. The timestamp arrays are required for
day-aware sequence construction and for verifying the sampling interval.

## Repository contents

| Path | Description |
| --- | --- |
| [`01-download.ipynb`](./01-download.ipynb) | Downloads the processed SKIPP'D benchmark and timestamps. |
| [`02-eda.ipynb`](./02-eda.ipynb) | Inspects images, PV generation, timestamps, daily patterns, and data quality. |
| [`forecast/01-lstm.ipynb`](./forecast/01-lstm.ipynb) | PV/time-feature Conv1D + stacked LSTM baseline. |
| [`forecast/02-cnn-lstm.ipynb`](./forecast/02-cnn-lstm.ipynb) | Multimodal sky-image CNN + PV-LSTM baseline. |
| [`SKIPPD_RESEARCH_AND_TRAINING_GUIDE.md`](./SKIPPD_RESEARCH_AND_TRAINING_GUIDE.md) | Detailed research plan, training procedure, evaluation protocol, and journal-paper guidance. |

## Existing forecasting models

The current notebooks use a 16-sample historical window and predict three
future horizons:

```text
15 minutes, 30 minutes, and 60 minutes
```

The PV-only notebook uses raw PV, cyclical time-of-day/year features, and a
rolling-volatility feature. Its model is:

```text
causal Conv1D -> stacked LSTM layers -> shared dense layer
             -> horizon-specific regression heads
```

The multimodal notebook combines:

```text
sky-image CNN features + historical PV sequence -> LSTM fusion -> forecasts
```

Both notebooks use date-blocked validation to prevent overlapping sequences
from the same calendar day being split between training and validation. They
train multiple folds, save the best checkpoints, average fold predictions, and
report horizon-specific MAE and RMSE.

## Running the notebooks

1. Download or extract the SKIPP'D data.
2. Place the six expected `.npy` files in a local data directory.
3. Open [`02-eda.ipynb`](./02-eda.ipynb) and verify:
   - array shapes and dtypes;
   - image channel order and value range;
   - PV missing values and range;
   - timestamp ordering and sampling cadence;
   - train/test date separation.
4. Update the hard-coded data path in the forecasting notebook. On Windows,
   use a raw path such as:

   ```python
   DATA_DIR = r"D:\path\to\data\extracted"
   ```

5. Run the PV-only baseline in
   [`forecast/01-lstm.ipynb`](./forecast/01-lstm.ipynb).
6. Run the image-plus-PV baseline in
   [`forecast/02-cnn-lstm.ipynb`](./forecast/02-cnn-lstm.ipynb).
7. Save model checkpoints, training histories, predictions, and evaluation
   metrics in separate experiment directories.

Before interpreting the results, confirm that one array step really equals one
minute. The notebooks use `[15, 30, 60]` array steps as the forecast horizons,
so this assumption must be checked against `times_trainval.npy` and
`times_test.npy`.

## Research protocol

For a defensible comparison, use the same date folds, preprocessing, target
mask, and untouched test set for every model. Include at least:

1. persistence forecasting;
2. the PV-only LSTM baseline;
3. the CNN-LSTM baseline;
4. the proposed model.

Report MAE, RMSE, normalized errors, persistence skill, bias, per-day
confidence intervals, and results for each forecast horizon. For a journal
paper, also include ablations, cloudy/sunny and ramp-event analysis, parameter
counts, runtime, and repeated-seed results.

The full step-by-step procedure is in
[`SKIPPD_RESEARCH_AND_TRAINING_GUIDE.md`](./SKIPPD_RESEARCH_AND_TRAINING_GUIDE.md).
That guide also explains how to develop a motion-aware or probabilistic model
without claiming unsupported accuracy or leaking information from the test
set.

## Related research

- SKIPP'D dataset paper: <https://arxiv.org/abs/2207.00913>
- Multi-location and transfer-learning study: <https://arxiv.org/abs/2211.02108>
- SkyGPT probabilistic forecasting study: <https://arxiv.org/abs/2306.11682>
