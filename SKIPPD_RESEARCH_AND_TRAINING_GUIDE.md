# SKIPP'D Solar Forecasting: Reproducible Research and Training Guide

## 1. Purpose and scope

This document is a practical research plan for the SKIPP'D (SKy Images and Photovoltaic Power Generation Dataset) project in this repository. It explains:

1. What the supplied notebooks currently do.
2. What the original SKIPP'D work and related work established.
3. How to train, evaluate, and compare models without test leakage.
4. Which experiments could make a defensible journal contribution.
5. What must be reported in a paper.

This is a research roadmap, not a guarantee of a particular score. No honest forecasting study can guarantee 100% accuracy: future cloud motion is partly unpredictable, measurements contain noise, and the achievable error depends on the test period and metric. A paper can beat a previous result only after it uses the same data split, target definition, preprocessing, and metric, and the improvement is statistically supported.

The strongest goal should be **a fair, reproducible, statistically significant improvement over clearly defined baselines**, not a very small improvement obtained by changing the evaluation protocol.

---

## 2. Evidence inspected in this repository

The repository contains:

- [`01-download.ipynb`](./01-download.ipynb): downloads the processed 2017–2019 SKIPP'D benchmark HDF5 file and timestamp arrays.
- [`02-eda.ipynb`](./02-eda.ipynb): loads HDF5 or extracted `.npy` files and explores timestamps, PV, images, daily patterns, and weather/solar-position-related quantities.
- [`forecast/01-lstm.ipynb`](./forecast/01-lstm.ipynb): PV-only/time-feature forecasting.
- [`forecast/02-cnn-lstm.ipynb`](./forecast/02-cnn-lstm.ipynb): image plus PV forecasting.
- [`README.md`](./README.md): identifies the Stanford Solar Forecasting Dataset.
- [`paper/main.pdf`](./paper/main.pdf) and [`skippd-research paper.pdf`](./skippd-research%20paper.pdf): local papers supplied with the project.

The notebook source was inspected directly. The descriptions below refer to the code that is present, not to an assumed implementation.

### 2.1 Dataset files expected by the notebooks

The extracted data directory is expected to contain:

```text
trainval_images_log.npy
trainval_pv_log.npy
test_images_log.npy
test_pv_log.npy
times_trainval.npy
times_test.npy
```

The notebook currently uses Linux/Colab paths such as `/home/saadasim_1/content/data/extracted/`. On Windows, replace this with the real local path, for example:

```python
from pathlib import Path

DATA_DIR = Path(r"D:\path\to\data\extracted")
```

Do not copy the literal `/home/...` path into a Windows run.

The dataset README links the benchmark and dataset paper:

- Dataset repository: <https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset>
- Dataset paper: <https://doi.org/10.1016/j.solener.2023.03.043>
- Benchmark data: <https://purl.stanford.edu/dj417rh1007>
- SKIPP'D arXiv description: <https://arxiv.org/abs/2207.00913>

The repository README describes SKIPP'D as three years of quality-controlled, down-sampled sky images paired with PV generation data, with a benchmark split and code for processing and baseline modelling.

---

## 3. What the existing notebooks actually implement

### 3.1 Common forecasting setup

Both forecasting notebooks use:

```text
History window: 16 one-minute samples
Forecast horizons: 15, 30, and 60 steps
Assumed interpretation: 15, 30, and 60 minutes if the data cadence is one minute
Maximum epochs: 200
Batch size: 256
Adam learning rate: 0.001
Early stopping patience: 10
Validation: 10 date-block folds
Loss: mean squared error (MSE)
Metric during training: mean absolute error (MAE)
```

The code groups samples by calendar date. It creates a sample only when the input history and target remain within one day. This is important because a sequence should not use the previous evening as the history for the next morning.

The targets are the raw PV value at each horizon. The test sequence builder can create partial end-of-day samples; unavailable long-horizon targets are represented by `NaN` and excluded from the corresponding metric.

### 3.2 PV-only/time-feature model

The file [`forecast/01-lstm.ipynb`](./forecast/01-lstm.ipynb) loads PV and timestamps, then constructs six input features:

1. Raw PV value.
2. Sine of time of day.
3. Cosine of time of day.
4. Sine of day of year.
5. Cosine of day of year.
6. Five-sample rolling PV standard deviation.

The model function is named `build_lstm_forecast_model`, but it is not a plain LSTM. Its effective structure is:

```text
Input (16 time steps × 6 features)
    -> causal Conv1D(32, kernel_size=3, ReLU)
    -> LSTM(96, return_sequences=True)
    -> LSTM(96)
    -> Dense(128, ReLU)
    -> three independent Dense(64, ReLU) heads
    -> one linear output per horizon
```

The three heads allow the network to learn horizon-specific corrections. The model is trained separately in each of ten date folds. Each fold's best checkpoint is then used to predict the test set, and the predictions are averaged.

### 3.3 Image plus PV model

The file [`forecast/02-cnn-lstm.ipynb`](./forecast/02-cnn-lstm.ipynb) uses:

```text
Image sequence: 16 × 64 × 64 × 3
PV sequence:    16 × 1
```

Its visual branch applies:

```text
Rescaling(1/255)
Conv2D -> BatchNorm -> MaxPooling
Conv2D -> BatchNorm -> MaxPooling
GlobalAveragePooling2D
Dense visual projection
```

The visual features are combined with the PV history and passed through stacked LSTM layers. A dense regression head outputs the three horizons. The notebook streams image batches from a memory-mapped array using a generator to avoid loading all image sequences into RAM.

This is a sensible baseline because:

- CNN layers learn cloud texture, brightness, and spatial structure.
- A sequence of CNN features can represent cloud evolution.
- PV history provides the current generation state and corrects image-only ambiguity.
- LSTM layers model temporal persistence and short-term ramps.
- Multi-horizon output shares information among 15-, 30-, and 60-minute forecasts.

However, the current model is still a baseline rather than a guaranteed state-of-the-art method. It does not explicitly estimate cloud motion, uncertainty, or physical irradiance constraints.

### 3.4 Current validation and ensemble procedure

The notebooks:

1. Shuffle unique calendar dates with a fixed seed.
2. Partition dates into ten folds.
3. Put complete dates, rather than individual sequences, into validation.
4. Train one model per fold.
5. Save the best validation-loss checkpoint.
6. Predict the fixed test set with all fold models.
7. Average fold predictions.
8. Compute RMSE and MAE separately for each horizon, ignoring missing targets.

Grouping by date is much safer than randomly splitting overlapping windows. Randomly splitting adjacent windows would put near-duplicate histories in both train and validation and inflate performance.

---

## 4. What prior research contributes

### 4.1 SKIPP'D dataset paper

The SKIPP'D work addresses a central reproducibility problem: researchers need a standardized paired dataset of sky images and PV power. It provides processed benchmark data, higher-resolution/raw data, and baseline code. Its purpose is to make short-term image-based solar forecasting comparisons possible.

Use the dataset paper to understand:

- how images and PV were collected and quality-controlled;
- the benchmark sampling frequency and split;
- the definition of the forecast task;
- the original SUNSET baseline and its comparison protocol.

Do not compare a new number with a number from a different task definition. A 15-minute forecast, a 1-hour forecast, a nowcast, and a probabilistic interval forecast are different tasks.

### 4.2 Transfer learning and multiple locations

The related multi-location work, [arXiv:2211.02108](https://arxiv.org/abs/2211.02108), compares local training, joint/global training, and transfer learning. Its main research lesson is that:

- local models can work well at their own site;
- off-site deployment can produce large errors;
- global pretraining plus transfer can reduce the amount of target-site data needed;
- geographic and climate diversity can improve generalization.

For a SKIPP'D-only paper, this motivates experiments with temporal transfer, seasonal transfer, and external pretraining only if the data and protocol are available and clearly documented.

### 4.3 Probabilistic forecasting and SkyGPT

SkyGPT, [arXiv:2306.11682](https://arxiv.org/abs/2306.11682), treats future cloud motion as stochastic. It generates multiple possible future sky videos and evaluates probabilistic PV forecasts with CRPS and Winkler score. The published abstract reports better reliability and sharpness than its cited baselines for a 15-minute probabilistic task.

The important lesson is not “use generative AI.” It is that a single point prediction cannot express uncertainty during ambiguous cloud transitions. A publishable improvement could instead be a well-calibrated probabilistic model using quantile regression or an ensemble, which is easier to audit than a large generative system.

### 4.4 Research gap suitable for this project

A realistic contribution should combine several of the following:

1. **Strong leakage-safe evaluation** with persistence and climatology baselines.
2. **Motion-aware image representation**, not only framewise CNN features.
3. **Physics-aware target normalization or constraints**.
4. **Uncertainty estimates** and calibration, not only RMSE.
5. **Cloud-condition and ramp-event analysis**.
6. **Ablation evidence** showing which component creates the gain.
7. **Reproducible code, seeds, preprocessing, and split manifests**.

Do not claim novelty merely because a different layer count or optimizer was used.

---

## 5. First priority: audit the data before training

Create and run a small audit notebook before modifying the model. The audit must save its output (shapes, dtypes, date ranges, cadence, missing counts, and value ranges) as a JSON or CSV artifact.

### 5.1 Load and inspect arrays

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_DIR = Path(r"D:\path\to\data\extracted")

names = [
    "trainval_images_log.npy",
    "trainval_pv_log.npy",
    "test_images_log.npy",
    "test_pv_log.npy",
    "times_trainval.npy",
    "times_test.npy",
]

for name in names:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)

trainval_pv = np.load(DATA_DIR / "trainval_pv_log.npy")
test_pv = np.load(DATA_DIR / "test_pv_log.npy")
trainval_images = np.load(DATA_DIR / "trainval_images_log.npy", mmap_mode="r")
test_images = np.load(DATA_DIR / "test_images_log.npy", mmap_mode="r")
times_trainval = pd.to_datetime(
    np.load(DATA_DIR / "times_trainval.npy", allow_pickle=True)
)
times_test = pd.to_datetime(
    np.load(DATA_DIR / "times_test.npy", allow_pickle=True)
)

print("trainval PV:", trainval_pv.shape, trainval_pv.dtype)
print("test PV:    ", test_pv.shape, test_pv.dtype)
print("trainval images:", trainval_images.shape, trainval_images.dtype)
print("test images:    ", test_images.shape, test_images.dtype)
print("trainval time:", times_trainval[0], times_trainval[-1])
print("test time:    ", times_test[0], times_test[-1])
```

### 5.2 Verify cadence instead of assuming it

```python
def cadence_report(times):
    delta = pd.Series(times).diff().dropna().dt.total_seconds() / 60
    return {
        "min_minutes": float(delta.min()),
        "median_minutes": float(delta.median()),
        "max_minutes": float(delta.max()),
        "common_minutes": delta.round(6).value_counts().head(10).to_dict(),
    }

print(cadence_report(times_trainval))
print(cadence_report(times_test))
```

If the median cadence is not exactly one minute, `FORECAST_HORIZONS = [15, 30, 60]` does not mean 15/30/60 minutes. Convert minutes into step counts after verifying the sampling interval.

### 5.3 Verify alignment

Check that:

```python
len(trainval_pv) == len(trainval_images) == len(times_trainval)
len(test_pv) == len(test_images) == len(times_test)
```

Also verify that timestamps are strictly increasing within each split, that image and PV records refer to the same timestamp, and that the train/validation and test date sets do not overlap.

### 5.4 Check missing values and image range

```python
print("PV NaNs:", np.isnan(trainval_pv).sum(), np.isnan(test_pv).sum())
print("PV train min/max:", np.nanmin(trainval_pv), np.nanmax(trainval_pv))

sample = np.asarray(trainval_images[0])
print("Image sample:", sample.shape, sample.dtype, sample.min(), sample.max())
```

If PV contains missing values, decide before sequence construction whether to remove affected windows, interpolate only according to a documented rule, or use a missingness mask. Never silently convert missing targets to zero.

---

## 6. Correct preprocessing protocol

### 6.1 Split by time/date before fitting transforms

The official test set must remain untouched until the final evaluation. For development:

1. Keep the supplied test dates as the final test set.
2. Split train/validation by complete date.
3. Fit scalers using training dates only.
4. Apply those scalers to validation and test.

The current notebooks add time features before splitting. Cyclic calendar features are deterministic and safe, but the rolling statistic should be implemented carefully at date boundaries and should use only historical values. A robust implementation creates rolling features separately per date or explicitly resets the rolling state at sunrise/day boundaries.

### 6.2 Target scaling

PV regression often trains more reliably when the target is scaled. A safe option is:

```python
from sklearn.preprocessing import RobustScaler

target_scaler = RobustScaler()
target_scaler.fit(train_targets.reshape(-1, 1))
```

Fit the scaler only on training targets. Inverse-transform predictions before reporting kW metrics. If the supplied PV values are already transformed (for example, logarithmic), determine the exact transform from the dataset documentation before applying another transform. Never report a transformed-space MAE as kW.

### 6.3 Image normalization

For `uint8` RGB images, `x / 255.0` is appropriate. Check whether the stored arrays are channel-first or channel-last before using `Conv2D`. If necessary:

```python
if image.shape[-1] != 3 and image.shape[1] == 3:
    image = np.transpose(image, (0, 1, 3, 4, 2))
```

Do not assume this conversion without printing the actual shape.

---

## 7. Baselines that must be included

A model does not “beat previous papers” unless it beats simple baselines under the same evaluation protocol.

### 7.1 Persistence

For horizon \(h\), predict the most recent observed PV value:

\[
\hat y\_{t+h} = y_t.
\]

This is often surprisingly strong for short horizons. Report persistence RMSE, MAE, nRMSE, and skill score:

\[
\text{Skill} = 1 - \frac{\text{RMSE}_{model}}{\text{RMSE}_{persistence}}.
\]

### 7.2 Same-time-of-day baseline

Predict using the PV value from a previous comparable day at the same time, with all choices defined using training data only. This tests whether a neural model learns more than the diurnal cycle.

### 7.3 Seasonal/clear-sky baseline

If clear-sky irradiance or solar position is calculated, compare a clear-sky-scaled persistence model. Do not use future measurements when constructing it.

### 7.4 Existing neural baselines

At minimum compare:

1. Persistence.
2. PV-only LSTM/Conv1D-LSTM from [`forecast/01-lstm.ipynb`](./forecast/01-lstm.ipynb).
3. CNN + PV LSTM from [`forecast/02-cnn-lstm.ipynb`](./forecast/02-cnn-lstm.ipynb).
4. Proposed model.

Use the same date folds, target masks, test dates, and inverse-transform procedure for all models.

---

## 8. Recommended proposed model

The most defensible first proposal is a **motion-aware, multi-horizon, probabilistic multimodal model**, developed incrementally:

```text
Past sky images
    -> shared CNN or pretrained visual encoder
    -> temporal attention / ConvLSTM / temporal transformer
    -> cloud-motion representation

Past PV + time + solar-position features
    -> small temporal encoder

Image and PV representations
    -> gated fusion
    -> shared representation
    -> horizon-specific heads
    -> point output plus quantiles
```

### 8.1 Why each component is used

- **CNN or visual encoder:** extracts cloud edges, texture, brightness, and spatial context.
- **Temporal encoder:** captures cloud movement and changing illumination rather than treating frames independently.
- **PV branch:** supplies the plant's immediate operating state, which images alone do not fully determine.
- **Time/solar-position features:** represent deterministic day/night and seasonal structure.
- **Gated fusion:** lets the model rely more on images during cloud-driven ramps and more on PV history during stable periods.
- **Horizon-specific heads:** 15 minutes and 60 minutes have different error mechanisms.
- **Quantile heads:** produce prediction intervals, enabling reliability and calibration analysis.

Start with the existing CNN + LSTM, then add one component at a time. This prevents an untraceable “large model” result.

### 8.2 Practical model sequence

Train these in order:

1. Persistence baseline.
2. PV-only linear/MLP baseline.
3. PV-only LSTM from the current notebook.
4. Single-image CNN + PV history.
5. CNN feature sequence + LSTM.
6. CNN feature sequence + temporal attention.
7. Proposed gated-fusion model.
8. Proposed model with quantile loss.

Only the last model that gives a statistically credible improvement should be presented as the main contribution.

---

## 9. Sequence generation: safe reference logic

Use timestamps rather than array indices to define horizons. The following sketch assumes verified one-minute sampling; adapt it if the audit finds gaps.

```python
def make_sequences(pv, times, images=None, history=16, horizons=(15, 30, 60)):
    times = pd.to_datetime(times)
    dates = times.date
    x_pv, y, x_img, sample_dates = [], [], [], []

    for date in np.unique(dates):
        idx = np.flatnonzero(dates == date)
        if len(idx) <= history + max(horizons):
            continue

        for end in range(history, len(idx) - max(horizons)):
            hist_idx = idx[end - history:end]
            target_idx = [idx[end + h] for h in horizons]

            if not np.all(np.diff(hist_idx) == 1):
                continue
            if not np.all(np.diff(target_idx) >= 1):
                continue

            x_pv.append(pv[hist_idx])
            y.append([pv[i] for i in target_idx])
            sample_dates.append(date)
            if images is not None:
                x_img.append(images[hist_idx])

    result = {
        "pv": np.asarray(x_pv, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "dates": np.asarray(sample_dates),
    }
    if images is not None:
        result["images"] = np.asarray(x_img)
    return result
```

For production-scale images, do not materialize `np.asarray(x_img)` if it exhausts memory. Store integer indices and stream the image batches, as the current CNN–LSTM notebook does.

---

## 10. Training procedure, step by step

### Step 1: Freeze the environment

Record:

```text
Python version
TensorFlow version
CUDA/cuDNN version, if used
GPU model
NumPy/pandas/scikit-learn versions
Git commit or source archive hash
random seeds
```

Use the versions supported by the project, or update deliberately and record the change. The upstream README lists an older TensorFlow environment; do not assume old notebooks will run unchanged on current TensorFlow.

### Step 2: Run the data audit

Save:

- array shapes and dtypes;
- image range and channel order;
- PV range and missing count;
- time range and cadence;
- number of unique dates;
- train/test date overlap result;
- sequence counts at each horizon.

Stop if alignment or cadence is wrong.

### Step 3: Create a fixed split manifest

Save the exact date lists used for each fold in a CSV or JSON file:

```text
fold,role,date
0,validation,2017-...
0,train,2017-...
...
test,test,2019-...
```

Every model must read this same manifest. This is necessary for an apples-to-apples comparison.

### Step 4: Build sequences once

Create a preprocessing script or notebook that:

1. sorts by timestamp;
2. verifies cadence;
3. creates day-safe histories;
4. creates targets for 15, 30, and 60 minutes;
5. stores sample dates and target timestamps;
6. stores a mask for valid targets;
7. writes a versioned artifact.

Do not regenerate sequences differently for each model.

### Step 5: Train the baselines

For each fold:

1. Fit any scaler on the training portion only.
2. Construct train and validation datasets.
3. Train with early stopping on validation loss.
4. Save the best checkpoint and history.
5. Save validation predictions.

Use a deterministic seed, but repeat the full training with several seeds later. One favorable seed is not sufficient evidence.

### Step 6: Train the CNN–LSTM baseline

Use memory mapping and streaming batches. Keep image augmentation conservative. Horizontal flips, arbitrary rotations, or aggressive color changes may destroy camera geometry and solar cues. If augmentations are used, justify them and apply them only to training images.

### Step 7: Train the proposed model

Change one factor at a time:

- replace framewise CNN aggregation with temporal attention;
- add motion features or optical-flow-like displacement;
- add gated image/PV fusion;
- add quantile heads;
- add robust or ramp-aware loss.

Record all hyperparameters in a machine-readable file.

### Step 8: Evaluate the untouched test set once per finalized model

The test set is not for selecting hyperparameters. If you inspect test performance and modify the model repeatedly, it becomes a validation set. Keep a locked final test evaluation script.

### Step 9: Compute all metrics

For each horizon, report:

- MAE in kW;
- RMSE in kW;
- nMAE and nRMSE, with the normalization denominator defined;
- persistence skill score;
- bias (mean error);
- ramp-event MAE/RMSE;
- daytime-only metrics;
- cloudy and sunny subset metrics;
- per-day distribution of errors;
- confidence intervals over days.

Avoid MAPE near zero PV; it is unstable at night and dawn/dusk. If used, use a clearly stated daytime mask and epsilon, and also report MAE/RMSE.

### Step 10: Test significance

For each model comparison, compute paired per-day errors on the same samples. Use bootstrap resampling by day, not by overlapping sequence, to obtain a 95% confidence interval for the metric difference. A small average improvement whose interval crosses zero should not be claimed as a reliable win.

---

## 11. Probabilistic extension

For quantile levels \(q \in \{0.1, 0.5, 0.9\}\), use pinball loss:

\[
L_q(y,\hat y) =
\begin{cases}
q(y-\hat y), & y \ge \hat y,\\
(q-1)(y-\hat y), & y < \hat y.
\end{cases}
\]

Report:

- interval coverage probability;
- average interval width;
- weighted interval score;
- pinball loss;
- CRPS if a full predictive distribution is produced;
- reliability diagrams.

An interval that is always extremely wide is not useful. The goal is calibrated and sharp uncertainty, especially during cloud transitions.

---

## 12. Essential ablation study

The paper should contain a table like this:

| Variant               | PV history | Time features | Sky images | Temporal attention/motion | Gated fusion | Quantiles |
| --------------------- | ---------: | ------------: | ---------: | ------------------------: | -----------: | --------: |
| Persistence           |        yes |            no |         no |                        no |           no |        no |
| PV-only               |        yes |      optional |         no |                        no |           no |        no |
| Existing PV-LSTM      |        yes |           yes |         no |                      LSTM |           no |        no |
| Existing CNN–LSTM     |        yes |    no/limited |        yes |                      LSTM |           no |        no |
| Proposed minus motion |        yes |           yes |        yes |                        no |          yes |  optional |
| Proposed full         |        yes |           yes |        yes |                       yes |          yes |       yes |

Each row must use the same folds and training budget policy. The ablation must show whether the improvement comes from the proposed scientific idea, more parameters, or simply more training.

---

## 13. Error analysis that can become a contribution

Do not report only one overall number. Analyze:

1. **Horizon:** 15 vs 30 vs 60 minutes.
2. **Cloud condition:** clear, partly cloudy, overcast.
3. **Ramp events:** rapid PV increases/decreases.
4. **Solar elevation:** dawn, midday, dusk.
5. **Season/month:** seasonal distribution shift.
6. **PV level:** low, medium, high generation.
7. **Forecast lead:** error growth with lead time.
8. **Failure cases:** examples where the model misses a cloud edge or ramp.

Define all categories before looking at the final test result. If the labels are hand-selected after seeing predictions, the analysis is exploratory and must be labelled as such.

Useful visualizations:

- actual vs predicted PV for representative days;
- error versus solar elevation;
- error versus recent PV volatility;
- reliability diagrams for intervals;
- per-day boxplots;
- confusion-style tables for ramp detection;
- saliency or occlusion maps only as supporting evidence, not proof of causality.

---

## 14. Common problems in the current notebooks to resolve

### 14.1 Windows path portability

Replace hard-coded Linux paths with a configurable `Path` variable. Use raw Windows strings such as `r"D:\data\extracted"` and never rely on the current working directory.

### 14.2 Verify that “steps” equal minutes

The notebooks label 15, 30, and 60 array steps as minutes. Confirm the timestamp cadence first. Missing records can also make index-based horizons differ from time-based horizons.

### 14.3 Avoid incomplete-target ambiguity

The test builder permits partial end-of-day sequences and pads unavailable horizons with `NaN`. This can be valid, but report the valid sample count separately for every horizon. Do not report one “overall” number without explaining that horizons have different masks.

### 14.4 Rolling statistics at boundaries

The rolling PV standard deviation is computed over the flat split array. Reset it at date boundaries or use an explicitly causal, date-aware window. Document whether the first samples of each day use fewer than five observations.

### 14.5 Validation score versus test score

The notebook prints the minimum validation loss from each fold. That is useful for model selection diagnostics, but it is not a final generalization estimate. The paper's main result should be the locked test metric, with fold/seed variability and confidence intervals.

### 14.6 Fold averaging is not automatically a superior ensemble

The ten models have different validation dates but all see most training dates. Describe the average as a cross-fold ensemble and compare it with a single final model trained on all development dates. Do not claim that averaging is unbiased without measuring it.

### 14.7 Parameter count and compute budget

Report trainable parameter counts, training time, inference time, and peak memory. A small accuracy gain from a much larger model may not be a useful practical improvement.

---

## 15. Paper structure

### Title

Use a precise title, for example:

> Motion-aware multimodal probabilistic solar forecasting on the SKIPP'D benchmark

Change the title after the final contribution is known. Do not claim “state of the art” before a complete literature and benchmark comparison.

### Abstract

Include:

1. problem and horizons;
2. dataset and split;
3. proposed method;
4. main point and probabilistic metrics;
5. improvement over persistence and strongest baseline;
6. limitation.

### Introduction

Explain intermittency, cloud-driven ramps, the need for standardized data, and the gap your method addresses. State contributions as testable claims.

### Related work

Discuss:

- persistence and statistical models;
- CNN/RNN sky-image forecasting;
- multimodal PV/image fusion;
- transfer learning;
- probabilistic forecasting;
- SKIPP'D and its baseline;
- recent models using future-image synthesis.

### Data and preprocessing

Give the source, years, sampling interval, image shape, PV units/transforms, quality filtering, split dates, sequence construction, and missing-target policy.

### Method

Include equations, architecture diagram, number of parameters, loss function, scaler fitting, and inference procedure.

### Experimental protocol

State seeds, folds, hardware, software, batch size, optimizer, stopping rule, hyperparameter selection, and whether test data was ever used during model selection.

### Results

Give per-horizon tables, baseline skill, confidence intervals, ablations, runtime, and error-condition analysis.

### Limitations

Discuss single-site generalization, camera changes, missing images, unusual weather, distribution shift, uncertainty calibration, and compute cost. Transparent limitations increase credibility.

---

## 16. Minimum result table template

Fill this table only after running the locked evaluation script:

| Model             | Horizon | MAE (kW) | RMSE (kW) | nRMSE | Persistence skill | 95% CI |
| ----------------- | ------: | -------: | --------: | ----: | ----------------: | -----: |
| Persistence       |  15 min |      TBD |       TBD |   TBD |                 0 |    TBD |
| PV-only baseline  |  15 min |      TBD |       TBD |   TBD |               TBD |    TBD |
| Existing CNN–LSTM |  15 min |      TBD |       TBD |   TBD |               TBD |    TBD |
| Proposed model    |  15 min |      TBD |       TBD |   TBD |               TBD |    TBD |
| Persistence       |  30 min |      TBD |       TBD |   TBD |                 0 |    TBD |
| Proposed model    |  30 min |      TBD |       TBD |   TBD |               TBD |    TBD |
| Persistence       |  60 min |      TBD |       TBD |   TBD |                 0 |    TBD |
| Proposed model    |  60 min |      TBD |       TBD |   TBD |               TBD |    TBD |

Never invent values for this table. `TBD` should remain until the experiment has produced the result.

---

## 17. A realistic publication strategy

### Phase A: Reproduce

Run the current PV-only and CNN–LSTM notebooks after fixing paths and verifying data. Reproduce their metrics as closely as possible. If they cannot be reproduced, investigate the discrepancy before proposing a new model.

### Phase B: Establish the benchmark

Add persistence, same-time-of-day, and a simple linear/MLP model. Create the fixed split manifest and locked evaluation script.

### Phase C: Improve scientifically

Implement one motion-aware or uncertainty-aware idea. Keep the existing CNN–LSTM as a controlled baseline. Run ablations.

### Phase D: Stress-test

Repeat the best configurations across at least three to five random seeds, use day-level bootstrap confidence intervals, and analyze cloudy/ramp cases.

### Phase E: Write and release

Release preprocessing code, split manifest, configuration files, evaluation code, trained checkpoint hashes, and enough information to reconstruct the tables. Include the exact dataset citation and license.

---

## 18. Definition of success

The work is ready for submission when all of the following are true:

- the data audit passes;
- horizons are verified in real minutes;
- test dates are untouched during tuning;
- persistence and existing notebook baselines are reproduced;
- the proposed model improves the strongest baseline on the same test samples;
- the improvement is stable across seeds and day-level bootstrap intervals;
- ablations support the claimed mechanism;
- results are reported separately for all horizons and conditions;
- uncertainty metrics are calibrated if probabilistic forecasts are claimed;
- code and configurations are reproducible;
- limitations are stated honestly.

The next concrete action is therefore not to add more layers immediately. It is to run the audit, reproduce both current baselines, and create the locked comparison table. Only then should the proposed architecture be selected.

---

## References and links

1. Nie et al., “SKIPP'D: a SKy Images and Photovoltaic Power Generation Dataset for Short-term Solar Forecasting,” Solar Energy, 2023. <https://doi.org/10.1016/j.solener.2023.03.043>
2. Nie et al., “SKIPP'D: a SKy Images and Photovoltaic Power Generation Dataset for Short-term Solar Forecasting,” arXiv:2207.00913. <https://arxiv.org/abs/2207.00913>
3. Nie et al., “Sky-image-based solar forecasting using deep learning with multi-location data: training models locally, globally or via transfer learning?”, arXiv:2211.02108. <https://arxiv.org/abs/2211.02108>
4. Nie et al., “SkyGPT: Probabilistic Short-term Solar Forecasting Using Synthetic Sky Videos from Physics-constrained VideoGPT,” arXiv:2306.11682. <https://arxiv.org/abs/2306.11682>
5. Stanford SKIPP'D dataset repository. <https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset>
