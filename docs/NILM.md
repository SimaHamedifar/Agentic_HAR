# NILM component: Sima's latest seven-appliance requirement

The current NILM code uses one shared CNN and one checkpoint to classify seven appliances simultaneously:

1. Television Site
2. Toaster
3. Microwave
4. Kettle
5. Computer Site
6. Washing Machine
7. Dishwasher

This replaces the earlier implementation that trained a separate model and checkpoint for each appliance.

### Input and labels

Every time step has six model features in this exact order:

1. `Aggregate` household power
2. `Hour_X`
3. `Hour_Y`
4. `DoW_X`
5. `DoW_Y`
6. `is_weekend`

The cache builder resamples readings into fixed eight-second bins. A 37-point window therefore covers about 296 seconds (approximately five minutes). Windows do not cross houses, non-increasing timestamps, or gaps longer than the configured tolerance.

The centre point of each window supplies seven labels. Code regenerates those labels from `REFIT_multi_appliance.csv` with the thresholds in `nilm/example_config.json`. If an appliance value is NaN, its mask is 0 and that label contributes nothing to loss or metrics. `REFIT_multi_appliance_binary.csv` is a reference file; it is not needed during training.

### Install

Python 3.10 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
```

This workstation has Windows, Python 3.10, and an NVIDIA GeForce RTX 5070 Ti. Its reproducible GPU install is:

```powershell
python -m pip install -r requirements-gpu-win-py310.txt
```

That file installs `torch 2.14.0+cu130` from the official PyTorch CUDA 13.0 wheel repository. It is specific to 64-bit Windows and Python 3.10. For another Python or operating-system version, select the matching CUDA wheel from the official PyTorch installer.

Verify CUDA before training:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected on this machine: `2.14.0+cu130`, `True`, and `NVIDIA GeForce RTX 5070 Ti`.

### Step 1: build the disk cache

Run this once from the repository root:

```powershell
python -m nilm.cache --config nilm/example_config.json
```

The raw merged CSV is about 6.4 GB and contains about 50.8 million rows. Cache creation scans it twice in chunks and can take substantial time. It writes per-house `.npy` memory maps under `artifacts/nilm/refit_cache_8s`; it does not load the full CSV into RAM. Use `--overwrite` only when intentionally rebuilding an existing generated cache.

For a small pipeline check, copy the config and set `max_rows_per_house` to a bounded value. A short prefix may not contain positive examples of every rare appliance, so it is suitable for cache testing rather than meaningful training.

### Step 2: train, validate, and test

```powershell
python -m nilm.train --config nilm/example_config.json --device cuda
```

Training uses house-level splits to prevent readings from one house appearing in more than one split. Aggregate normalization and seven positive-class weights are fitted only from training houses. Loss and metrics ignore unobserved labels through the mask.

The checked-in batch size is 4,096. On this RTX 5070 Ti it used under 1 GB of PyTorch-allocated GPU memory in a real-cache throughput check and processed about 21,000 windows per second. Reduce `batch_size` if other GPU applications leave too little free memory.

The checked-in split is a coverage-valid provisional split:

- train: H1, H6, H7, H15
- validation: H2, H5
- test: H3, H11

It gives all seven appliances observed data in every split, but Microwave training data comes only from H6. Confirm the final experimental split with Sima before reporting research results.

Outputs are written to `artifacts/nilm/multi_appliance`:

- `best_model.pt`: one seven-output model plus feature/appliance order, thresholds, normalization, house splits, class weights, history, and metrics;
- `metrics.json`: test loss and per-appliance precision, recall, F1, balanced accuracy, binned approximate average precision, accuracy, and confusion counts. Average precision uses 1,000 probability bins so evaluation memory stays bounded on the full dataset.
- `figures/training_history.png`: train/validation weighted BCE loss and macro F1 across epochs;
- `figures/test_metrics.png`: per-appliance test F1 and average precision.

The training command generates both figures automatically after test evaluation and records their paths in the checkpoint metadata and `metrics.json`.

### Step 3: inference

Create a CSV with exactly 37 rows and these six columns:

```text
Aggregate,Hour_X,Hour_Y,DoW_X,DoW_Y,is_weekend
```

Then run:

```powershell
python -m nilm.inference `
  --checkpoint artifacts/nilm/multi_appliance/best_model.pt `
  --window-file path/to/window.csv
```

The output is one activation probability per appliance:

```json
{
  "Television Site": 0.12,
  "Toaster": 0.03,
  "Microwave": 0.41,
  "Kettle": 0.87,
  "Computer Site": 0.22,
  "Washing Machine": 0.08,
  "Dishwasher": 0.15
}
```

These values mean the model's estimated probability that each appliance is ON at the centre of the input window. They are model outputs, not measured wattages. The JSON values above are illustrative; use `metrics.json` for the saved real-run evaluation results.

### Important status and open decisions

- Full REFIT cache construction and one real GPU training run have completed. The run stopped after 11 epochs by early stopping; epoch 6 had the best validation loss (`5.772730`) and the final test loss was `1.949556`.
- A trained seven-output checkpoint is available at `artifacts/nilm/multi_appliance/best_model.pt`, with the matching evaluation summary at `artifacts/nilm/multi_appliance/metrics.json`.
- Current test F1 scores are: Television Site `0.637`, Washing Machine `0.329`, Dishwasher `0.218`, Kettle `0.073`, Computer Site `0.040`, Microwave `0.019`, and Toaster `0.011`. This is a completed baseline run, not a production-quality model. The rare appliances have high recall but very low precision and therefore many false positives.
- The six NILM test modules currently contain 27 passing tests.
- Sima should confirm the final house split or cross-validation protocol.
- H6 contains `MJY Computer` and `PGM Computer` in upstream data rather than a literal `Computer Site` channel. The provided merged CSV currently omits H6 from `Computer Site`; confirm whether those two channels should be combined in a future dataset revision.
- The current evaluation converts probabilities to ON/OFF at `0.5`. Any future per-appliance probability-threshold calibration is separate from Sima's fixed watt thresholds used to create labels.
- The repository currently implements the CNN baseline only. No Transformer comparison has been implemented or trained.
- The NILM probabilities are intended as evidence for activity reasoning. Connection to HAR/LLM-agent APIs remains a later integration task.

### Developer checks

```powershell
python -m pytest tests/test_nilm_config.py tests/test_nilm_preprocessing.py tests/test_nilm_cache.py tests/test_nilm_model.py tests/test_nilm_training.py tests/test_nilm_inference.py -q
python -m nilm.cache --help
python -m nilm.train --help
python -m nilm.inference --help
```
