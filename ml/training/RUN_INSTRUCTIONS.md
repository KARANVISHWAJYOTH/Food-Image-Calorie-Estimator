# Training on Google Colab

This guide runs `ml/training/train.py` from the repository root. The model needs
Food-101 images and a `combined_labels.csv` of class-level nutrition targets.
Those targets are text-matched Nutrition5k proxies, not verified Food-101
ground truth.

## 1. Start a Colab runtime and get the code

In Colab, select **Runtime > Change runtime type > T4 GPU** (or another GPU),
then run:

```python
!git clone https://github.com/KARANVISHWAJYOTH/Food-Image-Calorie-Estimator.git
%cd /content/Food-Image-Calorie-Estimator
!python -m pip install -r ml/requirements.txt
```

Confirm Colab can see the accelerator:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

If this prints `False`, select a GPU runtime before training. The training
script automatically uses CUDA when available; `--device cuda` can also be
specified explicitly.

## 2. Make the dataset accessible

The defaults in `ml/configs/base.yaml` expect this layout relative to `ml/`:

```text
ml/data/
├── food101/
│   └── food-101/
│       ├── images/
│       └── meta/
│           ├── classes.txt
│           ├── train.txt
│           └── test.txt
└── combined_labels.csv
```

Food-101 can be downloaded into the configured default location with:

```python
!python ml/data/download_food101.py
```

This downloads and extracts the dataset and requires several gigabytes of free
Colab storage. Alternatively, mount Drive if the dataset is already there:

```python
from google.colab import drive
drive.mount("/content/drive")
```

For a Drive copy, the folder configured as `dataset.food101_dir` must contain
the `food-101/` directory shown above. Point `dataset.food101_dir` and
`dataset.combined_labels_csv` at the mounted paths by editing
`ml/configs/base.yaml`, for example:

```python
from pathlib import Path
import yaml

config_path = Path("ml/configs/base.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
config["dataset"]["food101_dir"] = "/content/drive/MyDrive/FoodData/food101"
config["dataset"]["combined_labels_csv"] = (
    "/content/drive/MyDrive/FoodData/combined_labels.csv"
)
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
```

The example assumes
`/content/drive/MyDrive/FoodData/food101/food-101/` contains the extracted
Food-101 tree. Keep model and training settings in the same YAML file; only the
paths need to be changed for this example.

### Creating `combined_labels.csv` (if needed)

If you have Nutrition5k metadata but not the combined labels, make these files
accessible:

```text
<nutrition5k directory>/
└── metadata/
    ├── dish_metadata_cafe1.csv
    └── dish_metadata_cafe2.csv
```

Mount Drive as above, then run (substitute your actual Drive folder):

```python
!python ml/data/prepare_nutrition5k.py \
    --data-dir /content/drive/MyDrive/FoodData/nutrition5k_raw
!python ml/data/align_labels.py
```

`prepare_nutrition5k.py` writes the clean nutrition CSV to the configured
`dataset.nutrition5k_clean_csv`. `align_labels.py` combines it with the
Food-101 class list and writes `dataset.combined_labels_csv`. Ensure the
configured paths point to the Drive data locations before running these
commands. Only the two Nutrition5k metadata CSVs are needed by this preparation
step; the full Nutrition5k image/video download is not required.

The data loader skips Food-101 classes without resolved nutrition values. Check
the label-alignment output and `ml/data/manual_overrides.yaml` if the loader
reports too few samples or no usable samples.

## 3. Run a small smoke test

With the repository root as the current directory, run two epochs on up to
1,024 train and 256 validation samples. This command overrides Stage 1 to one
epoch, so it exercises both the frozen-head and fine-tuning stages:

```python
!python -m ml.training.train \
    --epochs 2 \
    --batch-size 32 \
    --max-train-samples 1024 \
    --max-val-samples 256 \
    --freeze-backbone-epochs 1 \
    --unfreeze-layers 1 \
    --device cuda
```

This runs one epoch in each stage to exercise the transition. Sample caps are
selected deterministically from the manifest and do not change the default
full-dataset behavior. The split manifest is written to `ml/data/splits.json`.

On a Colab T4, budget roughly **a few minutes to around 15 minutes** for the
first smoke run, including model initialization/weight download and data
access; Drive I/O, accelerator availability, and startup time can change this.
This is an estimate, not a benchmark. A CPU runtime will take substantially
longer.

## 4. Run full training

The default command uses the configured two-stage schedule: two epochs with a
frozen backbone and head learning rate `0.001`, followed by eight epochs with
the deepest backbone stage unfrozen at learning rate `0.0001`. Batch size
defaults to 32:

```python
!python -m ml.training.train --device cuda
```

No stage flags are needed for the normal run: the schedule is read from
`ml/configs/base.yaml`:

```yaml
training:
  stage1_epochs: 2
  stage1_learning_rate: 0.001
  stage2_epochs: 8
  stage2_learning_rate: 0.0001
  stage2_unfreeze_layers: 1
  use_amp: true
```

Stage 1 updates only the classification and regression heads. At Stage 2, the
configured number of deepest backbone stages is unfrozen and a new AdamW
optimizer is created at the lower learning rate. Set `stage2_unfreeze_layers`
to the number of parameterized stages in the selected backbone to fully
unfreeze it. ResNet-50 counts the stem convolution and batch normalization
separately, followed by its four residual stages; EfficientNet-B0 counts its
parameterized feature blocks. A value of 1 fine-tunes only the deepest stage.
The shared `--scheduler` option creates a fresh scheduler for each training
stage.

Automatic mixed precision is controlled by `training.use_amp`. It defaults to
`true` for CUDA training and uses autocast with gradient scaling; set it to
`false` in `ml/configs/base.yaml` to force full precision. If enabled while
running on CPU, training warns and safely continues in full precision.

The flow is:

1. **Stage 1:** freeze the backbone and train only the two task heads for
   `stage1_epochs` at `stage1_learning_rate`.
2. **Stage 2:** unfreeze the `stage2_unfreeze_layers` deepest backbone stages
   and continue for `stage2_epochs` at `stage2_learning_rate`, using a fresh
   AdamW optimizer.

To change the normal schedule, edit those five `training` values in the YAML.
The following command uses the config schedule while overriding the batch size
and enabling a cosine scheduler for each stage:

```python
!python -m ml.training.train \
    --batch-size 32 \
    --scheduler cosine \
    --device cuda
```

For a short two-stage run without editing the YAML, `--epochs` sets the total
epoch count and `--freeze-backbone-epochs` selects how many of those epochs run
in Stage 1. `--unfreeze-layers` can override the configured Stage 2 unfreeze
depth. For example, the smoke-test command above runs one epoch per stage.
`--learning-rate` sets both stage learning rates to one value; omit it to keep
the configured higher Stage 1 and lower Stage 2 rates.

The base config defaults to two frozen-backbone epochs followed by eight
fine-tuning epochs. Setting `--freeze-backbone-epochs 0` skips the frozen-head
stage and runs all epochs in Stage 2. ResNet-50 exposes its stem and four
residual stages; EfficientNet-B0 uses its parameterized feature stages.
Batch-normalization state in fully frozen stages remains in evaluation mode
until those stages are unfrozen.

For the complete configured split on a Colab T4, a rough planning estimate is
**about 1–4 hours for 10 epochs**. This can vary considerably with the number of
classes that have nutrition labels, image storage location, Colab GPU type,
input pipeline performance, and session load. CPU-only training can take many
hours or longer. Treat these figures as approximate budgeting guidance rather
than measured guarantees.

## 5. Outputs and resuming

The run writes:

- Timestamped per-epoch metrics CSV files under `ml/training/logs/`.
- One checkpoint per epoch and `best.pt` under `ml/training/checkpoints/`.
- The deterministic split manifest at `ml/data/splits.json`.

To preserve outputs after the Colab session ends, copy these folders to Drive
before disconnecting, for example:

```python
!cp -r ml/training/logs /content/drive/MyDrive/FoodData/training-logs
!cp -r ml/training/checkpoints /content/drive/MyDrive/FoodData/checkpoints
!cp ml/data/splits.json /content/drive/MyDrive/FoodData/splits.json
```

The current training script saves checkpoints but does not resume optimizer or
epoch state from them.

## 6. Compare augmentation runs

Three training augmentation presets are available: `light`, `standard`, and
`strong`. The selected preset is read from `preprocessing.augmentation_preset`
in `ml/configs/base.yaml` and is included in checkpoint training metadata.
Validation always uses the same deterministic resize, center-crop, and
normalization transform.

The default remains `standard` until validation results are compared. To run
each configuration, edit the preset and use distinct log paths so the CSVs are
preserved:

```python
import yaml
from pathlib import Path

config_path = Path("ml/configs/base.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
config["preprocessing"]["augmentation_preset"] = "light"  # or "standard"/"strong"
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
```

```python
!python -m ml.training.train --log-file ml/training/logs/light.csv --device cuda
```

Repeat after setting `augmentation_preset` to `standard` and `strong`, using
`standard.csv` and `strong.csv` as log paths. Keep the dataset split, sample
limits, model, and training schedule the same between runs for a fair
comparison.

Plot validation loss and accuracy for all three runs:

```python
!python -m ml.training.compare_runs \
    ml/training/logs/light.csv \
    ml/training/logs/standard.csv \
    ml/training/logs/strong.csv \
    --labels light standard strong \
    --output ml/training/logs/augmentation_comparison.png
```

The plot has validation-loss and validation-accuracy panels and is saved at the
specified output path. Choose a preset based on validation behavior (ideally
across more than one seed): stronger augmentation may reduce overfitting but
can slow convergence or underfit; lighter augmentation can converge faster but
may generalize less well. No run metrics are currently included here, so the
default has not been changed. Share the three CSVs' epoch, validation-loss, and
validation-accuracy columns to select a winner based on observed results.
