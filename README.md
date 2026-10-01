# 3D Hippocampus Segmentation from MRI

A compact, end-to-end 3D segmentation project using the Medical Segmentation Decathlon `Task04_Hippocampus` dataset. One notebook covers data exploration, preprocessing, a small 3D U-Net, validation, and final held-out evaluation.

The model assigns each MRI voxel to background (0), anterior hippocampus (1), or posterior hippocampus (2).

## Results

The final checkpoint was selected at epoch 25 using the mean foreground Dice on the validation split. It was then evaluated on the held-out test split.

| Split | Scans | Anterior Dice | Posterior Dice | Mean foreground Dice |
| --- | ---: | ---: | ---: | ---: |
| Validation | 39 | 0.8569 | 0.8549 | 0.8559 |
| Test | 39 | 0.8715 | 0.8610 | **0.8663** |

Dice is calculated per scan and foreground class, then averaged across scans. Background Dice is excluded from the headline metric. The test results were not used for training or checkpoint selection.

## Dataset

The dataset is listed on the [Medical Segmentation Decathlon AWS downloads page](https://medicaldecathlon.com/dataaws/). Download the [official public `Task04_Hippocampus.tar` archive](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar) to `data/Task04_Hippocampus.tar` and extract it so that the project contains:

```text
data/Task04_Hippocampus/dataset.json
data/Task04_Hippocampus/imagesTr/
data/Task04_Hippocampus/labelsTr/
```

MSD lists the dataset license as **CC BY-SA 4.0**. The downloaded archive used for this project has SHA-256:

```text
282d808a3e84e5a52f090d9dd4c0b0057b94a6bd51ad41569aef5ff303287771
```

You can check your download with `sha256sum data/Task04_Hippocampus.tar`. The `data/` directory is ignored by Git.

The project uses all 260 publicly labeled volumes: 182 training, 39 validation, and 39 test scans. A NumPy seed of 42 determines the split, which the notebook writes to or verifies against `splits.json`. Original NIfTI files are never modified.

## Method

The notebook `hippocampus_segmentation.ipynb` walks through these steps:

1. Check image–mask pairing, spatial grids, voxel spacing, shapes, labels, and intensity distributions. Investigate the unusually scaled `hippocampus_210` scan and its bright boundary voxels.
2. Normalize each MRI independently: clip nonzero intensities at its 1st and 99th percentiles, then scale to `[0, 1]`. The mask does not influence normalization.
3. Symmetrically pad images and masks to `(48, 64, 48)`. A Boolean `valid_region` excludes artificial padding while keeping genuine background voxels valid.
4. Train a small 3D U-Net (85,035 parameters) with one input channel and three output-logit channels. Use masked cross-entropy, Adam at `1e-3`, batch size 2, and CPU training.
5. Calculate anterior and posterior Dice per validation scan after each epoch. Train a fresh model for 15 epochs, then continue the same live model and optimizer through epoch 25. Save the best validation checkpoints separately.
6. Inspect validation score distributions and prediction overlays. Load the selected epoch-25 checkpoint and evaluate the 39 held-out test scans.

The five-epoch diagnostic reached 0.4229 mean foreground validation Dice, with posterior Dice of only 0.1810. The same training setup reached 0.8381 mean validation Dice at epoch 15 and 0.8559 at epoch 25. The early result was therefore insufficient to judge this model.

## Reproduce

Install the locked dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

From the project directory, open `hippocampus_segmentation.ipynb` in VS Code or Jupyter and select `.venv/bin/python` as its kernel. Running every cell from a fresh kernel regenerates or checks the split, trains for 25 epochs, and evaluates the selected checkpoint. The saved notebook already includes the outputs and figures; a fresh run takes several minutes and writes checkpoints.

The dependency lock uses a CUDA-enabled PyTorch build. The complete training run explicitly uses CPU because, on the development machine's GTX 1650 Ti, this small model's GPU epoch was slower than the measured CPU epoch.

## Files

- `hippocampus_segmentation.ipynb`: the complete experiment and its saved outputs.
- `pyproject.toml`, `uv.lock`: project dependencies.
- `splits.json`: reproducible scan assignments.
- `best_small_unet3d_15epochs.pt`: best checkpoint during the first 15 epochs.
- `best_small_unet3d_25epochs.pt`: final model selected on validation.

## Limitations

This is a single-split baseline on one dataset, without external-site evaluation or a systematic architecture comparison. The available dataset metadata does not verify patient identities, so patient-level separation across splits cannot be confirmed. The reported test Dice should be understood within those limits.
