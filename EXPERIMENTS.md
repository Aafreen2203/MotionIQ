# Experiments

## Current status

No research model has been trained and no benchmark score is reported. Model classes and metric functions are implemented; evaluation requires a dataset, labels, and participant-level split. The optional application adapters do not imply accuracy or clinical validity.

## Baseline protocol

1. Select a dataset whose license permits the intended academic/non-commercial use and document its exact version/checksum.
2. Keep all records from one participant in one split; use `split_by_participant` or an equivalent fixed split file.
3. Extract frame-wise landmarks; document confidence thresholds, missing-point treatment, camera view, and normalization.
4. Build per-repetition feature vectors (joint angles, ranges, symmetry, tempo) for ExtraTrees; use sequence features for the GRU.
5. Select hyperparameters on validation participants only; report held-out test metrics with class counts and confusion matrix.
6. Report MPJPE/PCK only against matching pose ground truth with explicit coordinate units, IoU/Dice only against masks, and classification metrics only against labels.
7. Measure latency after warmup and report hardware, model weights, frame sampling, and preprocessing. Report tracking stability and measurement consistency separately from task accuracy.

## Google Colab

For GPU experiments, use a Python 3.11 runtime. Clone/copy the project, install `python -m pip install -e '.[training,dev]'`, and install `.[vision]` only if extracting poses in Colab. Upload or mount data only if the dataset terms allow this. Set a fixed seed, create participant splits, train the GRU with `train_temporal_model(..., device='cuda')`, and save checkpoints to Drive. Do not use the repository's synthetic clip for reported scores. Record the runtime image, CUDA/PyTorch versions, code commit, split file, and dataset checksums.
