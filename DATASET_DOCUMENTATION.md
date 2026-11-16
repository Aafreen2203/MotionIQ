# Dataset documentation and provenance

No research dataset is bundled or downloaded into this repository. The local `data/raw/` and `data/processed/` paths are ignored by Git. Dataset records were inspected online on 2026-11-16; see the source pages linked below for authoritative files and terms.

## UI-PRMD

- User-provided original URL: https://zenodo.org/records/1118037 — Zenodo redirects this URL to record 1118038, which is an unrelated figure for a stonefly taxonomy paper. No UI-PRMD data is present at the requested URL.
- Current Zenodo record found during discovery: https://zenodo.org/records/23104965
- Original project documentation: https://www.idahofallshighered.org/vakanski/ui-prmd.html
- Modality: joint positions/angles from Vicon optical motion capture and Kinect skeleton tracking. It is **not ordinary RGB exercise video**.
- Directory/file layout described in its README: `Vicon/` and `Kinect/`, each with `Positions/` and `Angles/`; recordings are comma-delimited text named by movement and subject, with episode identifiers in segmented files. Incorrect examples add an `_inc` suffix. A documented missing sample is Vicon inline lunge `m03_s03` positions/angles.
- Scope: 10 healthy subjects, 10 exercises, repeated recordings; reduced and segmented variants are available.
- License: project page states Open Data Commons Public Domain Dedication and License (PDDL) v1.0. Check the current record and source terms before reuse.
- Current Zenodo listing: seven files totaling about 1.26 GB (archives from 0.2–0.29 GB; README 8 KB). Zenodo's metadata API reports no license field on this current record; the original project page states PDDL 1.0. The small README was inspected remotely; no archives were downloaded.
- Suitability: useful for skeleton/kinematic feature and exercise-quality research, not for training RGB segmentation or ordinary-video pose extraction.

## Selected for MotionIQ's first video-feature experiment: REHAB24-6

- Source/version: [Zenodo record 13305826](https://zenodo.org/records/13305826), version 1; [dataset paper](https://doi.org/10.1007/978-3-031-75823-2_2).
- Verified content: 65 recordings, 184,825 RGB frames at 30 FPS from Camera17 and Camera18, 10 participants, six named exercises, and 1,072 repetition annotations. The official `Segmentation.csv` was fetched and its MD5 (`90b8fbd7445dd050bf27b17126c78fbe`) matches the record. It uses semicolon-separated columns `video_id`, `repetition_number`, `exercise_id`, `person_id`, `first_frame`, `last_frame`, view/error/lighting fields, and binary `correctness`. The matching `Segmentation.txt` documents frame indexing and columns.
- Exercise IDs: 1 arm abduction, 2 arm VW, 3 push-ups on a table, 4 leg abduction, 5 leg lunge, 6 squat. The inspected annotation rows have 178, 208, 107, 210, 174, and 195 repetitions for these classes respectively. The current model target is **exercise identity**; correctness remains a separate ground-truth field and is not silently treated as an exercise class.
- License: the record states academic or nonprofit-organization noncommercial research use only, requires citation, and directs commercial users to contact VisionCraft. Do not use outside those terms without written permission. The video archive and derived artifacts are local ignored files, not tracked by this repo. Zenodo lists `videos.zip` at about 2.7 GB (5.7 GB total across the record).
- Limitations: only 10 people and a controlled lab setup. In the official repetition annotations, additional people are marked in 386 Camera17 repetitions versus 42 Camera18 repetitions, so the adapter defaults to Camera18 (still preserving the camera choice in provenance). Actual exercise-class coverage after YOLO failures must be inspected after preprocessing. The source also has 2D/3D motion-capture ground truth, but MotionIQ's classifier adapter deliberately extracts features from the actual RGB video with its own YOLO11 pipeline. These labels do not establish clinical validity or generalization to user phone videos.

## Other candidates reviewed

- [MM-Fit](https://mmfit.github.io/) includes actual RGB-D video, synchronized wearable sensors and ten exercises; its video-only Zenodo release contains 20 sessions (about 39 GB). The inspected record did not clearly expose a reuse license, so it is not selected until rights are clarified.
- [FineGym](https://sdolivia.github.io/FineGym/) has actual gymnastics video identifiers and hierarchical, timestamped event/subaction labels. Its annotation license is CC BY-NC 4.0, while video files are hosted on YouTube and remain subject to source-video rights and availability. It is less aligned with rehabilitation exercise classes and is not selected.
- [UI-PRMD](https://www.idahofallshighered.org/vakanski/ui-prmd.html) contains Kinect/Vicon skeleton positions and angles, not ordinary RGB exercise videos; it cannot validate MotionIQ's RGB-to-pose feature pipeline.
- [Penn Action](https://dreamdragon.github.io/PennAction/) has 2,326 RGB sequences, 15 action labels and per-frame joints, but the public documentation reviewed here does not establish a participant ID suitable for the required subject-held-out split or a clear reuse license. It is not selected.
- [UCO Physical Rehabilitation](https://github.com/AVAuco/ucophyrehab) contains 2,160 RGB sequences, 16 exercise activities, 27 subjects, and five camera views. Its official repository says data access is by request with name, affiliation and research purpose, so it cannot be acquired automatically here. It is a strong future option after access and terms are confirmed.

## Implemented experiment path

`python -m motioniq.training prepare-rehab24` reads the official annotation CSV and locally extracted Camera17/Camera18 videos, runs MotionIQ pose/measurement extraction once per source video, and emits repetition-level temporal features plus timestamped pose/measurement JSONL and checksummed provenance. It preserves `person_id` as `subject_id`, label IDs/names, repetition/frame intervals, orientation and source file metadata. It drops repetitions with fewer than two sampled frames or no detected measurement features and reports exclusions. It does not download data or bypass the license.

The training command uses a deterministic grouped subject split (default 64/16/20 percent by recordings), validation-only selection among a small Random Forest parameter grid, then refits train+validation and evaluates the test subjects once. Median imputation and missingness indicators are fitted inside the training pipeline; class-weighted Random Forest handles imbalance. Trees do not need feature scaling. XGBoost is not added as a dependency for this initial, small-subject baseline; compare it later only under the same group split. An optional bidirectional LSTM helper remains an experimental later temporal model, not a shipped prediction model or substitute for the RF baseline. The 10-subject dataset may be too small for reliable generalization even when every split runs.

### Run completed in this workspace

The authorized local run processed Camera18 at 2 sampled FPS, up to 300 samples per recording, using the actual YOLO11n-pose model and `motioniq-temporal-v1` features. It retained 1,045 of 1,072 annotated repetitions (27 excluded for insufficient sampled/valid measurements), spanning all six exercise classes and all 10 participants. Counts after extraction: arm abduction 172, arm VW 193, leg abduction 210, leg lunge 173, push-up 107, squat 190. Source checksums, runtime versions, YOLO weight hashes and row-level provenance are in the ignored `data/processed/rehab24/provenance.json`.

Seed 42 grouped the participants into train subjects 1, 2, 3, 5, 6, 7 (686 repetitions), validation subjects 4, 9 (239), and held-out test subjects 8, 10 (120). Validation-only selection chose a 500-tree Random Forest, `min_samples_leaf=3`, `max_features=0.7`. On the held-out subjects: accuracy 0.925; macro precision 0.942, recall 0.900, F1 0.894; weighted precision 0.951, recall 0.925, F1 0.917. Class supports are 17, 15, 23, 25, 18, 22 in label order `arm_abduction`, `arm_vw`, `leg_abduction`, `leg_lunge`, `push_up`, `squat`. `arm_vw` recall was 0.40 (6/15); its 9 misses were predicted as `arm_abduction`. The small two-participant test set and controlled camera setup make these preliminary dataset-specific results, not a guarantee of new-user performance. The checkpoint and complete metrics are local in `models/motioniq-rf.joblib` and `models/motioniq-rf.status.json`; both are ignored by version control.

No manually annotated pose or segmentation ground-truth files were supplied, so MPJPE/PCK and IoU/Dice remain not evaluated. A clean clone also lacks the local checkpoint and reports prediction as `not_configured` until the experiment is reproduced or a compatible checkpoint is supplied.

## Local dataset conventions

Use `data/raw/<dataset>/` for untouched source files and `data/processed/<dataset>/` for generated artifacts. Preserve source filenames and checksums. Store a manifest with source URL, record version, license, download date, modality, participant ID, and file checksum. Do not commit recordings. Use participant-level splits, never frame-level random splits. `motioniq.video.build_manifest` inspects local videos; `split_by_participant` separates records without participant overlap.

## Synthetic fallback

`create_synthetic_video` creates a simple rendered stick-figure MP4 for codec, upload, and frame-extraction software checks only. It is artificial, unannotated test material and must not be included in research results or dataset manifests presented as real samples.
