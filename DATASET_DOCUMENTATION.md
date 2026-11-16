# Dataset documentation and provenance

No research dataset is bundled or downloaded into this repository. The local `data/raw/` and `data/processed/` paths are ignored by Git. Dataset records were inspected online on 2026-11-15; see the source pages linked below for authoritative files and terms.

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

## REHAB24-6 (RGB alternative)

- Source: https://zenodo.org/records/13305826
- Modality: 65 recordings, two RGB camera views, 30 FPS, plus 2D/3D positions for 26 joints and 41 markers; repetition boundaries and binary correctness labels.
- License/access: Zenodo record says academic or non-profit organization non-commercial research use only; cite the associated SISAP 2024 paper; contact VisionCraft for commercial use. Confirm the record terms before use.
- Size: the record lists a 2.7 GB `videos.zip`; total listed files are about 5.7 GB. The CSV annotation is about 53 KB, and names/description files are small.
- Status: record/page inspected; no archive downloaded. The video archive is large, and the restrictive use terms make it suitable only for qualifying non-commercial research.
- Limitations: 10 participants, controlled laboratory cameras, exercise set limited to six movements. Labels reflect a physiotherapist's instructed correctness categories; they are not clinical outcome labels.

## Local dataset conventions

Use `data/raw/<dataset>/` for untouched source files and `data/processed/<dataset>/` for generated artifacts. Preserve source filenames and checksums. Store a manifest with source URL, record version, license, download date, modality, participant ID, and file checksum. Do not commit recordings. Use participant-level splits, never frame-level random splits. `motioniq.video.build_manifest` inspects local videos; `split_by_participant` separates records without participant overlap.

## Synthetic fallback

`create_synthetic_video` creates a simple rendered stick-figure MP4 for codec, upload, and frame-extraction software checks only. It is artificial, unannotated test material and must not be included in research results or dataset manifests presented as real samples.
