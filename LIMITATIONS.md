# Limitations

- This is research software, not a clinically validated measurement or diagnosis system. The included pretrained YOLO11n models were smoke-tested but have not been evaluated against a MotionIQ dataset.
- Pose is COCO-17. It does not include toe/foot landmarks, so this implementation does not compute ankle angles. 2D knee/hip angles and trunk inclination depend on viewpoint, occlusion, pose confidence and perspective.
- Displacements are pixel or normalized-image values, not calibrated physical distances. Symmetry is a descriptive normalized trajectory comparison, not a normative score.
- Repetition counting is unconfigured because supported exercise-specific thresholds are not defined. This local research workspace has a trained Random Forest checkpoint and real video inference; a clean checkout lacks that ignored checkpoint and reports prediction as `not_configured` until trained or supplied.
- REHAB24-6 is the first supported dataset adapter. Its RGB archive is not bundled, and its license restricts use to academic/nonprofit noncommercial research. Do not download or train on it unless your use complies or permission is obtained.
- Evaluation requires genuine matching annotations. API routes compute pose MPJPE/PCK and segmentation IoU/Dice only when ground truth is submitted. This workspace has classification metrics from an independent participant-held-out REHAB24 test split; these metrics are not clinical validation and do not measure pose or segmentation quality.
- Segmentation is COCO person-class instance segmentation, not a medical or fine-boundary segmenter. Evaluation masks are compared at the model's lower output resolution after nearest-neighbor ground-truth resizing.
- Analysis samples up to 300 frames at 3 FPS by default. Longer videos may be truncated; the response and dashboard report this. Processing is synchronous and can take time on CPU.
- FastAPI keeps result records and segmentation evaluation masks in memory, has no authentication or upload-size limit, and is intended for trusted local use. Restarting the API clears result lookup and annotation evaluation state.
- Python support is pinned to 3.11/3.12. GPU model training is optional and requires labeled data; the local inference pipeline runs on CPU.
