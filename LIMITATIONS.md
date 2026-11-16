# Limitations

- This is research software, not a clinically validated measurement or diagnosis system. The included pretrained YOLO11n models were smoke-tested but have not been evaluated against a MotionIQ dataset.
- Pose is COCO-17. It does not include toe/foot landmarks, so this implementation does not compute ankle angles. 2D knee/hip angles and trunk inclination depend on viewpoint, occlusion, pose confidence and perspective.
- Displacements are pixel or normalized-image values, not calibrated physical distances. Symmetry is a descriptive normalized trajectory comparison, not a normative score.
- Repetition counting is unconfigured because supported exercises and movement-specific thresholds are not defined. No movement classification checkpoint or class mapping is present, so prediction is explicitly `not_configured`.
- Evaluation requires matching annotations. API routes compute pose MPJPE/PCK and segmentation IoU/Dice when corresponding ground truth is submitted. Classification precision, recall, F1 and confusion matrix helpers exist but there are no model predictions to evaluate.
- Segmentation is COCO person-class instance segmentation, not a medical or fine-boundary segmenter. Evaluation masks are compared at the model's lower output resolution after nearest-neighbor ground-truth resizing.
- Analysis samples up to 300 frames at 3 FPS by default. Longer videos may be truncated; the response and dashboard report this. Processing is synchronous and can take time on CPU.
- FastAPI keeps result records and segmentation evaluation masks in memory, has no authentication or upload-size limit, and is intended for trusted local use. Restarting the API clears result lookup and annotation evaluation state.
- Python support is pinned to 3.11/3.12. GPU model training is optional and requires labeled data; the local inference pipeline runs on CPU.
