# Limitations

- This is research infrastructure, not a clinically validated measurement or diagnosis system.
- No real dataset or trained checkpoint is included. Current environment could not resolve the user-specified UI-PRMD Zenodo URL; available datasets are large, and REHAB24-6 has restrictive academic/non-commercial terms.
- MediaPipe pose and Ultralytics segmentation are optional. Inference quality is unmeasured here. YOLO requires a pre-provisioned `models/yolo11n-seg.pt` file. In this headless macOS environment, MediaPipe 0.10.18 could not initialize its OpenGL context (`NSOpenGLPixelFormat`); the API reports pose as unavailable instead of failing the upload.
- The dashboard can display a pose/segmentation overlay image for the first sampled frame when optional model outputs are available; it does not generate an overlaid full-length video. CSV contains normalized model landmarks, not clinical angles. Segmentation masks are represented only through the preview image.
- Joint angles and trunk orientation are 2D geometry unless a calibrated 3D source is supplied; perspective, occlusion, camera placement, clothing, and image quality can change results.
- Repetition detection is a simple peak detector and needs movement-specific tuning. Symmetry is a bounded descriptive score, not a normative measure.
- FastAPI stores results in memory, has no authentication or upload-size cap, and is intended for trusted local use only.
- Frontend development currently expects API CORS to be allowed by a same-origin proxy or browser configuration; production deployment should configure explicit CORS and secure storage.
- Python support is pinned to 3.11/3.12 due package compatibility; Python 3.14 is outside the declared support range.
