import numpy as np
import pandas as pd
import pytest

from motioniq.metrics import classification_scores
from motioniq.models import (
    load_classifier_bundle,
    make_random_forest,
    make_temporal_model,
    predict_classifier_bundle,
    save_classifier_bundle,
    sequence_features,
    train_sklearn_baseline,
)
from motioniq.training import grouped_split, prepare_rehab24_features


def test_sklearn_baseline_inference():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array(["a", "a", "b", "b"])
    model = train_sklearn_baseline(x, y)
    assert model.predict([[0, 0]])[0] == "a"


def test_temporal_feature_extraction_uses_timestamped_values_and_velocity():
    frames = [
        {"timestamp_seconds": 0.0, "left_knee_flexion_deg": 0.0},
        {"timestamp_seconds": 0.5, "left_knee_flexion_deg": 30.0},
        {"timestamp_seconds": 1.0, "left_knee_flexion_deg": 60.0},
    ]
    features = sequence_features(frames, 640, 480)
    assert features["left_knee_flexion_mean"] == pytest.approx(30)
    assert features["left_knee_flexion_range"] == pytest.approx(60)
    assert features["left_knee_flexion_mean_abs_velocity"] == pytest.approx(60)
    assert features["right_knee_flexion_valid_fraction"] == 0
    assert np.isnan(features["right_knee_flexion_mean"])


def test_group_split_keeps_subjects_disjoint_and_is_reproducible():
    subjects = np.repeat([f"s{i}" for i in range(10)], 3)
    first = grouped_split(subjects, seed=19)
    second = grouped_split(subjects, seed=19)
    assert np.array_equal(first, second)
    assigned = {part: set(subjects[first == part]) for part in ("train", "validation", "test")}
    assert not assigned["train"] & assigned["validation"]
    assert not assigned["train"] & assigned["test"]
    assert not assigned["validation"] & assigned["test"]


def test_random_forest_bundle_round_trip_and_unseen_video_inference(tmp_path):
    x = np.array([[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.1]])
    y = np.array(["squat", "squat", "lunge", "lunge"])
    model = make_random_forest(seed=12, n_estimators=30, min_samples_leaf=1)
    model.fit(x, y)
    checkpoint = tmp_path / "rf.joblib"
    save_classifier_bundle(
        model,
        checkpoint,
        feature_names=["f1", "f2"],
        class_labels=["lunge", "squat"],
        metadata={"confidence_threshold": 0.5, "held_out_test_metrics": {"accuracy": 1.0}},
    )
    bundle = load_classifier_bundle(checkpoint)
    result = predict_classifier_bundle(bundle, {"f1": 0.0, "f2": 1.0}, confidence_threshold=0.5)
    assert result["status"] == "completed"
    assert result["predicted_class"] == "squat"
    assert result["model_version"] == "motioniq-random-forest-v1"
    assert set(result["probabilities"]) == {"squat", "lunge"}


def test_missing_checkpoint_is_not_loaded():
    from motioniq.models import load_classifier_bundle

    with pytest.raises(FileNotFoundError):
        load_classifier_bundle("missing-motioniq-checkpoint.joblib")


def test_classifier_rejects_unseen_video_with_no_valid_measurements(tmp_path):
    x = np.array([[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.1]])
    model = make_random_forest(seed=1, n_estimators=10, min_samples_leaf=1).fit(
        x, ["squat", "squat", "lunge", "lunge"]
    )
    path = tmp_path / "bundle.joblib"
    save_classifier_bundle(
        model, path, feature_names=["a", "b"], class_labels=["lunge", "squat"], metadata={}
    )
    result = predict_classifier_bundle(load_classifier_bundle(path), {"a": np.nan, "b": np.nan})
    assert result["status"] == "failed"
    assert "predicted_class" not in result
    no_pose = predict_classifier_bundle(
        load_classifier_bundle(path),
        {"a": 1.0, "b": 2.0, "left_knee_flexion_valid_fraction": 0.0},
    )
    assert no_pose["status"] == "failed"


def test_classification_metrics_match_known_example():
    scores = classification_scores(
        np.array(["a", "a", "b", "b"]),
        np.array(["a", "b", "b", "b"]),
        labels=["a", "b"],
    )
    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["precision_macro"] == pytest.approx(5 / 6)
    assert scores["recall_macro"] == pytest.approx(0.75)
    assert scores["f1_macro"] == pytest.approx(11 / 15)
    assert scores["confusion_matrix"] == [[1, 1], [0, 2]]
    assert scores["per_class"]["a"]["support"] == 2


def test_optional_temporal_experiment_uses_lstm():
    pytest.importorskip("torch")
    model = make_temporal_model(input_features=5, classes=3)
    output = model(__import__("torch").zeros((2, 8, 5)))
    assert tuple(output.shape) == (2, 3)
    assert hasattr(model, "lstm")


def test_rehab24_adapter_preserves_subject_labels_frames_and_provenance(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    annotation_rows = []
    for person_id, exercise_id in ((1, 1), (2, 2)):
        video_id = f"PM_{person_id:03}"
        source = videos / f"{video_id}-Camera18-30fps.mp4"
        source.write_bytes(b"test-only placeholder video bytes")
        annotation_rows.append(
            {
                "video_id": video_id,
                "repetition_number": "1",
                "exercise_id": str(exercise_id),
                "person_id": str(person_id),
                "first_frame": "0",
                "last_frame": "30",
                "cam17_orientation": "front",
                "mocap_erroneous": "0",
                "exercise_subtype": "",
                "lights_on": "1",
                "extra_person_in_cam17": "0",
                "extra_person_in_cam18": "0",
                "correctness": "1",
            }
        )
    annotations = tmp_path / "Segmentation.csv"
    pd.DataFrame(annotation_rows).to_csv(annotations, sep=";", index=False)

    def fake_analyze(path, **_kwargs):
        measurement_frames = [
            {"frame": frame, "timestamp_seconds": frame / 30, "left_knee_flexion_deg": float(frame)}
            for frame in (0, 15, 30)
        ]
        pose_frames = [
            {
                "frame": frame["frame"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "landmarks": [],
            }
            for frame in measurement_frames
        ]
        return {
            "video": {"path": str(path), "width": 640, "height": 480},
            "sampling": {"sampled_frames": 3},
            "pose": {"frames": pose_frames},
            "movement_measurements": {"frames": measurement_frames},
        }

    monkeypatch.setattr("motioniq.training.analyze_video", fake_analyze)
    feature_path = prepare_rehab24_features(annotations, videos, tmp_path / "prepared")
    features = pd.read_csv(feature_path)
    assert features["subject_id"].tolist() == [1, 2]
    assert features["label"].tolist() == ["arm_abduction", "arm_vw"]
    assert features["correctness"].tolist() == [1, 1]
    assert features["left_knee_flexion_mean"].tolist() == pytest.approx([15, 15])
    assert (tmp_path / "prepared" / "provenance.json").is_file()
    frame_lines = (tmp_path / "prepared" / "pose_measurement_frames.jsonl").read_text().splitlines()
    assert len(frame_lines) == 6
    assert '"timestamp_seconds": 0.5' in frame_lines[1]
