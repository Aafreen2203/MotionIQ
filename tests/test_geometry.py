import numpy as np

from motioniq.geometry import (
    cycle_durations,
    detect_repetitions,
    joint_angle,
    knee_flexion,
    landmark_displacement,
    movement_symmetry,
    trunk_inclination,
)
from motioniq.metrics import (
    classification_scores,
    mpjpe,
    pck,
    segmentation_scores,
    tracking_stability,
)


def test_joint_angle_and_degenerate_case():
    assert joint_angle([1, 0], [0, 0], [0, 1]) == 90
    assert np.isnan(joint_angle([0, 0], [0, 0], [1, 1]))


def test_internal_knee_angles_and_flexion_conversion():
    straight = joint_angle([0, -1], [0, 0], [0, 1])
    right_angle = joint_angle([0, -1], [0, 0], [1, 0])
    intermediate = joint_angle([0, -1], [0, 0], [np.sqrt(0.5), np.sqrt(0.5)])
    assert straight == 180
    assert right_angle == 90
    np.testing.assert_allclose(intermediate, 135)
    assert knee_flexion(straight) == 0
    assert knee_flexion(right_angle) == 90
    np.testing.assert_allclose(knee_flexion(intermediate), 45)
    assert np.isnan(knee_flexion(float("nan")))
    assert np.isnan(joint_angle([0, 0], [np.nan, 1], [1, 1]))


def test_trunk_angle_and_displacement():
    assert trunk_inclination([0, -1], [0, 0]) == 0
    np.testing.assert_allclose(landmark_displacement(np.array([[[0, 0]], [[3, 4]]])), [[5]])


def test_trunk_inclination_uses_pixel_axes_and_image_right_sign():
    hip_mid = np.array([500.0, 400.0])
    assert trunk_inclination([500.0, 300.0], hip_mid) == 0
    right_x = 500.0 + 100.0 * np.tan(np.deg2rad(15))
    left_x = 500.0 - 100.0 * np.tan(np.deg2rad(15))
    np.testing.assert_allclose(trunk_inclination([right_x, 300.0], hip_mid), 15.0)
    np.testing.assert_allclose(trunk_inclination([left_x, 300.0], hip_mid), -15.0)
    assert np.isnan(trunk_inclination([np.inf, 300], hip_mid))


def test_symmetry_repetitions_and_duration():
    assert movement_symmetry(np.array([1, 2]), np.array([1, 2])) == 1
    peaks = detect_repetitions(np.array([0, 1, 0, 1, 0]), min_distance=2)
    np.testing.assert_array_equal(peaks, [1, 3])
    np.testing.assert_allclose(cycle_durations(peaks, 2), [1])
    score = movement_symmetry(np.array([0.2, 0.1]), np.array([0.1, 0.05]))
    assert 0 <= score <= 1
    assert np.isnan(movement_symmetry(np.array([np.nan]), np.array([0.1])))
    assert np.isnan(movement_symmetry(np.zeros(2), np.zeros(2)))


def test_evaluation_metrics():
    assert mpjpe(np.array([[0, 0]]), np.array([[3, 4]])) == 5
    assert pck(np.array([[0, 0]]), np.array([[3, 4]]), 5) == 1
    assert segmentation_scores(np.array([1, 1, 0]), np.array([1, 0, 0])) == {
        "iou": 0.5,
        "dice": 2 / 3,
    }
    assert tracking_stability(np.array([[1, 1], [1, 0], [1, 1]])) == 0.5


def test_classification_evaluation_metrics_include_confusion_matrix():
    scores = classification_scores(np.array(["good", "poor"]), np.array(["good", "good"]))
    assert scores["accuracy"] == 0.5
    assert scores["precision_weighted"] == 0.25
    assert scores["recall_weighted"] == 0.5
    assert scores["confusion_matrix"] == [[1, 0], [1, 0]]
