import numpy as np

from motioniq.geometry import (
    cycle_durations,
    detect_repetitions,
    joint_angle,
    landmark_displacement,
    movement_symmetry,
    trunk_inclination,
)
from motioniq.metrics import mpjpe, pck, segmentation_scores, tracking_stability


def test_joint_angle_and_degenerate_case():
    assert joint_angle([1, 0], [0, 0], [0, 1]) == 90
    assert np.isnan(joint_angle([0, 0], [0, 0], [1, 1]))


def test_trunk_angle_and_displacement():
    assert trunk_inclination([0, -1], [0, 0]) == 0
    np.testing.assert_allclose(landmark_displacement(np.array([[[0, 0]], [[3, 4]]])), [[5]])


def test_symmetry_repetitions_and_duration():
    assert movement_symmetry(np.array([1, 2]), np.array([1, 2])) == 1
    peaks = detect_repetitions(np.array([0, 1, 0, 1, 0]), min_distance=2)
    np.testing.assert_array_equal(peaks, [1, 3])
    np.testing.assert_allclose(cycle_durations(peaks, 2), [1])


def test_evaluation_metrics():
    assert mpjpe(np.array([[0, 0]]), np.array([[3, 4]])) == 5
    assert pck(np.array([[0, 0]]), np.array([[3, 4]]), 5) == 1
    assert segmentation_scores(np.array([1, 1, 0]), np.array([1, 0, 0])) == {
        "iou": 0.5,
        "dice": 2 / 3,
    }
    assert tracking_stability(np.array([[1, 1], [1, 0], [1, 1]])) == 0.5
