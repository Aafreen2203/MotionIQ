import numpy as np

from motioniq.models import train_sklearn_baseline


def test_sklearn_baseline_inference():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array(["a", "a", "b", "b"])
    model = train_sklearn_baseline(x, y)
    assert model.predict([[0, 0]])[0] == "a"
