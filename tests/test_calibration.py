import pytest
import numpy as np
from calibration.calibrate_metric import _fit_calibration

def test_fit_calibration_basic():
    true_scale = 2.5
    true_offset = 0.1
    raw_depths = np.array([0.5, 1.0, 2.0])
    distances = true_scale / raw_depths + true_offset
    samples = list(zip(raw_depths.tolist(), distances.tolist()))
    scale, offset = _fit_calibration(samples)
    assert pytest.approx(scale, rel=1e-6) == true_scale
    assert pytest.approx(offset, rel=1e-6) == true_offset

def test_fit_calibration_insufficient_samples():
    with pytest.raises(ValueError, match="At least two calibration samples"):
        _fit_calibration([(1.0, 2.0)])

def test_fit_calibration_invalid_values():
    with pytest.raises(ValueError, match="Invalid raw depth"):
        _fit_calibration([(0.0, 1.0), (1.0, 2.0)])
    with pytest.raises(ValueError, match="Invalid measured distance"):
        _fit_calibration([(1.0, -1.0), (2.0, 1.0)])
    with pytest.raises(ValueError, match="Invalid raw depth"):
        _fit_calibration([(float('nan'), 1.0), (1.0, 2.0)])
