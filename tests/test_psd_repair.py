"""Covariance repair used by the MFA adapter (runs without Stone Soup).

    python -m pytest tests/test_psd_repair.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cidgen.trackers.mfa_ss import psd_repair  # noqa: E402


def test_valid_covariance_is_unchanged():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (6, 6)); P = A @ A.T + np.eye(6)
    Pf, worst, bad = psd_repair(P)
    assert not bad and worst > 0 and np.allclose(Pf, P)


def test_indefinite_covariance_is_repaired():
    P = np.diag([1e4, 1e4, -50.0, 100.0, 100.0, 100.0])        # negative variance, as in the warnings
    P[0, 3] = P[3, 0] = 900.0
    Pf, worst, bad = psd_repair(P)
    assert bad and worst < 0
    assert np.allclose(Pf, Pf.T) and np.linalg.eigvalsh(Pf).min() > 0
    assert np.trace(Pf[:3, :3]) > 0                              # sqrt(trace) is now defined
    assert np.abs(Pf - P)[np.ix_([0, 1], [0, 1])].max() < 1.0  # the healthy part is barely touched


def test_asymmetric_round_off_is_symmetrised():
    P = np.eye(3) * 100.0; P[0, 1] = 1e-9
    Pf, _, bad = psd_repair(P)
    assert not bad and np.array_equal(Pf, Pf.T)


def test_non_finite_is_reported():
    P = np.eye(3); P[1, 1] = np.nan
    Pf, worst, bad = psd_repair(P)
    assert Pf is None and bad and np.isnan(worst)
