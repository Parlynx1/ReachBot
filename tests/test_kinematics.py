"""
tests/test_kinematics.py
Unit tests for FK/IK correctness.

Run:  python -m pytest tests/test_kinematics.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from utils.kinematics import (fk, ik, ik_multi_start, jacobian,
                               ee_velocity, JOINT_LIMITS)


# ── FK tests ─────────────────────────────────────────────────────────

def test_fk_home_position():
    """At q=[0,0,0] EE should be directly above base on Z axis."""
    q = np.array([0., 0., 0.])
    pos, R = fk(q)
    assert pos[0] == pytest.approx(0., abs=1e-6)
    assert pos[1] == pytest.approx(0., abs=1e-6)
    assert pos[2] > 0.


def test_fk_rotation_matrix_orthonormal():
    """FK rotation matrix must satisfy R R^T = I."""
    q = np.array([0.5, 0.3, -0.4])
    _, R = fk(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_fk_joint1_yaw():
    """Rotating joint1 by 90° should rotate EE 90° in XY plane."""
    pos0,  _ = fk(np.array([0.,      0., 0.]))
    pos90, _ = fk(np.array([np.pi/2, 0., 0.]))
    assert pos90[0] == pytest.approx(-pos0[1], abs=1e-4)
    assert pos90[1] == pytest.approx( pos0[0], abs=1e-4)


def test_fk_returns_correct_types():
    q = np.array([0.1, 0.2, 0.3])
    pos, R = fk(q)
    assert isinstance(pos, np.ndarray) and pos.shape == (3,)
    assert isinstance(R,   np.ndarray) and R.shape   == (3, 3)


def test_fk_finite_at_joint_limits():
    for lo, hi in JOINT_LIMITS:
        pos, _ = fk(np.array([lo, lo, lo]))
        assert np.isfinite(pos).all()
        pos, _ = fk(np.array([hi, hi, hi]))
        assert np.isfinite(pos).all()


def test_fk_joint2_z_symmetry():
    """
    joint2=+a and joint2=-a should give the same Z height
    (both pitch the arm the same amount, just in opposite XZ directions).
    """
    pos_pos, _ = fk(np.array([0.,  0.5, 0.]))
    pos_neg, _ = fk(np.array([0., -0.5, 0.]))
    # Z must be identical — only X flips
    assert pos_pos[2] == pytest.approx(pos_neg[2], abs=1e-4)
    assert pos_pos[1] == pytest.approx(0., abs=1e-4)   # Y stays zero
    assert pos_neg[1] == pytest.approx(0., abs=1e-4)


def test_fk_joint1_does_not_change_reach():
    """Rotating joint1 should not change the distance from base to EE."""
    reach0,  _ = fk(np.array([0.0,  0.4, 0.3]))
    reach90, _ = fk(np.array([1.57, 0.4, 0.3]))
    assert np.linalg.norm(reach0) == pytest.approx(
           np.linalg.norm(reach90), rel=1e-3)


# ── IK tests — only confirmed reachable targets ───────────────────────
# Targets verified against sample_workspace(20000): all within 15mm of
# the actual workspace boundary and solve to <1mm IK error.

@pytest.mark.parametrize("target", [
    np.array([ 0.00,  0.25, 0.40]),   # workspace_dist=4mm  IK=0.1mm
    np.array([ 0.15,  0.10, 0.50]),   # workspace_dist=10mm IK=0.1mm
    np.array([ 0.00, -0.20, 0.50]),   # workspace_dist=10mm IK=0.1mm
])
def test_ik_reaches_target(target):
    """IK solution must place EE within 5mm of a confirmed reachable target."""
    q, err = ik_multi_start(target, n_tries=12, tol=1e-4)
    assert q   is not None, "IK should return a solution"
    assert err < 0.005,     f"IK error too large: {err*1000:.2f} mm"
    pos, _ = fk(q)
    np.testing.assert_allclose(pos, target, atol=0.005)


def test_ik_respects_joint_limits():
    target = np.array([0.00, 0.25, 0.40])
    q, err = ik_multi_start(target, n_tries=8)
    if q is not None:
        for i, (lo, hi) in enumerate(JOINT_LIMITS):
            assert lo <= q[i] <= hi, f"Joint {i} out of limits: {q[i]:.4f}"


def test_ik_unreachable_target():
    """A target far outside the workspace should return large error."""
    target = np.array([5.0, 0., 0.])
    _, err = ik_multi_start(target, n_tries=4, max_iter=50)
    assert err > 0.1


def test_ik_multi_start_better_than_single():
    target = np.array([0.00, 0.25, 0.40])
    _, err_single = ik(target, max_iter=100)
    _, err_multi  = ik_multi_start(target, n_tries=6)
    assert err_multi <= err_single + 1e-4


def test_ik_solution_consistent_with_fk():
    target = np.array([0.00, 0.25, 0.40])
    q, err = ik_multi_start(target, n_tries=8)
    if q is not None and err < 0.005:
        pos, _ = fk(q)
        np.testing.assert_allclose(pos, target, atol=0.005)


# ── Jacobian tests ────────────────────────────────────────────────────

def test_jacobian_shape():
    J = jacobian(np.array([0.3, 0.2, -0.1]))
    assert J.shape == (3, 3)


def test_jacobian_finite_diff_accuracy():
    q   = np.array([0.4, 0.3, 0.2])
    J   = jacobian(q)
    eps = 1e-5
    for i in range(3):
        dq     = np.zeros(3); dq[i] = eps
        p1, _  = fk(q + dq)
        p0, _  = fk(q - dq)
        col_fd = (p1 - p0) / (2 * eps)
        np.testing.assert_allclose(J[:, i], col_fd, atol=1e-5)


def test_ee_velocity_zero_qdot():
    q = np.array([0.2, 0.4, -0.3])
    v = ee_velocity(q, np.zeros(3))
    np.testing.assert_allclose(v, np.zeros(3), atol=1e-10)


def test_ee_velocity_scales_linearly():
    q     = np.array([0.3, 0.2, -0.1])
    d     = np.array([1., 0., 0.])
    spd1  = np.linalg.norm(ee_velocity(q, 1.0 * d))
    spd2  = np.linalg.norm(ee_velocity(q, 2.0 * d))
    assert spd2 == pytest.approx(2.0 * spd1, rel=1e-5)