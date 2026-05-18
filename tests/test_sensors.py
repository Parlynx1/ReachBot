"""
tests/test_sensors.py
Sensor unit tests — no PyBullet required (uses mock loader).

Run:  python -m pytest tests/test_sensors.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ── Mock robot loader ─────────────────────────────────────────────────

def make_mock_loader():
    loader = MagicMock()
    loader.get_base_pose.return_value = (
        np.array([0., 0., 0.05]),
        np.array([0., 0., 0., 1.])   # identity quaternion xyzw
    )
    loader.arm_id  = 1
    loader.base_id = 0
    return loader


# ── Ultrasonic sensor ─────────────────────────────────────────────────

class TestUltrasonicSensor:

    def setup_method(self):
        self.loader = make_mock_loader()

    @patch("utils.sensors.p")
    def test_read_returns_four_directions(self, mock_p):
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.rayTestBatch.return_value = [(None, None, 0.5, None, None)] * 4
        from utils.sensors import UltrasonicSensor
        us      = UltrasonicSensor(self.loader)
        reading = us.read()
        assert set(reading.distances.keys()) == {"front", "back", "left", "right"}

    @patch("utils.sensors.p")
    def test_distances_clipped_to_max_range(self, mock_p):
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.rayTestBatch.return_value = [(None, None, 1.0, None, None)] * 4
        from utils.sensors import UltrasonicSensor
        us = UltrasonicSensor(self.loader)
        r  = us.read()
        for d in r.distances.values():
            assert 0. <= d <= us.max_range

    @patch("utils.sensors.p")
    def test_noise_is_applied(self, mock_p):
        """50 readings with the same hit fraction must not all be identical."""
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.rayTestBatch.return_value = [(None, None, 0.5, None, None)] * 4
        from utils.sensors import UltrasonicSensor
        us   = UltrasonicSensor(self.loader)
        vals = [us.read().distances["front"] for _ in range(50)]
        assert len(set(round(v, 6) for v in vals)) > 1

    @patch("utils.sensors.p")
    def test_all_distances_are_floats(self, mock_p):
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.rayTestBatch.return_value = [(None, None, 0.3, None, None)] * 4
        from utils.sensors import UltrasonicSensor
        us = UltrasonicSensor(self.loader)
        r  = us.read()
        for d in r.distances.values():
            assert isinstance(d, float)


# ── Camera sensor ─────────────────────────────────────────────────────

class TestCameraSensor:

    def setup_method(self):
        self.loader = make_mock_loader()

    @patch("utils.sensors.p")
    def test_rgb_shape_correct(self, mock_p):
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.computeProjectionMatrixFOV.return_value = [1] * 16
        mock_p.computeViewMatrix.return_value          = [1] * 16
        fake_rgba  = np.ones(320*240*4, dtype=np.uint8) * 128
        fake_depth = np.full(320*240, 0.5, dtype=np.float32)
        mock_p.getCameraImage.return_value = (320, 240, fake_rgba, fake_depth, None)
        from utils.sensors import CameraSensor
        cam = CameraSensor(self.loader)
        r   = cam.read()
        assert r.rgb.shape == (240, 320, 3)

    @patch("utils.sensors.p")
    def test_depth_shape_correct(self, mock_p):
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.computeProjectionMatrixFOV.return_value = [1] * 16
        mock_p.computeViewMatrix.return_value          = [1] * 16
        fake_rgba  = np.ones(320*240*4, dtype=np.uint8)
        fake_depth = np.full(320*240, 0.5, dtype=np.float32)
        mock_p.getCameraImage.return_value = (320, 240, fake_rgba, fake_depth, None)
        from utils.sensors import CameraSensor
        cam = CameraSensor(self.loader)
        r   = cam.read()
        assert r.depth.shape == (240, 320)

    @patch("utils.sensors.p")
    def test_depth_linearised_positive(self, mock_p):
        """Linearised depth values must all be positive metres."""
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.computeProjectionMatrixFOV.return_value = [1] * 16
        mock_p.computeViewMatrix.return_value          = [1] * 16
        fake_rgba  = np.ones(320*240*4, dtype=np.uint8)
        fake_depth = np.full(320*240, 0.8, dtype=np.float32)
        mock_p.getCameraImage.return_value = (320, 240, fake_rgba, fake_depth, None)
        from utils.sensors import CameraSensor
        cam = CameraSensor(self.loader)
        r   = cam.read()
        assert (r.depth > 0).all()

    @patch("utils.sensors.p")
    def test_rgb_dtype_uint8(self, mock_p):
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.computeProjectionMatrixFOV.return_value = [1] * 16
        mock_p.computeViewMatrix.return_value          = [1] * 16
        fake_rgba  = np.ones(320*240*4, dtype=np.uint8) * 200
        fake_depth = np.full(320*240, 0.5, dtype=np.float32)
        mock_p.getCameraImage.return_value = (320, 240, fake_rgba, fake_depth, None)
        from utils.sensors import CameraSensor
        cam = CameraSensor(self.loader)
        r   = cam.read()
        assert r.rgb.dtype == np.uint8


# ── LiDAR sensor ──────────────────────────────────────────────────────

class TestLiDARSensor:

    def setup_method(self):
        self.loader = make_mock_loader()

    @patch("utils.sensors.p")
    def test_returns_360_rays(self, mock_p):
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.rayTestBatch.return_value = [(None, None, 0.5, None, None)] * 360
        from utils.sensors import LiDARSensor
        lidar = LiDARSensor(self.loader)
        r     = lidar.read()
        assert len(r.angles)      == 360
        assert len(r.distances)   == 360
        assert r.hit_points.shape == (360, 2)

    @patch("utils.sensors.p")
    def test_distances_within_range(self, mock_p):
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.rayTestBatch.return_value = [(None, None, 0.6, None, None)] * 360
        from utils.sensors import LiDARSensor
        lidar = LiDARSensor(self.loader)
        r     = lidar.read()
        assert (r.distances >= lidar.min_r).all()
        assert (r.distances <= lidar.max_r).all()

    @patch("utils.sensors.p")
    def test_angles_span_full_circle(self, mock_p):
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.rayTestBatch.return_value = [(None, None, 0.5, None, None)] * 360
        from utils.sensors import LiDARSensor
        lidar = LiDARSensor(self.loader)
        r     = lidar.read()
        assert r.angles[0]  == pytest.approx(0., abs=1e-6)
        assert r.angles[-1] <  2 * np.pi      # endpoint=False in linspace

    @patch("utils.sensors.p")
    def test_hit_points_are_2d(self, mock_p):
        mock_p.getMatrixFromQuaternion.return_value = [1,0,0, 0,1,0, 0,0,1]
        mock_p.rayTestBatch.return_value = [(None, None, 0.4, None, None)] * 360
        from utils.sensors import LiDARSensor
        lidar = LiDARSensor(self.loader)
        r     = lidar.read()
        assert r.hit_points.ndim == 2
        assert r.hit_points.shape[1] == 2


# ── Force sensor ──────────────────────────────────────────────────────

class TestForceSensor:

    def setup_method(self):
        self.loader = make_mock_loader()

    @patch("utils.sensors.p")
    def test_no_contact_near_zero(self, mock_p):
        """No contact points → force magnitude should be near zero (noise only)."""
        mock_p.getContactPoints.return_value = []
        from utils.sensors import ForceSensor
        fs = ForceSensor(self.loader)
        r  = fs.read()
        assert r.magnitude < 0.5
        assert r.force.shape == (3,)

    @patch("utils.sensors.p")
    def test_contact_produces_nonzero_force(self, mock_p):
        """A contact with 10 N normal force must give a nonzero magnitude."""
        contact = (None,None,None,None,None,None,None,
                   (1., 0., 0.), None, 10., None, None)
        mock_p.getContactPoints.return_value = [contact]
        from utils.sensors import ForceSensor
        fs = ForceSensor(self.loader)
        r  = fs.read()
        assert r.magnitude > 0.

    @patch("utils.sensors.p")
    def test_force_clipped_to_max(self, mock_p):
        """A 1000 N contact must be clipped to max_force."""
        contact = (None,None,None,None,None,None,None,
                   (1., 0., 0.), None, 1000., None, None)
        mock_p.getContactPoints.return_value = [contact]
        from utils.sensors import ForceSensor
        fs = ForceSensor(self.loader)
        r  = fs.read()
        assert abs(r.force[0]) <= fs.max_f + 1.0   # +1 for noise tolerance

    @patch("utils.sensors.p")
    def test_torque_shape(self, mock_p):
        mock_p.getContactPoints.return_value = []
        from utils.sensors import ForceSensor
        fs = ForceSensor(self.loader)
        r  = fs.read()
        assert r.torque.shape == (3,)