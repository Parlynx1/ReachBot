"""
env/robot_loader.py
Loads the robot into PyBullet.

Two loading modes:
  1. SPLIT (default) — loads base.urdf + kuka_arm.urdf separately,
     attaches arm with a fixed constraint. Most reliable for PyBullet.

  2. FULL — loads robot_full.urdf as a single body.
     Use RobotLoader(mode='full') for this.
     Note: robot_full.urdf references mesh files — PyBullet falls back
     to primitives if meshes are not found (this is intentional).

Usage:
    loader = RobotLoader()           # split mode (default)
    loader = RobotLoader(mode='full') # full URDF mode
    loader.load()
"""

import pybullet as p
import yaml, os
import numpy as np

_DIR      = os.path.dirname(os.path.abspath(__file__))
SIM_CFG   = os.path.join(_DIR, "../assets/configs/sim_params.yaml")
BASE_URDF = os.path.join(_DIR, "../urdf/base/base.urdf")
ARM_URDF  = os.path.join(_DIR, "../urdf/arm/kuka_arm.urdf")
FULL_URDF = os.path.join(_DIR, "../urdf/robot_full.urdf")


class RobotLoader:
    """
    Loads base + arm into PyBullet and provides unified control interface.

    Attributes
    ----------
    base_id       : int   — PyBullet body id of mobile base
    arm_id        : int   — PyBullet body id of arm  (== base_id in full mode)
    joint_map     : dict  — arm joint name → joint index
    wheel_joints  : list  — 4 wheel joint indices (on base body)
    finger_indices: list  — 2 finger joint indices
    """

    ARM_JOINTS    = ["joint1", "joint2", "joint3"]
    FINGER_JOINTS = ["finger_left_joint", "finger_right_joint"]
    WHEEL_NAMES   = ["joint_wheel_fl", "joint_wheel_fr",
                     "joint_wheel_rl", "joint_wheel_rr"]

    def __init__(self, sim_cfg=SIM_CFG,
                 base_urdf=BASE_URDF, arm_urdf=ARM_URDF,
                 full_urdf=FULL_URDF, mode="split"):
        with open(sim_cfg) as f:
            cfg = yaml.safe_load(f)

        self._mode       = mode
        self._base_urdf  = base_urdf
        self._arm_urdf   = arm_urdf
        self._full_urdf  = full_urdf
        self._start_pos  = list(cfg["robot"]["start_pos"])

        # Defensive: YAML may return a scalar 0.0 instead of [0,0,0]
        # if the list was written oddly. Always normalise to 3-element list.
        raw_orn = cfg["robot"]["start_orn"]
        if isinstance(raw_orn, (int, float)):
            self._start_orn = [0.0, 0.0, float(raw_orn)]
        else:
            orn_list = list(raw_orn)
            if len(orn_list) == 3:
                self._start_orn = [float(v) for v in orn_list]
            else:
                self._start_orn = [0.0, 0.0, 0.0]

        self.base_id       = None
        self.arm_id        = None
        self._constraint   = None
        self.joint_map     = {}
        self.wheel_joints  = []
        self.finger_indices= []

    # ── Public API ───────────────────────────────────────────────────

    def load(self):
        """Load robot. Returns (base_id, arm_id)."""
        if self._mode == "full":
            return self._load_full()
        return self._load_split()

    def get_base_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.base_id)
        return np.array(pos), np.array(orn)

    def get_ee_pose(self):
        idx = self._link_idx(self.arm_id, "end_effector_link")
        st  = p.getLinkState(self.arm_id, idx, computeForwardKinematics=True)
        return np.array(st[4]), np.array(st[5])

    def get_joint_positions(self) -> np.ndarray:
        return np.array([
            p.getJointState(self.arm_id, self.joint_map[n])[0]
            for n in self.ARM_JOINTS
        ])

    def set_joint_positions(self, angles, max_force=50.0):
        for name, angle in zip(self.ARM_JOINTS, angles):
            p.setJointMotorControl2(
                self.arm_id, self.joint_map[name],
                p.POSITION_CONTROL,
                targetPosition=float(angle),
                force=max_force)

    def set_gripper(self, open_frac=1.0, max_force=20.0):
        target = 0.04 * float(np.clip(open_frac, 0, 1))
        for idx in self.finger_indices:
            p.setJointMotorControl2(
                self.arm_id, idx,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=max_force)

    def set_wheel_velocities(self, vfl, vfr, vrl, vrr, max_force=10.0):
        for jidx, vel in zip(self.wheel_joints, [vfl, vfr, vrl, vrr]):
            p.setJointMotorControl2(
                self.base_id, jidx,
                p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=max_force)

    def reset_pose(self):
        p.resetBasePositionAndOrientation(
            self.base_id, self._start_pos,
            p.getQuaternionFromEuler([0, 0, 0]))
        for name in self.ARM_JOINTS:
            p.resetJointState(self.arm_id, self.joint_map[name], 0.0)

    # ── Private: split mode ──────────────────────────────────────────

    def _load_split(self):
        orn_q = p.getQuaternionFromEuler(self._start_orn)
        pos   = self._start_pos

        # Load base
        self.base_id = p.loadURDF(
            self._base_urdf,
            basePosition=pos,
            baseOrientation=orn_q,
            useFixedBase=False)
        self._configure_friction(self.base_id)
        self._map_joints_on(self.base_id, body_is_base=True)

        # Load arm slightly above mount
        arm_pos = [pos[0]+0.05, pos[1], pos[2]+0.13]
        self.arm_id = p.loadURDF(
            self._arm_urdf,
            basePosition=arm_pos,
            baseOrientation=orn_q,
            useFixedBase=False)
        self._map_joints_on(self.arm_id, body_is_base=False)
        self._attach_arm_to_base()
        return self.base_id, self.arm_id

    def _attach_arm_to_base(self):
        mount_idx = self._link_idx(self.base_id, "arm_mount")
        self._constraint = p.createConstraint(
            self.base_id, mount_idx,
            self.arm_id, -1,
            p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

    # ── Private: full mode ───────────────────────────────────────────

    def _load_full(self):
        orn_q = p.getQuaternionFromEuler(self._start_orn)
        body  = p.loadURDF(
            self._full_urdf,
            basePosition=self._start_pos,
            baseOrientation=orn_q,
            useFixedBase=False)
        self.base_id = body
        self.arm_id  = body
        self._configure_friction(body)
        self._map_joints_on(body, body_is_base=True)
        self._map_joints_on(body, body_is_base=False)
        return body, body

    # ── Private: helpers ─────────────────────────────────────────────

    def _configure_friction(self, body_id):
        for i in range(p.getNumJoints(body_id)):
            p.changeDynamics(body_id, i,
                             lateralFriction=0.8,
                             spinningFriction=0.02,
                             rollingFriction=0.002)

    def _map_joints_on(self, body_id, body_is_base: bool):
        for i in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, i)
            name = info[1].decode()
            if body_is_base and name in self.WHEEL_NAMES:
                if i not in self.wheel_joints:
                    self.wheel_joints.append(i)
            if name in self.ARM_JOINTS + self.FINGER_JOINTS:
                self.joint_map[name] = i
            if name in self.FINGER_JOINTS and i not in self.finger_indices:
                self.finger_indices.append(i)

    @staticmethod
    def _link_idx(body_id, link_name) -> int:
        for i in range(p.getNumJoints(body_id)):
            if p.getJointInfo(body_id, i)[12].decode() == link_name:
                return i
        raise ValueError(f"Link '{link_name}' not found in body {body_id}")