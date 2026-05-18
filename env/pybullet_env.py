"""
env/pybullet_env.py
Initialises a PyBullet simulation: physics server, ground plane, camera view.

macOS note: PyBullet GUI can hang on macOS if rendering features are enabled
before the window is ready. This version disables problematic features at
startup and adds a short settle delay to prevent the freeze.
"""

import pybullet as p
import pybullet_data
import yaml
import os
import time
import platform

_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_DIR, "../assets/configs/sim_params.yaml")

# Detect macOS for platform-specific workarounds
_IS_MACOS = platform.system() == "Darwin"


class PyBulletEnv:
    """Wrapper around a PyBullet physics server."""

    def __init__(self, config_path: str = CONFIG_PATH, gui: bool = True):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg       = cfg
        self.gui       = gui
        self.time_step = cfg["simulation"]["time_step"]
        self._client   = None
        self.plane_id  = None

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> int:
        mode = p.GUI if self.gui else p.DIRECT
        self._client = p.connect(mode)

        if self.gui and _IS_MACOS:
            # Disable all rendering during load to prevent macOS OpenGL hang
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        gx, gy, gz = self.cfg["simulation"]["gravity"]
        p.setGravity(gx, gy, gz)
        p.setPhysicsEngineParameter(
            numSolverIterations=self.cfg["simulation"]["solver_iterations"],
            fixedTimeStep=self.time_step,
        )
        self.plane_id = p.loadURDF("plane.urdf")

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0],
            )
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

            if _IS_MACOS:
                # Re-enable rendering after all URDFs are loaded
                # Small sleep lets the macOS window server catch up
                time.sleep(0.5)
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

        return self._client

    def step(self):
        p.stepSimulation()
        if self.cfg["simulation"]["real_time"]:
            time.sleep(self.time_step)

    def step_n(self, n: int):
        for _ in range(n):
            self.step()

    def reset(self):
        p.resetSimulation()
        gx, gy, gz = self.cfg["simulation"]["gravity"]
        p.setGravity(gx, gy, gz)
        self.plane_id = p.loadURDF("plane.urdf")

    def close(self):
        if self._client is not None:
            try:
                p.disconnect(self._client)
            except Exception:
                pass
            self._client = None

    @property
    def client(self) -> int:
        return self._client

    def set_real_time(self, enabled: bool):
        self.cfg["simulation"]["real_time"] = enabled
        p.setRealTimeSimulation(int(enabled))

    def add_debug_text(self, text: str, position, color=(1, 1, 1), size=1.2):
        return p.addUserDebugText(text, position, color, size)