# ReachBot — Reach Mobility Bot Simulation

A PyBullet simulation of a combined mobile manipulator:
- **Base**: 4-omni-wheel mobile platform with ultrasonic, camera, and LiDAR sensors
- **Arm**: 3-DOF KUKA-style arm with parallel gripper and end-effector force sensor

## Project Structure

```
reach_mobility_bot/
├── urdf/           Robot URDF definitions
├── assets/         Config files, maps, meshes
├── env/            PyBullet environment + robot loader
├── utils/          Kinematics, controllers, sensors, path planning
├── notebooks/      Modules 1-3 (Jupyter .ipynb)
├── scripts/        Modules 4-6 (runnable .py)
└── tests/          Unit tests
```

## Modules

| Module | Type | Description |
|--------|------|-------------|
| 1 | Notebook | Base basics: position estimation, control, sensor tests |
| 2 | Notebook | Hand basics: FK, IK, workspace, velocity, force sensor |
| 3 | Notebook | Line trace: follow a path from an image file |
| 4 | Script | A* global path planning through obstacle environment |
| 5 | Script | Whole-bot controller: pick ball with gripper |
| 6 | Script | Main delivery: detect → plan → navigate → deliver |

## Setup

```bash
conda create -n reachbot python=3.10
conda activate reachbot
conda install -c conda-forge pybullet -y
pip install -r requirements.txt
```

## Running

```bash
# Notebooks (Modules 1-3)
jupyter notebook

# Scripts (Modules 4-6)
python scripts/module4_astar_planning.py
python scripts/module5_bot_controller.py
python scripts/module6_main_delivery.py
```

## Notes
- Always run from the project root directory
- Close the PyBullet GUI window between runs
- Only one PyBullet connection at a time