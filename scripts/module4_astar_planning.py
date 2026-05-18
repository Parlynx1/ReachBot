"""
scripts/module4_astar_planning.py
A* global path planning through an obstacle environment.

Run:  python scripts/module4_astar_planning.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time, numpy as np, pybullet as p
from env.pybullet_env    import PyBulletEnv
from env.robot_loader    import RobotLoader
from env.obstacle_builder import ObstacleBuilder
from utils.path_planning  import OccupancyGrid, AStarPlanner, PurePursuitTracker
from utils.controllers    import OmniDriveController
from utils.visualizer     import DebugVisualizer

START_XY = (0.0, 0.0)
GOAL_XY  = (3.0, 3.0)


def main():
    print("=" * 55)
    print("MODULE 4 — A* Global Path Planning")
    print("=" * 55)

    env    = PyBulletEnv(gui=True);  env.start()
    loader = RobotLoader();          loader.load()
    loader.set_joint_positions([0.0, -0.5, 0.8])
    env.step_n(60)

    obs = ObstacleBuilder();  obs.build_default_maze()
    env.step_n(20)

    print("[Planner] Building occupancy grid ...")
    grid = OccupancyGrid(resolution=0.1, size_x=10.0, size_y=10.0, inflation=0.30)
    grid.load_from_obstacle_builder(obs)

    print(f"[Planner] A* from {START_XY} → {GOAL_XY}")
    planner   = AStarPlanner(grid)
    waypoints = planner.plan(START_XY, GOAL_XY)

    if waypoints is None:
        print("[Planner] ✗ No path found."); input("Enter to exit"); env.close(); return

    print(f"[Planner] ✓ {len(waypoints)} waypoints found")

    viz = DebugVisualizer()
    viz.draw_path(waypoints, color=(0.,1.,0.2))
    viz.draw_waypoints(waypoints[::5])
    viz.label("START", list(START_XY)+[0.3], color=(0,1,0))
    viz.label("GOAL",  list(GOAL_XY) +[0.3], color=(1,0.3,0))

    omni    = OmniDriveController(loader)
    tracker = PurePursuitTracker([w.astype(float) for w in waypoints],
                                  look_ahead=0.35, max_lin=0.4)
    step = 0
    while not tracker.done and step < 15_000:
        pos, orn = loader.get_base_pose()
        yaw = float(np.arctan2(2*(orn[3]*orn[2]+orn[0]*orn[1]),
                               1-2*(orn[1]**2+orn[2]**2)))
        vx, vy, wz = tracker.step(pos[:2], yaw)
        omni.cmd_vel_world(vx, vy, wz)
        env.step(); step += 1
        if step % 500 == 0:
            print(f"  step {step:5d}  dist {np.linalg.norm(pos[:2]-np.array(GOAL_XY)):.2f} m")

    omni.stop(); env.step_n(30)
    pos, _ = loader.get_base_pose()
    dist   = np.linalg.norm(pos[:2] - np.array(GOAL_XY))
    print(f"\n[Result] Final dist to goal: {dist:.3f} m")
    print("[Result] ✓ Done." if dist < 0.25 else "[Result] ✗ Did not reach goal.")

    print("\nClose PyBullet window to exit.")
    try:
        while True: env.step(); time.sleep(0.01)
    except Exception: pass
    env.close()


if __name__ == "__main__":
    main()
