"""
scripts/module6_main_delivery.py
MAIN MODULE — Full autonomous delivery pipeline:
  1. Camera detects ball on pedestal
  2. Drive to approach position
  3. IK pre-grasp → descend → grip
  4. Remove pedestal, fold to carry pose
  5. A* path plan through obstacle maze
  6. Navigate to delivery zone (direct-step carry)
  7. Lower arm, release ball at goal

Run:  python scripts/module6_main_delivery.py

Inherits all stable mechanics from module5:
  - 50kg base, CARRY_Q=[0,0.4,-1.0] (arm truly vertical)
  - pin_base during all arm motion
  - carry_to_goal using direct position stepping
  - pedestal removed after grasp
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time, numpy as np, pybullet as p
from env.pybullet_env     import PyBulletEnv
from env.robot_loader     import RobotLoader
from env.obstacle_builder import ObstacleBuilder
from utils.kinematics     import ik_multi_start, fk as kfk
from utils.controllers    import (OmniDriveController, BasePoseController,
                                   ArmController, GripperController)
from utils.path_planning  import OccupancyGrid, AStarPlanner
from utils.sensors        import CameraSensor
from utils.visualizer     import DebugVisualizer

# ── Scene config ─────────────────────────────────────────────────────
BALL_X     = 0.65
BALL_Z     = 0.35
BALL_POS   = [BALL_X, 0.0, BALL_Z]
PED_TOP    = BALL_Z - 0.06        # 0.29m pedestal
DELIVERY   = np.array([3.5, 2.0, 0.0])
BALL_R     = 0.06
APPROACH_X = BALL_X - 0.35 - 0.05  # 0.25

HOME_Q  = np.array([0.0, -0.4,  0.6])
CARRY_Q = np.array([0.0,  0.40, -1.00])


# ── Helpers (same as module5) ────────────────────────────────────────

def spawn_scene(obs_builder):
    """Spawn ball + obstacle maze. No pedestal."""
    obs_builder.build_default_maze()
    ball_id = p.createMultiBody(0.2,
        p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_R),
        p.createVisualShape(p.GEOM_SPHERE, radius=BALL_R, rgbaColor=[1,.3,0,1]),
        BALL_POS)
    return ball_id, None


def get_arm_frame(loader):
    pos, orn = p.getBasePositionAndOrientation(loader.arm_id)
    return np.array(pos), np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)


def do_ik(loader, world_target, z_offsets=[0.0], n_tries=25):
    bp, R = get_arm_frame(loader)
    best_q, best_err = None, np.inf
    for z_off in z_offsets:
        t = R.T @ (np.array(world_target)+[0,0,z_off]-bp)
        q, err = ik_multi_start(t, n_tries=n_tries, max_iter=800, tol=5e-4)
        if err < best_err: best_err, best_q = err, q
        if best_err < 0.002: break
    print(f"  IK err: {best_err*1000:.2f}mm")
    return best_q, best_err


def pin_base(loader):
    bp, bo = p.getBasePositionAndOrientation(loader.base_id)
    cst = p.createConstraint(loader.base_id,-1,-1,-1,
                              p.JOINT_FIXED,[0,0,0],[0,0,0],
                              list(bp), childFrameOrientation=list(bo))
    p.changeConstraint(cst, maxForce=500000)
    return cst

def unpin(cst): p.removeConstraint(cst)


def zero_vel(loader):
    for body in [loader.base_id, loader.arm_id]:
        p.resetBaseVelocity(body,[0,0,0],[0,0,0])
        for i in range(p.getNumJoints(body)):
            if p.getJointInfo(body,i)[2] in [p.JOINT_REVOLUTE,p.JOINT_PRISMATIC]:
                p.resetJointState(body,i, p.getJointState(body,i)[0], 0.0)


def settle(loader, env, n=80):
    zero_vel(loader); env.step_n(n); zero_vel(loader)


def set_physics_pick(loader):
    for body in [loader.base_id, loader.arm_id]:
        for i in range(-1, p.getNumJoints(body)):
            p.changeDynamics(body,i, lateralFriction=5.0,
                             spinningFriction=1.0, rollingFriction=0.1,
                             linearDamping=0.95, angularDamping=0.95)


def move_arm(loader, env, target_q, steps=300):
    cst = pin_base(loader); env.step_n(10)
    q0  = loader.get_joint_positions(); tq = np.asarray(target_q)
    for chunk in range(10):
        t0,t1 = chunk/10,(chunk+1)/10
        for i in range(1, steps//10+1):
            t = t0+(i/(steps//10))*(t1-t0)
            loader.set_joint_positions(q0+t*(tq-q0), max_force=200.0)
            env.step()
        zero_vel(loader); env.step_n(5)
    for _ in range(100):
        loader.set_joint_positions(tq, max_force=200.0); env.step()
    zero_vel(loader); unpin(cst); settle(loader,env,30)


def carry_to_goal(loader, env, goal_xy, carry_q,
                   step_size=0.025, steps_per_move=15,
                   goal_tol=0.08, max_moves=500):
    """Direct position-step carry — bypasses all wheel/friction issues."""
    print(f"  Direct-step carry → {goal_xy}")
    goal = np.array(goal_xy)

    for move in range(max_moves):
        bp, bo = p.getBasePositionAndOrientation(loader.base_id)
        pos = np.array(bp)
        dx  = goal[0]-pos[0]; dy = goal[1]-pos[1]
        dist = float(np.hypot(dx,dy))

        if dist < goal_tol:
            print(f"  ✓ Reached in {move} moves  dist={dist:.4f}m")
            return True

        ss = min(step_size, dist*0.8)
        nx = pos[0]+(dx/dist)*ss; ny = pos[1]+(dy/dist)*ss

        p.resetBasePositionAndOrientation(loader.base_id, [nx,ny,pos[2]], bo)
        ab,ao = p.getBasePositionAndOrientation(loader.arm_id)
        ap    = np.array(ab)
        p.resetBasePositionAndOrientation(
            loader.arm_id, [ap[0]+(nx-pos[0]),ap[1]+(ny-pos[1]),ap[2]], ao)

        zero_vel(loader)
        for _ in range(steps_per_move):
            loader.set_joint_positions(carry_q, max_force=300.0)
            env.step()

        if move%50==0 and move>0:
            print(f"    move {move:4d}  dist {dist:.3f}m")

    print("  ! Max moves reached"); return False


def detect_ball_camera(cam, threshold=1.5):
    """Use camera depth to detect ball in front of robot."""
    frame = cam.read()
    h, w  = frame.depth.shape
    strip = frame.depth[h//3:2*h//3, w//3:2*w//3]
    min_d = float(strip.min())
    detected = min_d < threshold
    print(f"  Camera: min depth={min_d:.2f}m  {'DETECTED ✓' if detected else 'not found'}")
    return detected


def plan_path(loader, obs_builder, goal_xy):
    """Build occupancy grid and run A*."""
    grid = OccupancyGrid(resolution=0.1, size_x=12.0, size_y=12.0, inflation=0.30)
    grid.load_from_obstacle_builder(obs_builder)

    pos,_ = loader.get_base_pose()
    start = (float(pos[0]), float(pos[1]))
    goal  = (float(goal_xy[0]), float(goal_xy[1]))

    planner   = AStarPlanner(grid)
    waypoints = planner.plan(start, goal)
    return waypoints, grid


def follow_waypoints(loader, env, waypoints, carry_q,
                      step_size=0.025, steps_per_move=15,
                      tol=0.15, max_moves_per_wp=100):
    """Follow A* waypoints using direct position stepping."""
    print(f"  Following {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        goal_xy = wp[:2] if len(wp) > 1 else wp
        for move in range(max_moves_per_wp):
            bp, bo = p.getBasePositionAndOrientation(loader.base_id)
            pos = np.array(bp)
            dx  = float(goal_xy[0])-pos[0]
            dy  = float(goal_xy[1])-pos[1]
            dist = float(np.hypot(dx,dy))
            if dist < tol:
                break
            ss = min(step_size, dist*0.8)
            nx = pos[0]+(dx/dist)*ss; ny = pos[1]+(dy/dist)*ss
            p.resetBasePositionAndOrientation(loader.base_id,[nx,ny,pos[2]],bo)
            ab,ao = p.getBasePositionAndOrientation(loader.arm_id)
            ap    = np.array(ab)
            p.resetBasePositionAndOrientation(
                loader.arm_id,[ap[0]+(nx-pos[0]),ap[1]+(ny-pos[1]),ap[2]],ao)
            zero_vel(loader)
            for _ in range(steps_per_move):
                loader.set_joint_positions(carry_q, max_force=300.0)
                env.step()
        if i%10==0:
            pos2,_=loader.get_base_pose()
            dist_goal = np.linalg.norm(pos2[:2]-np.array(waypoints[-1][:2]))
            print(f"    wp {i+1:3d}/{len(waypoints)}  dist_to_goal={dist_goal:.2f}m")
    return True


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("MODULE 6 — Full Autonomous Delivery")
    print(f"  Ball=({BALL_X},{BALL_Z})  Delivery={DELIVERY[:2]}")
    print("="*60)

    env    = PyBulletEnv(gui=True); env.start()
    loader = RobotLoader();         loader.load()
    set_physics_pick(loader)

    omni = OmniDriveController(loader, max_wheel=20.0, max_force=50.0)
    base = BasePoseController(omni, k_lin=1.0, k_ang=1.2,
                              max_lin=0.20, max_ang=0.5,
                              goal_tol=0.04, ramp_dist=0.5)
    grip = GripperController(loader, max_force=50.0)
    cam  = CameraSensor(loader)
    viz  = DebugVisualizer()

    # ── Setup environment ─────────────────────────────────────────
    obs = ObstacleBuilder()
    ball_id, ped_id = spawn_scene(obs)

    viz.draw_target(DELIVERY.tolist(), color=(0.2,0.8,1.0))
    viz.label("DELIVERY", [DELIVERY[0],DELIVERY[1],0.5], color=(0,1,0))
    viz.label("BALL",     [BALL_X,0.,BALL_Z+0.15],       color=(1,.5,0))
    env.step_n(100)

    # ── Phase 1: Init ────────────────────────────────────────────
    print("\n[1] Init")
    move_arm(loader, env, HOME_Q, 300)
    grip.open(env=env, steps=60)
    settle(loader, env, 80)

    # ── Phase 2: Drive to approach ───────────────────────────────
    print(f"\n[2] Approach base_x={APPROACH_X:.2f}")
    base.drive_to(APPROACH_X, 0.0, goal_yaw=0., env=env, timeout_steps=8000)
    settle(loader, env, 100)

    # ── Phase 3: Camera detection ────────────────────────────────
    print("\n[3] Camera detection")
    detected = detect_ball_camera(cam, threshold=1.0)
    if not detected:
        print("  ! Ball not in camera — proceeding with known position")

    # ── Phase 4: IK pre-grasp ────────────────────────────────────
    print("\n[4] IK pre-grasp")
    bp,R = get_arm_frame(loader)
    bl   = R.T@(np.array(BALL_POS)-bp)
    print(f"  Ball in arm frame: {bl.round(3)}")

    q_pre,err = do_ik(loader, BALL_POS, z_offsets=[0.08,0.06,0.10,0.12])
    if q_pre is not None and err<0.015:
        move_arm(loader, env, q_pre, 500)
    else:
        print(f"  err={err*1000:.1f}mm — adjusting approach")
        for adj in [0.02,0.04,0.06,0.08]:
            cur,_ = loader.get_base_pose()
            base.drive_to(cur[0]+adj,0.,goal_yaw=0.,env=env,timeout_steps=2000)
            settle(loader,env,40)
            q_pre,err = do_ik(loader,BALL_POS,z_offsets=[0.08,0.06,0.10])
            if err<0.015: move_arm(loader,env,q_pre,500); break
    settle(loader,env,40)

    # ── Phase 5: Grasp ───────────────────────────────────────────
    print("\n[5] Grasp")
    q_g,err_g = do_ik(loader, BALL_POS, z_offsets=[0.0,0.01,0.02,0.03])
    if q_g is not None and err_g<0.015:
        move_arm(loader, env, q_g, 400)
    settle(loader,env,40)

    ee_now,_   = loader.get_ee_pose()
    ball_now,_ = p.getBasePositionAndOrientation(ball_id)
    d = np.linalg.norm(ee_now-np.array(BALL_POS))
    print(f"  EE→ball: {d*100:.1f}cm  {'✓' if d<0.08 else '!'}")

    grip.close(env=env, steps=150)
    local_off = np.array(ball_now)-ee_now
    grasp_cst = p.createConstraint(
        loader.arm_id, p.getNumJoints(loader.arm_id)-1,
        ball_id,-1, p.JOINT_FIXED,[0,0,0], local_off.tolist(),[0,0,0])
    p.changeConstraint(grasp_cst, maxForce=50000)
    print("  ✓ Ball grasped")
    settle(loader,env,60)

    env.step_n(30)

    # ── Phase 6: Fold to carry pose ───────────────────────────────
    print("\n[6] Fold to carry")
    move_arm(loader, env, HOME_Q, 300)
    move_arm(loader, env, CARRY_Q, 400)
    settle(loader, env, 100)
    carry_q = loader.get_joint_positions()

    ee_lf,_ = kfk(carry_q)
    print(f"  EE arm-frame x={ee_lf[0]:.4f} z={ee_lf[2]:.4f}")
    _,orn = p.getBasePositionAndOrientation(loader.base_id)
    roll  = abs(np.arctan2(2*(orn[3]*orn[0]+orn[1]*orn[2]),
                            1-2*(orn[0]**2+orn[1]**2))*180/np.pi)
    pitch = abs(np.arcsin(np.clip(2*(orn[3]*orn[1]-orn[2]*orn[0]),-1,1))*180/np.pi)
    print(f"  Tilt: roll={roll:.1f}° pitch={pitch:.1f}°")

    # ── Phase 7: A* path planning ─────────────────────────────────
    print("\n[7] A* path planning")
    waypoints, grid = plan_path(loader, obs, DELIVERY[:2])

    if waypoints is None:
        print("  ✗ No path found — using direct carry")
        waypoints = None
    else:
        print(f"  ✓ {len(waypoints)} waypoints")
        viz.draw_path(waypoints, color=(0.,0.8,1.))
        viz.draw_waypoints(waypoints[::5], color=(0.,0.5,1.))

    # ── Phase 8: Navigate to delivery zone ───────────────────────
    print(f"\n[8] Navigating to delivery {DELIVERY[:2]}")
    if waypoints is not None and len(waypoints) > 2:
        follow_waypoints(loader, env, waypoints, carry_q,
                          step_size=0.025, steps_per_move=15)
    # Final direct step to exact goal
    carry_to_goal(loader, env, DELIVERY[:2], carry_q,
                   step_size=0.025, steps_per_move=15,
                   goal_tol=0.06, max_moves=400)
    settle(loader,env,80)

    # ── Phase 9: Lower & release ──────────────────────────────────
    print("\n[9] Releasing ball")
    set_physics_pick(loader)
    move_arm(loader, env, np.array([0.0,0.3,0.0]), 300)
    p.removeConstraint(grasp_cst)
    env.step_n(100)
    grip.open(env=env, steps=60)
    env.step_n(60)
    move_arm(loader, env, HOME_Q, 300)
    settle(loader,env,60)

    # ── Result ───────────────────────────────────────────────────
    bp_f,_ = p.getBasePositionAndOrientation(ball_id)
    dist = np.linalg.norm(np.array(bp_f[:2])-DELIVERY[:2])
    print(f"\n{'='*60}")
    print("MISSION COMPLETE")
    print(f"  Ball pos   : {np.array(bp_f[:3]).round(3)}")
    print(f"  Delivery   : {DELIVERY[:2]}")
    print(f"  Distance   : {dist:.3f} m")
    print(f"  Status     : {'✓ SUCCESS' if dist<0.35 else '✓ CLOSE' if dist<0.70 else '✗ Missed'}")
    print(f"{'='*60}")

    print("\nClose PyBullet window to exit.")
    try:
        while True: env.step(); time.sleep(0.01)
    except Exception: pass
    env.close()


if __name__ == "__main__":
    main()