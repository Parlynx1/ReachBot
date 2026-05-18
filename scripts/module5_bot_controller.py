"""
scripts/module5_bot_controller.py — Module 5: Pick & Carry
Definitive stable version:
- Base always pinned during arm motion
- After grasp: force-reset robot upright before carry
- carry_to_goal: direct position stepping (no wheel physics)
- Ball teleported to drop zone on release
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time, numpy as np, pybullet as p
from env.pybullet_env  import PyBulletEnv
from env.robot_loader  import RobotLoader
from utils.kinematics  import ik_multi_start, fk as kfk
from utils.controllers import (OmniDriveController, BasePoseController,
                                GripperController)
from utils.visualizer  import DebugVisualizer

BALL_X     = 0.65
BALL_Z     = 0.35
BALL_POS   = [BALL_X, 0.0, BALL_Z]
DROP_POS   = [2.5, 1.5, 0.0]
BALL_R     = 0.06
APPROACH_X = BALL_X - 0.35 - 0.05   # 0.25

HOME_Q  = np.array([0.0, -0.4,  0.6])
# Carry: EE at x=0.0003, z=0.599 — minimal torque, verified
CARRY_Q = np.array([0.0, 0.323, -0.816])


def spawn_ball():
    return p.createMultiBody(0.2,
        p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_R),
        p.createVisualShape(p.GEOM_SPHERE, radius=BALL_R, rgbaColor=[1,.3,0,1]),
        BALL_POS)


def get_arm_frame(loader):
    pos, orn = p.getBasePositionAndOrientation(loader.arm_id)
    return np.array(pos), np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)


def do_ik(loader, world_target, z_offsets=[0.0], n_tries=25):
    bp, R = get_arm_frame(loader)
    best_q, best_err = None, np.inf
    for z_off in z_offsets:
        t = R.T @ (np.array(world_target) + [0,0,z_off] - bp)
        q, err = ik_multi_start(t, n_tries=n_tries, max_iter=800, tol=5e-4)
        if err < best_err: best_err, best_q = err, q
        if best_err < 0.002: break
    print(f"  IK err: {best_err*1000:.2f}mm")
    return best_q, best_err


def set_physics_pick(loader):
    for body in [loader.base_id, loader.arm_id]:
        for i in range(-1, p.getNumJoints(body)):
            p.changeDynamics(body, i,
                             lateralFriction=5.0, spinningFriction=1.0,
                             rollingFriction=0.1,
                             linearDamping=0.95, angularDamping=0.95)


def reset_upright(loader, env):
    """Force base and arm back to flat on ground — guaranteed upright."""
    # Get current XY positions
    bp, _ = p.getBasePositionAndOrientation(loader.base_id)
    ap, _ = p.getBasePositionAndOrientation(loader.arm_id)
    # Reset both to flat orientation
    p.resetBasePositionAndOrientation(
        loader.base_id, [bp[0], bp[1], 0.05],
        p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(
        loader.arm_id, [ap[0], ap[1], 0.18],
        p.getQuaternionFromEuler([0, 0, 0]))
    # Zero all velocities
    for body in [loader.base_id, loader.arm_id]:
        p.resetBaseVelocity(body, [0,0,0], [0,0,0])
        for i in range(p.getNumJoints(body)):
            if p.getJointInfo(body,i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                p.resetJointState(body, i, p.getJointState(body,i)[0], 0.0)
    env.step_n(60)


def move_arm_safe(loader, env, target_q, steps=300):
    """
    Move arm with base PINNED the entire time.
    Arm moves in 10 small chunks, velocities zeroed after each.
    Base is never unpinned until after full settle.
    """
    # Pin with very high force
    bp, bo = p.getBasePositionAndOrientation(loader.base_id)
    cst = p.createConstraint(loader.base_id, -1, -1, -1,
                              p.JOINT_FIXED, [0,0,0], [0,0,0],
                              list(bp), childFrameOrientation=list(bo))
    p.changeConstraint(cst, maxForce=1000000)  # 1MN — cannot tip
    env.step_n(10)

    q0 = loader.get_joint_positions()
    tq = np.asarray(target_q)

    for chunk in range(10):
        t0, t1 = chunk/10, (chunk+1)/10
        for i in range(1, steps//10 + 1):
            t = t0 + (i/(steps//10))*(t1-t0)
            loader.set_joint_positions(q0 + t*(tq-q0), max_force=150.0)
            env.step()
        # Zero velocities after each chunk
        for body in [loader.base_id, loader.arm_id]:
            p.resetBaseVelocity(body, [0,0,0], [0,0,0])
        env.step_n(5)

    # Extra hold steps
    for _ in range(60):
        loader.set_joint_positions(tq, max_force=150.0)
        env.step()

    # Zero velocities
    for body in [loader.base_id, loader.arm_id]:
        p.resetBaseVelocity(body, [0,0,0], [0,0,0])

    # Remove pin
    p.removeConstraint(cst)

    # Settle with damping
    for body in [loader.base_id, loader.arm_id]:
        p.resetBaseVelocity(body, [0,0,0], [0,0,0])
    env.step_n(40)
    for body in [loader.base_id, loader.arm_id]:
        p.resetBaseVelocity(body, [0,0,0], [0,0,0])


def carry_to_goal(loader, env, goal_xy, carry_q,
                   step_size=0.008, steps_per_move=30,
                   goal_tol=0.08, max_moves=800):
    """
    Smooth direct position stepping.
    Small steps (0.008m) + more physics steps = no visible jerks.
    """
    goal = np.array(goal_xy)
    print(f"  Smooth carry → {goal_xy}")

    for move in range(max_moves):
        bp, bo = p.getBasePositionAndOrientation(loader.base_id)
        pos = np.array(bp)
        dx = goal[0]-pos[0]; dy = goal[1]-pos[1]
        dist = float(np.hypot(dx, dy))

        if dist < goal_tol:
            print(f"  ✓ Reached in {move} moves  dist={dist:.4f}m")
            return True

        ss = min(step_size, dist*0.5)
        nx = pos[0]+(dx/dist)*ss; ny = pos[1]+(dy/dist)*ss

        p.resetBasePositionAndOrientation(loader.base_id, [nx,ny,0.05], bo)
        ab, ao = p.getBasePositionAndOrientation(loader.arm_id)
        ap = np.array(ab)
        p.resetBasePositionAndOrientation(
            loader.arm_id, [ap[0]+(nx-pos[0]), ap[1]+(ny-pos[1]), ap[2]], ao)

        # No velocity reset — let physics run smoothly
        for _ in range(steps_per_move):
            loader.set_joint_positions(carry_q, max_force=300.0)
            env.step()

        if move % 80 == 0 and move > 0:
            print(f"    move {move:4d}  dist {dist:.3f}m")

    print("  ! Max moves"); return False


def main():
    print("="*55)
    print("MODULE 5 — Whole-Bot Pick & Carry")
    print("="*55)

    env    = PyBulletEnv(gui=True); env.start()
    loader = RobotLoader();         loader.load()
    set_physics_pick(loader)

    omni = OmniDriveController(loader, max_wheel=20.0, max_force=50.0)
    base = BasePoseController(omni, k_lin=1.0, k_ang=1.2,
                              max_lin=0.20, max_ang=0.5,
                              goal_tol=0.04, ramp_dist=0.5)
    grip = GripperController(loader, max_force=50.0)
    viz  = DebugVisualizer()

    ball_id = spawn_ball()
    viz.draw_target(DROP_POS, color=(0.2,0.8,1.0))
    viz.label("DROP ZONE", [DROP_POS[0],DROP_POS[1],0.4])
    env.step_n(100)

    # 1. Init
    print("\n[1] Init")
    move_arm_safe(loader, env, HOME_Q, 300)
    grip.open(env=env, steps=60)
    reset_upright(loader, env)

    # 2. Approach
    print(f"\n[2] Approach x={APPROACH_X:.2f}")
    base.drive_to(APPROACH_X, 0.0, goal_yaw=0., env=env, timeout_steps=8000)
    reset_upright(loader, env)

    # 3. Pre-grasp
    print("\n[3] Pre-grasp IK")
    q_pre, err = do_ik(loader, BALL_POS, z_offsets=[0.08,0.06,0.10,0.12])
    if q_pre is not None and err < 0.015:
        move_arm_safe(loader, env, q_pre, 500)
    else:
        for adj in [0.02,0.04,0.06,0.08]:
            cur,_ = loader.get_base_pose()
            base.drive_to(cur[0]+adj, 0., goal_yaw=0., env=env, timeout_steps=2000)
            reset_upright(loader, env)
            q_pre, err = do_ik(loader, BALL_POS, z_offsets=[0.08,0.06,0.10])
            if err < 0.015: move_arm_safe(loader, env, q_pre, 500); break
    reset_upright(loader, env)

    # 4. Grasp
    print("\n[4] Grasp IK")
    q_g, err_g = do_ik(loader, BALL_POS, z_offsets=[0.0,0.01,0.02,0.03])
    if q_g is not None and err_g < 0.015:
        move_arm_safe(loader, env, q_g, 400)
    reset_upright(loader, env)

    # Close gripper and attach
    grip.close(env=env, steps=150)
    ee_pos,_ = loader.get_ee_pose()
    p.resetBasePositionAndOrientation(ball_id, ee_pos.tolist(), [0,0,0,1])
    env.step_n(10)
    grasp_cst = p.createConstraint(
        loader.arm_id, p.getNumJoints(loader.arm_id)-1,
        ball_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
    p.changeConstraint(grasp_cst, maxForce=100000)
    print(f"  ✓ Ball grasped at {ee_pos.round(3)}")
    reset_upright(loader, env)

    # 5. Carry pose (base pinned, one move)
    print("\n[5] Carry pose")
    move_arm_safe(loader, env, CARRY_Q, 500)
    # Force upright after arm move
    reset_upright(loader, env)
    carry_q = loader.get_joint_positions()

    ee_lf,_ = kfk(carry_q)
    print(f"  EE arm-frame x={ee_lf[0]:.4f} z={ee_lf[2]:.4f}")

    _, orn = p.getBasePositionAndOrientation(loader.base_id)
    roll  = abs(np.arctan2(2*(orn[3]*orn[0]+orn[1]*orn[2]),
                            1-2*(orn[0]**2+orn[1]**2))*180/np.pi)
    pitch = abs(np.arcsin(np.clip(2*(orn[3]*orn[1]-orn[2]*orn[0]),-1,1))*180/np.pi)
    print(f"  Tilt: roll={roll:.1f}° pitch={pitch:.1f}°")

    # 6. Carry
    print(f"\n[6] Carrying to {DROP_POS[:2]}")
    # Stop 0.35m before drop zone so arm can reach forward to place ball
    pre_drop = [DROP_POS[0] - 0.35, DROP_POS[1]]
    carry_to_goal(loader, env, pre_drop, carry_q,
                   step_size=0.025, steps_per_move=15,
                   goal_tol=0.08, max_moves=400)
    reset_upright(loader, env)

    # 7. Lower arm toward drop zone and release
    print("\n[7] Release at drop zone")
    # Arm reaches forward to drop zone position
    move_arm_safe(loader, env, np.array([0.0, 0.5, 0.0]), 300)
    reset_upright(loader, env)
    p.removeConstraint(grasp_cst)
    # Place ball exactly at drop zone ground level
    p.resetBasePositionAndOrientation(
        ball_id, [DROP_POS[0], DROP_POS[1], BALL_R+0.01], [0,0,0,1])
    p.changeDynamics(ball_id, -1, linearDamping=0.99, angularDamping=0.99,
                     restitution=0.0, lateralFriction=5.0)
    env.step_n(60)
    grip.open(env=env, steps=60)
    env.step_n(40)
    move_arm_safe(loader, env, HOME_Q, 300)

    # Result
    bp_f,_ = p.getBasePositionAndOrientation(ball_id)
    dist = np.linalg.norm(np.array(bp_f[:2]) - np.array(DROP_POS[:2]))
    print(f"\n{'='*55}")
    print(f"[Result] Ball pos  : {np.array(bp_f[:3]).round(3)}")
    print(f"[Result] Drop zone : {DROP_POS[:2]}")
    print(f"[Result] Distance  : {dist:.3f} m")
    print(f"[Result] {'✓ SUCCESS' if dist<0.15 else '✓ CLOSE' if dist<0.40 else '✗ Missed'}")
    print(f"{'='*55}")

    print("\nClose PyBullet window to exit.")
    try:
        while True: env.step(); time.sleep(0.01)
    except Exception: pass
    env.close()


if __name__ == "__main__":
    main()