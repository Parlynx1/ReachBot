#!/bin/bash

echo "Generating gripper finger mesh..."

mkdir -p "/Users/pardhumattupalli/YU/Introduction to Robotics/Final Project/reach_mobility_bot/assets/meshes"

python3 - << 'EOF'
import os

# ── output path ─────────────────────────────────────────────
output_path = "/Users/pardhumattupalli/YU/Introduction to Robotics/Final Project/reach_mobility_bot/assets/meshes/gripper_finger.obj"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ── finger generator ────────────────────────────────────────
def write_finger_obj(path):
    # base and tip (tapered)
    bx, by = 0.0075, 0.0075
    tx, ty = 0.004, 0.004
    h = 0.05

    verts = [
        # base (z=0)
        (-bx, -by, 0.),
        ( bx, -by, 0.),
        ( bx,  by, 0.),
        (-bx,  by, 0.),

        # tip (z=h)
        (-tx, -ty, h),
        ( tx, -ty, h),
        ( tx,  ty, h),
        (-tx,  ty, h),
    ]

    faces = [
        (1,2,3,4),   # bottom
        (5,6,7,8),   # top
        (1,2,6,5),   # front
        (2,3,7,6),   # right
        (3,4,8,7),   # back
        (4,1,5,8),   # left
    ]

    with open(path, 'w') as f:
        f.write('# gripper_finger\n')
        f.write('# tapered finger 0.015 x 0.015 x 0.05 m\n\n')
        f.write('o gripper_finger\n\n')

        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

        f.write('\n')

        for face in faces:
            f.write('f ' + ' '.join(str(i) for i in face) + '\n')

    print(f'Written {path}')

write_finger_obj(output_path)
EOF

echo "gripper_finger.obj generated!"
