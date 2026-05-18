#!/bin/bash

echo "Generating omni wheel mesh..."

mkdir -p "/Users/pardhumattupalli/YU/Introduction to Robotics/Final Project/reach_mobility_bot/assets/meshes"

python3 - << 'EOF'
import numpy as np
import math
import os

# ── output path ─────────────────────────────────────────────
output_path = "/Users/pardhumattupalli/YU/Introduction to Robotics/Final Project/reach_mobility_bot/assets/meshes/omni_wheel.obj"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ── cylinder generator ──────────────────────────────────────
def write_cylinder_obj(path, radius, length, n_sides, label):
    half = length / 2
    verts = []

    # centres
    verts.append((0, -half, 0))  # bottom centre
    verts.append((0,  half, 0))  # top centre

    # bottom ring
    for side in range(n_sides):
        angle = 2 * math.pi * side / n_sides
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        verts.append((x, -half, z))

    # top ring
    for side in range(n_sides):
        angle = 2 * math.pi * side / n_sides
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        verts.append((x, half, z))

    with open(path, 'w') as f:
        f.write(f'# {label}\n')
        f.write(f'# omni wheel cylinder r={radius} l={length} sides={n_sides}\n\n')
        f.write(f'o {label}\n\n')

        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

        f.write('\n')

        # bottom cap
        for i in range(n_sides):
            a = 3 + i
            b = 3 + (i + 1) % n_sides
            f.write(f'f 1 {b} {a}\n')

        # top cap
        for i in range(n_sides):
            a = 3 + n_sides + i
            b = 3 + n_sides + (i + 1) % n_sides
            f.write(f'f 2 {a} {b}\n')

        # side faces
        for i in range(n_sides):
            bl = 3 + i
            br = 3 + (i + 1) % n_sides
            tl = 3 + n_sides + i
            tr = 3 + n_sides + (i + 1) % n_sides
            f.write(f'f {bl} {br} {tr} {tl}\n')

    print(f'Written {path}')

write_cylinder_obj(output_path, radius=0.06, length=0.03, n_sides=20, label='omni_wheel')
EOF

echo "omni_wheel.obj generated!"
