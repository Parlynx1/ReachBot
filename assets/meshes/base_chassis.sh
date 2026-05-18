pip install numpy --quiet 2>/dev/null

python3 - << 'EOF'
import numpy as np
import os

# ── output path  ─────────────────────────────
output_path = "/Users/pardhumattupalli/YU/Introduction to Robotics/Final Project/reach_mobility_bot/assets/meshes/base_chassis.obj"

# ensure folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ── geometry ─────────────────────────────────────────────────────────
def write_box_obj(path, lx, ly, lz, label):
    hx, hy, hz = lx/2, ly/2, lz/2

    verts = [
        (-hx,-hy,-hz),(hx,-hy,-hz),(hx,hy,-hz),(-hx,hy,-hz),
        (-hx,-hy, hz),(hx,-hy, hz),(hx,hy, hz),(-hx,hy, hz),
    ]

    faces = [
        (1,2,3,4),(5,8,7,6),(1,5,6,2),
        (2,6,7,3),(3,7,8,4),(4,8,5,1)
    ]

    with open(path,'w') as f:
        f.write(f'# {label}\n')
        f.write('# Units: metres\n')
        f.write(f'# Dimensions: {lx} x {ly} x {lz} m\n\n')
        f.write(f'o {label}\n\n')

        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

        f.write('\n')

        for n in [(0,-1,0),(0,1,0),(-1,0,0),(1,0,0),(0,0,-1),(0,0,1)]:
            f.write(f'vn {n[0]} {n[1]} {n[2]}\n')

        f.write('\n')

        for fi, face in enumerate(faces):
            ni = fi + 1
            vstr = ' '.join(f'{vi}//{ni}' for vi in face)
            f.write(f'f {vstr}\n')

    print(f'Written {path}')

write_box_obj(output_path, 0.50, 0.40, 0.10, 'base_chassis')
EOF