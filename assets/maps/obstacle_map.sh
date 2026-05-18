#!/bin/bash

echo "Generating obstacle map..."

OUTPUT_DIR="/Users/pardhumattupalli/YU/Introduction to Robotics/Final Project/reach_mobility_bot/assets/maps"
mkdir -p "$OUTPUT_DIR"

python3 - << 'EOF'
import numpy as np
from PIL import Image, ImageDraw

import os

W, H = 512, 512
img  = Image.new('RGB', (W,H), (255,255,255))
draw = ImageDraw.Draw(img)

# Draw obstacles as gray rectangles in arena
obstacles = [
    (330,180,370,420),
    (80, 280,120,512),
    (50, 50, 290,90),
    (150,330,250,370),
    (380,50, 420,220),
]

for box in obstacles:
    draw.rectangle(box, fill=(80,80,80))

# Start and goal markers
draw.ellipse([20,220,60,260], fill=(0,180,0))
draw.ellipse([440,220,480,260], fill=(180,0,0))

output_path = "/Users/pardhumattupalli/YU/Introduction to Robotics/Final Project/reach_mobility_bot/assets/maps/obstacle_map_01.png"
img.save(output_path)

print("obstacle_map_01.png created at", output_path)
EOF

echo "Obstacle map generation complete!"