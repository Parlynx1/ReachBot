#!/bin/bash

echo "Generating S-curve map..."

python3 - << 'EOF'
import numpy as np
from PIL import Image, ImageDraw

W, H = 512, 512
img  = Image.new('RGB', (W,H), (255,255,255))
draw = ImageDraw.Draw(img)

# Draw a smooth S-curve path as thick black line
pts = []
for t in np.linspace(0, 1, 300):
    x = int(50 + t*(W-100))
    y = int(H/2 + 150*np.sin(2*np.pi*t))
    pts.append((x,y))

for i in range(len(pts)-1):
    draw.line([pts[i], pts[i+1]], fill=(0,0,0), width=8)

# Mark start/end
draw.ellipse([pts[0][0]-10, pts[0][1]-10, pts[0][0]+10, pts[0][1]+10],
             fill=(0,200,0))
draw.ellipse([pts[-1][0]-10, pts[-1][1]-10, pts[-1][0]+10, pts[-1][1]+10],
             fill=(200,0,0))

img.save('line_path_01.png')
print('line_path_01.png created')
EOF

echo "Done!"