"""
env/obstacle_builder.py
Spawns static box obstacles and tracks AABBs for the path planner.
"""

import pybullet as p
import numpy as np


class ObstacleBuilder:
    def __init__(self):
        self.obstacles = []

    def build_default_maze(self):
        layout = [
            ( 1.5,  0.0, 0.25, 0.20, 1.20, 0.50),
            (-1.5,  0.5, 0.25, 0.20, 1.20, 0.50),
            ( 0.0,  1.5, 0.25, 1.20, 0.20, 0.50),
            ( 0.5, -1.5, 0.25, 1.20, 0.20, 0.50),
            ( 0.8,  0.8, 0.25, 0.20, 0.20, 0.50),
        ]
        for x, y, z, lx, ly, lz in layout:
            self.add_box([x, y, z], [lx/2, ly/2, lz/2])

    def build_sparse(self, n=6, arena=3.0, seed=42):
        rng = numpy.random.default_rng(seed)
        for _ in range(n):
            x  = rng.uniform(-arena/2, arena/2)
            y  = rng.uniform(-arena/2, arena/2)
            if abs(x) < 0.5 and abs(y) < 0.5:
                continue
            lx = rng.uniform(0.15, 0.40)
            ly = rng.uniform(0.15, 0.40)
            self.add_box([x, y, 0.20], [lx/2, ly/2, 0.20])

    def add_box(self, position, half_extents, color=(0.55, 0.35, 0.20, 1.0)):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                  rgbaColor=color)
        body_id = p.createMultiBody(0, col, vis, position)
        self.obstacles.append({"id": body_id,
                               "pos": np.array(position),
                               "half": np.array(half_extents)})
        return body_id

    def get_obstacle_aabbs(self):
        aabbs = []
        for obs in self.obstacles:
            mn, mx = p.getAABB(obs["id"])
            aabbs.append((np.array(mn[:2]), np.array(mx[:2])))
        return aabbs

    def clear(self):
        for obs in self.obstacles:
            p.removeBody(obs["id"])
        self.obstacles.clear()
