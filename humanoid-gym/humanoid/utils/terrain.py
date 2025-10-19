# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
import numpy as np
from isaacgym import terrain_utils
from humanoid.envs.custom.legged_robot_config import LeggedRobotCfg

class Terrain:
    """
    Base class for terrain generation.
    Handles the creation of the heightfield, trimesh, and the data structures
    required for curriculum learning.
    """
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots, device) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.device = device
        self.type = cfg.mesh_type
        
        if self.type in ["none", 'plane']:
            return

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        
        # Initialize curriculum data structures
        # Assign a random terrain type to each environment
        self.terrain_types = torch.randint(0, self.cfg.num_cols, (self.num_robots,), device=self.device)
        
        # Pre-calculate the origins for each terrain type and difficulty level
        self.terrain_origins = torch.zeros(self.cfg.num_rows, self.cfg.num_cols, 3, device=self.device, dtype=torch.float)
        for i in range(self.cfg.num_rows):
            for j in range(self.cfg.num_cols):
                origin_x = (i + 0.5) * self.env_length
                origin_y = (j + 0.5) * self.env_width
                self.terrain_origins[i, j, 0] = origin_x
                self.terrain_origins[i, j, 1] = origin_y
                self.terrain_origins[i, j, 2] = 0.0

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        if cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold
            )

    def randomized_terrain(self):
        """Default terrain generation method. Can be overridden by child classes."""
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            terrain = terrain_utils.SubTerrain("terrain",
                                             width=self.width_per_env_pixels,
                                             length=self.width_per_env_pixels,
                                             vertical_scale=self.cfg.vertical_scale,
                                             horizontal_scale=self.cfg.horizontal_scale)
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        """Default terrain creation method. Can be overridden by child classes."""
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        """
        Adds a subtierrain to the main heightfield map.
        NOTE: This method no longer calculates env_origins.
        """
        i = row
        j = col
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

# Helper functions for terrain generation
def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


class HumanoidTerrain(Terrain):
    """
    Specific terrain implementation for the Humanoid robot.
    Inherits the curriculum-ready Terrain base class and overrides
    the terrain generation logic with its own methods.
    """
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots, device) -> None:
        # Pass the device to the parent class constructor
        super().__init__(cfg, num_robots, device)
        print("HumanoidTerrain initialized with curriculum support.")

    def randomized_terrain(self):
        """Overrides the base method to use humanoid-specific logic."""
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0, 1) # Using continuous difficulty
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        """Overrides the base method for humanoid-specific terrain types."""
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        discrete_obstacles_height = 0.005
        r_height = 0.04
        h_slope = difficulty * 0.15
        
        if choice < self.proportions[0]:
            pass # Flat terrain
        elif choice < self.proportions[1]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[2]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-r_height, max_height=r_height, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=h_slope, platform_size=0.1)
        elif choice < self.proportions[4]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=-h_slope, platform_size=0.1)
        elif choice < self.proportions[5]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=discrete_obstacles_height, platform_size=1.)
        elif choice < self.proportions[6]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=-discrete_obstacles_height, platform_size=1.)
        else:
            pass # Flat terrain as a fallback
        return terrain