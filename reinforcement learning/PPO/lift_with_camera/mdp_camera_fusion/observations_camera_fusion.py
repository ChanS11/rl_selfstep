# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .. import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position in robot-root frame.

    Kept as a standalone copy for camera-fusion experiments.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def gated_image_after_lift(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_camera"),
    data_type: str = "distance_to_image_plane",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    minimal_height: float = 0.04,
    value_pre_lift: float = 1.0,
    value_post_lift: float = 0.1,
    transition_scale: float = 0.01,
) -> torch.Tensor:
    """Return camera image with smooth post-lift visual gating.

    This keeps observation shape unchanged (checkpoint-compatible) while reducing
    visual branch influence after successful lift to suppress high-frequency jitter.
    """
    image = mdp.image(env=env, sensor_cfg=sensor_cfg, data_type=data_type)
    object: RigidObject = env.scene[object_cfg.name]

    # Smooth gate: before lift -> ~value_pre_lift, after lift -> ~value_post_lift
    height = object.data.root_pos_w[:, 2]
    scaled = (minimal_height - height) / transition_scale
    gate = value_post_lift + (value_pre_lift - value_post_lift) * torch.sigmoid(scaled)

    # Broadcast [N] gate to image tensor shape [N, ...]
    while gate.dim() < image.dim():
        gate = gate.unsqueeze(-1)

    return image * gate
