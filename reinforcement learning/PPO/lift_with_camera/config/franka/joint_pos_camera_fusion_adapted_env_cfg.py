# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.lift_camera_fusion_adapted_env_cfg import (
    CameraFusionObservationsCfg,
    CameraFusionRewardsCfg,
    LiftCameraSceneCfg,
)

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class FrankaCubeLiftCameraFusionAdaptedEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka camera-fusion lift config using adapted observation/reward copies."""

    scene: LiftCameraSceneCfg = LiftCameraSceneCfg(num_envs=32, env_spacing=5.0)
    observations: CameraFusionObservationsCfg = CameraFusionObservationsCfg()
    rewards: CameraFusionRewardsCfg = CameraFusionRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.gripper_camera.height = 128
        self.scene.gripper_camera.width = 128
        self.curriculum = None
        self.commands.object_pose.ranges.pos_x = (0.45, 0.55)
        self.commands.object_pose.ranges.pos_y = (-0.12, 0.12)
        self.commands.object_pose.ranges.pos_z = (0.25, 0.40)
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.12, 0.12),
            "z": (0.0, 0.0),
        }
