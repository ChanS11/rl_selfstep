# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.lift_camera_fusion_env_cfg import (
    DepthProprioObservationsCfg,
    LiftCameraSceneCfg,
)

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class FrankaCubeLiftCameraEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka lift with wrist depth camera and visual-proprio observations."""

    scene: LiftCameraSceneCfg = LiftCameraSceneCfg(num_envs=64, env_spacing=5.0)
    observations: DepthProprioObservationsCfg = DepthProprioObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep camera input lightweight for faster visual policy learning.
        self.scene.gripper_camera.height = 128
        self.scene.gripper_camera.width = 128
        # Disable the default lift reward curriculum for camera-fusion training.
        # The original curriculum ramps action penalties from -1e-4 to -1e-1,
        # which can dominate rewards before visual grasping stabilizes.
        self.curriculum = None
        # Start from clean observations; add corruption only after a stable grasp policy appears.
        self.observations.policy.enable_corruption = False
        self.observations.proprio.enable_corruption = False
        self.observations.perception.enable_corruption = False
