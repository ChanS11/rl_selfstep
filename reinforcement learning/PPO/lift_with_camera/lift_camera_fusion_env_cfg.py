# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .lift_camera_env_cfg import LiftCameraEnvCfg, LiftCameraSceneCfg


@configclass
class DepthProprioObservationsCfg:
    """Depth image + proprioception observations for camera-based lift."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Task-level low-dimensional observations (kept as a flat tensor)."""

        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        """Robot proprioception observations."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PerceptionCfg(ObsGroup):
        """Depth camera observations."""

        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("gripper_camera"), "data_type": "distance_to_image_plane"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            # Single image term remains a tensor; keep it as a separate observation group.
            self.concatenate_terms = True

    policy: ObsGroup = PolicyCfg()
    proprio: ObsGroup = ProprioCfg()
    perception: ObsGroup = PerceptionCfg()


@configclass
class LiftCameraFusionEnvCfg(LiftCameraEnvCfg):
    """Lift environment with wrist depth camera and fused visual-proprio observations."""

    scene: LiftCameraSceneCfg = LiftCameraSceneCfg(num_envs=64, env_spacing=5.0)
    observations: DepthProprioObservationsCfg = DepthProprioObservationsCfg()
