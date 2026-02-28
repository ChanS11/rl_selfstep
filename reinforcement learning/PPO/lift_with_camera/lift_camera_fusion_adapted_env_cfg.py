# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from . import mdp_camera_fusion as mdp_cf
from .lift_camera_env_cfg import LiftCameraEnvCfg, LiftCameraSceneCfg


@configclass
class CameraFusionObservationsCfg:
    """Camera-fusion observation groups for lift task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy group with visual + state observations."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp_cf.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        image = ObsTerm(
            func=mdp_cf.gated_image_after_lift,
            params={
                "sensor_cfg": SceneEntityCfg("gripper_camera"),
                "data_type": "distance_to_image_plane",
                "minimal_height": 0.04,
                "value_pre_lift": 1.0,
                "value_post_lift": 0.1,
                "transition_scale": 0.01,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: ObsGroup = PolicyCfg()


@configclass
class CameraFusionRewardsCfg:
    """Camera-fusion specific reward shaping."""

    # Slightly reduce pre-grasp dominance so policy does not over-optimize reach behavior after lift.
    reaching_object = RewTerm(func=mdp_cf.object_ee_distance, params={"std": 0.12}, weight=1.5)
    lifting_object = RewTerm(func=mdp_cf.object_is_lifted, params={"minimal_height": 0.04}, weight=20.0)
    object_goal_tracking = RewTerm(
        func=mdp_cf.object_goal_distance,
        params={"std": 0.35, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=10.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp_cf.object_goal_distance,
        params={"std": 0.08, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=3.0,
    )
    # Increase smoothness penalties to suppress post-lift high-frequency action oscillation.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-5)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-3e-5, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class CameraFusionTerminationsCfg:
    """Camera-fusion specific terminations."""

    time_out = DoneTerm(func=mdp_cf.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp_cf.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class LiftCameraFusionAdaptedEnvCfg(LiftCameraEnvCfg):
    """Adapted camera-fusion lift env with separate MDP copy and conservative shaping."""

    scene: LiftCameraSceneCfg = LiftCameraSceneCfg(num_envs=32, env_spacing=5.0)
    observations: CameraFusionObservationsCfg = CameraFusionObservationsCfg()
    rewards: CameraFusionRewardsCfg = CameraFusionRewardsCfg()
    terminations: CameraFusionTerminationsCfg = CameraFusionTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Disable default curriculum from LiftEnvCfg for this adapted variant.
        self.curriculum = None
        # Narrow command distribution for initial visual learning stability.
        self.commands.object_pose.ranges.pos_x = (0.45, 0.55)
        self.commands.object_pose.ranges.pos_y = (-0.12, 0.12)
        self.commands.object_pose.ranges.pos_z = (0.25, 0.40)
        # Narrow reset distribution so the agent sees easier grasp layouts first.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.12, 0.12),
            "z": (0.0, 0.0),
        }
