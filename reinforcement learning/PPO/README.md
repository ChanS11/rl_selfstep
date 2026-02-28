# Franka Lift Camera-Fusion: Change Log

## Goal
Add camera-based perception to the Franka Lift task in IsaacLab and train with skrl PPO using fused visual + proprioceptive inputs.  
Target behavior: stable grasp, lift, and post-lift target tracking.

## Initial Network Change (CNN + State Fusion)
The policy/value model was changed from state-only to fused input:

- Image branch (CNN):
  - `conv2d(32, k=8, s=4)`
  - `conv2d(64, k=4, s=2)`
  - `conv2d(64, k=3, s=1)`
  - `flatten -> linear(512) -> linear(64)`
- State branch:
  - `linear(128) -> linear(64)`
- Fusion head:
  - `concat(image_feat, state_feat) -> linear(256) -> linear(128)`

Main file:

- `config/franka/agents/skrl_camera_fusion_ppo_cfg.yaml`

## First Training Issues

### Startup / config errors
- Camera not enabled:
  - `A camera was spawned without the --enable_cameras flag`
  - Fixed by adding `--enable_cameras` to train/play commands.
- Observation structure mismatch:
  - `AttributeError: 'dict' object has no attribute 'shape'`
  - Fixed by restructuring observations for skrl-compatible access.
- Syntax import error:
  - `unexpected character after line continuation character`
  - Root cause: accidental Windows path text pasted into Python source.

### Training behavior issues
- Agent could grasp/lift, but showed high-frequency jitter after lift.
- `object_goal_tracking` degraded in mid/late training.
- Total reward rose early, then drifted down; min reward became worse.
- `action_rate` penalty became more negative, matching observed jitter.

## Core Modifications

### 1) Create isolated camera-fusion config path
Added separate files to avoid polluting the base lift config:

- `lift_camera_env_cfg.py`
- `lift_camera_fusion_env_cfg.py`
- `lift_camera_fusion_adapted_env_cfg.py`
- `config/franka/joint_pos_camera_env_cfg.py`
- `config/franka/joint_pos_camera_fusion_adapted_env_cfg.py`
- `mdp_camera_fusion/` (`observations`, `rewards`, `terminations`)

Registered task:

- `Isaac-Lift-Cube-Franka-Camera-Fusion-Adapted-v0`

### 2) Fix observation format and model input alignment
Observations were unified into a `policy` dict (`concatenate_terms=False`):

- `joint_pos`
- `joint_vel`
- `object_position`
- `target_object_position`
- `actions`
- `image`

This enabled direct, stable use of:

- `OBSERVATIONS["image"]`
- `OBSERVATIONS["...state..."]`

### 3) Stabilization tuning (checkpoint-compatible)
To reduce jitter and policy drift, only reward/PPO hyperparameters were adjusted.
No action dimension or network shape changes were made.

Reward changes in `lift_camera_fusion_adapted_env_cfg.py`:

- `reaching_object: 2.0 -> 1.5`
- `object_goal_tracking: 8.0 -> 10.0`
- `object_goal_tracking_fine_grained: 2.0 -> 3.0`
- `action_rate: -1e-5 -> -5e-5`
- `joint_vel: -1e-5 -> -3e-5`

PPO changes in `skrl_camera_fusion_ppo_cfg.yaml`:

- `learning_rate: 1e-4 -> 6e-5`
- `learning_epochs: 8 -> 5`
- `ratio_clip: 0.2 -> 0.15`
- `grad_norm_clip: 1.0 -> 0.5`
- `entropy_loss_scale: 0.001 -> 0.0005`
- stricter KL control

### 4) Post-lift visual gating
Added:

- `mdp_camera_fusion/observations_camera_fusion.py::gated_image_after_lift`

Behavior:

- High visual weight before lift (`1.0`)
- Smoothly reduced visual weight after lift (default `0.1`)
- Sigmoid transition to avoid hard switching
- Output tensor shape unchanged (checkpoint-safe)

## Checkpoint Compatibility Policy
To keep training resumable from existing checkpoints:

- Do not change action dimension.
- Do not change model layer shapes or parameter names.
- Do not change final observation tensor shapes.

## Commands

### Continue training
```bash
cd /root/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Lift-Cube-Franka-Camera-Fusion-Adapted-v0 \
  --algorithm PPO \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --checkpoint /root/IsaacLab/logs/skrl/franka_lift_camera_fusion/<run>/checkpoints/best_agent.pt \
  --max_iterations 30000
```

### Play / evaluate
```bash
cd /root/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task Isaac-Lift-Cube-Franka-Camera-Fusion-Adapted-v0 \
  --algorithm PPO \
  --enable_cameras \
  --num_envs 1 \
  --checkpoint /root/IsaacLab/logs/skrl/franka_lift_camera_fusion/<run>/checkpoints/best_agent.pt
```

### TensorBoard
```bash
cd /root/IsaacLab
tensorboard --logdir /root/IsaacLab/logs/skrl/franka_lift_camera_fusion --host 0.0.0.0 --port 6006
```
