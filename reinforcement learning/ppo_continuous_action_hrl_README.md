# PPO Continuous Action HRL Notes (Action Hold + Async + Resume)

## Summary of Today's Changes

We modified `cleanrl/ppo_continuous_action_hrl.py` with two practical improvements:

- Switched vectorized environment support to allow `AsyncVectorEnv` (default), with a runtime switch to `SyncVectorEnv`
- Added resumable training checkpoints (model, optimizer, global step, iteration, env normalization stats, and action-hold state)

These changes are intended to improve throughput on multi-core CPUs and allow interrupted training to continue.

## New CLI Options

- `--vec-env-mode async|sync`
- `--resume-from <checkpoint_path>`
- `--save-checkpoint True|False`

Default behavior now uses `async` vector envs and saves a resumable checkpoint at the end of training.

## Resume Training Behavior

The new checkpoint restores:

- `agent.state_dict()`
- `optimizer.state_dict()`
- `global_step`
- completed `iteration`
- per-environment normalization statistics:
  - `NormalizeObservation.obs_rms`
  - `NormalizeReward.return_rms`
  - `NormalizeReward.discounted_reward`
- `ActionHoldWrapper.decision_count`

Note: this does **not** restore the exact internal simulator state of each environment. Training resumes correctly in practice, but it is not a bitwise-identical continuation.

## Why GPU Usage Looks Low (and CPU Usage Looks High)

This is expected for PPO + MuJoCo-style continuous control:

- Environment stepping (`env.step`) is CPU-heavy
- PPO MLPs here are small (2x64), so GPU compute is brief
- `AsyncVectorEnv` increases CPU process usage, which is usually the real bottleneck

Result: high CPU usage (e.g., ~600%) and near-zero visible GPU utilization can still mean CUDA is working.

## Why Performance Became Worse After Adding Action Hold

This is also common and does not necessarily mean the code is broken.

### Main reasons

- **Action hold may be too large**
  - Example: `0.3s` can correspond to several simulator steps, which reduces control frequency too much.
- **Annealing hold changes the task during training**
  - `action_hold_start_seconds -> action_hold_end_seconds` changes effective dynamics and decision frequency.
- **Time-scale mismatch in PPO targets**
  - Current rollout rewards are aggregated over repeated actions, but GAE/value targets still use a fixed single-step `gamma`.
  - With variable repeat length, this creates a mismatch in the training objective.
- **Original PPO hyperparameters are tuned for per-step control**
  - Action hold changes horizon, exploration frequency, and credit assignment.

### Practical consequence

`policy_loss`, `value_loss`, and `explained_variance` formulas are still valid mathematically, but they are optimizing/measuring a **macro-step approximation** of the original problem when action repeat is used this way.

## Recommended Next Steps

1. Test smaller/fixed action hold first (`0.0`, `0.05`, `0.1`) before annealing.
2. Disable hold annealing during ablations (`--action-hold-anneal False`).
3. Compare `sync` vs `async` using `SPS` and final return.
4. If theoretical correctness matters, implement variable-discount GAE using per-transition repeat counts (`gamma ** repeat_steps`).

## Example Commands

Train (async, save checkpoint):

```bash
python cleanrl/ppo_continuous_action_hrl.py \
  --vec-env-mode async \
  --save-checkpoint True
```

Resume training:

```bash
python cleanrl/ppo_continuous_action_hrl.py \
  --resume-from runs/<run_name>/ppo_continuous_action_hrl.cleanrl_checkpoint \
  --vec-env-mode async
```
