# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A fork of [ImageWAM](https://github.com/yuyangalin/ImageWAM) (a world–action model built on image-editing
foundation models) that instantiates **LIT — Latent Interface Training** on the FLUX.2 Klein 4B backbone.
Upstream documentation lives in `README_upstream.md`; the LIT reproduction guide is `README.md`.
Work happens on branch `feat/goal-prior-bottleneck-fix`.

The LIT / "goal-pose prior" feature lives entirely on the path from the FLUX.2 DiT's per-block features to
the action head. Three docs carry the design rationale and are worth reading before touching that path:

- `docs/goal_pose_prior_flux2.md` — the operations manual (Chinese): locked decisions, launch commands, expected logs, known risks.
- `docs/CHANGELOG_v2.md` — why each mechanism exists, with the ablations and the bugs that real training exposed (Chinese).
- `docs/dependencies.md` — pinned external checkouts (FLUX.2, OmniGen2, DIM) and vendored code.

Two evaluated revisions exist. The **v1** tag `goal-pose-prior-v1-20260825` (hard firewall) scores *below*
baseline on LIBERO-Plus; **v2** tag `goal-pose-prior-v2-20260825` (gated context, `action_sees_ref`) is the
version in the paper tables. Do not treat `e81335b` as final — its commit message's "(evaluated version)"
refers to v1.

## Environment

```bash
uv sync --python 3.11 --extra shared && source .venv/bin/activate
cp .env.example .env.local          # scripts/common.sh sources this automatically
```

`scripts/common.sh:imagewam_init` `cd`s to the repo root, sources `.env.local`, and prepends the venv's
`nvidia/npp/lib` to `LD_LIBRARY_PATH` (required by TorchCodec's dlopen). Every `scripts/**/run_*.sh`
entrypoint goes through it, and calls `imagewam_require_env` for the variables it needs — so missing
configuration fails with exit code 2 rather than a stack trace. Required: `DATA_ROOT`, `FLUX2_SRC`,
`FLUX2_MODEL_PATH`, `FLUX2_AE_MODEL_PATH`, `FLUX2_QWEN3_MODEL_SPEC`.

FLUX.2 is **not vendored**: clone `https://github.com/black-forest-labs/flux2` to `third_party/flux2` and
check out the commit pinned in `docs/dependencies.md`. Launchers export
`PYTHONPATH=$REPO_ROOT/src:$FLUX2_SRC/src:$FLUX2_SRC`. FLUX.2 variants need `transformers==4.56.1`; the
OmniGen2/Ovis variants need the `transformers==4.51.3` pinned in `pyproject.toml` — switch back before
running those.

## Commands

All commands run from the repository root.

```bash
# Tests. Most test files are pytest-style functions; tests/test_goal_pose_prior.py is unittest-style.
# pytest is not in pyproject.toml/uv.lock — install it in the venv if it is missing.
PYTHONPATH=src python -m pytest tests/ -q
PYTHONPATH=src python -m unittest tests.test_goal_pose_prior    # the unittest-only file

# Import-check the LIT-critical modules without a GPU (from docs/goal_pose_prior_flux2.md)
.venv/bin/python -m py_compile src/imagewam/models/backbones/goal_pose_prior.py \
  src/imagewam/models/backbones/imagewam.py src/imagewam/models/backbones/mot.py \
  src/imagewam/trainer.py src/imagewam/runtime.py src/imagewam/datasets/lerobot/robot_video_dataset.py

# LIT training — Stage 1 (vision-free), then Stage 2 (needs Stage 1)
bash scripts/flux2/run_train_flux2_klein_goal_prior_stage1.sh
STAGE1_CHECKPOINT=<run>/checkpoints/weights/step_010000.pt \
  bash scripts/flux2/run_train_flux2_klein_goal_prior_stage2.sh
# smoke test on 8 GPUs, ~20 steps each
GPU_PER_NODE=8 bash scripts/flux2/run_train_flux2_klein_goal_prior_stage1.sh max_steps=20 save_every=20 eval_every=0 log_every=1

# Baseline (non-LIT) training, and evaluation
GPU_PER_NODE=8 TASK_TYPE=libero FLUX2_VARIANT=4b PRECOMPUTE_QWEN3_CACHE=true bash scripts/flux2/run_train_flux2_klein_imagewam.sh
NUM_GPUS=8 FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero.sh
LIBERO_PLUS_FIX_LANG=1 NUM_GPUS=8 FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero_plus.sh
```

Any extra argv after a launcher is passed straight through as Hydra overrides:
`bash scripts/flux2/run_train_flux2_klein_goal_prior_stage2.sh batch_size=10 resume=<dir> wandb.mode=offline`.
Useful overrides: `batch_size`, `max_steps`, `learning_rate`, `warmup_steps`, `eval_every=0`, `resume`,
`ZERO_STAGE=1|2`, `TRAIN_NORM_STATS`, `GPU_PER_NODE`, `DRY_RUN=true`, `IMAGEWAM_QUIET=false`.

Evaluation can also be driven from a training run instead of an explicit checkpoint:
`EXP_PATH=<run> EVAL_TRAIN_STEP=10000` resolves `CKPT_PATH=$EXP_PATH/checkpoints/weights/step_<N>.pt`
and `DATASET_STATS_PATH=$EXP_PATH/dataset_stats.json` (`scripts/common.sh:imagewam_ckpt_from_exp`).

## Architecture

### Entry points and config flow

`scripts/train.py` is the single Hydra entrypoint (`configs/train.yaml` as base). `scripts/train_zero{1,2}.sh`
wrap it in `accelerate launch` with `scripts/accelerate_configs/accelerate_*.yaml` and DeepSpeed configs, and
compute the run directory `runs/<task>/<timestamp>/`. `runtime.run_training` then:

1. `instantiate(cfg.model)` — the `_target_` in a model config is one of the `create_imagewam_*` factories in
   `src/imagewam/runtime.py`, each a thin validating shim over an `ImageWAM.from_*_pretrained` classmethod.
2. `build_datasets(cfg.data)` — hydra-instantiates the dataset and processor.
3. Constructs `Wan22Trainer` and calls `.train()`.

Configs compose in three layers under `configs/`: `task/*` overrides `data/*` and `model/*` via Hydra
`defaults:`. Note `configs/data/libero_omnigen2_pair.yaml` is the data config used by the *FLUX.2* tasks too —
the name is historical.

A run directory (`runs/<task>/<timestamp>/`) ends up containing `config.yaml` (dumped before the trainer is
built), `checkpoints/weights/step_XXXXXX.pt` (main process only), `checkpoints/state/step_XXXXXX/` (the
resumable DeepSpeed/optimizer/scheduler shards, all ranks, `keep_latest_state_only` prunes older ones),
`eval/*.mp4`, and `dataset_stats.json` — which is written by the *dataset*, not the trainer. `resume=<dir>`
restores full state plus the sampler position; `resume=<file.pt>` restores weights only and leaves the
optimizer fresh.

### Model

`ImageWAM` (`src/imagewam/models/backbones/imagewam.py`) is one class that dispatches on `self.stack`
(`wan22` | `omnigen2` | `ovis_u1` | `flux2` | `dim`). Most methods branch on stack; **only `flux2` implements
the LIT / goal-pose-prior path**. It holds a video DiT, an `ActionDiT` action head, a VAE, a text encoder, an
optional `proprio_encoder`, and four flow-matching schedulers (separate train/infer shifts for video vs
action). `self.dit` is an alias for `self.mot` (the `MoT` wrapper holding both experts as
`mixtures["video"]` / `mixtures["action"]`), kept for the trainer/optimizer.

The action head is `ActionDiTFlux2`: a "slim" FLUX.2 — same 24 heads × 128 head-dim (attention width 3072,
matching the video expert so q/k/v concatenate with no projection) but a 1024-wide residual stream. Its
weights are initialised from the FLUX.2 checkpoint by `scripts/flux2/preprocess_action_dit_flux2.py`, which
linearly interpolates mismatched shapes and applies a `sqrt(src/dst)` alpha scaling to preserve variance.

`imagewam.py` is ~5.4k lines; `mot.py` (~1.8k) holds the per-layer joint attention and the FLUX.2 forward
paths; `goal_pose_prior.py` (~780) holds the LIT modules in isolation and is the most self-contained place to
read the mechanism.

### Training losses

Flow matching, dispatched by stage in `_training_loss_flux2_*`:

| Stage | Loss | Notes |
| --- | --- | --- |
| baseline | `0.5·L_video + 1.0·L_action` | both streams noised by independently sampled timesteps |
| `stage1` | `1.0·L_action` only | the image stream is empty (or `stage1_null_image_tokens`); the oracle chunk-end pose is encoded and concatenated onto the text tokens, giving `[txt \| pose \| action]` |
| `stage2` | `0.5·L_video + 1.0·L_action + 0.3·L_pose` | 8 of the 100 latents decode to the chunk-end SE(3) target |

`_goal_prior_diagnostics` emits the gate bias, blackout fraction, and per-regime losses
(`regime/loss_{both,ref_only,syn_only}`, `loss_action_fallback`) used to read a run.

Validation during training re-renders decoded video and computes PSNR/SSIM — except in stage 1, where the
image stream does not exist, so `evaluate()` returns only loss and action/pose metrics.

### Checkpoint contract and the Stage1→Stage2 bridge

`save_checkpoint` writes a dict with `mot`, `step`, `torch_dtype`, `goal_prior_stage`, plus whichever of
`proprio_encoder`, `goal_pose_encoder`, `stage1_null_image_tokens`, `semantic_visual_aggregator`,
`semantic_visual_pose_norm`, `semantic_visual_pose_decoder`, `optimizer` exist.

`load_checkpoint(..., goal_prior_bridge=True)` (used when a stage-2 model loads a stage-1 payload, forced by
the trainer unless `resume` is set) is **fail-closed**: any MoT or `proprio_encoder` key that is not inherited
exactly raises. The only permitted exceptions come from the prefix constants in `goal_pose_prior.py`:

- dropped: `goal_pose_encoder.*`, `stage1_null_image_tokens` (`STAGE1_ONLY_CHECKPOINT_PREFIXES`)
- randomly initialised: `semantic_visual_aggregator.*`, `semantic_visual_pose_norm.*`,
  `semantic_visual_pose_decoder.*` (`STAGE2_ONLY_CHECKPOINT_PREFIXES`)

Stage 2 **requires** `stage1_checkpoint` unless `resume` is set, so it can never silently train from the
ActionDiT init.

### Trainable-parameter whitelist

`Wan22Trainer._apply_dit_only_train_mode` calls `model.eval()` + `model.requires_grad_(False)`, re-enables
`model.dit` and `proprio_encoder`, then delegates to `ImageWAM.apply_trainable_policy()` — which is a no-op
for every stack except `flux2`. For `flux2`:

- **stage1**: action expert trainable; video expert frozen and `.eval()`; `goal_pose_encoder` trainable.
- **stage2**: both experts trainable plus the three aggregator modules.
- **baseline**: everything in the MoT trainable, or only `.lora_A`/`.lora_B` when `flux2_lora_enabled`.

`collect_trainable_parameters()` / `summarize_trainable_parameters()` build and log the optimiser's parameter
list from `requires_grad`.

### LIT mechanics (the part that needs several files to understand)

`goal_pose_prior.py` defines the interface; `mot.py` wires it into the FLUX.2 forward.

- `GoalPoseEncoder` (stage 1) turns an 8-D pose into **8×3072** FLUX-hidden tokens.
- `SemanticVisualAggregator` (stage 2) holds **100 learnable latents (8 pose + 92 context, dim 768)** that
  cross-attend to the FLUX.2 block features and Qwen3 text tokens, per layer group (25 layers → 5 groups via
  `layer_idx // 5`), and project to synthetic K/V (`inner_dim` 512) injected into the action expert.
- `GoalPoseDecoder` maps the 8 pose latents back to the SE(3) target for `L_pose`.
- The **firewall** is `build_stage2_action_attention_mask` with `ref_len=0`: the action queries' K/V layout is
  `[txt | ref? | syn | action]`, so omitting the ref block literally removes the raw image columns.
- The **cold-start gate** (`mot.py:_flux2_gate_action_mask`) adds a learnable per-layer logit bias to the
  synthetic columns so stage 2 begins as a near no-op. It is *additive on a float mask*, not a bool mask.

Config knobs (`configs/model/imagewam_flux2_klein_4b_goal_prior_stage2.yaml`) and their defaults are all
"off / evaluated behaviour": `context_token_dropout`, `context_blackout_prob`, `zero_init_value`,
`syn_gate_bias_init`, `gate_pose_tokens`, `latent_layout` (B1 grid), and the `action_sees_ref` plan-B switch
with `p_both`/`p_ref_only`/`p_syn_only`. `tests/test_bottleneck_fixes.py::test_defaults_match_evaluated_revision`
guards that invariant.

### Inference modes

`imagewam._resolve_infer_mode` picks which channels the action expert may attend, defaulting to `full` when the
checkpoint was trained with `action_sees_ref` and `firewall` otherwise, and overridable with the
`IMAGEWAM_INFER_MODE` env var (`full` | `firewall` | `ref_only`). Requesting `full`/`ref_only` from a
checkpoint trained without the reference channel raises.

Naming trap: the README's conditioning modes `both` / `ref_only` / `syn_only` are the **training regimes**
(`p_*`); the code's inference modes are `full` / `firewall` / `ref_only`. Reported numbers use `both` ⇔ `full`.

### Datasets

`RobotVideoDataset` reads LeRobot-format directories (`dataset_dirs`) of preprocessed LIBERO / RoboTwin data.
`num_frames: 17` = horizon 16 + current frame; actions are 7-D with the gripper dim excluded from
delta-action conversion; state/proprio is 8-D. Normalisation is **min/max** from
`pretrained_norm_stats` (a `dataset_stats.json`), not mean/std; a dataset with no `pretrained_norm_stats`
computes stats and writes them into the run's work dir, which is why evaluation needs
`DATASET_STATS_PATH` alongside the checkpoint. Two optional caches: Qwen text embeddings
(`qwen_text_cache_dir`, `qwen_text_cache_format=qwen3_flux2`, `qwen_context_len`, precomputed by
`scripts/flux2/precompute_flux2_qwen3_embeds.py`) and the ActionDiT init weights
(`ACTION_INIT`, default `checkpoints/action_dit_flux2_<variant>_libero_init.pt`, auto-generated when absent,
`REBUILD_ACTION_INIT=true` to force). `vision_free: true` (stage 1) skips image decoding entirely and sets
`goal_pose = proprio[-1]` clamped to the episode end.

### Evaluation

`experiments/libero/run_libero_manager.py` is a Hydra entrypoint that fans tasks across GPUs
(`MULTIRUN.num_gpus`, `max_tasks_per_gpu`, `chunk_size`, `task_sample_ratio`, `task_suite_names`) and spawns
`eval_libero_single.py` workers — hence `LIBERO_WORKER_ENV_SOURCE` must point at the venv activation script.
Results land in `evaluate_results/<family>/<ckpt_tag>/<run_ts>/` as `gpu*_task*_results.json` plus
`summary.json` and `task_success_rates.csv` (`summarize_results.py`). RoboTwin uses the parallel
`experiments/robotwin/` manager and the `imagewam_policy` adapter symlinked into `third_party/RoboTwin`.

## Gotchas

- **Metrics must be emitted on every rank, every step.** The trainer all-gathers one `accelerator.gather` call
  per loss-dict key; a key present on some ranks and absent on others desynchronises NCCL and hangs the job.
  This is why the blackout sampler draws a fixed count per rank and `_goal_prior_diagnostics` always emits
  every key. `tests/test_diagnostics.py` and `tests/test_planb.py` exist to hold that property.
- **Dropout-style mechanisms read the *module's* train/eval mode, not the top-level model's.** The trainer
  never calls `.train()` on `ImageWAM` itself (only on `dit`/experts/aggregator), so anything keyed off
  `self.training` at the top level silently no-ops. Getting this wrong once produced a 35-hour run in which
  every new mechanism was inactive.
- **Stage 1 must reuse the baseline's `dataset_stats.json`** (`TRAIN_NORM_STATS`, defaulted in the launchers
  to a specific baseline run path). Recomputing stats would break the min/max used by the comparison run.
- **`scripts/audit_goal_prior_v2.py` has the author's absolute paths hardcoded** (`/data2/JM/Code/ImageWAM-v2`)
  in both its `sys.path` insert and `ROOT` — repoint them before running it. It is a wiring audit (does the
  shipped YAML actually produce the intended model?), not a unit test.
- **`configs/task/libero_flux2_klein_4b_goal_prior_stage*.yaml` contain author-specific absolute paths** for
  `pretrained_norm_stats`; the launchers override them, so don't rely on the YAML directly.
- `third_party/flux2` is gitignored and must be cloned separately; `third_party/RoboTwin/assets/` ships only
  `_download.py`.
- Stage-2 video decoding needs the `LD_LIBRARY_PATH` fix from `imagewam_init`, otherwise dataloader workers
  fail to load `libnppicc.so.11`.
- Frozen-FLUX stage 1 still runs the 25 frozen layers through autograd (gradient must reach the goal encoder),
  so `mot_checkpoint_mixed_attn=true` checkpoints the whole FLUX/Action MLP per layer — checkpointing only
  mixed attention OOMs at bs=128.
- `.gitignore` excludes `.claude/`, `runs/`, `evaluate_results/`, `checkpoints`, `*.pt`, and `*.txt` — the
  last one means any plain-text notes you create will not show up in `git status`.
- `ImageWAMCacheIDM` / `ImageWAMNoiseIDM` are subclasses that reuse `ImageWAM`; changes to the loss paths or
  checkpoint payload affect them.
