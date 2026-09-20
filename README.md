# LIT on ImageWAM

The **Latent Interface Training (LIT)** instantiation of *Breaking the Vision–Action Shortcut: Latent
Interface Training for Generalizable Robot Foundation Models* on ImageWAM (FLUX.2 Klein 4B).
Hub, project page, checkpoints: https://github.com/jianmanlincjx/LIT · https://jianmanlincjx.github.io/LIT/ ·
https://huggingface.co/linjianman/LIT (public; Stage 1 and Stage 2 for every backbone)

This is a fork of [ImageWAM](https://github.com/yuyangalin/ImageWAM) (original README:
[`README_upstream.md`](./README_upstream.md) — installation, FLUX.2 weights, data preparation).
Use branch **`feat/goal-prior-bottleneck-fix`**.

**Environment and base weights** (from `README_upstream.md`; the FLUX.2 repositories on Hugging Face may
require accepting a licence first):

```bash
git clone -b feat/goal-prior-bottleneck-fix https://github.com/jianmanlincjx/ImageWAM.git && cd ImageWAM
uv sync --python 3.11 --extra shared && source .venv/bin/activate
cp .env.example .env.local           # the scripts read paths from here

git clone https://github.com/black-forest-labs/flux2 third_party/flux2          # FLUX.2 source (pinned commit in README_upstream)
hf download black-forest-labs/FLUX.2-klein-base-4B --local-dir checkpoints/flux2/FLUX.2-klein-base-4B
hf download black-forest-labs/FLUX.2-dev ae.safetensors --local-dir checkpoints/flux2/FLUX.2-dev   # gated
# in .env.local
FLUX2_SRC=$PWD/third_party/flux2
FLUX2_MODEL_PATH=$PWD/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors
FLUX2_AE_MODEL_PATH=$PWD/checkpoints/flux2/FLUX.2-dev/ae.safetensors
FLUX2_QWEN3_MODEL_SPEC=Qwen/Qwen3-4B
```

Every LIBERO-Plus number was produced with **`LIBERO_PLUS_FIX_LANG=1`**; Overall is the mean over the seven
perturbation axes. ImageWAM evaluates in three conditioning modes (`both` / `ref_only` / `syn_only`); the
reported numbers use `both`.

---

## 1. Evaluate the released checkpoint

```bash
hf download linjianman/LIT --include "imagewam/*" --local-dir ./LIT_ckpt
export CKPT_PATH=./LIT_ckpt/imagewam/lit_stage2/model.pt
export DATASET_STATS_PATH=./LIT_ckpt/imagewam/lit_stage2/dataset_stats.json
```

```bash
# LIBERO (in-distribution)
# NUM_GPUS=8 FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero.sh
NUM_GPUS=1 MAX_TASKS_PER_GPU=2 FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero.sh

# LIBERO-Plus (out of distribution): 10,030 tasks, one episode each
LIBERO_PLUS_FIX_LANG=1 NUM_GPUS=8 FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero_plus.sh
```

Both launchers run the manager in `experiments/libero/`; results are written per task as
`gpu*_task*_results.json` with `summary.json` and `task_success_rates.csv` alongside. Aggregate the
LIBERO-Plus run per axis with `scripts/aggregate.py` in the LIT hub.

---

## 2. Train, then evaluate

The **baseline** is the released ImageWAM FLUX.2 Klein 4B LIBERO checkpoint evaluated as-is
(see the Hugging Face collection linked from `README_upstream.md`); we do not retrain it. Data: LIBERO, all
four suites, `no_noops`, prepared as in `README_upstream.md` (`scripts/data/`). **LIT** is two runs on
`configs/task/libero_flux2_klein_4b_goal_prior_stage{1,2}.yaml`:

```bash
# Stage 1 — vision-free SE(3)-conditioned action prior: batch 64, lr 2e-4, warmup 2000, 10K steps
bash scripts/flux2/run_train_flux2_klein_goal_prior_stage1.sh

# Stage 2 — pose-supervised latent interface, initialised from Stage 1: batch 10, lr 1e-4, warmup 5%, 10 epochs
bash scripts/flux2/run_train_flux2_klein_goal_prior_stage2.sh
```

To skip Stage 1, use the released prior: `STAGE1_CHECKPOINT=./LIT_ckpt/imagewam/lit_stage1/model.pt`.

Then evaluate `<stage2_run>/checkpoints/weights/step_034720.pt` with `<stage2_run>/dataset_stats.json`
exactly as in §1 (the released `lit_stage2/model.pt` is that file, renamed; 34,720 steps = 10 epochs).
`scripts/audit_goal_prior_v2.py` checks a Stage-2 run's latent usage before you trust it.

---

## 3. How LIT is integrated in ImageWAM

ImageWAM is a world–action model built on an image-editing foundation model: the FLUX.2 Klein DiT edits
the current observation towards the future, and an action head generates the chunk from the DiT's
per-block features (`src/imagewam/`). LIT touches only the path from those features to the action head:

| Piece | Where | What it does |
| --- | --- | --- |
| Latent interface | `src/imagewam/models/backbones/goal_pose_prior.py`, wired in `backbones/imagewam.py` / `mot.py` | 100 learnable latents cross-attend to the FLUX.2 block features and the Qwen3 text tokens and become the action head's only visual input |
| Firewall | `backbones/imagewam.py`, switched by `goal_prior_stage: stage2` in `configs/model/imagewam_flux2_klein_4b_goal_prior_stage2.yaml` | the direct feature path from the DiT into the action head is closed |
| Spatial supervision | `GoalPoseDecoder` in `goal_pose_prior.py`, `goal_prior.lambda_pose: 0.3` | 8 latents decode to the chunk-end SE(3) target; MSE added to the action objective |
| Stage-1 conditioning | `GoalPoseEncoder` in `goal_pose_prior.py`, `goal_prior_stage: stage1` | terminal pose encoded into the action head's conditioning while the image backbone is off |
| Launchers | `scripts/flux2/run_train_flux2_klein_goal_prior_stage{1,2}.sh` | Stage 2 reads the Stage-1 weights |

Interface settings match the other three backbones and were not tuned per model: `num_latents=100`,
`num_pose_tokens=8`, `latent_dim=768`, `inner_dim=512`, `lambda_pose=0.3`. Action representation, horizon
(16) and replanning (12) are the upstream ImageWAM defaults.
