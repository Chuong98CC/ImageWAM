# FLUX.2 ImageWAM 两阶段 Goal-Pose Prior

本文是实现落地后的操作手册。这是官方 FLUX.2 ImageWAM baseline 的 **add-on**，不替换图像 flow matching，也不改现有 baseline 实验路径。

## 锁定决策

- Topology：MolmoAct2 V3 软瓶颈。Stage1 = **8** 个 oracle goal token；Stage2 = **100 = 8 pose + 92 context**。
- Horizon：**H=16**（`num_frames=17`）。`goal_pose` 是 ImageWAM min/max 归一化后的 `state[t+16]`，不是任务终点 object pose。
- 归一化：沿用 ImageWAM **min/max**。Stage1/Stage2 **不重算** stats，复用当前 ImageWAM baseline run 的 `dataset_stats.json`（min/max 与官方 release 相同；mean/std 有浮点差，训练不用）。
- Stage1：**真正无图**。不解码视频、不算 `L_image`、不跑 VAE。冻结预训练 FLUX.2；训练 ActionDiT（baseline init）+ `proprio_encoder` + GoalEncoder。
- GoalEncoder 输出 **8×3072**（FLUX hidden），插入在 `txt_in` **之后**，不是 7680D Qwen 接口。
- Stage1 超参：LR **`2e-4`**（baseline `1e-4` 的两倍，无图所以略高，但不再用 `5e-4`）。固定 **10,000 steps**，**warmup 2000**。单卡 **batch size 128**（全局 8×128，所以不再用 20k）。
- Padding goal：LeRobot 把越界的 `t+16` **clamp 到本条 episode 末帧**（不跨 episode）。Stage1 **跟 MolmoAct2**：8 个 goal token 始终可见，clamp 末帧仍作为条件；Action loss 只靠 `action_is_pad`。Stage2 **`L_pose` 用 `goal_pose_is_pad` mask**。不开启 `skip_padding_as_possible`。
- W&B：**offline**。事后 `wandb sync -e <entity> -p imagewam <offline-run>`。
- Stage2：独立 follow baseline（LIBERO 4B：bs=10、10 epoch、**峰值 LR `1e-4` 与 baseline 完全一致**、wd `1e-2`、bf16）。**不扣除** Stage1 的 10k。公平对比的是两阶段策略，不是另一套优化配方。Warmup **5000**（比 baseline 的 5%≈1.7k 长）只改变爬升，cosine 仍到达 `1e-4` 再衰减。  
  Loss = `0.5 L_image + 1.0 L_action + 0.3 L_pose`。
- Stage2 启动：无 `resume` 时 **必须** 提供 `stage1_checkpoint`，否则直接报错，避免静默从 `ACTION_INIT` 开训。
- Stage2：ActionDiT **不能**看 raw ref / noisy target，只看 text/state + 100 synthetic KV + action self。Aggregator 的 visual context **只取 ref**，target 永不进入。
- 5 groups 覆盖 25 层：`group_idx = layer_idx // 5`（5 double + 20 single）。
- Stage1 初始化：原始 FLUX.2 Klein 4B（冻）+ `action_dit_flux2_4b_libero_init.pt`。不加载已训 ImageWAM/VLA ckpt。
- Stage1→Stage2 bridge：继承 ActionDiT + proprio；**丢弃 GoalEncoder**；aggregator / pose decoder **随机初始化**。fail-closed key 检查。
- 冻结 FLUX 时 **禁止** 用包住整条 text stream 的 `no_grad()`，梯度要回到 GoalEncoder / proprio。因此 Stage1 仍会把 25 层冻结 FLUX 放进 autograd；`mot_checkpoint_mixed_attn=true` 时 `_forward_flux2` **按层 checkpoint** FLUX/Action MLP（不只 checkpoint mixed attn），否则 8 卡 per-device bs=128 会在 single-block MLP OOM。这不改变峰值 LR，也不对 text stream 包 `no_grad()`。
- Stage2 推理接口 **不接受** future goal；LIBERO rollout 只给当前图、语言、current state。

## 文件地图

| 角色 | 路径 |
| --- | --- |
| GoalEncoder / Aggregator / PoseDecoder / mask / checkpoint 契约 | `src/imagewam/models/backbones/goal_pose_prior.py` |
| Stage1 无图数据、`goal_pose=proprio[-1]` | `src/imagewam/datasets/lerobot/robot_video_dataset.py`（`vision_free`） |
| 不解码图像 | `src/imagewam/datasets/lerobot/base_lerobot_dataset.py`、`.../processors/imagewam_processor.py` |
| Stage1/Stage2 loss、trainable policy、checkpoint、infer | `src/imagewam/models/backbones/imagewam.py` |
| Stage2 逐层 aggregator、synthetic KV、attention 隔离 | `src/imagewam/models/backbones/mot.py`（`_forward_flux2_stage2`） |
| Optimizer 白名单、Stage1 bridge、Stage1 eval 跳过 PSNR | `src/imagewam/trainer.py` |
| Hydra factory | `src/imagewam/runtime.py`（`create_imagewam_flux2_klein`） |
| Stage1 配置 | `configs/model/imagewam_flux2_klein_4b_goal_prior_stage1.yaml`、`configs/task/libero_flux2_klein_4b_goal_prior_stage1.yaml` |
| Stage2 配置 | `configs/model/imagewam_flux2_klein_4b_goal_prior_stage2.yaml`、`configs/task/libero_flux2_klein_4b_goal_prior_stage2.yaml` |
| 启动脚本 | `scripts/flux2/run_train_flux2_klein_goal_prior_stage1.sh`、`..._stage2.sh` |
| CPU 单测 | `tests/test_goal_pose_prior.py` |

官方 baseline 脚本 `scripts/flux2/run_train_flux2_klein_imagewam.sh` **不要改、不要纳入本功能提交**。

## Checkpoint 契约

Stage1 `save_checkpoint` 载荷：

- `mot`：完整 MoT（冻结 FLUX + 已训 ActionDiT）
- `proprio_encoder`
- `goal_pose_encoder`
- `goal_prior_stage: "stage1"`

Stage2 从原始 FLUX.2 + ActionDiT init 构建后，在 `accelerator.prepare` **之前**、且 `resume is None` 时加载 `stage1_checkpoint`，`goal_prior_bridge=True`：

- **必须精确继承**：`mot`（ActionDiT + FLUX）和 `proprio_encoder`
- **必须丢弃**：`goal_pose_encoder.*`
- **必须保持随机初始化**：`semantic_visual_aggregator.*`、`semantic_visual_pose_norm.*`、`semantic_visual_pose_decoder.*`
- 任何其它 missing / unexpected key 直接报错

续训 Stage2 时用 `resume=`，不要再走 Stage1 bridge。

## 启动命令

环境与官方 FLUX.2 ImageWAM 相同：`DATA_ROOT`、`FLUX2_SRC`、`FLUX2_AE_MODEL_PATH`，以及可选 `.env.local`。  
Qwen cache **复用 baseline**（`qwen3_flux2`，默认 `${DATA_ROOT}/flux2_qwen3_cache_4b`）。不必为 Stage1 重算。

Stage1（无图，10k step，LR `2e-4`，warmup 2000，per-device bs=128，wandb offline）：

```bash
bash scripts/flux2/run_train_flux2_klein_goal_prior_stage1.sh
# 可选：TRAIN_NORM_STATS=/path/to/dataset_stats.json
```

Stage2（独立 follow baseline 预算；必须提供 Stage1 权重）：

```bash
STAGE1_CHECKPOINT=/path/to/stage1/checkpoints/weights/step_010000.pt \
  bash scripts/flux2/run_train_flux2_klein_goal_prior_stage2.sh
```

Stage2 + 视频专家 LoRA（可选，**默认关闭**）：

```bash
STAGE1_CHECKPOINT=/path/to/stage1/checkpoints/weights/step_010000.pt \
FLUX2_LORA_ENABLED=true FLUX2_LORA_RANK=16 \
  bash scripts/flux2/run_train_flux2_klein_goal_prior_stage2.sh
```

打开后冻结视频专家的 3.876B 底座权重，只训练 80 个 Linear 上的低秩适配器（rank16 ≈ 23.6M，
占该专家 2.36%）；Action Expert 与三个聚合器模块仍全量训练。rank/alpha/dropout 见
`FLUX2_LORA_RANK`、`FLUX2_LORA_ALPHA`、`FLUX2_LORA_DROPOUT`。

两点硬约束：

- **只有 Stage2 支持**。Stage1 整个视频专家都被冻结（梯度经激活回传到 `goal_pose_encoder`，
  不经权重），适配器永远不会更新；脚本此时会打 warning，不要开。
- **必须合并保存**。`FLUX2_LORA_ENABLED=true` 时启动脚本自动追加
  `model.flux2_lora_config.save_lora_merged=true`。若手工用 Hydra override 开启 LoRA 而忘了这个
  开关，写出的 checkpoint 键名带 `base_layer` 且体积翻倍，Stage1→Stage2 桥接与评测加载器都会
  拒绝加载。

注意这**不是**同一次实验的省显存版本：视频专家不再能适应 LIT 训练域，Stage2 的
`0.5 · L_video` 从全量微调退化为每 block 八个投影族的 rank-16 修正。见
`docs/flux2_architecture.md` §5.2。

常用 Hydra override：

- `batch_size=`
- `max_steps=`（Stage1 默认 10000；Stage2 默认按 10 epoch 估算）
- `learning_rate=`
- `warmup_steps=`（Stage1 默认 2000；Stage2 默认 5000）
- `eval_every=0`（Stage1 默认已关）
- `resume=`（优先于 `stage1_checkpoint`）
- `ZERO_STAGE=1|2`
- `TRAIN_NORM_STATS=`（默认当前 LIBERO baseline run 的 `dataset_stats.json`）
- `wandb.mode=`（默认 `offline`）

8 卡 ZeRO 与 baseline 相同，脚本默认 `GPU_PER_NODE=8`、`ZERO_STAGE=1`。

## 期望日志

Stage1 启动应出现 trainable 白名单，大致只有：

- `mot.mixtures.action`（ActionDiT）
- `proprio_encoder`
- `goal_pose_encoder`

不应出现 `mot.mixtures.video` / VAE / Qwen。训练 log 只应有 `loss_action`（没有 `loss_video` / `loss_pose`）。

Stage2 白名单应包含：

- `mot.mixtures.video`
- `mot.mixtures.action`
- `proprio_encoder`
- `semantic_visual_aggregator`
- `semantic_visual_pose_norm`
- `semantic_visual_pose_decoder`

训练 log 同时出现 `loss_video`、`loss_action`、`loss_pose`。图像 PSNR/SSIM 只在 Stage2 eval 记录。

## 无 GPU 可做

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_goal_pose_prior
.venv/bin/python -m py_compile \
  src/imagewam/models/backbones/goal_pose_prior.py \
  src/imagewam/models/backbones/imagewam.py \
  src/imagewam/models/backbones/mot.py \
  src/imagewam/trainer.py \
  src/imagewam/runtime.py \
  src/imagewam/datasets/lerobot/robot_video_dataset.py
```

单测覆盖：goal 提取、8×3072 / 100×768 / decoder 8、25 层→5 group、pose pad mask、Stage2 action mask 无 raw image 列、Stage1→Stage2 key 契约、Stage1 freeze 白名单。

## 8 卡 smoke（模拟正式训练，各 20 step）

`GPU_PER_NODE=8`、`ZERO_STAGE=1`，batch / 数据 / 脚本与正式训练相同，只改 `max_steps=20`。到达 max_steps 会保存 `step_000020.pt`。

```bash
GPU_PER_NODE=8 bash scripts/flux2/run_train_flux2_klein_goal_prior_stage1.sh \
  max_steps=20 save_every=20 eval_every=0 log_every=1

GPU_PER_NODE=8 STAGE1_CHECKPOINT=<stage1>/checkpoints/weights/step_000020.pt \
  bash scripts/flux2/run_train_flux2_klein_goal_prior_stage2.sh \
  max_steps=20 save_every=20 eval_every=0 log_every=1
```

Stage1 通过：只有 `loss_action`；白名单是 ActionDiT / proprio / GoalEncoder；`img_len=0` 不炸。  
Stage2 通过：bridge 丢弃 GoalEncoder；同时出现 `loss_video/loss_action/loss_pose`。20 step 仍在 warmup 内，瞬时 LR 小于 peak `1e-4` 是预期。

Smoke 通过后再开正式 Stage1 10k / Stage2 10 epoch 和 LIBERO 评测。

## 已知风险

- V3 软瓶颈：92 个无 `L_pose` 的 context token 可能漏外观捷径。
- synthetic KV 必须对齐 24×128、RMSNorm、RoPE；`time_value=3.0`。
- Stage1 的 `2e-4` 仍可能让已初始化的 ActionDiT 漂移，但比 `5e-4` 保守。
- `state[t+16]` 是 chunk 终点，不是任务终点。越界时 clamp 为本条末帧；Stage1 仍注入，Stage2 只 mask `L_pose`。
- Stage2 显存：100 latent × 每层三段 attention，并且仍训练 FLUX。
- empty-image GPU 路径用 8 卡 20-step smoke 验证。
- Stage1 显存：autograd 穿过冻结 FLUX 时必须按层 checkpoint；只 checkpoint mixed attn 不够。
- Stage2 视频解码：启动脚本会把 venv 的 `nvidia/npp/lib` 加进 `LD_LIBRARY_PATH`，否则 dataloader worker 找不到 `libnppicc.so.11`，torchcodec 无法加载。

## 评测结果

两个版本都在 LIBERO 与 LIBERO-Plus 上跑过全量评测，结论相反，**不要把 `e81335b`
当成最终版**——它的 commit message 里那句 "(evaluated version)" 指的是 v1，而 v1 低于
baseline。进入对外表格的是 v2。

| 版本 | commit | tag | LIBERO in-dist | LIBERO-Plus | vs baseline |
|:---|:---|:---|---:|---:|---:|
| baseline | — | — | 98.1 | 83.01 | — |
| v1 硬防火墙 | `e81335b` | `goal-pose-prior-v1-20260825` | 97.4 | 79.73 | **−3.28** |
| v2 门控上下文 | `e08402c` | `goal-pose-prior-v2-20260825` | 98.4 | 84.45 | **+1.44** |

LIBERO-Plus 全量 10,030 个任务，baseline 与 ours 在同一批 task id 上评测，逐任务配对。

### 分轴（n 加权，跨四个 suite 合并）

| 轴 | n | baseline | v1 | v2 |
|:---|---:|---:|---:|---:|
| Camera      | 1599 | 82.93 | 68.29 (−14.63) | 83.86 (+0.94) |
| Noise       | 1601 | 97.31 | 92.19 (−5.12)  | 95.63 (−1.69) |
| Light       | 1142 | 98.42 | 96.06 (−2.36)  | 96.76 (−1.66) |
| Background  | 1076 | 89.03 | 91.91 (+2.88)  | 90.24 (+1.21) |
| Robot       | 1550 | 48.58 | 58.65 (+10.06) | **62.06 (+13.48)** |
| Layout      | 1525 | 78.49 | 79.34 (+0.85)  | 83.48 (+4.98) |
| Language    | 1537 | 91.74 | 79.64 (−12.10) | 83.73 (−8.00) |
| **总计**    | 10030 | 83.01 | 79.73 (−3.28) | **84.45 (+1.44)** |
| **w/o Language** | 8493 | 81.43 | 79.75 (−1.68) | **84.58 (+3.14)** |

### 两版之间改了什么

v1 把原始图像 token 对 Action Expert 完全屏蔽（硬防火墙），AE 只能通过 8 个 pose token
看世界。结果是 Camera 掉了 14.63、Language 掉了 12.10——切断得太彻底，AE 失去了它本来
就需要的参照信息。

v2 把参考图像还给 AE，改为对 92 个无监督 context latent 做门控（8 个 pose token 全程保留，
不参与 dropout / blackout）。Camera 从 −14.63 回到 +0.94，总体转正。

轴的分配按 LIBERO-Plus 各 suite 内连续的 task-id 区间判定，不是从任务描述正则匹配——
描述里的措辞会把 Language 改写误判进 Background。

### 复现

训练见 `scripts/flux2/run_train_flux2_klein_goal_prior_stage{1,2}.sh`，
评测见 `scripts/flux2/run_eval_flux2_libero_plus.sh`。
