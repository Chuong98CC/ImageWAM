# FLUX.2 architecture in ImageWAM

A reference for the `flux2` stack, in two parts:

- **Part 1 — Upstream FLUX.2 Klein image model.** What the DiT is, and what every block inside it does.
- **Part 2 — LIT goal-pose prior model.** What this fork adds on top: the joint video/action attention,
  and the two-stage goal-pose prior.

Each part opens with a **macro block diagram** of the whole model, then gives a **micro block diagram**
for each macro block.

This doc describes structure and data flow only. For *why* the goal-pose-prior / LIT mechanisms exist
and what they cost, see `goal_pose_prior_flux2.md` (operations manual) and `CHANGELOG_v2.md`
(ablations and the bugs behind each decision).

## Where the code lives

| Piece | File | Notes |
| --- | --- | --- |
| Upstream DiT | `third_party/flux2/src/flux2/model.py` | Not vendored — clone separately, see `dependencies.md` |
| VAE | `third_party/flux2/src/flux2/autoencoder.py` | |
| Text encoder | `third_party/flux2/src/flux2/text_encoder.py` | Qwen3, `OUTPUT_LAYERS_QWEN3` |
| Video expert wrapper | `src/imagewam/models/backbones/flux2_video_expert.py` | 182 lines |
| Action expert | `src/imagewam/models/backbones/action_dit_flux2.py` | 288 lines |
| Joint attention | `src/imagewam/models/backbones/mot.py` | `_forward_flux2` and friends |
| LIT modules | `src/imagewam/models/backbones/goal_pose_prior.py` | 784 lines, self-contained |
| Assembly | `src/imagewam/models/backbones/imagewam.py` | `from_flux2_klein_pretrained`, loss paths |

```text
                     upstream (third_party/flux2)               this repo (src/imagewam)
  ┌───────────────────────────────────────────────┐   ┌──────────────────────────────────────┐
  │  flux2.model.Flux2                            │   │  Flux2VideoExpert   (video expert)   │
  │    double_blocks[5] + single_blocks[20]       │◄──┤    .pre_dit / .post_dit              │
  │  flux2.autoencoder.AutoEncoder                │   │  ActionDiTFlux2     (action expert)  │
  │  flux2.text_encoder (Qwen3)                   │   │  MoT                (joint attention)│
  └───────────────────────────────────────────────┘   │  goal_pose_prior    (LIT modules)    │
                                                      │  ImageWAM           (loss / infer)   │
                                                      └──────────────────────────────────────┘
```

### A note on the name "video expert"

The `MoT` has exactly two expert slots, and ImageWAM's vocabulary names them `"video"` and `"action"`
(`mixtures={"video": …, "action": …}`). Every backbone in the repo fills the first slot under that same
name — `wan_video_expert.py`, `omnigen2_video_expert.py`, `ovis_u1_video_expert.py`,
`dim_video_expert.py`, `flux2_video_expert.py` — and the MoT only requires the filler to agree on
`num_layers` / `num_heads` / `num_kv_heads` / `attn_head_dim` / `block_protocol`. It is a structural
role, not a claim about the backbone.

**The FLUX.2 video expert is a pure image model.** `flux2/autoencoder.py` contains no temporal code at
all — no 3D convolutions, no causal time padding, no frame compression. The one place this repo handles
temporal alignment is `_compute_video_loss_per_sample`, which reads `vae.temporal_downsample_factor`;
that lives in the DIM loss path (`imagewam.py:3032`) and is never called for FLUX.2. All four FLUX.2
loss paths use a plain `mse_loss(...).flatten(1).mean(dim=1)` with no temporal grouping.

"Video" describes the **task**, not the architecture. The data is video and the target is a future
frame: `build_inputs_flux2` takes `video[:, :, 0]` as the reference and `video[:, :, -1]` as the target.
So the model does one-step video prediction with a backbone that has no notion of time — two stills,
encoded independently, concatenated into a single sequence with no frame-to-frame attention. The name
would be less confusing as `visual_expert`, but it is baked into the `mixtures` dict and into every
stack's filename.

---

# Part 1 — Upstream FLUX.2 Klein image model

## 1.0 Macro block diagram

```text
   image [B,3,H,W]                       prompt
        │                                │
        ▼                                ▼
   ┌─────────┐                    ┌──────────────┐
   │   VAE   │                    │    Qwen3     │
   │  encode │                    │    4B/8B     │
   └────┬────┘                    └──────┬───────┘
        │                                │
        ▼                                ▼
 tokens [B,256,128]              txt [B,L,7680]
        │                                │
      img_in                          txt_in
     (128→3072)                     (7680→3072)
        │                                │
        └────────────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │ DoubleStreamBlock×5 │
              └──────────┬──────────┘
                         │
                         ▼
                 concat [txt | img]
                         │
              ┌──────────┴──────────┐
              │SingleStreamBlock×20 │
              └──────────┬──────────┘
                         │
                         ▼
                   drop txt tokens
                         │
                         ▼
                   ┌───────────┐
                   │ LastLayer │
                   └─────┬─────┘
                         │
                         ▼
               velocity [B,256,128]
```

Two global signals are injected into **every** block and are not drawn as stages:

| Signal | Path | Consumed by |
| --- | --- | --- |
| timestep | `timestep_embedding(t,256)` → `time_in` → `vec` → `Modulation` | every block (§1.3) |
| token ids | 4-axis id vector → `EmbedND` → RoPE freqs | every attention (§1.4) |

The variants differ only in these numbers (`model.py:11`, `model.py:25`, `model.py:39`):

| | Klein 4B | Klein 9B | full FLUX.2 |
| --- | ---: | ---: | ---: |
| residual stream | 3072 | 4096 | 6144 |
| heads × head_dim | 24 × 128 | 32 × 128 | 48 × 128 |
| double blocks | 5 | 8 | 8 |
| single blocks | 20 | 24 | 48 |
| latent channels | 128 | 128 | 128 |
| text context | 7680 | 12288 | 15360 |
| mlp_ratio | 3.0 | 3.0 | 3.0 |
| guidance embed | no | no | yes |

Only the 4B and 9B Klein variants are accepted (`flux2_video_expert.py:49`).

### 1.0.1 Parameter counts (Klein 4B)

Measured, not estimated: the `Flux2(Klein4BParams())` build on the meta device reproduces the shipped
safetensors tensor-for-tensor, and both agree to the parameter.

| Component | Params | Share | Role |
| --- | ---: | ---: | --- |
| **Transformer** (`flux-2-klein-base-4b.safetensors`) | **3,875,544,576** | 48.6% | the "4B" |
| **Text encoder** (`Qwen3ForCausalLM`) | **4,022,468,096** | 50.4% | LLM, hidden 2560 |
| **VAE** (`AutoencoderKLFlux2`) | **84,046,375** | 1.1% | image ↔ latent |
| **Total** | **7,982,059,047** | | |

The "4B" name counts the **transformer alone** — the released pipeline is ~8B. Note also that
`model_index.json` lists only `scheduler, text_encoder, tokenizer, transformer, vae`: there is **no
vision tower**. The text encoder is `Qwen3ForCausalLM`, a text-only LLM, and images enter through the
84M VAE. The `4B/8B` label in the diagram above is the text encoder's own size, not a vision model's.

Inside the transformer, the single blocks dominate:

| Group | Params | Share of transformer |
| --- | ---: | ---: |
| `single_blocks` (×20) | 2,453,672,960 | 63.3% |
| `double_blocks` (×5) | 1,226,836,480 | 31.7% |
| `double_stream_modulation_img` | 56,623,104 | 1.5% |
| `double_stream_modulation_txt` | 56,623,104 | 1.5% |
| `single_stream_modulation` | 28,311,552 | 0.7% |
| `txt_in` | 23,592,960 | 0.6% |
| `final_layer` | 19,267,584 | 0.5% |
| `time_in` | 10,223,616 | 0.3% |
| `img_in` | 393,216 | 0.01% |

A single block is 122.7M of that: `linear1` is (27648, 3072) — a ~9× combined attention + both-MLP
expansion, not the `mlp_ratio` you would guess from the config — and `linear2` is (3072, 12288).

## 1.1 Micro — VAE (`autoencoder.py:271`)

4-level conv autoencoder, `z_channels=32`, `ch_mult=[1,2,4,4]`, `ps=[2,2]`. `encode` takes the mean of
the moments and then **patchifies 2×2**, so the DiT sees 128 channels at 1/16 resolution:

```text
  256×256×3 image
        │
   Encoder  (4 × stride-2 downsampling, ch_mult [1,2,4,4])
        │
   moments: 32×32×64  ──► take mean ──► 32×32×32
        │
   patchify ps=[2,2]   "... c (i pi) (j pj) -> ... (c pi pj) i j"
        │
   latent: 16×16×128   ──► BatchNorm2d(128, affine=False) normalisation
        │
   pack_latents        "b c h w -> b (h w) c"
        │
   tokens: [B, 256, 128]
        │
   img_in: Linear(128 → 3072)
```

Spatial dims must be multiples of 16 (`imagewam.py:1838`). `decode` reverses this and the VAE is kept
in `eval()` mode (`imagewam.py:686`).

## 1.2 Micro — text encoder (Qwen3)

Hidden states of `OUTPUT_LAYERS_QWEN3 = [9, 18, 27]` (`text_encoder.py:27`) are stacked:

```text
  prompt ──► Qwen3-4B (hidden 2560) ──► layers [9, 18, 27] ──► stack
                                                                 │
                                              [B, L, 3 × 2560 = 7680]
                                                                 │
                                                    txt_in: Linear(7680 → 3072)
```

3 × 2560 = 7680 = `context_in_dim` for the 4B variant; the 9B uses Qwen3-8B (3 × 4096 = 12288).
In training these are usually read from the precomputed cache rather than run online
(`imagewam.py:1865`); `load_text_encoder: false` in the model config is the default.

## 1.3 Micro — conditioning (`Modulation`)

One `vec`, three `Modulation` modules turning it into `(shift, scale, gate)` triples:

```text
  timestep
    │
    ▼
  timestep_embedding(t, 256)
    │
    ▼
  MLPEmbedder(256 → 3072)
    │
    ▼
  vec
    │
    ├─► double_stream_modulation_img ──► (shift, scale, gate) × 2 ──► img attn + img mlp
    │
    ├─► double_stream_modulation_txt ──► (shift, scale, gate) × 2 ──► txt attn + txt mlp
    │
    └─► single_stream_modulation ──────► (shift, scale, gate)     ──► attn + mlp (fused)
```

`Modulation.forward` (`model.py:407`) chunks its output into `multiplier` pieces and returns
`(first_triple, rest_if_double)` — which is why callers write `single_mod, _ = ...`.

## 1.4 Micro — positional encoding (`EmbedND`)

`EmbedND` over `axes_dim = [32, 32, 32, 32]` (sums to head_dim 128), theta 2000. IDs are 4-vectors and
the four axes are used **disjointly by modality**, so tokens of different kinds can never collide:

| token kind | axis 0 (t) | axis 1 (h) | axis 2 (w) | axis 3 (seq) | built by |
| --- | --- | --- | --- | --- | --- |
| text | 0 | 0 | 0 | `arange(L)` | `flux2_video_expert.py:63` |
| image | `time_value` | row | col | 0 | `flux2_video_expert.py:75` |
| action | 2.0 | `arange(T)` | 0 | 0 | `action_dit_flux2.py:237` |
| synthetic (LIT) | 3.0 | `arange(K)` | 0 | 0 | `goal_pose_prior.py:617` |

`time_value` is how the repo separates reference from target: **ref = 10.0, target = 0.0**
(`imagewam.py:2106`, `imagewam.py:2119`). Upstream instead gives ref tokens a fixed timestep and
blends per-position modulations (`_blend_double_mods`, `model.py:346`); this wrapper does **not** use
that path — ref and target share one timestep and are told apart positionally.

## 1.5 Micro — `DoubleStreamBlock` (`model.py:524`)

Separate img/txt weights, joint attention over `[txt | img]`:

```text
    txt                             img
    │                               │
    txt_norm1 ─(1+scale)·x+shift    img_norm1 ─(1+scale)·x+shift
    │                               │
    txt_attn.qkv                    img_attn.qkv
    │                               │
    QKNorm                          QKNorm
    │                               │
    └───────────────┬───────────────┘
                    ▼
        cat([txt_q, img_q]) ◄── sequence order: txt first, then img
                    │
        apply_rope(pe_full) ◄── pe_full = cat([txt_pe, img_pe])
                    │
                SDPA over the union
                    │
    ┌───────────────┴───────────────┐
    ▼                               ▼
    txt_attn                        img_attn
    │                               │
    txt_attn.proj                   img_attn.proj
    │                               │
    txt += gate₁ · out              img += gate₁ · out
    │                               │
    txt_mlp: Linear(3072→2m)        img_mlp: Linear(3072→2m)
    → SiLUActivation                → SiLUActivation
    → Linear(m→3072)                → Linear(m→3072)
    │                               │
    txt += gate₂ · out              img += gate₂ · out
```

`img_attn` and `txt_attn` are two separate `SelfAttention` modules (`model.py:375`) — one weight set
per stream. Each has its own `qkv` (`Linear(3072 → 9216, bias=False)`, one fused matmul producing
Q/K/V, split by an einops reshape) and `proj` (`Linear(3072 → 3072, bias=False)`). Only the SDPA
between them is joint.

**`SDPA`** throughout this doc is PyTorch's fused scaled-dot-product attention,
`F.scaled_dot_product_attention` (`mot.py:195`) — that is, `softmax(QKᵀ/√d)·V`, dispatched at runtime
to a backend (FlashAttention, memory-efficient, or math). `mot_force_flash_attention` pins it to
FlashAttention.

## 1.6 Micro — `SingleStreamBlock` (`model.py:437`)

One stream, fused matmuls:

```text
                     x
                     │
              pre_norm ──(1+scale)·x+shift
                     │
   linear1: Linear(3072 → 3·3072 + 2m)          m = 3072 · mlp_ratio = 9216
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   qkv [3·3072]              mlp [2m = 18432]
        │                         │
   QKNorm + apply_rope      SiLUActivation          ◄── GATED: chunks in half
        │                    (-> [m = 9216])
      SDPA                        │
        └────────────┬────────────┘
                     ▼
        linear2: Linear(3072 + m → 3072)
                     │
            x += gate · out
```

**The non-obvious part**: `SiLUActivation` (`model.py:390`) is not elementwise — it chunks its input
in half and returns `silu(x1) * x2`. That is what consumes the `mlp_mult_factor = 2` width and makes
`linear2`'s declared input (`hidden + m`) match the concatenation. Reading the shapes without knowing
this makes `linear2` look like it is sized wrong by exactly one `mlp_hidden_dim`.

**The two branches run in parallel, not in sequence.** Both split off the same `x_mod` and do not meet
until `linear2` concatenates them along the feature dim (`model.py:483`) — so the MLP never sees the
attention output. One pre-norm, one modulation triple, one gate, and **one** residual add.

This is the opposite of the double block in §1.5, which has the conventional shape: two norms, two
residuals, and the MLP reading the post-attention stream (`model.py:626`). FLUX.2 carries both patterns
in the same model, and applying the "attn then MLP" mental model to a single block will misread the
diagram above.

The fusion buys two matmuls — one `Linear(3072 → 3·3072 + 2m)` instead of separate qkv and MLP
projections, one `Linear(3072 + m → 3072)` instead of separate output projections — and it is inherited
from FLUX.1. A practical consequence: `linear1` carries both projections, so LoRA cannot target them
separately in a single block; the `linear1` entry in `target_suffixes` covers both.

## 1.7 Micro — `LastLayer` (`model.py:415`)

```text
   img tokens
        │
   norm_final (no affine)
        │
   (1 + scale) · x + shift          ◄── adaLN from vec
        │
   linear: Linear(3072 → 128, no bias)
        │
   velocity [B, N, 128]
```

## Part 1 — things that are easy to get wrong

- **`SiLUActivation` is gated, not elementwise.** It halves the width; §1.6 shape arithmetic looks off
  by one `mlp_hidden_dim` until you account for it.
- **Single blocks are parallel; double blocks are sequential.** In a `SingleStreamBlock` the attention
  and MLP branches both read the same pre-norm activation, never see each other, and share one residual
  add. In a `DoubleStreamBlock` they are sequential with two. The "attn then MLP" mental model applies
  to the double block only (§1.6).
- **The four RoPE axes are modality-disjoint.** Text uses axis 3, image uses 0–2, action and synthetic
  tokens use axis 1 with a distinct time value. Ref vs target is distinguished by the image
  `time_value` (10.0 vs 0.0), *not* by the upstream ref-timestep blending, which this wrapper never calls.
- **Latent channels are 128, not 32.** The VAE's `z_channels` is 32; the 2×2 patchify is what makes it
  128 by the time the DiT sees it.

---

# Part 2 — LIT goal-pose prior model

The prior is trained in **two stages**, and they exist for different reasons:

- **Stage 1 pretrains the action expert.** No images at all. The action expert learns to map
  `[txt | pose | action]` → action, conditioning on an oracle goal pose. The action expert, the MoT
  that drives it, and the goal-token interface are all introduced here, because this is where they
  first exist.
- **Stage 2 integrates vision.** The reference image comes back, and a `SemanticVisualAggregator`
  reads the FLUX.2 video features and steers the action expert through synthetic K/V.

Everything in Part 1 is frozen or reused as-is by both stages.

---

## Stage 1 — pretraining the action expert

### 2.1 Macro — stage 1

```text
         prompt               goal_pose               noisy action
            │                     │                         │
            ▼                     ▼                         ▼
      ┌── text ───┐     ┌───── stage 1 ─────┐     ┌───── action ──────┐
      │   Qwen3   │     │  GoalPoseEncoder  │     │   action_expert   │
      │ 9, 18, 27 │     │   8-D → 8 × 3072  │     │      .pre_dit     │   §2.2  ActionDiTFlux2 — the action expert
      └─────┬─────┘     └─────────┬─────────┘     └─────────┬─────────┘
     txt [B,L,7680]               │                action [B,16,1024]
            │                     │                         │
┌──────── video ────────┐         │                         │
│  video_expert.pre_dit │         │                         │
│   (FROZEN — Part 1)   │         │                         │
└───────────┬───────────┘         │                         │
            │                     │                         │
     txt [B,L,3072]               │                         │
            │                     │                         │
            │                     │                         │
   txt = [txt | pose]  ◄──────────┘                         │   §2.6  GoalPoseEncoder output
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    ▼
                  ┌────────────── loop ───────────────┐
                  │          MoT — 25 layers          │
                  │       video block  (frozen)       │   §2.4  MoT joint attention (the loop)
                  │      action block (trainable)     │   §2.5  action K/V layout + mask (in SDPA)
                  └─────────────────┬─────────────────┘
                                    │
                                    ▼
                         action_expert.post_dit     §2.2  same module — the decoder half
                                    │
                                    ▼
                          pred_action [B,16,7]
                                    │
                                    ▼
                       L_action   ← the only loss
```

Where each Stage 1 section lives in that diagram:

| Section | Appears as |
| --- | --- |
| §2.2 `ActionDiTFlux2` | **three** places, one module: the `action_expert .pre_dit` box, the `action block` row inside the MoT box, and `action_expert.post_dit` |
| §2.3 `pre_dit` / `post_dit` | every `.pre_dit` box, every `post_dit` line — they are the halves the loop sits between |
| §2.4 MoT joint attention | the `loop` box — the whole 25-layer stack |
| §2.5 action K/V layout + mask | inside that loop, at the `SDPA + mask` step — not a box of its own |
| §2.6 `GoalPoseEncoder` | the `stage 1` box; its output merges into `txt = [txt | pose]` |
| the video expert (Part 1, §1.5–1.6) | the `video_expert.pre_dit` box and the `video block` row inside the MoT box |

The image stream is absent by construction: `x` is a zero-length tensor, and the reference slot holds
either nothing or a constant learnable stand-in (`stage1_null_image_tokens: 32`). The video expert
still runs all 25 layers — that is what gives the goal tokens their context — it is simply frozen.

`goal_pose` comes from the dataset as `proprio[-1]` clamped to the episode end; with `vision_free: true`
no frame is ever decoded.

What is trainable (`apply_trainable_policy`, `imagewam.py:5364`):

| module | stage 1 |
| --- | --- |
| action expert (incl. `action_encoder`) | **trainable** |
| `goal_pose_encoder` | **trainable** |
| video expert — all 25 blocks | frozen, forced `.eval()` |

Stage 2 and the baseline use the same two gates with different whitelists, and LoRA is a third
path reachable only from the baseline — Part 5 covers all of them.

### 2.1.1 The instantiated FLUX.2 stack

Every module `ImageWAM` actually materialises on the 4B Klein path, measured by building each class on
the meta device with the shipped config (`configs/model/imagewam_flux2_klein_4b_goal_prior_stage2.yaml`):

| Module | Params | Share of stack | Stage 2 status |
| --- | ---: | ---: | --- |
| Video Expert — the Part 1 transformer, unmodified | 3,875,544,576 | 81.0% | trainable (LoRA-able, §5.2) |
| Action Expert (`ActionDiTFlux2`) | 642,012,416 | 13.4% | trainable |
| `SemanticVisualAggregator` | 165,439,365 | 3.5% | trainable |
| VAE (`AutoencoderKLFlux2`) | 84,046,375 | 1.8% | frozen, `.eval()` |
| `GoalPoseEncoder` | 12,874,752 | 0.27% | stage 1 only |
| `semantic_visual_pose_decoder` (`GoalPoseDecoder`) | 3,413,000 | 0.07% | trainable |
| `semantic_visual_pose_norm` (`LayerNorm(768)`) | 1,536 | 0.00003% | trainable |
| `ProprioEncoder` (`Linear(8 → 7680)`) | 61,448 | 0.001% | trainable |
| **In-process total** | **4,783,393,468** | | |

Names are the checkpoint payload keys; `GoalPoseDecoder` *is* `semantic_visual_pose_decoder`
(`imagewam.py:217`), not a separate module, and the two are saved under the latter
(`imagewam.py:5246`).

The Qwen3-4B text encoder (4.02B, §1.0.1) is **not** in this table because it is never instantiated:
every flux2 config sets `load_text_encoder: false` and reads precomputed embeddings from the Qwen cache
(`imagewam.py:701`). Counting it, the nominal pipeline is ~8.8B.

Two things this makes concrete:

- The Action Expert is **642M, not a small head**. Its residual stream is 1024-wide but its attention
  geometry is 24 × 128 = 3072 to match the video expert, so the attention projections dominate its
  parameter count. It is 6× the aggregator and the second-largest module here.
- The aggregator is **165M** — 3.5% of the stack. At LoRA rank 16 the FLUX adapters are 23.6M
  (§5.2), so the LIT modules beside them are 7× larger and train in full.

Loss: `1.0 · L_action`, nothing else. No image loss and no pose loss exist in this stage.

### 2.2 Micro — `ActionDiTFlux2`, the expert being pretrained (`action_dit_flux2.py:146`)

A structural clone of the Part 1 blocks with a **1024-wide residual stream but unchanged attention
geometry**:

| | video expert | action expert |
| --- | ---: | ---: |
| residual stream | 3072 | 1024 |
| heads × head_dim | 24 × 128 | 24 × 128 |
| q/k/v width | 3072 | 3072 |
| linear1 out (single) | 9216 + 18432 | 9216 + 8192 |
| linear2 in (single) | 3072 + 9216 | 3072 + 4096 |
| layers | 5 + 20 | 5 + 20 |
| input | 128-ch latent | 7-D action |

Because q/k/v width matches, concatenation is free; because the residual width differs, everything
else is separate weights. It carries its own `action_encoder`, `time_in`, and modulation modules
(`action_dit_flux2.py:178`), and a `Flux2ActionHead` doing adaLN + `Linear(1024 → action_dim)`
(`action_dit_flux2.py:134`).

Weights are initialised from the FLUX.2 checkpoint by `scripts/flux2/preprocess_action_dit_flux2.py`,
which linearly interpolates mismatched shapes with a `sqrt(src/dst)` alpha scaling to preserve variance.

### 2.3 Micro — `pre_dit` / `post_dit`: the split around the DiT

Both experts expose the same two-method interface, and the MoT loop runs **between** them:

```text
   video  ──► Flux2VideoExpert.pre_dit  ──┐                    ┌── post_dit ──► pred_video
                                          ├──► MoT loop (25) ──┤
   action ──► ActionDiTFlux2.pre_dit    ──┘                    └── post_dit ──► pred_action
```

**`pre_dit` is everything before the transformer blocks.** For the video expert
(`flux2_video_expert.py:100`) that is: concatenating ref before target in the image stream, `img_in` and
`txt_in`, the `EmbedND` positional embeddings for both streams, `time_in` for the timestep, and the
three `Modulation` triples. For the action expert (`action_dit_flux2.py:249`) it is `action_encoder`
(`action_dim → 1024`), `time_in`, and its two modulation triples. Neither does any cross-expert
computation — `pre_dit` is a pure per-stream embedding step, and it ignores the `context` argument it
is given (`action_dit_flux2.py:256`).

**`post_dit` is everything after.** The video expert slices the target tokens back out —
`img[:, cond_len : cond_len+target_len]`, which drops the ref tokens — and applies `final_layer`
(`flux2_video_expert.py:177`). The action expert applies `Flux2ActionHead`: adaLN from `vec`, then
`Linear(1024 → action_dim)` (`action_dit_flux2.py:287`).

**Why the split exists.** `pre_dit` returns more than tokens — it returns a *state dict* carrying the
positional frequencies, the modulation vectors, and the length bookkeeping (`txt_len`, `cond_len`,
`target_len`). The MoT layers consume that state, and `post_dit` needs it again at the far end to know
which slice of the output is the prediction. That is why the three pieces are strictly ordered —
`pre_dit` for both experts, then the 25-layer loop, then `post_dit` for both — and why the loop can be
dropped in between without either expert noticing.

The name reads "pre/post the **DiT blocks**", not "pre/post the diffusion".

### 2.4 Micro — MoT joint attention (`mot.py:22`)

`MoT` holds `mixtures = {"video": Flux2VideoExpert, "action": ActionDiTFlux2}` and validates that both
agree on `num_layers`, `num_heads`, `num_kv_heads`, `attn_head_dim`, `block_protocol` (`mot.py:53`).

Because both experts emit q/k/v of width 24 × 128 = 3072, they concatenate **with no projection**.
Only the residual streams differ in width:

```text
  layer_idx i of 25   (i < 5 → double blocks,  i ≥ 5 → single blocks)

     video expert                                    action expert
  ┌──────────────────┐                            ┌──────────────────┐
  │   video block i  │                            │  action block i  │
  │  3072-wide       │                            │  1024-wide       │
  │  own weights     │                            │  own weights     │
  └────────┬─────────┘                            └────────┬─────────┘
           │ q,k,v  [B, L_v, 3072]                         │ q,k,v  [B, L_a, 3072]
           └───────────────────────┬───────────────────────┘
                                   ▼
                       cat along sequence (dim=1)
                                   │
                             SDPA + mask
                                   │
                           split back at L_v
           ┌───────────────────────┴───────────────────────┐
           ▼                                               ▼
     video attn out                                  action attn out
     (residual + MLP in video weights)               (residual + MLP in action weights)
```

**This is drawn once because it holds for both block types.** The concatenate–attend–split core is
byte-for-byte the same idea in the double loop (`mot.py:626`) and the single loop (`mot.py:670`): each
expert projects q/k/v in its own weights, the two are concatenated along the sequence, one SDPA runs,
and the result is split back at `L_v`.

It is safe to reuse one mask for both because `L_v` is the same in either case: the double block's
`_prepare_qkv` concatenates `[txt | img]` internally, and the single block's stream is already
`cat([txt, img])`. So the total sequence length is `txt + ref + target + action` throughout, which is
why `_build_mot_attention_mask_flux2` can return one mask and clone it for `single`.

What differs is what each side does *around* the shared core:

| | double block (`i < 5`) | single block (`i ≥ 5`) |
| --- | --- | --- |
| video q/k/v produced by | `v_block._prepare_qkv(img, txt, …)` | `_flux2_video_single_io(v_block, stream, …)` (`mot.py:543`) |
| video output path | `_apply_residuals` — splits the attention output back into txt and img, **two** residuals | `_out` — one residual over the whole stream |
| video MLP timing | **after** the attention; reads the post-attention stream | fused into `linear1`, computed *alongside* q/k/v **before** the attention |
| action side | `a_block.prepare_qkv` → `apply_post` | same two methods (`action_dit_flux2.py:106`) |

The action side is uniform: both `SlimFlux2DoubleBlock` and `SlimFlux2SingleBlock` expose
`prepare_qkv` / `apply_post`, so the MoT loop calls them identically and only their internals differ
(§1.5 vs §1.6 for what those internals are).

The seam is upstream's *private* split, not its public API: `DoubleStreamBlock._prepare_qkv` /
`_apply_residuals` (`model.py:569`, `model.py:614`) and `SingleStreamBlock._qkv` / `_out`
(`model.py:468`, `model.py:482`). Upstream's own `forward_kv_extract` / `forward_kv_cached`
(`model.py:170`, `model.py:267`) are a decoy — causal, single-stream, and for reference images; the
MoT calls none of them.

**Stage 2 breaks the joint call.** Everything above describes `_forward_flux2`, which baseline and
stage 1 use. `_forward_flux2_stage2` (`mot.py:775`) runs *two* attentions per layer instead: the
video stream attends only itself (`mot.py:863`) and the action stream attends
`[txt | ref? | syn | action]` (`mot.py:891`). The video expert stops reading action tokens — which
is what lets §3.1 prefill the video half alone and cache it.

The per-layer function is wrapped in `torch.utils.checkpoint` when `mot_checkpoint_mixed_attn` is set
and the module is training (`mot.py:557`) — needed because stage 1 runs autograd through all 25 frozen
video layers to reach the goal encoder.

**The `SDPA` box is the forward path only.** When `_mixed_attention` is called with
`return_attn_probs=True` it does not use SDPA — it computes `softmax(q·kᵀ·D^-0.5)` explicitly with
matmuls, because SDPA does not return probabilities (`mot.py:151`). That branch is only taken when
attention-capture diagnostics are active (`mot.py:1112`).

**Pairing constraint.** The action expert must have exactly the same layer counts as the video expert,
because the loop pairs them by index (`mot.py:598`, `mot.py:655`). `from_flux2_klein_pretrained`
enforces this by overwriting whatever the config said (`imagewam.py:661`):

```python
expected_action_shape = {
    "num_heads": 24, "attn_head_dim": 128,
    "num_layers_double": 5, "num_layers_single": 20,
}
# hidden_dim is the ONE thing left free (1024)
```

### 2.5 Micro — the action K/V layout and its baseline mask (`imagewam.py:2312`)

`_build_mot_attention_mask_flux2` produces a block-structured **bidirectional** keep-mask (there is no
causal masking here, despite `causal_attn_fn` existing in the upstream ref-cache path). The same tensor
is used for both `double_joint` and `single`:

```text
           key ►   text     ref     target   action
                ┌────────┬────────┬────────┬────────┐
           text │   ✓    │   ✓    │   ·    │   ·    │
                ├────────┼────────┼────────┼────────┤
            ref │   ✓    │   ✓    │   ·    │   ·    │
                ├────────┼────────┼────────┼────────┤
         target │   ✓    │   ✓    │   ✓    │   ·    │
                ├────────┼────────┼────────┼────────┤
         action │   ✓    │   ✓    │   ·    │   ✓    │
                └────────┴────────┴────────┴────────┘
```

The four rows are the four row-blocks the code assigns with five separate rules (`imagewam.py:2329`):
text rows read `txt + ref`; ref rows read `txt + ref`; target rows read `txt + ref + target`; action rows
read `txt + ref + action`. Nothing in the mask distinguishes `txt` from `ref` *as keys* — every rule
that admits one admits the other — which is what makes the first two rows identical and earns them the
name **stable prefix**: they are the only blocks that never read anything noisy.

Two consequences worth reading off the table:

- **The target block is invisible to the action block**, and vice versa. The action block therefore sees
  no predicted future; that is the separation the stage-2 firewall later widens.
- **Stage 1 passes `target_len=0`**, which deletes the `target` row and column entirely rather than
  masking them. The action row then reads `txt + ref + action`, and if the reference is also absent
  (`cond_len=0`, no null tokens), `txt + action` — stage 1's `[txt | pose | action]`.

This is the layout the baseline (non-prior) path uses with every block populated.
`text_attention_mask` additionally zeroes padding columns of the txt block (`imagewam.py:2343`).

### 2.6 Micro — `GoalPoseEncoder` (`goal_pose_prior.py:85`)

```text
   goal_pose [B, 8]
        │
   Linear(8 → 512) ─► GELU ─► Linear(512 → 512) ─► GELU ─► Linear(512 → 8·3072)
        │
   reshape [B, 8, 3072]
        │
   appended to the text stream, after txt_in
```

The 8 output tokens use the **FLUX hidden interface (3072)**, not the 7680-D Qwen one. That matters:
they are appended *after* `txt_in`, by `_append_goal_tokens_to_flux2_pre` (`imagewam.py:2060`), which
also extends `txt_pe` and `text_mask` to match the new length. The goal tokens therefore have the same
positional treatment as text tokens — axis 3 of the 4-axis id, not the image axes.

---

## Stage 2 — integrating vision

### 2.7 Macro — stage 2

Same spine as stage 1, with the image rail restored and the LIT block spliced into the layer loop. The
loop box is drawn open: inside every one of the 25 layers the order is video block →
`SemanticVisualAggregator` → action block, and the aggregator never touches the video stream itself.
§2.8 walks that interior; the sections after it take the three boxes apart one at a time.

```text
        ref image              prompt                 noisy action
            │                     │                         │
            ▼                     ▼                         ▼
      ┌─── VAE ───┐        ┌─── text ────┐           action [B,16,7]
      │   encode  │        │    Qwen3    │                  │
      └─────┬─────┘        │  9, 18, 27  │                  ▼
            │              └──────┬──────┘        ┌───── action ──────┐
            ▼                     ▼               │   action_expert   │
       ref tokens          txt [B,L,7680]         │      .pre_dit     │
            │                     │               └─────────┬─────────┘
            └──────────┬──────────┘                         │
                       ▼                                    │
             ┌────── video ──────┐                          │
             │    video_expert   │                          │
             │      .pre_dit     │                          │
             └─────────┬─────────┘                          │
                       │                                    │
                       └─────────────────┬──────────────────┘
                                         ▼
                 ┌───────────────────── loop ─────────────────────┐   §2.8  how the loop is wired
                 │                MoT — 25 layers                 │
                 │              5 double + 20 single              │
                 │    ┌──────────────────────────────────────┐    │   §1.5–1.6  frozen in stage 1, trainable now
                 │    │            video block i             │    │
                 │    │          txt + img features          │    │
                 │    └──────────────────┬───────────────────┘    │
                 │                       ▼                        │
                 │    ┌──────────────────────────────────────┐    │   §2.9  the LIT block — 100 learnable queries
                 │    │       SemanticVisualAggregator       │    │   §2.10  group = layer_idx // 5
                 │    │  100 latents × 768  ·  group i // 5  │    │
                 │    └──────────────────┬───────────────────┘    │
                 │             syn K/V  ·  100 × 3072             │   §2.11  projected + RoPE'd
                 │                       ▼                        │
                 │    ┌──────────────────────────────────────┐    │   §2.12  cold-start gate + firewall
                 │    │            action block i            │    │
                 │    │  K/V ← [txt | ref? | syn | action]   │    │
                 │    └──────────────────────────────────────┘    │
                 └───────────────────────┬────────────────────────┘   latents are carried across all 25 layers
                                         │
            ┌────────────────────────────┴──────────────────┐
            ▼                                               ▼
  video_expert.post_dit                          action_expert.post_dit
            │                                               │
            ▼                                               ▼
  pred_video [B,N,128]                            pred_action [B,16,7]
```

The VAE (`self.vae`) and Qwen3 (`self.text_encoder`) are separate modules feeding the video expert, but
the action chunk goes straight into `action_expert.pre_dit` — its first step is
`action_expert.action_encoder`. Unlike stage 1, the video expert becomes trainable here, and the goal
tokens are no longer used.

Where each stage-2 block attaches — these are the only new pieces:

| Block | Reads | Writes into | Section |
| --- | --- | --- | --- |
| the layer loop | — | the order inside every layer: video block → aggregate → action block | §2.8 |
| `SemanticVisualAggregator` | txt + ref-image features out of each video block | `goal_latents`, carried across all 25 layers | §2.9 |
| `SemanticVisualAggregatorGroup` — `to_key` / `to_value` | `goal_latents`, 100 × 768 | this layer's synthetic K/V | §2.10 |
| `project_kv` + RoPE | those K/V | the `syn` columns of the action K/V | §2.11 |
| `pose_norm` + `GoalPoseDecoder` | `goal_latents[:, :8]` | `L_pose` (weight 0.3) | §2.9 |
| cold-start gate | — | the action attention logits, as a float mask | §2.12 |
| firewall / plan-B regime | — | which of those channels are visible at all | §2.12 |

The prior writes to the **action** stream only. The video stream runs as Part 1 describes; stage 2
reads from it but never writes to it.

Loss: `0.5 · L_video + 1.0 · L_action + 0.3 · L_pose`.

### 2.8 Micro — inside the layer loop

The macro draws one layer. That box repeats 25 times, and the only state the *aggregator* carries from
one iteration to the next is `goal_latents` — it is re-run after *every* video block and hands the
updated latents forward, so layer `i` reads what layer `i-1` saw:

```text
latents — 100 learnable queries × 768
                 │                      initialised once, before layer 0  (§2.9)
                 ▼                      carried across all 25 layers
  ┌─────────── video ────────────┐
  │        video block i         │
  │    txt + img features out    │
  └──────────────┬───────────────┘
                 │    i < 5 → double block      i >= 5 → single block
                 ▼
  ┌───────── aggregate ──────────┐
  │   SemanticVisualAggregator   │
  │   self → semantic → visual   │
  └──────────────┬───────────────┘
                 │    group = i // 5
                 │    updated latents ──► carried to the next layer
    ┌────────────┴───────────────┐
    ▼                            ▼
  latents[:, :8]             latents (all 100)
    │                            │
    ▼                            ▼
  pose_norm +                to_key / to_value
  GoalPoseDecoder                │
    │                            ▼
    ▼                       syn K/V [B,100,3072]
    L_pose  (×0.3)               │
                                 │
                                 ▼
                    ┌──────── attend ─────────┐
                    │      action block i     │
                    │   Q   ← action tokens   │
                    │ K/V ← [txt|ref?|syn|act]│
                    └────────────┬────────────┘
                                 │
                                 ▼
                              action tokens → next layer
```

Note that the aggregator and the action block both live *inside* layer `i`, between the video block and
the next layer — the action block's queries see the synthetic K/V that was just produced.

### 2.9 Micro — `SemanticVisualAggregator` (`goal_pose_prior.py:309`)

100 learnable latents (8 pose + 92 context, dim 768) updated **once per FLUX layer**, with the 25
layers grouped into 5 parameter sets (`layer_idx // 5`):

```text
   queries: 100 × 768  (learnable, trunc_normal std=0.02)
        │
        │   for layer_idx in 0..24:
        │       group = groups[layer_idx // 5]        ← 5 groups, 5 layers each
        │
        ├──────────────────────────────────────────────────────────┐
        ▼                                                          │
   SemanticVisualAggregatorGroup(group)                            │
        │  inputs:  txt (3072) ── semantic cross-attention         │
        │           ref img (3072) ── visual cross-attention       │
        │                                                          │
        ├──► latents' ─────────────────────────────────────────────┘
        │
        ├──► latents[:, :8]  ──► pose_norm ──► GoalPoseDecoder ──► L_pose
        │
        └──► latents (all 100) ──► to_key / to_value ──► synthetic K/V for this layer
```

Key facts:

- **Visual context is the reference image only** — `_flux2_update_goal_latents` takes `img[:, :cond_len]`
  (`mot.py:757`), so the noisy target never enters the aggregator.
- The latents are re-projected to K/V through that layer's group, not carried across wholesale.
- **B1 `latent_layout: grid`** (`goal_pose_prior.py:513`) ties the context latents to a 7×7 grid so each
  sees only its own neighbourhood of image tokens; `free` (the evaluated default) gives all latents
  global attention.
- `pose_norm` is a `LayerNorm` (`imagewam.py:216`) applied to the first 8 latents before
  `GoalPoseDecoder`; along with the decoder it is randomly initialised when bridging from stage 1.

### 2.10 Micro — `SemanticVisualAggregatorGroup` (`goal_pose_prior.py:243`)

```text
   queries (100 × 768)
        │
   self_block        ── LayerNorm ─► MHA(self) ─► + ─► LayerNorm ─► FFN ─► +
        │
   semantic_block    ── cross-attn over txt tokens   (3072), padding-masked
        │
   visual_block      ── cross-attn over ref img tokens (3072), padding-masked
        │
        ├──► updated queries
        │
        └──► to_key / to_value ──► [B, 100, 3072]
                    │
              key_norm (_RMSNorm over attn_head_dim = 128)
                    │
              view as num_kv_heads × 128  →  3072-wide K/V
```

`gate_bias_init` seeds `syn_gate_bias`, a learnable scalar on the group used by the cold-start gate
(§2.12). `zero_init_value` optionally zeroes `to_value`; it is **mutually exclusive** with
`gate_pose_tokens=False` and raises (`goal_pose_prior.py:373`) — see the comment there for why the
gate must then be carried by the logit bias alone.

### 2.11 Micro — synthetic K/V projection (`mot.py:694`)

The 100 latents are not usable as attention keys directly: they are 768-wide and live in the
aggregator's own space, while the action attention needs 3072-wide keys in FLUX head layout with
positional encoding applied.

```text
   goal_latents [B, 100, 768]
        │
   project_kv  ──► to_key / to_value  ──► [B, 100, 3072]
        │
   view as 24 heads × 128, apply RoPE
        │
   ids: axis 0 = 3.0 (synthetic), axis 1 = arange(100)   (goal_pose_prior.py:617)
        │
   synthetic K/V  ──► concatenated into the action queries' K/V
```

The synthetic time value `3.0` keeps these tokens positionally distinct from text (axis 3), image
(0.0 / 10.0) and action (2.0) — see the id table in §1.4.

### 2.12 Micro — firewall, plan B, cold-start gate

`build_stage2_action_attention_mask` (`goal_pose_prior.py:631`) builds the action queries' K/V layout.
`ref_len=0` **literally removes the raw image columns** — the firewall is an architectural absence, not
a masked-out region:

```text
  firewall (ref_len = 0)                     plan B (action_sees_ref = true)
  [ txt | syn | action ]                     [ txt | ref | syn | action ]
    ▲            ▲                             ▲     ▲           ▲
    │            │                             │     │           │
  text/state  100 synthetic K/V              + reference image columns
```

Training samples which channels are visible, with fixed counts so every rank sees every regime
(`goal_pose_prior.py:478`):

| regime | p | K/V layout | purpose |
| --- | ---: | --- | --- |
| both | 0.55 | `[txt \| ref \| syn \| action]` | the deployment condition |
| ref_only | 0.15 | `[txt \| ref \| action]` | breaks dependence on the synthetic channel |
| syn_only | 0.30 | `[txt \| syn \| action]` | keeps firewall mode trained well enough to report |

**Cold-start gate** (`_flux2_gate_action_mask`, `mot.py:723`): a per-layer learnable scalar added to the
synthetic columns' attention logits, so stage 2 begins as a near no-op instead of first having to learn
to ignore noise. It is *additive on a float mask*, not a bool mask:

```text
   action_mask (bool)  ──►  additive float 0 / -inf
                                  │
                                  ▼
                        += syn_gate_bias on [syn_start+lo, syn_start+hi)
                                  │
                                  ▼
                        SDPA with a float mask  (mot.py:146)
```

`gated_span()` (`goal_pose_prior.py:425`) returns `(8, 100)` under the default
`gate_pose_tokens=false`: the **92 context columns are gated, the 8 pose columns stay open**, so step 0
reproduces stage 1's `[txt | pose | action]` topology — which is also exactly what B2+ blackout falls
back to.

### 2.13 Micro — the Stage 1 → Stage 2 bridge

Stage 2 **requires** `stage1_checkpoint` unless `resume` is set, so it can never silently train from the
ActionDiT init. The checkpoint bridge is fail-closed; the only permitted key differences come from the
prefix constants at `goal_pose_prior.py:64`:

| | keys |
| --- | --- |
| dropped when bridging stage1 → stage2 | `goal_pose_encoder.*`, `stage1_null_image_tokens` |
| randomly initialised | `semantic_visual_aggregator.*`, `semantic_visual_pose_norm.*`, `semantic_visual_pose_decoder.*` |

## Part 2 — things that are easy to get wrong

- **The action expert is shape-locked to the video expert** except for `hidden_dim`. Config values for
  `num_heads` / `attn_head_dim` / layer counts are silently overwritten (`imagewam.py:661`).
- **q/k/v concatenation is free only because both experts are 24 × 128.** Any change to the video
  variant's head geometry forces a matching change in the action expert.
- **Float masks are additive biases, bool masks are keep-masks** (`mot.py:101`). Passing a float mask
  where a bool was expected will not raise.
- **Dropout-style mechanisms read the *module's* train/eval mode**, not the top-level model's. The
  trainer never calls `.train()` on `ImageWAM` itself, so anything keyed off `self.training` at the top
  level silently no-ops. `sample_context_keep_mask` and `sample_channel_regime` both default to
  `self.training` for this reason (`goal_pose_prior.py:449`, `goal_pose_prior.py:491`).
- **Stage 1 freezes the video expert but still runs autograd through it.** `apply_trainable_policy`
  calls `.eval()` and `requires_grad_(False)` on the video expert, yet the backward pass still traverses
  all 25 layers to reach `goal_pose_encoder` — which is why per-layer checkpointing is mandatory there.
- **Per-regime samplers use fixed counts, not independent coins** (`goal_pose_prior.py:462`). Every rank
  must see every regime in every batch, or the per-key all-gather in the trainer desynchronises NCCL
  and hangs the job.
- **Metrics must be emitted on every rank, every step** — `_goal_prior_diagnostics` always emits every
  key for the same reason.

---

# Part 3 — inference

Everything above is training. Deployment runs the same modules in a different order, and the difference
is not cosmetic: **the video stream is computed once per action chunk and cached; only the action expert
is re-run per denoising step.** That hoist is what makes a 10–20 step action diffusion affordable.

## 3.0 Macro block diagram

```text
        ref image              prompt              noise x_T [B,16,7]
            │                     │                         │
            ▼                     ▼                         │
       ┌── VAE ──┐         ┌─── text ────┐                  │
       │  encode │         │    Qwen3    │                  │
       └────┬────┘         │  9, 18, 27  │                  │
            │              └──────┬──────┘                  │
            ▼                     ▼                         │
       ref tokens          txt [B,L,7680]                   │
            │                     │                         │
            └──────────┬──────────┘                         │
                       ▼                                    │
            ┌─────── video ───────┐                         │
            │     video_expert    │                         │
            │ .pre_dit (0 tokens) │                         │
            └──────────┬──────────┘                         │
                       │                                    │
                       ▼                                    │
   ┌─────────────── prefill ───────────────┐                │          §3.1  the video stream runs once
   │    video blocks ×25  +  aggregator    │                │
   │   cache per layer:  txt · ref? · syn  │                │
   │   all of it computed once per chunk   │                │
   └───────────────────┬───────────────────┘                │
                       │                                    │
                       └─────────────────┬──────────────────┘
                                         ▼
                ┌──────────────────── denoise ─────────────────────┐   §3.3  action-only, ×N
                │  action_expert.pre_dit → blocks ×25 → post_dit   │   §3.2  the layout is the infer mode
                │       K/V ← [ txt | ref? | syn | action ]        │
                │   x ← x + v · Δσ     (N = num_inference_steps)   │   §3.3  Euler step, σ: 1 → 0
                └────────────────────────┬─────────────────────────┘
                                         │
                                         ▼
                               action chunk [B,16,7]
                                         │
                                         ▼
                       execute replan_steps, then re-encode            §3.5  the closed loop
```

| Phase | What runs | How often |
| --- | --- | --- |
| encode + prefill | VAE, Qwen3, `video_expert.pre_dit`, 25 video blocks, the aggregator | once per chunk |
| denoise | 25 action blocks | `num_inference_steps` times |

`prefill` and `denoise` are the two halves of the *same* 25-layer stack — §2.4's MoT picture still
describes one layer, inference just runs the halves a different number of times.

## 3.1 Micro — prefill: the video stream runs once

Two entry points, one per stage family:

| | baseline and stage 1 | stage 2 |
| --- | --- | --- |
| called from | `_infer_action_flux2_baseline` (`imagewam.py:4121`) | `_infer_action_flux2_stage2` (`imagewam.py:4360`) |
| prefill | `prefill_flux2_video_cache` (`mot.py:1023`) | `prefill_flux2_goal_prior_cache` (`mot.py:1158`) |
| cached per layer | `k`, `v` | `txt_k/v`, `ref_k/v`, `syn_k/v`, `gate_bias`, `gated_span`, `mode` |
| aggregator | — | run once per layer during prefill |

```text
        training                                    inference
   ┌─────────────────────┐                   ┌─────────────────────┐
   │   video block   i   │  both experts     │   video block   i   │   ×25, once
   │   action block  i   │  run every layer  │   + aggregator      │
   └─────────────────────┘                   └──────────┬──────────┘
                                                        │ cache K/V
                                                        ▼
                                              ┌─────────────────────┐
                                              │   action block  i   │   ×25, per step
                                              └─────────────────────┘
```

The stage-2 prefill does one thing the baseline's does not: it also runs `_flux2_update_goal_latents` and
`_flux2_project_synthetic_kv` on each layer's updated video features, and caches the resulting synthetic
K/V. (It also keeps the text and reference K/V separate rather than as one block, because the action
forward re-assembles the layout per mode.) So the aggregator runs **25 times per chunk**, not 25 times
per denoising step — the LIT path is a per-chunk cost, and the gate bias is frozen for the chunk too.

The prefill pass is masked with `action_len=0` (`imagewam.py:4178`): the video stream attends to itself,
exactly as at training time, and never sees the action tokens that do not exist yet.

## 3.2 Micro — the action K/V at inference: the three modes

`_resolve_infer_mode` (`imagewam.py:4238`) answers §2.12's firewall question at deployment. The default
is read off the checkpoint, not off a flag:

| mode | action K/V layout | available when |
| --- | --- | --- |
| `full` | `[txt \| ref \| syn \| action]` | `action_sees_ref=true` — and then it is the default |
| `firewall` | `[txt \| syn \| action]` | always; the default when the reference channel was never trained |
| `ref_only` | `[txt \| ref \| action]` | `action_sees_ref=true` |

`IMAGEWAM_INFER_MODE` overrides the default for the ablation numbers. Asking for `full` or `ref_only`
from a checkpoint trained without the reference channel raises (`imagewam.py:4254`) — it fails closed
rather than scoring an untrained channel.

The mode is applied in two places that cannot disagree: `build_stage2_action_attention_mask(ref_len=…,
synthetic_len=…)` (`goal_pose_prior.py:631`) builds the mask — the same function training uses, so
`ref_len=0` removes the columns rather than masking them, and the prefill cache is stamped with the mode
(`mot.py:1281`) for the action forward to read back (`mot.py:1315`).

The cold-start gate travels the same route: the bias comes out of the cache and is added to the float
mask only when it is non-zero (`mot.py:1332`), so an open gate leaves the mask untouched and a closed one
costs nothing.

## 3.3 Micro — the denoising loop

`build_inference_schedule` (`scheduler_continuous.py:63`) lays down σ from 1 → 0 over
`num_inference_steps`, warped by the shift (`_phi`, `scheduler_continuous.py:18`); `step` is plain Euler
(`scheduler_continuous.py:83`). The network is trained to predict velocity (`noise - sample`,
`scheduler_continuous.py:59`), so each step is `x ← x + v · Δσ` with Δσ < 0:

```text
   σ:   1.000 ──► 0.978 ──► 0.952 ──► 0.921 ──► ... ──► 0.357 ──► 0.000
        x_T                                                        action chunk
        └────────────────────── x ← x + v · Δσ  per step ──────────────┘
```

Those are the real numbers for `num_inference_steps=10` and the default `shift=5.0`
(`configs/model/imagewam_flux2_klein_4b_base.yaml:50`) — the shipped LIBERO setting. The shift makes the
schedule deliberately non-uniform: the early steps creep along near σ≈1 and the last one covers
0.36 of the range on its own.

Cost is linear in `num_inference_steps` and in nothing else: it is the number of times the 25 action
blocks run. The video stack does not run again.

## 3.4 Micro — what inference never runs

- **No noisy video target.** Every inference prefill passes `x = empty_target` (`imagewam.py:4151`) with
  `target_len=0`, so the video expert's `post_dit` and `final_layer` never execute at deploy time and
  there is no `pred_video`.
- **Stage 1 needs an oracle goal pose, and nothing supplies one.** `_infer_action_flux2_stage1` raises
  without `goal_pose` (`imagewam.py:4277`), and the public `infer_action` has no `goal_pose` parameter
  (`imagewam.py:3607`) — the shipped LIBERO and RoboTwin evaluators can therefore only run baseline and
  stage-2 checkpoints. Stage 1 is reachable only by calling `infer_action_flux2(goal_pose=…)` directly.
- **The stage-1 null tokens are training-only.** The 32 learnable `stage1_null_image_tokens` stand in
  for the missing reference image during training (`imagewam.py:2776`), but the stage-1 *inference* path
  passes `ref_image_hidden_states=None` (`imagewam.py:4310`) — its prefix is `[txt | pose]` with no
  image slots at all. A train/deploy difference to know about before scoring stage 1 directly.
- **`negative_prompt` / `text_cfg_scale` are accepted and then dropped.** `infer_action` takes both
  (`imagewam.py:3607`); the FLUX.2 branch forwards neither. There is no classifier-free guidance here.
- **`visualize_future_video` cannot run on FLUX.2.** It calls `infer_joint` (`imagewam.py:3427`), which
  reads `vae.temporal_downsample_factor` and `vae.upsampling_factor` (`imagewam.py:3502`) — the FLUX.2
  autoencoder defines neither (see the note in *Where the code lives*). Leave it off.

## 3.5 Micro — the closed loop

`_predict_action_chunk` (`eval_libero_single.py:514`) builds the prompt, passes the **current camera
frame** as `input_image` — the "ref image" at deployment is the present observation, not a goal — and
calls `infer_action`. Around it:

- **Denormalization is min/max**, applied per action dim from the processor's normalizer
  (`eval_libero_single.py:414`) — the stats come from the run's `dataset_stats.json`.
- **Gripper handling.** The dataloader flips the sign of the gripper dim, so the evaluator flips it back
  and optionally binarizes it (`eval_libero_single.py:580`).
- **Receding horizon.** `run_single_episode` (`eval_libero_single.py:600`) waits `num_steps_wait` steps,
  predicts a chunk, executes `replan_steps` of it open-loop, then re-encodes the fresh observation and
  replans.
- **Optional `ActionEnsembler`** (`experiments/libero/action_ensembler.py:5`) averages the overlapping
  predictions of successive chunks per timestep; off by default.

LIBERO defaults (`configs/sim_libero.yaml`): `action_horizon = num_frames - 1 = 16`, `replan_steps: 10`,
`num_steps_wait: 30`, `num_inference_steps: ${eval_num_inference_steps} = 10` (`configs/train.yaml:28`),
`binarize_gripper: true`. RoboTwin drives the same `infer_action` API through
`experiments/robotwin/imagewam_policy/deploy_policy.py:324`.

## Part 3 — things that are easy to get wrong

- **The inference mode belongs to the checkpoint, not to the command line.** The same eval command
  scores v2 (`action_sees_ref=true`) as `full` and v1 as `firewall`. `IMAGEWAM_INFER_MODE` only
  overrides, and requesting a channel the checkpoint never trained raises (`imagewam.py:4254`).
- **Reported `both` numbers are `full`.** The README's regimes and the code's modes are different
  vocabularies — `both` ⇔ `full` (the naming trap in *Where the code lives*).
- **The prefill cache freezes the whole LIT path for the chunk.** Synthetic K/V, gate bias and ref
  columns are computed once per replan; only the action expert runs again per denoising step.
- **Two different default step counts.** `infer_action_flux2` defaults to `num_inference_steps=20`
  (`imagewam.py:4064`) while the shipped sim configs pass 10. Numbers from different step counts are not
  comparable.
- **An eval needs the training run's min/max stats.** `DATASET_STATS_PATH` is not optional, and the
  denormalization is silent about wrong stats.

# Part 4 — building LIT on top of FLUX.2

Parts 1–3 describe what this code *is*. This part describes what it *added*, in the order you would add
it if you started from stock FLUX.2 knowing nothing about LIT — the reverse-engineering view. Every
milestone has the same three lines:

```text
FLUX.2 gives:   what already exists upstream and is not modified
You add:        the piece with no upstream equivalent
Check:          the experiment that says the seam is in the right place
```

The order is a dependency order, not a reading order. M0–M2 are the general-purpose machinery both
stages share; M3 is stage 1; M4–M5 are stage 2; M6 is training; M7 is deployment.

## 4.0 Build order

```text
   M0  inventory      FLUX.2 as shipped — read it, change nothing
        │
   M1  action expert  a slim FLUX copy: 1024-wide stream, FLUX's 24 × 128 attention
        │
   M2  MoT            two experts, one SDPA, split back
        │
        ├───────────────────► M3  stage 1: the oracle pose, [txt | pose | action]
        ▼
   M4  aggregator     100 latents → synthetic K/V                          (stage 2)
        │
   M5  firewall       what the action may see, and how quietly the channel opens
        │
   M6  losses         L_video + L_action + L_pose, and the stage1 → stage2 contract
        │
   M7  inference      prefill once, denoise N times
```

| § | milestone | the one-line version |
| --- | --- | --- |
| 4.1 | M0 inventory | the seam is in the blocks' private q/k/v split, not in their public API |
| 4.2 | M1 action expert | clone FLUX's attention geometry, shrink the residual stream to 1024 |
| 4.3 | M2 MoT | concatenate q/k/v from both experts, one SDPA, split back at `L_v` |
| 4.4 | M3 stage 1 | encode an oracle pose into 8 tokens and append them to the text stream |
| 4.5 | M4 vision | 100 recurrent latents produce per-layer synthetic K/V for the action |
| 4.6 | M5 firewall | the mask decides the channel layout, the gate decides how loud it starts |
| 4.7 | M6 losses | three losses, a fail-closed checkpoint contract, and rank-safe metrics |
| 4.8 | M7 inference | hoist the video stream out of the denoising loop |

## 4.1 M0 — what stock FLUX.2 gives you

```text
FLUX.2 gives:   Flux2.forward, the two block types, EmbedND, Modulation, LastLayer,
                AutoEncoder, Qwen3Embedder          — model.py:52, model.py:524
You add:        nothing. This milestone is reading.
Check:          Flux2(x, x_ids, t, ctx, ctx_ids, guidance) runs unchanged.
```

```python
# FLUX.2 as shipped. LIT modifies none of this.            # model.py:115
class Flux2(nn.Module):
    def forward(self, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        vec = self.time_in(timestep_embedding(timesteps, 256))     # model.py:710
        img, txt = self.img_in(x), self.txt_in(ctx)
        pe_x, pe_ctx = self.pe_embedder(x_ids), self.pe_embedder(ctx_ids)   # model.py:694
        for block in self.double_blocks:                           # model.py:524
            img, txt, _ = block.forward_kv_extract(img, txt, pe_x, pe_ctx, ...)
        img = cat([txt, img], dim=1)
        pe = cat([pe_ctx, pe_x], dim=2)
        for block in self.single_blocks:                           # model.py:437
            img, _ = block.forward_kv_extract(img, pe, ..., num_txt_tokens)
        return self.final_layer(img[:, num_txt_tokens:], vec)      # model.py:415
```

**The decoy.** Upstream already has a caching story: `forward_kv_extract` (`model.py:170`) and
`forward_kv_cached` (`model.py:267`) run the first step with reference tokens and reuse their K/V on
later steps, through `causal_attn_fn` (`model.py:758`). It looks like exactly the hook a world–action
model wants, and LIT **does not use any of it** — that path is causal, single-stream, and for reference
*images*; LIT needs bidirectional attention between two experts. What the MoT actually calls is one
level deeper, below the block's public API:

| what you need | where it lives |
| --- | --- |
| double block: q/k/v of `[txt \| img]` | `DoubleStreamBlock._prepare_qkv` (`model.py:569`) |
| double block: the two residual branches | `DoubleStreamBlock._apply_residuals` (`model.py:614`) |
| single block: q/k/v **and** the fused MLP | `SingleStreamBlock._qkv` (`model.py:468`) |
| single block: output projection + gate | `SingleStreamBlock._out` (`model.py:482`) |

Those four are what every later milestone builds on. They are private, so a reimplementation should
plan on vendoring the block rather than importing it.

The other thing to read off at this stage is the id convention (`flux2_video_expert.py:63`,
`flux2_video_expert.py:75`), because LIT's synthetic tokens need ids that collide with nothing:

| axis 0 (`t`) | axis 1 | axis 2 | axis 3 |
| --- | --- | --- | --- |
| 0.0 target image, 10.0 reference image, 2.0 action, **3.0 synthetic** | row | col | text position |

## 4.2 M1 — the action expert

```text
FLUX.2 gives:   QKNorm, SiLUActivation, MLPEmbedder, Modulation, timestep_embedding,
                and the block layout to copy   — model.py:375, model.py:390,
                model.py:683, model.py:400, model.py:746
You add:        SlimFlux2DoubleBlock, SlimFlux2SingleBlock, Flux2ActionHead,
                ActionDiTFlux2      — action_dit_flux2.py:30, action_dit_flux2.py:83,
                action_dit_flux2.py:134, action_dit_flux2.py:146
Check:          pre_dit → post_dit at zero depth returns [B, T, 7].
```

The whole design is one decision: **keep FLUX's attention geometry, shrink the residual stream.**

```python
class ActionDiTFlux2:                                # action_dit_flux2.py:146
    action_dim, hidden_dim = 7, 1024                 # hidden_dim is the ONE free number
    num_heads, attn_head_dim = 24, 128               # copied from the video expert, not chosen
    attn_dim = num_heads * attn_head_dim             # 3072 — width the two experts share

    def pre_dit(self, action, t):                    # action_dit_flux2.py:249
        tokens = self.action_encoder(action)         # [B, 16, 7] → [B, 16, 1024]
        vec    = self.time_in(timestep_embedding(t, 256))
        return dict(tokens=tokens,
                    ids=build_action_ids(...),       # axis 0 = 2.0      action_dit_flux2.py:237
                    t_mod=dict(vec=vec,
                               double_img=self.double_stream_modulation_img(vec),
                               single=self.single_stream_modulation(vec)[0]))

    def post_dit(self, tokens, pre):                 # action_dit_flux2.py:287
        return self.head(tokens, pre["t_mod"]["vec"])   # adaLN → Linear(1024 → 7)
```

Both block types expose the same two-method split, which is what makes M2 a loop instead of a branch:

```python
class SlimFlux2DoubleBlock:                          # action_dit_flux2.py:30
    def prepare_qkv(self, x, pe, mod):               # action_dit_flux2.py:52
        mod1, mod2 = mod
        x_mod = (1 + mod1.scale) * self.img_norm1(x) + mod1.shift
        q, k, v = rearrange(self.img_attn.qkv(x_mod), "B L (K H D) -> K B H L D", K=3, H=24)
        q, k = self.img_attn.norm(q, k, v)           # QKNorm, upstream's class
        q, k = apply_rope(q, k, pe)                  # model.py:828
        return dict(q=..., k=..., v=..., residual_x=x, mod2=mod2, mod1_gate=mod1.gate)

    def apply_post(self, mixed_attn_out, state):     # action_dit_flux2.py:75
        x = state.residual_x + state.mod1_gate * self.img_attn.proj(mixed_attn_out)
        return x + state.mod2_gate * self.img_mlp(
            (1 + state.mod2.scale) * self.img_norm2(x) + state.mod2.shift)
```

The single block fuses the MLP into `linear1` and computes it *before* attention, exactly as
`SingleStreamBlock` does (§1.6) — that asymmetry between the two block types is inherited, not chosen.

**Initialisation is a separate script, not a `from_pretrained`.** `ActionDiTFlux2.from_pretrained`
(`action_dit_flux2.py:211`) loads a *pre-converted* checkpoint; producing that checkpoint is
`preprocess_action_dit_flux2.py`, which copies FLUX.2's img/single branches into the slim blocks by
name and repairs every shape mismatch:

```python
def convert(flux2_state, dst_model):                 # preprocess_action_dit_flux2.py:142
    for name, dst in dst_model.named_parameters():
        src = flux2_state[name]
        if src.shape != dst.shape:
            src = interpolate(src, dst.shape)        # linear along the mismatched axis
            if src.ndim >= 2 and src.shape[-1] != dst.shape[-1]:
                src = src * sqrt(src.shape[-1] / dst.shape[-1])   # alpha, preserves variance
        dst.copy_(src)
```

## 4.3 M2 — the MoT: two experts, one SDPA

```text
FLUX.2 gives:   the four block seams from M0
You add:        MoT: the expert registry, _mixed_attention, one forward per stage
                — mot.py:22, mot.py:132
Check:          an empty action stream degenerates to the stock video pass.
```

Everything works because both experts emit q/k/v of the **same width**, so concatenation needs no
projection — only the residual streams differ (3072 vs 1024):

```python
# baseline and stage 1 — one joint SDPA      # mot.py:626, mot.py:670
def layer(img, txt, action):
    video_q, video_k, video_v, pe, n_txt, mods = v_block._prepare_qkv(img, txt, pe_x, pe_ctx, ...)
    video_q, video_k = apply_rope(video_q, video_k, pe)
    a = a_block.prepare_qkv(action, action_pe, mod)      # [B, L_a, 3072]
    mixed = sdpa(cat([video_q, a.q], 1),
                 cat([video_k, a.k], 1),
                 cat([video_v, a.v], 1), mask)
    video_out, action_out = split(mixed, [L_v, L_a])
    img, txt = v_block._apply_residuals(img, txt, *split(video_out, [n_txt, L_img]), mods)
    return img, txt, a_block.apply_post(action_out, a)
```

**Stage 2 severs that joint call**, and this is the single most surprising thing in the reverse-engineering
path. The video stream stops reading the action tokens; the action stream becomes a reader of the video's
K/V rather than a mutual partner:

```python
# stage 2 — two separate attentions        # mot.py:863, mot.py:891
video_attn  = sdpa(video_q, video_k, video_v, video_mask)     # video attends itself only
action_attn = sdpa(a.q, cat([txt_k | ref_k? | syn_k | a.k]), gated_action_mask)
```

Nothing in Parts 2–3 would tell you this by looking at the macro diagram — it is visible only in the
second forward (`mot.py:775`). It also explains §3.1's cache design: because the video stream never
reads the action stream, prefill can run the video half alone and hand the action half a frozen K/V.
The action expert is still *trained* in stage 2, but as a consumer.

## 4.4 M3 — the goal-pose interface (stage 1)

```text
FLUX.2 gives:   txt_in (7680 → 3072), EmbedND, build_txt_ids  — flux2_video_expert.py:63
You add:        GoalPoseEncoder, _append_goal_tokens_to_flux2_pre,
                _stage1_null_image_stream
                — goal_pose_prior.py:85, imagewam.py:2060, imagewam.py:2038
Check:          prefix length is txt_len + 8, and the 8 pose columns are never masked.
```

```python
def stage1_prefix(text, text_mask, pose):            # imagewam.py:2060
    goal = goal_pose_encoder(pose)                   # [B, 8] → [B, 8, 3072]
    txt  = cat([text, goal], dim=1)                  # [txt | pose]
    ids  = build_txt_ids(seq_len=txt.shape[1])       # pose rides axis 3, after the text
    return txt, pe_embedder(ids), cat([text_mask, ones(B, 8)], dim=1)
```

The encoder is three linears wide enough to be worth stating exactly — `8 → 512 → 512 → 8·3072`
(`goal_pose_prior.py:99`), i.e. it emits the FLUX hidden interface directly, not the 7680-D Qwen one.
The image stream is **empty**, and the two flags that dress it up are both training-only:

```python
def stage1_loss(sample):                             # imagewam.py:2731
    x     = zeros(B, 0, 128)                         # no noisy target at all
    ref   = null_image_tokens.expand(B, -1, -1) or None   # learnable stand-in, image ids, t = 10.0
    t_vid = train_video_scheduler.sample() if cfg.stage1_sample_video_timestep else zeros(B)
    pre   = video_expert.pre_dit(x=x, timestep=t_vid, context=txt,
                                 ref_image_hidden_states=ref, target_img_ids=empty)
    pre   = append_goal_tokens_to_flux2_pre(pre, pose)
    mask  = build_mask(txt_len=pre.txt_len, target_len=0,
                       cond_len=pre.cond_len, action_len=L_a)          # imagewam.py:2312
    out   = mot(video=pre, action=action_pre, attention_mask=mask)
    return loss_lambda_action * weighted_mse(action_expert.post_dit(out.action, action_pre), target)
```

`target_len=0` deletes the target row and column from the mask rather than masking them (§2.5), which
is what leaves `[txt | pose | action]`.

## 4.5 M4 — the aggregator and the synthetic channel (stage 2)

```text
FLUX.2 gives:   the per-layer video features (post-attention txt and img) — nothing else
You add:        SemanticVisualAggregator + SemanticVisualAggregatorGroup,
                _flux2_update_goal_latents, _flux2_project_synthetic_kv
                — goal_pose_prior.py:309, goal_pose_prior.py:243,
                  mot.py:745, mot.py:694
Check:          the latents differ at every one of the 25 layers; with cond_len = 0 the visual
                stream is a single masked-out token instead of an empty tensor.
```

This is the one place where the design is not derivable from FLUX.2 — it is the contribution. The loop
body, in full:

```python
def stage2_layer(i, img, txt, action, latents, a_block, v_block):
    video_attn = sdpa(...)                                   # video, self-only (§4.3)
    img, txt = v_block._apply_residuals(...)

    latents = aggregator.forward_layer(                      # mot.py:745 — after EVERY block
        latents,
        semantic_hidden=txt,                                 # the text tokens
        visual_hidden=img[:, :cond_len],                     # the REFERENCE image only
        semantic_mask=text_mask,
        image_mask=ones(B, cond_len),
        layer_idx=i, num_layers=25)                          # → group i // 5

    syn_k, syn_v = group.project_kv(latents)                 # [B, 100, 3072]
    syn_k = rope_on_k_only(syn_k, ids(axis0=3.0, axis1=arange(100)))

    action_attn = sdpa(a.q, cat([txt_k | ref_k? | syn_k | a.k]), gate(action_mask))
    return img, txt, a_block.apply_post(action_attn, a.state), latents
```

Two details in that body are easy to get subtly wrong, and both are load-bearing:

- **The residual stream that survives 25 iterations is `latents` alone.** The aggregator is re-run from
  the updated queries each layer and re-projected through *that layer's* group; nothing else carries
  across layers. `layer_idx // 5` gives five parameter sets over 25 layers (`goal_pose_prior.py:592`).
- **The K/V projection order is `to_key → view(…, 128) → RMSNorm → reshape`** (`goal_pose_prior.py:299`),
  not a norm over the 3072-wide vector. And RoPE is applied to the key with a dummy query
  (`mot.py:717`) — `apply_rope` needs two arguments, but only `k` is kept.

The group itself is a small transformer over the 100 latents:

```python
class SemanticVisualAggregatorGroup:                 # goal_pose_prior.py:243
    def forward(self, queries, semantic, visual, *, semantic_mask, image_mask, visual_attn_mask=None):
        q = self.self_block(queries)                             # 100 latents attend themselves
        q = self.semantic_block(q, semantic, semantic_mask)      # cross-attn over txt, padding-masked
        q = self.visual_block(q, visual, image_mask, visual_attn_mask)   # cross-attn over the ref image
        return q

    def project_kv(self, tokens):
        key = self.to_key(tokens).view(B, 100, self.num_kv_heads, 128)
        key = self.key_norm(key).reshape(B, 100, 3072)           # _RMSNorm over 128
        return key, self.to_value(tokens)
```

The 8 pose latents are the ones that carry the training signal — `latents[:, :8]` → `pose_norm` →
`GoalPoseDecoder` → `L_pose` (§4.7). The other 92 exist only as attention context.

## 4.6 M5 — firewall, regimes, cold-start gate

```text
FLUX.2 gives:   nothing — pure LIT
You add:        build_stage2_action_attention_mask, sample_context_keep_mask,
                sample_channel_regime, _flux2_gate_action_mask
                — goal_pose_prior.py:631, goal_pose_prior.py:437,
                  goal_pose_prior.py:478, mot.py:723
Check:          ref_len = 0 ⇒ key_len == txt_len + syn_len + action_len.
```

The mask is built by *layout*, not by masking — recalling that a column that does not exist costs
nothing, while a masked column still costs a softmax slot:

```python
def action_keep_mask(txt_len, ref_len, syn_len, action_len):     # goal_pose_prior.py:631
    mask = zeros(B, action_len, txt_len + ref_len + syn_len + action_len, dtype=bool)
    mask[:, :, :txt_len]                   &= text_mask         # padding columns
    mask[:, :, ref_cols]                   &= ref_keep          # per-sample, plan B
    mask[:, :, syn_cols]                   &= syn_keep          # B2 dropout / B2+ blackout
    mask[:, :, action_cols]                 = True              # the action always sees itself
    return mask
```

The gate is the second half of the cold-start story, and it must be **additive on a float mask** — a
bool mask has nowhere to put a bias:

```python
def gate(mask, syn_start, bias, span):                           # mot.py:723
    additive = where(mask, 0.0, -inf)                            # float, not bool
    lo, hi = span                                                # (8, 100) by default
    additive[..., syn_start + lo : syn_start + hi] += bias        # only the 92 context columns
    return additive
```

`zero_init_value=True` and `gate_pose_tokens=False` are mutually exclusive and the constructor raises
on the pair (`goal_pose_prior.py:373`): `to_value` is one linear shared by all 100 latents, so zeroing
it would silence the pose columns the split gate deliberately keeps open.

The regime samplers draw **fixed counts, never independent coins**:

```python
def sample_channel_regime(B):                                    # goal_pose_prior.py:478
    n_syn_only = round(p_syn_only * B)                           # 0.30 → no synthetic columns
    n_ref_only = round(p_ref_only * B)                           # 0.15 → no reference columns
    order = randperm(B)
    ref_keep[order[:n_syn_only]] = False
    syn_keep[order[n_syn_only : n_syn_only + n_ref_only]] = False
    return ref_keep, syn_keep
```

The reason is not statistical — it is §M6's metric contract: a coin flip can hand one rank a batch with
no blackout samples, and `loss_action_fallback` would then be missing on that rank while present on
another, desynchronising the all-gather. Fixed counts make every regime present in every batch.

## 4.7 M6 — losses, diagnostics, and the stage bridge

```text
FLUX.2 gives:   nothing; upstream's objective is image-only
You add:        _training_loss_flux2_{baseline,stage1,stage2}, _goal_prior_diagnostics,
                validate_goal_prior_checkpoint_keys
                — imagewam.py:2652, imagewam.py:2731, imagewam.py:2828,
                  imagewam.py:2971, goal_pose_prior.py:734
Check:          every rank emits every diagnostic key on every step.
```

The dispatcher is four lines: `goal_prior_stage` selects the loss (`imagewam.py:2644`).

```python
def loss_stage2(sample):                                     # imagewam.py:2828
    inputs = build_inputs_flux2(sample)                       # text, ref, target, action, proprio
    noise_v, t_v = train_video_scheduler.sample(...)          # independent timesteps per stream
    noise_a, t_a = train_action_scheduler.sample(...)

    syn_keep  = agg.sample_context_keep_mask(B, device)       # may be None
    ref_keep, syn_channel_keep = agg.sample_channel_regime(B, device)
    syn_keep = (syn_keep & syn_channel_keep[:, None]) if syn_channel_keep is not None else syn_keep

    action_mask = build_stage2_action_attention_mask(..., ref_len=cond_len if agg.action_sees_ref else 0)
    out = mot(video=..., action=..., attention_mask={... "action": action_mask},
              context_all={"goal_prior": {"aggregator": agg, ...}})

    pred_pose = semantic_visual_pose_decoder(                 # the 8 pose latents only
        semantic_visual_pose_norm(out.goal_latents[:, :8]))
    loss = (0.5 * mse(post_dit(out.video), target_video)      # imagewam.py:2952
          + 1.0 * weighted_action_loss(out.action, target_action)
          + 0.3 * compute_pose_reconstruction_loss(pred_pose, goal_pose))
```

The diagnostics are contractual, not decorative — the trainer all-gathers one collective per metric key,
so a key present on some ranks and absent on others hangs the job:

```python
def diagnostics(...):                                        # imagewam.py:2971
    dark = ~syn_keep[:, 8:].any(dim=1)                       # samples that lost the whole context
    out["gate/bias_mean"]      = mean([g.syn_gate_bias for g in agg.groups])
    out["blackout_frac"]       = dark.float().mean()
    out["loss_action_fallback"] = weighted[dark].mean()  if dark.any() else overall
    out["loss_action_steered"]  = weighted[~dark].mean() if (~dark).any() else overall
    # every key above is emitted unconditionally, even when the feature is off
```

The checkpoint contract is the last piece, and it is **fail-closed**: the bridge from stage 1 to stage 2
permits exactly two kinds of key difference and raises on everything else (`goal_pose_prior.py:734`):

| | keys | why |
| --- | --- | --- |
| dropped | `goal_pose_encoder.*`, `stage1_null_image_tokens` | stage 1's prior interface; stage 2 does not read it |
| randomly initialised | `semantic_visual_aggregator.*`, `semantic_visual_pose_norm.*`, `semantic_visual_pose_decoder.*` | the new vision path — there is nothing to inherit |
| anything else | — | raises: stage 2 must never silently train from the ActionDiT init |

## 4.8 M7 — inference: prefill once, denoise N times

```text
FLUX.2 gives:   Flux2.forward (image-only) plus the block seams from M0
You add:        prefill_flux2_video_cache / prefill_flux2_goal_prior_cache,
                forward_flux2_action_with_{video,goal_prior}_cache
                — mot.py:1023, mot.py:1158, mot.py:1080, mot.py:1297
Check:          the cached prefill reproduces recomputing the video stream every step.
```

The hoist is the whole idea: the video half depends only on the reference image and the prompt, which
do not change while the action chunk is being denoised (§4.3 is what makes this legal).

```python
def infer_chunk(image, prompt):                          # imagewam.py:4360
    text = qwen3(prompt)                                 # cached embeddings in practice
    ref, ref_ids = vae_encode(image)                     # one frame → 392 tokens at 1/16
    pre  = video_expert.pre_dit(x=zeros(B, 0, 128), timestep=0,
                                ref_image_hidden_states=ref, ...)     # target_len = 0
    cache = mot.prefill_flux2_goal_prior_cache(pre, ..., goal_prior)  # ONCE per chunk
    x = randn(B, 16, 7)
    for t, d_sigma in build_inference_schedule(N):        # scheduler_continuous.py:63
        a_pre = action_expert.pre_dit(x, t)
        tok   = mot.forward_flux2_action_with_goal_prior_cache(a_pre, cache, action_mask)
        x     = x + action_expert.post_dit(tok, a_pre) * d_sigma       # Euler, scheduler:83
    return x
```

What the cache holds per layer is worth stating, because it *is* the LIT path's runtime cost model:
`txt_k/v`, `ref_k/v`, `syn_k/v`, `gate_bias`, `gated_span`, and `mode` (`mot.py:1227`). The aggregator
therefore runs **25 times per chunk**, not 25 times per step — the synthetic channel and the gate bias
are frozen for the whole denoising loop, and the only thing that varies across steps is the action
stream itself.

The inference mode is resolved once and then expressed purely through the two lengths:

```python
mode = _resolve_infer_mode()                             # imagewam.py:4238
action_mask = build_stage2_action_attention_mask(
    synthetic_len = cache["synthetic_len"] if mode in ("full", "firewall") else 0,
    ref_len       = cond_len               if mode in ("full", "ref_only") else 0)
```

So `firewall` is literally the same call the firewall-mode *training* makes, and `full` the same call
plan B makes — deployment re-uses the mask builder rather than re-deriving the layout.

## Part 4 — things that are easy to get wrong

- **Upstream's `forward_kv_extract` is a decoy.** It exists, it caches K/V, and it is the wrong shape
  for this: causal, single-stream, reference-image-only (`model.py:170`). The seam LIT needs is the
  private q/k/v split — `_prepare_qkv` / `_apply_residuals` (`model.py:569`, `model.py:614`) and
  `_qkv` / `_out` (`model.py:468`, `model.py:482`). Expect to vendor the blocks.
- **Stage 2 is not a superset of stage 1's attention.** Baseline and stage 1 run one joint SDPA over
  `[video | action]`; stage 2 splits it into two, and the video stream stops reading action entirely
  (`mot.py:863` vs `mot.py:626`). Anyone re-deriving the architecture from Part 2 alone will get this
  wrong.
- **The two experts must agree on head geometry, and the config will not tell you.** `num_heads`,
  `attn_head_dim` and both layer counts are overwritten from the video expert at construction
  (`imagewam.py:661`); `hidden_dim` is the only free number.
- **Only the reference image reaches the aggregator.** `_flux2_update_goal_latents` passes
  `img[:, :cond_len]` (`mot.py:757`) — the noisy target is sliced away, and at inference there is no
  target at all.
- **`key_norm` is per-head, and RoPE applies to the key only.** Viewing to 24 × 128 is part of the
  norm, not just bookkeeping (`goal_pose_prior.py:299`), and `apply_rope` is called with a dummy query
  because only `k` survives (`mot.py:717`).
- **The gate cannot be a bool mask.** It is a learnable logit bias, so the mask has to become an
  additive float tensor first; `zero_init_value` + an open pose gate is a contradiction the
  constructor rejects (`goal_pose_prior.py:373`).
- **Fixed-count sampling is a correctness property, not a statistical one.** Regime counts and blackout
  counts are drawn so every rank emits every key on every step (`goal_pose_prior.py:478`,
  `imagewam.py:3003`); switching them to per-sample coins will hang the job rather than skew a metric.
- **The synthetic time id is 3.0 and it matters.** Synthetic tokens occupy axis 0 = 3.0, distinct from
  text (axis 3), reference (10.0), target (0.0) and action (2.0) — reusing any of those silently makes
  the synthetic channel positionally indistinguishable from a real one (`goal_pose_prior.py:617`).
- **`stage1_null_image_tokens` are training-only.** They stand in for the missing reference during
  training but not at inference (`imagewam.py:2038` vs `imagewam.py:4310`), so a stage-1 checkpoint
  should be scored by calling `infer_action_flux2(goal_pose=…)` directly — the shipped evaluators
  cannot supply one (§3.4).

# Part 5 — training, the original implementation

Parts 1–4 describe the model. This part describes what the training loop actually *optimises* — which is
not readable off the model, because two separate gates narrow the parameter set and the second is not
mentioned anywhere near the first.

## 5.0 The two gates

```text
   trainer.py:543        model.requires_grad_(False)
        │                     everything, the VAE included
        ▼
   trainer.py:545        model.dit.train()  /  requires_grad_(True)
        │                     dit IS mot — the same object
        ▼
   imagewam.py:5364      model.apply_trainable_policy()
        │                     the flux2-only refinement
        ▼
   trainer.py:552        proprio_encoder.requires_grad_(True)
                              unconditional, in every run type
```

**Gate 1** is `_apply_dit_only_train_mode` (`trainer.py:541`). It runs *before* the optimiser is built
(`trainer.py:112` vs `trainer.py:126`) so that ZeRO allocates state over the right tensors, and it is
re-applied at `train()` entry (`trainer.py:1150`) and again after validation restores the model
(`trainer.py:1010`) — the whitelist is not a one-time setup.

**Gate 2**, `apply_trainable_policy` (`imagewam.py:5364`), is a no-op for every stack except `flux2`.

What the optimiser actually receives is not a whitelist but the `requires_grad` flag:
`collect_trainable_parameters` (`imagewam.py:5404`) gathers `dit` + `proprio_encoder` + the goal-prior
modules and drops anything still frozen (`imagewam.py:5412`). The resulting counts are logged at INFO
(`trainer.py:484`), grouped by `mot.mixtures.video` / `mot.mixtures.action`
(`imagewam.py:5421`).

## 5.1 What each run trains

| run | video expert | action expert | LIT modules | proprio encoder |
| --- | --- | --- | --- | --- |
| baseline | fully trained | fully trained | — | trained |
| baseline + LoRA | LoRA only | fully trained | — | trained |
| stage 1 | frozen (`.eval()`) | fully trained | `goal_pose_encoder` | trained |
| stage 2 | fully trained | fully trained | aggregator, pose norm, pose decoder | trained |

Two things the table does not show:

- **`proprio_encoder` is trained in all four run types**, including the baselines. Gate 1 re-enables it
  unconditionally *after* gate 2 has had its say (`trainer.py:552`), and `imagewam.py` never
  distinguishes.
- **One optimiser, one param group.** `AdamW(trainable, lr=learning_rate, weight_decay=weight_decay,
  betas=(0.9, 0.95))` with the betas hardcoded (`trainer.py:126`). There is no per-group learning rate
  anywhere in the trainer; the only lr knob is the task config (`trainer.py:36`) — `2.0e-4` for stage 1
  (`configs/task/libero_flux2_klein_4b_goal_prior_stage1.yaml:22`), `1e-4` for stage 2 and the base runs.

Stage 1's freeze is unusual in one respect: the video expert is frozen and in `.eval()`, but it is
**not** under `no_grad`. All 25 layers still run through autograd because the gradient has to reach
`goal_pose_encoder` through them — and the checkpointing gate reads `mot.training` (`mot.py:557`), which
gate 1 sets and `video_expert.eval()` does not clear, so those frozen layers are still
gradient-checkpointed.

## 5.2 LoRA on the video expert

LoRA is injected at **construction**, not by the policy: `from_flux2_klein_pretrained` calls
`apply_lora_to_linear_suffixes(video_expert.transformer, ...)` (`imagewam.py:642`) and records the
outcome on `video_expert.flux2_lora_enabled` (`imagewam.py:649`). `LoRALinear` freezes its own base in
its constructor (`lora.py:38`), so the freeze travels with the module rather than with the policy.

**Which stage may use it.** Stage 2 only. `apply_trainable_policy` keeps its LoRA branch for the
baseline, and gives Stage 2 a LoRA variant beside the full-fine-tune one (`imagewam.py:5403`):

| Run | video expert |
| --- | --- |
| baseline, LoRA off | fully trainable |
| baseline, LoRA on | base frozen, adapters trainable |
| **stage 1** | fully frozen — LoRA or not |
| **stage 2**, LoRA off | fully trainable (the evaluated revision) |
| **stage 2**, LoRA on | **base frozen, adapters trainable** |

Stage 1 cannot use it, and the policy logs a warning when `enabled` is set there: the stage freezes the
whole video expert because the gradient that reaches `goal_pose_encoder` travels through activations,
not weights — so adapters built for a Stage 1 run would never move.

**What it trains.** The shipped `target_suffixes` hit **80 Linear layers** holding **3,774.9M** of the
expert's 3,875.5M (97.4%), with the adapter cost set by rank alone (bf16, on disk):

| rank | adapter params | % of video expert | file |
| ---: | ---: | ---: | ---: |
| 8 | 11.80M | 1.18% | 24 MB |
| **16** (default) | **23.59M** | **2.36%** | 47 MB |
| 32 | 47.19M | 4.72% | 94 MB |
| 64 | 94.37M | 9.44% | 189 MB |

Scaling is `alpha/rank`; the configs ship `alpha == rank` for a scaling of 1.0. For scale, the 23.6M
rank-16 adapter is *smaller than the 165M `SemanticVisualAggregator` training beside it* (§2.1.1).
Targets are `qkv`, `proj`, `linear1`, `linear2`, `img_mlp.0`, `img_mlp.2`, `txt_mlp.0`, `txt_mlp.2`
(`configs/model/imagewam_flux2_klein_4b_base.yaml:17`).

**Saving is the sharp edge.** A LoRA-trained checkpoint **must** be saved merged
(`save_lora_merged: true`), because the stage-1 bridge and the eval loader rebuild the model *without*
adapters and `load_checkpoint` is fail-closed on missing MoT keys (`imagewam.py:5280`). The launcher
sets the flag whenever `FLUX2_LORA_ENABLED=true`; enabling LoRA by hand in a YAML does not, and the
constructor warns. Saving *unmerged* also renames every key (`base_layer.weight`) and roughly doubles
the payload with frozen base weights, which nothing in the repo reads back.

For the same reason the constructor **rejects** `save_trainable_only` together with LoRA: under the LoRA
policy the base weights are frozen, so that filter would keep the adapters and drop the 3.876B they are
applied to.

Loading runs the MoT state dict through four steps before `load_state_dict`
(`imagewam.py:5299`). A plain-key checkpoint loads into a LoRA-built model, and a merged payload loads
into either.

| Step | Why |
| --- | --- |
| `merge_lora_state_dict_to_plain` | Folds an *unmerged* payload's adapters into base weights, so both checkpoint flavours read the same. |
| `strip_mot_prefix` | The payload is `self.mot.state_dict()`, but called on the expert those keys come back without the leading `mot.`. The helpers below derive names from `named_modules()`, so a prefixed payload matches nothing and every key is reported missing *and* unexpected. |
| `remap_plain_linear_keys_to_lora_base` | Renames a plain `….qkv.weight` to the wrapper's `….qkv.base.weight`. |
| `collapse_aliased_transformer_keys` | The alias problem below. |

The alias step is the one that bites, and it only exists because of how the video expert is built. It
binds the FLUX.2 blocks **twice** (`flux2_video_expert.py:22`):

```python
self.transformer = transformer
self.double_blocks = transformer.double_blocks   # flattened alias onto the expert
```

so a LoRA-built MoT state dict lists every block weight under both `…video.transformer.double_blocks.…`
and `…video.double_blocks.…`. The adapters live only on the `transformer` path, so the remap rewrites
that key and leaves its alias twin unmapped — the two halves then disagree, and because
`load_state_dict` walks each registered path separately it reports one missing plus one unexpected key
per block and the exact-load check rejects the load outright. Before LoRA this was invisible: plain keys
found a home under either path, so the duplicate never mattered.

`collapse_aliased_transformer_keys` repopulates both paths from the wrapped value, which is canonical —
it owns the adapters and the merged base weights — and drops any key that addresses no registered
parameter. Two consequences worth knowing:

- **The same bug lives on the save side.** `lora_merged_state_dict` walks the module state dict, so an
  un-collapsed save writes each block weight a second time under the flattened name.
- **`lora_A`/`lora_B` are parameters *of* the `LoRALinear`**, not children under a submodule, so the
  prefix filter that skips wrapped modules does not skip them. Without an explicit guard a checkpoint
  tagged `lora_merged` also carries the raw adapters — visible as `lora_*` and `.base.*` counts of the
  same size in a saved payload.

**Costs to weigh before enabling it.** Freezing the expert removes its ability to adapt to the LIT
regime: Stage 2's `0.5 · L_video` currently tunes the FLUX backbone to the goal-prior setup, and under
LoRA that becomes a rank-16 edit to eight projection families per block. The modulation projections
(`double_stream_modulation_img/txt`, `single_stream_modulation`, 141.6M total) and `final_layer` are
**not** targeted, so the timestep conditioning and output head stay frozen outright.

## 5.3 What is never trained

- **The VAE.** Built at `imagewam.py:686` and registered as a submodule, so gate 1's blanket freeze
  reaches it and nothing re-enables it.
- **The text encoder — which structurally cannot be trained.** The Qwen3 model is wrapped in a
  `types.SimpleNamespace` rather than an `nn.Module` (`imagewam.py:696`), so its parameters are invisible
  to `parameters()`, `state_dict()` and `requires_grad_()`; they never reach
  `collect_trainable_parameters`. Every flux2 config additionally sets `load_text_encoder: false` and
  reads precomputed embeddings instead.
- **`stage1_null_image_tokens`.** Despite the name and the config comment, this parameter is **frozen**:
  it is a bare `nn.Parameter` on the top-level module (`imagewam.py:159`), so gate 1's
  `model.requires_grad_(False)` reaches it; `model.dit.requires_grad_(True)` (`trainer.py:545`) covers
  only the MoT subtree; and stage 1's branch of gate 2 re-enables the action expert and the goal encoder
  only (`imagewam.py:5374`). The `requires_grad` filter in `collect_trainable_parameters`
  (`imagewam.py:5412`) then drops it from the optimiser. It *is* listed by `goal_prior_parameters()`
  (`imagewam.py:245`) and *is* saved and restored (`imagewam.py:5242`, `imagewam.py:5330`), so it
  round-trips through a checkpoint without ever being updated.

## Part 5 — things that are easy to get wrong

- **`stage1_null_image_tokens` is listed but not optimised.** Being present in `goal_prior_parameters()`
  is not sufficient — the `requires_grad` filter is what decides, and gate 1 has already frozen it. The
  A2 fix recorded in `CHANGELOG_v2.md:332` covers the listing plus save/load; the filter still excludes
  it, so the token keeps its `trunc_normal(std=0.02)` initialisation for the whole run. Fixing it means
  re-enabling it in stage 1's branch, next to `goal_pose_encoder`.
- **`save_trainable_only: true` produces a checkpoint that cannot be loaded back into a goal-prior
  run.** It whitelists the MoT subtree only, and a stage-1 run's trainable MoT set is the action expert
  alone — so the video expert's weights are absent. `load_checkpoint` raises on any missing MoT key when
  the loading model is in stage 1 or stage 2 (`imagewam.py:5280`). All shipped configs leave it `false`.
  The same flags do *not* restrict the other modules: `proprio_encoder`, `goal_pose_encoder`,
  `stage1_null_image_tokens` and the three aggregator modules are always saved in full
  (`imagewam.py:5241`). Combining it with `flux2_lora_config.enabled` is worse still — under the LoRA
  policy the base weights are frozen, so the filter keeps the adapters and drops the 3.876B they apply
  to. The constructor now raises on that combination (`imagewam.py:740`) rather than writing it out.
- **Enabling LoRA without `save_lora_merged` writes checkpoints nothing can read.** The payload keeps
  the wrapped module names (`*.base_layer.weight`) and carries every frozen base weight, so it roughly
  doubles in size and still fails the fail-closed MoT load. The constructor warns; the launcher forces
  the flag; a hand-written Hydra override does neither.
- **`save_lora_merged: true` with `enabled: false`.** `configs/model/imagewam_flux2_klein_9b_base.yaml`
  sets the flag at `:22` while disabling LoRA at `:18`. The save path takes the merge branch with no
  LoRA modules present, so the tensor content is an ordinary full state dict while
  `checkpoint_format` is tagged `lora_merged` (`imagewam.py:5213`). Nothing in the repo reads that tag
  back, so it is cosmetic — but it is a lie in the checkpoint metadata.
- **The checkpointing gate is `mot.training`, not `video_expert.training`** (`mot.py:557`). Stage 1's
  frozen video layers are therefore still recomputed for backward — that recomputation is required to
  carry gradient to `goal_pose_encoder`, not an oversight.
- **Gate 1 runs before the optimiser, so "what trains" is decided once and then re-asserted.** If a run
  appears to train a module that should be frozen, check `apply_trainable_policy`'s ordering rather than
  the config — three apply sites (`trainer.py:112`, `:1150`, `:1010`) all call the same policy.
