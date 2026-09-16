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
   action ──► ActionDiTFlux2.pre_dit   ──┘                    └── post_dit ──► pred_action
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
              key ►   txt      ref     target   action
          ┌────────┬────────┬────────┬────────┐
 txt/ref  │   ✓    │   ✓    │   ·    │   ·    │   stable prefix:
          ├────────┼────────┼────────┼────────┤   never reads noisy tokens
 target   │   ✓    │   ✓    │   ✓    │   ·    │
          ├────────┼────────┼────────┼────────┤
 action   │   ✓    │   ✓    │   ·    │   ✓    │   never reads the noisy target
          └────────┴────────┴────────┴────────┘
```

This is the layout the baseline (non-prior) path uses with every block populated. **Stage 1 passes
`target_len=0`**, so the target columns simply do not exist and the action queries see
`[txt | pose | null? | action]` — the "stable prefix" row and the action row are all that remain.
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

Same spine as stage 1, with the image rail restored and the aggregator spliced into the layer loop:

```text
        ref image            prompt                 noisy action
            │                   │                         │
            ▼                   ▼                         ▼
       ┌── VAE ──┐       ┌─── text ────┐           action [B,16,7]
       │   VAE   │       │    Qwen3    │                  │
       │  encode │       │  9, 18, 27  │                  ▼
       └────┬────┘       └──────┬──────┘        ┌───── action ──────┐
            │                   │               │   action_expert   │
            ▼                   ▼               │      .pre_dit     │
       ref tokens        txt [B,L,7680]         └─────────┬─────────┘
            │                   │                         │
            └─────────┬─────────┘                         │
                      ▼                                   │
            ┌────── video ──────┐               ┌───── action ──────┐
            │    video_expert   │               │   action_expert   │
            │      .pre_dit     │               │      .pre_dit     │
            └─────────┬─────────┘               └─────────┬─────────┘
                      │                                   │
                      └─────────────────┬─────────────────┘
                                        │ ← LIT blocks attach in here (§2.8–2.11)
                                        ▼
                     ┌─────────────── loop ────────────────┐
                     │           MoT — 25 layers           │
                     │         5 double + 20 single        │
                     │      video block | action block     │
                     └──────────────────┬──────────────────┘
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
            video_expert.post_dit              action_expert.post_dit
                      │                                   │
                      ▼                                   ▼
            pred_video [B,N,128]                pred_action [B,16,7]
```

The VAE (`self.vae`) and Qwen3 (`self.text_encoder`) are separate modules feeding the video expert, but
the action chunk goes straight into `action_expert.pre_dit` — its first step is
`action_expert.action_encoder`. Unlike stage 1, the video expert becomes trainable here, and the goal
tokens are no longer used.

Where each stage-2 block attaches — these are the only new pieces:

| Block | Reads | Writes into | Section |
| --- | --- | --- | --- |
| `SemanticVisualAggregator` | txt + ref-image features out of each video block | `goal_latents`, carried across all 25 layers | §2.8 |
| `to_key` / `to_value` | `goal_latents`, 100 × 768 | the action queries' K/V, per layer | §2.9 |
| `pose_norm` + `GoalPoseDecoder` | `goal_latents[:, :8]` | `L_pose` (weight 0.3) | §2.8 |
| cold-start gate | — | the action attention logits, as a float mask | §2.11 |
| firewall / plan-B regime | — | which of those channels are visible at all | §2.11 |

The prior writes to the **action** stream only. The video stream runs as Part 1 describes; stage 2
reads from it but never writes to it.

Loss: `0.5 · L_video + 1.0 · L_action + 0.3 · L_pose`.

### 2.8 Micro — `SemanticVisualAggregator` (`goal_pose_prior.py:309`)

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

### 2.9 Micro — `SemanticVisualAggregatorGroup` (`goal_pose_prior.py:243`)

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
(§2.11). `zero_init_value` optionally zeroes `to_value`; it is **mutually exclusive** with
`gate_pose_tokens=False` and raises (`goal_pose_prior.py:373`) — see the comment there for why the
gate must then be carried by the logit bias alone.

### 2.10 Micro — synthetic K/V projection (`mot.py:694`)

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

### 2.11 Micro — firewall, plan B, cold-start gate

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

### 2.12 Micro — inside the layer loop

The MoT box in the stage-2 macro is recurrent: the aggregator is re-run after *every* video block and
hands updated latents forward. This is the connection the flat spine can't show:

```text
latents — 100 learnable queries × 768
                 │                      initialised once, before layer 0  (§2.8)
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
