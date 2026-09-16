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
- **The four RoPE axes are modality-disjoint.** Text uses axis 3, image uses 0–2, action and synthetic
  tokens use axis 1 with a distinct time value. Ref vs target is distinguished by the image
  `time_value` (10.0 vs 0.0), *not* by the upstream ref-timestep blending, which this wrapper never calls.
- **Latent channels are 128, not 32.** The VAE's `z_channels` is 32; the 2×2 patchify is what makes it
  128 by the time the DiT sees it.

---

# Part 2 — LIT goal-pose prior model

Everything in Part 1 is frozen or reused as-is. Part 2 covers what this fork builds around it: the MoT
joint attention that attaches an action expert to the image model (§2.1–2.3), and the goal-pose prior
itself (§2.4–2.8).

## 2.0 Macro block diagram

```text
 ┌─ Stage 1 — vision-free prior ─────────────────────────────────────────┐
 │                                                                       │
 │   goal_pose [B,8] ──► GoalPoseEncoder ──► 8 tokens × 3072             │
 │                          (8→512→512→8·3072)                           │
 │                              │                                        │
 │                              ▼                                        │
 │                    appended to the text stream                        │
 │                    → action K/V layout [txt | pose | action]          │
 └───────────────────────────────────────────────────────────────────────┘

 ┌─ Stage 2 — synthetic K/V prior ───────────────────────────────────────┐
 │                                                                       │
 │   txt ──┐                                                             │
 │   ref ──┴──► SemanticVisualAggregator                                 │
 │                100 latents, dim 768, 5 layer-groups                   │
 │                     │                                                 │
 │        ┌────────────┴────────────┐                                    │
 │        │                         │                                    │
 │        ▼                         ▼                                    │
 │   8 pose latents            100 latents                               │
 │        │                         │                                    │
 │        ▼                         ▼                                    │
 │  pose_norm +             to_key / to_value (KV dim 3072)              │
 │  GoalPoseDecoder                 │                                    │
 │        │                         ▼                                    │
 │        ▼                  synthetic K/V, per layer                    │
 │  L_pose (×0.3)                   │                                    │
 │                                  ▼                                    │
 │                       action K/V layout [txt | ref? | syn | action]   │
 └───────────────────────────────────────────────────────────────────────┘
```

The prior attaches to the **action** stream only. The video stream keeps running as Part 1 describes;
Stage 1 replaces the image with 8 pose tokens, Stage 2 feeds the aggregator from the video stream's
per-layer features.

## 2.1 Micro — MoT joint attention (`mot.py:22`)

`MoT` holds `mixtures = {"video": Flux2VideoExpert, "action": ActionDiTFlux2}` and validates that both
agree on `num_layers`, `num_heads`, `num_kv_heads`, `attn_head_dim`, `block_protocol` (`mot.py:53`).

Because both experts emit q/k/v of width 24 × 128 = 3072, they concatenate **with no projection**.
Only the residual streams differ in width (3072 vs 1024):

```text
  layer_idx i of 25  (5 double, then 20 single)

     video expert                                    action expert
  ┌──────────────────┐                            ┌──────────────────┐
  │ DoubleStreamBlock│                            │SlimFlux2Double   │
  │  3072-wide       │                            │  Block, 1024-wide│
  │  own weights     │                            │  own weights     │
  └────────┬─────────┘                            └────────┬─────────┘
           │ q,k,v  [B, L_v, 3072]                         │ q,k,v  [B, L_a, 3072]
           └───────────────────┬───────────────────────────┘
                               ▼
                  cat along sequence (dim=1)
                               │
                          SDPA + mask
                               │
                  split back at L_v
           ┌───────────────────┴───────────────────────────┐
           ▼                                               ▼
     video attn out                                  action attn out
     (residual + MLP in video weights)               (residual + MLP in action weights)
```

The per-layer function is wrapped in `torch.utils.checkpoint` when `mot_checkpoint_mixed_attn` is set
and the module is training (`mot.py:557`) — needed because frozen-FLUX stage 1 still runs autograd
through all 25 layers to reach the goal encoder.

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

## 2.2 Micro — `ActionDiTFlux2` (`action_dit_flux2.py:146`)

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

## 2.3 Micro — baseline action attention mask (`imagewam.py:2312`)

A block-structured **bidirectional** keep-mask over `[txt | ref | target | action]` (there is no causal
masking here, despite `causal_attn_fn` existing in the upstream ref-cache path). The same tensor is
used for both `double_joint` and `single`:

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

`text_attention_mask` additionally zeroes padding columns of the txt block (`imagewam.py:2343`).
`_mixed_attention` accepts either a bool mask (keep-mask) or a **float mask treated as an additive
bias** (`mot.py:146`) — that second path is what the cold-start gate in §2.7 uses.

## 2.4 Micro — `GoalPoseEncoder` (stage 1, `goal_pose_prior.py:85`)

```text
   goal_pose [B, 8]
        │
   Linear(8 → 512) ─► GELU ─► Linear(512 → 512) ─► GELU ─► Linear(512 → 8·3072)
        │
   reshape [B, 8, 3072]
        │
   concatenated onto the text stream   (§2.8, imagewam.py:2060)
```

The 8 output tokens use the **FLUX hidden interface (3072)**, not the 7680-D Qwen one — they are
appended after `txt_in`, and `_append_goal_tokens_to_flux2_pre` extends `txt_pe` and `text_mask` to
match.

## 2.5 Micro — `SemanticVisualAggregator` (stage 2, `goal_pose_prior.py:309`)

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
- The **latent update happens every layer, but the projection to K/V also happens every layer** —
  the 100 latents are re-projected per layer through that layer's group, not carried across wholesale.
- **B1 `latent_layout: grid`** (`goal_pose_prior.py:513`) ties the context latents to a 7×7 grid so each
  sees only its own neighbourhood of image tokens; `free` (the evaluated default) gives all latents
  global attention.

## 2.6 Micro — `SemanticVisualAggregatorGroup` (`goal_pose_prior.py:243`)

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
(§2.7). `zero_init_value` optionally zeroes `to_value`; it is **mutually exclusive** with
`gate_pose_tokens=False` and raises (`goal_pose_prior.py:373`) — see the comment there for why the
gate must then be carried by the logit bias alone.

## 2.7 Micro — stage-2 action attention: firewall, plan B, cold-start gate

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

## 2.8 Micro — the two-stage training flow

```text
  Stage 1 — vision-free                       Stage 2 — needs a Stage 1 checkpoint

  goal_pose = proprio[-1]                     build_inputs_flux2 (real ref + target)
        │                                            │
        ▼                                            ▼
  GoalPoseEncoder ─► 8 tokens                 video_expert.pre_dit
        │                                            │
        ▼                                            ▼
  append to txt stream                        mot._forward_flux2_stage2
  [txt | pose | action]                              │
        │                                     ┌──────┴──────┐
        ▼                                     ▼             ▼
  mot._forward_flux2                   aggregator updates  action K/V
        │                              latents per layer   [txt|ref?|syn|act]
        ▼                                     │
  L_action only                               ▼
  (no image stream,                    0.5·L_video + 1.0·L_action + 0.3·L_pose
   no L_video, no L_pose)
```

Stage 2 **requires** `stage1_checkpoint` unless `resume` is set, so it can never silently train from
the ActionDiT init. The checkpoint bridge is fail-closed; the only permitted key differences come from
the prefix constants at `goal_pose_prior.py:64`:

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
- **Per-regime samplers use fixed counts, not independent coins** (`goal_pose_prior.py:462`). Every rank
  must see every regime in every batch, or the per-key all-gather in the trainer desynchronises NCCL
  and hangs the job.
- **Metrics must be emitted on every rank, every step** — `_goal_prior_diagnostics` always emits every
  key for the same reason.
