"""MolmoAct2 V3 goal-pose prior modules adapted to FLUX.2 ImageWAM.

Stage1 encodes an oracle future pose into 8 FLUX hidden tokens.
Stage2 infers 100 recurrent latents (8 pose + 92 context) from language/state
and current-image tokens, then steers ActionDiT through synthetic K/V.
Stage1b is Stage2 without the Action Expert: the aggregator keeps only the pose
latents and is trained by the pose reconstruction loss alone.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

GOAL_PRIOR_NUM_GOAL_TOKENS = 8
GOAL_PRIOR_NUM_LATENTS = 100
GOAL_PRIOR_NUM_POSE_TOKENS = 8
GOAL_PRIOR_NUM_GROUPS = 5
GOAL_PRIOR_LATENT_DIM = 768
GOAL_PRIOR_INNER_DIM = 512
GOAL_PRIOR_FFN_RATIO = 4.0
GOAL_PRIOR_NUM_HEADS = 8
GOAL_PRIOR_DROPOUT = 0.0

# --- bottleneck fixes; all default to the evaluated behaviour ---------------
# B2: fraction of *context* tokens whose synthetic K/V columns are masked out
# each training step. Pose tokens are never dropped. Forces the Action Expert
# to spread its dependence instead of leaning on one aggregate.
GOAL_PRIOR_CONTEXT_TOKEN_DROPOUT = 0.0
# B2+: probability of masking the *entire* context block for a given sample.
# The surviving state is [txt | pose | action] -- Stage 1's prior interface --
# so the Action Expert is forced to keep a working policy that does not depend
# on the context aggregate, and can fall back to it when that aggregate is
# corrupted out of distribution.
GOAL_PRIOR_CONTEXT_BLACKOUT_PROB = 0.0
# B4: zero-init `to_value` so the synthetic channel starts as a no-op, and bias
# its attention logits down so it also consumes no softmax mass at step 0.
GOAL_PRIOR_ZERO_INIT_VALUE = False
GOAL_PRIOR_SYN_GATE_BIAS_INIT = 0.0  # set to about -5.0 to cold-start
# Whether the cold-start gate also covers the 8 pose columns. False keeps them
# open, so step 0 matches Stage 1's [txt | pose | action] topology and agrees
# with what B2+ blackout falls back to. True reproduces the older behaviour of
# gating the whole synthetic block.
GOAL_PRIOR_GATE_POSE_TOKENS = False
# Plan B. The Action Expert attends the reference image again, and training
# samples which channels it may use. Probabilities must sum to 1; "both" is the
# inference condition, "syn only" preserves firewall mode as an evaluable
# capability rather than an architectural constraint.
GOAL_PRIOR_ACTION_SEES_REF = False
GOAL_PRIOR_P_BOTH = 0.55
GOAL_PRIOR_P_REF_ONLY = 0.15
GOAL_PRIOR_P_SYN_ONLY = 0.30
# B1: "free" = 100 unconstrained latents attending the whole image (evaluated
# behaviour). "grid" ties the context latents to a coarse spatial grid and lets
# each one attend only its own neighbourhood, so a global perturbation can no
# longer move every latent together.
GOAL_PRIOR_LATENT_LAYOUT = "free"
GOAL_PRIOR_GRID_HW = (7, 7)
GOAL_PRIOR_GRID_WINDOW = 1  # neighbourhood radius in coarse cells
GOAL_PRIOR_POSE_LOSS_WEIGHT = 0.3
GOAL_PRIOR_SYNTHETIC_TIME_VALUE = 3.0

STAGE1_ONLY_CHECKPOINT_PREFIXES = ("goal_pose_encoder.", "stage1_null_image_tokens")
STAGE2_ONLY_CHECKPOINT_PREFIXES = (
    "semantic_visual_aggregator.",
    "semantic_visual_pose_norm.",
    "semantic_visual_pose_decoder.",
)

# Stage 1b: Stage 2's aggregator and pose head, with the Action Expert excluded
# from the graph and the objective. The model still *builds* the Action Expert
# (deleting the module would break the checkpoint key contract, the Stage1->2
# bridge and the inference path, and the action stream is ~16 tokens against
# 784 image tokens, so nothing is saved by removing it) -- it is simply never
# invoked, never receives a loss, and is frozen.
STAGE1B = "stage1b"
GOAL_PRIOR_STAGES = ("stage1", "stage2", STAGE1B)
# The knobs that only shape the Action Expert's view of the synthetic channel.
# Under Stage 1b there is no Action Expert, so setting any of these to a
# non-default value is silently inert -- the failure mode that once produced a
# 35-hour run in which every new mechanism was inactive.
ACTION_FACING_GOAL_PRIOR_KNOBS = (
    "context_token_dropout",
    "context_blackout_prob",
    "action_sees_ref",
    "p_both",
    "p_ref_only",
    "p_syn_only",
    "zero_init_value",
    "syn_gate_bias_init",
    "gate_pose_tokens",
)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_f = x.float()
        rrms = torch.rsqrt(torch.mean(x_f**2, dim=-1, keepdim=True) + self.eps)
        return (x_f * rrms).to(dtype=x_dtype) * self.scale


class GoalPoseEncoder(nn.Module):
    """Maps a normalized goal pose vector to K FLUX hidden tokens."""

    def __init__(
        self,
        pose_dim: int,
        num_tokens: int = GOAL_PRIOR_NUM_GOAL_TOKENS,
        hidden_size: int = 3072,
        inner_dim: int = GOAL_PRIOR_INNER_DIM,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(self.pose_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, self.num_tokens * self.hidden_size),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 2:
            raise ValueError(f"`pose` must be [B, pose_dim], got {tuple(pose.shape)}")
        if pose.shape[1] != self.pose_dim:
            raise ValueError(f"`pose` last dim must be {self.pose_dim}, got {pose.shape[1]}")
        return self.net(pose).view(pose.shape[0], self.num_tokens, self.hidden_size)


class GoalPoseDecoder(nn.Module):
    """Reconstructs a normalized goal pose from concatenated pose latents."""

    def __init__(
        self,
        num_tokens: int = GOAL_PRIOR_NUM_POSE_TOKENS,
        hidden_size: int = GOAL_PRIOR_LATENT_DIM,
        pose_dim: int = 8,
        inner_dim: int = GOAL_PRIOR_INNER_DIM,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        self.pose_dim = int(pose_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_tokens * self.hidden_size, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, self.pose_dim),
        )

    def forward(self, goal_hidden: torch.Tensor) -> torch.Tensor:
        if goal_hidden.ndim != 3:
            raise ValueError(f"`goal_hidden` must be [B, K, D], got {tuple(goal_hidden.shape)}")
        if goal_hidden.shape[1] != self.num_tokens or goal_hidden.shape[2] != self.hidden_size:
            raise ValueError(
                f"`goal_hidden` must be [B, {self.num_tokens}, {self.hidden_size}], got {tuple(goal_hidden.shape)}"
            )
        return self.net(goal_hidden.reshape(goal_hidden.shape[0], -1))


class _SemanticVisualSelfAttentionBlock(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int, ffn_ratio: float, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        inner = max(1, int(round(latent_dim * ffn_ratio)))
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, latent_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(tokens)
        update, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.dropout(update)
        return tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))


class _SemanticVisualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
        )
        inner = max(1, int(round(latent_dim * ffn_ratio)))
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, latent_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        valid_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B, S, D], got {tuple(context.shape)}")
        if valid_mask.ndim != 2 or valid_mask.shape != context.shape[:2]:
            raise ValueError(
                f"`valid_mask` must be [B, S]={tuple(context.shape[:2])}, got {tuple(valid_mask.shape)}"
            )
        key_padding_mask = ~valid_mask.to(dtype=torch.bool)
        # MultiheadAttention forbids fully-masked keys; keep one dummy position.
        fully_masked = key_padding_mask.all(dim=1)
        if bool(fully_masked.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_masked, 0] = False
        context_normed = self.context_norm(context)
        if attn_mask is not None:
            # [L, S] bool, True = blocked. Static grid geometry, so no batch dim.
            if attn_mask.shape != (queries.shape[1], context.shape[1]):
                raise ValueError(
                    "`attn_mask` must be [num_queries, context_len]="
                    f"{(queries.shape[1], context.shape[1])}, got {tuple(attn_mask.shape)}"
                )
        update, _ = self.cross_attn(
            self.query_norm(queries),
            context_normed,
            context_normed,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        queries = queries + self.dropout(update)
        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class SemanticVisualAggregatorGroup(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        kv_dim: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
        attn_head_dim: int,
        zero_init_value: bool = GOAL_PRIOR_ZERO_INIT_VALUE,
        gate_bias_init: float = GOAL_PRIOR_SYN_GATE_BIAS_INIT,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.kv_dim = int(kv_dim)
        self.attn_head_dim = int(attn_head_dim)
        self.num_kv_heads = self.kv_dim // self.attn_head_dim
        if self.num_kv_heads * self.attn_head_dim != self.kv_dim:
            raise ValueError(f"`kv_dim` ({kv_dim}) must be divisible by attn_head_dim ({attn_head_dim}).")
        self.self_block = _SemanticVisualSelfAttentionBlock(latent_dim, num_heads, ffn_ratio, dropout)
        self.semantic_block = _SemanticVisualCrossAttentionBlock(
            latent_dim, context_dim, num_heads, ffn_ratio, dropout
        )
        self.visual_block = _SemanticVisualCrossAttentionBlock(
            latent_dim, context_dim, num_heads, ffn_ratio, dropout
        )
        self.to_key = nn.Linear(latent_dim, kv_dim, bias=False)
        self.to_value = nn.Linear(latent_dim, kv_dim, bias=False)
        self.key_norm = _RMSNorm(self.attn_head_dim)
        # B4. Zeroing `to_value` alone is not enough: the gated columns would
        # still take softmax mass away from text and action. The learnable logit
        # bias is what makes them a true no-op at initialisation.
        #
        # `to_value` is shared across all synthetic tokens, so zeroing it would
        # also silence the pose columns we deliberately keep open. The gate is
        # therefore carried entirely by the logit bias, which is applied per
        # column range by the caller.
        if bool(zero_init_value):
            nn.init.zeros_(self.to_value.weight)
        self.syn_gate_bias = nn.Parameter(torch.tensor(float(gate_bias_init)))

    def forward(
        self,
        queries: torch.Tensor,
        semantic_hidden: torch.Tensor,
        visual_hidden: torch.Tensor,
        *,
        semantic_mask: torch.Tensor,
        image_mask: torch.Tensor,
        visual_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        queries = self.self_block(queries)
        queries = self.semantic_block(queries, semantic_hidden, semantic_mask)
        return self.visual_block(queries, visual_hidden, image_mask, visual_attn_mask)

    def project_kv(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.to_key(tokens)
        value = self.to_value(tokens)
        batch, seq_len, _ = key.shape
        key = key.view(batch, seq_len, self.num_kv_heads, self.attn_head_dim)
        key = self.key_norm(key)
        key = key.reshape(batch, seq_len, self.kv_dim)
        return key, value


class SemanticVisualAggregator(nn.Module):
    """Recurrent 100-token V3 aggregator with depth-grouped parameters."""

    def __init__(
        self,
        num_tokens: int = GOAL_PRIOR_NUM_LATENTS,
        latent_dim: int = GOAL_PRIOR_LATENT_DIM,
        context_dim: int = 3072,
        kv_dim: int = 3072,
        attn_head_dim: int = 128,
        num_layer_groups: int = GOAL_PRIOR_NUM_GROUPS,
        num_heads: int = GOAL_PRIOR_NUM_HEADS,
        ffn_ratio: float = GOAL_PRIOR_FFN_RATIO,
        dropout: float = GOAL_PRIOR_DROPOUT,
        num_pose_tokens: int = GOAL_PRIOR_NUM_POSE_TOKENS,
        context_token_dropout: float = GOAL_PRIOR_CONTEXT_TOKEN_DROPOUT,
        context_blackout_prob: float = GOAL_PRIOR_CONTEXT_BLACKOUT_PROB,
        zero_init_value: bool = GOAL_PRIOR_ZERO_INIT_VALUE,
        gate_bias_init: float = GOAL_PRIOR_SYN_GATE_BIAS_INIT,
        latent_layout: str = GOAL_PRIOR_LATENT_LAYOUT,
        grid_hw: Sequence[int] = GOAL_PRIOR_GRID_HW,
        grid_window: int = GOAL_PRIOR_GRID_WINDOW,
        gate_pose_tokens: bool = GOAL_PRIOR_GATE_POSE_TOKENS,
        action_sees_ref: bool = GOAL_PRIOR_ACTION_SEES_REF,
        p_both: float = GOAL_PRIOR_P_BOTH,
        p_ref_only: float = GOAL_PRIOR_P_REF_ONLY,
        p_syn_only: float = GOAL_PRIOR_P_SYN_ONLY,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.num_pose_tokens = int(num_pose_tokens)
        if not 0 <= self.num_pose_tokens <= self.num_tokens:
            raise ValueError(
                f"`num_pose_tokens` ({num_pose_tokens}) must be in [0, num_tokens={self.num_tokens}]"
            )
        self.context_token_dropout = float(context_token_dropout)
        if not 0.0 <= self.context_token_dropout < 1.0:
            raise ValueError(
                f"`context_token_dropout` must be in [0, 1), got {context_token_dropout}"
            )
        self.context_blackout_prob = float(context_blackout_prob)
        if not 0.0 <= self.context_blackout_prob < 1.0:
            raise ValueError(
                f"`context_blackout_prob` must be in [0, 1), got {context_blackout_prob}"
            )
        self.action_sees_ref = bool(action_sees_ref)
        self.p_both = float(p_both)
        self.p_ref_only = float(p_ref_only)
        self.p_syn_only = float(p_syn_only)
        if self.action_sees_ref:
            total = self.p_both + self.p_ref_only + self.p_syn_only
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    "channel regime probabilities must sum to 1, got "
                    f"{self.p_both}+{self.p_ref_only}+{self.p_syn_only}={total}"
                )
            if min(self.p_both, self.p_ref_only, self.p_syn_only) < 0.0:
                raise ValueError("channel regime probabilities must be non-negative")
        self.gate_pose_tokens = bool(gate_pose_tokens)
        if bool(zero_init_value) and not self.gate_pose_tokens:
            # `to_value` is one Linear shared by every synthetic token, so there
            # is no way to zero it for the context columns only -- it would
            # silence the pose columns we are deliberately keeping open and the
            # split gate would be decorative. Carry the gate with the logit bias.
            raise ValueError(
                "`zero_init_value=True` silences every synthetic column, which "
                "contradicts `gate_pose_tokens=False`. Either gate the pose "
                "tokens too, or rely on `gate_bias_init` alone."
            )
        self.latent_layout = str(latent_layout)
        if self.latent_layout not in ("free", "grid"):
            raise ValueError(f"`latent_layout` must be 'free' or 'grid', got {latent_layout!r}")
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.grid_window = int(grid_window)
        if self.latent_layout == "grid":
            n_cells = self.grid_hw[0] * self.grid_hw[1]
            if self.num_pose_tokens + n_cells > self.num_tokens:
                raise ValueError(
                    f"grid layout needs num_pose_tokens ({self.num_pose_tokens}) + "
                    f"H*W ({n_cells}) <= num_tokens ({self.num_tokens})"
                )
        self._grid_mask_cache: dict[tuple, torch.Tensor] = {}
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.kv_dim = int(kv_dim)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layer_groups = int(num_layer_groups)
        if self.num_layer_groups < 1:
            raise ValueError(f"`num_layer_groups` must be >= 1, got {self.num_layer_groups}")
        queries = torch.empty(self.num_tokens, self.latent_dim)
        nn.init.trunc_normal_(queries, std=0.02)
        self.queries = nn.Parameter(queries)
        self.groups = nn.ModuleList(
            [
                SemanticVisualAggregatorGroup(
                    latent_dim=self.latent_dim,
                    context_dim=self.context_dim,
                    kv_dim=self.kv_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    attn_head_dim=self.attn_head_dim,
                    zero_init_value=zero_init_value,
                    gate_bias_init=gate_bias_init,
                )
                for _ in range(self.num_layer_groups)
            ]
        )

    def initial_queries(self, batch_size: int) -> torch.Tensor:
        return self.queries.unsqueeze(0).expand(int(batch_size), -1, -1).contiguous()

    def gate_bias(self, layer_idx: int, num_layers: int) -> torch.Tensor:
        """B4. Additive logit bias for this layer's *gated* synthetic columns."""
        return self.groups[self.layer_group_index(layer_idx, num_layers)].syn_gate_bias

    def gated_span(self) -> tuple[int, int]:
        """Half-open [start, end) of synthetic columns the gate covers.

        Pose columns stay open so that step 0 reproduces Stage 1's topology
        [txt | pose | action]; only the context block is cold-started. This is
        also exactly the state B2+ blackout falls back to, so the two
        mechanisms agree on what "no steering context" means.
        """
        if not self.gate_pose_tokens:
            return int(self.num_pose_tokens), int(self.num_tokens)
        return 0, int(self.num_tokens)

    def sample_context_keep_mask(
        self,
        batch_size: int,
        device: torch.device,
        training: Optional[bool] = None,
    ) -> Optional[torch.Tensor]:
        """B2. [B, num_tokens] bool, True = keep. Pose tokens are never dropped.

        Defaults to the module's own train/eval mode: defaulting to True would
        silently apply dropout at inference for any caller that forgot the flag.
        Returns None when the feature is off, so the caller can skip the AND.
        """
        if training is None:
            training = self.training
        if not training:
            return None
        if self.context_token_dropout <= 0.0 and self.context_blackout_prob <= 0.0:
            return None
        keep = torch.ones(int(batch_size), self.num_tokens, dtype=torch.bool, device=device)
        n_ctx = self.num_tokens - self.num_pose_tokens
        if n_ctx <= 0:
            return keep
        if self.context_token_dropout > 0.0:
            drop = torch.rand(int(batch_size), n_ctx, device=device) < self.context_token_dropout
            keep[:, self.num_pose_tokens :] = ~drop
        if self.context_blackout_prob > 0.0:
            # Fixed count rather than an independent coin per sample: a batch
            # then always carries both the steered and the fallback regime, so
            # the per-regime training metrics are always defined on every rank.
            # Ranks that disagree about which metric keys exist deadlock the
            # trainer's per-key all-gather.
            n = int(batch_size)
            k = int(round(self.context_blackout_prob * n))
            if k <= 0 and self.context_blackout_prob > 0.0:
                k = 1 if n > 1 else 0
            k = min(k, max(n - 1, 0))  # never black out the whole batch
            if k > 0:
                idx = torch.randperm(n, device=device)[:k]
                keep[idx, self.num_pose_tokens :] = False
        return keep

    def sample_channel_regime(
        self,
        batch_size: int,
        device: torch.device,
        training: Optional[bool] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Plan B. Per-sample (ref_keep, syn_keep) booleans, or (None, None).

        Fixed counts rather than independent coins: every rank then sees every
        regime in every batch, so the per-regime training metrics are always
        defined. Ranks that disagree about which metric keys exist deadlock the
        trainer's per-key all-gather.
        """
        if training is None:
            training = self.training
        if not self.action_sees_ref or not training:
            return None, None
        n = int(batch_size)
        if n <= 0:
            return None, None
        n_ref_only = int(round(self.p_ref_only * n))
        n_syn_only = int(round(self.p_syn_only * n))
        if n >= 3:  # keep every regime represented
            n_ref_only = max(1, min(n_ref_only, n - 2))
            n_syn_only = max(1, min(n_syn_only, n - 1 - n_ref_only))
        else:
            n_ref_only = min(n_ref_only, n)
            n_syn_only = min(n_syn_only, n - n_ref_only)
        order = torch.randperm(n, device=device)
        ref_keep = torch.ones(n, dtype=torch.bool, device=device)
        syn_keep = torch.ones(n, dtype=torch.bool, device=device)
        ref_keep[order[:n_syn_only]] = False                       # syn-only samples
        syn_keep[order[n_syn_only : n_syn_only + n_ref_only]] = False  # ref-only samples
        return ref_keep, syn_keep

    def visual_attn_mask(
        self,
        context_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """B1. [num_tokens, context_len] bool, True = blocked.

        Pose tokens and any spare latents keep global access; the H*W grid
        latents see only their own neighbourhood of image tokens. Returns None
        under the "free" layout so nothing changes.
        """
        if self.latent_layout != "grid" or int(context_len) <= 0:
            return None
        key = (int(context_len), str(device))
        cached = self._grid_mask_cache.get(key)
        if cached is not None:
            return cached
        gh, gw = self.grid_hw
        side = int(round(float(int(context_len)) ** 0.5))
        if side * side != int(context_len):
            # Non-square image token grid: fall back to global attention rather
            # than guessing a layout that would silently mis-align the windows.
            self._grid_mask_cache[key] = None
            return None
        mask = torch.ones(self.num_tokens, int(context_len), dtype=torch.bool, device=device)
        mask[: self.num_pose_tokens] = False  # pose tokens stay global
        rows = torch.arange(side, device=device).view(side, 1).expand(side, side).reshape(-1)
        cols = torch.arange(side, device=device).view(1, side).expand(side, side).reshape(-1)
        for cell in range(gh * gw):
            tok = self.num_pose_tokens + cell
            if tok >= self.num_tokens:
                break
            cy, cx = divmod(cell, gw)
            r0 = int(cy * side / gh) - self.grid_window
            r1 = int((cy + 1) * side / gh) + self.grid_window
            c0 = int(cx * side / gw) - self.grid_window
            c1 = int((cx + 1) * side / gw) + self.grid_window
            keep = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
            mask[tok] = ~keep
        mask[self.num_pose_tokens + gh * gw :] = False  # spare latents stay global
        self._grid_mask_cache[key] = mask
        return mask

    def layer_group_index(self, layer_idx: int, num_layers: int) -> int:
        return layer_group_index(layer_idx, num_layers, self.num_layer_groups)

    def forward_layer(
        self,
        queries: torch.Tensor,
        semantic_hidden: torch.Tensor,
        visual_hidden: torch.Tensor,
        *,
        semantic_mask: torch.Tensor,
        image_mask: torch.Tensor,
        layer_idx: int,
        num_layers: int,
        visual_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        group = self.groups[self.layer_group_index(layer_idx, num_layers)]
        return group(
            queries,
            semantic_hidden,
            visual_hidden,
            semantic_mask=semantic_mask,
            image_mask=image_mask,
            visual_attn_mask=visual_attn_mask,
        )

    def project_kv(
        self,
        tokens: torch.Tensor,
        *,
        layer_idx: int,
        num_layers: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group = self.groups[self.layer_group_index(layer_idx, num_layers)]
        return group.project_kv(tokens)


def layer_group_index(layer_idx: int, num_layers: int, num_layer_groups: int) -> int:
    if int(num_layers) <= 0:
        raise ValueError(f"`num_layers` must be positive, got {num_layers}")
    if int(num_layers) % int(num_layer_groups) != 0:
        raise ValueError(
            f"`num_layers` ({num_layers}) must be divisible by num_layer_groups ({num_layer_groups})"
        )
    if not 0 <= int(layer_idx) < int(num_layers):
        raise ValueError(f"`layer_idx` must be in [0, {num_layers}), got {layer_idx}")
    return int(layer_idx) // (int(num_layers) // int(num_layer_groups))


def extract_goal_pose_from_proprio(proprio: torch.Tensor, proprio_is_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the last observation as goal before aligning proprio with the action horizon."""
    if proprio.ndim != 2:
        raise ValueError(f"`proprio` must be [num_obs, dim], got {tuple(proprio.shape)}")
    if proprio.shape[0] < 2:
        raise ValueError(f"`proprio` must contain current state and future goal, got {tuple(proprio.shape)}")
    if proprio_is_pad.ndim != 1 or proprio_is_pad.shape[0] != proprio.shape[0]:
        raise ValueError(
            f"`proprio_is_pad` must be [{proprio.shape[0]}], got {tuple(proprio_is_pad.shape)}"
        )
    return proprio[-1].clone(), proprio_is_pad[-1].clone()


def build_synthetic_token_ids(
    batch_size: int,
    seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    time_value: float = GOAL_PRIOR_SYNTHETIC_TIME_VALUE,
) -> torch.Tensor:
    ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
    ids[..., 0] = float(time_value)
    ids[..., 1] = torch.arange(seq_len, device=device, dtype=dtype)[None, :]
    return ids


def build_stage2_action_attention_mask(
    batch_size: int,
    txt_len: int,
    synthetic_len: int,
    action_len: int,
    device: torch.device,
    text_attention_mask: Optional[torch.Tensor] = None,
    synthetic_keep_mask: Optional[torch.Tensor] = None,
    ref_len: int = 0,
    ref_keep_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Action queries attend text/state, the reference image, the synthetic
    tokens, and themselves. Layout is [txt | ref | syn | action]; `ref_len=0`
    reproduces the firewalled layout exactly."""
    ref_len = int(ref_len)
    syn_start = int(txt_len) + ref_len
    key_len = syn_start + int(synthetic_len) + int(action_len)
    mask = torch.zeros(batch_size, action_len, key_len, dtype=torch.bool, device=device)
    mask[:, :, :txt_len] = True
    mask[:, :, txt_len:syn_start] = True
    mask[:, :, syn_start : syn_start + synthetic_len] = True
    mask[:, :, syn_start + synthetic_len :] = True
    if text_attention_mask is not None:
        if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (batch_size, txt_len):
            raise ValueError(
                "`text_attention_mask` must be [B, txt_len], "
                f"got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
            )
        mask[:, :, :txt_len] &= text_attention_mask.to(device=device, dtype=torch.bool)[:, None, :]
    if synthetic_keep_mask is not None:
        if synthetic_keep_mask.ndim != 2 or tuple(synthetic_keep_mask.shape) != (
            batch_size,
            synthetic_len,
        ):
            raise ValueError(
                "`synthetic_keep_mask` must be [B, synthetic_len], "
                f"got {tuple(synthetic_keep_mask.shape)} for B={batch_size}, "
                f"synthetic_len={synthetic_len}"
            )
        keep = synthetic_keep_mask.to(device=device, dtype=torch.bool)[:, None, :]
        mask[:, :, syn_start : syn_start + synthetic_len] &= keep
    if ref_keep_mask is not None:
        if ref_len <= 0:
            raise ValueError("`ref_keep_mask` given but `ref_len` is 0")
        if ref_keep_mask.ndim != 1 or int(ref_keep_mask.shape[0]) != batch_size:
            raise ValueError(
                f"`ref_keep_mask` must be [B]={batch_size}, got {tuple(ref_keep_mask.shape)}"
            )
        mask[:, :, txt_len:syn_start] &= ref_keep_mask.to(
            device=device, dtype=torch.bool
        )[:, None, None]
    return mask


def build_concatenated_hidden(
    semantic: torch.Tensor,
    visual: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([semantic, visual], dim=1)


def build_concatenated_masks(
    semantic_mask: torch.Tensor,
    visual_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic_full = torch.cat(
        [semantic_mask, torch.zeros_like(visual_mask, dtype=torch.bool)],
        dim=1,
    )
    visual_full = torch.cat(
        [torch.zeros_like(semantic_mask, dtype=torch.bool), visual_mask],
        dim=1,
    )
    return semantic_full, visual_full


def compute_pose_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_pad: Optional[torch.Tensor] = None,
    dim_is_pad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pose pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    per_dim = F.mse_loss(pred.float(), target.float(), reduction="none")
    if dim_is_pad is not None:
        dim_valid = (~dim_is_pad).to(device=per_dim.device, dtype=per_dim.dtype)
        if dim_valid.ndim == 1:
            dim_valid = dim_valid.unsqueeze(0).expand(per_dim.shape[0], -1)
        dim_valid_sum = dim_valid.sum(dim=1).clamp(min=1.0)
        per_sample = (per_dim * dim_valid).sum(dim=1) / dim_valid_sum
    else:
        per_sample = per_dim.mean(dim=1)
    if is_pad is None:
        return per_sample.mean()
    valid = (~is_pad.view(-1)).to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)


def validate_pose_only_latent_layout(num_latents: int, num_pose_tokens: int) -> None:
    """Stage 1b keeps only the pose latents.

    The other 92 exist solely to be projected into synthetic K/V for the Action
    Expert. Stage 1b does not run an Action Expert, so they would be computed
    every layer and read by nothing: no gradient reaches them, which is both
    wasted work and the unused-parameter condition that stalls the trainer's
    per-key all-gather.
    """
    if int(num_latents) != int(num_pose_tokens):
        raise ValueError(
            "Stage 1b has no Action Expert, so the non-pose latents have no consumer. "
            f"Set num_latents == num_pose_tokens (got {num_latents} and {num_pose_tokens})."
        )
    if int(num_pose_tokens) < 1:
        raise ValueError(f"`num_pose_tokens` must be >= 1, got {num_pose_tokens}")


def non_default_action_facing_knobs(goal_prior_cfg: Mapping[str, object]) -> list[tuple[str, object]]:
    """Stage 1b knobs that were set away from their default, hence inert.

    Returned rather than logged so the caller owns the message and the check is
    testable without caplog.
    """
    defaults = {
        "context_token_dropout": GOAL_PRIOR_CONTEXT_TOKEN_DROPOUT,
        "context_blackout_prob": GOAL_PRIOR_CONTEXT_BLACKOUT_PROB,
        "action_sees_ref": GOAL_PRIOR_ACTION_SEES_REF,
        "p_both": GOAL_PRIOR_P_BOTH,
        "p_ref_only": GOAL_PRIOR_P_REF_ONLY,
        "p_syn_only": GOAL_PRIOR_P_SYN_ONLY,
        "zero_init_value": GOAL_PRIOR_ZERO_INIT_VALUE,
        "syn_gate_bias_init": GOAL_PRIOR_SYN_GATE_BIAS_INIT,
        "gate_pose_tokens": GOAL_PRIOR_GATE_POSE_TOKENS,
    }
    found: list[tuple[str, object]] = []
    for name in ACTION_FACING_GOAL_PRIOR_KNOBS:
        if name not in goal_prior_cfg:
            continue
        value = goal_prior_cfg[name]
        if value != defaults[name]:
            found.append((name, value))
    return found


def _starts_with_any(key: str, prefixes: Sequence[str]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def validate_goal_prior_checkpoint_keys(
    *,
    current_stage: str,
    payload_stage: Optional[str],
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    bridge_from_stage1: bool = False,
) -> None:
    """Fail-closed key check for Stage1/Stage2 checkpoints."""
    current_stage = str(current_stage)
    missing = list(missing_keys)
    unexpected = list(unexpected_keys)
    if current_stage not in GOAL_PRIOR_STAGES:
        raise ValueError(f"Unsupported goal-prior stage {current_stage!r}")

    if current_stage == "stage1":
        if missing:
            raise RuntimeError(f"Stage1 checkpoint is missing keys: {missing[:20]}")
        if unexpected:
            raise RuntimeError(f"Stage1 checkpoint has unexpected keys: {unexpected[:20]}")
        return

    allowed_missing = STAGE2_ONLY_CHECKPOINT_PREFIXES if bridge_from_stage1 else ()
    allowed_unexpected = STAGE1_ONLY_CHECKPOINT_PREFIXES if bridge_from_stage1 else ()
    bad_missing = [key for key in missing if not _starts_with_any(key, allowed_missing)]
    bad_unexpected = [key for key in unexpected if not _starts_with_any(key, allowed_unexpected)]
    if bad_missing:
        raise RuntimeError(
            "Stage2 checkpoint load has disallowed missing keys "
            f"(payload_stage={payload_stage!r}, bridge={bridge_from_stage1}): {bad_missing[:20]}"
        )
    if bad_unexpected:
        raise RuntimeError(
            "Stage2 checkpoint load has disallowed unexpected keys "
            f"(payload_stage={payload_stage!r}, bridge={bridge_from_stage1}): {bad_unexpected[:20]}"
        )


def stage2_action_mask_excludes_raw_images(
    action_mask: torch.Tensor,
    *,
    txt_len: int,
    synthetic_len: int,
    action_len: int,
) -> bool:
    """Stage2 action K/V layout is [txt | synthetic | action], with no raw image columns."""
    expected = (action_mask.shape[-2], action_mask.shape[-1])
    got = (action_len, txt_len + synthetic_len + action_len)
    if expected != got:
        raise ValueError(f"Stage2 action mask must be [..., {got[0]}, {got[1]}], got {tuple(action_mask.shape)}")
    return True
