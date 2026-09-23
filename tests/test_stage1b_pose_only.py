"""Stage 1b: the Action Expert is out of the graph, out of the loss, and frozen.

Stage 1b reuses Stage 2's forward, so the risk lives in the branches that decide
what *not* to run. The load-bearing claim is that dropping the Action Expert
cannot perturb the video stream -- the Stage 2 video mask is built with
`action_len=0`, so video rows already attend no action column, and the synthetic
K/V the aggregator produces had the Action Expert as its only consumer. That is
checked here by running the same inputs both ways and comparing the video output
bit for bit, not by reading the code.

The MoT forward needs no FLUX install: `apply_rope` is stubbed, exactly as
`tests/test_mixed_attention_gate.py` stands in for `_mixed_attention`.
"""
import logging
import sys
import types

import pytest
import torch
import torch.nn as nn

from imagewam.models.backbones.goal_pose_prior import (
    STAGE1B,
    GoalPoseDecoder,
    SemanticVisualAggregator,
    non_default_action_facing_knobs,
    validate_goal_prior_checkpoint_keys,
    validate_pose_only_latent_layout,
)
from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.models.backbones.mot import MoT

# --- geometry ---------------------------------------------------------------
B = 2
NUM_HEADS = 2
HEAD_DIM = 4
HIDDEN = NUM_HEADS * HEAD_DIM  # the FLUX residual width, shared by txt and img
TXT_LEN = 3
COND_LEN = 2  # reference-image tokens
TARGET_LEN = 3
IMG_LEN = COND_LEN + TARGET_LEN
VIDEO_LEN = TXT_LEN + IMG_LEN
ACTION_LEN = 2
LATENT_DIM = 16
NUM_TOKENS = 2
NUM_POSE_TOKENS = 2


class _Bomb:
    """Raises on any attribute access, so 'never touched' is an assertion."""

    def __getattr__(self, name):
        raise AssertionError(f"Stage 1b reached into the Action Expert ({name!r})")


def _nope(*args, **kwargs):
    raise AssertionError("Stage 1b projected synthetic K/V with no consumer for it")


class _StubDoubleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def _prepare_qkv(self, img_tokens, txt_tokens, img_pe, txt_pe, tmod_img, tmod_txt):
        seq = txt_tokens.shape[1] + img_tokens.shape[1]
        shape = (txt_tokens.shape[0], NUM_HEADS, seq, HEAD_DIM)
        return (
            torch.randn(shape),
            torch.randn(shape),
            torch.randn(shape),
            None,
            txt_tokens.shape[1],
            None,
        )

    def _apply_residuals(self, img_tokens, txt_tokens, img_attn, txt_attn, mods):
        return img_tokens, txt_tokens


class _StubSingleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def _qkv(self, x, mod):
        shape = (x.shape[0], NUM_HEADS, x.shape[1], HEAD_DIM)
        return torch.randn(shape), torch.randn(shape), torch.randn(shape), None, None

    def _out(self, residual_x, attn, mlp, gate):
        return residual_x


class _StubActionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def prepare_qkv(self, action_tokens, action_pe, tmod):
        return {
            "q": torch.randn(action_tokens.shape[0], action_tokens.shape[1], HIDDEN),
            "k": torch.randn(action_tokens.shape[0], action_tokens.shape[1], HIDDEN),
            "v": torch.randn(action_tokens.shape[0], action_tokens.shape[1], HIDDEN),
        }

    def apply_post(self, mixed, state):
        return mixed


class _StubVideoExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_layers = 1
        self.single_layers = 1
        self.double_blocks = nn.ModuleList([_StubDoubleBlock()])
        self.single_blocks = nn.ModuleList([_StubSingleBlock()])
        self.transformer = nn.Module()
        self.transformer.pe_embedder = lambda ids: torch.zeros(
            ids.shape[0], ids.shape[1], NUM_HEADS, HEAD_DIM
        )


class _StubActionExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([_StubActionBlock()])
        self.single_blocks = nn.ModuleList([_StubActionBlock()])


def _aggregator():
    return SemanticVisualAggregator(
        num_tokens=NUM_TOKENS,
        latent_dim=LATENT_DIM,
        context_dim=HIDDEN,
        kv_dim=HIDDEN,
        attn_head_dim=HEAD_DIM,
        num_layer_groups=1,
        num_heads=2,
        num_pose_tokens=NUM_POSE_TOKENS,
    )


class _PoseMoT(MoT):
    def __init__(self, action_expert):
        nn.Module.__init__(self)
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_HEADS
        self.attn_head_dim = HEAD_DIM
        self.gqa_implementation = "repeat"
        self.force_flash_attention = False
        # Off, so `_flux2_maybe_checkpoint` runs the layer directly instead of
        # routing a None through torch.utils.checkpoint.
        self.mot_checkpoint_mixed_attn = False
        self.mixtures = {"video": _StubVideoExpert(), "action": action_expert}


@pytest.fixture
def stub_flux2(monkeypatch):
    """`apply_rope` is identity: this is a plumbing test, not a RoPE test."""
    pkg = types.ModuleType("flux2")
    model = types.ModuleType("flux2.model")
    model.apply_rope = lambda q, k, pe: (q, k)
    monkeypatch.setitem(sys.modules, "flux2", pkg)
    monkeypatch.setitem(sys.modules, "flux2.model", model)


def _forward(mot, aggregator, mode, action_expert_present):
    torch.manual_seed(0)
    video_tokens = {
        "txt": torch.randn(B, TXT_LEN, HIDDEN),
        "img": torch.randn(B, IMG_LEN, HIDDEN),
    }
    freqs = {
        "txt": torch.zeros(B, NUM_HEADS, TXT_LEN, HEAD_DIM),
        "img": torch.zeros(B, NUM_HEADS, IMG_LEN, HEAD_DIM),
    }
    embeds = {"video": video_tokens}
    context = {
        "video": None,
        "goal_prior": {
            "mode": mode,
            "aggregator": aggregator,
            "text_mask": torch.ones(B, TXT_LEN, dtype=torch.bool),
            "cond_len": COND_LEN,
            "target_len": TARGET_LEN,
        },
    }
    t_mod = {"video": {"double_img": None, "double_txt": None, "single": None}}
    if action_expert_present:
        embeds["action"] = torch.randn(B, ACTION_LEN, HIDDEN)
        context["action"] = {"ids": torch.zeros(B, ACTION_LEN, 3)}
        t_mod["action"] = {"double_img": None, "single": None}

    mask = torch.ones(B, VIDEO_LEN, VIDEO_LEN, dtype=torch.bool)
    attention_mask = {"double_joint": mask, "single": mask.clone()}
    if action_expert_present:
        key_len = TXT_LEN + NUM_TOKENS + ACTION_LEN
        attention_mask["action"] = torch.ones(B, ACTION_LEN, key_len, dtype=torch.bool)

    return MoT._forward_flux2_stage2(
        mot,
        embeds_all=embeds,
        attention_mask=attention_mask,
        freqs_all={"video": freqs},
        context_all=context,
        t_mod_all=t_mod,
    )


# ------------------------------------------------------------ the MoT forward
def test_pose_only_never_touches_the_action_expert(stub_flux2):
    """The whole point of the stage: the Action Expert is out of the graph."""
    mot = _PoseMoT(action_expert=_Bomb())
    out = _forward(mot, _aggregator(), STAGE1B, action_expert_present=False)
    assert set(out) == {"video", "goal_latents"}


def test_pose_only_projects_no_synthetic_kv(stub_flux2, monkeypatch):
    """Synthetic K/V had the Action Expert as its only consumer."""
    monkeypatch.setattr(MoT, "_flux2_project_synthetic_kv", _nope)
    mot = _PoseMoT(action_expert=_Bomb())
    out = _forward(mot, _aggregator(), STAGE1B, action_expert_present=False)
    assert set(out) == {"video", "goal_latents"}


def test_pose_only_video_stream_matches_the_action_expert_run(stub_flux2):
    """The invariant that makes the ablation readable.

    If dropping the Action Expert changed the video stream, a Stage 1b run would
    not be comparable to Stage 2 at all -- every difference would be confounded
    with a different video forward.
    """
    with_action = _forward(
        _PoseMoT(action_expert=_StubActionExpert()),
        _aggregator(),
        "stage2",
        action_expert_present=True,
    )
    without_action = _forward(
        _PoseMoT(action_expert=_Bomb()),
        _aggregator(),
        STAGE1B,
        action_expert_present=False,
    )
    for key in ("txt", "img"):
        assert torch.equal(with_action["video"][key], without_action["video"][key]), key


def test_pose_only_still_returns_the_latents_the_pose_head_reads(stub_flux2):
    """`loss_pose` reads `latents[:, :num_pose_tokens]`; nothing else supplies it."""
    mot = _PoseMoT(action_expert=_Bomb())
    out = _forward(mot, _aggregator(), STAGE1B, action_expert_present=False)
    latents = out["goal_latents"]
    assert latents.shape == (B, NUM_TOKENS, LATENT_DIM)
    assert latents.requires_grad


# ------------------------------------------------------------- the pure guards
def test_pose_only_latent_layout_rejects_orphan_context_latents():
    """Stage 2's 92 context latents would be trained on a loss that never sees them."""
    with pytest.raises(ValueError) as excinfo:
        validate_pose_only_latent_layout(num_latents=100, num_pose_tokens=8)
    assert "no consumer" in str(excinfo.value)


def test_pose_only_latent_layout_accepts_the_matching_split():
    validate_pose_only_latent_layout(num_latents=8, num_pose_tokens=8)


def test_action_facing_knobs_report_only_non_defaults():
    assert non_default_action_facing_knobs({}) == []
    assert non_default_action_facing_knobs({"num_latents": 8}) == []
    assert non_default_action_facing_knobs({"zero_init_value": False}) == []
    found = dict(
        non_default_action_facing_knobs(
            {"action_sees_ref": True, "context_token_dropout": 0.05, "num_latents": 8}
        )
    )
    assert found == {"action_sees_ref": True, "context_token_dropout": 0.05}


def test_stage1b_checkpoint_keys_are_checked_like_stage2():
    validate_goal_prior_checkpoint_keys(
        current_stage=STAGE1B, payload_stage=STAGE1B, missing_keys=[], unexpected_keys=[]
    )
    with pytest.raises(RuntimeError):
        validate_goal_prior_checkpoint_keys(
            current_stage=STAGE1B, payload_stage=STAGE1B, missing_keys=["mot.x"], unexpected_keys=[]
        )


def test_unknown_stage_is_still_rejected():
    with pytest.raises(ValueError):
        validate_goal_prior_checkpoint_keys(
            current_stage="stage9", payload_stage="stage9", missing_keys=[], unexpected_keys=[]
        )


# ------------------------------------------------------- the trainable whitelist
class _StubExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))


class _PolicyImageWAM(ImageWAM):
    """Just enough of the model for `apply_trainable_policy` to run."""

    def __init__(self, stage):
        nn.Module.__init__(self)
        self.stack = "flux2"
        self.goal_prior_stage = stage
        mot = nn.Module()
        mot.mixtures = nn.ModuleDict({"video": _StubExpert(), "action": _StubExpert()})
        self.mot = mot
        self.dit = mot
        self.semantic_visual_aggregator = _aggregator()
        self.semantic_visual_pose_norm = nn.LayerNorm(LATENT_DIM)
        self.semantic_visual_pose_decoder = GoalPoseDecoder(
            num_tokens=NUM_POSE_TOKENS, hidden_size=LATENT_DIM, pose_dim=8
        )
        self.goal_pose_encoder = None
        self.proprio_encoder = None


def _grad_state(module):
    return {name: param.requires_grad for name, param in module.named_parameters()}


def test_stage1b_freezes_the_action_expert_and_the_orphan_kv_head():
    """Both would otherwise be trained on a loss that never reaches them.

    A parameter with `requires_grad=True` that produces no gradient is what makes
    DDP report an unused parameter and stall the per-key all-gather, so this is a
    correctness test, not a tidiness one.
    """
    model = _PolicyImageWAM(STAGE1B)
    model.apply_trainable_policy()

    assert not any(_grad_state(model.mot.mixtures["action"]).values())
    assert not model.mot.mixtures["action"].training

    for group in model.semantic_visual_aggregator.groups:
        for name in ("to_key", "to_value", "key_norm"):
            assert not any(_grad_state(getattr(group, name)).values()), name
        assert not group.syn_gate_bias.requires_grad

    # The latents themselves are the whole point of the stage, so they must move.
    # A bare Parameter rather than a module, so it is checked directly.
    assert model.semantic_visual_aggregator.queries.requires_grad
    assert all(_grad_state(group.self_block).values())
    assert all(_grad_state(group.semantic_block).values())
    assert all(_grad_state(group.visual_block).values())
    assert all(_grad_state(model.semantic_visual_pose_norm).values())
    assert all(_grad_state(model.semantic_visual_pose_decoder).values())

    # No LoRA configured here, so the video expert fine-tunes in full.
    assert all(_grad_state(model.mot.mixtures["video"]).values())


def test_stage1b_trainable_params_exclude_the_frozen_ones():
    model = _PolicyImageWAM(STAGE1B)
    model.apply_trainable_policy()

    trainable = {id(p) for p in model.collect_trainable_parameters()}
    assert id(model.mot.mixtures["action"].weight) not in trainable
    for group in model.semantic_visual_aggregator.groups:
        assert id(group.to_key.weight) not in trainable
        assert id(group.to_value.weight) not in trainable
        assert id(group.syn_gate_bias) not in trainable
    assert id(group.self_block.ffn[0].weight) in trainable


def test_stage2_still_trains_the_action_expert_and_the_kv_head():
    """The evaluated revision must be untouched by Stage 1b's exclusions."""
    model = _PolicyImageWAM("stage2")
    model.apply_trainable_policy()

    assert all(_grad_state(model.mot.mixtures["action"]).values())
    for group in model.semantic_visual_aggregator.groups:
        assert group.to_key.weight.requires_grad
        assert group.syn_gate_bias.requires_grad


# --------------------------------------------------------------- config surface
class _ConfigImageWAM(ImageWAM):
    """Just enough of the model for `_configure_goal_prior` to run."""

    def __init__(self):
        nn.Module.__init__(self)
        self.stack = "flux2"
        self.proprio_dim = 8
        self.torch_dtype = torch.float32
        video_expert = nn.Module()
        video_expert.hidden_dim = HIDDEN
        video_expert.num_heads = NUM_HEADS
        self.video_expert = video_expert
        self.loss_lambda_pose = 0.3


STAGE1B_GOAL_PRIOR = {
    "num_latents": 8,
    "num_pose_tokens": 8,
    "num_layer_groups": 5,
    "latent_dim": 768,
    "inner_dim": 512,
    "lambda_pose": 0.3,
}


def test_stage1b_config_builds_the_pose_only_aggregator():
    model = _ConfigImageWAM()
    model._configure_goal_prior(STAGE1B, dict(STAGE1B_GOAL_PRIOR))
    assert model.semantic_visual_aggregator.num_tokens == 8
    assert model.semantic_visual_aggregator.num_pose_tokens == 8
    assert model.goal_prior_num_pose_tokens == 8
    # Stage 1's encoder is dropped by the bridge, so Stage 1b never builds one.
    assert model.goal_pose_encoder is None
    assert model.semantic_visual_pose_decoder is not None


def test_stage1b_config_rejects_stage2s_latent_count():
    """A copy-paste of the Stage 2 config must fail loudly, not train 92 dead latents."""
    model = _ConfigImageWAM()
    with pytest.raises(ValueError) as excinfo:
        model._configure_goal_prior(STAGE1B, {**STAGE1B_GOAL_PRIOR, "num_latents": 100})
    assert "no consumer" in str(excinfo.value)


def test_stage1b_config_warns_about_inert_action_knobs(caplog):
    """Silent inertness is the failure this stage is most exposed to."""
    model = _ConfigImageWAM()
    with caplog.at_level(logging.WARNING):
        model._configure_goal_prior(
            STAGE1B,
            {
                **STAGE1B_GOAL_PRIOR,
                "action_sees_ref": True,
                "p_both": 0.55,
                "p_ref_only": 0.15,
                "p_syn_only": 0.30,
            },
        )
    assert "no effect on training" in caplog.text
    assert "action_sees_ref" in caplog.text
    # p_both is already at its default, so it must not be reported as a problem.
    assert "p_both" not in caplog.text


def test_stage2_still_accepts_its_own_latent_count():
    """The evaluated revision keeps its 100 latents and takes no new guard."""
    model = _ConfigImageWAM()
    model._configure_goal_prior("stage2", {"num_latents": 100, "num_pose_tokens": 8})
    assert model.semantic_visual_aggregator.num_tokens == 100


# ------------------------------------------------------------------- dispatch
def test_training_loss_dispatches_stage1b(monkeypatch):
    sentinel = (torch.tensor(0.0), {"loss_pose": 0.0})
    monkeypatch.setattr(
        ImageWAM,
        "_training_loss_flux2_stage1b",
        lambda self, sample, tiled=False: sentinel,
    )
    model = _PolicyImageWAM(STAGE1B)
    assert ImageWAM._training_loss_flux2(model, sample=None) is sentinel
