#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from imagewam.models.backbones.lora import (
    apply_lora_to_linear_suffixes,
    merge_lora_state_dict_to_plain,
    remap_plain_linear_keys_to_lora_base,
)
from imagewam.models.backbones.goal_pose_prior import (
    GOAL_PRIOR_LATENT_DIM,
    GOAL_PRIOR_NUM_GOAL_TOKENS,
    GOAL_PRIOR_NUM_GROUPS,
    GOAL_PRIOR_NUM_LATENTS,
    GOAL_PRIOR_NUM_POSE_TOKENS,
    GoalPoseDecoder,
    GoalPoseEncoder,
    SemanticVisualAggregator,
    build_stage2_action_attention_mask,
    compute_pose_reconstruction_loss,
    extract_goal_pose_from_proprio,
    layer_group_index,
    stage2_action_mask_excludes_raw_images,
    validate_goal_prior_checkpoint_keys,
)
from imagewam.models.backbones.imagewam import ImageWAM


class DummyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(4, 4)
        self.flux2_lora_enabled = False


class DummyMoT(nn.Module):
    def __init__(self, video: nn.Module, action: nn.Module):
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": video, "action": action})


class GoalPosePriorTests(unittest.TestCase):
    def test_goal_encoder_production_shape(self):
        encoder = GoalPoseEncoder(pose_dim=8, num_tokens=GOAL_PRIOR_NUM_GOAL_TOKENS, hidden_size=3072)
        tokens = encoder(torch.randn(2, 8))
        self.assertEqual(tuple(tokens.shape), (2, 8, 3072))

    def test_aggregator_production_query_shape(self):
        aggregator = SemanticVisualAggregator(
            num_tokens=GOAL_PRIOR_NUM_LATENTS,
            latent_dim=GOAL_PRIOR_LATENT_DIM,
            context_dim=64,
            kv_dim=64,
            attn_head_dim=8,
            num_layer_groups=GOAL_PRIOR_NUM_GROUPS,
            num_heads=4,
        )
        queries = aggregator.initial_queries(3)
        self.assertEqual(tuple(queries.shape), (3, 100, 768))
        semantic = torch.randn(3, 5, 64)
        visual = torch.randn(3, 7, 64)
        semantic_mask = torch.ones(3, 5, dtype=torch.bool)
        image_mask = torch.ones(3, 7, dtype=torch.bool)
        image_mask[:, -2:] = False
        out = aggregator.forward_layer(
            queries,
            semantic,
            visual,
            semantic_mask=semantic_mask,
            image_mask=image_mask,
            layer_idx=7,
            num_layers=25,
        )
        self.assertEqual(tuple(out.shape), (3, 100, 768))
        key, value = aggregator.project_kv(out, layer_idx=7, num_layers=25)
        self.assertEqual(tuple(key.shape), (3, 100, 64))
        self.assertEqual(tuple(value.shape), (3, 100, 64))

    def test_pose_decoder_production_shape(self):
        decoder = GoalPoseDecoder(
            num_tokens=GOAL_PRIOR_NUM_POSE_TOKENS,
            hidden_size=GOAL_PRIOR_LATENT_DIM,
            pose_dim=8,
        )
        pose = decoder(torch.randn(2, 8, 768))
        self.assertEqual(tuple(pose.shape), (2, 8))

    def test_layer_group_index_covers_25_layers(self):
        groups = [layer_group_index(idx, 25, 5) for idx in range(25)]
        self.assertEqual(groups[:5], [0, 0, 0, 0, 0])
        self.assertEqual(groups[5:10], [1, 1, 1, 1, 1])
        self.assertEqual(groups[10:15], [2, 2, 2, 2, 2])
        self.assertEqual(groups[15:20], [3, 3, 3, 3, 3])
        self.assertEqual(groups[20:], [4, 4, 4, 4, 4])
        self.assertEqual(set(groups), {0, 1, 2, 3, 4})

    def test_pose_loss_respects_pad_masks(self):
        pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.zeros_like(pred)
        dim_is_pad = torch.tensor([[False, False, True], [False, False, True]])
        is_pad = torch.tensor([False, True])
        loss = compute_pose_reconstruction_loss(pred, target, is_pad=is_pad, dim_is_pad=dim_is_pad)
        expected = ((1.0**2) + (2.0**2)) / 2.0
        self.assertTrue(torch.allclose(loss, torch.tensor(expected)))

    def test_extract_goal_pose_from_proprio(self):
        proprio = torch.arange(17 * 8, dtype=torch.float32).view(17, 8)
        proprio_is_pad = torch.zeros(17, dtype=torch.bool)
        proprio_is_pad[-1] = True
        goal, goal_is_pad = extract_goal_pose_from_proprio(proprio, proprio_is_pad)
        self.assertTrue(torch.equal(goal, proprio[-1]))
        self.assertTrue(bool(goal_is_pad.item()))

    def test_stage2_action_mask_layout_has_no_raw_image_columns(self):
        txt_len, synthetic_len, action_len = 12, 100, 16
        mask = build_stage2_action_attention_mask(
            batch_size=2,
            txt_len=txt_len,
            synthetic_len=synthetic_len,
            action_len=action_len,
            device=torch.device("cpu"),
            text_attention_mask=torch.tensor(
                [[True] * 10 + [False, False], [True] * 12],
                dtype=torch.bool,
            ),
        )
        self.assertEqual(tuple(mask.shape), (2, action_len, txt_len + synthetic_len + action_len))
        self.assertTrue(
            stage2_action_mask_excludes_raw_images(
                mask,
                txt_len=txt_len,
                synthetic_len=synthetic_len,
                action_len=action_len,
            )
        )
        self.assertFalse(bool(mask[0, 0, 10].item()))
        self.assertFalse(bool(mask[0, 0, 11].item()))
        self.assertTrue(bool(mask[1, 0, 11].item()))
        self.assertTrue(torch.all(mask[:, :, txt_len : txt_len + synthetic_len]).item())
        self.assertTrue(torch.all(mask[:, :, txt_len + synthetic_len :]).item())

    def test_checkpoint_bridge_allows_stage1_only_and_stage2_only_keys(self):
        validate_goal_prior_checkpoint_keys(
            current_stage="stage2",
            payload_stage="stage1",
            missing_keys=[
                "semantic_visual_aggregator.queries",
                "semantic_visual_pose_norm.weight",
                "semantic_visual_pose_decoder.net.0.weight",
            ],
            unexpected_keys=["goal_pose_encoder.net.0.weight"],
            bridge_from_stage1=True,
        )
        with self.assertRaises(RuntimeError):
            validate_goal_prior_checkpoint_keys(
                current_stage="stage2",
                payload_stage="stage1",
                missing_keys=["mot.mixtures.action.double_blocks.0.qkv.weight"],
                unexpected_keys=[],
                bridge_from_stage1=True,
            )
        with self.assertRaises(RuntimeError):
            validate_goal_prior_checkpoint_keys(
                current_stage="stage2",
                payload_stage="stage2",
                missing_keys=["semantic_visual_aggregator.queries"],
                unexpected_keys=[],
                bridge_from_stage1=False,
            )

    def test_stage1_trainable_policy_freezes_video_and_keeps_goal_encoder(self):
        video = DummyExpert()
        action = DummyExpert()
        model = ImageWAM.__new__(ImageWAM)
        nn.Module.__init__(model)
        model.stack = "flux2"
        model.mot = DummyMoT(video, action)
        model.dit = model.mot
        model.goal_prior_stage = "stage1"
        model.goal_pose_encoder = GoalPoseEncoder(pose_dim=8, num_tokens=8, hidden_size=16, inner_dim=8)
        model.semantic_visual_aggregator = None
        model.semantic_visual_pose_norm = None
        model.semantic_visual_pose_decoder = None
        model.proprio_encoder = nn.Linear(8, 16)

        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        ImageWAM.apply_trainable_policy(model)
        model.proprio_encoder.train()
        model.proprio_encoder.requires_grad_(True)

        self.assertFalse(any(param.requires_grad for param in video.parameters()))
        self.assertTrue(all(param.requires_grad for param in action.parameters()))
        self.assertTrue(all(param.requires_grad for param in model.goal_pose_encoder.parameters()))
        self.assertTrue(all(param.requires_grad for param in model.proprio_encoder.parameters()))
        trainable = ImageWAM.collect_trainable_parameters(model)
        trainable_ids = {id(param) for param in trainable}
        self.assertTrue(all(id(param) in trainable_ids for param in action.parameters()))
        self.assertTrue(all(id(param) in trainable_ids for param in model.goal_pose_encoder.parameters()))
        self.assertTrue(all(id(param) in trainable_ids for param in model.proprio_encoder.parameters()))
        self.assertFalse(any(id(param) in trainable_ids for param in video.parameters()))

    @staticmethod
    def _loadable_model(video: nn.Module, action: nn.Module, stage) -> ImageWAM:
        """A minimal ImageWAM wired for `load_checkpoint`, MoT registered as `mot`.

        The registration matters: only then do the model's own keys carry the
        `mot.` prefix that a real checkpoint payload uses.
        """
        model = ImageWAM.__new__(ImageWAM)
        nn.Module.__init__(model)
        model.stack = "flux2"
        model.torch_dtype = torch.float32
        model.add_module("mot", DummyMoT(video, action))
        for attr in (
            "proprio_encoder", "goal_pose_encoder", "stage1_null_image_tokens",
            "semantic_visual_aggregator", "semantic_visual_pose_norm",
            "semantic_visual_pose_decoder",
        ):
            setattr(model, attr, None)
        model.goal_prior_stage = stage
        return model

    @staticmethod
    def _build_goal_prior_model(stage: str, *, video: nn.Module, action: nn.Module) -> ImageWAM:
        """A goal-prior ImageWAM with only the attributes `apply_trainable_policy` reads."""
        model = ImageWAM.__new__(ImageWAM)
        nn.Module.__init__(model)
        model.stack = "flux2"
        model.mot = DummyMoT(video, action)
        model.dit = model.mot
        model.goal_prior_stage = stage
        model.goal_pose_encoder = None
        model.semantic_visual_aggregator = SemanticVisualAggregator(
            num_tokens=4, num_pose_tokens=2, latent_dim=8, context_dim=16, kv_dim=16,
            attn_head_dim=8, num_layer_groups=1, num_heads=2,
        )
        model.semantic_visual_pose_norm = None
        model.semantic_visual_pose_decoder = None
        model.proprio_encoder = None
        return model

    def test_stage2_lora_policy_freezes_base_and_keeps_adapters_trainable(self):
        from imagewam.models.backbones.lora import LoRALinear

        video = DummyExpert()
        video.transformer = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        apply_lora_to_linear_suffixes(video.transformer, target_suffixes=("0",), rank=2, alpha=2.0)
        video.flux2_lora_enabled = True
        video.flux2_lora_target_suffixes = ("0",)
        action = DummyExpert()
        model = self._build_goal_prior_model("stage2", video=video, action=action)
        aggregator = model.semantic_visual_aggregator

        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        ImageWAM.apply_trainable_policy(model)

        wrapped = video.transformer[0]
        self.assertIsInstance(wrapped, LoRALinear)
        # The base weights stop training; the adapters replace them.
        self.assertFalse(wrapped.base.weight.requires_grad)
        self.assertTrue(wrapped.lora_A.requires_grad)
        self.assertTrue(wrapped.lora_B.requires_grad)
        # The untouched Linear keeps its original weight and stays frozen.
        self.assertFalse(video.transformer[1].weight.requires_grad)
        # Stage 2's own modules are unaffected by the LoRA branch.
        self.assertTrue(all(param.requires_grad for param in action.parameters()))
        self.assertTrue(all(param.requires_grad for param in aggregator.parameters()))

        trainable_ids = {id(param) for param in ImageWAM.collect_trainable_parameters(model)}
        self.assertIn(id(wrapped.lora_A), trainable_ids)
        self.assertIn(id(wrapped.lora_B), trainable_ids)
        self.assertNotIn(id(wrapped.base.weight), trainable_ids)
        self.assertNotIn(id(video.transformer[1].weight), trainable_ids)

    def test_stage2_without_lora_keeps_the_whole_video_expert_trainable(self):
        video = DummyExpert()
        action = DummyExpert()
        model = self._build_goal_prior_model("stage2", video=video, action=action)

        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        ImageWAM.apply_trainable_policy(model)

        self.assertTrue(all(param.requires_grad for param in video.parameters()))

    def test_stage1_warns_that_lora_adapters_never_train(self):
        video = DummyExpert()
        video.flux2_lora_enabled = True
        model = self._build_goal_prior_model("stage1", video=video, action=DummyExpert())
        model.goal_pose_encoder = GoalPoseEncoder(pose_dim=8, num_tokens=8, hidden_size=16, inner_dim=8)

        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        with self.assertLogs("imagewam.models.backbones.imagewam", level="WARNING") as logs:
            ImageWAM.apply_trainable_policy(model)

        self.assertTrue(
            any("Stage 1 freezes the whole video expert" in line for line in logs.output)
        )
        self.assertFalse(any(param.requires_grad for param in video.parameters()))

    def test_load_checkpoint_maps_plain_keys_into_lora_wrapped_expert(self):
        """A plain-key payload must survive the LoRA remap and the alias filter.

        `Flux2VideoExpert` binds both `self.transformer` and `self.double_blocks =
        transformer.double_blocks`, so a LoRA-built MoT state dict lists every block
        weight twice -- once through the wrapped `transformer` path and once through
        the bare alias. The remap rewrites the wrapped key and leaves the alias
        behind, which the exact-load check reports as one missing plus one
        unexpected key per block and refuses the Stage 1 bridge.
        """
        from unittest import mock

        class AliasExpert(DummyExpert):
            """Mirrors Flux2VideoExpert's duplicate binding of the block list.

            The real wrapper binds `self.transformer` (the FLUX.2 module) and then
            `self.double_blocks = transformer.double_blocks`, flattening the blocks
            onto the expert as well -- so their weights appear under both paths.
            """

            def __init__(self):
                super().__init__()
                inner = nn.Module()
                inner.blocks = nn.ModuleList([nn.Linear(4, 4)])
                self.transformer = inner
                self.blocks = inner.blocks  # the flattened alias

        video = AliasExpert()
        apply_lora_to_linear_suffixes(video.transformer, target_suffixes=("blocks.0",), rank=2, alpha=2.0)

        # `None` skips the exact-load guard so this test sees the wiring result.
        model = self._loadable_model(video, DummyExpert(), stage=None)

        # A Stage 1 payload holds plain Linear weights -- it was saved without
        # adapters -- and carries both registered paths, as the real one does.
        payload = {
            "mot": {
                "mot.mixtures.video.transformer.blocks.0.weight": torch.full((4, 4), 0.25),
                "mot.mixtures.video.blocks.0.weight": torch.full((4, 4), 0.25),
            },
            "step": 1,
        }
        # Capture everything the loader logs; the alias warning is unconditional at
        # this stage, so string-matching it is the assertion.
        with mock.patch("torch.load", return_value=payload), \
                self.assertLogs("imagewam.models.backbones.imagewam", level="INFO") as logs:
            model.load_checkpoint("unused.pt", optimizer=None, goal_prior_bridge=True)
        output = "\n".join(logs.output)

        self.assertIn(
            "unexpected_keys=0", output,
            "alias twin was not dropped, so it was reported as an unexpected MoT key",
        )
        # `transformer.blocks.0.base.weight` is absent from the missing list, i.e.
        # the plain key was remapped into the wrapper and consumed.
        self.assertNotIn("transformer.blocks.0.base.weight'", output)
        wrapped = video.transformer.blocks[0]
        self.assertTrue(
            torch.equal(wrapped.base.weight, torch.full((4, 4), 0.25)),
            "plain-key weight did not land in the LoRA wrapper's base weight",
        )
        # The adapter starts at zero and is untouched by a plain-key payload.
        self.assertTrue(torch.equal(wrapped.lora_B, torch.zeros(4, 2)))

    def test_lora_bridge_tolerates_missing_adapters_but_not_missing_flux(self):
        """A Stage 2 LoRA run has to load a Stage 1 payload that carries no adapters.

        The adapters cannot exist in that payload, so they are initialised fresh --
        the exact-load guard must let them through on the bridge. A genuinely
        absent FLUX weight must still raise, or the guard stops guarding anything.
        """
        from unittest import mock

        def _lora_model() -> ImageWAM:
            video = DummyExpert()
            video.transformer = nn.Sequential(nn.Linear(4, 4))
            apply_lora_to_linear_suffixes(video.transformer, target_suffixes=("0",), rank=2, alpha=2.0)
            return self._loadable_model(video, DummyExpert(), stage="stage2")

        # Every non-adapter weight is supplied, so the only missing keys are the
        # adapters -- exactly the shape of a real Stage 1 payload loaded by a
        # LoRA-built Stage 2 model.
        payload = {
            "mot": {
                "mot.mixtures.video.transformer.0.weight": torch.full((4, 4), 0.25),
                "mot.mixtures.video.transformer.0.bias": torch.zeros(4),
                "mot.mixtures.video.w.weight": torch.zeros(4, 4),
                "mot.mixtures.video.w.bias": torch.zeros(4),
                "mot.mixtures.action.w.weight": torch.zeros(4, 4),
                "mot.mixtures.action.w.bias": torch.zeros(4),
            },
            "step": 1,
            "goal_prior_stage": "stage1",
        }
        model = _lora_model()
        with mock.patch("torch.load", return_value=payload), \
                self.assertLogs("imagewam.models.backbones.imagewam", level="INFO") as logs:
            model.load_checkpoint("unused.pt", optimizer=None, goal_prior_bridge=True)
        self.assertTrue(
            any("adapter tensors initialised fresh" in line for line in logs.output),
            "bridge load did not report the freshly initialised adapters",
        )

        # Drop a weight the checkpoint genuinely should have carried. The guard
        # must still fire, and still list that weight rather than wave it through
        # along with the adapters.
        model = _lora_model()
        empty_payload = {"mot": {}, "step": 1, "goal_prior_stage": "stage1"}
        with mock.patch("torch.load", return_value=empty_payload):
            with self.assertRaises(RuntimeError) as ctx:
                model.load_checkpoint("unused.pt", optimizer=None, goal_prior_bridge=True)
        message = str(ctx.exception)
        self.assertIn("must be exact", message)
        self.assertIn("base.weight", message)
        self.assertNotIn("lora_A", message, "adapters must not be reported as disallowed")

    def test_merged_save_folds_adapters_and_keeps_one_copy(self):
        """A `lora_merged` payload must be plain, single-copy and adapter-free.

        Three ways the merged save can go wrong, all silent until a loader trips:
        the adapters leak in as `lora_A`/`lora_B` (they are parameters *of* the
        wrapper, so a prefix filter over wrapped modules misses them), the
        doubly-bound FLUX blocks are written twice, and the merged weights are
        never folded at all.
        """
        from imagewam.models.backbones.lora import lora_merged_state_dict

        video = DummyExpert()
        video.transformer = nn.ModuleList([nn.Linear(4, 4)])
        video.blocks = video.transformer  # the flattened alias
        apply_lora_to_linear_suffixes(video.transformer, target_suffixes=("0",), rank=2, alpha=2.0)
        with torch.no_grad():
            video.transformer[0].lora_B.fill_(1.0)

        merged = lora_merged_state_dict(DummyMoT(video, DummyExpert()))

        self.assertFalse([k for k in merged if ".lora_" in k], "adapters leaked into a merged payload")
        self.assertFalse([k for k in merged if ".base." in k], "wrapped names leaked into a merged payload")
        # One copy per parameter: the `blocks` alias is the same module as
        # `transformer.0`, so it must not appear a second time.
        self.assertIn("mixtures.video.transformer.0.weight", merged)
        self.assertNotIn("mixtures.video.blocks.0.weight", merged)
        # scaling is alpha/rank = 1.0, so the fold is base + B @ A.
        expected = video.transformer[0].base.weight + video.transformer[0].lora_B @ video.transformer[0].lora_A
        self.assertTrue(torch.allclose(merged["mixtures.video.transformer.0.weight"], expected.cpu(), atol=1e-6))

    def test_plain_model_loads_a_merged_payload_unchanged(self):
        """A merged checkpoint must still load into a model built WITHOUT LoRA.

        That is what evaluation does: it builds the model from the config, where
        `enabled: false` injects no wrappers, then loads the run's checkpoint. The
        helpers run on every flux2 load, so if they rewrote a plain `….qkv.weight`
        into the wrapper's `….qkv.base.weight` they would leave the unwrapped model
        with a missing key and nothing to load.
        """
        from imagewam.models.backbones.lora import (
            collapse_aliased_transformer_keys,
            merge_lora_state_dict_to_plain,
            remap_plain_linear_keys_to_lora_base,
            strip_mot_prefix,
        )

        class UnwrappedExpert(DummyExpert):
            """No LoRA wrappers -- the blocks are plain Linears, bound twice."""

            def __init__(self):
                super().__init__()
                inner = nn.Module()
                inner.blocks = nn.ModuleList([nn.Linear(4, 4)])
                self.transformer = inner
                self.blocks = inner.blocks  # the flattened alias

        video = UnwrappedExpert()
        model = self._loadable_model(video, DummyExpert(), stage=None)

        payload = {
            "mot": {
                "mot.mixtures.video.transformer.blocks.0.weight": torch.full((4, 4), 0.5),
                "mot.mixtures.video.blocks.0.weight": torch.full((4, 4), 0.5),
            },
            "step": 1,
        }
        state = strip_mot_prefix(merge_lora_state_dict_to_plain(payload["mot"]))
        state = remap_plain_linear_keys_to_lora_base(model.mot, state)
        state = collapse_aliased_transformer_keys(model.mot, state)

        self.assertFalse(
            [k for k in state if ".base." in k],
            "a plain key was rewritten into the wrapper's form, which an unwrapped model cannot load",
        )
        result = model.mot.load_state_dict(state, strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(
            torch.equal(video.transformer.blocks[0].weight, torch.full((4, 4), 0.5)),
            "the merged weight did not reach the unwrapped layer",
        )

    def test_flux2_layer_checkpoint_keeps_input_grads(self):
        from imagewam.models.backbones.mot import MoT

        class _Flags:
            mot_checkpoint_mixed_attn = True
            training = True

        frozen = torch.randn(4, 4)
        tokens = torch.randn(2, 4, requires_grad=True)

        def _layer(x):
            return x @ frozen

        out = MoT._flux2_maybe_checkpoint(_Flags(), _layer, tokens)
        out.sum().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertIsNone(frozen.grad)

        flags_eval = _Flags()
        flags_eval.training = False
        tokens2 = torch.randn(2, 4, requires_grad=True)
        out2 = MoT._flux2_maybe_checkpoint(flags_eval, _layer, tokens2)
        out2.sum().backward()
        self.assertIsNotNone(tokens2.grad)

    def test_stage2_requires_stage1_checkpoint_unless_resuming(self):
        from imagewam.trainer import Wan22Trainer

        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.resume = None
        trainer.stage1_checkpoint = None
        trainer.model = type("M", (), {"goal_prior_stage": "stage2"})()
        with self.assertRaisesRegex(ValueError, "requires `stage1_checkpoint`"):
            Wan22Trainer._load_stage1_bridge_checkpoint_before_prepare(trainer)

        trainer.model.goal_prior_stage = "stage1"
        Wan22Trainer._load_stage1_bridge_checkpoint_before_prepare(trainer)

        trainer.model.goal_prior_stage = "stage2"
        trainer.resume = "/tmp/stage2_resume"
        Wan22Trainer._load_stage1_bridge_checkpoint_before_prepare(trainer)


if __name__ == "__main__":
    unittest.main()
