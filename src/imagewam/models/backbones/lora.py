from __future__ import annotations

import math
from dataclasses import dataclass
from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from imagewam.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapper for an existing Linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"`rank` must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)

    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        delta = (self.lora_B.float() @ self.lora_A.float()) * float(self.scaling)
        merged.weight.data.copy_((self.base.weight.float() + delta).to(dtype=self.base.weight.dtype))
        if self.base.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged


@dataclass
class LoRAApplyResult:
    replaced: int
    target_suffixes: tuple[str, ...]


def is_lora_adapter_key(key: str) -> bool:
    """Whether a state-dict key names a LoRA adapter tensor rather than a base weight.

    Adapters are stage-2-only: a Stage 2 run wraps the video expert and then loads
    a Stage 1 payload saved before the wrappers existed, so `lora_A`/`lora_B` are
    missing by construction and are initialised fresh. Callers use this to tell
    those apart from a genuinely absent FLUX or ActionDiT weight.
    """
    return key.endswith(".lora_A") or key.endswith(".lora_B")


def strip_mot_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Re-key a payload so it is relative to the MoT subtree.

    `save_checkpoint` stores `self.mot.state_dict()`, but called on the expert
    itself those keys come back stripped of the leading `mot.` -- Pytorch only
    adds a component for a *child* module. The LoRA helpers derive names from
    `named_modules()`, so a prefixed payload would match nothing and every key
    would be reported missing and unexpected at once.
    """
    prefix = None
    for key in state_dict:
        parts = key.split(".")
        if len(parts) > 1 and parts[1] == "mixtures":
            prefix = parts[0] + "."
            break
    if prefix is None:
        return state_dict
    stripped = {}
    for key, value in state_dict.items():
        stripped[key[len(prefix) :] if key.startswith(prefix) else key] = value
    logger.info("Stripped %r from %d checkpoint keys before the LoRA remap.", prefix, len(stripped))
    return stripped


def _iter_named_linears(module: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            yield name, child


def _get_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora_to_linear_suffixes(
    module: nn.Module,
    *,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> LoRAApplyResult:
    suffixes = tuple(str(item) for item in target_suffixes)
    replaced = 0
    for name, linear in list(_iter_named_linears(module)):
        if isinstance(linear, LoRALinear):
            continue
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        parent, child_name = _get_parent(module, name)
        setattr(parent, child_name, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1
    logger.info("Applied LoRA to %d Linear layers; target_suffixes=%s", replaced, suffixes)
    return LoRAApplyResult(replaced=replaced, target_suffixes=suffixes)


def merge_lora_linear_layers(module: nn.Module) -> int:
    """Replace all LoRALinear modules in-place with merged plain Linear layers."""
    replaced = 0
    for name, child in list(module.named_modules()):
        if not isinstance(child, LoRALinear):
            continue
        parent, child_name = _get_parent(module, name)
        setattr(parent, child_name, child.merged_linear())
        replaced += 1
    logger.info("Merged %d LoRA Linear layers into base weights.", replaced)
    return replaced


def _module_path_groups(module: nn.Module) -> dict[int, list[str]]:
    """Every path each module is reachable by, keyed by object identity.

    `named_modules()` walks depth-first and stops at the first name it meets for a
    submodule, so a module bound twice is reported once. `remove_duplicate=False`
    reports every registration -- which is what tells us that
    `mixtures.video.transformer.double_blocks.0` and
    `mixtures.video.double_blocks.0` are the same layer.
    """
    groups: dict[int, list[str]] = {}
    for name, child in module.named_modules(remove_duplicate=False):
        groups.setdefault(id(child), []).append(name)
    return groups


def lora_merged_state_dict(module: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Return a state dict where LoRALinear modules appear as plain Linear layers.

    The result carries the *unwrapped* names a model built without LoRA expects, so
    every trace of the wrapper has to go: the `lora_A`/`lora_B` parameters, the
    `.base` segment that only exists because the wrapper keeps the original layer
    as a child, and the flattened twin of each doubly-bound FLUX block -- which
    would otherwise be a second, un-merged copy of the same weight.
    """
    groups = _module_path_groups(module)
    # The path `named_modules()` reports is the one `load_state_dict` will walk --
    # it dedupes by object identity too, so writing the alias as well would add a
    # key no unwrapped model expects.
    canonical_path = {}
    for name, child in module.named_modules():
        canonical_path[id(child)] = name
    # The wrapper's own parameters, named under each path it is reachable by. They
    # are folded into `.weight`/`.bias` above, so copying them through as well would
    # leave the wrapper's bookkeeping in a payload meant for an unwrapped model.
    wrapper_leaves = ("base.weight", "base.bias", "lora_A", "lora_B")

    merged = OrderedDict()
    replaced: set[str] = set()
    for child in module.modules():
        if not isinstance(child, LoRALinear):
            continue
        linear = child.merged_linear()
        path = canonical_path[id(child)]
        merged[f"{path}.weight"] = linear.weight.detach().cpu()
        if linear.bias is not None:
            merged[f"{path}.bias"] = linear.bias.detach().cpu()
        for alias in groups[id(child)]:
            replaced |= {f"{alias}.{leaf}" for leaf in wrapper_leaves}

    for key, value in module.state_dict().items():
        if key in replaced:
            continue
        merged[key] = value.detach().cpu()
    return merged


def remap_plain_linear_keys_to_lora_base(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map plain Linear checkpoint keys into LoRALinear `.base` keys for resume."""
    lora_names = {name for name, child in module.named_modules() if isinstance(child, LoRALinear)}
    if not lora_names:
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        mapped_key = key
        for name in lora_names:
            if key == f"{name}.weight":
                mapped_key = f"{name}.base.weight"
                break
            if key == f"{name}.bias":
                mapped_key = f"{name}.base.bias"
                break
        remapped[mapped_key] = value
    return remapped


def collapse_aliased_transformer_keys(
    module: nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Fold the flattened half of a doubly-bound `transformer` path onto the real one.

    `Flux2VideoExpert.__init__` binds the transformer *and* its two block lists:

        self.transformer = transformer
        self.double_blocks = transformer.double_blocks

    so the same parameters appear under two names, and a state dict built from a
    LoRA-wrapped MoT lists each block weight twice -- once as
    `…video.transformer.double_blocks.0.…` and once as `…video.double_blocks.0.…`.

    Only the `transformer` form is wrapped, so after the LoRA remap the two halves
    no longer agree: the wrapped form carries `.base.` and the flattened form does
    not. `load_state_dict` walks each registered path separately and demands a key
    for both, so the mismatched half surfaces as a missing *and* an unexpected key
    per block, which the exact-load check rejects.

    Both paths are therefore populated from the wrapped value, which is canonical:
    it owns the adapters and the merged base weights. Keys that address no
    registered parameter at all are dropped, since they can only be rejected.
    """
    # Both paths are registered, so every key needs a value under each name. The
    # wrapped name is canonical: it owns the adapters and the merged base weights,
    # so it wins whichever form the remap produced.
    wrapped = {key: value for key, value in state_dict.items() if ".video.transformer." in key}
    flattened = {key: value for key, value in state_dict.items() if ".video.transformer." not in key}
    collapsed = dict(state_dict)
    folded = 0

    # The wrapped half owns the adapters, so it wins wherever both forms appear.
    for key, value in wrapped.items():
        head, sep, tail = key.partition(".video.transformer.")
        for alias in (f"{head}.video.{tail}", tail):
            if alias in collapsed and collapsed[alias] is not value:
                collapsed[alias] = value
                folded += 1

    # Then the reverse: a flattened path the payload never carried still needs a
    # value, because the expert registers it a second time.
    for key, value in wrapped.items():
        head, sep, tail = key.partition(".video.transformer.")
        alias = f"{head}.video.{tail}"
        if alias not in collapsed:
            collapsed[alias] = value
            folded += 1
    for key, value in flattened.items():
        head, sep, tail = key.partition(".video.")
        canonical = f"{head}.video.transformer.{tail}" if sep else key
        if canonical not in collapsed:
            collapsed[canonical] = value
            folded += 1

    # Finally, drop anything that addresses no parameter at all -- a stale form of
    # a name the model no longer registers would be rejected as unexpected.
    resolvable = set(module.state_dict())
    kept = {key: value for key, value in collapsed.items() if key in resolvable}
    if folded or len(kept) != len(collapsed):
        logger.info(
            "Folded %d aliased video-expert keys; dropped %d that address no parameter.",
            folded, len(collapsed) - len(kept),
        )
    return kept


def merge_lora_state_dict_to_plain(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a LoRA-wrapper state dict into plain Linear keys.

    This is a compatibility path for checkpoints saved before LoRA weights were
    merged at save time. The original config used alpha=rank, so scaling=1.0;
    future checkpoints should prefer `checkpoint_format=lora_merged`.
    """
    prefixes = set()
    for key in state_dict:
        if key.endswith(".lora_A"):
            prefixes.add(key[: -len(".lora_A")])
    if not prefixes:
        return state_dict

    merged: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for prefix in prefixes:
        base_key = f"{prefix}.base.weight"
        a_key = f"{prefix}.lora_A"
        b_key = f"{prefix}.lora_B"
        if base_key not in state_dict or a_key not in state_dict or b_key not in state_dict:
            continue
        base = state_dict[base_key]
        lora_a = state_dict[a_key]
        lora_b = state_dict[b_key]
        # Historical FLUX.2 LoRA configs used alpha=rank, so alpha/rank=1.
        delta = lora_b.float() @ lora_a.float()
        merged[f"{prefix}.weight"] = (base.float() + delta).to(dtype=base.dtype)
        consumed.update({base_key, a_key, b_key})
        bias_key = f"{prefix}.base.bias"
        if bias_key in state_dict:
            merged[f"{prefix}.bias"] = state_dict[bias_key]
            consumed.add(bias_key)

    for key, value in state_dict.items():
        if key in consumed:
            continue
        if ".lora_A" in key or ".lora_B" in key or ".base." in key:
            continue
        merged[key] = value
    return merged
