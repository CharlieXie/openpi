"""PEFT-style LoRA for Waypoint Joint Model (PI0WaypointJoint).

Key design choices (following HuggingFace PEFT):
  - **Wrapper pattern**: LoRALinear wraps the original nn.Linear as
    ``self.base_layer`` — zero-copy, no ``.clone()`` of weights.
  - **Default-all + skip-list**: every nn.Linear gets LoRA unless its
    full path matches an entry in ``modules_to_not_lora``.
  - **LoRA-only save/load**: ``get_lora_state_dict`` / ``load_lora_state_dict``
    only touch trainable parameters (LoRA A/B + whitelisted non-LoRA modules).
  - **DDP-friendly**: frozen base weights don't participate in gradient sync;
    only LoRA + trainable-module params are synced.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from safetensors.torch import save_file as safetensors_save

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Per-layer LoRA hyper-parameters."""

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    # rsLoRA: scale by alpha/sqrt(rank) instead of alpha/rank
    use_rslora: bool = False
    # Initialisation for lora_A.
    #   True / "kaiming" → kaiming_uniform (PEFT / LoRA paper default)
    #   "gaussian"       → normal(0, 1/rank)  (PEFT gaussian option)
    init_lora_weights: bool | str = True

    @property
    def scaling(self) -> float:
        if self.use_rslora:
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank


# ---------------------------------------------------------------------------
# LoRALinear — PEFT-style wrapper (zero-copy)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps an existing ``nn.Linear`` with LoRA adapters.

    The original layer is stored as ``self.base_layer`` **by reference** —
    no ``.clone()`` or memory duplication.  During forward the output is::

        y = base_layer(x) + lora_B(lora_A(dropout(x))) * scaling
    """

    def __init__(self, base_layer: nn.Linear, config: LoRAConfig):
        super().__init__()
        # Store original layer — zero copy
        self.base_layer = base_layer
        self.config = config
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA low-rank matrices
        self.lora_A = nn.Linear(in_features, config.rank, bias=False)
        self.lora_B = nn.Linear(config.rank, out_features, bias=False)
        self.scaling = config.scaling

        if config.dropout > 0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = nn.Identity()

        # Initialise: B = zero (always), A depends on config
        # This ensures the initial LoRA contribution is exactly 0.
        nn.init.zeros_(self.lora_B.weight)
        init = config.init_lora_weights
        if init is True or (isinstance(init, str) and init.lower() == "kaiming"):
            # PEFT / LoRA paper default: same as nn.Linear
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif isinstance(init, str) and init.lower() == "gaussian":
            # PEFT gaussian option: normal(0, 1/rank)
            nn.init.normal_(self.lora_A.weight, std=1.0 / config.rank)
        else:
            raise ValueError(f"Unknown init_lora_weights={init!r}")

        # Place LoRA params on same device / dtype as base
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A.to(device=device, dtype=dtype)
        self.lora_B.to(device=device, dtype=dtype)

    # -- forward --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        # Cast input for LoRA branch if needed (e.g. fp32 input, bf16 LoRA)
        lora_input = self.lora_dropout(x)
        if lora_input.dtype != self.lora_A.weight.dtype:
            lora_input = lora_input.to(self.lora_A.weight.dtype)
        lora_out = self.lora_B(self.lora_A(lora_input))
        return result + lora_out * self.scaling

    # -- merge / unmerge for inference ----------------------------------
    def merge(self) -> None:
        """Merge LoRA weights into base layer (irreversible without saved delta)."""
        with torch.no_grad():
            delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.base_layer.weight.data += delta.to(self.base_layer.weight.dtype)

    def get_delta_weight(self) -> torch.Tensor:
        """Return B @ A * scaling without modifying base."""
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    # -- convenience properties ----------------------------------------
    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"lora_rank={self.config.rank}, lora_alpha={self.config.alpha}"
        )


# ---------------------------------------------------------------------------
# Training-level config
# ---------------------------------------------------------------------------

# Default modules to NOT apply LoRA to (full-train or freeze instead).
# Reasoning for each entry — see WAYPOINT_LORA_DESIGN.md.
_DEFAULT_MODULES_TO_NOT_LORA: list[str] = [
    # AE-specific projections: input dim too small (32) for LoRA to compress
    "action_in_proj",
    "action_out_proj",
    "proprio_encoder",
    # Time conditioning MLPs: new layers in joint model, must full-train
    "time_mlp_in",
    "time_mlp_out",
    # Vision-language bridge: single small layer
    "multi_modal_projector",
    # embed_tokens is nn.Embedding (not nn.Linear) — won't be LoRA'd anyway,
    # but listed here to also exclude from trainable_non_lora_modules.
    "embed_tokens",
    # lm_head (nn.Linear) is weight-tied with embed_tokens (same tensor).
    # Joint model never calls lm_head.forward() — it uses embed_tokens.weight
    # directly via F.linear(). LoRA on lm_head would:
    #   (a) allocate dead params (forward never called → no gradients)
    #   (b) break weight tying (state_dict key changes to .base_layer.weight)
    # PEFT also explicitly excludes get_output_embeddings() for this reason.
    "lm_head",
]

# Default modules among the skip-list that should remain trainable
# (the rest are frozen).
# embed_tokens (527M, nn.Embedding) and lm_head (tied) are trainable
# because they contain waypoint token embeddings that must be learned.
_DEFAULT_TRAINABLE_NON_LORA: list[str] = [
    "action_in_proj",
    "action_out_proj",
    "proprio_encoder",
    "time_mlp_in",
    "time_mlp_out",
    "multi_modal_projector",
    "embed_tokens",
    "lm_head",
]


@dataclass
class LoRATrainingConfig:
    """Training-level LoRA configuration for the Waypoint Joint Model."""

    enabled: bool = False

    # LoRA hyper-parameters (applied uniformly to all LoRA'd layers)
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    use_rslora: bool = False
    # Initialisation for lora_A: True/"kaiming" or "gaussian"
    init_lora_weights: bool | str = True

    # Skip-list: full path substrings — any nn.Linear whose full
    # ``named_modules()`` path contains one of these strings is **not**
    # wrapped with LoRA.
    modules_to_not_lora: list[str] = field(
        default_factory=lambda: list(_DEFAULT_MODULES_TO_NOT_LORA),
    )

    # Among skipped modules, which should remain trainable (the rest freeze).
    trainable_non_lora_modules: list[str] = field(
        default_factory=lambda: list(_DEFAULT_TRAINABLE_NON_LORA),
    )

    # Vision encoder (SigLIP) handling
    vision_encoder_mode: Literal["freeze", "lora", "full"] = "freeze"

    # Which sub-models to apply LoRA to
    apply_to: Literal["all", "backbone_only", "expert_only"] = "all"

    def get_lora_config(self) -> LoRAConfig:
        return LoRAConfig(
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            use_rslora=self.use_rslora,
            init_lora_weights=self.init_lora_weights,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_parent_and_attr(model: nn.Module, dotted_name: str):
    """Return (parent_module, attr_name) for a dotted module path."""
    parts = dotted_name.rsplit(".", 1)
    if len(parts) == 2:
        return model.get_submodule(parts[0]), parts[1]
    return model, dotted_name


def _is_vision_tower(name: str) -> bool:
    return "vision_tower" in name or "vision_model" in name


def _should_skip_lora(name: str, config: LoRATrainingConfig) -> bool:
    """Return True if this module path should NOT be wrapped with LoRA."""
    # Check skip-list
    for pattern in config.modules_to_not_lora:
        if pattern in name:
            return True

    # Vision tower: skip unless vision_encoder_mode == "lora"
    if _is_vision_tower(name) and config.vision_encoder_mode != "lora":
        return True

    # apply_to filter
    if config.apply_to == "backbone_only":
        if "gemma_expert" in name:
            return True
    elif config.apply_to == "expert_only":
        if "paligemma" in name:
            return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    model: nn.Module,
    config: LoRATrainingConfig,
) -> tuple[int, int]:
    """Apply LoRA to a PI0WaypointJoint model.

    Every ``nn.Linear`` whose full path does **not** match any entry in
    ``config.modules_to_not_lora`` (and passes the vision / apply_to
    filters) gets wrapped with ``LoRALinear``.

    After wrapping, parameters are frozen / unfrozen according to
    ``config.trainable_non_lora_modules`` and ``config.vision_encoder_mode``.

    Returns:
        (frozen_param_count, trainable_param_count)
    """
    if not config.enabled:
        logger.info("LoRA is disabled, skipping")
        total = sum(p.numel() for p in model.parameters())
        return 0, total

    lora_config = config.get_lora_config()
    lora_count = 0
    lora_details: list[str] = []

    # Phase 1: wrap Linear layers with LoRALinear
    # Use list() to snapshot because we mutate the tree in-place.
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if _should_skip_lora(name, config):
            continue

        parent, attr = _get_parent_and_attr(model, name)
        wrapped = LoRALinear(module, lora_config)
        setattr(parent, attr, wrapped)
        lora_count += 1
        lora_details.append(
            f"  LoRA: {name}  ({module.in_features}→{module.out_features})"
        )

    logger.info(f"Applied LoRA (rank={lora_config.rank}) to {lora_count} layers:")
    for line in lora_details:
        logger.info(line)

    # Phase 2: freeze / unfreeze
    frozen, trainable = _freeze_for_lora(model, config)
    return frozen, trainable


def _freeze_for_lora(
    model: nn.Module,
    config: LoRATrainingConfig,
) -> tuple[int, int]:
    """Set requires_grad for LoRA training.

    Rules (in priority order):
      1. LoRA params (lora_A, lora_B)  → trainable
      2. Vision tower                   → depends on vision_encoder_mode
      3. Params matching trainable_non_lora_modules → trainable
      4. Everything else                → frozen
    """
    trainable_set = set(config.trainable_non_lora_modules)

    frozen_count = 0
    trainable_count = 0
    vision_count = 0

    for name, param in model.named_parameters():
        should_train = False

        # Rule 1: LoRA params always trainable
        if "lora_A" in name or "lora_B" in name:
            should_train = True

        # Rule 2: vision tower
        elif _is_vision_tower(name):
            if config.vision_encoder_mode == "full":
                should_train = True
            elif config.vision_encoder_mode == "lora":
                # LoRA params handled by rule 1; base weights stay frozen
                should_train = False
            else:  # "freeze"
                should_train = False
            if should_train:
                vision_count += param.numel()

        # Rule 3: trainable non-LoRA modules
        elif any(pattern in name for pattern in trainable_set):
            should_train = True

        param.requires_grad = should_train
        if should_train:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    total = trainable_count + frozen_count
    ratio = trainable_count / total * 100 if total > 0 else 0
    logger.info(
        f"LoRA freeze: {trainable_count:,} trainable, "
        f"{frozen_count:,} frozen  ({ratio:.2f}% trainable)"
    )
    if config.vision_encoder_mode == "freeze":
        logger.info("  Vision encoder (SigLIP): FROZEN")
    elif config.vision_encoder_mode == "full":
        logger.info(f"  Vision encoder (SigLIP): FULL-TRAIN ({vision_count:,} params)")
    else:
        logger.info("  Vision encoder (SigLIP): LoRA-only")

    return frozen_count, trainable_count


# ---------------------------------------------------------------------------
# LoRA-only save / load
# ---------------------------------------------------------------------------

def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect only trainable parameters (LoRA + non-LoRA trainable modules).

    This is the minimal set needed to resume LoRA training or run inference
    (after merging LoRA weights back into the base model).
    """
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data
    return state


def save_lora_checkpoint(
    model: nn.Module,
    path: str,
    *,
    use_safetensors: bool = True,
) -> None:
    """Save only the trainable (LoRA + non-LoRA) parameters.

    Args:
        model: The model (possibly wrapped in DDP).
        path: Output file path (.safetensors or .pt).
        use_safetensors: If True, use safetensors format (recommended).
    """
    # Unwrap DDP if necessary
    raw = model.module if hasattr(model, "module") else model
    state = get_lora_state_dict(raw)

    if use_safetensors:
        safetensors_save(state, path)
    else:
        torch.save(state, path)

    logger.info(f"Saved LoRA checkpoint ({len(state)} tensors) to {path}")


def load_lora_checkpoint(
    model: nn.Module,
    path: str,
) -> None:
    """Load a LoRA checkpoint (trainable params only) into *model*.

    The model must already have LoRA layers injected (via ``apply_lora_to_model``).
    Missing keys in the checkpoint are silently ignored (they keep their
    init values), which is fine for a fresh LoRA start from base weights.

    Args:
        model: The model with LoRA layers already applied.
        path: Path to ``.safetensors`` or ``.pt`` file.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)

    missing, unexpected = [], []
    model_state = model.state_dict()
    for key, val in state.items():
        if key in model_state:
            if model_state[key].shape == val.shape:
                model_state[key].copy_(val)
            else:
                logger.warning(
                    f"Shape mismatch for {key}: model={model_state[key].shape}, "
                    f"ckpt={val.shape} — skipping"
                )
                missing.append(key)
        else:
            unexpected.append(key)

    if missing:
        logger.warning(f"Missing / shape-mismatch keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in LoRA checkpoint: {unexpected}")
    logger.info(
        f"Loaded LoRA checkpoint from {path} "
        f"({len(state) - len(missing) - len(unexpected)} tensors applied)"
    )


# ---------------------------------------------------------------------------
# Merge all LoRA layers (for inference deployment)
# ---------------------------------------------------------------------------

def merge_lora_weights(model: nn.Module) -> int:
    """Merge all LoRALinear layers back into their base layers.

    After merging, the model behaves identically to a fully fine-tuned
    model with no LoRA overhead.  This is **irreversible** unless you
    saved the LoRA checkpoint separately.

    Returns:
        Number of layers merged.
    """
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            module.merge()
            # Replace LoRALinear with the now-merged base_layer
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, module.base_layer)
            count += 1
    logger.info(f"Merged {count} LoRA layers into base weights")
    return count


# ---------------------------------------------------------------------------
# Utility: print model LoRA summary
# ---------------------------------------------------------------------------

def print_lora_summary(model: nn.Module) -> None:
    """Print a summary of LoRA layers and trainable parameter counts."""
    lora_params = 0
    trainable_non_lora = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params += param.numel()
        elif param.requires_grad:
            trainable_non_lora += param.numel()
        else:
            frozen_params += param.numel()

    total = lora_params + trainable_non_lora + frozen_params
    logger.info("=" * 60)
    logger.info("LoRA Summary")
    logger.info(f"  LoRA params:            {lora_params:>12,}")
    logger.info(f"  Trainable non-LoRA:     {trainable_non_lora:>12,}")
    logger.info(f"  Frozen params:          {frozen_params:>12,}")
    logger.info(f"  Total params:           {total:>12,}")
    logger.info(f"  Trainable ratio:        {(lora_params + trainable_non_lora) / total * 100:.2f}%")
    logger.info("=" * 60)
