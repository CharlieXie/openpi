"""Joint VLM + Action Expert model for waypoint VLA.

Shares a single PaliGemma backbone between:
  - VLM: autoregressive waypoint token prediction (CE loss)
  - AE: flow-matching action prediction (MSE loss)

Gradient strategy options:
  - "none": both losses freely update all parameters
  - "stop_gradient": Knowledge Insulation (Pi0.5 §5.2) — detach backbone
    K/V inside every attention layer so MSE loss cannot update backbone
    weights through cross-attention. Backbone only receives VLM CE gradients.
  - "scale_gradient": Soft KI — scale (not detach) AE gradients flowing
    into backbone K/V by ``gradient_scale`` (e.g. 0.1 = 10% of AE gradient
    reaches backbone). Lets backbone slowly adapt to AE needs while VLM CE
    remains the dominant training signal.
  - "freeze_backbone": freeze all paligemma backbone parameters entirely
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    sample_beta,
)
from openpi.waypoint.ae_model import DURATION_MAX
from openpi.waypoint.tokenizer import (
    _DURATION_BASE,
    _GRIP_CLOSE_ID,
    _GRIP_OPEN_ID,
    _PROPRIO_BASE,
    DURATION_N_BINS,
    PROPRIO_N_BINS,
)

logger = logging.getLogger(__name__)


class PI0WaypointJoint(nn.Module):
    """Joint VLM + AE model sharing a single PaliGemma backbone.

    The VLM head uses ``paligemma_with_expert.paligemma`` for autoregressive
    waypoint token prediction. The AE head uses the full
    ``paligemma_with_expert`` (backbone + gemma_expert) for flow-matching
    action prediction, exactly as in the standalone AE model.
    """

    def __init__(
        self,
        config,
        vlm_max_token_len: int = 256,
        gradient_strategy: str = "none",
        gradient_scale: float = 0.1,
        aug_cfg: dict | None = None,
    ):
        super().__init__()
        assert gradient_strategy in ("none", "stop_gradient", "scale_gradient", "freeze_backbone"), (
            f"Unknown gradient_strategy: {gradient_strategy}"
        )
        self.config = config
        self.vlm_max_token_len = vlm_max_token_len
        self.gradient_strategy = gradient_strategy
        self.gradient_scale = gradient_scale
        self.aug_cfg = aug_cfg
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.duration_max = getattr(config, "duration_max", DURATION_MAX)

        # --- Shared backbone + expert (single instance) ---
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        expert_width = action_expert_config.width

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        # --- AE-specific projections (identical to ae_model.py) ---
        self.action_in_proj = nn.Linear(self.action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, self.action_dim)
        self.proprio_encoder = nn.Linear(self.action_dim, expert_width)
        self.time_mlp_in = nn.Linear(2 * expert_width, expert_width)
        self.time_mlp_out = nn.Linear(expert_width, expert_width)

        self.gradient_checkpointing_enabled = False

        # Apply freeze_backbone strategy
        if self.gradient_strategy == "freeze_backbone":
            for param in self.paligemma_with_expert.paligemma.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        # VLM path: use HF API so that the inner GemmaModel / SiglipEncoder
        # (which actually check the flag) get gradient_checkpointing=True.
        # Just setting the attribute on the outer wrapper does NOT propagate.
        self.paligemma_with_expert.paligemma.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # AE path: the custom paligemma_with_expert.forward() uses its own
        # torch.utils.checkpoint.checkpoint() calls, keyed off this attribute.
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.gradient_checkpointing_disable()
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

    def _ckpt(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    # ------------------------------------------------------------------
    # LM logits helper (supports LoRAEmbedding)
    # ------------------------------------------------------------------
    def _compute_lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute LM logits from hidden states.

        If embed_tokens has been wrapped with LoRAEmbedding, uses
        ``compute_logits()`` which applies the LoRA correction.
        Otherwise falls back to plain ``F.linear(h, weight)``.
        """
        embed_layer = self.paligemma_with_expert.paligemma.language_model.embed_tokens
        if hasattr(embed_layer, "compute_logits"):
            return embed_layer.compute_logits(hidden)
        # Cast hidden to weight dtype to avoid creating a large float32
        # copy of the embedding weight (~2 GB) in the autograd graph.
        w = embed_layer.weight
        out_dtype = hidden.dtype
        h = hidden.to(w.dtype) if hidden.dtype != w.dtype else hidden
        return F.linear(h, w).to(out_dtype)

    # ------------------------------------------------------------------
    # DDP-compatible forward dispatcher
    # ------------------------------------------------------------------
    def forward(self, mode: str, **kwargs):
        """Dispatch to vlm_forward or ae_forward.

        This method exists so that DDP can route through ``__call__`` and
        properly track which parameters are used in each forward pass.

        Args:
            mode: ``"vlm"`` or ``"ae"``.
            **kwargs: forwarded to the corresponding method.
        """
        if mode == "vlm":
            return self.vlm_forward(kwargs["batch"])
        elif mode == "ae":
            return self.ae_forward(**kwargs)
        else:
            raise ValueError(f"Unknown forward mode: {mode}")

    # ------------------------------------------------------------------
    # AE embedding helpers (from ae_model.py)
    # ------------------------------------------------------------------
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Embed images + language tokens for PaliGemma prefix."""
        embs, pad_masks, att_masks = [], [], []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._ckpt(self.paligemma_with_expert.embed_image, img)
            bsize, n_img = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, n_img))
            att_masks += [0] * n_img

        lang_emb = self._ckpt(self.paligemma_with_expert.embed_language_tokens, lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        bsize = pad_masks.shape[0]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, -1)
        return embs, pad_masks, att_masks

    def embed_suffix(self, start_proprio, end_proprio, noisy_actions, timestep, duration):
        """Embed proprio conditioning + noisy actions + time/duration for expert."""
        embs, pad_masks, att_masks = [], [], []
        device = noisy_actions.device
        bsize = noisy_actions.shape[0]
        expert_width = self.action_in_proj.out_features

        proprio_input = torch.stack([start_proprio, end_proprio], dim=1)
        proprio_input = proprio_input.to(self.proprio_encoder.weight.dtype)
        proprio_emb = self._ckpt(self.proprio_encoder, proprio_input)
        embs.append(proprio_emb)
        pad_masks.append(torch.ones(bsize, 2, dtype=torch.bool, device=device))
        att_masks += [1, 0]

        time_emb = create_sinusoidal_pos_embedding(
            timestep, expert_width, min_period=4e-3, max_period=4.0, device=device,
        ).to(timestep.dtype)

        dur_normalized = duration / self.duration_max
        dur_emb = create_sinusoidal_pos_embedding(
            dur_normalized, expert_width, min_period=4e-3, max_period=4.0, device=device,
        ).to(timestep.dtype)

        combined = torch.cat([time_emb, dur_emb], dim=-1)
        combined = combined.to(self.time_mlp_in.weight.dtype)
        x = self.time_mlp_in(combined)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        adarms_cond = F.silu(x)

        noisy_actions_cast = noisy_actions.to(self.action_in_proj.weight.dtype)
        action_emb = self._ckpt(self.action_in_proj, noisy_actions_cast)
        embs.append(action_emb)
        pad_masks.append(torch.ones(bsize, self.action_horizon, dtype=torch.bool, device=device))
        att_masks += [1] + [0] * (self.action_horizon - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=device)
        att_masks = att_masks[None, :].expand(bsize, -1)

        return embs, pad_masks, att_masks, adarms_cond

    # ------------------------------------------------------------------
    # VLM forward (CE loss on waypoint tokens)
    # ------------------------------------------------------------------
    def vlm_forward(self, batch: dict) -> torch.Tensor:
        """VLM training forward: compute CE loss on waypoint tokens.

        Routes through ``self.paligemma_with_expert.paligemma`` (the shared
        backbone). Identical logic to PI0WaypointVLM.forward().
        """
        tokens = batch["tokens"]
        token_mask = batch["token_mask"]
        ar_mask = batch["ar_mask"]
        loss_mask = batch["loss_mask"]
        device = tokens.device
        B, L = tokens.shape

        embs_list = []
        pad_list = []
        ar_list = []

        image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for key in image_keys:
            if key not in batch["images"]:
                continue
            img = batch["images"][key]
            if img.shape[-1] == 3 and img.ndim == 4:
                img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW

            img_emb = self._ckpt(
                self.paligemma_with_expert.paligemma.model.get_image_features, img
            )
            n_img_tokens = img_emb.shape[1]
            embs_list.append(img_emb)

            img_mask = batch["image_masks"][key]
            pad_list.append(img_mask[:, None].expand(B, n_img_tokens))
            ar_list.append(torch.zeros(B, n_img_tokens, dtype=torch.int32, device=device))

        token_embs = self.paligemma_with_expert.paligemma.language_model.embed_tokens(tokens)
        token_embs = token_embs * math.sqrt(token_embs.shape[-1])
        embs_list.append(token_embs)
        pad_list.append(token_mask)
        ar_list.append(ar_mask)

        all_embs = torch.cat(embs_list, dim=1)
        all_pad = torch.cat(pad_list, dim=1)
        all_ar = torch.cat(ar_list, dim=1)

        n_img_total = all_embs.shape[1] - L
        full_loss_mask = torch.cat([
            torch.zeros(B, n_img_total, dtype=torch.bool, device=device),
            loss_mask,
        ], dim=1)

        att_2d = make_att_2d_masks(all_pad, all_ar)
        position_ids = torch.cumsum(all_pad, dim=1) - 1

        att_4d = att_2d[:, None, :, :]
        att_4d = torch.where(att_4d, 0.0, -2.3819763e38)

        model_dtype = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.dtype
        if all_embs.dtype != model_dtype:
            all_embs = all_embs.to(model_dtype)
        if att_4d.dtype != model_dtype:
            att_4d = att_4d.to(model_dtype)

        outputs = self.paligemma_with_expert.paligemma.language_model.forward(
            inputs_embeds=all_embs[:, :-1],
            attention_mask=att_4d[:, :, :-1, :-1],
            position_ids=position_ids[:, :-1],
            use_cache=False,
        )

        hidden = outputs.last_hidden_state

        targets = torch.cat([
            torch.zeros(B, n_img_total, dtype=torch.long, device=device),
            tokens,
        ], dim=1)[:, 1:]

        shift_loss_mask = full_loss_mask[:, 1:]

        mask = shift_loss_mask
        if mask.any():
            active_hidden = hidden[mask]
            active_targets = targets[mask]
            active_logits = self._compute_lm_logits(active_hidden.float())
            loss = F.cross_entropy(active_logits, active_targets)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss

    # ------------------------------------------------------------------
    # AE forward (MSE loss on flow-matching)
    # ------------------------------------------------------------------
    def ae_forward(
        self,
        observation,
        start_proprio,
        end_proprio,
        actions,
        duration,
        action_pad_mask=None,
        action_dim_mask=None,
        noise=None,
        time=None,
    ):
        """AE training forward: compute per-element MSE loss.

        When ``gradient_strategy == "stop_gradient"``, prefix embeddings are
        detached so that the MSE loss does not propagate gradients into the
        shared PaliGemma backbone.
        """
        import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
        obs = _preprocessing.preprocess_observation_pytorch(observation, train=True, aug_cfg=self.aug_cfg)
        images = list(obs.images.values())
        img_masks = list(obs.image_masks.values())
        lang_tokens = obs.tokenized_prompt
        lang_masks = obs.tokenized_prompt_mask

        bsize = actions.shape[0]
        device = actions.device

        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time_beta = sample_beta(1.5, 1.0, bsize, device)
            time = (time_beta * 0.999 + 0.001).float()

        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(
            start_proprio, end_proprio, x_t, time, duration,
        )

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)

        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_4d = att_2d[:, None, :, :]
        att_4d = torch.where(att_4d, 0.0, -2.3819763e38)

        # Knowledge Insulation: control gradient flow from AE MSE loss into
        # backbone weights through cross-attention K/V.
        #   - "stop_gradient": full detach (True)
        #   - "scale_gradient": partial gradient (float scale in (0,1))
        #   - "none" / "freeze_backbone": no insulation (False)
        if self.gradient_strategy == "stop_gradient":
            ki_value: bool | float = True
        elif self.gradient_strategy == "scale_gradient":
            ki_value = self.gradient_scale
        else:
            ki_value = False

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            knowledge_insulation=ki_value,
        )

        suffix_out = suffix_out[:, -self.action_horizon:].float()
        v_t = self.action_out_proj(suffix_out)

        loss_per_element = F.mse_loss(u_t, v_t, reduction="none")

        if action_pad_mask is not None:
            step_mask = (~action_pad_mask).float().unsqueeze(-1)
            loss_per_element = loss_per_element * step_mask

        if action_dim_mask is not None:
            dim_mask = action_dim_mask[:, None, :].float()
            loss_per_element = loss_per_element * dim_mask

        return loss_per_element.sum() / max(loss_per_element.nonzero().shape[0], 1)

    # ------------------------------------------------------------------
    # VLM inference: constrained AR generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_waypoints(
        self,
        images: dict[str, torch.Tensor],
        image_masks: dict[str, torch.Tensor],
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        wp_tokenizer,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[list[tuple]]:
        """AR generation of waypoint tokens using the shared backbone.

        Identical logic to PI0WaypointVLM.generate_waypoints(), routing
        through ``self.paligemma_with_expert.paligemma``.
        """
        paligemma = self.paligemma_with_expert.paligemma

        device = prompt_tokens.device
        B = prompt_tokens.shape[0]

        embs_list = []
        pad_list = []

        image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for key in image_keys:
            if key not in images:
                continue
            img = images[key]
            if img.shape[-1] == 3 and img.ndim == 4:
                img = img.permute(0, 3, 1, 2)
            img_emb = paligemma.model.get_image_features(img)
            n_img = img_emb.shape[1]
            embs_list.append(img_emb)
            pad_list.append(image_masks[key][:, None].expand(B, n_img))

        token_embs = paligemma.language_model.embed_tokens(prompt_tokens)
        token_embs = token_embs * math.sqrt(token_embs.shape[-1])
        embs_list.append(token_embs)
        pad_list.append(prompt_mask)

        prefix_embs = torch.cat(embs_list, dim=1)
        prefix_pad = torch.cat(pad_list, dim=1)

        prefix_len = prefix_embs.shape[1]
        prefix_ar = torch.zeros_like(prefix_pad, dtype=torch.int32)
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_ar)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1

        prefix_att_padded = F.pad(prefix_att_2d, (0, max_new_tokens), value=False)
        prefix_att_4d = prefix_att_padded[:, None, :, :]
        prefix_att_4d = torch.where(prefix_att_4d, 0.0, -2.3819763e38)

        model_dtype = paligemma.language_model.embed_tokens.weight.dtype
        if prefix_embs.dtype != model_dtype:
            prefix_embs = prefix_embs.to(model_dtype)
        if prefix_att_4d.dtype != model_dtype:
            prefix_att_4d = prefix_att_4d.to(model_dtype)

        outputs = paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        last_hidden = outputs.last_hidden_state[:, -1:]

        last_logits = self._compute_lm_logits(last_hidden.float())

        tpw = wp_tokenizer.tokens_per_waypoint
        prefill_len = torch.sum(prefix_pad, dim=-1)

        action_header = wp_tokenizer._pg_tokenizer.encode("Action: ")
        header_injected = [False] * B
        header_pos = [0] * B

        all_output_tokens = [[] for _ in range(B)]

        vocab_size = last_logits.shape[-1]
        proprio_lo = _PROPRIO_BASE - PROPRIO_N_BINS + 1
        proprio_hi = _PROPRIO_BASE
        duration_lo = _DURATION_BASE - DURATION_N_BINS + 1
        duration_hi = _DURATION_BASE

        # Pre-compute position constants from tokenizer
        grip_pos = wp_tokenizer.gripper_pos_in_wp
        dur_delim_pos = wp_tokenizer.dur_delimiter_pos_in_wp
        dur_val_pos = wp_tokenizer.duration_pos_in_wp
        use_grip = wp_tokenizer.use_gripper_token

        for step in range(max_new_tokens):
            # --- Logit masking (constrained decoding) ---
            for b in range(B):
                if header_injected[b]:
                    wp_count = len(all_output_tokens[b])
                    pos_in_wp = wp_count % tpw
                    is_proprio_pos = 1 <= pos_in_wp <= wp_tokenizer.proprio_dim
                    is_gripper_pos = use_grip and pos_in_wp == grip_pos
                    is_duration_pos = pos_in_wp == dur_val_pos

                    if is_proprio_pos:
                        mask_val = torch.full((vocab_size,), float('-inf'), device=device)
                        mask_val[proprio_lo : proprio_hi + 1] = 0.0
                        last_logits[b, 0] += mask_val
                    elif is_gripper_pos:
                        mask_val = torch.full((vocab_size,), float('-inf'), device=device)
                        mask_val[_GRIP_OPEN_ID] = 0.0
                        mask_val[_GRIP_CLOSE_ID] = 0.0
                        last_logits[b, 0] += mask_val
                    elif is_duration_pos:
                        mask_val = torch.full((vocab_size,), float('-inf'), device=device)
                        mask_val[duration_lo : duration_hi + 1] = 0.0
                        last_logits[b, 0] += mask_val

            if temperature > 0:
                probs = F.softmax(last_logits[:, 0] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(last_logits[:, 0], dim=-1, keepdim=True)

            # --- Force-inject structural delimiters ---
            for b in range(B):
                if not header_injected[b]:
                    if header_pos[b] < len(action_header):
                        next_token[b, 0] = action_header[header_pos[b]]
                        header_pos[b] += 1
                        continue
                    else:
                        header_injected[b] = True

                wp_count = len(all_output_tokens[b])
                pos_in_wp = wp_count % tpw

                if pos_in_wp == 0:
                    next_token[b, 0] = wp_tokenizer.wp_token_id
                elif pos_in_wp == dur_delim_pos:
                    next_token[b, 0] = wp_tokenizer.dur_token_id

                all_output_tokens[b].append(next_token[b, 0].item())

            token_emb = paligemma.language_model.embed_tokens(next_token)
            token_emb = token_emb * math.sqrt(token_emb.shape[-1])
            if token_emb.dtype != model_dtype:
                token_emb = token_emb.to(model_dtype)

            positions = prefill_len[:, None] + step
            attn_len = prefix_len + step + 1
            mask = torch.zeros(B, 1, 1, attn_len, device=device, dtype=model_dtype)

            outputs = paligemma.language_model.forward(
                inputs_embeds=token_emb,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            last_hidden = outputs.last_hidden_state
            last_logits = self._compute_lm_logits(last_hidden.float())

            all_done = all(len(t) >= tpw * wp_tokenizer.num_waypoints for t in all_output_tokens)
            if all_done:
                break

        results = []
        for b in range(B):
            tids = all_output_tokens[b]
            n_proprio_ok = sum(
                1 for i, t in enumerate(tids)
                if 1 <= (i % tpw) <= wp_tokenizer.proprio_dim
                and proprio_lo <= t <= proprio_hi
            )
            n_duration_ok = sum(
                1 for i, t in enumerate(tids)
                if (i % tpw) == dur_val_pos
                and duration_lo <= t <= duration_hi
            )
            n_proprio_total = sum(
                1 for i in range(len(tids)) if 1 <= (i % tpw) <= wp_tokenizer.proprio_dim
            )
            n_duration_total = sum(
                1 for i in range(len(tids)) if (i % tpw) == dur_val_pos
            )
            if n_proprio_total > 0 or n_duration_total > 0:
                logger.debug(
                    f"VLM raw tokens: proprio_in_range={n_proprio_ok}/{n_proprio_total}, "
                    f"duration_in_range={n_duration_ok}/{n_duration_total}"
                )
            waypoints = wp_tokenizer.decode_waypoints(tids)
            results.append(waypoints)

        return results

    # ------------------------------------------------------------------
    # AE inference: iterative denoising
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        start_proprio,
        end_proprio,
        duration,
        num_steps: int = 10,
        noise=None,
    ):
        """Inference: generate action chunk via iterative denoising."""
        import openpi.models_pytorch.preprocessing_pytorch as _prep
        obs = _prep.preprocess_observation_pytorch(observation, train=False)
        images = list(obs.images.values())
        img_masks = list(obs.image_masks.values())
        lang_tokens = obs.tokenized_prompt
        lang_masks = obs.tokenized_prompt_mask

        bsize = start_proprio.shape[0]
        device = start_proprio.device

        if noise is None:
            noise = torch.randn(bsize, self.action_horizon, self.action_dim, device=device)

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        prefix_att_4d = prefix_att_2d[:, None, :, :]
        prefix_att_4d = torch.where(prefix_att_4d, 0.0, -2.3819763e38)

        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if prefix_embs.dtype != model_dtype:
            prefix_embs = prefix_embs.to(model_dtype)
        if prefix_att_4d.dtype != model_dtype:
            prefix_att_4d = prefix_att_4d.to(model_dtype)

        _, past_kv = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        t_val = torch.tensor(1.0, device=device)

        while t_val >= -dt / 2:
            expanded_t = t_val.expand(bsize)

            suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(
                start_proprio, end_proprio, x_t, expanded_t, duration,
            )

            suffix_len = suffix_pad.shape[1]
            prefix_len = prefix_pad.shape[1]

            prefix_2d = prefix_pad[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
            full_att = torch.cat([prefix_2d, suffix_att_2d], dim=2)
            full_att_4d = full_att[:, None, :, :]
            full_att_4d = torch.where(full_att_4d, 0.0, -2.3819763e38)
            if full_att_4d.dtype != model_dtype:
                full_att_4d = full_att_4d.to(model_dtype)

            prefix_offsets = torch.sum(prefix_pad, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1

            if suffix_embs.dtype != model_dtype:
                suffix_embs = suffix_embs.to(model_dtype)

            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=position_ids,
                past_key_values=past_kv,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = outputs[1][:, -self.action_horizon:].float()
            v_t = self.action_out_proj(suffix_out)

            x_t = x_t + dt * v_t
            t_val = t_val + dt

        return x_t

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_pretrained_weights(model, weight_path, device):
        """Load pretrained Pi0.5 weights with shape-mismatch tolerance.

        Handles the time_mlp_in resize from [W,W] to [2W,W] and any other
        shape differences gracefully.
        """
        import safetensors.torch

        state_dict = safetensors.torch.load_file(weight_path, device=str(device))
        own_state = model.state_dict()
        loaded, skipped, partial = 0, 0, 0
        for name, param in state_dict.items():
            if name not in own_state:
                skipped += 1
                continue
            if own_state[name].shape != param.shape:
                if param.dim() == own_state[name].dim() and all(
                    ps <= ns for ps, ns in zip(param.shape, own_state[name].shape)
                ):
                    slices = tuple(slice(0, s) for s in param.shape)
                    own_state[name][slices].copy_(param)
                    logger.info(
                        f"  Partial load {name}: pretrained {tuple(param.shape)} "
                        f"-> new {tuple(own_state[name].shape)} (preserved pretrained region)"
                    )
                    partial += 1
                else:
                    logger.info(f"  Skipping {name}: shape {param.shape} != {own_state[name].shape}")
                    skipped += 1
                continue
            own_state[name].copy_(param)
            loaded += 1
        logger.info(f"Loaded {loaded} weight tensors, {partial} partial, skipped {skipped}")
