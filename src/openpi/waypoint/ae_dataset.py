"""Action Expert training dataset for waypoint VLA.

Reads original RLDS data + waypoint_indices.json to produce waypoint-pair
training samples for the flow-matching Action Expert.

Each sample: (images, instruction, start_proprio, end_proprio, duration,
              padded_actions[horizon, model_dim], action_pad_mask, action_dim_mask)
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from openpi.waypoint.normalize import (
    NormalizationHelper,
    extract_proprio_from_obs,
    make_dim_mask,
    pad_to_dim,
)
from openpi.waypoint.robot_config import RobotConfig

logger = logging.getLogger(__name__)

MODEL_ACTION_DIM = 32
MODEL_PROPRIO_DIM = 32


class WaypointAEDataset(IterableDataset):
    """RLDS-based Action Expert dataset with global shuffle buffer.

    Iterates over episodes, extracts waypoint pairs, normalizes, pads to
    unified model dimensions, and yields training samples.
    """

    def __init__(
        self,
        original_rlds_dir: str,
        wp_indices_path: str,
        robot_config: RobotConfig,
        dataset_statistics: dict[str, Any],
        norm_type: str = "q99",
        max_duration: int = 32,
        horizon_steps: int = 32,
        model_action_dim: int = MODEL_ACTION_DIM,
        model_proprio_dim: int = MODEL_PROPRIO_DIM,
        shuffle_buffer_size: int = 10000,
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.original_rlds_dir = original_rlds_dir
        self.robot_config = robot_config
        self.norm_type = norm_type
        self.max_duration = max_duration
        self.horizon_steps = horizon_steps
        self.model_action_dim = model_action_dim
        self.model_proprio_dim = model_proprio_dim
        self.shuffle_buffer_size = shuffle_buffer_size
        self.image_size = image_size

        self.norm_helper = NormalizationHelper(dataset_statistics, norm_type)

        self.action_dim_mask = make_dim_mask(robot_config.actual_action_dim, model_action_dim)
        self.proprio_dim_mask = make_dim_mask(robot_config.actual_proprio_dim, model_proprio_dim)

        logger.info(f"Loading waypoint indices from {wp_indices_path}")
        with open(wp_indices_path) as f:
            wp_data = json.load(f)

        self.episode_wp_map: dict[int, list[tuple[int, int, int]]] = {}
        total_pairs = 0
        skipped = 0
        for ep in wp_data["episodes"]:
            wp_indices = ep["waypoint_indices"]
            valid_pairs = []
            for i in range(len(wp_indices) - 1):
                dur = wp_indices[i + 1] - wp_indices[i]
                if dur <= max_duration:
                    valid_pairs.append((wp_indices[i], wp_indices[i + 1], dur))
                else:
                    skipped += 1
            if valid_pairs:
                self.episode_wp_map[ep["src_ep_idx"]] = valid_pairs
                total_pairs += len(valid_pairs)

        self.total_pairs = total_pairs
        logger.info(
            f"WaypointAEDataset: {len(self.episode_wp_map)} episodes, "
            f"{total_pairs} valid pairs, {skipped} skipped (dur>{max_duration}), "
            f"actual_action={robot_config.actual_action_dim}→{model_action_dim}, "
            f"actual_proprio={robot_config.actual_proprio_dim}→{model_proprio_dim}"
        )

    def __len__(self) -> int:
        return self.total_pairs

    def _raw_sample_iter(self):
        """Yield raw samples from RLDS. DDP-aware, repeats indefinitely."""
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        try:
            import torch.distributed as dist
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0
        except Exception:
            world_size, rank = 1, 0

        rc = self.robot_config

        while True:
            builder = tfds.builder_from_directory(self.original_rlds_dir)
            dataset = builder.as_dataset(split="train")

            for ep_idx, episode in enumerate(dataset):
                if ep_idx % world_size != rank:
                    continue

                wp_pairs = self.episode_wp_map.get(ep_idx)
                if wp_pairs is None:
                    continue

                steps = list(episode["steps"])
                if not steps:
                    continue

                all_actions_raw = np.stack([s["action"].numpy() for s in steps])
                all_actions = rc.normalize_gripper(all_actions_raw)
                all_actions = all_actions[:, rc.action_dim_indices]
                all_actions = self.norm_helper.normalize_actions(all_actions)

                instruction = steps[0]["language_instruction"].numpy()
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")

                for w_start, w_end, duration in wp_pairs:
                    if w_end >= len(steps):
                        continue

                    images = {}
                    for view, rlds_key in rc.camera_rlds_keys.items():
                        model_key = rc.camera_model_keys[view]
                        img_data = steps[w_start]["observation"][rlds_key].numpy()
                        img = Image.fromarray(img_data)
                        if img.size != self.image_size:
                            img = img.resize(self.image_size, Image.BILINEAR)
                        images[model_key] = np.array(img, dtype=np.uint8)

                    start_proprio_raw = extract_proprio_from_obs(
                        {k: steps[w_start]["observation"][k] for k in steps[w_start]["observation"]},
                        rc.state_obs_keys,
                    )
                    end_proprio_raw = extract_proprio_from_obs(
                        {k: steps[w_end]["observation"][k] for k in steps[w_end]["observation"]},
                        rc.state_obs_keys,
                    )

                    start_proprio = pad_to_dim(
                        self.norm_helper.normalize_proprio(start_proprio_raw),
                        self.model_proprio_dim,
                    )
                    end_proprio = pad_to_dim(
                        self.norm_helper.normalize_proprio(end_proprio_raw),
                        self.model_proprio_dim,
                    )

                    seg_actions = all_actions[w_start:w_end]
                    actual_len = len(seg_actions)
                    padded_actions = np.zeros(
                        (self.horizon_steps, self.model_action_dim), dtype=np.float32
                    )
                    padded_actions[:actual_len, : rc.actual_action_dim] = seg_actions[:actual_len]

                    action_pad_mask = np.zeros(self.horizon_steps, dtype=bool)
                    action_pad_mask[actual_len:] = True

                    yield {
                        "images": images,
                        "instruction": instruction,
                        "start_proprio": start_proprio.astype(np.float32),
                        "end_proprio": end_proprio.astype(np.float32),
                        "duration": float(duration),
                        "actions": padded_actions.astype(np.float32),
                        "action_pad_mask": action_pad_mask,
                        "action_dim_mask": self.action_dim_mask.copy(),
                        "proprio_dim_mask": self.proprio_dim_mask.copy(),
                    }

                del steps, all_actions_raw, all_actions
                gc.collect()

    def __iter__(self):
        """Yields shuffled samples via a reservoir-style buffer."""
        buffer = []
        for sample in self._raw_sample_iter():
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer_size:
                idx = np.random.randint(len(buffer))
                yield buffer.pop(idx)


class WaypointAECollator:
    """Collates AE samples into batched tensors for the model."""

    def __init__(self, tokenizer_max_len: int = 64):
        self.tokenizer_max_len = tokenizer_max_len

        path_obj = None
        try:
            import sentencepiece
            import openpi.shared.download as download
            path_obj = download.maybe_download(
                "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
            )
            with path_obj.open("rb") as f:
                self._pg_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        except Exception:
            self._pg_tokenizer = None

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)

        all_images = {}
        all_image_masks = {}
        for key in batch[0]["images"]:
            imgs = np.stack([s["images"][key] for s in batch])
            imgs = imgs.astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
            all_images[key] = torch.from_numpy(imgs)
            all_image_masks[key] = torch.ones(B, dtype=torch.bool)

        for model_key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            if model_key not in all_images:
                all_images[model_key] = torch.zeros(B, 224, 224, 3, dtype=torch.float32)
                all_image_masks[model_key] = torch.zeros(B, dtype=torch.bool)

        start_proprio = torch.from_numpy(np.stack([s["start_proprio"] for s in batch]))
        end_proprio = torch.from_numpy(np.stack([s["end_proprio"] for s in batch]))
        actions = torch.from_numpy(np.stack([s["actions"] for s in batch]))
        action_pad_mask = torch.from_numpy(np.stack([s["action_pad_mask"] for s in batch]))
        action_dim_mask = torch.from_numpy(np.stack([s["action_dim_mask"] for s in batch]))
        duration = torch.tensor([s["duration"] for s in batch], dtype=torch.float32)

        prompt_tokens = torch.zeros(B, self.tokenizer_max_len, dtype=torch.long)
        prompt_masks = torch.zeros(B, self.tokenizer_max_len, dtype=torch.bool)

        if self._pg_tokenizer is not None:
            for i, s in enumerate(batch):
                text = f"Task: {s['instruction'].strip().replace('_', ' ').lower()}, \n"
                tids = self._pg_tokenizer.encode(text, add_bos=True)
                length = min(len(tids), self.tokenizer_max_len)
                prompt_tokens[i, :length] = torch.tensor(tids[:length], dtype=torch.long)
                prompt_masks[i, :length] = True

        return {
            "images": all_images,
            "image_masks": all_image_masks,
            "start_proprio": start_proprio,
            "end_proprio": end_proprio,
            "actions": actions,
            "action_pad_mask": action_pad_mask,
            "action_dim_mask": action_dim_mask,
            "duration": duration,
            "prompt_tokens": prompt_tokens,
            "prompt_masks": prompt_masks,
        }
