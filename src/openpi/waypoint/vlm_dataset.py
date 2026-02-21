"""VLM waypoint prediction dataset.

Reads waypoint-filtered RLDS data to produce autoregressive training samples
for the VLM waypoint predictor.

Each sample contains tokenized: images + "Task: instruction, State: s;\nAction: <wp>p...<dur>d...|"
"""

import gc
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from openpi.waypoint.normalize import NormalizationHelper, extract_proprio_from_obs, pad_to_dim
from openpi.waypoint.robot_config import RobotConfig
from openpi.waypoint.tokenizer import WaypointTokenizer

logger = logging.getLogger(__name__)


class WaypointVLMDataset(IterableDataset):
    """RLDS-based VLM waypoint training dataset.

    Reads waypoint-filtered RLDS (where each step IS a waypoint, with
    `waypoint_duration` and `is_waypoint_end` fields) and produces
    tokenized AR training samples.
    """

    def __init__(
        self,
        wp_rlds_dir: str,
        robot_config: RobotConfig,
        dataset_statistics: dict[str, Any],
        wp_tokenizer: WaypointTokenizer,
        norm_type: str = "q99",
        num_waypoints: int = 7,
        stride: int = 1,
        image_size: tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 5000,
    ):
        super().__init__()
        self.wp_rlds_dir = wp_rlds_dir
        self.robot_config = robot_config
        self.wp_tokenizer = wp_tokenizer
        self.norm_type = norm_type
        self.num_waypoints = num_waypoints
        self.stride = stride
        self.image_size = image_size
        self.shuffle_buffer_size = shuffle_buffer_size

        self.norm_helper = NormalizationHelper(dataset_statistics, norm_type)

        logger.info(
            f"WaypointVLMDataset: dir={wp_rlds_dir}, M={num_waypoints}, stride={stride}, "
            f"robot={robot_config.robot_type}"
        )

    def _raw_sample_iter(self):
        """Yield tokenized samples from waypoint-filtered RLDS."""
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
        M = self.num_waypoints

        while True:
            builder = tfds.builder_from_directory(self.wp_rlds_dir)
            dataset = builder.as_dataset(split="train")

            for ep_idx, episode in enumerate(dataset):
                if ep_idx % world_size != rank:
                    continue

                steps = list(episode["steps"])
                num_steps = len(steps)
                if num_steps < 2:
                    continue

                all_proprios_raw = []
                all_durations = []
                all_is_end = []
                for s in steps:
                    proprio_raw = extract_proprio_from_obs(
                        {k: s["observation"][k] for k in s["observation"]},
                        rc.state_obs_keys,
                    )
                    all_proprios_raw.append(proprio_raw)
                    all_durations.append(int(s["waypoint_duration"].numpy()))
                    all_is_end.append(bool(s.get("is_waypoint_end", s["is_last"]).numpy()))

                all_proprios_norm = np.stack([
                    self.norm_helper.normalize_proprio(p) for p in all_proprios_raw
                ])

                instruction = steps[0]["language_instruction"].numpy()
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")

                for start_idx in range(0, num_steps, self.stride):
                    end_idx = min(start_idx + M, num_steps)
                    actual_wps = end_idx - start_idx
                    if actual_wps < 1:
                        continue

                    wp_proprios = np.zeros((M, rc.actual_proprio_dim), dtype=np.float32)
                    wp_durations = np.zeros(M, dtype=np.int32)
                    wp_pad_mask_proprio = np.ones(M, dtype=bool)
                    wp_pad_mask_duration = np.ones(M, dtype=bool)

                    for j in range(actual_wps):
                        wp_idx = start_idx + j
                        wp_proprios[j] = all_proprios_norm[wp_idx]
                        wp_pad_mask_proprio[j] = False

                        if j > 0:
                            wp_durations[j - 1] = all_durations[wp_idx - 1 + start_idx] if (wp_idx - 1 + start_idx) < len(all_durations) else 0
                            wp_pad_mask_duration[j - 1] = False

                    if actual_wps >= 1:
                        if start_idx + actual_wps - 1 < len(all_durations):
                            wp_durations[actual_wps - 1] = all_durations[start_idx + actual_wps - 1]
                            wp_pad_mask_duration[actual_wps - 1] = False

                    if all_is_end[min(start_idx + actual_wps - 1, num_steps - 1)]:
                        if actual_wps < M:
                            wp_durations[actual_wps] = 0
                            wp_pad_mask_duration[actual_wps] = False

                    wp_proprios_padded = np.zeros((M, self.wp_tokenizer.proprio_dim), dtype=np.float32)
                    wp_proprios_padded[:, :rc.actual_proprio_dim] = wp_proprios

                    current_proprio_raw = all_proprios_raw[start_idx]
                    current_proprio_norm = self.norm_helper.normalize_proprio(current_proprio_raw)
                    state_padded = np.zeros(self.wp_tokenizer.proprio_dim, dtype=np.float32)
                    state_padded[:len(current_proprio_norm)] = current_proprio_norm

                    images = {}
                    image_masks = {}
                    for view, rlds_key in rc.camera_rlds_keys.items():
                        model_key = rc.camera_model_keys[view]
                        img_data = steps[start_idx]["observation"][rlds_key].numpy()
                        img = Image.fromarray(img_data)
                        if img.size != self.image_size:
                            img = img.resize(self.image_size, Image.BILINEAR)
                        images[model_key] = np.array(img, dtype=np.uint8)
                        image_masks[model_key] = True

                    for model_key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
                        if model_key not in images:
                            images[model_key] = np.zeros((*self.image_size, 3), dtype=np.uint8)
                            image_masks[model_key] = False

                    tokens, token_mask, ar_mask, loss_mask = self.wp_tokenizer.tokenize(
                        prompt=instruction,
                        state=state_padded,
                        wp_proprios=wp_proprios_padded,
                        wp_durations=wp_durations,
                        wp_pad_mask_proprio=wp_pad_mask_proprio,
                        wp_pad_mask_duration=wp_pad_mask_duration,
                    )

                    yield {
                        "images": images,
                        "image_masks": image_masks,
                        "tokens": tokens,
                        "token_mask": token_mask,
                        "ar_mask": ar_mask,
                        "loss_mask": loss_mask,
                    }

                del steps, all_proprios_raw, all_proprios_norm
                gc.collect()

    def __iter__(self):
        buffer = []
        for sample in self._raw_sample_iter():
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer_size:
                idx = np.random.randint(len(buffer))
                yield buffer.pop(idx)


class WaypointVLMCollator:
    """Collates VLM samples into batched tensors."""

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)

        all_images = {}
        all_image_masks = {}
        for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            imgs = np.stack([s["images"][key] for s in batch])  # (B, H, W, C)
            imgs = imgs.astype(np.float32) / 127.5 - 1.0
            imgs = imgs.transpose(0, 3, 1, 2)  # (B, C, H, W)
            all_images[key] = torch.from_numpy(imgs)
            masks = [s["image_masks"].get(key, False) for s in batch]
            all_image_masks[key] = torch.tensor(masks, dtype=torch.bool)

        tokens = torch.from_numpy(np.stack([s["tokens"] for s in batch]))
        token_mask = torch.from_numpy(np.stack([s["token_mask"] for s in batch]))
        ar_mask = torch.from_numpy(np.stack([s["ar_mask"] for s in batch]))
        loss_mask = torch.from_numpy(np.stack([s["loss_mask"] for s in batch]))

        return {
            "images": all_images,
            "image_masks": all_image_masks,
            "tokens": tokens,
            "token_mask": token_mask,
            "ar_mask": ar_mask,
            "loss_mask": loss_mask,
        }
