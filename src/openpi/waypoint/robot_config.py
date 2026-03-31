"""Robot configuration registry for waypoint VLA.

Defines per-robot configurations for cameras, action/proprio dimensions,
gripper normalization, and RLDS observation key mappings.
Supports LIBERO (single-arm Franka Panda) and Galaxea R1 Lite (dual-arm).
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Gripper normalization functions
# ---------------------------------------------------------------------------

def normalize_gripper_libero(actions: np.ndarray) -> np.ndarray:
    """LIBERO gripper: raw [-1,1] -> clip [0,1] -> invert (1-x).

    Result: 0=close, 1=open  (matches flow matching training convention).
    """
    actions = actions.copy()
    gripper = actions[:, -1]
    gripper = np.clip(gripper, 0, 1)
    actions[:, -1] = 1.0 - gripper
    return actions


def normalize_gripper_calvin(actions: np.ndarray) -> np.ndarray:
    """CALVIN gripper: raw [-1,+1] (+1=open, -1=close) -> clip [0,1].

    Result: 0=close, 1=open (matches flow matching convention).
    No inversion needed — CALVIN's +1=open already aligns with convention.
    """
    actions = actions.copy()
    gripper = actions[:, -1]
    gripper = np.clip(gripper, 0, 1)
    actions[:, -1] = gripper
    return actions


def normalize_gripper_r1_lite(actions: np.ndarray) -> np.ndarray:
    """R1 Lite gripper: raw [0,100] -> [0,1] for both arms."""
    QPOS_DIM = 6
    actions = actions.copy()
    actions[:, QPOS_DIM] = np.clip(actions[:, QPOS_DIM] / 100.0, 0, 1)
    actions[:, 2 * QPOS_DIM + 1] = np.clip(actions[:, 2 * QPOS_DIM + 1] / 100.0, 0, 1)
    return actions


# ---------------------------------------------------------------------------
# Camera key mappings: canonical view name -> RLDS observation key
# ---------------------------------------------------------------------------

CAMERA_KEY_MAPS: dict[str, dict[str, str]] = {
    "galaxea_r1_lite": {
        "head": "image_camera_head",
        "wrist_left": "image_camera_wrist_left",
        "wrist_right": "image_camera_wrist_right",
    },
    "libero": {
        "primary": "image",
        "wrist": "wrist_image",
    },
    "calvin": {
        "primary": "rgb_static",
        "wrist": "rgb_gripper",
    },
}

# Mapping from RLDS camera names to openpi model image keys
CAMERA_TO_MODEL_KEY: dict[str, dict[str, str]] = {
    "galaxea_r1_lite": {
        "head": "base_0_rgb",
        "wrist_left": "left_wrist_0_rgb",
        "wrist_right": "right_wrist_0_rgb",
    },
    "libero": {
        "primary": "base_0_rgb",
        "wrist": "left_wrist_0_rgb",
    },
    "calvin": {
        "primary": "base_0_rgb",
        "wrist": "left_wrist_0_rgb",
    },
}


# ---------------------------------------------------------------------------
# Robot action/proprio dimension configs
# ---------------------------------------------------------------------------

ROBOT_ACTION_DIM_CONFIGS: dict[str, dict[str, list[int]]] = {
    "galaxea_r1_lite": {
        "with_left_arm": list(range(0, 7)),
        "with_right_arm": list(range(7, 14)),
        "with_torso": list(range(14, 20)),
        "with_chassis": list(range(20, 26)),
    },
    "libero": {
        "with_arm": list(range(0, 7)),
    },
    "calvin": {
        "with_arm": list(range(0, 7)),
    },
}

# State observation keys in RLDS
ROBOT_STATE_KEYS: dict[str, dict[str, list[str]]] = {
    "galaxea_r1_lite": {
        "with_left_arm": ["joint_position_arm_left", "gripper_state_left"],
        "with_right_arm": ["joint_position_arm_right", "gripper_state_right"],
        "with_torso": ["joint_position_torso"],
        "with_chassis": ["base_velocity"],
    },
    "libero": {
        "with_arm": ["state"],
    },
    "calvin": {
        "with_arm": ["state"],
    },
}


def get_action_dims(robot_type: str, robot_cfg: dict[str, bool]) -> list[int]:
    """Get filtered action dimension indices for a robot configuration."""
    if robot_type not in ROBOT_ACTION_DIM_CONFIGS:
        raise ValueError(f"Unknown robot type: {robot_type}")
    config = ROBOT_ACTION_DIM_CONFIGS[robot_type]
    return [i for key, idxs in config.items() if robot_cfg.get(key, True) for i in idxs]


def get_state_obs_keys(robot_type: str, robot_cfg: dict[str, bool]) -> list[str]:
    """Get state observation keys from RLDS for a robot configuration."""
    if robot_type not in ROBOT_STATE_KEYS:
        raise ValueError(f"Unknown robot type: {robot_type}")
    config = ROBOT_STATE_KEYS[robot_type]
    return [v for key, vs in config.items() if robot_cfg.get(key, True) for v in vs]


# ---------------------------------------------------------------------------
# RobotConfig dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotConfig:
    """Complete configuration for a robot type in waypoint VLA."""
    robot_type: str

    # Actual dimensions before padding
    actual_action_dim: int
    actual_proprio_dim: int  # total raw state dims (e.g. 8 for LIBERO)

    # Continuous proprio dims (excluding gripper) for VLM tokenization
    continuous_proprio_dim: int = 6

    # Gripper binarization settings for VLM waypoint tokens
    gripper_dim_index: int = 6  # index in raw state for gripper binarization
    gripper_threshold: float = 0.02  # > threshold = open

    # Indices into the raw RLDS action array
    action_dim_indices: list[int] = field(default_factory=list)

    # Observation keys for extracting proprio from RLDS
    state_obs_keys: list[str] = field(default_factory=list)

    # Camera views and their RLDS keys
    camera_views: list[str] = field(default_factory=list)
    camera_rlds_keys: dict[str, str] = field(default_factory=dict)
    camera_model_keys: dict[str, str] = field(default_factory=dict)

    # Gripper normalization function (for action processing)
    normalize_gripper: Callable[[np.ndarray], np.ndarray] = field(default=lambda x: x)

    # Normalization mask for actions (False = skip normalization, e.g. gripper)
    action_norm_mask: np.ndarray | None = None

    def binarize_gripper(self, state: np.ndarray) -> int:
        """Binarize gripper from raw state: 1=open, 0=close."""
        return int(state[self.gripper_dim_index] > self.gripper_threshold)

    def split_proprio(self, state: np.ndarray) -> tuple[np.ndarray, int]:
        """Split raw state into continuous proprio and binary gripper.

        Returns:
            (continuous_proprio, gripper_binary): continuous is the first
            ``continuous_proprio_dim`` dimensions, gripper is 0 or 1.
        """
        continuous = state[:self.continuous_proprio_dim].copy()
        gripper = self.binarize_gripper(state)
        return continuous, gripper


def make_libero_config() -> RobotConfig:
    action_dims = list(range(7))
    # Gripper (dim 6) is already in [0,1] after normalize_gripper_libero → skip normalization
    norm_mask = np.array([True] * 6 + [False], dtype=bool)
    return RobotConfig(
        robot_type="libero",
        actual_action_dim=7,
        actual_proprio_dim=8,
        continuous_proprio_dim=6,  # dims 0-5 (EEF xyz + rot)
        gripper_dim_index=6,  # grip_qpos_L
        gripper_threshold=0.02,  # > 0.02 → open
        action_dim_indices=action_dims,
        state_obs_keys=["state"],
        camera_views=["primary", "wrist"],
        camera_rlds_keys=CAMERA_KEY_MAPS["libero"],
        camera_model_keys=CAMERA_TO_MODEL_KEY["libero"],
        normalize_gripper=normalize_gripper_libero,
        action_norm_mask=norm_mask,
    )


def make_calvin_config() -> RobotConfig:
    action_dims = list(range(7))
    # Gripper (dim 6) is already in [0,1] after normalize_gripper_calvin → skip normalization
    norm_mask = np.array([True] * 6 + [False], dtype=bool)
    return RobotConfig(
        robot_type="calvin",
        actual_action_dim=7,
        actual_proprio_dim=15,  # CALVIN robot_obs is 15D
        continuous_proprio_dim=6,  # dims 0-5 (TCP pos + euler)
        gripper_dim_index=14,  # binary gripper state (-1=close, +1=open)
        gripper_threshold=0.0,  # > 0 → open
        action_dim_indices=action_dims,
        state_obs_keys=["state"],
        camera_views=["primary", "wrist"],
        camera_rlds_keys=CAMERA_KEY_MAPS["calvin"],
        camera_model_keys=CAMERA_TO_MODEL_KEY["calvin"],
        normalize_gripper=normalize_gripper_calvin,
        action_norm_mask=norm_mask,
    )


def make_r1_lite_config(
    with_left_arm: bool = True,
    with_right_arm: bool = True,
    with_torso: bool = False,
    with_chassis: bool = False,
) -> RobotConfig:
    robot_cfg = {
        "with_left_arm": with_left_arm,
        "with_right_arm": with_right_arm,
        "with_torso": with_torso,
        "with_chassis": with_chassis,
    }
    action_dims = get_action_dims("galaxea_r1_lite", robot_cfg)
    state_keys = get_state_obs_keys("galaxea_r1_lite", robot_cfg)

    actual_action_dim = len(action_dims)
    gripper_indices = []
    if with_left_arm:
        gripper_indices.append(6)
    if with_right_arm:
        gripper_indices.append(13 if with_left_arm else 6)
    norm_mask = np.ones(actual_action_dim, dtype=bool)
    for gi in gripper_indices:
        if gi < actual_action_dim:
            norm_mask[gi] = False

    return RobotConfig(
        robot_type="galaxea_r1_lite",
        actual_action_dim=actual_action_dim,
        actual_proprio_dim=actual_action_dim,
        action_dim_indices=action_dims,
        state_obs_keys=state_keys,
        camera_views=["head", "wrist_left", "wrist_right"],
        camera_rlds_keys=CAMERA_KEY_MAPS["galaxea_r1_lite"],
        camera_model_keys=CAMERA_TO_MODEL_KEY["galaxea_r1_lite"],
        normalize_gripper=normalize_gripper_r1_lite,
        action_norm_mask=norm_mask,
    )


def get_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    """Factory function to get a robot configuration."""
    if robot_type == "libero":
        return make_libero_config()
    elif robot_type == "calvin":
        return make_calvin_config()
    elif robot_type == "galaxea_r1_lite":
        return make_r1_lite_config(**kwargs)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")
