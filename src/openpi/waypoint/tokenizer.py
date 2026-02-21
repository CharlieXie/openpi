"""Waypoint tokenizer for VLM autoregressive waypoint prediction.

Maps continuous proprio values and discrete duration values to PaliGemma
vocabulary token IDs, following the same convention as Pi0-FAST.

Token ID layout (mapped into PaliGemma vocab tail):
  - Proprio bins 0-255:  vocab_size - 1 - SKIP - i     (i=0..255)
  - Duration 0-33:       vocab_size - 1 - SKIP - 256 - d  (d=0..33)
  - <wp> delimiter:      vocab_size - 1 - SKIP - 256 - 34
  - <dur> delimiter:     vocab_size - 1 - SKIP - 256 - 35
"""

import logging

import numpy as np
import sentencepiece

import openpi.shared.download as download

logger = logging.getLogger(__name__)

PALIGEMMA_VOCAB_SIZE = 257152
SKIP_TOKENS = 128  # Last 128 tokens reserved by PaliGemma (same as FAST)

PROPRIO_N_BINS = 256
DURATION_MAX = 33
DURATION_N_BINS = DURATION_MAX + 1  # 0..33 inclusive

# Token ID offsets from vocab_size - 1 - SKIP_TOKENS
_PROPRIO_BASE = PALIGEMMA_VOCAB_SIZE - 1 - SKIP_TOKENS  # 257023
_DURATION_BASE = _PROPRIO_BASE - PROPRIO_N_BINS  # 256767
_WP_TOKEN_ID = _DURATION_BASE - DURATION_N_BINS  # 256733
_DUR_TOKEN_ID = _WP_TOKEN_ID - 1  # 256732

# Total dedicated tokens: 256 + 34 + 2 = 292
# Token ID range: 256732 .. 257023


class ProprioTokenizer:
    """Discretizes continuous proprio values in [-1,1] into 256-bin tokens."""

    def __init__(self, n_bins: int = PROPRIO_N_BINS, min_val: float = -1.0, max_val: float = 1.0):
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Encode continuous values to token IDs.

        Args:
            values: shape (...,) continuous values in [-1, 1].

        Returns:
            Token IDs array of same shape, dtype int64.
        """
        clipped = np.clip(values, self.min_val, self.max_val)
        bin_indices = np.digitize(clipped, self.bin_edges[1:-1])  # 0..n_bins-1
        token_ids = _PROPRIO_BASE - bin_indices
        return token_ids.astype(np.int64)

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        """Decode token IDs back to continuous bin-center values."""
        bin_indices = _PROPRIO_BASE - np.asarray(token_ids)
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return self.bin_centers[bin_indices].astype(np.float32)


class WaypointTokenizer:
    """Tokenizes waypoint sequences for VLM autoregressive training.

    Combines ProprioTokenizer with duration tokens and structural delimiters.
    Interfaces with PaliGemma's sentencepiece tokenizer for text prefix.
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        proprio_dim: int = 8,
        num_waypoints: int = 7,
        max_token_len: int = 256,
    ):
        self.proprio_dim = proprio_dim
        self.num_waypoints = num_waypoints
        self.max_token_len = max_token_len

        self.proprio_tokenizer = ProprioTokenizer()

        self.wp_token_id = _WP_TOKEN_ID
        self.dur_token_id = _DUR_TOKEN_ID

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._pg_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    @property
    def tokens_per_waypoint(self) -> int:
        return 1 + self.proprio_dim + 1 + 1  # <wp> + proprio + <dur> + duration

    @property
    def max_waypoint_tokens(self) -> int:
        return self.num_waypoints * self.tokens_per_waypoint

    def encode_duration(self, duration: int) -> int:
        d = int(np.clip(duration, 0, DURATION_MAX))
        return _DURATION_BASE - d

    def decode_duration(self, token_id: int) -> int:
        return int(_DURATION_BASE - token_id)

    def is_proprio_token(self, token_id: int) -> bool:
        return (_PROPRIO_BASE - PROPRIO_N_BINS + 1) <= token_id <= _PROPRIO_BASE

    def is_duration_token(self, token_id: int) -> bool:
        return (_DURATION_BASE - DURATION_N_BINS + 1) <= token_id <= _DURATION_BASE

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray,
        wp_proprios: np.ndarray | None,
        wp_durations: np.ndarray | None,
        wp_pad_mask_proprio: np.ndarray | None = None,
        wp_pad_mask_duration: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a full VLM waypoint training sample.

        Args:
            prompt: Language instruction text.
            state: Current proprio state, shape (proprio_dim,). Discretized into prefix.
            wp_proprios: Waypoint proprio values, shape (M, proprio_dim), normalized to [-1,1].
                         None during inference.
            wp_durations: Waypoint durations, shape (M,), integers 0..33.
                          None during inference.
            wp_pad_mask_proprio: (M,) bool, True = this WP is padding (no loss).
            wp_pad_mask_duration: (M,) bool, True = this duration is padding (no loss).

        Returns:
            tokens: (max_token_len,) int array.
            token_mask: (max_token_len,) bool, True = valid (not sequence padding).
            ar_mask: (max_token_len,) int, 0 = bidirectional, 1 = causal.
            loss_mask: (max_token_len,) bool, True = compute CE loss on this token.
        """
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ").lower()

        discretized_state = np.digitize(
            np.clip(state, -1, 1),
            bins=np.linspace(-1, 1, 256 + 1)[:-1],
        ) - 1
        state_str = " ".join(map(str, discretized_state.astype(int)))

        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._pg_tokenizer.encode(prefix, add_bos=True)

        if wp_proprios is not None:
            postfix_tokens, postfix_loss_mask = self._encode_waypoint_postfix(
                wp_proprios, wp_durations, wp_pad_mask_proprio, wp_pad_mask_duration,
            )
        else:
            postfix_tokens = []
            postfix_loss_mask = []

        action_header = self._pg_tokenizer.encode("Action: ")
        action_footer = self._pg_tokenizer.encode("|", add_eos=True)

        full_postfix = action_header + postfix_tokens + action_footer
        full_postfix_loss = [False] * len(action_header) + postfix_loss_mask + [False] * len(action_footer)

        tokens = prefix_tokens + full_postfix
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(full_postfix)
        loss_mask = [False] * len(prefix_tokens) + full_postfix_loss

        tokens_len = len(tokens)
        if tokens_len < self.max_token_len:
            pad_len = self.max_token_len - tokens_len
            tokens = tokens + [0] * pad_len
            token_mask = token_mask + [False] * pad_len
            ar_mask = ar_mask + [0] * pad_len
            loss_mask = loss_mask + [False] * pad_len
        else:
            if tokens_len > self.max_token_len:
                logger.warning(
                    f"Token length ({tokens_len}) exceeds max ({self.max_token_len}), truncating."
                )
            tokens = tokens[: self.max_token_len]
            token_mask = token_mask[: self.max_token_len]
            ar_mask = ar_mask[: self.max_token_len]
            loss_mask = loss_mask[: self.max_token_len]

        return (
            np.asarray(tokens, dtype=np.int64),
            np.asarray(token_mask, dtype=bool),
            np.asarray(ar_mask, dtype=np.int32),
            np.asarray(loss_mask, dtype=bool),
        )

    def _encode_waypoint_postfix(
        self,
        wp_proprios: np.ndarray,
        wp_durations: np.ndarray,
        wp_pad_mask_proprio: np.ndarray | None,
        wp_pad_mask_duration: np.ndarray | None,
    ) -> tuple[list[int], list[bool]]:
        """Encode M waypoints into token IDs + per-token loss mask."""
        M = len(wp_proprios)
        tokens = []
        loss_mask = []

        for i in range(M):
            is_pad_proprio = wp_pad_mask_proprio is not None and wp_pad_mask_proprio[i]
            is_pad_duration = wp_pad_mask_duration is not None and wp_pad_mask_duration[i]

            tokens.append(self.wp_token_id)
            loss_mask.append(False)

            proprio_tids = self.proprio_tokenizer.encode(wp_proprios[i])
            for tid in proprio_tids.flatten():
                tokens.append(int(tid))
                loss_mask.append(not is_pad_proprio)

            tokens.append(self.dur_token_id)
            loss_mask.append(False)

            dur_tid = self.encode_duration(int(wp_durations[i]))
            tokens.append(dur_tid)
            loss_mask.append(not is_pad_duration)

        return tokens, loss_mask

    def decode_waypoints(self, token_ids: list[int] | np.ndarray) -> list[tuple[np.ndarray, int]]:
        """Decode generated token sequence into list of (proprio, duration) tuples."""
        token_ids = list(token_ids)
        waypoints = []
        tpw = self.tokens_per_waypoint

        for start in range(0, len(token_ids), tpw):
            block = token_ids[start : start + tpw]
            if len(block) < tpw:
                break

            proprio_tids = np.array(block[1 : 1 + self.proprio_dim])
            proprio_values = self.proprio_tokenizer.decode(proprio_tids)

            dur_tid = block[1 + self.proprio_dim + 1]
            duration = self.decode_duration(dur_tid)

            waypoints.append((proprio_values, duration))

        return waypoints

    def extract_waypoint_tokens_from_output(self, output_tokens: np.ndarray) -> list[int]:
        """Extract waypoint-relevant tokens from a full generated sequence.

        Looks for the 'Action:' header and '|' footer, returning only the
        waypoint tokens between them.
        """
        decoded = self._pg_tokenizer.decode(output_tokens.tolist())
        if "Action: " not in decoded:
            return []

        action_text = decoded.split("Action: ")[1].split("|")[0]
        wp_token_ids = self._pg_tokenizer.encode(action_text)
        return wp_token_ids
