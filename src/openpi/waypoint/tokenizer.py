"""Waypoint tokenizer for VLM autoregressive waypoint prediction.

Maps continuous proprio values, discrete duration values, and binary gripper
state to PaliGemma vocabulary token IDs, following the same convention as
Pi0-FAST.

Token ID layout (mapped into PaliGemma vocab tail):
  - Proprio bins 0-299:  vocab_size - 1 - SKIP - i        (i=0..299)
  - Duration 0-33:       vocab_size - 1 - SKIP - 300 - d  (d=0..33)
  - <wp> delimiter:      vocab_size - 1 - SKIP - 300 - 34
  - <dur> delimiter:     vocab_size - 1 - SKIP - 300 - 35
  - <grip_open>:         vocab_size - 1 - SKIP - 300 - 36
  - <grip_close>:        vocab_size - 1 - SKIP - 300 - 37
"""

import logging

import numpy as np
import sentencepiece

import openpi.shared.download as download

logger = logging.getLogger(__name__)

PALIGEMMA_VOCAB_SIZE = 257152
SKIP_TOKENS = 128  # Last 128 tokens reserved by PaliGemma (same as FAST)

PROPRIO_N_BINS = 300
DURATION_MAX = 33
DURATION_N_BINS = DURATION_MAX + 1  # 0..33 inclusive

# Token ID offsets from vocab_size - 1 - SKIP_TOKENS
_PROPRIO_BASE = PALIGEMMA_VOCAB_SIZE - 1 - SKIP_TOKENS  # 257023
_DURATION_BASE = _PROPRIO_BASE - PROPRIO_N_BINS  # 256723
_WP_TOKEN_ID = _DURATION_BASE - DURATION_N_BINS  # 256689
_DUR_TOKEN_ID = _WP_TOKEN_ID - 1  # 256688
_GRIP_OPEN_ID = _DUR_TOKEN_ID - 1  # 256687
_GRIP_CLOSE_ID = _GRIP_OPEN_ID - 1  # 256686

# Total dedicated tokens: 300 + 34 + 2 + 2 = 338
# Token ID range: 256686 .. 257023


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
        token_ids = np.asarray(token_ids)
        bin_indices = _PROPRIO_BASE - token_ids
        out_of_range = (bin_indices < 0) | (bin_indices >= self.n_bins)
        if np.any(out_of_range):
            bad_ids = token_ids[out_of_range].tolist()
            logger.warning(
                f"ProprioTokenizer.decode: {int(np.sum(out_of_range))}/{token_ids.size} "
                f"token(s) outside proprio range [{_PROPRIO_BASE - self.n_bins + 1}, "
                f"{_PROPRIO_BASE}]: {bad_ids[:5]}{'...' if len(bad_ids) > 5 else ''}"
            )
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return self.bin_centers[bin_indices].astype(np.float32)


class WaypointTokenizer:
    """Tokenizes waypoint sequences for VLM autoregressive training.

    Combines ProprioTokenizer with duration tokens, gripper tokens, and
    structural delimiters.  Interfaces with PaliGemma's sentencepiece
    tokenizer for text prefix.

    Token layout per waypoint (use_gripper_token=True, proprio_dim=6):
        <wp> p1 p2 p3 p4 p5 p6 G <dur> d
        pos:  0  1  2  3  4  5  6 7   8  9
        G ∈ {grip_open, grip_close}
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        proprio_dim: int = 6,
        num_waypoints: int = 7,
        max_token_len: int = 256,
        use_gripper_token: bool = True,
    ):
        self.proprio_dim = proprio_dim  # continuous dims only (excludes gripper)
        self.num_waypoints = num_waypoints
        self.max_token_len = max_token_len
        self.use_gripper_token = use_gripper_token

        self.proprio_tokenizer = ProprioTokenizer()

        self.wp_token_id = _WP_TOKEN_ID
        self.dur_token_id = _DUR_TOKEN_ID
        self.grip_open_id = _GRIP_OPEN_ID
        self.grip_close_id = _GRIP_CLOSE_ID

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._pg_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    @property
    def tokens_per_waypoint(self) -> int:
        # <wp> + proprio + [grip] + <dur> + duration
        grip = 1 if self.use_gripper_token else 0
        return 1 + self.proprio_dim + grip + 1 + 1

    @property
    def max_waypoint_tokens(self) -> int:
        return self.num_waypoints * self.tokens_per_waypoint

    # --- Gripper token position (0-indexed within a waypoint block) ---

    @property
    def gripper_pos_in_wp(self) -> int:
        """Position of the gripper token within a waypoint block."""
        return 1 + self.proprio_dim  # right after the last proprio token

    @property
    def dur_delimiter_pos_in_wp(self) -> int:
        """Position of the <dur> delimiter within a waypoint block."""
        return self.gripper_pos_in_wp + (1 if self.use_gripper_token else 0)

    @property
    def duration_pos_in_wp(self) -> int:
        """Position of the duration value token within a waypoint block."""
        return self.dur_delimiter_pos_in_wp + 1

    def encode_duration(self, duration: int) -> int:
        d = int(np.clip(duration, 0, DURATION_MAX))
        return _DURATION_BASE - d

    def decode_duration(self, token_id: int) -> int:
        dur = int(_DURATION_BASE - token_id)
        if dur < 0 or dur > DURATION_MAX:
            logger.warning(
                f"decode_duration: token_id={token_id} decoded to duration={dur} "
                f"(valid: 0-{DURATION_MAX}), token outside duration range "
                f"[{_DURATION_BASE - DURATION_N_BINS + 1}, {_DURATION_BASE}]"
            )
        return dur

    def encode_gripper(self, gripper_open: bool) -> int:
        return _GRIP_OPEN_ID if gripper_open else _GRIP_CLOSE_ID

    def decode_gripper(self, token_id: int) -> float:
        """Decode gripper token to float: 1.0=open, 0.0=close."""
        if token_id == _GRIP_OPEN_ID:
            return 1.0
        elif token_id == _GRIP_CLOSE_ID:
            return 0.0
        else:
            logger.warning(f"decode_gripper: unexpected token_id={token_id}, defaulting to 0.0 (close)")
            return 0.0

    def is_proprio_token(self, token_id: int) -> bool:
        return (_PROPRIO_BASE - PROPRIO_N_BINS + 1) <= token_id <= _PROPRIO_BASE

    def is_duration_token(self, token_id: int) -> bool:
        return (_DURATION_BASE - DURATION_N_BINS + 1) <= token_id <= _DURATION_BASE

    def is_gripper_token(self, token_id: int) -> bool:
        return token_id in (_GRIP_OPEN_ID, _GRIP_CLOSE_ID)

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray,
        wp_proprios: np.ndarray | None,
        wp_durations: np.ndarray | None,
        current_gripper: int = 0,
        wp_grippers: np.ndarray | None = None,
        wp_pad_mask_proprio: np.ndarray | None = None,
        wp_pad_mask_duration: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a full VLM waypoint training sample.

        Args:
            prompt: Language instruction text.
            state: Current proprio state, shape (proprio_dim,) — continuous dims
                   only (no gripper), normalized to [-1,1]. Discretized into prefix.
            wp_proprios: Waypoint proprio values, shape (M, proprio_dim), continuous
                         dims only, normalized to [-1,1].  None during inference.
            wp_durations: Waypoint durations, shape (M,), integers 0..33.
                          None during inference.
            current_gripper: 0=close, 1=open.  Encoded in prefix text.
            wp_grippers: (M,) int/float, 0=close, 1=open per waypoint.
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
            bins=np.linspace(-1, 1, PROPRIO_N_BINS + 1)[:-1],
        ) - 1
        state_str = " ".join(map(str, discretized_state.astype(int)))

        if self.use_gripper_token:
            grip_str = "open" if current_gripper else "closed"
            prefix = f"Task: {cleaned_text}, State: {state_str}, Gripper: {grip_str};\n"
        else:
            prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._pg_tokenizer.encode(prefix, add_bos=True)

        if wp_proprios is not None:
            postfix_tokens, postfix_loss_mask = self._encode_waypoint_postfix(
                wp_proprios, wp_durations, wp_grippers,
                wp_pad_mask_proprio, wp_pad_mask_duration,
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
        wp_grippers: np.ndarray | None,
        wp_pad_mask_proprio: np.ndarray | None,
        wp_pad_mask_duration: np.ndarray | None,
    ) -> tuple[list[int], list[bool]]:
        """Encode M waypoints into token IDs + per-token loss mask.

        Per-waypoint layout:
          <wp>  p1..pN  [G]  <dur>  d
        where G is the optional gripper token (grip_open / grip_close).
        """
        M = len(wp_proprios)
        tokens = []
        loss_mask = []

        for i in range(M):
            is_pad_proprio = wp_pad_mask_proprio is not None and wp_pad_mask_proprio[i]
            is_pad_duration = wp_pad_mask_duration is not None and wp_pad_mask_duration[i]

            # <wp> delimiter (forced, no loss)
            tokens.append(self.wp_token_id)
            loss_mask.append(False)

            # Continuous proprio tokens
            proprio_tids = self.proprio_tokenizer.encode(wp_proprios[i])
            for tid in proprio_tids.flatten():
                tokens.append(int(tid))
                loss_mask.append(not is_pad_proprio)

            # Gripper token
            if self.use_gripper_token:
                if wp_grippers is not None:
                    grip_tid = self.encode_gripper(bool(wp_grippers[i]))
                else:
                    grip_tid = self.grip_close_id  # default for padding
                tokens.append(grip_tid)
                # Gripper shares pad mask with proprio
                loss_mask.append(not is_pad_proprio)

            # <dur> delimiter (forced, no loss)
            tokens.append(self.dur_token_id)
            loss_mask.append(False)

            # Duration token
            dur_tid = self.encode_duration(int(wp_durations[i]))
            tokens.append(dur_tid)
            loss_mask.append(not is_pad_duration)

        return tokens, loss_mask

    def decode_waypoints(self, token_ids: list[int] | np.ndarray) -> list[tuple[np.ndarray, int]]:
        """Decode generated token sequence into list of (proprio, duration) tuples.

        Returns:
            List of (proprio, duration) tuples.
            If use_gripper_token: proprio is (proprio_dim+1,) with gripper
            value (1.0=open, 0.0=close) appended as the last element.
            Otherwise: proprio is (proprio_dim,).
        """
        token_ids = list(token_ids)
        waypoints = []
        tpw = self.tokens_per_waypoint

        for start in range(0, len(token_ids), tpw):
            block = token_ids[start : start + tpw]
            if len(block) < tpw:
                break

            # Decode continuous proprio (positions 1..proprio_dim)
            proprio_tids = np.array(block[1 : 1 + self.proprio_dim])
            proprio_values = self.proprio_tokenizer.decode(proprio_tids)

            # Decode gripper → append to proprio as last dim
            if self.use_gripper_token:
                grip_tid = block[self.gripper_pos_in_wp]
                grip_val = self.decode_gripper(grip_tid)
                proprio_values = np.append(proprio_values, grip_val)

            # Decode duration
            dur_tid = block[self.duration_pos_in_wp]
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
