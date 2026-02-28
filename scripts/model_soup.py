"""Model Soup: average weights of multiple safetensors checkpoints."""

import argparse
import os
import shutil
import safetensors.torch
import torch


def uniform_soup(ckpt_paths: list[str], output_path: str):
    print(f"Loading {len(ckpt_paths)} checkpoints for uniform soup...")

    state_dicts = []
    for p in ckpt_paths:
        sf_path = os.path.join(p, "model.safetensors")
        print(f"  Loading {sf_path}")
        sd = safetensors.torch.load_file(sf_path, device="cpu")
        state_dicts.append(sd)

    keys = list(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        assert set(sd.keys()) == set(keys), (
            f"Key mismatch between checkpoint 0 and {i}"
        )

    n = len(state_dicts)
    print(f"Averaging {len(keys)} tensors across {n} checkpoints...")
    averaged = {}
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in state_dicts])
        averaged[k] = (stacked.sum(dim=0) / n).to(state_dicts[0][k].dtype)

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "model.safetensors")
    print(f"Saving averaged model to {out_file}")
    safetensors.torch.save_file(averaged, out_file)

    meta_src = os.path.join(ckpt_paths[0], "metadata.pt")
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, os.path.join(output_path, "metadata.pt"))

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Soup: uniform weight averaging")
    parser.add_argument(
        "checkpoints", nargs="+",
        help="Paths to checkpoint directories containing model.safetensors",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for the averaged checkpoint",
    )
    args = parser.parse_args()
    uniform_soup(args.checkpoints, args.output)
