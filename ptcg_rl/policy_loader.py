from __future__ import annotations

import os

from ptcg_rl.numpy_policy import NumpyPolicy


def load_policy(path: str, *, device: str = "cpu"):
    """Load either legacy NumPy BC checkpoints or v14 torch sequence checkpoints."""
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix in {".pt", ".pth"}:
        from ptcg_rl.seq.torch_policy import TorchSequencePolicy

        return TorchSequencePolicy.load(path, device=device)
    return NumpyPolicy.load(path)
