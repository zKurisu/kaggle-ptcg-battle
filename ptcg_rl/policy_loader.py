from __future__ import annotations

import os

from ptcg_rl.numpy_policy import NumpyPolicy


def load_policy(path: str, *, device: str = "cpu"):
    """Load legacy NumPy BC checkpoints or torch sequence/plan checkpoints."""
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix in {".pt", ".pth"}:
        import torch

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        cfg = dict(ckpt.get("config") or ckpt.get("model_config") or {})
        feature = str(cfg.get("feature_version", ""))
        if feature.startswith("v15_") or ("model" in ckpt and "config" in ckpt):
            from ptcg_rl.v15.torch_policy import TorchV15Policy

            return TorchV15Policy.load(path, device=device)
        from ptcg_rl.seq.torch_policy import TorchSequencePolicy

        return TorchSequencePolicy.load(path, device=device)
    return NumpyPolicy.load(path)
