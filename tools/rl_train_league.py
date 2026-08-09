#!/usr/bin/env python3
"""From-scratch league PPO entry point.

This wrapper intentionally names the non-BC path. Pass a checkpoint through
`--policy-init` only as an architecture/template source and add
`--init-mode random`; do not use BC anchors or reference KL for scratch RL.
"""
from __future__ import annotations

from rl_finetune_vs_pool import main


if __name__ == "__main__":
    main()
