#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2 import BCCorpus, discover_npz_paths, sequence_nll
from ptcg_rl.deck_plans import CARD_NAMES
from ptcg_rl.model import PolicyValueNet


CONTEXT_IDS = {
    "MAIN": 0,
    "SETUP_ACTIVE": 1,
    "SETUP_BENCH": 2,
    "SWITCH": 3,
    "TO_ACTIVE": 4,
    "TO_BENCH": 5,
    "TO_FIELD": 6,
    "TO_HAND": 7,
    "DISCARD": 8,
    "TO_DECK": 9,
    "TO_DECK_BOTTOM": 10,
    "TO_PRIZE": 11,
    "NOT_MOVE": 12,
    "DAMAGE_COUNTER": 13,
    "DAMAGE_COUNTER_ANY": 14,
    "DAMAGE": 15,
    "REMOVE_DAMAGE_COUNTER": 16,
    "HEAL": 17,
    "EVOLVES_FROM": 18,
    "EVOLVES_TO": 19,
    "DEVOLVE": 20,
    "ATTACH_FROM": 21,
    "ATTACH_TO": 22,
    "DETACH_FROM": 23,
    "LOOK": 24,
    "EFFECT_TARGET": 25,
    "DISCARD_ENERGY_CARD": 26,
    "DISCARD_TOOL_CARD": 27,
    "SWITCH_ENERGY_CARD": 28,
    "DISCARD_CARD_OR_ATTACHED_CARD": 29,
    "DISCARD_ENERGY": 30,
    "TO_HAND_ENERGY": 31,
    "TO_DECK_ENERGY": 32,
    "SWITCH_ENERGY": 33,
    "SKILL_ORDER": 34,
    "ATTACK": 35,
    "DISABLE_ATTACK": 36,
    "EVOLVE": 37,
    "DRAW_COUNT": 38,
    "DAMAGE_COUNTER_COUNT": 39,
    "REMOVE_DAMAGE_COUNTER_COUNT": 40,
    "IS_FIRST": 41,
    "MULLIGAN": 42,
    "ACTIVATE": 43,
}

TYPE_IDS = {
    "NUMBER": 0,
    "YES": 1,
    "NO": 2,
    "CARD": 3,
    "TOOL_CARD": 4,
    "ENERGY_CARD": 5,
    "ENERGY": 6,
    "PLAY": 7,
    "ATTACH": 8,
    "EVOLVE": 9,
    "ABILITY": 10,
    "DISCARD": 11,
    "RETREAT": 12,
    "ATTACK": 13,
    "END": 14,
    "SKILL": 15,
    "SPECIAL_CONDITION": 16,
}

CARD_IDS = {str(name).upper(): int(card_id) for card_id, name in CARD_NAMES.items()}


def _parse_weight_specs(specs: list[str], names: dict[str, int], label: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{label} weight must be NAME=WEIGHT or ID=WEIGHT, got {spec!r}")
        key, value = spec.split("=", 1)
        key = key.strip().upper()
        if key.lstrip("-").isdigit():
            idx = int(key)
        elif key in names:
            idx = names[key]
        else:
            known = ", ".join(sorted(names)[:12])
            raise ValueError(f"unknown {label} {key!r}; use numeric id or one of: {known}, ...")
        out[idx] = float(value)
    return out


def _parse_text_weight_specs(specs: list[str], label: str, *, lower: bool = False) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{label} weight must be NAME=WEIGHT, got {spec!r}")
        key, value = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{label} weight has an empty key in {spec!r}")
        out[key.lower() if lower else key] = float(value)
    return out


def _save_npz(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def _load_npz_init(model: torch.nn.Module, path: str, device: torch.device, *, partial: bool = False) -> tuple[int, list[str]]:
    with np.load(path) as z:
        checkpoint = {
            k: torch.as_tensor(z[k], device=device)
            for k in z.files
        }
    if not partial:
        model.load_state_dict(checkpoint, strict=True)
        return len(checkpoint), []

    current = model.state_dict()
    loaded = {}
    skipped: list[str] = []
    for key, tensor in checkpoint.items():
        if key not in current:
            skipped.append(f"{key}: unexpected")
            continue
        if tuple(tensor.shape) != tuple(current[key].shape):
            skipped.append(f"{key}: {tuple(tensor.shape)} != {tuple(current[key].shape)}")
            continue
        loaded[key] = tensor.to(dtype=current[key].dtype)
    if not loaded:
        raise RuntimeError(f"No compatible tensors found in --init checkpoint: {path}")
    current.update(loaded)
    model.load_state_dict(current, strict=True)
    return len(loaded), skipped


def _checkpoint_feature_dims(path: str) -> tuple[int, int]:
    with np.load(path) as z:
        ec = z["card_emb.weight"].shape[1]
        state_in = z["state_fc1.weight"].shape[1]
        slot_feat_dim = state_in - 5 * ec
        legacy_feat_dim = state_in - 3 * ec
        state_feat_dim = slot_feat_dim if 8 <= slot_feat_dim <= 256 else legacy_feat_dim
        option_context = "context_emb.weight" in z.files
        opt_extra = 0
        if option_context:
            opt_extra = (
                z["context_emb.weight"].shape[1]
                + z["select_type_emb.weight"].shape[1]
                + z["area_emb.weight"].shape[1]
                + z["index_emb.weight"].shape[1]
                + z["inplay_area_emb.weight"].shape[1]
                + z["inplay_index_emb.weight"].shape[1]
            )
        opt_feat_dim = z["opt_fc.weight"].shape[1] - (
            2 * ec
            + z["attack_emb.weight"].shape[1]
            + z["opt_type_emb.weight"].shape[1]
            + opt_extra
        )
    return int(state_feat_dim), int(opt_feat_dim)


def _configure_cuda_memory_limit(device: torch.device, *, gb: float = 0.0, fraction: float = 0.0) -> str:
    if device.type != "cuda":
        return ""
    if gb <= 0 and fraction <= 0:
        return ""
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / (1024 ** 3)
    if gb > 0:
        fraction = float(gb) / max(total_gb, 1e-9)
    fraction = min(max(float(fraction), 0.01), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    limit_gb = total_gb * fraction
    return f"cuda_memory_limit={limit_gb:.2f}GB/{total_gb:.1f}GB fraction={fraction:.3f}"


def _run_epoch(model, corpus, indices, batch_size, device, optimizer=None,
               first_action_weight=1.0, value_weight=0.0,
               set_loss_weight=0.0, set_loss_min_count=2,
               set_loss_negative_weight=0.25):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    steps = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if len(batch_idx) < 2:
            continue
        batch = corpus.collate(batch_idx, device)
        loss = sequence_nll(
            model,
            batch,
            first_action_weight=first_action_weight,
            value_weight=value_weight,
            set_loss_weight=set_loss_weight,
            set_loss_min_count=set_loss_min_count,
            set_loss_negative_weight=set_loss_negative_weight,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1), steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/bc_corpus_banded_v4")
    parser.add_argument("--aux-corpus", action="append", default=[],
                        help="extra corpus root mixed into training, e.g. generated rollout BC; repeatable")
    parser.add_argument("--aux-score-bands", nargs="+", default=["generated"],
                        help="score bands to read from each --aux-corpus")
    parser.add_argument("--aux-archetype", default="",
                        help="archetype name for --aux-corpus; defaults to --archetype")
    parser.add_argument("--aux-repeat", type=int, default=1,
                        help="repeat aux corpus paths N times as a coarse sample multiplier")
    parser.add_argument("--archetype", default="Marnie Grimmsnarl")
    parser.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    parser.add_argument("--deck-sig", action="append", default=[],
                        help="filter to one or more deck signatures; repeatable. Requires freshly extracted corpus metadata.")
    parser.add_argument("--team-name", action="append", default=[],
                        help="filter to one or more exact team names; repeatable. Use with --deck-sig for trajectory specialists.")
    parser.add_argument("--opponent-deck-sig", action="append", default=[],
                        help="filter to decisions from games against one or more opponent deck signatures")
    parser.add_argument("--opponent-archetype", action="append", default=[],
                        help="filter to decisions from games against one or more opponent archetypes")
    parser.add_argument("--opponent-team-name", action="append", default=[],
                        help="filter to decisions from games against one or more exact opponent team names")
    parser.add_argument("--opponent-deck-sig-weight", action="append", default=[],
                        help="repeatable opponent deck signature multiplier without filtering, e.g. 697a82e582d5=2.0")
    parser.add_argument("--opponent-archetype-weight", action="append", default=[],
                        help="repeatable opponent archetype multiplier without filtering, e.g. 'Crustle Wall=2.0'")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-memory-gb", type=float, default=0.0,
                        help="cap this process' CUDA allocator to approximately N GiB; 0 disables")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0,
                        help="cap this process' CUDA allocator to a fraction of the visible GPU; 0 disables")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init", default="",
                        help="optional .npz checkpoint used to initialize the model before training")
    parser.add_argument("--init-partial", action="store_true",
                        help="load only matching tensors from --init; default requires an exact architecture match")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--load-progress-every", type=int, default=200000,
                        help="print corpus indexing progress every N raw decisions; 0 disables")
    parser.add_argument("--winner-only", action="store_true",
                        help="train only on decisions from games this player won; requires outcome metadata")
    parser.add_argument("--win-weight", type=float, default=1.0,
                        help="sample weight multiplier for winning-game decisions when outcome metadata exists")
    parser.add_argument("--loss-weight", type=float, default=1.0,
                        help="sample weight multiplier for losing-game decisions when outcome metadata exists")
    parser.add_argument("--draw-weight", type=float, default=1.0,
                        help="sample weight multiplier for drawn-game decisions when outcome metadata exists")
    parser.add_argument("--legacy-state-pool", action="store_true",
                        help="use old pooled board encoder instead of slot-aware active/bench encoder")
    parser.add_argument("--state-feat-dim", type=int, default=0,
                        help="override state feature width; 0 uses current encoder default")
    parser.add_argument("--opt-feat-dim", type=int, default=0,
                        help="override per-option feature width; 0 uses current encoder default")
    parser.add_argument("--first-action-weight", type=float, default=1.5)
    parser.add_argument("--value-weight", type=float, default=0.0,
                        help="optional auxiliary outcome-value MSE weight; requires outcome metadata")
    parser.add_argument("--set-loss-weight", type=float, default=0.0,
                        help="optional order-free multi-select auxiliary loss weight; 0 disables")
    parser.add_argument("--set-loss-min-count", type=int, default=2,
                        help="minimum labeled action length for set auxiliary loss")
    parser.add_argument("--set-loss-negative-weight", type=float, default=0.25,
                        help="relative weight for unselected options inside the set auxiliary loss")
    parser.add_argument("--option-weight", type=float, default=0.15)
    parser.add_argument("--context-weight", action="append", default=[],
                        help="repeatable context multiplier, e.g. MAIN=2.0 or 21=2.5")
    parser.add_argument("--type-weight", action="append", default=[],
                        help="repeatable true first option type multiplier, e.g. ATTACK=2.5")
    parser.add_argument("--card-weight", action="append", default=[],
                        help="repeatable true first option card multiplier, e.g. 647=2.5")
    parser.add_argument("--multi-select-weight", type=float, default=1.0,
                        help="sample multiplier when the labeled action selects more than one option")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--save", default="checkpoints/bc2_marnie_w2.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    memory_limit_msg = _configure_cuda_memory_limit(
        device,
        gb=args.cuda_memory_gb,
        fraction=args.cuda_memory_fraction,
    )
    paths = discover_npz_paths(args.corpus, args.archetype, args.score_bands)
    base_path_count = len(paths)
    aux_path_count = 0
    aux_details: list[tuple[str, int]] = []
    aux_repeat = max(1, int(args.aux_repeat))
    for aux_root in args.aux_corpus:
        aux_paths = discover_npz_paths(
            aux_root,
            args.aux_archetype or args.archetype,
            args.aux_score_bands,
        )
        aux_path_count += len(aux_paths)
        aux_details.append((aux_root, len(aux_paths)))
        paths.extend(aux_paths * aux_repeat)
    context_weights = _parse_weight_specs(args.context_weight, CONTEXT_IDS, "context")
    type_weights = _parse_weight_specs(args.type_weight, TYPE_IDS, "type")
    card_weights = _parse_weight_specs(args.card_weight, CARD_IDS, "card")
    opponent_deck_sig_weights = _parse_text_weight_specs(
        args.opponent_deck_sig_weight,
        "opponent deck sig",
    )
    opponent_archetype_weights = _parse_text_weight_specs(
        args.opponent_archetype_weight,
        "opponent archetype",
        lower=True,
    )
    inferred_from_init = False
    state_feat_dim = int(args.state_feat_dim) if args.state_feat_dim else None
    opt_feat_dim = int(args.opt_feat_dim) if args.opt_feat_dim else None
    if args.init and not args.init_partial and (state_feat_dim is None or opt_feat_dim is None):
        init_state_dim, init_opt_dim = _checkpoint_feature_dims(args.init)
        if state_feat_dim is None:
            state_feat_dim = init_state_dim
            inferred_from_init = True
        if opt_feat_dim is None:
            opt_feat_dim = init_opt_dim
            inferred_from_init = True
    corpus = BCCorpus(
        paths,
        include_empty=args.include_empty,
        option_weight=args.option_weight,
        **({"state_feat_dim": state_feat_dim} if state_feat_dim is not None else {}),
        **({"opt_feat_dim": opt_feat_dim} if opt_feat_dim is not None else {}),
        deck_sigs=args.deck_sig,
        team_names=args.team_name,
        opponent_deck_sigs=args.opponent_deck_sig,
        opponent_archetypes=args.opponent_archetype,
        opponent_team_names=args.opponent_team_name,
        opponent_deck_sig_weights=opponent_deck_sig_weights,
        opponent_archetype_weights=opponent_archetype_weights,
        winner_only=args.winner_only,
        win_weight=args.win_weight,
        loss_weight=args.loss_weight,
        draw_weight=args.draw_weight,
        context_weights=context_weights,
        type_weights=type_weights,
        card_weights=card_weights,
        multi_select_weight=args.multi_select_weight,
        load_progress_every=args.load_progress_every,
    )
    if corpus.stats["kept"] <= 0:
        raise RuntimeError(
            "No training samples kept after filters. Check --score-bands, --deck-sig, "
            "--opponent-*, --winner-only, and corpus path before training."
        )
    train_idx, val_idx = corpus.split_indices(args.val_fraction, args.seed)
    if len(train_idx) < 2 or len(val_idx) < 1:
        raise RuntimeError(
            f"Not enough samples after split: train={len(train_idx)} val={len(val_idx)}. "
            "Relax filters or lower --val-fraction."
        )

    model_kwargs = {}
    if state_feat_dim is not None:
        model_kwargs["state_feat_dim"] = state_feat_dim
    if opt_feat_dim is not None:
        model_kwargs["opt_feat_dim"] = opt_feat_dim
    model = PolicyValueNet(width=args.width, slot_state=not args.legacy_state_pool, **model_kwargs).to(device)
    if args.init:
        loaded, skipped = _load_npz_init(model, args.init, device, partial=args.init_partial)
        msg = f"Init: loaded={loaded} path={args.init}"
        if skipped:
            msg += f" skipped={len(skipped)}"
        print(msg, flush=True)
        if skipped:
            print(f"Init skipped examples: {skipped[:8]}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_batches = (len(train_idx) + args.batch_size - 1) // args.batch_size
    total_steps = max(args.epochs * train_batches, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.03)
    params = sum(p.numel() for p in model.parameters())

    print(
        f"BC2: {args.archetype} {args.score_bands} device={device} "
        f"{memory_limit_msg + ' ' if memory_limit_msg else ''}"
        f"width={args.width} slot_state={not args.legacy_state_pool} "
        f"state_feat_dim={state_feat_dim or 'default'} opt_feat_dim={opt_feat_dim or 'default'} "
        f"{'feature_dims_from_init ' if inferred_from_init else ''}"
        f"deck_sigs={args.deck_sig or 'all'} team_names={args.team_name or 'all'} "
        f"opponent_deck_sigs={args.opponent_deck_sig or 'all'} "
        f"opponent_archetypes={args.opponent_archetype or 'all'} "
        f"opponent_team_names={args.opponent_team_name or 'all'} "
        f"opponent_deck_sig_weights={opponent_deck_sig_weights or '{}'} "
        f"opponent_archetype_weights={opponent_archetype_weights or '{}'} "
        f"winner_only={args.winner_only} "
        f"win/loss/draw_weight={args.win_weight}/{args.loss_weight}/{args.draw_weight} "
        f"context_weights={context_weights or '{}'} type_weights={type_weights or '{}'} "
        f"card_weights={card_weights or '{}'} "
        f"multi_select_weight={args.multi_select_weight} "
        f"set_loss={args.set_loss_weight}/{args.set_loss_min_count}/{args.set_loss_negative_weight} "
        f"aux_corpus={aux_details or 'none'} aux_bands={args.aux_score_bands} aux_repeat={aux_repeat} "
        f"params={params/1e6:.1f}M",
        flush=True,
    )
    print(
        f"Corpus: files={len(paths)} base_files={base_path_count} "
        f"aux_files={aux_path_count} stats={corpus.stats}",
        flush=True,
    )
    print(f"Split: train={len(train_idx)} val={len(val_idx)} batch={args.batch_size}", flush=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(train_idx)
        start = time.time()
        model.train()
        total = 0.0
        steps = 0
        for batch_start in range(0, len(train_idx), args.batch_size):
            batch_idx = train_idx[batch_start : batch_start + args.batch_size]
            if len(batch_idx) < 2:
                continue
            loss = sequence_nll(
                model,
                corpus.collate(batch_idx, device),
                first_action_weight=args.first_action_weight,
                value_weight=args.value_weight,
                set_loss_weight=args.set_loss_weight,
                set_loss_min_count=args.set_loss_min_count,
                set_loss_negative_weight=args.set_loss_negative_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach().cpu())
            steps += 1
            if steps == 1 or steps % 25 == 0 or steps == train_batches:
                print(
                    f"  epoch {epoch:02d} {steps:4d}/{train_batches} "
                    f"loss={total/max(steps,1):.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
        train_loss = total / max(steps, 1)
        with torch.no_grad():
            val_loss, val_steps = _run_epoch(
                model,
                corpus,
                val_idx,
                args.batch_size,
                device,
                optimizer=None,
                first_action_weight=args.first_action_weight,
                value_weight=args.value_weight,
                set_loss_weight=args.set_loss_weight,
                set_loss_min_count=args.set_loss_min_count,
                set_loss_negative_weight=args.set_loss_negative_weight,
            )
        elapsed = time.time() - start
        print(f"  done epoch {epoch}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f} {elapsed:.0f}s", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            _save_npz(model, args.save)
            print(f"  saved best {best_val:.4f} -> {args.save}", flush=True)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            ckpt = args.save.replace(".npz", f"_ep{epoch:03d}.npz")
            _save_npz(model, ckpt)
            print(f"  checkpoint {ckpt}", flush=True)

    print(f"Best val={best_val:.4f} -> {args.save}")


if __name__ == "__main__":
    main()
