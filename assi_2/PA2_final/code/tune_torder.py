from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


NHID_CHOICES = np.array([50, 75, 100, 125], dtype=np.int64)
LENGTH_CHOICES = np.array([75, 100, 125, 150], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    results_root = project_root.parent / "results"
    parser = argparse.ArgumentParser(description="GP-based temporal-order tuning for the extra-credit setup.")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--valid-seed", type=int, default=12345)
    parser.add_argument("--short-iters", type=int, default=4000)
    parser.add_argument("--full-iters", type=int, default=10000)
    parser.add_argument("--checkFreq", type=int, default=20)
    parser.add_argument("--n-initial", type=int, default=6)
    parser.add_argument("--n-bo", type=int, default=6)
    parser.add_argument("--n-finalists", type=int, default=3)
    parser.add_argument("--candidate-pool", type=int, default=1024)
    parser.add_argument("--force", action="store_true", help="Rerun even if checkpoints already exist.")
    parser.add_argument("--checkpoint-dir", type=Path, default=results_root / "checkpoints")
    parser.add_argument("--out-dir", type=Path, default=results_root / "torder")
    return parser.parse_args()


def _sample_random(rng: np.random.RandomState) -> dict[str, float | int]:
    fixed_length = bool(rng.rand() < 0.65)
    length = int(rng.choice(LENGTH_CHOICES))
    if fixed_length:
        min_length = length
        max_length = length
    else:
        min_length = 50
        max_length = 200
    return {
        "cutoff": float(10 ** rng.uniform(math.log10(0.05), math.log10(2.0))),
        "alpha": float(rng.uniform(0.5, 4.0)),
        "lr": float(10 ** rng.uniform(math.log10(0.003), math.log10(0.03))),
        "nhid": int(rng.choice(NHID_CHOICES)),
        "min_length": int(min_length),
        "max_length": int(max_length),
    }


def _initial_design() -> list[dict[str, float | int]]:
    return [
        {"cutoff": 0.05, "alpha": 2.0, "lr": 0.01, "nhid": 50, "min_length": 50, "max_length": 200},
        {"cutoff": 1.0, "alpha": 2.0, "lr": 0.01, "nhid": 50, "min_length": 50, "max_length": 200},
        {"cutoff": 1.0, "alpha": 2.0, "lr": 0.01, "nhid": 100, "min_length": 100, "max_length": 100},
        {"cutoff": 0.5, "alpha": 1.0, "lr": 0.01, "nhid": 100, "min_length": 100, "max_length": 100},
        {"cutoff": 1.0, "alpha": 1.0, "lr": 0.005, "nhid": 100, "min_length": 100, "max_length": 100},
        {"cutoff": 0.25, "alpha": 4.0, "lr": 0.01, "nhid": 75, "min_length": 75, "max_length": 75},
    ]


def _round_cfg(cfg: dict[str, float | int]) -> dict[str, float | int]:
    nhid = int(NHID_CHOICES[np.argmin(np.abs(NHID_CHOICES - float(cfg["nhid"])))])
    min_length = int(cfg["min_length"])
    max_length = int(cfg["max_length"])
    if min_length == max_length:
        length = int(LENGTH_CHOICES[np.argmin(np.abs(LENGTH_CHOICES - min_length))])
        min_length = length
        max_length = length
    else:
        min_length = 50
        max_length = 200
    return {
        "cutoff": float(np.clip(cfg["cutoff"], 0.05, 2.0)),
        "alpha": float(np.clip(cfg["alpha"], 0.5, 4.0)),
        "lr": float(np.clip(cfg["lr"], 0.003, 0.03)),
        "nhid": nhid,
        "min_length": min_length,
        "max_length": max_length,
    }


def _cfg_key(cfg: dict[str, float | int]) -> tuple[float, float, float, int, int, int]:
    rounded = _round_cfg(cfg)
    return (
        round(float(rounded["cutoff"]), 6),
        round(float(rounded["alpha"]), 6),
        round(float(rounded["lr"]), 6),
        int(rounded["nhid"]),
        int(rounded["min_length"]),
        int(rounded["max_length"]),
    )


def _cfg_tag(cfg: dict[str, float | int]) -> str:
    rounded = _round_cfg(cfg)
    cutoff = f"{rounded['cutoff']:.3f}".replace(".", "")
    alpha = f"{rounded['alpha']:.3f}".replace(".", "")
    lr = f"{rounded['lr']:.4f}".replace(".", "")
    if rounded["min_length"] == rounded["max_length"]:
        length_tag = f"fix{rounded['min_length']}"
    else:
        length_tag = "rand50_200"
    return f"clip{cutoff}_alpha{alpha}_lr{lr}_nh{rounded['nhid']}_{length_tag}"


def _encode_cfg(cfg: dict[str, float | int]) -> np.ndarray:
    rounded = _round_cfg(cfg)
    is_fixed = 1.0 if rounded["min_length"] == rounded["max_length"] else 0.0
    length = float(rounded["min_length"] if is_fixed else 100)
    return np.array(
        [
            math.log10(float(rounded["cutoff"])),
            float(rounded["alpha"]),
            math.log10(float(rounded["lr"])),
            float(rounded["nhid"]),
            is_fixed,
            length,
        ],
        dtype=np.float64,
    )


def _metrics(npz_path: Path) -> dict[str, float]:
    with np.load(npz_path, allow_pickle=True) as data:
        train = data["train_nll"]
        valid = data["valid_error"]
        grad = data["gradient_norm"]
        rho = data["rho_Whh"]

        train = train[np.isfinite(train) & (train >= 0)]
        valid = valid[np.isfinite(valid) & (valid >= 0)]
        grad = grad[np.isfinite(grad) & (grad >= 0)]
        rho = rho[np.isfinite(rho) & (rho >= 0)]

        return {
            "final_train_nll": float(train[-1]) if train.size else float("nan"),
            "best_train_nll": float(train.min()) if train.size else float("nan"),
            "final_valid_error": float(valid[-1]) if valid.size else float("nan"),
            "best_valid_error": float(valid.min()) if valid.size else float("nan"),
            "final_grad_norm": float(grad[-1]) if grad.size else float("nan"),
            "max_grad_norm": float(grad.max()) if grad.size else float("nan"),
            "final_rho": float(rho[-1]) if rho.size else float("nan"),
        }


def _surrogate_loss(metrics: dict[str, float]) -> float:
    stall_penalty = max(0.0, metrics["best_train_nll"] - 1.0) * 40.0
    late_penalty = max(0.0, metrics["final_train_nll"] - metrics["best_train_nll"]) * 10.0
    return float(metrics["best_valid_error"] + stall_penalty + late_penalty)


def _train_command(
    args: argparse.Namespace,
    cfg: dict[str, float | int],
    name: str,
    maxiters: int,
) -> list[str]:
    ckpt_prefix = args.checkpoint_dir / name
    rounded = _round_cfg(cfg)
    return [
        sys.executable,
        str(args.project_root / "train.py"),
        "--task",
        "torder",
        "--model",
        "rnn",
        "--init",
        "smart_tanh",
        "--clipstyle",
        "rescale",
        "--cutoff",
        str(rounded["cutoff"]),
        "--alpha",
        str(rounded["alpha"]),
        "--lr",
        str(rounded["lr"]),
        "--nhid",
        str(rounded["nhid"]),
        "--bs",
        "20",
        "--min_length",
        str(rounded["min_length"]),
        "--max_length",
        str(rounded["max_length"]),
        "--maxiters",
        str(maxiters),
        "--ebs",
        "10000",
        "--cbs",
        "1000",
        "--checkFreq",
        str(args.checkFreq),
        "--seed",
        str(args.seed),
        "--valid_seed",
        str(args.valid_seed),
        "--collectDiags",
        "--diagBins",
        "60",
        "--satThresh",
        "0.05",
        "--device",
        args.device,
        "--name",
        str(ckpt_prefix),
    ]


def _run_trial(
    args: argparse.Namespace,
    cfg: dict[str, float | int],
    stage: str,
    trial_id: int,
    maxiters: int,
) -> dict[str, float | int | str]:
    rounded = _round_cfg(cfg)
    run_name = f"ECgp_{stage}_{trial_id:02d}_{_cfg_tag(rounded)}"
    final_path = args.checkpoint_dir / f"{run_name}_final_state.npz"
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    if args.force or not final_path.exists():
        cmd = _train_command(args, rounded, run_name, maxiters)
        with log_path.open("w", encoding="ascii") as handle:
            handle.write(" ".join(cmd) + "\n\n")
            subprocess.run(cmd, cwd=args.project_root, stdout=handle, stderr=subprocess.STDOUT, check=True)

    metrics = _metrics(final_path)
    row: dict[str, float | int | str] = {
        "stage": stage,
        "run_name": run_name,
        "cutoff": float(rounded["cutoff"]),
        "alpha": float(rounded["alpha"]),
        "lr": float(rounded["lr"]),
        "nhid": int(rounded["nhid"]),
        "min_length": int(rounded["min_length"]),
        "max_length": int(rounded["max_length"]),
        "surrogate_loss": _surrogate_loss(metrics),
    }
    row.update(metrics)
    return row


def _fit_gp(rows: list[dict[str, float | int | str]]) -> GaussianProcessRegressor:
    X = np.stack([_encode_cfg(row) for row in rows])
    y = np.array([float(row["surrogate_loss"]) for row in rows], dtype=np.float64)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(
        noise_level=1e-5,
        noise_level_bounds=(1e-8, 1e-2),
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0)
    gp.fit(X, y)
    return gp


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    z = (best - mu) / sigma
    return (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def _suggest_next(
    rng: np.random.RandomState,
    gp: GaussianProcessRegressor,
    seen: set[tuple[float, float, float, int, int, int]],
    candidate_pool: int,
) -> dict[str, float | int]:
    candidates = []
    while len(candidates) < candidate_pool:
        cfg = _sample_random(rng)
        key = _cfg_key(cfg)
        if key not in seen:
            candidates.append(_round_cfg(cfg))
    X = np.stack([_encode_cfg(cfg) for cfg in candidates])
    mu, sigma = gp.predict(X, return_std=True)
    y_best = float(np.min(gp.y_train_))
    ei = _expected_improvement(mu, sigma, y_best)
    return candidates[int(np.argmax(ei))]


def _write_report(rows: list[dict[str, float | int | str]], out_dir: Path) -> None:
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "summary.md"
    extra_path = out_dir / "extra_credit_summary.md"

    fieldnames = [
        "stage",
        "run_name",
        "cutoff",
        "alpha",
        "lr",
        "nhid",
        "min_length",
        "max_length",
        "surrogate_loss",
        "final_train_nll",
        "best_train_nll",
        "final_valid_error",
        "best_valid_error",
        "final_grad_norm",
        "max_grad_norm",
        "final_rho",
    ]

    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    short_rows = [row for row in rows if row["stage"] == "short"]
    full_rows = [row for row in rows if row["stage"] == "full"]
    lines = ["# Temporal-order GP search", ""]
    if short_rows:
        best_short = min(short_rows, key=lambda row: float(row["surrogate_loss"]))
        lines.append(
            f"Best short-run surrogate: `{best_short['run_name']}` with surrogate loss {float(best_short['surrogate_loss']):.3f}, "
            f"best valid error {float(best_short['best_valid_error']):.3f}%, and best train nll {float(best_short['best_train_nll']):.4f}."
        )
        lines.append("")
    if full_rows:
        best_full = min(full_rows, key=lambda row: float(row["best_valid_error"]))
        lines.append(
            f"Best full rerun: `{best_full['run_name']}` with best valid error {float(best_full['best_valid_error']):.3f}%, "
            f"best train nll {float(best_full['best_train_nll']):.4f}, and final rho {float(best_full['final_rho']):.3f}."
        )
        lines.append("")
    for row in rows:
        lines.append(
            f"- `{row['run_name']}` [{row['stage']}]: cutoff={float(row['cutoff']):.4f}, alpha={float(row['alpha']):.4f}, "
            f"lr={float(row['lr']):.5f}, nhid={int(row['nhid'])}, len={int(row['min_length'])}-{int(row['max_length'])}, "
            f"surrogate={float(row['surrogate_loss']):.3f}, best valid={float(row['best_valid_error']):.3f}%, "
            f"best train nll={float(row['best_train_nll']):.4f}"
        )
    text = "\n".join(lines)
    md_path.write_text(text, encoding="ascii")

    extra_lines = ["# Extra-credit temporal order note", ""]
    if short_rows:
        best_short = min(short_rows, key=lambda row: float(row["surrogate_loss"]))
        extra_lines.append(
            f"The GP search used a surrogate loss of `best_valid_error + 40 * max(best_train_nll - 1.0, 0) + 10 * max(final_train_nll - best_train_nll, 0)`."
        )
        extra_lines.append(
            f"The best short-run surrogate was `{best_short['run_name']}` with cutoff={float(best_short['cutoff']):.4f}, alpha={float(best_short['alpha']):.4f}, "
            f"lr={float(best_short['lr']):.5f}, nhid={int(best_short['nhid'])}, and length {int(best_short['min_length'])}-{int(best_short['max_length'])}."
        )
        extra_lines.append("")
    if full_rows:
        best_full = min(full_rows, key=lambda row: float(row["best_valid_error"]))
        extra_lines.append("The best kept full rerun was:")
        extra_lines.append("")
        extra_lines.append(f"- Run name: `{best_full['run_name']}`")
        extra_lines.append(
            f"- Key settings: `--min_length {int(best_full['min_length'])} --max_length {int(best_full['max_length'])} --nhid {int(best_full['nhid'])} --cutoff {float(best_full['cutoff']):.4f} --alpha {float(best_full['alpha']):.4f} --lr {float(best_full['lr']):.5f}`"
        )
        extra_lines.append(f"- Best observed validation error: `{float(best_full['best_valid_error']):.2f}%`")
        extra_lines.append(f"- Best train NLL: `{float(best_full['best_train_nll']):.4f}`")
        extra_lines.append(f"- Final rho(W_hh): `{float(best_full['final_rho']):.4f}`")
    extra_path.write_text("\n".join(extra_lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(20260310)
    rows: list[dict[str, float | int | str]] = []
    seen: set[tuple[float, float, float, int, int, int]] = set()

    design = _initial_design()
    while len(design) < args.n_initial:
        design.append(_sample_random(rng))

    trial_id = 0
    for cfg in design[: args.n_initial]:
        rounded = _round_cfg(cfg)
        key = _cfg_key(rounded)
        if key in seen:
            continue
        seen.add(key)
        rows.append(_run_trial(args, rounded, "short", trial_id, args.short_iters))
        trial_id += 1

    for _ in range(args.n_bo):
        short_rows = [row for row in rows if row["stage"] == "short"]
        gp = _fit_gp(short_rows)
        candidate = _suggest_next(rng, gp, seen, args.candidate_pool)
        key = _cfg_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        rows.append(_run_trial(args, candidate, "short", trial_id, args.short_iters))
        trial_id += 1

    short_rows = [row for row in rows if row["stage"] == "short"]
    short_rows.sort(key=lambda row: (float(row["surrogate_loss"]), float(row["best_valid_error"])))
    finalists = short_rows[: args.n_finalists]
    for full_id, row in enumerate(finalists):
        cfg = {
            "cutoff": float(row["cutoff"]),
            "alpha": float(row["alpha"]),
            "lr": float(row["lr"]),
            "nhid": int(row["nhid"]),
            "min_length": int(row["min_length"]),
            "max_length": int(row["max_length"]),
        }
        rows.append(_run_trial(args, cfg, "full", full_id, args.full_iters))

    rows.sort(key=lambda row: (row["stage"], float(row["surrogate_loss"]) if row["stage"] == "short" else float(row["best_valid_error"])))
    _write_report(rows, args.out_dir)
    full_rows = [row for row in rows if row["stage"] == "full"]
    if full_rows:
        best_full = min(full_rows, key=lambda row: float(row["best_valid_error"]))
        print(
            f"Best full rerun: {best_full['run_name']} | best valid error={float(best_full['best_valid_error']):.3f}% | "
            f"best train nll={float(best_full['best_train_nll']):.4f}"
        )


if __name__ == "__main__":
    main()
