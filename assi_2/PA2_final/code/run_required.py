from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RequiredRun:
    run_id: str
    name: str
    task: str
    model: str
    clipstyle: str
    cutoff: float
    init: str = "basic_tanh"
    nhid: int = 50
    alpha: float = 0.0


RUNS: dict[str, RequiredRun] = {
    "A1": RequiredRun("A1", "A1_mem_rnn_tanh_noclip", "mem", "rnn", "nothing", 1.0),
    "A2": RequiredRun("A2", "A2_mem_rnn_tanh_clip005", "mem", "rnn", "rescale", 0.05),
    "A3": RequiredRun("A3", "A3_mem_rnn_tanh_clip001", "mem", "rnn", "rescale", 0.01),
    "A4": RequiredRun("A4", "A4_mem_gru_noclip", "mem", "gru", "nothing", 1.0),
    "A5": RequiredRun("A5", "A5_mem_gru_clip005", "mem", "gru", "rescale", 0.05),
    "B1": RequiredRun("B1", "B1_mul_rnn_tanh_noclip", "mul", "rnn", "nothing", 1.0),
    "B2": RequiredRun("B2", "B2_mul_gru_noclip", "mul", "gru", "nothing", 1.0),
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    results_root = project_root.parent / "results"
    parser = argparse.ArgumentParser(description="Rerun the required A and B experiments and rebuild the plots.")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(RUNS.keys()),
        choices=sorted(RUNS.keys()),
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--maxiters", type=int, default=50000)
    parser.add_argument("--checkFreq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--valid-seed", type=int, default=12345)
    parser.add_argument("--force", action="store_true", help="Rerun even if the final checkpoint already exists.")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--checkpoint-dir", type=Path, default=results_root / "checkpoints")
    parser.add_argument("--log-dir", type=Path, default=results_root / "logs" / "required")
    parser.add_argument("--plot-dir", type=Path, default=results_root / "plots")
    return parser.parse_args()


def _command(args: argparse.Namespace, cfg: RequiredRun) -> list[str]:
    ckpt_prefix = args.checkpoint_dir / cfg.name
    cmd = [
        sys.executable,
        str(args.project_root / "train.py"),
        "--task",
        cfg.task,
        "--model",
        cfg.model,
        "--init",
        cfg.init,
        "--nhid",
        str(cfg.nhid),
        "--alpha",
        str(cfg.alpha),
        "--lr",
        "0.01",
        "--min_length",
        "50",
        "--max_length",
        "200",
        "--bs",
        "20",
        "--ebs",
        "10000",
        "--cbs",
        "1000",
        "--checkFreq",
        str(args.checkFreq),
        "--maxiters",
        str(args.maxiters),
        "--seed",
        str(args.seed),
        "--valid_seed",
        str(args.valid_seed),
        "--clipstyle",
        cfg.clipstyle,
        "--cutoff",
        str(cfg.cutoff),
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
    if cfg.model == "gru":
        cmd.append("--diagGates")
    return cmd


def _completed_iters(path: Path) -> int:
    if not path.exists():
        return 0
    with np.load(path, allow_pickle=True) as data:
        train = np.asarray(data["train_nll"])
    return int(np.sum(np.isfinite(train) & (train >= 0.0)))


def _run_one(args: argparse.Namespace, cfg: RequiredRun) -> None:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.checkpoint_dir / f"{cfg.name}_final_state.npz"
    log_path = args.log_dir / f"{cfg.name}.log"
    completed = _completed_iters(final_path)
    if completed >= args.maxiters and not args.force:
        return
    cmd = _command(args, cfg)
    if 0 < completed < args.maxiters:
        cmd.extend(["--resume", str(final_path)])
    with log_path.open("w", encoding="ascii") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        subprocess.run(cmd, cwd=args.project_root, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _rebuild_plots(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(args.project_root / "plotter.py"),
        "--device",
        args.device,
        "--base-dir",
        str(args.checkpoint_dir),
        "--out-dir",
        str(args.plot_dir),
        "--runs",
        *args.runs,
    ]
    subprocess.run(cmd, cwd=args.project_root, check=True)


def main() -> None:
    args = parse_args()
    for run_id in args.runs:
        _run_one(args, RUNS[run_id])
    if not args.skip_plot:
        _rebuild_plots(args)


if __name__ == "__main__":
    main()
