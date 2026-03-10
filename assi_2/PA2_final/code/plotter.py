from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


EPS = 1e-12
SAT_THRESH = 0.05


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    name: str
    task: str
    model: str
    diag_gates: bool = False


RUNS: dict[str, RunConfig] = {
    "A1": RunConfig("A1", "A1_mem_rnn_tanh_noclip", "mem", "rnn"),
    "A2": RunConfig("A2", "A2_mem_rnn_tanh_clip005", "mem", "rnn"),
    "A3": RunConfig("A3", "A3_mem_rnn_tanh_clip001", "mem", "rnn"),
    "A4": RunConfig("A4", "A4_mem_gru_noclip", "mem", "gru", diag_gates=True),
    "A5": RunConfig("A5", "A5_mem_gru_clip005", "mem", "gru", diag_gates=True),
    "B1": RunConfig("B1", "B1_mul_rnn_tanh_noclip", "mul", "rnn"),
    "B2": RunConfig("B2", "B2_mul_gru_noclip", "mul", "gru", diag_gates=True),
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Plot diagnostics from saved *_final_state.npz files.")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["A1", "A2", "A3", "A4", "A5", "B1", "B2"],
        choices=sorted(RUNS.keys()),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=project_root / "results" / "checkpoints",
        help="Directory containing *_final_state.npz files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "results" / "plots",
        help="Directory where plots and summary.md will be written.",
    )
    return parser.parse_args()


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (14, 8)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13


def _npz_path(base_dir: Path, cfg: RunConfig) -> Path:
    return base_dir / f"{cfg.name}_final_state.npz"


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _finite_curve(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    mask = np.isfinite(arr)
    if np.issubdtype(arr.dtype, np.number):
        mask &= arr >= 0
    return arr[mask]


def _finite_indexed_curve(arr: np.ndarray, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr)
    mask = np.isfinite(arr)
    if np.issubdtype(arr.dtype, np.number):
        mask &= arr >= 0
    return np.flatnonzero(mask) * step, arr[mask]


def _finite_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        return np.empty((0, 0), dtype=np.float64)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    row_mask = np.isfinite(arr).any(axis=1)
    col_mask = np.isfinite(arr).any(axis=0)
    if not row_mask.any() or not col_mask.any():
        return np.empty((0, 0), dtype=np.float64)
    return arr[row_mask][:, col_mask]


def _last_diag_row(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return np.array([], dtype=np.float64)
    for row in arr[::-1]:
        mask = np.isfinite(row)
        if mask.any():
            return row[mask]
    return np.array([], dtype=np.float64)


def _checkpoint_curve(payload: dict[str, np.ndarray], key: str) -> tuple[np.ndarray, np.ndarray]:
    check_freq = int(payload.get("checkFreq", 20))
    return _finite_indexed_curve(payload[key], step=check_freq)


def _downsample_rows(arr: np.ndarray, max_rows: int = 240) -> tuple[np.ndarray, np.ndarray]:
    if arr.shape[0] <= max_rows:
        return arr, np.arange(arr.shape[0])
    idx = np.unique(np.linspace(0, arr.shape[0] - 1, max_rows).round().astype(int))
    return arr[idx], idx


def _run_dir(out_dir: Path, run_id: str) -> Path:
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _compare_dir(out_dir: Path) -> Path:
    compare_dir = out_dir / "comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)
    return compare_dir


def _heatmap(ax: plt.Axes, arr: np.ndarray, title: str, xlabel: str, ylabel: str, cmap: str, vmin: float | None = None, vmax: float | None = None) -> None:
    if arr.size == 0:
        ax.text(0.5, 0.5, "not enough data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    view, row_idx = _downsample_rows(arr)
    sns.heatmap(
        view,
        ax=ax,
        cmap=cmap,
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{ylabel} ({row_idx[0]} to {row_idx[-1]})")


def _final_diag(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    grad_time = _last_diag_row(payload.get("grad_time", np.empty((0, 0))))
    sat_time = _last_diag_row(payload.get("sat_time", np.empty((0, 0))))
    z_sat_time = _last_diag_row(payload.get("gate_z_sat_time", np.empty((0, 0))))
    r_sat_time = _last_diag_row(payload.get("gate_r_sat_time", np.empty((0, 0))))

    return {
        "grad_time": grad_time,
        "grad_log10": np.log10(np.maximum(grad_time, EPS)),
        "hidden_sat": sat_time,
        "z_sat": z_sat_time,
        "r_sat": r_sat_time,
    }


def _gradient_label(values: np.ndarray) -> str:
    if values.size == 0:
        return "not enough data"
    p50 = float(np.median(values))
    p95 = float(np.quantile(values, 0.95))
    if p95 < -4.0:
        return "mostly vanishing"
    if p50 > 0.0:
        return "mostly exploding"
    return "spread across scales"


def _saturation_label(values: np.ndarray) -> str:
    if values.size == 0:
        return "not enough data"
    frac = float((values < SAT_THRESH).mean())
    if frac > 0.60:
        return "heavily saturated"
    if frac > 0.30:
        return "partly saturated"
    return "mostly unsaturated"


def _gate_label(values: np.ndarray) -> str:
    if values.size == 0:
        return "not enough data"
    edge = float((values < 0.1).mean())
    middle = float((values > 0.4).mean())
    if edge > middle * 1.2:
        return "more mass near 0 or 1 (saturated)"
    if middle > edge * 1.2:
        return "more mass near 0.5 (unsaturated)"
    return "mixed"


def _rho_grad_corr(payload: dict[str, np.ndarray]) -> float:
    rho = np.asarray(payload["rho_Whh"])
    grad_time = np.asarray(payload["grad_time"])
    valid_rows = np.isfinite(rho) & (rho >= 0)
    g_stat = np.full_like(rho, np.nan, dtype=np.float64)

    if grad_time.ndim == 2:
        for idx, row in enumerate(grad_time):
            finite = np.isfinite(row) & (row >= 0)
            if finite.any():
                g_stat[idx] = float(np.median(np.log10(row[finite] + EPS)))

    mask = valid_rows & np.isfinite(g_stat)
    if mask.sum() < 2:
        return float("nan")
    if np.nanstd(rho[mask]) < 1e-12 or np.nanstd(g_stat[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rho[mask], g_stat[mask])[0, 1])


def _run_summary(cfg: RunConfig, payload: dict[str, np.ndarray], diag: dict[str, np.ndarray]) -> dict[str, float | str]:
    grad_curve = _finite_curve(payload["gradient_norm"])
    valid_curve = _finite_curve(payload["valid_error"])
    rho_curve = _finite_curve(payload["rho_Whh"])

    summary = {
        "run_id": cfg.run_id,
        "name": cfg.name,
        "task": cfg.task,
        "model": cfg.model,
        "final_valid_error": float(valid_curve[-1]) if valid_curve.size else float("nan"),
        "best_valid_error": float(valid_curve.min()) if valid_curve.size else float("nan"),
        "final_grad_norm": float(grad_curve[-1]) if grad_curve.size else float("nan"),
        "max_grad_norm": float(grad_curve.max()) if grad_curve.size else float("nan"),
        "rho_start": float(rho_curve[0]) if rho_curve.size else float("nan"),
        "rho_end": float(rho_curve[-1]) if rho_curve.size else float("nan"),
        "rho_max": float(rho_curve.max()) if rho_curve.size else float("nan"),
        "rho_grad_corr": _rho_grad_corr(payload),
        "grad_behavior": _gradient_label(diag["grad_log10"]),
        "sat_behavior": _saturation_label(diag["hidden_sat"]),
        "sat_frac": float((diag["hidden_sat"] < SAT_THRESH).mean()) if diag["hidden_sat"].size else float("nan"),
    }
    if cfg.model == "gru":
        summary["z_gate_behavior"] = _gate_label(diag["z_sat"])
        summary["r_gate_behavior"] = _gate_label(diag["r_sat"])
    return summary


def _save_base_plot(cfg: RunConfig, payload: dict[str, np.ndarray], diag: dict[str, np.ndarray], run_dir: Path) -> None:
    _style()
    valid_x, valid_err = _checkpoint_curve(payload, "valid_error")
    rho_x, rho = _checkpoint_curve(payload, "rho_Whh")

    ncols = 3 if cfg.model == "gru" else 2
    fig, axes = plt.subplots(2, ncols, figsize=(7 * ncols, 10))
    axes = np.asarray(axes)

    sns.histplot(diag["grad_log10"], bins=60, color="#1f77b4", ax=axes[0, 0])
    axes[0, 0].set_title("log10 ||dL/dh_t||")
    axes[0, 0].set_xlabel("log10 g_t")

    sns.histplot(diag["hidden_sat"], bins=60, binrange=(0.0, 1.0), color="#e76f51", ax=axes[0, 1])
    axes[0, 1].set_title("hidden saturation distance")
    axes[0, 1].set_xlabel("distance to saturation")
    axes[0, 1].axvline(SAT_THRESH, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xlim(0.0, 1.0)

    if cfg.model == "gru":
        sns.histplot(diag["z_sat"], bins=60, binrange=(0.0, 0.5), color="#588157", ax=axes[0, 2])
        axes[0, 2].set_title("gate z saturation distance")
        axes[0, 2].set_xlabel("distance to saturation")
        axes[0, 2].set_xlim(0.0, 0.5)

    sns.lineplot(x=valid_x, y=valid_err, color="#2a9d8f", ax=axes[1, 0])
    axes[1, 0].set_title("validation error (%)")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].set_ylabel("validation error (%)")

    sns.lineplot(x=rho_x, y=rho, color="#9b2226", ax=axes[1, 1])
    axes[1, 1].set_title("rho(W_hh)")
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel("rho(W_hh)")

    if cfg.model == "gru":
        sns.histplot(diag["r_sat"], bins=60, binrange=(0.0, 0.5), color="#6d597a", ax=axes[1, 2])
        axes[1, 2].set_title("gate r saturation distance")
        axes[1, 2].set_xlabel("distance to saturation")
        axes[1, 2].set_xlim(0.0, 0.5)

    fig.tight_layout()
    fig.savefig(run_dir / "base_plots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_training_plot(cfg: RunConfig, payload: dict[str, np.ndarray], run_dir: Path) -> None:
    _style()
    train_nll = _finite_curve(payload["train_nll"])
    grad_norm = _finite_curve(payload["gradient_norm"])
    valid_x, valid_err = _checkpoint_curve(payload, "valid_error")
    rho_x, rho = _checkpoint_curve(payload, "rho_Whh")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sns.lineplot(x=np.arange(len(train_nll)), y=train_nll, color="#355070", ax=axes[0, 0])
    axes[0, 0].set_title(f"{cfg.run_id}: train NLL")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].set_ylabel("train NLL")

    sns.lineplot(x=valid_x, y=valid_err, color="#2a9d8f", ax=axes[0, 1])
    axes[0, 1].set_title(f"{cfg.run_id}: validation error")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].set_ylabel("validation error (%)")

    sns.lineplot(x=np.arange(len(grad_norm)), y=grad_norm, color="#023047", ax=axes[1, 0])
    axes[1, 0].set_title(f"{cfg.run_id}: gradient norm")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].set_ylabel("global grad norm")
    axes[1, 0].set_yscale("log")

    sns.lineplot(x=rho_x, y=rho, color="#9b2226", ax=axes[1, 1])
    axes[1, 1].set_title(f"{cfg.run_id}: rho(W_hh)")
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel("rho(W_hh)")

    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_gradient_plot(cfg: RunConfig, payload: dict[str, np.ndarray], diag: dict[str, np.ndarray], run_dir: Path) -> None:
    _style()
    grad_norm = _finite_curve(payload["gradient_norm"])
    grad_norm_log = np.log10(np.maximum(grad_norm, EPS))
    grad_heat = _finite_matrix(payload["grad_time"])
    if grad_heat.size:
        grad_heat = np.log10(np.maximum(grad_heat, EPS))

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    sns.histplot(diag["grad_log10"], bins=60, kde=True, color="#1f77b4", ax=axes[0, 0])
    axes[0, 0].set_title(f"{cfg.run_id}: final log10 g_t histogram")
    axes[0, 0].set_xlabel("log10 g_t")

    sns.lineplot(x=np.arange(diag["grad_log10"].shape[0]), y=diag["grad_log10"], color="#264653", ax=axes[0, 1])
    axes[0, 1].set_title(f"{cfg.run_id}: final log10 g_t by timestep")
    axes[0, 1].set_xlabel("timestep")
    axes[0, 1].set_ylabel("log10 g_t")

    if grad_heat.size:
        checkpoint_median = np.nanmedian(grad_heat, axis=1)
        sns.lineplot(x=np.arange(checkpoint_median.shape[0]), y=checkpoint_median, color="#3a86ff", ax=axes[0, 2])
    axes[0, 2].set_title(f"{cfg.run_id}: checkpoint median log10 g_t")
    axes[0, 2].set_xlabel("checkpoint index")
    axes[0, 2].set_ylabel("median log10 g_t")

    _heatmap(
        axes[1, 0],
        grad_heat,
        f"{cfg.run_id}: log10 g_t over checkpoints",
        "time index",
        "checkpoint",
        cmap="mako",
        vmin=-12.0,
        vmax=2.0,
    )

    sns.lineplot(x=np.arange(len(grad_norm)), y=grad_norm_log, color="#457b9d", ax=axes[1, 1])
    axes[1, 1].set_title(f"{cfg.run_id}: log10 global grad norm")
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel("log10 global grad norm")

    sns.histplot(grad_norm_log, bins=70, kde=True, color="#6aaed6", ax=axes[1, 2])
    axes[1, 2].set_title(f"{cfg.run_id}: log10 global grad norm histogram")
    axes[1, 2].set_xlabel("log10 global grad norm")

    fig.tight_layout()
    fig.savefig(run_dir / "gradient_views.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_saturation_plot(cfg: RunConfig, payload: dict[str, np.ndarray], diag: dict[str, np.ndarray], run_dir: Path) -> None:
    _style()
    sat_heat = _finite_matrix(payload["sat_time"])
    final_sat = diag["hidden_sat"]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    sns.histplot(final_sat, bins=60, binrange=(0.0, 1.0), kde=True, color="#e76f51", ax=axes[0, 0])
    axes[0, 0].set_title(f"{cfg.run_id}: final hidden saturation distance")
    axes[0, 0].set_xlabel("distance to saturation")
    axes[0, 0].axvline(SAT_THRESH, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xlim(0.0, 1.0)

    sns.ecdfplot(final_sat, color="#d62828", ax=axes[0, 1])
    axes[0, 1].set_title(f"{cfg.run_id}: hidden saturation ECDF")
    axes[0, 1].set_xlabel("distance to saturation")
    axes[0, 1].set_ylabel("ECDF")
    axes[0, 1].axvline(SAT_THRESH, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xlim(0.0, 1.0)

    sns.lineplot(x=np.arange(final_sat.shape[0]), y=final_sat, color="#bc4749", ax=axes[0, 2])
    axes[0, 2].set_title(f"{cfg.run_id}: final hidden saturation by timestep")
    axes[0, 2].set_xlabel("timestep")
    axes[0, 2].set_ylabel("distance to saturation")
    axes[0, 2].axhline(SAT_THRESH, color="black", linestyle="--", linewidth=1)

    _heatmap(
        axes[1, 0],
        sat_heat,
        f"{cfg.run_id}: hidden saturation over checkpoints",
        "time index",
        "checkpoint",
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
    )

    if sat_heat.size:
        checkpoint_mean = np.nanmean(sat_heat, axis=1)
        sns.lineplot(x=np.arange(checkpoint_mean.shape[0]), y=checkpoint_mean, color="#b56576", ax=axes[1, 1])
    axes[1, 1].set_title(f"{cfg.run_id}: checkpoint mean saturation")
    axes[1, 1].set_xlabel("checkpoint index")
    axes[1, 1].set_ylabel("mean distance to saturation")

    if sat_heat.size:
        time_mean = np.nanmean(sat_heat, axis=0)
        sns.lineplot(x=np.arange(time_mean.shape[0]), y=time_mean, color="#e09f3e", ax=axes[1, 2])
    axes[1, 2].set_title(f"{cfg.run_id}: mean saturation by timestep")
    axes[1, 2].set_xlabel("time index")
    axes[1, 2].set_ylabel("mean distance to saturation")

    fig.tight_layout()
    fig.savefig(run_dir / "saturation_views.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_gate_plot(cfg: RunConfig, payload: dict[str, np.ndarray], diag: dict[str, np.ndarray], run_dir: Path) -> None:
    if cfg.model != "gru":
        return

    _style()
    z_heat = _finite_matrix(payload["gate_z_sat_time"])
    r_heat = _finite_matrix(payload["gate_r_sat_time"])
    z_sat = diag["z_sat"]
    r_sat = diag["r_sat"]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    sns.histplot(z_sat, bins=60, binrange=(0.0, 0.5), kde=True, color="#588157", ax=axes[0, 0])
    axes[0, 0].set_title(f"{cfg.run_id}: update gate saturation")
    axes[0, 0].set_xlabel("distance to saturation")
    axes[0, 0].set_xlim(0.0, 0.5)

    sns.histplot(r_sat, bins=60, binrange=(0.0, 0.5), kde=True, color="#6d597a", ax=axes[0, 1])
    axes[0, 1].set_title(f"{cfg.run_id}: reset gate saturation")
    axes[0, 1].set_xlabel("distance to saturation")
    axes[0, 1].set_xlim(0.0, 0.5)

    sns.ecdfplot(z_sat, color="#588157", ax=axes[0, 2], label="update")
    sns.ecdfplot(r_sat, color="#6d597a", ax=axes[0, 2], label="reset")
    axes[0, 2].set_title(f"{cfg.run_id}: gate saturation ECDF")
    axes[0, 2].set_xlabel("distance to saturation")
    axes[0, 2].set_xlim(0.0, 0.5)
    axes[0, 2].legend()

    sns.lineplot(x=np.arange(z_sat.shape[0]), y=z_sat, color="#588157", ax=axes[1, 0], label="update")
    sns.lineplot(x=np.arange(r_sat.shape[0]), y=r_sat, color="#6d597a", ax=axes[1, 0], label="reset")
    axes[1, 0].set_title(f"{cfg.run_id}: final gate saturation by timestep")
    axes[1, 0].set_xlabel("timestep")
    axes[1, 0].set_ylabel("distance to saturation")
    axes[1, 0].legend()

    _heatmap(
        axes[1, 1],
        z_heat,
        f"{cfg.run_id}: update gate over checkpoints",
        "time index",
        "checkpoint",
        cmap="crest",
        vmin=0.0,
        vmax=0.5,
    )

    _heatmap(
        axes[1, 2],
        r_heat,
        f"{cfg.run_id}: reset gate over checkpoints",
        "time index",
        "checkpoint",
        cmap="flare",
        vmin=0.0,
        vmax=0.5,
    )

    fig.tight_layout()
    fig.savefig(run_dir / "gate_views.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compare_runs(run_ids: list[str], payloads: dict[str, dict[str, np.ndarray]], diags: dict[str, dict[str, np.ndarray]], out_path: Path, title: str) -> None:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    palette = sns.color_palette("Set2", n_colors=len(run_ids))

    for color, run_id in zip(palette, run_ids):
        payload = payloads[run_id]
        diag = diags[run_id]
        grad_norm = _finite_curve(payload["gradient_norm"])
        valid_x, valid_err = _checkpoint_curve(payload, "valid_error")
        rho_x, rho = _checkpoint_curve(payload, "rho_Whh")
        sns.lineplot(x=np.arange(len(grad_norm)), y=grad_norm, color=color, label=run_id, ax=axes[0, 0])
        sns.lineplot(x=valid_x, y=valid_err, color=color, label=run_id, ax=axes[0, 1])
        sns.lineplot(x=rho_x, y=rho, color=color, label=run_id, ax=axes[0, 2])
        sns.lineplot(x=np.arange(diag["grad_log10"].shape[0]), y=diag["grad_log10"], color=color, label=run_id, ax=axes[1, 0])
        sns.ecdfplot(diag["hidden_sat"], color=color, label=run_id, ax=axes[1, 1])
        sns.histplot(diag["grad_log10"], bins=50, stat="density", element="step", fill=False, color=color, label=run_id, ax=axes[1, 2])

    axes[0, 0].set_title("gradient norm over training")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_title("validation error over training")
    axes[0, 1].set_ylabel("validation error (%)")
    axes[0, 2].set_title("rho(W_hh) over training")
    axes[1, 0].set_title("final log10 g_t by timestep")
    axes[1, 0].set_xlabel("timestep")
    axes[1, 0].set_ylabel("log10 g_t")
    axes[1, 1].set_title("hidden saturation ECDF")
    axes[1, 1].set_xlabel("distance to saturation")
    axes[1, 1].axvline(SAT_THRESH, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xlim(0.0, 1.0)
    axes[1, 2].set_title("final log10 g_t histogram")
    axes[1, 2].set_xlabel("log10 g_t")

    for ax in axes.ravel():
        ax.legend()

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compare_gates(run_ids: list[str], diags: dict[str, dict[str, np.ndarray]], out_path: Path, title: str) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    palette = sns.color_palette("deep", n_colors=len(run_ids))

    for color, run_id in zip(palette, run_ids):
        diag = diags[run_id]
        sns.histplot(diag["z_sat"], bins=50, stat="density", element="step", fill=False, color=color, label=run_id, ax=axes[0, 0])
        sns.histplot(diag["r_sat"], bins=50, stat="density", element="step", fill=False, color=color, label=run_id, ax=axes[0, 1])
        sns.lineplot(x=np.arange(diag["z_sat"].shape[0]), y=diag["z_sat"], color=color, label=f"{run_id} z", ax=axes[1, 0])
        sns.lineplot(x=np.arange(diag["r_sat"].shape[0]), y=diag["r_sat"], color=color, label=f"{run_id} r", ax=axes[1, 1])

    axes[0, 0].set_title("update gate saturation histogram")
    axes[0, 0].set_xlim(0.0, 0.5)
    axes[0, 1].set_title("reset gate saturation histogram")
    axes[0, 1].set_xlim(0.0, 0.5)
    axes[1, 0].set_title("update gate saturation by timestep")
    axes[1, 0].set_xlabel("timestep")
    axes[1, 0].set_ylabel("distance to saturation")
    axes[1, 1].set_title("reset gate saturation by timestep")
    axes[1, 1].set_xlabel("timestep")
    axes[1, 1].set_ylabel("distance to saturation")

    for ax in axes.ravel():
        ax.legend()

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _report_text(selected_runs: list[str], summaries: dict[str, dict[str, float | str]]) -> str:
    lines = ["# PA2 dynamics summary", ""]
    for run_id in selected_runs:
        s = summaries[run_id]
        lines.append(f"## {run_id} ({s['name']})")
        lines.append(
            f"- Final valid error: {s['final_valid_error']:.3f}% | best: {s['best_valid_error']:.3f}% | final grad norm: {s['final_grad_norm']:.4g}"
        )
        lines.append(
            f"- Gradient histogram: {s['grad_behavior']} | hidden saturation: {s['sat_behavior']} | saturated fraction (<{SAT_THRESH}): {100.0 * s['sat_frac']:.1f}%"
        )
        lines.append(
            f"- rho(W_hh): start {s['rho_start']:.3f}, end {s['rho_end']:.3f}, max {s['rho_max']:.3f}, corr(rho, median log10 g_t) {s['rho_grad_corr']:.3f}"
        )
        if s["model"] == "gru":
            lines.append(
                f"- Gate saturation histograms: z {s['z_gate_behavior']}, r {s['r_gate_behavior']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = _compare_dir(out_dir)

    selected_runs = args.runs
    payloads: dict[str, dict[str, np.ndarray]] = {}
    diags: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, float | str]] = {}

    for run_id in selected_runs:
        cfg = RUNS[run_id]
        path = _npz_path(args.base_dir, cfg)
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {run_id}: {path}")
        payload = _load_payload(path)
        diag = _final_diag(payload)
        payloads[run_id] = payload
        diags[run_id] = diag
        summaries[run_id] = _run_summary(cfg, payload, diag)

        run_dir = _run_dir(out_dir, run_id)
        _save_base_plot(cfg, payload, diag, run_dir)
        _save_training_plot(cfg, payload, run_dir)
        _save_gradient_plot(cfg, payload, diag, run_dir)
        _save_saturation_plot(cfg, payload, diag, run_dir)
        _save_gate_plot(cfg, payload, diag, run_dir)

    if all(run_id in selected_runs for run_id in ("A1", "A2", "A3")):
        _compare_runs(["A1", "A2", "A3"], payloads, diags, compare_dir / "mem_rnn_clipping_compare.png", "Memorization: RNN clipping comparison")
    if all(run_id in selected_runs for run_id in ("A4", "A5")):
        _compare_runs(["A4", "A5"], payloads, diags, compare_dir / "mem_gru_clipping_compare.png", "Memorization: GRU clipping comparison")
        _compare_gates(["A4", "A5"], diags, compare_dir / "mem_gru_gate_compare.png", "Memorization: GRU gate comparison")
    if all(run_id in selected_runs for run_id in ("A1", "A4")):
        _compare_runs(["A1", "A4"], payloads, diags, compare_dir / "mem_rnn_vs_gru_noclip.png", "Memorization: RNN vs GRU without clipping")
    if all(run_id in selected_runs for run_id in ("B1", "B2")):
        _compare_runs(["B1", "B2"], payloads, diags, compare_dir / "mul_rnn_vs_gru_noclip.png", "Multiplication: RNN vs GRU without clipping")
        _compare_gates(["B2"], diags, compare_dir / "mul_gru_gate_compare.png", "Multiplication: GRU gate plots")

    report = _report_text(selected_runs, summaries)
    (out_dir / "summary.md").write_text(report, encoding="ascii")
    print(f"Wrote plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
