from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# USER SETTINGS
# ============================================================

OUT_DIR = Path(
    r"C:\Files from USB\Carleton Year 5\MAAE 4907-N (Capstone Project - Micro Flapping-Wing Flyer (MFWF))\Neural Network (Fall 2025 - Winter 2026)\Data\Output\Output_folder_earlystopX"
)

TRAINING_LOG_FILE = OUT_DIR / "training_log.txt"
MAT_DIR = OUT_DIR / "matfiles"

FIG_DIR = OUT_DIR / "figures"
METRICS_FILE = OUT_DIR / "metrics_summary.csv"

SHOW_PLOTS = True

# Input labels aligned to the notation used in the companion paper.
INPUT_VARIABLE_NAMES = [
    "AoA",
    "|v|",
    "a_x",
    "a_y",
    "a_z",
    "alpha_dot",
    "alpha_ddot",
]

# Output labels aligned to the notation used in the companion paper.
OUTPUT_VARIABLE_NAMES = [
    "CF_x",
    "CF_y",
    "CM_x",
    "CM_y",
    "CM_z",
]

OUTPUT_VARIABLE_DESCRIPTIONS = {
    "CF_x": "normal force coefficient",
    "CF_y": "chordwise force coefficient",
    "CM_x": "normal moment coefficient",
    "CM_y": "chordwise moment coefficient",
    "CM_z": "spanwise moment coefficient",
}


# ============================================================
# GLOBAL FIGURE STORAGE
# ============================================================

GENERATED_FIGURES: List[plt.Figure] = []


# ============================================================
# PATH HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    long_path = to_windows_long_path(path)
    os.makedirs(long_path, exist_ok=True)


def to_windows_long_path(path: Path) -> str:
    abs_path = str(path.resolve())

    if os.name != "nt":
        return abs_path

    if abs_path.startswith("\\\\?\\"):
        return abs_path

    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")

    return "\\\\?\\" + abs_path


def path_exists_long(path: Path) -> bool:
    try:
        return os.path.exists(to_windows_long_path(path))
    except Exception:
        return False


def find_file_case_insensitive(folder: Path, target_name: str) -> Path:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    target_lower = target_name.lower()

    for name in os.listdir(str(folder)):
        if name.lower() == target_lower:
            return folder / name

    available = ", ".join(repr(name) for name in os.listdir(str(folder)))
    raise FileNotFoundError(
        f"Could not find {target_name!r} in {folder}\nAvailable files: {available}"
    )


def load_clean_mat(path: Path) -> Dict:
    long_path = to_windows_long_path(path)
    raw = loadmat(long_path)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def read_text_long(path: Path, encoding: str = "utf-8") -> str:
    long_path = to_windows_long_path(path)
    with open(long_path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()


# ============================================================
# METRIC HELPERS
# ============================================================

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if np.isclose(ss_tot, 0.0):
        return np.nan

    return 1.0 - (ss_res / ss_tot)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_percentile_envelope(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentile: float = 98.0,
) -> float:
    abs_err = np.abs(np.asarray(y_pred, dtype=float).ravel() - np.asarray(y_true, dtype=float).ravel())
    return float(np.percentile(abs_err, percentile))


def get_output_variable_names(n_outputs: int) -> List[str]:
    if len(OUTPUT_VARIABLE_NAMES) != n_outputs:
        raise ValueError(
            f"OUTPUT_VARIABLE_NAMES has length {len(OUTPUT_VARIABLE_NAMES)} but model has {n_outputs} outputs."
        )
    return OUTPUT_VARIABLE_NAMES


def format_output_label(variable_name: str) -> str:
    description = OUTPUT_VARIABLE_DESCRIPTIONS.get(variable_name)
    if description is None:
        return variable_name
    return f"{variable_name} ({description})"


# ============================================================
# TRAINING LOG PARSER
# ============================================================

def parse_training_log(log_file: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not path_exists_long(log_file):
        raise FileNotFoundError(f"Missing file: {log_file}")

    text = read_text_long(log_file)
    lines = text.splitlines()

    epoch_pattern = re.compile(
        r"\[(\d+)\]:\s*Train\s+([0-9eE+\-.]+),\s*Test\s+([0-9eE+\-.]+)"
    )
    time_pattern = re.compile(
        r"Epoch time:\s*([0-9eE+\-.]+)s\s*\|\s*Elapsed:\s*([0-9eE+\-.]+)\s*min"
    )
    total_time_pattern = re.compile(
        r"Total training time:\s*([0-9eE+\-.]+)\s*seconds"
    )
    best_test_pattern = re.compile(
        r"Best test loss was\s*([0-9eE+\-.]+)\s*at epoch\s*(\d+)"
    )

    rows = []
    for i, line in enumerate(lines):
        m_epoch = epoch_pattern.search(line)
        if m_epoch:
            epoch = int(m_epoch.group(1))
            train_loss = float(m_epoch.group(2))
            test_loss = float(m_epoch.group(3))

            epoch_time_sec = np.nan
            elapsed_min = np.nan

            if i + 1 < len(lines):
                m_time = time_pattern.search(lines[i + 1])
                if m_time:
                    epoch_time_sec = float(m_time.group(1))
                    elapsed_min = float(m_time.group(2))

            rows.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "epoch_time_sec": epoch_time_sec,
                "elapsed_min": elapsed_min,
            })

    history_df = pd.DataFrame(rows)

    summary: Dict[str, float] = {}

    m_total = total_time_pattern.search(text)
    if m_total:
        summary["total_training_time_sec"] = float(m_total.group(1))

    m_best = best_test_pattern.search(text)
    if m_best:
        summary["best_test_loss"] = float(m_best.group(1))
        summary["best_test_epoch"] = int(m_best.group(2))

    return history_df, summary


# ============================================================
# MATLAB DATA EXTRACTION
# ============================================================

def extract_training_loss(loss_mat: Dict) -> pd.DataFrame:
    if "train_loss" not in loss_mat or "test_loss" not in loss_mat:
        raise KeyError("training_loss.mat must contain 'train_loss' and 'test_loss'")

    train_loss = np.asarray(loss_mat["train_loss"], dtype=float).squeeze()
    test_loss = np.asarray(loss_mat["test_loss"], dtype=float).squeeze()

    n = min(len(train_loss), len(test_loss))
    return pd.DataFrame({
        "epoch": np.arange(n),
        "train_loss": train_loss[:n],
        "test_loss": test_loss[:n],
    })


def reshape_predictions(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        return arr.reshape(-1, 1)

    if arr.ndim == 2:
        return arr

    n_outputs = arr.shape[-1]
    return arr.reshape(-1, n_outputs)


def extract_prediction_arrays(pred_mat: Dict) -> Dict[str, np.ndarray]:
    required = ["gp_mean_train", "gt_train", "gp_mean_test", "gt_test"]
    for key in required:
        if key not in pred_mat:
            raise KeyError(f"predict_train_n_test.mat must contain '{key}'")

    pred_train = reshape_predictions(pred_mat["gp_mean_train"])
    gt_train = reshape_predictions(pred_mat["gt_train"])
    pred_test = reshape_predictions(pred_mat["gp_mean_test"])
    gt_test = reshape_predictions(pred_mat["gt_test"])

    if pred_train.shape != gt_train.shape:
        raise ValueError(
            f"Training prediction and truth shapes do not match: "
            f"{pred_train.shape} vs {gt_train.shape}"
        )

    if pred_test.shape != gt_test.shape:
        raise ValueError(
            f"Testing prediction and truth shapes do not match: "
            f"{pred_test.shape} vs {gt_test.shape}"
        )

    return {
        "pred_train": pred_train,
        "gt_train": gt_train,
        "pred_test": pred_test,
        "gt_test": gt_test,
    }


# ============================================================
# PLOTTING HELPERS
# ============================================================

def register_figure(fig: plt.Figure) -> None:
    GENERATED_FIGURES.append(fig)


def finalize_figure(fig: plt.Figure, save_path: Path) -> None:
    ensure_dir(save_path.parent)
    fig.tight_layout()
    fig.savefig(to_windows_long_path(save_path), dpi=300)
    register_figure(fig)


# ============================================================
# PLOTTING
# ============================================================

def plot_loss_vs_epoch_log(history_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Training Loss")
    ax.plot(history_df["epoch"], history_df["test_loss"], label="Testing Loss")
    ax.set_title("Training and Testing Loss vs Epoch (Log Scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finalize_figure(fig, save_path)


def plot_loss_vs_epoch_zoomed(history_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Training Loss")
    ax.plot(history_df["epoch"], history_df["test_loss"], label="Testing Loss")
    ax.set_title("Training and Testing Loss vs Epoch (Zoomed Linear Scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    # Plot the full training history on the x-axis, but ignore the first few
    # epochs when setting the y-axis so the initial drop does not dominate.
    zoom_history_df = history_df.loc[history_df["epoch"] >= 5]
    if zoom_history_df.empty:
        zoom_history_df = history_df

    zoom_losses = np.concatenate([
        zoom_history_df["train_loss"].to_numpy(dtype=float),
        zoom_history_df["test_loss"].to_numpy(dtype=float),
    ])

    ymin = float(np.min(zoom_losses))
    ymax = float(np.max(zoom_losses))
    span = ymax - ymin
    scale = max(abs(ymin), abs(ymax), 1.0)
    lower_pad = 0.05 * span if span > 0 else 0.03 * scale
    upper_pad = 0.20 * span if span > 0 else 0.10 * scale
    ax.set_xlim(float(history_df["epoch"].min()), float(history_df["epoch"].max()))
    ax.set_ylim(ymin - lower_pad, ymax + upper_pad)

    ax.grid(True, alpha=0.3)
    ax.legend()
    finalize_figure(fig, save_path)


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variable_name: str,
    split_name: str,
    save_path: Path,
) -> Tuple[float, float, float]:
    r2 = compute_r2(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    envelope = compute_percentile_envelope(y_true, y_pred, percentile=98.0)

    data_min = float(min(np.min(y_true), np.min(y_pred)))
    data_max = float(max(np.max(y_true), np.max(y_pred)))
    span = data_max - data_min
    pad = 0.05 * span if span > 0 else 1.0

    lo = data_min - pad
    hi = data_max + pad
    xline = np.linspace(lo, hi, 300)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8)
    ax.plot(xline, xline, linestyle="--", label="Perfect Parity")
    ax.plot(xline, xline + envelope, linestyle=":", label="98th Pct. Abs. Error Band")
    ax.plot(xline, xline - envelope, linestyle=":")

    display_name = format_output_label(variable_name)

    ax.set_title(f"{split_name} Parity Plot - {display_name}")
    ax.set_xlabel(f"True {variable_name}")
    ax.set_ylabel(f"Predicted {variable_name}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    txt = (
        f"N = {len(y_true)}\n"
        f"R² = {r2:.5f}\n"
        f"RMSE = {rmse:.6f}\n"
        f"98th pct. abs. error = ±{envelope:.6f}"
    )
    ax.text(
        0.05,
        0.95,
        txt,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.2),
    )

    ax.legend(loc="lower right")

    finalize_figure(fig, save_path)
    return r2, rmse, envelope


def plot_combined_parity_grid(
    gt_train: np.ndarray,
    pred_train: np.ndarray,
    gt_test: np.ndarray,
    pred_test: np.ndarray,
    variable_names: List[str],
    save_path: Path,
) -> None:
    n_vars = len(variable_names)
    fig, axes = plt.subplots(
        nrows=n_vars,
        ncols=2,
        figsize=(13, 5.5 * n_vars),
        squeeze=False
    )

    for j, var_name in enumerate(variable_names):
        for col, (split_name, y_true, y_pred) in enumerate([
            ("Train", gt_train[:, j], pred_train[:, j]),
            ("Test", gt_test[:, j], pred_test[:, j]),
        ]):
            ax = axes[j, col]

            r2 = compute_r2(y_true, y_pred)
            rmse = compute_rmse(y_true, y_pred)
            envelope = compute_percentile_envelope(y_true, y_pred, percentile=98.0)

            data_min = float(min(np.min(y_true), np.min(y_pred)))
            data_max = float(max(np.max(y_true), np.max(y_pred)))
            span = data_max - data_min
            pad = 0.05 * span if span > 0 else 1.0

            lo = data_min - pad
            hi = data_max + pad
            xline = np.linspace(lo, hi, 300)

            ax.scatter(y_true, y_pred, alpha=0.25, s=8)
            ax.plot(xline, xline, linestyle="--", label="Perfect Parity")
            ax.plot(xline, xline + envelope, linestyle=":", label="98th Pct. Abs. Error Band")
            ax.plot(xline, xline - envelope, linestyle=":")

            display_name = format_output_label(var_name)

            ax.set_title(f"{split_name} - {display_name}")
            ax.set_xlabel(f"True {var_name}")
            ax.set_ylabel(f"Predicted {var_name}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")

            txt = (
                f"N = {len(y_true)}\n"
                f"R² = {r2:.5f}\n"
                f"RMSE = {rmse:.6f}\n"
                f"98th pct. abs. error = ±{envelope:.6f}"
            )
            ax.text(
                0.05,
                0.95,
                txt,
                transform=ax.transAxes,
                va="top",
                bbox=dict(boxstyle="round", alpha=0.2),
            )

            ax.legend(loc="lower right")

    finalize_figure(fig, save_path)


def plot_training_time_vs_accuracy(
    history_df: pd.DataFrame,
    total_training_time_sec: float | None,
    metrics_df: pd.DataFrame,
    save_path_epoch: Path,
    save_path_total: Path,
) -> None:
    if "epoch_time_sec" in history_df.columns and history_df["epoch_time_sec"].notna().any():
        epoch_df = history_df.copy()
        epoch_df["cumulative_time_sec"] = epoch_df["epoch_time_sec"].fillna(0.0).cumsum()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epoch_df["cumulative_time_sec"], epoch_df["test_loss"], label="Testing Loss")
        ax.set_title("Testing Loss vs Cumulative Training Time")
        ax.set_xlabel("Cumulative Training Time [s]")
        ax.set_ylabel("Testing Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        finalize_figure(fig, save_path_epoch)

    test_metrics = metrics_df[metrics_df["split"] == "test"]
    overall_r2 = float(test_metrics["r2"].mean()) if len(test_metrics) > 0 else float(metrics_df["r2"].mean())

    if total_training_time_sec is not None:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.bar(["Model"], [total_training_time_sec])
        ax1.set_ylabel("Total Training Time [s]")
        ax1.set_title("Total Training Time vs Overall Model Accuracy (R²)")
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(["Model"], [overall_r2], marker="o", linestyle="--")
        ax2.set_ylabel("Overall R²")

        finalize_figure(fig, save_path_total)


def show_all_figures() -> None:
    if SHOW_PLOTS and GENERATED_FIGURES:
        plt.show(block=True)


def run_for_output_dir(out_dir: Path, show_plots: bool = False) -> None:
    global OUT_DIR, TRAINING_LOG_FILE, MAT_DIR, FIG_DIR, METRICS_FILE, SHOW_PLOTS, GENERATED_FIGURES

    OUT_DIR = Path(out_dir)
    TRAINING_LOG_FILE = OUT_DIR / "training_log.txt"
    MAT_DIR = OUT_DIR / "matfiles"
    FIG_DIR = OUT_DIR / "figures"
    METRICS_FILE = OUT_DIR / "metrics_summary.csv"
    SHOW_PLOTS = show_plots
    GENERATED_FIGURES = []
    main()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dir(FIG_DIR)

    print("RUNNING SCRIPT:", __file__)
    print("TRAINING_LOG_FILE exists:", path_exists_long(TRAINING_LOG_FILE))
    print("OUT_DIR exists:", path_exists_long(OUT_DIR))
    print("MAT_DIR exists:", path_exists_long(MAT_DIR))

    if path_exists_long(MAT_DIR):
        print("MAT_DIR raw contents:")
        for name in os.listdir(str(MAT_DIR)):
            print("  ", repr(name))

    loss_mat_file = find_file_case_insensitive(MAT_DIR, "training_loss.mat")
    predict_mat_file = find_file_case_insensitive(MAT_DIR, "predict_train_n_test.mat")

    print("Resolved loss_mat_file:", repr(str(loss_mat_file)))
    print("Resolved predict_mat_file:", repr(str(predict_mat_file)))

    loss_mat = load_clean_mat(loss_mat_file)
    pred_mat = load_clean_mat(predict_mat_file)
    log_summary: Dict[str, float] = {}

    if path_exists_long(TRAINING_LOG_FILE):
        log_history_df, log_summary = parse_training_log(TRAINING_LOG_FILE)
    else:
        print(f"Warning: training log not found, continuing without it: {TRAINING_LOG_FILE}")
        log_history_df = pd.DataFrame()

    loss_df = extract_training_loss(loss_mat)

    if not log_history_df.empty and len(log_history_df) == len(loss_df):
        loss_df["epoch_time_sec"] = log_history_df["epoch_time_sec"].to_numpy()
        loss_df["elapsed_min"] = log_history_df["elapsed_min"].to_numpy()
    elif not log_history_df.empty:
        print("Warning: log length and training_loss.mat length do not match. Using losses from .mat only.")

    arrays = extract_prediction_arrays(pred_mat)

    pred_train = arrays["pred_train"]
    gt_train = arrays["gt_train"]
    pred_test = arrays["pred_test"]
    gt_test = arrays["gt_test"]

    print("pred_train shape:", pred_train.shape)
    print("gt_train shape:", gt_train.shape)
    print("pred_test shape:", pred_test.shape)
    print("gt_test shape:", gt_test.shape)

    n_outputs = gt_train.shape[1]
    variable_names = get_output_variable_names(n_outputs)

    print("Input variable labels:", INPUT_VARIABLE_NAMES)
    print("Number of output variables:", n_outputs)
    print("Output variable labels used for parity plots:", variable_names)

    plot_loss_vs_epoch_log(loss_df, FIG_DIR / "loss_vs_epoch_log.png")
    plot_loss_vs_epoch_zoomed(loss_df, FIG_DIR / "loss_vs_epoch_zoomed.png")

    metrics_rows = []

    for j, var_name in enumerate(variable_names):
        print(f"{var_name}: train points = {len(gt_train[:, j])}, test points = {len(gt_test[:, j])}")

        r2, rmse, env = plot_parity(
            gt_train[:, j],
            pred_train[:, j],
            var_name,
            "Train",
            FIG_DIR / f"parity_train_{var_name}.png",
        )
        metrics_rows.append({
            "split": "train",
            "variable": var_name,
            "r2": r2,
            "rmse": rmse,
            "envelope_98_abs_error": env,
        })

        r2, rmse, env = plot_parity(
            gt_test[:, j],
            pred_test[:, j],
            var_name,
            "Test",
            FIG_DIR / f"parity_test_{var_name}.png",
        )
        metrics_rows.append({
            "split": "test",
            "variable": var_name,
            "r2": r2,
            "rmse": rmse,
            "envelope_98_abs_error": env,
        })

    metrics_df = pd.DataFrame(metrics_rows)
    ensure_dir(METRICS_FILE.parent)
    metrics_df.to_csv(to_windows_long_path(METRICS_FILE), index=False)

    plot_combined_parity_grid(
        gt_train=gt_train,
        pred_train=pred_train,
        gt_test=gt_test,
        pred_test=pred_test,
        variable_names=variable_names,
        save_path=FIG_DIR / "combined_parity_grid.png",
    )

    plot_training_time_vs_accuracy(
        history_df=loss_df,
        total_training_time_sec=log_summary.get("total_training_time_sec"),
        metrics_df=metrics_df,
        save_path_epoch=FIG_DIR / "test_loss_vs_cumulative_training_time.png",
        save_path_total=FIG_DIR / "total_training_time_vs_overall_r2.png",
    )

    print("\nDone.")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Metrics saved to: {METRICS_FILE}")

    if "best_test_loss" in log_summary:
        print(
            f"Best test loss from log: {log_summary['best_test_loss']:.6f} "
            f"at epoch {int(log_summary['best_test_epoch'])}"
        )

    print("\nMetrics summary:")
    print(metrics_df.to_string(index=False))

    show_all_figures()


if __name__ == "__main__":
    main()
