from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.io import loadmat


OUTPUT_VARIABLE_NAMES = ["CF_x", "CF_y", "CM_x", "CM_y", "CM_z"]
OUTPUT_VARIABLE_DESCRIPTIONS = {
    "CF_x": "normal force coefficient",
    "CF_y": "chordwise force coefficient",
    "CM_x": "normal moment coefficient",
    "CM_y": "chordwise moment coefficient",
    "CM_z": "spanwise moment coefficient",
}
INPUT_ARROW_COLORS = {
    "AoA": "#7b6cff",
    "|v|": "#1f77b4",
    "a_x": "#2a9d8f",
    "a_y": "#2a9d8f",
    "a_z": "#2a9d8f",
    "alpha_dot": "#bc6c25",
    "alpha_ddot": "#a4243b",
}
OUTPUT_ARROW_COLORS = {
    "CF_x": "#00876c",
    "CF_y": "#00a6ca",
    "CM_x": "#ef476f",
    "CM_y": "#f4a261",
    "CM_z": "#8338ec",
}
PREDICTED_LINE_COLOR = "#000000"
INPUT_VARIABLE_NAMES = [
    "AoA",
    "|v|",
    "a_x",
    "a_y",
    "a_z",
    "alpha_dot",
    "alpha_ddot",
]
INPUT_VARIABLE_DESCRIPTIONS = {
    "AoA": "angle of attack",
    "|v|": "wing speed magnitude",
    "a_x": "x-acceleration in the wing frame",
    "a_y": "y-acceleration in the wing frame",
    "a_z": "z-acceleration in the wing frame",
    "alpha_dot": "wing pitch rate",
    "alpha_ddot": "wing pitch acceleration",
}
POSITION_NAMES = ["stroke", "deviation", "rotation"]
INPUT_INDEX = {name: i for i, name in enumerate(INPUT_VARIABLE_NAMES)}
OUTPUT_INDEX = {name: i for i, name in enumerate(OUTPUT_VARIABLE_NAMES)}
AXIS_LABELS = {
    "x": "stroke plane direction",
    "y": "deviation direction",
    "z": "span / tip direction",
}
SPAN_LENGTH = 1.0
CHORD_LENGTH = 0.34
AXIS_OF_ROTATION_FRAC = 0.15
CATEGORY_ORDER = [
    "All",
    "High stroke",
    "High deviation",
    "Early rotation",
    "Late rotation",
    "High speed",
    "Easy prediction",
    "Hard prediction",
]
CATEGORY_RULES = {
    "All": "shows every trajectory in the selected split",
    "High stroke": "stroke amplitude is in the top third of the selected split",
    "High deviation": "deviation amplitude is in the top third of the selected split",
    "Early rotation": "peak |rotation| happens in the first third of the cycle",
    "Late rotation": "peak |rotation| happens in the last third of the cycle",
    "High speed": "mean |v| is in the top third of the selected split",
    "Easy prediction": "trajectory RMSE is in the lowest third of the selected split",
    "Hard prediction": "trajectory RMSE is in the highest third of the selected split",
}
FILTER_METRIC_ORDER = [
    "Stroke amplitude [deg]",
    "Deviation amplitude [deg]",
    "Rotation amplitude [deg]",
    "Mean AoA",
    "Mean |v|",
    "Peak |v|",
    "Mean CF_x",
    "Mean CF_y",
    "RMSE (all outputs)",
]
FILTER_METRIC_DISPLAY = {
    "Stroke amplitude [deg]": "Stroke ampl [deg]",
    "Deviation amplitude [deg]": "Deviation ampl [deg]",
    "Rotation amplitude [deg]": "Rotation ampl [deg]",
    "Mean AoA": "Mean AoA",
    "Mean |v|": "Mean |v|",
    "Peak |v|": "Peak |v|",
    "Mean CF_x": "Mean CF_x",
    "Mean CF_y": "Mean CF_y",
    "RMSE (all outputs)": "RMSE all outputs",
}
FILTER_METRIC_RULES = {
    "Stroke amplitude [deg]": "peak-to-peak stroke angle across the trajectory",
    "Deviation amplitude [deg]": "peak-to-peak deviation angle across the trajectory",
    "Rotation amplitude [deg]": "peak-to-peak wing pitch rotation across the trajectory",
    "Mean AoA": "average angle of attack over the trajectory",
    "Mean |v|": "average wing-speed magnitude over the trajectory",
    "Peak |v|": "maximum wing-speed magnitude over the trajectory",
    "Mean CF_x": "average normal force coefficient over the trajectory",
    "Mean CF_y": "average chordwise force coefficient over the trajectory",
    "RMSE (all outputs)": "root-mean-square prediction error across all outputs",
}
SLIDER_LABELS = {
    "frame": "Frame",
    "trajectory": "Trajectory",
    "speed": "Speed",
    "filter_min": "Metric Min",
    "filter_max": "Metric Max",
}


def to_windows_long_path(path: Path) -> str:
    abs_path = str(path.resolve())
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


def load_clean_mat(path: Path) -> Dict[str, np.ndarray]:
    if not path_exists_long(path):
        raise FileNotFoundError(
            f"Missing MATLAB file: {path}\n"
            f"Expected a model output folder containing a prediction MAT file."
        )
    raw = loadmat(to_windows_long_path(path))
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def list_mat_files_long(folder: Path) -> List[Path]:
    if not path_exists_long(folder):
        return []

    try:
        names = os.listdir(to_windows_long_path(folder))
    except (FileNotFoundError, OSError):
        return []

    mat_files = [folder / name for name in names if name.lower().endswith(".mat")]
    return sorted(mat_files)


def denormalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=float).reshape(1, 1, -1)
    std = np.asarray(std, dtype=float).reshape(1, 1, -1)
    return arr * std + mean


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros_like(vec)
    return vec / norm


def get_output_roots() -> List[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "Data" / "Output",
        repo_root.parent / "Data" / "Output",
    ]


def build_default_out_dir(out_folder_name: str) -> Path:
    for root in get_output_roots():
        candidate = root / out_folder_name
        if candidate.exists():
            return candidate
    return get_output_roots()[0] / out_folder_name


def has_prediction_mat_file(out_dir: Path) -> bool:
    mat_dir = out_dir / "matfiles"
    if not path_exists_long(mat_dir):
        return False

    mat_files = list_mat_files_long(mat_dir)
    for path in mat_files:
        name = path.name.lower()
        if "predict" in name or ("train" in name and "test" in name):
            return True

    return bool(mat_files)


def list_available_output_dirs() -> List[Path]:
    candidates: List[Path] = []
    seen = set()
    for output_root in get_output_roots():
        if not output_root.exists():
            continue
        for child in sorted(output_root.iterdir()):
            if not child.is_dir():
                continue
            if has_prediction_mat_file(child):
                resolved = str(child.resolve())
                if resolved not in seen:
                    candidates.append(child)
                    seen.add(resolved)
    return candidates


def choose_output_dir_interactive() -> Path:
    candidates = list_available_output_dirs()
    if not candidates:
        raise FileNotFoundError(
            "No output folders with usable MAT files were found under:\n"
            + "\n".join(str(root) for root in get_output_roots())
        )

    print("Available output folders with prediction files:")
    for idx, path in enumerate(candidates, start=1):
        print(f"  {idx}. {path.name}")
        print(f"     {path}")

    if len(candidates) == 1:
        print(f"\nOnly one valid folder was found. Press Enter to use: {candidates[0].name}")

    while True:
        raw = input(
            "\nChoose by number or exact folder name"
            + (f" [default: {candidates[0].name}]" if len(candidates) == 1 else "")
            + ": "
        ).strip()

        if raw == "" and len(candidates) == 1:
            return candidates[0]

        for path in candidates:
            if raw.lower() == path.name.lower():
                return path

        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a listed number or folder name.")
            continue

        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]

        print("Selection out of range.")


def list_prediction_mat_files(out_dir: Path) -> List[Path]:
    mat_dir = out_dir / "matfiles"
    if not path_exists_long(mat_dir):
        return []

    mat_files = list_mat_files_long(mat_dir)
    preferred = [
        p for p in mat_files
        if ("predict" in p.name.lower()) and ("train" in p.name.lower() or "test" in p.name.lower())
    ]
    if preferred:
        return preferred
    return mat_files


def choose_prediction_file_interactive(out_dir: Path) -> Path:
    candidates = list_prediction_mat_files(out_dir)
    if not candidates:
        raise FileNotFoundError(f"No .mat files were found in {out_dir / 'matfiles'}.")

    print(f"\nAvailable MAT files in {out_dir.name}:")
    for idx, path in enumerate(candidates, start=1):
        print(f"  {idx}. {path.name}")

    if len(candidates) == 1:
        print(f"\nOnly one MAT file was found. Press Enter to use: {candidates[0].name}")

    while True:
        raw = input(
            "Choose MAT file by number or exact filename"
            + (f" [default: {candidates[0].name}]" if len(candidates) == 1 else "")
            + ": "
        ).strip()

        if raw == "" and len(candidates) == 1:
            return candidates[0]

        for path in candidates:
            if raw.lower() == path.name.lower():
                return path

        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a listed number or file name.")
            continue

        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]

        print("Selection out of range.")


def classify_tertile(values: np.ndarray, low_label: str, mid_label: str, high_label: str) -> List[str]:
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    labels: List[str] = []
    for value in values:
        if value <= q1:
            labels.append(low_label)
        elif value >= q2:
            labels.append(high_label)
        else:
            labels.append(mid_label)
    return labels


def build_category_data(
    pos: np.ndarray,
    inputs_raw: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    n_traj, n_steps, _ = pos.shape
    stroke_amp = np.degrees(np.ptp(pos[:, :, 0], axis=1))
    deviation_amp = np.degrees(np.ptp(pos[:, :, 1], axis=1))
    rotation_phase = np.argmax(np.abs(pos[:, :, 2]), axis=1) / max(n_steps - 1, 1)
    mean_speed = np.mean(np.abs(inputs_raw[:, :, 1]), axis=1)
    traj_rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=(1, 2)))

    stroke_q = np.quantile(stroke_amp, 2 / 3)
    deviation_q = np.quantile(deviation_amp, 2 / 3)
    phase_low, phase_high = np.quantile(rotation_phase, [1 / 3, 2 / 3])
    speed_q = np.quantile(mean_speed, 2 / 3)
    rmse_low, rmse_high = np.quantile(traj_rmse, [1 / 3, 2 / 3])

    categories = {
        "All": np.arange(n_traj, dtype=int),
        "High stroke": np.where(stroke_amp >= stroke_q)[0],
        "High deviation": np.where(deviation_amp >= deviation_q)[0],
        "Early rotation": np.where(rotation_phase <= phase_low)[0],
        "Late rotation": np.where(rotation_phase >= phase_high)[0],
        "High speed": np.where(mean_speed >= speed_q)[0],
        "Easy prediction": np.where(traj_rmse <= rmse_low)[0],
        "Hard prediction": np.where(traj_rmse >= rmse_high)[0],
    }

    stroke_labels = classify_tertile(stroke_amp, "low stroke", "medium stroke", "high stroke")
    deviation_labels = classify_tertile(deviation_amp, "low deviation", "medium deviation", "high deviation")
    speed_labels = classify_tertile(mean_speed, "low speed", "medium speed", "high speed")
    rmse_labels = classify_tertile(traj_rmse, "easy prediction", "moderate prediction", "hard prediction")
    phase_labels: List[str] = []
    for value in rotation_phase:
        if value <= phase_low:
            phase_labels.append("early rotation")
        elif value >= phase_high:
            phase_labels.append("late rotation")
        else:
            phase_labels.append("mid-cycle rotation")

    summaries = [
        f"{stroke_labels[i]}, {deviation_labels[i]}, {phase_labels[i]}, {speed_labels[i]}, {rmse_labels[i]}"
        for i in range(n_traj)
    ]
    return categories, summaries


def build_filter_metrics(
    pos: np.ndarray,
    inputs_raw: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "Stroke amplitude [deg]": np.degrees(np.ptp(pos[:, :, 0], axis=1)),
        "Deviation amplitude [deg]": np.degrees(np.ptp(pos[:, :, 1], axis=1)),
        "Rotation amplitude [deg]": np.degrees(np.ptp(pos[:, :, 2], axis=1)),
        "Mean AoA": np.mean(inputs_raw[:, :, 0], axis=1),
        "Mean |v|": np.mean(np.abs(inputs_raw[:, :, 1]), axis=1),
        "Peak |v|": np.max(np.abs(inputs_raw[:, :, 1]), axis=1),
        "Mean CF_x": np.mean(truth[:, :, 0], axis=1),
        "Mean CF_y": np.mean(truth[:, :, 1], axis=1),
        "RMSE (all outputs)": np.sqrt(np.mean((pred - truth) ** 2, axis=(1, 2))),
    }


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.eye(3)
    axis = axis / axis_norm
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def wing_geometry(stroke: float, deviation: float, rotation: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    span_dir = np.array([
        np.cos(deviation) * np.sin(stroke),
        -np.sin(deviation),
        np.cos(deviation) * np.cos(stroke),
    ])
    span_dir /= np.linalg.norm(span_dir)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(span_dir, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])

    chord_seed = np.cross(world_up, span_dir)
    chord_seed /= np.linalg.norm(chord_seed)
    chord_dir = rotation_matrix(span_dir, rotation) @ chord_seed
    chord_dir /= np.linalg.norm(chord_dir)

    hinge = np.zeros(3)
    tip = hinge + SPAN_LENGTH * span_dir
    leading = tip + AXIS_OF_ROTATION_FRAC * CHORD_LENGTH * chord_dir
    trailing = tip - (1.0 - AXIS_OF_ROTATION_FRAC) * CHORD_LENGTH * chord_dir
    return hinge, leading, trailing


class SplitData:
    def __init__(
        self,
        pos: np.ndarray,
        inputs_norm: np.ndarray,
        inputs_raw: np.ndarray,
        truth: np.ndarray,
        pred: np.ndarray,
        n_traj: int,
        n_steps: int,
        category_indices: Dict[str, np.ndarray],
        category_summaries: List[str],
        filter_metrics: Dict[str, np.ndarray],
    ) -> None:
        self.pos = pos
        self.inputs_norm = inputs_norm
        self.inputs_raw = inputs_raw
        self.truth = truth
        self.pred = pred
        self.n_traj = n_traj
        self.n_steps = n_steps
        self.category_indices = category_indices
        self.category_summaries = category_summaries
        self.filter_metrics = filter_metrics


class WingTrajectoryAnimator:
    def __init__(
        self,
        out_dir: Path,
        mat_file: Path,
        split: str = "test",
        traj_index: int = 0,
        speed: float = 1.0,
    ):
        self.out_dir = out_dir
        self.mat_dir = out_dir / "matfiles"
        self.mat_file = mat_file
        self.mat = load_clean_mat(self.mat_file)
        self.split = split
        self.speed = speed
        self.trace_mode = "2D Full Traces"
        self.category_name = "All"
        self.filter_metric_name = FILTER_METRIC_ORDER[0]
        self.filter_min = None
        self.filter_max = None
        self.tip_tracer_mode = "3D Tip Tracer ON"
        self.extrema_mode = "Extrema OFF"
        self.error_metrics_mode = "Errors OFF"
        self.instant_error_mode = "Inst Error OFF"
        self.input_arrows_mode = "Input Arrows OFF"
        self.output_arrows_mode = "Output Arrows OFF"
        self.legend_popup_fig = None
        self.input_selector_popup_fig = None
        self.output_selector_popup_fig = None
        self.input_selector_checks = None
        self.output_selector_checks = None
        self.input_selector_check_texts = []
        self.output_selector_check_texts = []
        self.selected_input_arrow_vars = set(INPUT_VARIABLE_NAMES)
        self.selected_output_arrow_vars = set(OUTPUT_VARIABLE_NAMES)
        self.paused = True
        self.frame_idx = 0
        self.frame_float = 0.0
        self.last_slider_frame = 0
        self.playback_dt = 0.004

        self.splits = self._prepare_splits()
        self.current = self.splits[self.split]
        self.traj_index = int(np.clip(traj_index, 0, self.current.n_traj - 1))
        self.filtered_indices = np.arange(self.current.n_traj, dtype=int)

        self.fig = plt.figure(figsize=(17.4, 10.4), constrained_layout=False, facecolor="#f4f1ea")
        self.fig.subplots_adjust(left=0.028, right=0.988, top=0.89, bottom=0.055)
        self.gs = GridSpec(1, 3, figure=self.fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.08)
        self._build_figure()
        self._load_trajectory(self.traj_index)
        self._update_frame(0)

    def _prepare_splits(self) -> Dict[str, SplitData]:
        mean_in = np.asarray(self.mat["mean_in"], dtype=float).reshape(-1)
        std_in = np.asarray(self.mat["std_in"], dtype=float).reshape(-1)
        mean_out = np.asarray(self.mat["mean_out"], dtype=float).reshape(-1)
        std_out = np.asarray(self.mat["std_out"], dtype=float).reshape(-1)

        splits: Dict[str, SplitData] = {}
        for split in ["train", "test"]:
            pos = np.asarray(self.mat[f"pos_{split}"], dtype=float)
            inputs_norm = np.asarray(self.mat[f"in_{split}"], dtype=float)
            truth_norm = np.asarray(self.mat[f"gt_{split}"], dtype=float)
            pred_norm = np.asarray(self.mat[f"gp_mean_{split}"], dtype=float)
            inputs_raw = denormalize(inputs_norm, mean_in, std_in)
            truth = denormalize(truth_norm, mean_out, std_out)
            pred = denormalize(pred_norm, mean_out, std_out)
            category_indices, category_summaries = build_category_data(pos, inputs_raw, truth, pred)
            filter_metrics = build_filter_metrics(pos, inputs_raw, truth, pred)

            splits[split] = SplitData(
                pos=pos,
                inputs_norm=inputs_norm,
                inputs_raw=inputs_raw,
                truth=truth,
                pred=pred,
                n_traj=pos.shape[0],
                n_steps=pos.shape[1],
                category_indices=category_indices,
                category_summaries=category_summaries,
                filter_metrics=filter_metrics,
            )
        return splits

    def _build_figure(self) -> None:
        self.fig.suptitle("Flapping Wing Trajectory Animation", fontsize=18, y=0.975, fontweight="bold")
        self.header_text = self.fig.text(
            0.5,
            0.924,
            "",
            fontsize=8.6,
            ha="center",
            va="center",
            color="#5b5345",
        )
        left_gs = self.gs[0, 0].subgridspec(
            12,
            1,
            height_ratios=[1, 1, 1, 1, 1, 1, 0.75, 0.9, 0.9, 0.9, 0.52, 0.12],
            hspace=0.48,
        )
        center_gs = self.gs[0, 1].subgridspec(
            13,
            1,
            height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.18, 0.98, 1.0],
            hspace=0.74,
        )
        right_gs = self.gs[0, 2].subgridspec(
            18,
            2,
            width_ratios=[1.0, 1.0],
            height_ratios=[
                0.86,
                0.86,
                0.26,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.30,
                1.0,
                1.0,
                1.0,
                1.0,
                0.86,
                0.56,
                0.56,
                0.62,
                0.62,
            ],
            hspace=0.60,
            wspace=0.18,
        )

        self.ax3d = self.fig.add_subplot(left_gs[:6, 0], projection="3d")
        self.ax3d.set_title("Wing Motion in Wing Frame", pad=20, fontsize=13, fontweight="bold")
        self.ax3d.set_xlabel("x_w (stroke axis)", labelpad=4)
        self.ax3d.set_ylabel("y_w (deviation axis)", labelpad=4)
        self.ax3d.set_zlabel("z_w (span axis)", labelpad=4)
        self.ax3d.tick_params(labelsize=8.5, pad=1)
        self.ax3d.view_init(elev=24, azim=-118)
        self.ax3d.set_box_aspect((1.2, 1.0, 1.0))
        self.ax3d.grid(False)
        self.ax3d.set_facecolor("#fbfaf7")
        self.ax3d.xaxis.pane.set_facecolor((0.97, 0.96, 0.93, 1.0))
        self.ax3d.yaxis.pane.set_facecolor((0.97, 0.96, 0.93, 1.0))
        self.ax3d.zaxis.pane.set_facecolor((0.98, 0.97, 0.95, 1.0))

        self.hinge_trace, = self.ax3d.plot([], [], [], color="#1f77b4", alpha=0.35, linewidth=1.5)
        self.wing_span, = self.ax3d.plot([], [], [], color="black", linewidth=3.0)
        self.leading_edge, = self.ax3d.plot([], [], [], color="#c44e52", linewidth=2.0)
        self.trailing_edge, = self.ax3d.plot([], [], [], color="#4c72b0", linewidth=2.0)
        self.stroke_history, = self.ax3d.plot([], [], [], color="#7f7f7f", alpha=0.25, linewidth=1.0)
        self.leading_tip_trace, = self.ax3d.plot([], [], [], color="#c44e52", alpha=0.5, linewidth=1.2, linestyle="--")
        self.trailing_tip_trace, = self.ax3d.plot([], [], [], color="#4c72b0", alpha=0.5, linewidth=1.2, linestyle="--")
        self.leading_tip_marker, = self.ax3d.plot([], [], [], marker="o", color="#c44e52", markersize=5)
        self.trailing_tip_marker, = self.ax3d.plot([], [], [], marker="o", color="#4c72b0", markersize=5)
        self.wing_patch = Poly3DCollection([], facecolors="#d9d3c3", edgecolors="black", linewidths=1.0, alpha=0.65)
        self.ax3d.add_collection3d(self.wing_patch)
        self.input_arrow_artists: List = []
        self.output_arrow_artists: List = []

        self.ax_details = self.fig.add_subplot(left_gs[7:10, 0])
        self.ax_details.axis("off")
        self.ax_details.set_facecolor("#fffdf8")
        self.ax_details.set_title("Selection Details", fontsize=10.5, fontweight="bold", pad=5)
        self.info_text = self.ax_details.text(
            0.02,
            0.98,
            "",
            va="top",
            ha="left",
            fontsize=7.2,
            family="monospace",
            linespacing=1.02,
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffdf8", edgecolor="#d8d2c2"),
        )

        self.output_axes = []
        self.truth_lines = []
        self.pred_lines = []
        self.cursor_lines = []
        self.actual_markers = []
        self.pred_markers = []
        self.metric_texts = []
        self.instant_error_texts = []
        self.extrema_markers = []
        self.extrema_texts = []
        for i, name in enumerate(OUTPUT_VARIABLE_NAMES):
            ax = self.fig.add_subplot(center_gs[2 * i:2 * i + 2, 0])
            ax.set_title(
                f"{name} ({OUTPUT_VARIABLE_DESCRIPTIONS[name]})",
                fontsize=9.4,
                pad=7,
                fontweight="bold",
            )
            ax.set_ylabel(f"{name} [-]", fontsize=8.2, labelpad=7)
            ax.grid(True, alpha=0.22, color="#b8b1a1")
            ax.set_xlim(0.0, 1.0)
            ax.set_facecolor("#fffdf8")
            ax.tick_params(labelsize=8.2, pad=1.5)
            for spine in ax.spines.values():
                spine.set_color("#d8d2c2")
            line_color = OUTPUT_ARROW_COLORS[name]
            truth_line, = ax.plot([], [], color=line_color, linewidth=2.0, label="Measured")
            pred_line, = ax.plot([], [], color=PREDICTED_LINE_COLOR, linewidth=1.8, linestyle="--", label="Predicted")
            cursor = ax.axvline(0.0, color="black", alpha=0.25, linewidth=1.0)
            actual_marker, = ax.plot([], [], marker="o", color=line_color, markersize=5)
            pred_marker, = ax.plot([], [], marker="o", color=PREDICTED_LINE_COLOR, markersize=5, markerfacecolor="white")
            if i != len(OUTPUT_VARIABLE_NAMES) - 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Cycle time [-]", labelpad=3, fontsize=9.0)
            self.output_axes.append(ax)
            self.truth_lines.append(truth_line)
            self.pred_lines.append(pred_line)
            self.cursor_lines.append(cursor)
            self.actual_markers.append(actual_marker)
            self.pred_markers.append(pred_marker)
            self.metric_texts.append(
                ax.text(
                    0.985,
                    0.06,
                    "",
                    transform=ax.transAxes,
                    fontsize=7.0,
                    color="#332f26",
                    ha="right",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#fffdf8", edgecolor="#d8d2c2", alpha=0.92),
                )
            )
            self.instant_error_texts.append(
                ax.text(
                    0.015,
                    0.06,
                    "",
                    transform=ax.transAxes,
                    fontsize=7.0,
                    color="#332f26",
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#fffdf8", edgecolor="#d8d2c2", alpha=0.92),
                    visible=False,
                )
            )
            truth_extrema_color = line_color
            pred_extrema_color = PREDICTED_LINE_COLOR
            extrema_markers = [
                ax.plot([], [], marker="^", color=truth_extrema_color, markersize=5, linestyle="None", visible=False)[0],
                ax.plot([], [], marker="v", color=truth_extrema_color, markersize=5, linestyle="None", visible=False)[0],
                ax.plot([], [], marker="^", color=pred_extrema_color, markersize=5, linestyle="None", visible=False)[0],
                ax.plot([], [], marker="v", color=pred_extrema_color, markersize=5, linestyle="None", visible=False)[0],
            ]
            extrema_texts = [
                ax.text(0.0, 0.0, "", fontsize=7.0, color=truth_extrema_color, visible=False, ha="left", va="bottom"),
                ax.text(0.0, 0.0, "", fontsize=7.0, color=truth_extrema_color, visible=False, ha="left", va="top"),
                ax.text(0.0, 0.0, "", fontsize=7.0, color=pred_extrema_color, visible=False, ha="right", va="bottom"),
                ax.text(0.0, 0.0, "", fontsize=7.0, color=pred_extrema_color, visible=False, ha="right", va="top"),
            ]
            self.extrema_markers.append(extrema_markers)
            self.extrema_texts.append(extrema_texts)

        frame_gs = center_gs[11:13, 0].subgridspec(2, 1, height_ratios=[0.72, 0.28], hspace=0.16)
        self.ax_frame = self.fig.add_subplot(frame_gs[0, 0])
        self.ax_plot_legend = self.fig.add_subplot(frame_gs[1, 0])
        self.ax_plot_legend.axis("off")
        self.ax_plot_legend.set_facecolor("#f4f1ea")
        self.ax_plot_legend.legend(
            [self.truth_lines[0], self.pred_lines[0]],
            ["Measured", "Predicted"],
            loc="center",
            bbox_to_anchor=(0.5, 0.22),
            ncol=2,
            frameon=True,
            facecolor="#fffdf8",
            edgecolor="#d8d2c2",
            fontsize=9,
        )

        self.ax_split = self.fig.add_subplot(right_gs[0:2, 0])
        self.ax_category = self.fig.add_subplot(right_gs[3:7, 0])
        self.ax_metric = self.fig.add_subplot(right_gs[8:15, 0])
        self.ax_traj = self.fig.add_subplot(right_gs[0:2, 1])
        self.ax_traj_box = self.fig.add_subplot(right_gs[2:4, 1])
        self.ax_speed = self.fig.add_subplot(right_gs[5:7, 1])
        self.ax_filter_min = self.fig.add_subplot(right_gs[7:9, 1])
        self.ax_filter_max = self.fig.add_subplot(right_gs[10:12, 1])
        self.ax_min_box = self.fig.add_subplot(right_gs[12:13, 1])
        self.ax_max_box = self.fig.add_subplot(right_gs[13:14, 1])
        action_gs = right_gs[15:18, :].subgridspec(4, 3, wspace=0.18, hspace=0.18)
        self.ax_var_legend = self.fig.add_subplot(action_gs[0, 0])
        self.ax_tracer = self.fig.add_subplot(action_gs[0, 1])
        self.ax_input_arrows = self.fig.add_subplot(action_gs[0, 2])
        self.ax_output_arrows = self.fig.add_subplot(action_gs[1, 0])
        self.ax_extrema = self.fig.add_subplot(action_gs[1, 1])
        self.ax_errors = self.fig.add_subplot(action_gs[1, 2])
        self.ax_instant_error = self.fig.add_subplot(action_gs[2, 0])
        self.ax_play = self.fig.add_subplot(action_gs[2, 1])
        self.ax_reset = self.fig.add_subplot(action_gs[2, 2])
        self.ax_trace = self.fig.add_subplot(action_gs[3, 1])

        self.radio_split = RadioButtons(self.ax_split, ("test", "train"), active=0 if self.split == "test" else 1)
        self._style_control_axis(self.ax_split, "Split")
        self._style_radio_buttons(self.radio_split, label_size=6.9, marker_size=135)

        self.radio_category = RadioButtons(
            self.ax_category,
            CATEGORY_ORDER,
            active=CATEGORY_ORDER.index(self.category_name),
        )
        self._style_control_axis(self.ax_category, "Category")
        self._style_radio_buttons(self.radio_category, label_size=6.9)

        metric_display_order = [FILTER_METRIC_DISPLAY[name] for name in FILTER_METRIC_ORDER]
        self.radio_metric = RadioButtons(
            self.ax_metric,
            metric_display_order,
            active=FILTER_METRIC_ORDER.index(self.filter_metric_name),
        )
        self._style_control_axis(self.ax_metric, "Filter Metric")
        self._style_radio_buttons(self.radio_metric, label_size=6.9)
        self.metric_range_text = self.ax_metric.text(
            0.03,
            0.02,
            "",
            transform=self.ax_metric.transAxes,
            fontsize=6.2,
            color="#5b5345",
            va="bottom",
            ha="left",
        )

        self.slider_traj = Slider(self.ax_traj, "", 0, max(len(self.filtered_indices) - 1, 0), valinit=0, valstep=1)
        self.slider_speed = Slider(self.ax_speed, "", 0.25, 7.0, valinit=min(self.speed, 7.0), valstep=0.25)
        self.slider_frame = Slider(self.ax_frame, "", 0, max(self.current.n_steps - 1, 0), valinit=0, valstep=1)
        self.slider_filter_min = Slider(self.ax_filter_min, "", 0.0, 1.0, valinit=0.0)
        self.slider_filter_max = Slider(self.ax_filter_max, "", 0.0, 1.0, valinit=1.0)
        self.textbox_traj = TextBox(self.ax_traj_box, "Trajectory", initial="")
        self.textbox_min = TextBox(self.ax_min_box, "Metric Min", initial="")
        self.textbox_max = TextBox(self.ax_max_box, "Metric Max", initial="")
        self.button_play = Button(self.ax_play, "Play")
        self.button_reset = Button(self.ax_reset, "Reset")
        self.button_trace = Button(self.ax_trace, self.trace_mode)
        self.button_extrema = Button(self.ax_extrema, self.extrema_mode)
        self.button_errors = Button(self.ax_errors, self.error_metrics_mode)
        self.button_instant_error = Button(self.ax_instant_error, self.instant_error_mode)
        self.button_var_legend = Button(self.ax_var_legend, "Var Legend")
        self.button_tracer = Button(self.ax_tracer, self.tip_tracer_mode)
        self.button_input_arrows = Button(self.ax_input_arrows, self.input_arrows_mode)
        self.button_output_arrows = Button(self.ax_output_arrows, self.output_arrows_mode)
        self._style_slider(self.slider_frame, SLIDER_LABELS["frame"])
        self._style_slider(self.slider_traj, SLIDER_LABELS["trajectory"])
        self._style_slider(self.slider_speed, SLIDER_LABELS["speed"])
        self._style_slider(self.slider_filter_min, SLIDER_LABELS["filter_min"])
        self._style_slider(self.slider_filter_max, SLIDER_LABELS["filter_max"])
        self._style_textbox(self.textbox_traj, label_size=7.0)
        self._style_button(self.button_play)
        self._style_button(self.button_reset)
        self._style_button(self.button_trace)
        self._style_button(self.button_extrema)
        self._style_button(self.button_errors)
        self._style_button(self.button_instant_error)
        self._style_button(self.button_var_legend)
        self._style_button(self.button_tracer)
        self._style_button(self.button_input_arrows)
        self._style_button(self.button_output_arrows)
        self._style_textbox(self.textbox_min, label_size=6.2)
        self._style_textbox(self.textbox_max, label_size=6.2)

        self.radio_split.on_clicked(self._on_split_change)
        self.radio_category.on_clicked(self._on_category_change)
        self.radio_metric.on_clicked(self._on_metric_change)
        self.slider_traj.on_changed(self._on_traj_change)
        self.slider_speed.on_changed(self._on_speed_change)
        self.slider_frame.on_changed(self._on_frame_change)
        self.slider_filter_min.on_changed(self._on_filter_slider_change)
        self.slider_filter_max.on_changed(self._on_filter_slider_change)
        self.textbox_traj.on_submit(self._on_traj_text_submit)
        self.textbox_min.on_submit(self._on_filter_text_submit)
        self.textbox_max.on_submit(self._on_filter_text_submit)
        self.button_play.on_clicked(self._toggle_pause)
        self.button_reset.on_clicked(self._reset_animation)
        self.button_trace.on_clicked(self._toggle_trace_mode)
        self.button_extrema.on_clicked(self._toggle_extrema_mode)
        self.button_errors.on_clicked(self._toggle_error_metrics_mode)
        self.button_instant_error.on_clicked(self._toggle_instant_error_mode)
        self.button_var_legend.on_clicked(self._show_variable_legend)
        self.button_tracer.on_clicked(self._toggle_tracer_mode)
        self.button_input_arrows.on_clicked(self._toggle_input_arrows_mode)
        self.button_output_arrows.on_clicked(self._toggle_output_arrows_mode)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._reset_metric_filter_controls()
        self._refresh_info_panel()

    def _style_control_axis(self, ax, title: str) -> None:
        ax.set_facecolor("#fffdf8")
        ax.set_title(title, fontsize=8.7, pad=4, fontweight="bold")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d8d2c2")

    def _style_radio_buttons(self, radio: RadioButtons, label_size: float, marker_size: float = 120.0) -> None:
        for label in radio.labels:
            label.set_fontsize(label_size)
            label.set_color("#332f26")

        if hasattr(radio, "_buttons"):
            radio._buttons.set_sizes([marker_size] * len(radio.labels))
            radio._buttons.set_linewidths([1.0] * len(radio.labels))
            radio._buttons.set_edgecolors(["#1f1f1f"] * len(radio.labels))
            radio._buttons.set_zorder(3)

        if hasattr(radio, "activecolor"):
            radio.activecolor = "#1a1aff"

    def _style_slider(self, slider: Slider, title: str) -> None:
        slider.ax.set_facecolor("#fffdf8")
        if hasattr(slider, "track"):
            slider.track.set_color("#ddd6c4")
        if hasattr(slider, "poly"):
            slider.poly.set_facecolor("#5e8b7e")
        slider.label.set_text("")
        slider.valtext.set_color("#332f26")
        slider.valtext.set_fontsize(7.8)
        slider.valtext.set_horizontalalignment("right")
        slider.valtext.set_x(0.9)
        slider.ax.set_title(title, fontsize=7.9, pad=1, loc="left", color="#332f26", fontweight="bold")

    def _style_button(self, button: Button) -> None:
        button.ax.set_facecolor("#fffdf8")
        button.color = "#fffdf8"
        button.hovercolor = "#efe7d6"
        button.label.set_color("#332f26")
        button.label.set_fontsize(9.0)
        for spine in button.ax.spines.values():
            spine.set_edgecolor("#d8d2c2")

    def _style_textbox(self, textbox: TextBox, label_size: float = 8.1) -> None:
        textbox.ax.set_facecolor("#fffdf8")
        title = textbox.label.get_text()
        textbox.label.set_text("")
        textbox.text_disp.set_color("#332f26")
        textbox.text_disp.set_fontsize(8.0)
        textbox.ax.set_title(title, fontsize=label_size, pad=1, loc="center", color="#332f26", fontweight="bold")
        for spine in textbox.ax.spines.values():
            spine.set_edgecolor("#d8d2c2")

    def _update_metric_range_text(self) -> None:
        values = self._get_active_metric_values()
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        self.metric_range_text.set_text(f"available range: {vmin:.3f} to {vmax:.3f}")

    def _build_variable_legend_text(self) -> str:
        lines = [
            "Positions / wing kinematics",
            "stroke    : sweep angle in the wing frame",
            "deviation : out-of-plane angle in the wing frame",
            "rotation  : wing pitch angle",
            "",
            "Input variables",
        ]
        for name in INPUT_VARIABLE_NAMES:
            lines.append(f"{name:<10}: {INPUT_VARIABLE_DESCRIPTIONS[name]}")
        lines.extend([
            "",
            "Output variables",
        ])
        for name in OUTPUT_VARIABLE_NAMES:
            lines.append(f"{name:<10}: {OUTPUT_VARIABLE_DESCRIPTIONS[name]}")
        lines.extend([
            "",
            "Plot overlays",
            "True min/max : minimum or maximum of measured data",
            "Pred min/max : minimum or maximum of predicted data",
            "Errors ON    : overall trajectory RMSE and R^2 on each plot",
            "Inst Error ON: live pred-true value on each plot",
            "",
            "3D arrow overlays",
            "Input Arrows : current measured input values in the wing frame",
            "Output Arrows: current measured output values in the wing frame",
            "Arrow length : proportional to the current value, scaled by",
            "               that trajectory's own variable range",
            "Arrow popup  : choose which input/output variables appear",
            "",
            "3D axes",
            "x_w       : stroke axis",
            "y_w       : deviation axis",
            "z_w       : span axis",
            "",
            "Scale note",
            f"span = {SPAN_LENGTH:.2f} units, chord = {CHORD_LENGTH:.2f} units",
            "Cycle time [-] is normalized cycle fraction from 0 to 1.",
        ])
        return "\n".join(lines)

    def _show_variable_legend(self, _event) -> None:
        if self.legend_popup_fig is not None and plt.fignum_exists(self.legend_popup_fig.number):
            self.legend_popup_fig.canvas.draw_idle()
            return

        popup = plt.figure(figsize=(6.4, 7.2), facecolor="#f4f1ea")
        try:
            popup.canvas.manager.set_window_title("Variable Meanings")
        except Exception:
            pass

        ax = popup.add_axes([0.07, 0.06, 0.86, 0.88])
        ax.axis("off")
        ax.set_facecolor("#fffdf8")
        ax.text(
            0.0,
            1.0,
            self._build_variable_legend_text(),
            va="top",
            ha="left",
            fontsize=10.0,
            family="monospace",
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffdf8", edgecolor="#d8d2c2"),
        )
        self.legend_popup_fig = popup
        popup.show()

    def _show_arrow_selector_popup(self, kind: str) -> None:
        if kind == "input":
            fig_attr = "input_selector_popup_fig"
            checks_attr = "input_selector_checks"
            check_texts_attr = "input_selector_check_texts"
            title = "Input Arrow Selection"
            names = INPUT_VARIABLE_NAMES
            descriptions = INPUT_VARIABLE_DESCRIPTIONS
            selected = self.selected_input_arrow_vars
            on_toggle = self._on_input_arrow_selector_toggle
            color_map = INPUT_ARROW_COLORS
        else:
            fig_attr = "output_selector_popup_fig"
            checks_attr = "output_selector_checks"
            check_texts_attr = "output_selector_check_texts"
            title = "Output Arrow Selection"
            names = OUTPUT_VARIABLE_NAMES
            descriptions = OUTPUT_VARIABLE_DESCRIPTIONS
            selected = self.selected_output_arrow_vars
            on_toggle = self._on_output_arrow_selector_toggle
            color_map = OUTPUT_ARROW_COLORS

        popup = getattr(self, fig_attr)
        if popup is not None and plt.fignum_exists(popup.number):
            self._refresh_arrow_selector_checkmarks(kind)
            popup.canvas.draw_idle()
            return

        popup = plt.figure(figsize=(3.8, 4.6), facecolor="#f4f1ea")
        try:
            popup.canvas.manager.set_window_title(title)
        except Exception:
            pass

        ax = popup.add_axes([0.10, 0.08, 0.82, 0.84])
        ax.set_facecolor("#fffdf8")
        labels = [f"{name} ({descriptions[name]})" for name in names]
        checks = CheckButtons(ax, labels, [name in selected for name in names])

        for name, label in zip(names, checks.labels):
            label.set_fontsize(8.2)
            label.set_color(color_map[name])
        for line_group in checks.lines:
            for line in line_group:
                line.set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d8d2c2")
        ax.set_title(title, fontsize=10.0, fontweight="bold", pad=6)
        ax.text(
            0.02,
            0.98,
            "Checked variables are shown on the 3D graph.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            color="#5b5345",
        )

        check_texts = []
        for name, line_group in zip(names, checks.lines):
            x_data = line_group[0].get_xdata()
            y_data = line_group[0].get_ydata()
            x_center = float(np.mean(x_data))
            y_center = float(np.mean(y_data))
            check_text = ax.text(
                x_center,
                y_center,
                "✓",
                ha="center",
                va="center",
                fontsize=12.5,
                fontweight="bold",
                color=color_map[name],
                transform=ax.transAxes,
                visible=(name in selected),
            )
            check_texts.append(check_text)

        checks.on_clicked(on_toggle)
        setattr(self, fig_attr, popup)
        setattr(self, checks_attr, checks)
        setattr(self, check_texts_attr, check_texts)
        popup.show()

    def _on_input_arrow_selector_toggle(self, label: str) -> None:
        name = label.split(" (", 1)[0]
        if name in self.selected_input_arrow_vars:
            self.selected_input_arrow_vars.remove(name)
        else:
            self.selected_input_arrow_vars.add(name)
        self._refresh_arrow_selector_checkmarks("input")
        self._update_frame(self.frame_idx)

    def _on_output_arrow_selector_toggle(self, label: str) -> None:
        name = label.split(" (", 1)[0]
        if name in self.selected_output_arrow_vars:
            self.selected_output_arrow_vars.remove(name)
        else:
            self.selected_output_arrow_vars.add(name)
        self._refresh_arrow_selector_checkmarks("output")
        self._update_frame(self.frame_idx)

    def _refresh_arrow_selector_checkmarks(self, kind: str) -> None:
        if kind == "input":
            names = INPUT_VARIABLE_NAMES
            selected = self.selected_input_arrow_vars
            popup = self.input_selector_popup_fig
            check_texts = self.input_selector_check_texts
        else:
            names = OUTPUT_VARIABLE_NAMES
            selected = self.selected_output_arrow_vars
            popup = self.output_selector_popup_fig
            check_texts = self.output_selector_check_texts

        if popup is None or not plt.fignum_exists(popup.number):
            return

        for name, check_text in zip(names, check_texts):
            check_text.set_visible(name in selected)

        popup.canvas.draw_idle()

    def _on_split_change(self, label: str) -> None:
        self.split = label
        self.current = self.splits[label]
        self.traj_index = int(np.clip(self.traj_index, 0, self.current.n_traj - 1))
        self.slider_frame.valmax = max(self.current.n_steps - 1, 0)
        self.slider_frame.ax.set_xlim(self.slider_frame.valmin, self.slider_frame.valmax)
        self._reset_metric_filter_controls()

    def _on_traj_change(self, value: float) -> None:
        if len(self.filtered_indices) == 0:
            return
        traj_index = int(self.filtered_indices[int(value)])
        if traj_index == self.traj_index and hasattr(self, "trajectory_pos"):
            return
        self.traj_index = traj_index
        self._load_trajectory(traj_index)
        self._update_frame(0)

    def _on_speed_change(self, value: float) -> None:
        self.speed = float(value)

    def _on_frame_change(self, value: float) -> None:
        frame = int(value)
        if frame == self.last_slider_frame:
            return
        self.frame_idx = frame
        self.frame_float = float(frame)
        self.last_slider_frame = frame
        self._update_frame(frame)

    def _on_traj_text_submit(self, text: str) -> None:
        try:
            traj_index = int(float(text.strip()))
        except ValueError:
            return

        traj_index = int(np.clip(traj_index, 0, self.current.n_traj - 1))
        if len(self.filtered_indices) == 0:
            self.filtered_indices = np.array([traj_index], dtype=int)
            slider_index = 0
        elif traj_index in set(self.filtered_indices.tolist()):
            slider_index = int(np.where(self.filtered_indices == traj_index)[0][0])
        else:
            self.filtered_indices = np.array([traj_index], dtype=int)
            slider_index = 0
            self.slider_traj.valmax = 0
            self.slider_traj.ax.set_xlim(self.slider_traj.valmin, max(self.slider_traj.valmax, 0))

        self.slider_traj.eventson = False
        self.slider_traj.set_val(slider_index)
        self.slider_traj.eventson = True
        self.traj_index = traj_index
        self._load_trajectory(traj_index)
        self._update_frame(0)

    def _get_active_metric_values(self) -> np.ndarray:
        return self.current.filter_metrics[self.filter_metric_name]

    def _reset_metric_filter_controls(self) -> None:
        values = self._get_active_metric_values()
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        self._update_metric_range_text()
        self.filter_min = vmin
        self.filter_max = vmax

        for slider, val in [(self.slider_filter_min, vmin), (self.slider_filter_max, vmax)]:
            slider.valmin = vmin
            slider.valmax = vmax
            slider.ax.set_xlim(vmin, vmax)
            slider.eventson = False
            slider.set_val(val)
            slider.eventson = True

        self.textbox_min.set_val(f"{vmin:.4f}")
        self.textbox_max.set_val(f"{vmax:.4f}")
        self._apply_filters(keep_current=True)

    def _on_metric_change(self, label: str) -> None:
        for metric_name, display_name in FILTER_METRIC_DISPLAY.items():
            if label == display_name:
                self.filter_metric_name = metric_name
                break
        else:
            self.filter_metric_name = label
        self._reset_metric_filter_controls()

    def _on_filter_slider_change(self, _value: float) -> None:
        min_val = float(self.slider_filter_min.val)
        max_val = float(self.slider_filter_max.val)
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        self.filter_min = min_val
        self.filter_max = max_val
        self.textbox_min.set_val(f"{min_val:.4f}")
        self.textbox_max.set_val(f"{max_val:.4f}")
        self._apply_filters(keep_current=True)

    def _on_filter_text_submit(self, _text: str) -> None:
        try:
            min_val = float(self.textbox_min.text)
            max_val = float(self.textbox_max.text)
        except ValueError:
            return

        if min_val > max_val:
            min_val, max_val = max_val, min_val

        values = self._get_active_metric_values()
        lower_bound = float(np.min(values))
        upper_bound = float(np.max(values))
        min_val = float(np.clip(min_val, lower_bound, upper_bound))
        max_val = float(np.clip(max_val, lower_bound, upper_bound))
        self.filter_min = min_val
        self.filter_max = max_val

        self.slider_filter_min.eventson = False
        self.slider_filter_max.eventson = False
        self.slider_filter_min.set_val(min_val)
        self.slider_filter_max.set_val(max_val)
        self.slider_filter_min.eventson = True
        self.slider_filter_max.eventson = True
        self._apply_filters(keep_current=True)

    def _toggle_pause(self, _event) -> None:
        self.paused = not self.paused
        self.button_play.label.set_text("Play" if self.paused else "Pause")

    def _toggle_trace_mode(self, _event) -> None:
        self.trace_mode = "2D Live Traces" if self.trace_mode == "2D Full Traces" else "2D Full Traces"
        self.button_trace.label.set_text(self.trace_mode)
        self._update_frame(self.frame_idx)

    def _toggle_extrema_mode(self, _event) -> None:
        self.extrema_mode = "Extrema ON" if self.extrema_mode == "Extrema OFF" else "Extrema OFF"
        self.button_extrema.label.set_text(self.extrema_mode)
        self._update_frame(self.frame_idx)

    def _toggle_error_metrics_mode(self, _event) -> None:
        self.error_metrics_mode = "Errors ON" if self.error_metrics_mode == "Errors OFF" else "Errors OFF"
        self.button_errors.label.set_text(self.error_metrics_mode)
        self._update_frame(self.frame_idx)

    def _toggle_instant_error_mode(self, _event) -> None:
        self.instant_error_mode = "Inst Error ON" if self.instant_error_mode == "Inst Error OFF" else "Inst Error OFF"
        self.button_instant_error.label.set_text(self.instant_error_mode)
        self._update_frame(self.frame_idx)

    def _toggle_input_arrows_mode(self, _event) -> None:
        self.input_arrows_mode = "Input Arrows ON" if self.input_arrows_mode == "Input Arrows OFF" else "Input Arrows OFF"
        self.button_input_arrows.label.set_text(self.input_arrows_mode)
        if self.input_arrows_mode == "Input Arrows ON":
            self._show_arrow_selector_popup("input")
        self._update_frame(self.frame_idx)

    def _toggle_output_arrows_mode(self, _event) -> None:
        self.output_arrows_mode = "Output Arrows ON" if self.output_arrows_mode == "Output Arrows OFF" else "Output Arrows OFF"
        self.button_output_arrows.label.set_text(self.output_arrows_mode)
        if self.output_arrows_mode == "Output Arrows ON":
            self._show_arrow_selector_popup("output")
        self._update_frame(self.frame_idx)

    def _toggle_tracer_mode(self, _event) -> None:
        self.tip_tracer_mode = "3D Tip Tracer OFF" if self.tip_tracer_mode == "3D Tip Tracer ON" else "3D Tip Tracer ON"
        self.button_tracer.label.set_text(self.tip_tracer_mode)
        self._update_frame(self.frame_idx)

    def _refresh_info_panel(self) -> None:
        self.ax_details.set_title("Selection Details", fontsize=11, fontweight="bold", pad=8)
        self.info_text.set_text(self._format_info_text(self.frame_idx))
        self.fig.canvas.draw_idle()

    def _cycle_category(self, _event) -> None:
        idx = CATEGORY_ORDER.index(self.category_name)
        self._on_category_change(CATEGORY_ORDER[(idx + 1) % len(CATEGORY_ORDER)])

    def _on_category_change(self, label: str) -> None:
        self.category_name = label
        self._apply_filters(keep_current=True)

    def _apply_filters(self, keep_current: bool) -> None:
        category_indices = self.current.category_indices[self.category_name]
        metric_values = self._get_active_metric_values()
        min_val = float(self.filter_min)
        max_val = float(self.filter_max)
        metric_mask = (metric_values >= min_val) & (metric_values <= max_val)
        metric_indices = np.where(metric_mask)[0]
        self.filtered_indices = np.intersect1d(category_indices, metric_indices, assume_unique=False)
        if len(self.filtered_indices) == 0:
            self.filtered_indices = np.arange(self.current.n_traj, dtype=int)

        self._refresh_filtered_trajectory(keep_current=keep_current)
        self._load_trajectory(self.traj_index)
        self._update_frame(0)

    def _refresh_filtered_trajectory(self, keep_current: bool) -> None:
        if keep_current and self.traj_index in set(self.filtered_indices.tolist()):
            slider_index = int(np.where(self.filtered_indices == self.traj_index)[0][0])
        else:
            self.traj_index = int(self.filtered_indices[0])
            slider_index = 0

        self.slider_traj.valmax = max(len(self.filtered_indices) - 1, 0)
        self.slider_traj.ax.set_xlim(self.slider_traj.valmin, self.slider_traj.valmax)
        self.slider_traj.eventson = False
        self.slider_traj.set_val(slider_index)
        self.slider_traj.eventson = True
        self.textbox_traj.set_val(str(self.traj_index))

    def _reset_animation(self, _event) -> None:
        self.paused = True
        self.button_play.label.set_text("Play")
        self.frame_idx = 0
        self.frame_float = 0.0
        self.slider_frame.set_val(0)
        self._update_frame(0)

    def _on_key_press(self, event) -> None:
        if event.key == " ":
            self._toggle_pause(None)
        elif event.key == "t":
            self._toggle_trace_mode(None)
        elif event.key == "m":
            self._toggle_extrema_mode(None)
        elif event.key == "e":
            self._toggle_error_metrics_mode(None)
        elif event.key == "i":
            self._toggle_instant_error_mode(None)
        elif event.key == "u":
            self._toggle_input_arrows_mode(None)
        elif event.key == "o":
            self._toggle_output_arrows_mode(None)
        elif event.key == "r":
            self._toggle_tracer_mode(None)
        elif event.key == "c":
            self._cycle_category(None)
        elif event.key == "right":
            self.paused = True
            self.button_play.label.set_text("Play")
            self.slider_frame.set_val(min(self.frame_idx + 1, self.current.n_steps - 1))
        elif event.key == "left":
            self.paused = True
            self.button_play.label.set_text("Play")
            self.slider_frame.set_val(max(self.frame_idx - 1, 0))

    def _load_trajectory(self, traj_index: int) -> None:
        self.trajectory_pos = self.current.pos[traj_index]
        self.trajectory_inputs = self.current.inputs_raw[traj_index]
        self.trajectory_truth = self.current.truth[traj_index]
        self.trajectory_pred = self.current.pred[traj_index]
        self.trajectory_summary = self.current.category_summaries[traj_index]
        self.time_axis = np.linspace(0.0, 1.0, self.current.n_steps)
        self.frame_idx = 0
        self.frame_float = 0.0
        self.last_slider_frame = 0

        geometry = [wing_geometry(*angles) for angles in self.trajectory_pos]
        self.trajectory_hinge = np.array([item[0] for item in geometry])
        self.trajectory_leading = np.array([item[1] for item in geometry])
        self.trajectory_trailing = np.array([item[2] for item in geometry])
        self.trajectory_tip = 0.5 * (self.trajectory_leading + self.trajectory_trailing)
        self.scene_lim = 0.8
        self.input_scale_refs = {
            name: max(float(np.max(np.abs(self.trajectory_inputs[:, idx]))), 1e-6)
            for idx, name in enumerate(INPUT_VARIABLE_NAMES)
        }
        self.output_scale_refs = {
            name: max(float(np.max(np.abs(self.trajectory_truth[:, idx]))), 1e-6)
            for idx, name in enumerate(OUTPUT_VARIABLE_NAMES)
        }

        for i, ax in enumerate(self.output_axes):
            truth_series = self.trajectory_truth[:, i]
            pred_series = self.trajectory_pred[:, i]
            self.truth_lines[i].set_data(self.time_axis, truth_series)
            self.pred_lines[i].set_data(self.time_axis, pred_series)
            ymin = min(truth_series.min(), pred_series.min())
            ymax = max(truth_series.max(), pred_series.max())
            pad = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
            ax.set_ylim(ymin - pad, ymax + pad)

            span = (ymax + pad) - (ymin - pad)
            text_dx = 0.012
            text_dy = 0.04 * span

            truth_max_idx = int(np.argmax(truth_series))
            truth_min_idx = int(np.argmin(truth_series))
            pred_max_idx = int(np.argmax(pred_series))
            pred_min_idx = int(np.argmin(pred_series))

            extrema_points = [
                (self.time_axis[truth_max_idx], truth_series[truth_max_idx], f"True max {truth_series[truth_max_idx]:.3f}", text_dx, text_dy),
                (self.time_axis[truth_min_idx], truth_series[truth_min_idx], f"True min {truth_series[truth_min_idx]:.3f}", text_dx, -text_dy),
                (self.time_axis[pred_max_idx], pred_series[pred_max_idx], f"Pred max {pred_series[pred_max_idx]:.3f}", -text_dx, text_dy),
                (self.time_axis[pred_min_idx], pred_series[pred_min_idx], f"Pred min {pred_series[pred_min_idx]:.3f}", -text_dx, -text_dy),
            ]

            for marker, text, (xv, yv, label, dx, dy) in zip(
                self.extrema_markers[i],
                self.extrema_texts[i],
                extrema_points,
            ):
                marker.set_data([xv], [yv])
                text.set_position((np.clip(xv + dx, 0.0, 1.0), yv + dy))
                text.set_text(label)

            rmse = compute_rmse(truth_series, pred_series)
            r2 = compute_r2(truth_series, pred_series)
            self.metric_texts[i].set_text(f"RMSE {rmse:.3f}\nR$^2$ {r2:.3f}")

        all_points = np.vstack([self.trajectory_hinge, self.trajectory_leading, self.trajectory_trailing])
        lim = np.max(np.abs(all_points)) * 1.15
        lim = max(lim, 0.8)
        self.scene_lim = lim
        self.ax3d.set_xlim(-lim, lim)
        self.ax3d.set_ylim(-lim, lim)
        self.ax3d.set_zlim(-lim, lim)

    def _clear_3d_arrow_artists(self, artists: List) -> None:
        while artists:
            artist = artists.pop()
            try:
                artist.remove()
            except Exception:
                pass

    def _add_3d_arrow(self, container: List, origin: np.ndarray, vector: np.ndarray, color: str, label: str) -> None:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return
        quiver = self.ax3d.quiver(
            origin[0], origin[1], origin[2],
            vector[0], vector[1], vector[2],
            color=color,
            linewidth=2.0,
            arrow_length_ratio=0.18,
        )
        text = self.ax3d.text(
            origin[0] + vector[0],
            origin[1] + vector[1],
            origin[2] + vector[2],
            label,
            color=color,
            fontsize=7.0,
        )
        container.extend([quiver, text])

    def _draw_input_arrows(self, frame: int, hinge: np.ndarray, wing_tip: np.ndarray, leading: np.ndarray, trailing: np.ndarray) -> None:
        self._clear_3d_arrow_artists(self.input_arrow_artists)
        if self.input_arrows_mode != "Input Arrows ON":
            return

        base = 0.45 * self.scene_lim
        inputs = self.trajectory_inputs[frame]
        chord_dir = normalize_vector(leading - trailing)
        span_dir = normalize_vector(wing_tip - hinge)
        normal_dir = normalize_vector(np.cross(span_dir, chord_dir))
        prev_idx = max(frame - 1, 0)
        next_idx = min(frame + 1, self.current.n_steps - 1)
        vel_dir = normalize_vector(self.trajectory_tip[next_idx] - self.trajectory_tip[prev_idx])
        if np.linalg.norm(vel_dir) < 1e-9:
            vel_dir = np.array([1.0, 0.0, 0.0])

        axis_dirs = {
            "a_x": np.array([1.0, 0.0, 0.0]),
            "a_y": np.array([0.0, 1.0, 0.0]),
            "a_z": np.array([0.0, 0.0, 1.0]),
            "alpha_dot": np.array([0.0, 0.0, 1.0]),
            "alpha_ddot": np.array([0.0, 0.0, 1.0]),
        }

        for name in ("a_x", "a_y", "a_z"):
            if name not in self.selected_input_arrow_vars:
                continue
            value = inputs[INPUT_INDEX[name]]
            length = base * (value / self.input_scale_refs[name])
            self._add_3d_arrow(
                self.input_arrow_artists,
                hinge,
                axis_dirs[name] * length,
                INPUT_ARROW_COLORS[name],
                f"{name} {value:.2f}",
            )

        if "AoA" in self.selected_input_arrow_vars:
            aoa_val = inputs[INPUT_INDEX["AoA"]]
            self._add_3d_arrow(
                self.input_arrow_artists,
                wing_tip,
                normal_dir * base * (aoa_val / self.input_scale_refs["AoA"]),
                INPUT_ARROW_COLORS["AoA"],
                f"AoA {aoa_val:.2f}",
            )
        if "|v|" in self.selected_input_arrow_vars:
            vel_val = inputs[INPUT_INDEX["|v|"]]
            self._add_3d_arrow(
                self.input_arrow_artists,
                wing_tip,
                vel_dir * base * (vel_val / self.input_scale_refs["|v|"]),
                INPUT_ARROW_COLORS["|v|"],
                f"|v| {vel_val:.2f}",
            )
        for name, origin in (("alpha_dot", trailing), ("alpha_ddot", leading)):
            if name not in self.selected_input_arrow_vars:
                continue
            value = inputs[INPUT_INDEX[name]]
            length = base * (value / self.input_scale_refs[name])
            self._add_3d_arrow(
                self.input_arrow_artists,
                origin,
                axis_dirs[name] * length,
                INPUT_ARROW_COLORS[name],
                f"{name} {value:.2f}",
            )

    def _draw_output_arrows(self, frame: int, hinge: np.ndarray, wing_tip: np.ndarray) -> None:
        self._clear_3d_arrow_artists(self.output_arrow_artists)
        if self.output_arrows_mode != "Output Arrows ON":
            return

        base = 0.45 * self.scene_lim
        outputs = self.trajectory_truth[frame]
        force_axes = {
            "CF_x": np.array([1.0, 0.0, 0.0]),
            "CF_y": np.array([0.0, 1.0, 0.0]),
        }
        moment_axes = {
            "CM_x": np.array([1.0, 0.0, 0.0]),
            "CM_y": np.array([0.0, 1.0, 0.0]),
            "CM_z": np.array([0.0, 0.0, 1.0]),
        }

        for name, axis in force_axes.items():
            if name not in self.selected_output_arrow_vars:
                continue
            value = outputs[OUTPUT_INDEX[name]]
            length = base * (value / self.output_scale_refs[name])
            self._add_3d_arrow(
                self.output_arrow_artists,
                wing_tip,
                axis * length,
                OUTPUT_ARROW_COLORS[name],
                f"{name} {value:.2f}",
            )

        for name, axis in moment_axes.items():
            if name not in self.selected_output_arrow_vars:
                continue
            value = outputs[OUTPUT_INDEX[name]]
            length = base * (value / self.output_scale_refs[name])
            self._add_3d_arrow(
                self.output_arrow_artists,
                hinge,
                axis * length,
                OUTPUT_ARROW_COLORS[name],
                f"{name} {value:.2f}",
            )

    def _format_info_text(self, frame: int) -> str:
        inputs = self.trajectory_inputs[frame]
        pos = self.trajectory_pos[frame]
        lines = [
            f"Trajectory: {self.traj_index}",
            f"Frame     : {frame + 1}/{self.current.n_steps}",
            f"Split     : {self.split}",
            "",
            f"Category  : {self.category_name}",
            f"Rule      : {CATEGORY_RULES[self.category_name]}",
            f"Filter    : {self.filter_metric_name}",
            f"Meaning   : {FILTER_METRIC_RULES[self.filter_metric_name]}",
            f"Range     : {self.filter_min:.4f} to {self.filter_max:.4f}",
            "",
            "Summary",
            self.trajectory_summary,
            "",
            "Kinematics",
        ]
        kin_parts = [
            f"{name[:5]}={np.degrees(value):6.1f} deg"
            for name, value in zip(POSITION_NAMES, pos)
        ]
        lines.append(" | ".join(kin_parts))
        lines.append("")
        lines.append("Inputs")
        input_parts = [f"{name}={value:6.3f}" for name, value in zip(INPUT_VARIABLE_NAMES, inputs)]
        lines.append(" | ".join(input_parts[:3]))
        lines.append(" | ".join(input_parts[3:5]))
        lines.append(" | ".join(input_parts[5:]))
        lines.append("")
        lines.append("Instant error (pred-true)")
        err_parts = [
            f"{name}={self.trajectory_pred[frame, i] - self.trajectory_truth[frame, i]:6.3f}"
            for i, name in enumerate(OUTPUT_VARIABLE_NAMES)
        ]
        lines.append(" | ".join(err_parts[:3]))
        lines.append(" | ".join(err_parts[3:]))
        return "\n".join(lines)

    def _update_frame(self, frame: int) -> List:
        self.frame_idx = int(np.clip(frame, 0, self.current.n_steps - 1))
        t = self.time_axis[self.frame_idx]

        hinge = self.trajectory_hinge[self.frame_idx]
        leading = self.trajectory_leading[self.frame_idx]
        trailing = self.trajectory_trailing[self.frame_idx]
        wing_tip = self.trajectory_tip[self.frame_idx]
        leading_history = self.trajectory_leading[: self.frame_idx + 1]
        trailing_history = self.trajectory_trailing[: self.frame_idx + 1]

        self.hinge_trace.set_data([0.0], [0.0])
        self.hinge_trace.set_3d_properties([0.0])
        self.wing_span.set_data([hinge[0], wing_tip[0]], [hinge[1], wing_tip[1]])
        self.wing_span.set_3d_properties([hinge[2], wing_tip[2]])
        self.leading_edge.set_data([wing_tip[0], leading[0]], [wing_tip[1], leading[1]])
        self.leading_edge.set_3d_properties([wing_tip[2], leading[2]])
        self.trailing_edge.set_data([wing_tip[0], trailing[0]], [wing_tip[1], trailing[1]])
        self.trailing_edge.set_3d_properties([wing_tip[2], trailing[2]])
        self.stroke_history.set_data(leading_history[:, 0], leading_history[:, 1])
        self.stroke_history.set_3d_properties(leading_history[:, 2])
        if self.tip_tracer_mode == "3D Tip Tracer ON":
            self.leading_tip_trace.set_data(leading_history[:, 0], leading_history[:, 1])
            self.leading_tip_trace.set_3d_properties(leading_history[:, 2])
            self.trailing_tip_trace.set_data(trailing_history[:, 0], trailing_history[:, 1])
            self.trailing_tip_trace.set_3d_properties(trailing_history[:, 2])
            self.leading_tip_marker.set_data([leading[0]], [leading[1]])
            self.leading_tip_marker.set_3d_properties([leading[2]])
            self.trailing_tip_marker.set_data([trailing[0]], [trailing[1]])
            self.trailing_tip_marker.set_3d_properties([trailing[2]])
        else:
            self.leading_tip_trace.set_data([], [])
            self.leading_tip_trace.set_3d_properties([])
            self.trailing_tip_trace.set_data([], [])
            self.trailing_tip_trace.set_3d_properties([])
            self.leading_tip_marker.set_data([], [])
            self.leading_tip_marker.set_3d_properties([])
            self.trailing_tip_marker.set_data([], [])
            self.trailing_tip_marker.set_3d_properties([])
        self.wing_patch.set_verts([[hinge, leading, trailing]])
        self._draw_input_arrows(self.frame_idx, hinge, wing_tip, leading, trailing)
        self._draw_output_arrows(self.frame_idx, hinge, wing_tip)

        current_time_sec = self.frame_idx * self.playback_dt
        self.header_text.set_text(
            f"{'Paused' if self.paused else 'Playing'} | "
            f"time {current_time_sec:0.3f} s | "
            f"speed {self.speed:0.2f}x | "
            f"split {self.split} | "
            f"category {self.category_name} | "
            f"{self.mat_file.name}"
        )
        self.info_text.set_text(self._format_info_text(self.frame_idx))

        artists: List = [
            self.hinge_trace,
            self.wing_span,
            self.leading_edge,
            self.trailing_edge,
            self.stroke_history,
            self.leading_tip_trace,
            self.trailing_tip_trace,
            self.leading_tip_marker,
            self.trailing_tip_marker,
            self.wing_patch,
            self.header_text,
            self.info_text,
        ]
        artists.extend(self.input_arrow_artists)
        artists.extend(self.output_arrow_artists)
        for i in range(len(OUTPUT_VARIABLE_NAMES)):
            if self.trace_mode == "2D Live Traces":
                line_slice = slice(0, self.frame_idx + 1)
                self.truth_lines[i].set_data(self.time_axis[line_slice], self.trajectory_truth[line_slice, i])
                self.pred_lines[i].set_data(self.time_axis[line_slice], self.trajectory_pred[line_slice, i])
            else:
                self.truth_lines[i].set_data(self.time_axis, self.trajectory_truth[:, i])
                self.pred_lines[i].set_data(self.time_axis, self.trajectory_pred[:, i])
            truth_value = self.trajectory_truth[self.frame_idx, i]
            pred_value = self.trajectory_pred[self.frame_idx, i]
            inst_error = pred_value - truth_value
            self.cursor_lines[i].set_xdata([t, t])
            self.actual_markers[i].set_data([t], [truth_value])
            self.pred_markers[i].set_data([t], [pred_value])
            self.metric_texts[i].set_visible(self.error_metrics_mode == "Errors ON")
            self.instant_error_texts[i].set_text(f"pred-true {inst_error:.3f}")
            self.instant_error_texts[i].set_visible(self.instant_error_mode == "Inst Error ON")
            show_extrema = self.extrema_mode == "Extrema ON"
            for marker, text in zip(self.extrema_markers[i], self.extrema_texts[i]):
                marker.set_visible(show_extrema)
                text.set_visible(show_extrema)
            artists.extend([
                self.truth_lines[i],
                self.pred_lines[i],
                self.cursor_lines[i],
                self.actual_markers[i],
                self.pred_markers[i],
                self.metric_texts[i],
                self.instant_error_texts[i],
            ])
            artists.extend(self.extrema_markers[i])
            artists.extend(self.extrema_texts[i])

        return artists

    def _animate(self, _frame: int) -> List:
        if not self.paused:
            self.frame_float = (self.frame_float + self.speed) % self.current.n_steps
            self.frame_idx = int(self.frame_float)
            self.last_slider_frame = self.frame_idx
            self.slider_frame.eventson = False
            self.slider_frame.set_val(self.frame_idx)
            self.slider_frame.eventson = True
        return self._update_frame(self.frame_idx)

    def show(self) -> None:
        interval_ms = max(10, int(1000 * self.playback_dt / max(self.speed, 1e-6)))
        self.anim = animation.FuncAnimation(
            self.fig,
            self._animate,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate flapping-wing trajectories with synchronized outputs.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Full path to a model output folder containing usable MAT files in matfiles.",
    )
    parser.add_argument(
        "--out-folder-name",
        type=str,
        default=None,
        help="Folder name under ../Data/Output used when --out-dir is not provided.",
    )
    parser.add_argument(
        "--mat-file",
        type=str,
        default=None,
        help="Specific MAT filename inside the chosen output folder's matfiles directory.",
    )
    parser.add_argument(
        "--list-output-folders",
        action="store_true",
        help="List available output folders that contain usable MAT files and exit.",
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--trajectory", type=int, default=0, help="Zero-based trajectory index.")
    parser.add_argument("--speed", type=float, default=1.0, help="Initial playback speed multiplier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_output_folders:
        candidates = list_available_output_dirs()
        if not candidates:
            print("No valid output folders found under:")
            for root in get_output_roots():
                print(root)
        else:
            for path in candidates:
                print(path.name)
        return

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.out_folder_name:
        out_dir = build_default_out_dir(args.out_folder_name)
    else:
        out_dir = choose_output_dir_interactive()

    if args.mat_file:
        mat_file = out_dir / "matfiles" / args.mat_file
    else:
        mat_file = choose_prediction_file_interactive(out_dir)

    animator = WingTrajectoryAnimator(
        out_dir=out_dir,
        mat_file=mat_file,
        split=args.split,
        traj_index=args.trajectory,
        speed=args.speed,
    )
    animator.show()


if __name__ == "__main__":
    main()
