from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
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
INPUT_VARIABLE_NAMES = [
    "AoA",
    "|v|",
    "a_x",
    "a_y",
    "a_z",
    "alpha_dot",
    "alpha_ddot",
]
POSITION_NAMES = ["stroke", "deviation", "rotation"]
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


def to_windows_long_path(path: Path) -> str:
    abs_path = str(path.resolve())
    if abs_path.startswith("\\\\?\\"):
        return abs_path
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


def load_clean_mat(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MATLAB file: {path}\n"
            f"Expected a model output folder containing a prediction MAT file."
        )
    raw = loadmat(to_windows_long_path(path))
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def denormalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=float).reshape(1, 1, -1)
    std = np.asarray(std, dtype=float).reshape(1, 1, -1)
    return arr * std + mean


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


def list_available_output_dirs() -> List[Path]:
    candidates: List[Path] = []
    seen = set()
    for output_root in get_output_roots():
        if not output_root.exists():
            continue
        for child in sorted(output_root.iterdir()):
            if not child.is_dir():
                continue
            predict_file = child / "matfiles" / "predict_train_n_test.mat"
            if predict_file.exists():
                resolved = str(child.resolve())
                if resolved not in seen:
                    candidates.append(child)
                    seen.add(resolved)
    return candidates


def choose_output_dir_interactive() -> Path:
    candidates = list_available_output_dirs()
    if not candidates:
        raise FileNotFoundError(
            "No output folders with matfiles/predict_train_n_test.mat were found under:\n"
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
    if not mat_dir.exists():
        return []

    preferred = [
        p for p in sorted(mat_dir.glob("*.mat"))
        if ("predict" in p.name.lower()) and ("train" in p.name.lower() or "test" in p.name.lower())
    ]
    if preferred:
        return preferred
    return sorted(mat_dir.glob("*.mat"))


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
        self.trace_mode = "Full traces"
        self.category_name = "All"
        self.filter_metric_name = FILTER_METRIC_ORDER[0]
        self.filter_min = None
        self.filter_max = None
        self.tip_tracer_mode = "Tracer ON"
        self.info_mode = "Details"
        self.paused = True
        self.frame_idx = 0
        self.frame_float = 0.0
        self.last_slider_frame = 0
        self.playback_dt = 0.004

        self.splits = self._prepare_splits()
        self.current = self.splits[self.split]
        self.traj_index = int(np.clip(traj_index, 0, self.current.n_traj - 1))
        self.filtered_indices = np.arange(self.current.n_traj, dtype=int)

        self.fig = plt.figure(figsize=(17.5, 11.5), constrained_layout=False, facecolor="#f3f1eb")
        self.gs = GridSpec(
            10,
            6,
            figure=self.fig,
            width_ratios=[1.45, 1.25, 1.2, 1.2, 0.95, 1.05],
            height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.48, 0.48, 0.48, 0.48, 0.48],
            wspace=0.34,
            hspace=0.78,
        )
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

        self.ax3d = self.fig.add_subplot(self.gs[:5, :2], projection="3d")
        self.ax3d.set_title("Wing Motion in 3D", pad=16, fontsize=13, fontweight="bold")
        self.ax3d.set_xlabel("x (stroke plane direction)")
        self.ax3d.set_ylabel("y (deviation direction)")
        self.ax3d.set_zlabel("z (span / tip direction)")
        self.ax3d.view_init(elev=24, azim=-56)
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
        self.status_text = self.fig.text(
            0.045,
            0.905,
            "",
            fontsize=10,
            va="top",
            ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffdf8", edgecolor="#d8d2c2", alpha=0.95),
        )

        self.output_axes = []
        self.truth_lines = []
        self.pred_lines = []
        self.cursor_lines = []
        self.actual_markers = []
        self.pred_markers = []
        for i, name in enumerate(OUTPUT_VARIABLE_NAMES):
            ax = self.fig.add_subplot(self.gs[i, 2:4])
            ax.set_title(name, fontsize=11, pad=6, fontweight="bold")
            ax.grid(True, alpha=0.22, color="#b8b1a1")
            ax.set_xlim(0.0, 1.0)
            ax.set_facecolor("#fffdf8")
            for spine in ax.spines.values():
                spine.set_color("#d8d2c2")
            truth_line, = ax.plot([], [], color="#2a9d8f", linewidth=2.0, label="Measured")
            pred_line, = ax.plot([], [], color="#d55e00", linewidth=1.8, linestyle="--", label="Predicted")
            cursor = ax.axvline(0.0, color="black", alpha=0.25, linewidth=1.0)
            actual_marker, = ax.plot([], [], marker="o", color="#2a9d8f", markersize=5)
            pred_marker, = ax.plot([], [], marker="o", color="#d55e00", markersize=5)
            ax.text(
                0.01,
                0.88,
                OUTPUT_VARIABLE_DESCRIPTIONS[name],
                transform=ax.transAxes,
                fontsize=8,
                color="#5b5345",
                ha="left",
                va="top",
            )
            if i == len(OUTPUT_VARIABLE_NAMES) - 1:
                ax.set_xlabel("Normalized cycle time")
            self.output_axes.append(ax)
            self.truth_lines.append(truth_line)
            self.pred_lines.append(pred_line)
            self.cursor_lines.append(cursor)
            self.actual_markers.append(actual_marker)
            self.pred_markers.append(pred_marker)

        self.fig.legend(
            [self.truth_lines[0], self.pred_lines[0]],
            ["Measured", "Predicted"],
            loc="upper center",
            bbox_to_anchor=(0.55, 0.952),
            ncol=2,
            frameon=True,
            facecolor="#fffdf8",
            edgecolor="#d8d2c2",
            fontsize=9,
        )

        self.ax_info = self.fig.add_subplot(self.gs[:5, 4:6])
        self.ax_info.axis("off")
        self.ax_info.set_facecolor("#fffdf8")
        self.ax_info.set_title("Selection Details", fontsize=11, fontweight="bold", pad=10)
        self.info_text = self.ax_info.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            fontsize=8.2,
            family="monospace",
            linespacing=1.18,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffdf8", edgecolor="#d8d2c2"),
        )

        self.ax_split = self.fig.add_subplot(self.gs[5:8, 0])
        self.ax_category = self.fig.add_subplot(self.gs[5:10, 1])
        self.ax_metric = self.fig.add_subplot(self.gs[5:8, 4])
        self.ax_frame = self.fig.add_subplot(self.gs[5, 2:4])
        self.ax_traj = self.fig.add_subplot(self.gs[6, 2:4])
        self.ax_speed = self.fig.add_subplot(self.gs[7, 2:4])
        self.ax_filter_min = self.fig.add_subplot(self.gs[8, 2:4])
        self.ax_filter_max = self.fig.add_subplot(self.gs[9, 2:4])
        self.ax_min_box = self.fig.add_subplot(self.gs[8, 5])
        self.ax_max_box = self.fig.add_subplot(self.gs[9, 5])
        self.ax_play = self.fig.add_subplot(self.gs[5, 5])
        self.ax_reset = self.fig.add_subplot(self.gs[6, 5])
        self.ax_trace = self.fig.add_subplot(self.gs[7, 5])
        self.ax_tracer = self.fig.add_subplot(self.gs[8, 0])
        self.ax_toggle_guide = self.fig.add_subplot(self.gs[8, 5])
        self.ax_toggle_details = self.fig.add_subplot(self.gs[9, 5])

        self.radio_split = RadioButtons(self.ax_split, ("test", "train"), active=0 if self.split == "test" else 1)
        self._style_control_axis(self.ax_split, "Dataset Split")
        for label in self.radio_split.labels:
            label.set_fontsize(10)
            label.set_color("#332f26")

        self.radio_category = RadioButtons(
            self.ax_category,
            CATEGORY_ORDER,
            active=CATEGORY_ORDER.index(self.category_name),
        )
        self._style_control_axis(self.ax_category, "Trajectory Category")
        for label in self.radio_category.labels:
            label.set_fontsize(8.2)
            label.set_color("#332f26")

        self.radio_metric = RadioButtons(
            self.ax_metric,
            FILTER_METRIC_ORDER,
            active=FILTER_METRIC_ORDER.index(self.filter_metric_name),
        )
        self._style_control_axis(self.ax_metric, "Numeric Filter Metric")
        for label in self.radio_metric.labels:
            label.set_fontsize(7.9)
            label.set_color("#332f26")

        self.slider_traj = Slider(self.ax_traj, "Trajectory in Filter", 0, max(len(self.filtered_indices) - 1, 0), valinit=0, valstep=1)
        self.slider_speed = Slider(self.ax_speed, "Playback Speed", 0.25, 4.0, valinit=self.speed, valstep=0.25)
        self.slider_frame = Slider(self.ax_frame, "Frame", 0, max(self.current.n_steps - 1, 0), valinit=0, valstep=1)
        self.slider_filter_min = Slider(self.ax_filter_min, "Filter Min", 0.0, 1.0, valinit=0.0)
        self.slider_filter_max = Slider(self.ax_filter_max, "Filter Max", 0.0, 1.0, valinit=1.0)
        self.textbox_min = TextBox(self.ax_min_box, "Min", initial="")
        self.textbox_max = TextBox(self.ax_max_box, "Max", initial="")
        self.button_play = Button(self.ax_play, "Play")
        self.button_reset = Button(self.ax_reset, "Reset")
        self.button_trace = Button(self.ax_trace, self.trace_mode)
        self.button_tracer = Button(self.ax_tracer, self.tip_tracer_mode)
        self.button_toggle_guide = Button(self.ax_toggle_guide, "Guide")
        self.button_toggle_details = Button(self.ax_toggle_details, "Details")
        self._style_slider(self.slider_frame)
        self._style_slider(self.slider_traj)
        self._style_slider(self.slider_speed)
        self._style_slider(self.slider_filter_min)
        self._style_slider(self.slider_filter_max)
        self._style_button(self.button_play)
        self._style_button(self.button_reset)
        self._style_button(self.button_trace)
        self._style_button(self.button_tracer)
        self._style_button(self.button_toggle_guide)
        self._style_button(self.button_toggle_details)
        self._style_textbox(self.textbox_min)
        self._style_textbox(self.textbox_max)

        self.radio_split.on_clicked(self._on_split_change)
        self.radio_category.on_clicked(self._on_category_change)
        self.radio_metric.on_clicked(self._on_metric_change)
        self.slider_traj.on_changed(self._on_traj_change)
        self.slider_speed.on_changed(self._on_speed_change)
        self.slider_frame.on_changed(self._on_frame_change)
        self.slider_filter_min.on_changed(self._on_filter_slider_change)
        self.slider_filter_max.on_changed(self._on_filter_slider_change)
        self.textbox_min.on_submit(self._on_filter_text_submit)
        self.textbox_max.on_submit(self._on_filter_text_submit)
        self.button_play.on_clicked(self._toggle_pause)
        self.button_reset.on_clicked(self._reset_animation)
        self.button_trace.on_clicked(self._toggle_trace_mode)
        self.button_tracer.on_clicked(self._toggle_tracer_mode)
        self.button_toggle_guide.on_clicked(self._toggle_guide)
        self.button_toggle_details.on_clicked(self._toggle_details)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._reset_metric_filter_controls()
        self._refresh_info_panel()

    def _style_control_axis(self, ax, title: str) -> None:
        ax.set_facecolor("#fffdf8")
        ax.set_title(title, fontsize=10, pad=8, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#d8d2c2")

    def _style_slider(self, slider: Slider) -> None:
        slider.ax.set_facecolor("#fffdf8")
        if hasattr(slider, "track"):
            slider.track.set_color("#ddd6c4")
        if hasattr(slider, "poly"):
            slider.poly.set_facecolor("#5e8b7e")
        slider.label.set_color("#332f26")
        slider.valtext.set_color("#332f26")

    def _style_button(self, button: Button) -> None:
        button.ax.set_facecolor("#fffdf8")
        button.color = "#fffdf8"
        button.hovercolor = "#efe7d6"
        button.label.set_color("#332f26")
        button.label.set_fontsize(9.5)
        for spine in button.ax.spines.values():
            spine.set_edgecolor("#d8d2c2")

    def _style_textbox(self, textbox: TextBox) -> None:
        textbox.ax.set_facecolor("#fffdf8")
        textbox.label.set_color("#332f26")
        textbox.text_disp.set_color("#332f26")
        for spine in textbox.ax.spines.values():
            spine.set_edgecolor("#d8d2c2")

    def _build_variable_legend_text(self) -> str:
        lines = [
            "3D scale",
            "axes use normalized wing geometry",
            f"span      = {SPAN_LENGTH:.2f} units",
            f"chord     = {CHORD_LENGTH:.2f} units",
            "numbers show relative position only",
            "they do not represent meters unless",
            "you calibrate the wing dimensions",
            "",
            "Why it matters",
            "use the 3D axes to compare shape,",
            "orientation, and tip paths between",
            "trajectories rather than exact size",
            "",
            "3D axes",
            f"x         : {AXIS_LABELS['x']}",
            f"y         : {AXIS_LABELS['y']}",
            f"z         : {AXIS_LABELS['z']}",
            "",
            "Category rules",
        ]
        for name in CATEGORY_ORDER:
            lines.append(f"{name:10s}: {CATEGORY_RULES[name]}")
        lines.extend([
            "",
            "Refined filter",
            "choose a metric, then set",
            "min/max bounds to keep only",
            "trajectories inside that range",
        ])
        return "\n".join(lines)

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

    def _get_active_metric_values(self) -> np.ndarray:
        return self.current.filter_metrics[self.filter_metric_name]

    def _reset_metric_filter_controls(self) -> None:
        values = self._get_active_metric_values()
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
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
        self.trace_mode = "Live traces" if self.trace_mode == "Full traces" else "Full traces"
        self.button_trace.label.set_text(self.trace_mode)
        self._update_frame(self.frame_idx)

    def _toggle_tracer_mode(self, _event) -> None:
        self.tip_tracer_mode = "Tracer OFF" if self.tip_tracer_mode == "Tracer ON" else "Tracer ON"
        self.button_tracer.label.set_text(self.tip_tracer_mode)
        self._update_frame(self.frame_idx)

    def _toggle_guide(self, _event) -> None:
        self.info_mode = "Guide"
        self._refresh_info_panel()

    def _toggle_details(self, _event) -> None:
        self.info_mode = "Details"
        self._refresh_info_panel()

    def _refresh_info_panel(self) -> None:
        if self.info_mode == "Guide":
            self.ax_info.set_title("Guide", fontsize=11, fontweight="bold", pad=10)
            self.info_text.set_text(self._build_variable_legend_text())
            self.button_toggle_guide.color = "#efe7d6"
            self.button_toggle_details.color = "#fffdf8"
        else:
            self.ax_info.set_title("Selection Details", fontsize=11, fontweight="bold", pad=10)
            self.info_text.set_text(self._format_info_text(self.frame_idx))
            self.button_toggle_guide.color = "#fffdf8"
            self.button_toggle_details.color = "#efe7d6"
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
        elif event.key == "r":
            self._toggle_tracer_mode(None)
        elif event.key == "g":
            self._toggle_guide(None)
        elif event.key == "d":
            self._toggle_details(None)
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

        for i, ax in enumerate(self.output_axes):
            truth_series = self.trajectory_truth[:, i]
            pred_series = self.trajectory_pred[:, i]
            self.truth_lines[i].set_data(self.time_axis, truth_series)
            self.pred_lines[i].set_data(self.time_axis, pred_series)
            ymin = min(truth_series.min(), pred_series.min())
            ymax = max(truth_series.max(), pred_series.max())
            pad = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
            ax.set_ylim(ymin - pad, ymax + pad)

        all_points = np.vstack([self.trajectory_hinge, self.trajectory_leading, self.trajectory_trailing])
        lim = np.max(np.abs(all_points)) * 1.15
        lim = max(lim, 0.8)
        self.ax3d.set_xlim(-lim, lim)
        self.ax3d.set_ylim(-lim, lim)
        self.ax3d.set_zlim(-lim, lim)

    def _format_info_text(self, frame: int) -> str:
        inputs = self.trajectory_inputs[frame]
        pos = self.trajectory_pos[frame]
        lines = [
            f"Trajectory: {self.traj_index}",
            f"Frame     : {frame + 1}/{self.current.n_steps}",
            "",
            f"Category  : {self.category_name}",
            "",
            f"Filter    : {self.filter_metric_name}",
            f"Range     : {self.filter_min:.4f} to {self.filter_max:.4f}",
            "",
            "Summary",
            self.trajectory_summary,
            "",
            "Kinematics",
        ]
        for name, value in zip(POSITION_NAMES, pos):
            lines.append(f"{name:10s}: {np.degrees(value):8.2f} deg")
        lines.append("")
        lines.append("Inputs")
        for name, value in zip(INPUT_VARIABLE_NAMES, inputs):
            lines.append(f"{name:10s}: {value:8.4f}")
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
        if self.tip_tracer_mode == "Tracer ON":
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

        current_time_sec = self.frame_idx * self.playback_dt
        self.status_text.set_text(
            f"time = {current_time_sec:0.3f} s\n"
            f"speed = {self.speed:0.2f}x\n"
            f"state = {'paused' if self.paused else 'playing'}\n"
            f"split = {self.split} (train=fit, test=held-out)\n"
            f"category = {self.category_name}\n"
            f"mat file = {self.mat_file.name}"
        )
        if self.info_mode == "Details":
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
            self.status_text,
            self.info_text,
        ]
        for i in range(len(OUTPUT_VARIABLE_NAMES)):
            if self.trace_mode == "Live traces":
                line_slice = slice(0, self.frame_idx + 1)
                self.truth_lines[i].set_data(self.time_axis[line_slice], self.trajectory_truth[line_slice, i])
                self.pred_lines[i].set_data(self.time_axis[line_slice], self.trajectory_pred[line_slice, i])
            else:
                self.truth_lines[i].set_data(self.time_axis, self.trajectory_truth[:, i])
                self.pred_lines[i].set_data(self.time_axis, self.trajectory_pred[:, i])
            truth_value = self.trajectory_truth[self.frame_idx, i]
            pred_value = self.trajectory_pred[self.frame_idx, i]
            self.cursor_lines[i].set_xdata([t, t])
            self.actual_markers[i].set_data([t], [truth_value])
            self.pred_markers[i].set_data([t], [pred_value])
            artists.extend([
                self.truth_lines[i],
                self.pred_lines[i],
                self.cursor_lines[i],
                self.actual_markers[i],
                self.pred_markers[i],
            ])

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
        manager = plt.get_current_fig_manager()
        try:
            manager.window.state("zoomed")
        except Exception:
            try:
                manager.window.showMaximized()
            except Exception:
                try:
                    manager.full_screen_toggle()
                except Exception:
                    pass
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate flapping-wing trajectories with synchronized outputs.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Full path to a model output folder containing matfiles/predict_train_n_test.mat.",
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
        help="List available output folders that contain predict_train_n_test.mat and exit.",
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
