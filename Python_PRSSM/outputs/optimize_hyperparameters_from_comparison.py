from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_EXCEL_PATH = Path(
    r"C:\Files from USB\Carleton Year 5\MAAE 4907-N (Capstone Project - Micro Flapping-Wing Flyer (MFWF))\Neural Network (Fall 2025 - Winter 2026)\Data\Comparison of Model Accuracy.xlsx"
)

OUTPUT_COLUMNS = ["CF_X", "CF_Y", "CM_X", "CM_Y", "CM_Z"]
FEATURE_COLUMNS = [
    "number_of_layers",
    "layer1_activation_function",
    "layer1_filters",
    "layer1_kernel_size",
    "layer1_pool_size",
    "layer1_stride",
    "layer2_activation_function",
    "layer2_filters",
    "layer2_kernel_size",
    "layer2_pool_size",
    "layer2_stride",
    "learning_rate",
    "batch_size",
    "patience",
    "min_delta",
]

DEFAULT_PATIENCE = 3
DEFAULT_MIN_DELTA = 100000.0


def clean_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("r^2", "r2")
    text = text.replace("^2", "2")
    text = text.replace("/", "_")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("-", " ")
    text = text.replace("__", "_")
    tokens = [token for token in text.split() if token]
    return "_".join(tokens)


def build_column_names(header_top: pd.Series, header_bottom: pd.Series) -> List[str]:
    columns: List[str] = []
    current_top = ""

    for top, bottom in zip(header_top, header_bottom):
        if not pd.isna(top):
            current_top = clean_label(top)

        bottom_label = clean_label(bottom)
        if bottom_label == "notes":
            columns.append("notes")
            continue

        if bottom_label:
            if current_top and current_top not in {"", "nan"} and current_top != bottom_label:
                columns.append(f"{current_top}_{bottom_label}")
            else:
                columns.append(bottom_label)
        else:
            columns.append(current_top or "unnamed")

    deduped: List[str] = []
    seen: Dict[str, int] = {}
    for name in columns:
        count = seen.get(name, 0)
        if count:
            deduped.append(f"{name}_{count + 1}")
        else:
            deduped.append(name)
        seen[name] = count + 1
    return deduped


def load_experiment_table(excel_path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    try:
        if sheet_name is None:
            workbook = pd.ExcelFile(excel_path)
            sheet_name = workbook.sheet_names[0]
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    except ImportError as exc:
        raise ImportError(
            "Reading the Excel workbook requires the 'openpyxl' package.\n"
            "Install it in the active environment, for example:\n"
            "  conda install openpyxl\n"
            "or:\n"
            "  pip install openpyxl"
        ) from exc
    columns = build_column_names(raw.iloc[0], raw.iloc[1])
    df = raw.iloc[2:].copy()
    df.columns = columns

    rename_map = {
        "number_of_layers": "number_of_layers",
        "layer_1_activation_function": "layer1_activation_function",
        "layer_1_filters": "layer1_filters",
        "layer_1_kernel_size": "layer1_kernel_size",
        "layer_1_pool_size": "layer1_pool_size",
        "layer_1_stride": "layer1_stride",
        "layer_2_activation_function": "layer2_activation_function",
        "layer_2_filters": "layer2_filters",
        "layer_2_kernel_size": "layer2_kernel_size",
        "layer_2_pool_size": "layer2_pool_size",
        "layer_2_stride": "layer2_stride",
        "layer_2_learning_rate": "learning_rate",
        "layer_2_batch_size": "batch_size",
        "layer_2_patience": "patience",
        "layer_2_min_delta": "min_delta",
        "learning_rate": "learning_rate",
        "batch_size": "batch_size",
        "patience": "patience",
        "min_delta": "min_delta",
        "cf_x_test_rmse": "CF_X_RMSE",
        "cf_x_test_r2": "CF_X_R2",
        "cf_x_test_rr2": "CF_X_R2",
        "cf_y_test_rmse": "CF_Y_RMSE",
        "cf_y_test_r2": "CF_Y_R2",
        "cf_y_test_rr2": "CF_Y_R2",
        "cm_x_test_rmse": "CM_X_RMSE",
        "cm_x_test_r2": "CM_X_R2",
        "cm_x_test_rr2": "CM_X_R2",
        "cm_y_test_rmse": "CM_Y_RMSE",
        "cm_y_test_r2": "CM_Y_R2",
        "cm_y_test_rr2": "CM_Y_R2",
        "cm_z_test_rmse": "CM_Z_RMSE",
        "cm_z_test_r2": "CM_Z_R2",
        "cm_z_test_rr2": "CM_Z_R2",
        "average_test_rmse": "average_rmse_reported",
        "average_test_r2": "average_r2_reported",
        "average_test_rr2": "average_r2_reported",
        "chance_of_overtraining": "overtraining_label",
        "overtrained": "overtraining_label",
        "average_test_chance_of_overtraining": "overtraining_label",
        "average_test_overtrained": "overtraining_label",
        "average_test_training_classification": "training_classification",
        "training_classification": "training_classification",
        "epoch": "epoch",
        "total_training_time_seconds": "training_time_seconds",
        "total_training_time_hours": "training_time_hours",
        "notes": "notes",
    }
    df = df.rename(columns=rename_map)
    metric_markers = ["average_rmse_reported", "average_r2_reported"]
    df = df[df[metric_markers].notna().all(axis=1)].copy()
    df = df.reset_index(drop=True)

    numeric_columns = [
        "number_of_layers",
        "layer1_filters",
        "layer1_kernel_size",
        "layer1_pool_size",
        "layer1_stride",
        "layer2_filters",
        "layer2_kernel_size",
        "layer2_pool_size",
        "layer2_stride",
        "batch_size",
        "patience",
        "min_delta",
        "epoch",
        "training_time_seconds",
        "training_time_hours",
        "CF_X_RMSE",
        "CF_X_R2",
        "CF_Y_RMSE",
        "CF_Y_R2",
        "CM_X_RMSE",
        "CM_X_R2",
        "CM_Y_RMSE",
        "CM_Y_R2",
        "CM_Z_RMSE",
        "CM_Z_R2",
        "average_rmse_reported",
        "average_r2_reported",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in [
        "layer2_activation_function",
        "learning_rate",
        "layer1_activation_function",
        "overtraining_label",
        "training_classification",
    ]:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    return df


def compute_average_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rmse_columns = [f"{name}_RMSE" for name in OUTPUT_COLUMNS]
    r2_columns = [f"{name}_R2" for name in OUTPUT_COLUMNS]

    df = df.copy()
    computed_average_rmse = df[rmse_columns].mean(axis=1)
    computed_average_r2 = df[r2_columns].mean(axis=1)

    if "average_rmse_reported" in df.columns:
        df["average_rmse"] = df["average_rmse_reported"].fillna(computed_average_rmse)
    else:
        df["average_rmse"] = computed_average_rmse

    if "average_r2_reported" in df.columns:
        df["average_r2"] = df["average_r2_reported"].fillna(computed_average_r2)
    else:
        df["average_r2"] = computed_average_r2

    df["average_rmse_recomputed"] = computed_average_rmse
    df["average_r2_recomputed"] = computed_average_r2
    df["optimization_score"] = df["average_rmse"] + (1.0 - df["average_r2"])

    for column in ["epoch", "training_time_seconds", "training_time_hours"]:
        if column not in df.columns:
            df[column] = np.nan

    if "training_time_seconds" not in df.columns and "training_time_hours" in df.columns:
        df["training_time_seconds"] = pd.to_numeric(df["training_time_hours"], errors="coerce") * 3600.0
    if "training_time_hours" not in df.columns and "training_time_seconds" in df.columns:
        df["training_time_hours"] = pd.to_numeric(df["training_time_seconds"], errors="coerce") / 3600.0

    return df


def exclude_overtraining_rows(df: pd.DataFrame) -> pd.DataFrame:
    flagged = mark_overtraining_rows(df)
    mask = ~flagged["is_overtraining_flagged"]
    return df.loc[mask].reset_index(drop=True)


def normalize_training_classification(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    if text in {"yes", "y", "high", "overtrained", "true", "1", "overtraining"}:
        return "overtraining"
    if text in {"undertraining", "undertrained"}:
        return "undertraining"
    if text in {"well-balanced", "well balanced", "balanced", "well_balanced"}:
        return "well-balanced"
    if text in {"inconclusive"}:
        return "inconclusive"
    if text in {"unavailable"}:
        return "unavailable"
    if text in {"maybe", "medium", "moderate", "possible", "uncertain"}:
        return "inconclusive"
    if text in {"no", "n", "low", "false", "0", "not overtrained"}:
        return "well-balanced"
    return text


def mark_overtraining_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    notes = df.get("notes", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    classification = df.get("training_classification", pd.Series("", index=df.index)).map(
        normalize_training_classification
    )
    legacy_labels = df.get("overtraining_label", pd.Series("", index=df.index)).map(
        normalize_training_classification
    )
    notes_flag = notes.str.contains("over training|overtraining", regex=True)
    classification = classification.where(classification != "", legacy_labels)
    classification = classification.where(classification != "", np.where(notes_flag, "overtraining", ""))

    df["training_classification_normalized"] = classification
    df["is_overtraining_flagged"] = classification.eq("overtraining")
    df["is_undertraining_flagged"] = classification.eq("undertraining")
    df["is_inconclusive_flagged"] = classification.eq("inconclusive")
    df["is_unavailable_flagged"] = classification.eq("unavailable")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = [column for column in X.columns if column not in categorical_features]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="N/A")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )


def build_surrogate_model(X: pd.DataFrame, y: pd.Series, random_state: int) -> Pipeline:
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=500,
                    random_state=random_state,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def normalized_learning_rate(value: object) -> object:
    if pd.isna(value):
        return "N/A"

    text = str(value).strip()
    if text.lower() in {"exponential decay", "exp decay", "exponential"}:
        return "Exponential decay"

    try:
        return float(text)
    except ValueError:
        return text


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[FEATURE_COLUMNS].copy()
    features["learning_rate"] = features["learning_rate"].map(normalized_learning_rate)

    categorical_columns = [
        "layer1_activation_function",
        "layer2_activation_function",
        "learning_rate",
    ]

    for column in categorical_columns:
        features[column] = features[column].fillna("N/A").astype(str).str.strip()

    numeric_columns = [column for column in features.columns if column not in categorical_columns]
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    return features


def normalize_layer_counts(df_features: pd.DataFrame) -> pd.DataFrame:
    normalized = df_features.copy()
    layer2_columns = [
        "layer2_activation_function",
        "layer2_filters",
        "layer2_kernel_size",
        "layer2_pool_size",
        "layer2_stride",
    ]
    single_layer_mask = pd.to_numeric(
        normalized["number_of_layers"],
        errors="coerce",
    ).fillna(1).astype(int).eq(1)

    normalized.loc[single_layer_mask, "layer2_activation_function"] = "N/A"
    for column in layer2_columns[1:]:
        normalized.loc[single_layer_mask, column] = np.nan

    return normalized


def unique_sorted(values: Iterable[object]) -> List[object]:
    clean_values = [value for value in values if not pd.isna(value)]
    try:
        return sorted(set(clean_values))
    except TypeError:
        return sorted({str(value) for value in clean_values})


def unique_numeric(values: Iterable[object]) -> List[float]:
    clean = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().tolist()
    return sorted(set(float(value) for value in clean))


def compute_numeric_default(
    values: Iterable[object],
    fallback: float,
    integer: bool = False,
) -> float | int:
    observed = unique_numeric(values)
    if not observed:
        return int(fallback) if integer else float(fallback)

    counts = pd.Series(observed).value_counts()
    if not counts.empty:
        default = float(counts.index[0])
    else:
        default = float(np.median(observed))

    if integer:
        return int(round(default))
    return default


def get_early_stopping_defaults(df_features: pd.DataFrame) -> Dict[str, float | int]:
    return {
        "patience": compute_numeric_default(
            df_features["patience"],
            fallback=DEFAULT_PATIENCE,
            integer=True,
        ),
        "min_delta": compute_numeric_default(
            df_features["min_delta"],
            fallback=DEFAULT_MIN_DELTA,
            integer=False,
        ),
    }


def generate_integer_candidates(
    values: Iterable[object],
    minimum: int | None = None,
    maximum: int | None = None,
    extend: int = 1,
) -> List[int]:
    observed = [int(round(value)) for value in unique_numeric(values)]
    if not observed:
        return []

    low = min(observed)
    high = max(observed)
    if minimum is None:
        minimum = max(1, low - extend)
    if maximum is None:
        maximum = high + extend

    candidates = set(observed)
    candidates.update(range(minimum, maximum + 1))

    for a, b in zip(observed[:-1], observed[1:]):
        midpoint = int(round((a + b) / 2.0))
        if minimum <= midpoint <= maximum:
            candidates.add(midpoint)

    return sorted(value for value in candidates if minimum <= value <= maximum)


def generate_float_candidates(
    values: Iterable[object],
    expand_ratio: float = 0.25,
) -> List[float]:
    observed = unique_numeric(values)
    if not observed:
        return []
    if len(observed) == 1:
        value = observed[0]
        return [round(value, 6)]

    candidates = set(observed)
    for a, b in zip(observed[:-1], observed[1:]):
        candidates.add((a + b) / 2.0)

    low = min(observed)
    high = max(observed)
    span = high - low
    if span > 0:
        candidates.add(max(low - expand_ratio * span, 1e-12))
        candidates.add(high + expand_ratio * span)

    return sorted(round(value, 6) for value in candidates if value > 0)


def generate_log_candidates(values: Iterable[object]) -> List[float]:
    observed = unique_numeric(values)
    if not observed:
        return []
    positive = [value for value in observed if value > 0]
    if not positive:
        return observed
    if len(positive) == 1:
        return [round(positive[0], 6)]

    candidates = set(positive)
    logs = np.log10(np.asarray(positive, dtype=float))
    for a, b in zip(logs[:-1], logs[1:]):
        candidates.add(float(10 ** ((a + b) / 2.0)))

    low_log = float(logs.min())
    high_log = float(logs.max())
    if high_log > low_log:
        candidates.add(float(10 ** max(low_log - 0.5, -12)))
        candidates.add(float(10 ** (high_log + 0.5)))

    return sorted(round(value, 6) for value in candidates if value > 0)


def generate_learning_rate_candidates(values: Iterable[object]) -> List[object]:
    raw_values = [value for value in values if not pd.isna(value)]
    numeric_values: List[float] = []
    text_values: List[str] = []

    for value in raw_values:
        if isinstance(value, str):
            try:
                numeric_values.append(float(value))
            except ValueError:
                text_values.append(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            numeric_values.append(float(value))
        else:
            text_values.append(str(value))

    candidates: List[object] = []
    if numeric_values:
        candidates.extend(generate_float_candidates(numeric_values, expand_ratio=0.5))
    candidates.extend(sorted(set(text_values)))
    return candidates


def observed_configuration_signatures(df_features: pd.DataFrame) -> set[Tuple[str, ...]]:
    normalized = df_features.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: "N/A" if pd.isna(value) else str(value)
        )
    return {tuple(row) for row in normalized[FEATURE_COLUMNS].itertuples(index=False, name=None)}


def mark_novel_configurations(candidates: pd.DataFrame, observed_signatures: set[Tuple[str, ...]]) -> pd.DataFrame:
    candidates = candidates.copy()

    def signature(row: pd.Series) -> Tuple[str, ...]:
        values: List[str] = []
        for column in FEATURE_COLUMNS:
            value = row[column]
            values.append("N/A" if pd.isna(value) else str(value))
        return tuple(values)

    candidates["is_novel_configuration"] = ~candidates.apply(signature, axis=1).isin(observed_signatures)
    return candidates


def build_candidate_grid(
    df_features: pd.DataFrame,
    random_state: int = 42,
    max_candidates: int = 30000,
    early_stopping_defaults: Dict[str, float | int] | None = None,
) -> pd.DataFrame:
    early_stopping_defaults = early_stopping_defaults or {
        "patience": DEFAULT_PATIENCE,
        "min_delta": DEFAULT_MIN_DELTA,
    }
    search_space = {
        "number_of_layers": generate_integer_candidates(df_features["number_of_layers"], minimum=1, maximum=3),
        "layer1_activation_function": unique_sorted(df_features["layer1_activation_function"]),
        "layer1_filters": generate_integer_candidates(df_features["layer1_filters"], minimum=1, maximum=16),
        "layer1_kernel_size": generate_integer_candidates(df_features["layer1_kernel_size"], minimum=1, maximum=8),
        "layer1_pool_size": generate_integer_candidates(df_features["layer1_pool_size"], minimum=1, maximum=6),
        "layer1_stride": generate_integer_candidates(df_features["layer1_stride"], minimum=1, maximum=6),
        "learning_rate": generate_learning_rate_candidates(df_features["learning_rate"]),
        "batch_size": generate_integer_candidates(df_features["batch_size"], minimum=4, maximum=64),
        "patience": generate_integer_candidates(df_features["patience"], minimum=1, maximum=30)
        or [int(early_stopping_defaults["patience"])],
        "min_delta": generate_log_candidates(df_features["min_delta"])
        or [float(early_stopping_defaults["min_delta"])],
    }

    layer2_space = {
        "layer2_activation_function": unique_sorted(df_features["layer2_activation_function"]),
        "layer2_filters": generate_integer_candidates(df_features["layer2_filters"], minimum=1, maximum=16),
        "layer2_kernel_size": generate_integer_candidates(df_features["layer2_kernel_size"], minimum=1, maximum=8),
        "layer2_pool_size": generate_integer_candidates(df_features["layer2_pool_size"], minimum=1, maximum=6),
        "layer2_stride": generate_integer_candidates(df_features["layer2_stride"], minimum=1, maximum=6),
    }

    rng = np.random.RandomState(random_state)

    observed_rows = df_features[FEATURE_COLUMNS].copy()
    candidates: List[Dict[str, object]] = observed_rows.to_dict(orient="records")
    seen = {
        tuple("N/A" if pd.isna(row[col]) else str(row[col]) for col in FEATURE_COLUMNS)
        for _, row in observed_rows.iterrows()
    }

    base_keys = list(search_space.keys())
    layer2_keys = list(layer2_space.keys())

    while len(candidates) < max_candidates:
        row = {key: rng.choice(search_space[key]) for key in base_keys}
        row["number_of_layers"] = int(row["number_of_layers"])

        if row["number_of_layers"] == 1:
            row.update(
                {
                    "layer2_activation_function": "N/A",
                    "layer2_filters": np.nan,
                    "layer2_kernel_size": np.nan,
                    "layer2_pool_size": np.nan,
                    "layer2_stride": np.nan,
                }
            )
        else:
            for key in layer2_keys:
                row[key] = rng.choice(layer2_space[key])

        signature = tuple("N/A" if pd.isna(row[col]) else str(row[col]) for col in FEATURE_COLUMNS)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(row)

    return pd.DataFrame(candidates, columns=FEATURE_COLUMNS).reset_index(drop=True)


def summarize_configuration(
    row: pd.Series,
    early_stopping_defaults: Dict[str, float | int] | None = None,
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    early_stopping_defaults = early_stopping_defaults or {
        "patience": DEFAULT_PATIENCE,
        "min_delta": DEFAULT_MIN_DELTA,
    }
    for column in FEATURE_COLUMNS:
        value = row[column]
        if pd.isna(value):
            summary[column] = early_stopping_defaults.get(column, "N/A")
        elif isinstance(value, np.generic):
            summary[column] = value.item()
        else:
            summary[column] = value
    return summary


def summarize_prediction(row: pd.Series) -> Dict[str, object]:
    summary = {
        "configuration": summarize_configuration(row),
        "is_novel_configuration": bool(row["is_novel_configuration"]),
        "predicted_average_rmse": float(row["predicted_average_rmse"]),
        "predicted_average_r2": float(row["predicted_average_r2"]),
        "predicted_score_raw": float(row["predicted_score_raw"]),
        "predicted_score": float(row["predicted_score"]),
        "overtraining_risk": str(row["overtraining_risk"]),
        "overtraining_reason": str(row["overtraining_reason"]),
        "classification_preference": str(row.get("classification_preference", "")),
    }
    for column in ["predicted_epoch", "predicted_training_time_seconds", "predicted_training_time_hours"]:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            summary[column] = float(value)
    return summary


def summarize_recommendation(
    row: pd.Series,
    early_stopping_defaults: Dict[str, float | int] | None = None,
) -> Dict[str, object]:
    summary = summarize_prediction(row)
    summary["configuration"] = summarize_configuration(row, early_stopping_defaults)
    return summary


def _format_scalar(value: object, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return str(value)


def build_recommendation_comparison_table(results: Dict[str, object]) -> str:
    observed = results["best_observed"]
    predicted = results["best_accuracy_recommendation"]
    conservative = results["best_conservative_recommendation"]
    balanced = results["best_balanced_accuracy_recommendation"]

    rows = [
        {
            "Option": "Current Best",
            "Source": "Observed",
            "Avg RMSE": _format_scalar(observed.get("average_rmse")),
            "Avg R2": _format_scalar(observed.get("average_r2")),
            "Score": _format_scalar(observed.get("score")),
            "Risk": "high" if observed.get("is_overtraining_flagged") else observed.get("training_classification", "-"),
            "Novel": "No",
            "Layers": _format_scalar(observed["configuration"].get("number_of_layers"), decimals=0),
            "LR": _format_scalar(observed["configuration"].get("learning_rate"), decimals=6),
            "Batch": _format_scalar(observed["configuration"].get("batch_size"), decimals=0),
            "Patience": _format_scalar(observed["configuration"].get("patience"), decimals=0),
            "Min Delta": _format_scalar(observed["configuration"].get("min_delta"), decimals=6),
            "Epoch": _format_scalar(observed.get("epoch"), decimals=0),
            "Hours": _format_scalar(observed.get("training_time_hours"), decimals=2),
        },
        {
            "Option": "Accuracy Rec.",
            "Source": "Predicted",
            "Avg RMSE": _format_scalar(predicted.get("predicted_average_rmse")),
            "Avg R2": _format_scalar(predicted.get("predicted_average_r2")),
            "Score": _format_scalar(predicted.get("predicted_score")),
            "Risk": _format_scalar(predicted.get("overtraining_risk")),
            "Novel": "Yes" if predicted.get("is_novel_configuration") else "No",
            "Layers": _format_scalar(predicted["configuration"].get("number_of_layers"), decimals=0),
            "LR": _format_scalar(predicted["configuration"].get("learning_rate"), decimals=6),
            "Batch": _format_scalar(predicted["configuration"].get("batch_size"), decimals=0),
            "Patience": _format_scalar(predicted["configuration"].get("patience"), decimals=0),
            "Min Delta": _format_scalar(predicted["configuration"].get("min_delta"), decimals=6),
            "Epoch": _format_scalar(predicted.get("predicted_epoch"), decimals=0),
            "Hours": _format_scalar(predicted.get("predicted_training_time_hours"), decimals=2),
        },
        {
            "Option": "Conservative",
            "Source": "Predicted",
            "Avg RMSE": _format_scalar(conservative.get("predicted_average_rmse")),
            "Avg R2": _format_scalar(conservative.get("predicted_average_r2")),
            "Score": _format_scalar(conservative.get("predicted_score")),
            "Risk": _format_scalar(conservative.get("overtraining_risk")),
            "Novel": "Yes" if conservative.get("is_novel_configuration") else "No",
            "Layers": _format_scalar(conservative["configuration"].get("number_of_layers"), decimals=0),
            "LR": _format_scalar(conservative["configuration"].get("learning_rate"), decimals=6),
            "Batch": _format_scalar(conservative["configuration"].get("batch_size"), decimals=0),
            "Patience": _format_scalar(conservative["configuration"].get("patience"), decimals=0),
            "Min Delta": _format_scalar(conservative["configuration"].get("min_delta"), decimals=6),
            "Epoch": _format_scalar(conservative.get("predicted_epoch"), decimals=0),
            "Hours": _format_scalar(conservative.get("predicted_training_time_hours"), decimals=2),
        },
        {
            "Option": "Balanced Acc.",
            "Source": "Predicted",
            "Avg RMSE": _format_scalar(balanced.get("predicted_average_rmse")),
            "Avg R2": _format_scalar(balanced.get("predicted_average_r2")),
            "Score": _format_scalar(balanced.get("predicted_score")),
            "Risk": _format_scalar(balanced.get("overtraining_risk")),
            "Novel": "Yes" if balanced.get("is_novel_configuration") else "No",
            "Layers": _format_scalar(balanced["configuration"].get("number_of_layers"), decimals=0),
            "LR": _format_scalar(balanced["configuration"].get("learning_rate"), decimals=6),
            "Batch": _format_scalar(balanced["configuration"].get("batch_size"), decimals=0),
            "Patience": _format_scalar(balanced["configuration"].get("patience"), decimals=0),
            "Min Delta": _format_scalar(balanced["configuration"].get("min_delta"), decimals=6),
            "Epoch": _format_scalar(balanced.get("predicted_epoch"), decimals=0),
            "Hours": _format_scalar(balanced.get("predicted_training_time_hours"), decimals=2),
        },
    ]

    columns = list(rows[0].keys())
    widths = {
        column: max(len(column), max(len(str(row[column])) for row in rows))
        for column in columns
    }

    def render_row(row: Dict[str, object]) -> str:
        return " | ".join(str(row[column]).ljust(widths[column]) for column in columns)

    header = render_row({column: column for column in columns})
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = [render_row(row) for row in rows]
    return "\n".join([header, separator, *body])


def fit_auxiliary_surrogates(
    features: pd.DataFrame,
    experiments: pd.DataFrame,
    random_state: int,
) -> Dict[str, Pipeline]:
    targets = {
        "predicted_epoch": "epoch",
        "predicted_training_time_seconds": "training_time_seconds",
        "predicted_training_time_hours": "training_time_hours",
    }
    models: Dict[str, Pipeline] = {}

    for prediction_name, experiment_column in targets.items():
        if experiment_column not in experiments.columns:
            continue
        target = pd.to_numeric(experiments[experiment_column], errors="coerce")
        mask = target.notna()
        if mask.sum() < 3:
            continue
        models[prediction_name] = build_surrogate_model(
            features.loc[mask],
            target.loc[mask],
            random_state=random_state,
        )

    return models


def estimate_overtraining_risk(
    candidates: pd.DataFrame,
    experiments: pd.DataFrame,
) -> pd.DataFrame:
    candidates = candidates.copy()

    flagged = experiments[experiments["is_overtraining_flagged"]].copy()
    undertrained = experiments[experiments.get("is_undertraining_flagged", False)].copy()
    inconclusive = experiments[experiments.get("is_inconclusive_flagged", False)].copy()
    unavailable = experiments[experiments.get("is_unavailable_flagged", False)].copy()
    well_balanced = experiments[
        experiments["training_classification_normalized"].eq("well-balanced")
    ].copy()
    non_overtrained = experiments[~experiments["is_overtraining_flagged"]].copy()

    candidates["overtraining_risk"] = "low"
    candidates["overtraining_penalty"] = 0.0
    candidates["overtraining_reason"] = ""
    candidates["classification_preference"] = ""

    if flagged.empty or non_overtrained.empty:
        return candidates

    max_non_overtrained_patience = pd.to_numeric(non_overtrained["patience"], errors="coerce").max()
    min_non_overtrained_min_delta = pd.to_numeric(non_overtrained["min_delta"], errors="coerce").min()

    patience_values_flagged = set(pd.to_numeric(flagged["patience"], errors="coerce").dropna().tolist())
    min_delta_values_flagged = set(pd.to_numeric(flagged["min_delta"], errors="coerce").dropna().tolist())

    exact_flagged_mask = (
        candidates["patience"].isin(patience_values_flagged)
        & candidates["min_delta"].isin(min_delta_values_flagged)
    )
    extrapolated_mask = (
        (pd.to_numeric(candidates["patience"], errors="coerce") > max_non_overtrained_patience)
        | (pd.to_numeric(candidates["min_delta"], errors="coerce") < min_non_overtrained_min_delta)
    )

    candidates.loc[exact_flagged_mask, "overtraining_risk"] = "high"
    candidates.loc[exact_flagged_mask, "overtraining_penalty"] = 0.12
    candidates.loc[
        exact_flagged_mask,
        "overtraining_reason",
    ] = "Matches the patience/min_delta region already flagged as overtraining."

    medium_mask = extrapolated_mask & ~exact_flagged_mask
    candidates.loc[medium_mask, "overtraining_risk"] = "medium"
    candidates.loc[medium_mask, "overtraining_penalty"] = 0.03
    candidates.loc[
        medium_mask,
        "overtraining_reason",
    ] = "Moves beyond the non-overtrained early-stopping settings seen in the workbook."

    near_undertrained_mask = pd.Series(False, index=candidates.index)
    if not undertrained.empty:
        undertrained_patience = set(pd.to_numeric(undertrained["patience"], errors="coerce").dropna().tolist())
        undertrained_delta = set(pd.to_numeric(undertrained["min_delta"], errors="coerce").dropna().tolist())
        near_undertrained_mask = (
            candidates["patience"].isin(undertrained_patience)
            & candidates["min_delta"].isin(undertrained_delta)
            & ~exact_flagged_mask
            & ~medium_mask
        )
        candidates.loc[near_undertrained_mask, "overtraining_risk"] = "medium"
        candidates.loc[near_undertrained_mask, "overtraining_penalty"] = 0.01
        candidates.loc[
            near_undertrained_mask,
            "overtraining_reason",
        ] = "Matches an early-stopping region previously labeled as undertraining, which may still allow higher accuracy."

    for subset, reason in [
        (inconclusive, "Matches a region previously labeled inconclusive."),
        (unavailable, "Matches a region previously labeled unavailable."),
    ]:
        if subset.empty:
            continue
        subset_patience = set(pd.to_numeric(subset["patience"], errors="coerce").dropna().tolist())
        subset_delta = set(pd.to_numeric(subset["min_delta"], errors="coerce").dropna().tolist())
        subset_mask = (
            candidates["patience"].isin(subset_patience)
            & candidates["min_delta"].isin(subset_delta)
            & ~exact_flagged_mask
            & ~medium_mask
            & ~near_undertrained_mask
        )
        candidates.loc[subset_mask, "overtraining_risk"] = "medium"
        candidates.loc[subset_mask, "overtraining_penalty"] = 0.02
        candidates.loc[subset_mask, "overtraining_reason"] = reason

    if not well_balanced.empty:
        balanced_patience = set(pd.to_numeric(well_balanced["patience"], errors="coerce").dropna().tolist())
        balanced_delta = set(pd.to_numeric(well_balanced["min_delta"], errors="coerce").dropna().tolist())
        balanced_mask = (
            candidates["patience"].isin(balanced_patience)
            & candidates["min_delta"].isin(balanced_delta)
            & ~exact_flagged_mask
        )
        candidates.loc[balanced_mask, "overtraining_penalty"] -= 0.005
        candidates.loc[balanced_mask, "classification_preference"] = (
            "Small bonus for matching a previously well-balanced stopping region."
        )

    return candidates


def select_recommendations(candidate_predictions: pd.DataFrame) -> Dict[str, pd.Series]:
    aggressive_pool = candidate_predictions[
        candidate_predictions["overtraining_risk"] != "high"
    ].copy()
    if aggressive_pool.empty:
        aggressive_pool = candidate_predictions.copy()
    aggressive_pool = aggressive_pool.sort_values(
        by=[
            "predicted_average_r2",
            "predicted_average_rmse",
            "predicted_score",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    conservative_pool = candidate_predictions[
        candidate_predictions["overtraining_risk"] == "low"
    ].copy()
    if conservative_pool.empty:
        conservative_pool = aggressive_pool.copy()

    conservative_pool["conservative_score"] = (
        conservative_pool["predicted_score"]
        + np.where(conservative_pool["is_novel_configuration"], 0.01, 0.0)
        - np.where(conservative_pool["classification_preference"] != "", 0.005, 0.0)
    )
    conservative_pool = conservative_pool.sort_values(
        by=[
            "conservative_score",
            "predicted_average_r2",
            "predicted_average_rmse",
        ],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    balanced_accuracy_pool = candidate_predictions[
        candidate_predictions["overtraining_risk"] == "low"
    ].copy()
    if balanced_accuracy_pool.empty:
        balanced_accuracy_pool = aggressive_pool.copy()
    balanced_accuracy_pool["balanced_accuracy_score"] = (
        balanced_accuracy_pool["predicted_score"]
        + np.where(balanced_accuracy_pool["is_novel_configuration"], 0.005, 0.0)
    )
    balanced_accuracy_pool = balanced_accuracy_pool.sort_values(
        by=[
            "predicted_average_r2",
            "predicted_average_rmse",
            "balanced_accuracy_score",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return {
        "aggressive": aggressive_pool.loc[0],
        "conservative": conservative_pool.loc[0],
        "balanced_accuracy": balanced_accuracy_pool.loc[0],
    }


def optimize_hyperparameters(
    excel_path: Path,
    sheet_name: str | None = None,
    exclude_overtraining: bool = False,
    random_state: int = 42,
) -> Dict[str, object]:
    experiments = load_experiment_table(excel_path, sheet_name=sheet_name)
    experiments = compute_average_metrics(experiments)
    experiments = mark_overtraining_rows(experiments)

    if exclude_overtraining:
        experiments = exclude_overtraining_rows(experiments)
        experiments = mark_overtraining_rows(experiments)

    features = prepare_features(experiments)
    features = normalize_layer_counts(features)
    early_stopping_defaults = get_early_stopping_defaults(features)
    for column, default in early_stopping_defaults.items():
        features[column] = features[column].fillna(default)

    candidate_grid = build_candidate_grid(
        features,
        early_stopping_defaults=early_stopping_defaults,
    )
    candidate_grid = normalize_layer_counts(candidate_grid)
    observed_signatures = observed_configuration_signatures(features)
    candidate_grid = mark_novel_configurations(candidate_grid, observed_signatures)

    rmse_model = build_surrogate_model(features, experiments["average_rmse"], random_state=random_state)
    r2_model = build_surrogate_model(features, experiments["average_r2"], random_state=random_state + 1)
    auxiliary_models = fit_auxiliary_surrogates(features, experiments, random_state=random_state + 2)

    candidate_predictions = candidate_grid.copy()
    candidate_predictions["predicted_average_rmse"] = rmse_model.predict(candidate_grid)
    candidate_predictions["predicted_average_r2"] = r2_model.predict(candidate_grid)
    for prediction_name, model in auxiliary_models.items():
        candidate_predictions[prediction_name] = model.predict(candidate_grid)
    candidate_predictions = estimate_overtraining_risk(candidate_predictions, experiments)
    candidate_predictions["predicted_score_raw"] = (
        candidate_predictions["predicted_average_rmse"]
        + (1.0 - candidate_predictions["predicted_average_r2"])
    )
    candidate_predictions["predicted_score"] = (
        candidate_predictions["predicted_score_raw"] + candidate_predictions["overtraining_penalty"]
    )
    candidate_predictions = candidate_predictions.sort_values(
        by=["predicted_score", "predicted_average_rmse", "predicted_average_r2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    recommendations = select_recommendations(candidate_predictions)

    observed = pd.concat(
        [
            features,
            experiments[
                [
                    "average_rmse",
                    "average_r2",
                    "optimization_score",
                    "training_classification_normalized",
                    "is_overtraining_flagged",
                    "notes",
                ]
            ],
        ],
        axis=1,
    )
    best_observed_idx = observed["optimization_score"].idxmin()
    best_observed = observed.loc[best_observed_idx]
    best_predicted = recommendations["aggressive"]
    best_conservative = recommendations["conservative"]
    best_balanced_accuracy = recommendations["balanced_accuracy"]

    return {
        "excel_path": str(excel_path),
        "sheet_name": sheet_name,
        "rows_used": int(len(experiments)),
        "candidate_count": int(len(candidate_predictions)),
        "best_observed": {
            "configuration": summarize_configuration(best_observed, early_stopping_defaults),
            "average_rmse": float(best_observed["average_rmse"]),
            "average_r2": float(best_observed["average_r2"]),
            "score": float(best_observed["optimization_score"]),
            "epoch": None if pd.isna(best_observed.get("epoch")) else float(best_observed["epoch"]),
            "training_time_seconds": None
            if pd.isna(best_observed.get("training_time_seconds"))
            else float(best_observed["training_time_seconds"]),
            "training_time_hours": None
            if pd.isna(best_observed.get("training_time_hours"))
            else float(best_observed["training_time_hours"]),
            "training_classification": ""
            if pd.isna(best_observed.get("training_classification_normalized"))
            else str(best_observed.get("training_classification_normalized")),
            "is_overtraining_flagged": bool(best_observed.get("is_overtraining_flagged", False)),
            "notes": "" if pd.isna(best_observed.get("notes")) else str(best_observed.get("notes")),
        },
        "best_predicted": summarize_recommendation(best_predicted, early_stopping_defaults),
        "best_accuracy_recommendation": summarize_recommendation(best_predicted, early_stopping_defaults),
        "best_conservative_recommendation": summarize_recommendation(best_conservative, early_stopping_defaults),
        "best_balanced_accuracy_recommendation": summarize_recommendation(
            best_balanced_accuracy,
            early_stopping_defaults,
        ),
        "top_10_predicted": candidate_predictions.head(10).to_dict(orient="records"),
        "experiments_with_metrics": pd.concat(
            [
                features,
                experiments[
                    [
                        "average_rmse",
                        "average_r2",
                        "average_rmse_recomputed",
                        "average_r2_recomputed",
                        "optimization_score",
                        "epoch",
                        "training_time_seconds",
                        "training_time_hours",
                        "training_classification_normalized",
                        "is_overtraining_flagged",
                    ]
                ],
            ],
            axis=1,
        )
        .sort_values(by=["optimization_score", "average_rmse", "average_r2"], ascending=[True, True, False])
        .to_dict(orient="records"),
    }


def format_cli_report(results: Dict[str, object]) -> str:
    observed = results["best_observed"]
    predicted = results["best_accuracy_recommendation"]
    conservative = results["best_conservative_recommendation"]
    balanced = results["best_balanced_accuracy_recommendation"]
    comparison_table = build_recommendation_comparison_table(results)

    lines = [
        f"Rows used: {results['rows_used']}",
        f"Candidate configurations searched: {results['candidate_count']}",
        "",
        "Recommendation comparison table:",
        comparison_table,
        "",
        "Best observed configuration in the workbook:",
        json.dumps(observed, indent=2),
        "",
        "Best accuracy-focused recommendation (strongly avoids overtraining):",
        json.dumps(predicted, indent=2),
        "",
        "Best conservative recommendation (low-risk and closer to known-good regions):",
        json.dumps(conservative, indent=2),
        "",
        "Best balanced high-accuracy recommendation (maximize R^2 within the low-risk subset):",
        json.dumps(balanced, indent=2),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read the comparison workbook, compute average test RMSE/R^2 over the five output variables, "
            "and search for the hyperparameter configuration with the best predicted overall performance while strongly avoiding overtraining."
        )
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=DEFAULT_EXCEL_PATH,
        help="Path to the comparison workbook.",
    )
    parser.add_argument(
        "--sheet-name",
        default=None,
        help="Excel sheet name to analyze. Defaults to the workbook's first sheet.",
    )
    parser.add_argument(
        "--exclude-overtraining",
        action="store_true",
        help="Ignore rows whose training classification is overtraining.",
    )
    args = parser.parse_args()

    results = optimize_hyperparameters(
        excel_path=args.excel_path,
        sheet_name=args.sheet_name,
        exclude_overtraining=args.exclude_overtraining,
    )
    print(format_cli_report(results))


if __name__ == "__main__":
    main()
