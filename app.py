import json
import os
import copy
import io
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import base64
import sys

from engine import compute_scores, ahp_weights, topsis_rank, REQUIRED_COLUMNS

st.set_page_config(page_title="Project Prioritization UI", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------
# Helpers and config
# ----------------------------
def resource_path(rel_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.dirname(__file__), rel_path)


def get_importable_database_dirs(include_bundle_dir: bool = True) -> list[str]:
    folder_name = "Importable Database"
    base_dirs = []

    # Current working directory (dev / streamlit cloud runs)
    try:
        base_dirs.append(os.getcwd())
    except Exception:
        pass

    # Project/script directory
    base_dirs.append(os.path.dirname(__file__))

    # PyInstaller temp bundle dir (if bundled)
    if include_bundle_dir and hasattr(sys, "_MEIPASS"):
        base_dirs.append(sys._MEIPASS)

    # Folder next to executable (for distributed EXE usage)
    if getattr(sys, "frozen", False):
        base_dirs.append(os.path.dirname(sys.executable))

    dirs = []
    seen = set()
    for base in base_dirs:
        db_dir = os.path.abspath(os.path.join(base, folder_name))
        if db_dir in seen:
            continue
        seen.add(db_dir)
        dirs.append(db_dir)
    return dirs


def get_importable_database_files(extensions: tuple[str, ...] = (".csv",)) -> list[tuple[str, str]]:
    db_dirs = get_importable_database_dirs(include_bundle_dir=True)
    seen = set()
    files: list[tuple[str, str]] = []
    for db_dir in db_dirs:
        if not os.path.isdir(db_dir):
            continue
        for name in sorted(os.listdir(db_dir)):
            if not name.lower().endswith(tuple(ext.lower() for ext in extensions)):
                continue
            p = os.path.abspath(os.path.join(db_dir, name))
            if p in seen:
                continue
            seen.add(p)
            files.append((name, p))
    return files


def get_importable_workspace_json_files() -> list[tuple[str, str]]:
    return get_importable_database_files(extensions=(".json",))


def get_workspace_export_dir() -> str | None:
    # Prefer writable project/exe-adjacent Importable Database folder.
    candidates = get_importable_database_dirs(include_bundle_dir=False)
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            test_path = os.path.join(d, ".write_test.tmp")
            with open(test_path, "w", encoding="utf-8") as tf:
                tf.write("ok")
            os.remove(test_path)
            return d
        except Exception:
            continue
    return None


def get_importable_json_files() -> list[tuple[str, str]]:
    folder_name = "Importable Database"
    base_dirs = []
    try:
        base_dirs.append(os.getcwd())
    except Exception:
        pass
    base_dirs.append(os.path.dirname(__file__))
    if hasattr(sys, "_MEIPASS"):
        base_dirs.append(sys._MEIPASS)
    if getattr(sys, "frozen", False):
        base_dirs.append(os.path.dirname(sys.executable))

    seen = set()
    files: list[tuple[str, str]] = []
    for base in base_dirs:
        # JSONs in base folder
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                if not name.lower().endswith(".json"):
                    continue
                p = os.path.abspath(os.path.join(base, name))
                if p in seen:
                    continue
                seen.add(p)
                files.append((name, p))
        # JSONs in Importable Database subfolder
        db_dir = os.path.join(base, folder_name)
        if os.path.isdir(db_dir):
            for name in sorted(os.listdir(db_dir)):
                if not name.lower().endswith(".json"):
                    continue
                p = os.path.abspath(os.path.join(db_dir, name))
                if p in seen:
                    continue
                seen.add(p)
                files.append((name, p))
    return files

with open(resource_path("scoring_config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

maps = config.get("mappings", {})
labels = config.get("labels", {})
SVI_RECLASS_COL = "svi_value_reclassified"

def label_for(group: str, key: str) -> str:
    try:
        return labels.get(group, {}).get(key, key)
    except Exception:
        return key

def classify_svi(value: float) -> str:
    if value < 0.25:
        return "low"
    if value < 0.5:
        return "low_moderate"
    if value < 0.75:
        return "moderate_high"
    return "high"


def find_column_case_insensitive(df: pd.DataFrame, target: str) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == target.lower():
            return c
    return None


def _norm_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def find_column_by_aliases(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {_norm_col_name(c): c for c in df.columns}
    for a in aliases:
        key = _norm_col_name(a)
        if key in normalized:
            return normalized[key]
    return None


def find_column_by_contains(df: pd.DataFrame, token: str) -> str | None:
    t = str(token).strip().lower()
    for c in df.columns:
        if t in str(c).strip().lower():
            return c
    return None


def classify_svi_series(values: pd.Series, mode: str = "normalized") -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce")

    if mode == "normalized":
        valid = numeric.dropna()
        if valid.empty:
            normalized = numeric.copy()
        else:
            vmin = float(valid.min())
            vmax = float(valid.max())
            if vmax > vmin:
                # Re-scale to 0.01..0.99 so outputs avoid hard 0/1 edges.
                normalized = ((numeric - vmin) / (vmax - vmin)) * 0.98 + 0.01
            else:
                normalized = pd.Series(0.5, index=numeric.index, dtype=float)
        class_input = normalized
    else:
        normalized = numeric
        class_input = numeric

    classes = class_input.apply(lambda x: classify_svi(float(x)) if pd.notna(x) else "")
    return normalized, classes


def preprocess_uploaded_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    svi_col = find_column_case_insensitive(df, "svi")
    if svi_col is None:
        df_auto, _ = auto_fill_derived_classes(df, overwrite=False)
        return df_auto

    svi_raw = pd.to_numeric(df[svi_col], errors="coerce")
    valid = svi_raw.dropna()
    use_raw = (not valid.empty) and bool(((valid >= 0) & (valid <= 1)).all())
    mode = "raw" if use_raw else "normalized"
    svi_norm, svi_class_auto = classify_svi_series(svi_raw, mode=mode)

    if SVI_RECLASS_COL not in df.columns:
        df[SVI_RECLASS_COL] = svi_norm
    else:
        existing_svi_value = pd.to_numeric(df[SVI_RECLASS_COL], errors="coerce")
        df[SVI_RECLASS_COL] = existing_svi_value.where(existing_svi_value.notna(), svi_norm)

    # Backward compatibility for older files/logic that still reference svi_value.
    if "svi_value" not in df.columns:
        df["svi_value"] = df[SVI_RECLASS_COL]

    if "svi_class" not in df.columns:
        df["svi_class"] = ""

    svi_class_existing = df["svi_class"].astype(str).str.strip()
    missing_class = df["svi_class"].isna() | (svi_class_existing == "")
    df.loc[missing_class, "svi_class"] = svi_class_auto[missing_class]

    df_auto, _ = auto_fill_derived_classes(df, overwrite=False)
    return df_auto


def classify_excess_rainfall_series(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        normalized = numeric.copy()
    else:
        vmin = float(valid.min())
        vmax = float(valid.max())
        if vmax > vmin:
            normalized = (numeric - vmin) / (vmax - vmin)
        else:
            normalized = pd.Series(0.5, index=numeric.index, dtype=float)

    classes = normalized.apply(
        lambda x: ("low" if x < 0.33 else ("intermediate" if x < 0.66 else "high")) if pd.notna(x) else ""
    )
    return normalized, classes


def _score_from_bins_local(value: float, bins: list[dict]) -> float:
    if pd.isna(value):
        return np.nan
    for b in bins:
        if float(value) <= float(b["max"]):
            return float(b["score"])
    return float(bins[-1]["score"]) if bins else np.nan


def _is_missing_series_value(v) -> bool:
    if pd.isna(v):
        return True
    return isinstance(v, str) and v.strip() == ""


def auto_fill_derived_classes(df_in: pd.DataFrame, overwrite: bool = False) -> tuple[pd.DataFrame, dict]:
    df = df_in.copy()
    changes = {"svi_class": 0, "people_efficiency_class": 0, "structures_efficiency_class": 0}

    # SVI class from value columns if class is missing (or overwrite=True)
    svi_source = None
    for c in [SVI_RECLASS_COL, "svi_value", find_column_case_insensitive(df, "svi")]:
        if c and c in df.columns:
            svi_source = c
            break
    if svi_source:
        svi_norm, svi_class_auto = classify_svi_series(df[svi_source], mode="raw")
        # if values are outside 0..1, normalize first
        valid = pd.to_numeric(df[svi_source], errors="coerce").dropna()
        if (not valid.empty) and (not ((valid >= 0) & (valid <= 1)).all()):
            svi_norm, svi_class_auto = classify_svi_series(df[svi_source], mode="normalized")

        if "svi_class" not in df.columns:
            df["svi_class"] = ""
        if SVI_RECLASS_COL not in df.columns:
            df[SVI_RECLASS_COL] = svi_norm
        if "svi_value" not in df.columns:
            df["svi_value"] = svi_norm

        if overwrite:
            target_mask = pd.Series(True, index=df.index)
        else:
            target_mask = df["svi_class"].apply(_is_missing_series_value)
        changes["svi_class"] = int(target_mask.sum())
        df.loc[target_mask, "svi_class"] = svi_class_auto[target_mask]
        df.loc[target_mask, SVI_RECLASS_COL] = svi_norm[target_mask]
        df.loc[target_mask, "svi_value"] = svi_norm[target_mask]

    # Efficiency classes from cost/benefitted
    if all(c in df.columns for c in ["total_cost", "people_benefitted"]):
        cost = pd.to_numeric(df["total_cost"], errors="coerce")
        people = pd.to_numeric(df["people_benefitted"], errors="coerce")
        cpp = cost / people.replace(0, np.nan)
        bins_people = config.get("efficiency_bins", {}).get("people_cost_per_person", [])
        people_class = cpp.apply(lambda v: str(int(_score_from_bins_local(v, bins_people))) if not pd.isna(_score_from_bins_local(v, bins_people)) else "")
        if "people_efficiency_class" not in df.columns:
            df["people_efficiency_class"] = ""
        target_mask = pd.Series(True, index=df.index) if overwrite else df["people_efficiency_class"].apply(_is_missing_series_value)
        fill_mask = target_mask & people_class.ne("")
        changes["people_efficiency_class"] = int(fill_mask.sum())
        df.loc[fill_mask, "people_efficiency_class"] = people_class[fill_mask]

    if all(c in df.columns for c in ["total_cost", "structures_benefitted"]):
        cost = pd.to_numeric(df["total_cost"], errors="coerce")
        structs = pd.to_numeric(df["structures_benefitted"], errors="coerce")
        cps = cost / structs.replace(0, np.nan)
        bins_struct = config.get("efficiency_bins", {}).get("structures_cost_per_structure", [])
        struct_class = cps.apply(lambda v: str(int(_score_from_bins_local(v, bins_struct))) if not pd.isna(_score_from_bins_local(v, bins_struct)) else "")
        if "structures_efficiency_class" not in df.columns:
            df["structures_efficiency_class"] = ""
        target_mask = pd.Series(True, index=df.index) if overwrite else df["structures_efficiency_class"].apply(_is_missing_series_value)
        fill_mask = target_mask & struct_class.ne("")
        changes["structures_efficiency_class"] = int(fill_mask.sum())
        df.loc[fill_mask, "structures_efficiency_class"] = struct_class[fill_mask]

    return df, changes


def recalculate_efficiency_classes(df_in: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df_in.copy()
    changes = {"people_efficiency_class": 0, "structures_efficiency_class": 0}

    if all(c in df.columns for c in ["total_cost", "people_benefitted"]):
        cost = pd.to_numeric(df["total_cost"], errors="coerce")
        people = pd.to_numeric(df["people_benefitted"], errors="coerce")
        cpp = cost / people.replace(0, np.nan)
        bins_people = config.get("efficiency_bins", {}).get("people_cost_per_person", [])
        people_class = cpp.apply(lambda v: str(int(_score_from_bins_local(v, bins_people))) if not pd.isna(_score_from_bins_local(v, bins_people)) else "")
        if "people_efficiency_class" not in df.columns:
            df["people_efficiency_class"] = ""
        old_vals = df["people_efficiency_class"].astype(str)
        new_vals = people_class.astype(str)
        df["people_efficiency_class"] = new_vals
        changes["people_efficiency_class"] = int((old_vals != new_vals).sum())

    if all(c in df.columns for c in ["total_cost", "structures_benefitted"]):
        cost = pd.to_numeric(df["total_cost"], errors="coerce")
        structs = pd.to_numeric(df["structures_benefitted"], errors="coerce")
        cps = cost / structs.replace(0, np.nan)
        bins_struct = config.get("efficiency_bins", {}).get("structures_cost_per_structure", [])
        struct_class = cps.apply(lambda v: str(int(_score_from_bins_local(v, bins_struct))) if not pd.isna(_score_from_bins_local(v, bins_struct)) else "")
        if "structures_efficiency_class" not in df.columns:
            df["structures_efficiency_class"] = ""
        old_vals = df["structures_efficiency_class"].astype(str)
        new_vals = struct_class.astype(str)
        df["structures_efficiency_class"] = new_vals
        changes["structures_efficiency_class"] = int((old_vals != new_vals).sum())

    return df, changes


def read_uploaded_csv_with_id(uploaded_file) -> tuple[pd.DataFrame, str]:
    raw = uploaded_file.getvalue()
    file_hash = hashlib.md5(raw).hexdigest()
    upload_id = f"{uploaded_file.name}:{uploaded_file.size}:{file_hash}"
    df = pd.read_csv(io.BytesIO(raw))
    return df, upload_id


def _serialize_workspace_value(v):
    if isinstance(v, pd.DataFrame):
        return {"__type__": "dataframe", "value": v.to_dict(orient="split")}
    if isinstance(v, pd.Series):
        return {"__type__": "series", "value": v.to_list(), "index": v.index.to_list(), "name": v.name}
    if isinstance(v, np.ndarray):
        return {"__type__": "ndarray", "value": v.tolist()}
    if isinstance(v, dict):
        return {str(k): _serialize_workspace_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_serialize_workspace_value(x) for x in v]
    if isinstance(v, tuple):
        return {"__type__": "tuple", "value": [_serialize_workspace_value(x) for x in v]}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return {"__type__": "unsupported", "value": str(v)}


def _deserialize_workspace_value(v):
    if isinstance(v, dict) and "__type__" in v:
        t = v.get("__type__")
        if t == "dataframe":
            return pd.DataFrame(**v.get("value", {}))
        if t == "series":
            return pd.Series(v.get("value", []), index=v.get("index", None), name=v.get("name", None))
        if t == "ndarray":
            return np.array(v.get("value", []))
        if t == "tuple":
            return tuple(_deserialize_workspace_value(x) for x in v.get("value", []))
        if t == "unsupported":
            return v.get("value", "")
    if isinstance(v, dict):
        return {k: _deserialize_workspace_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_deserialize_workspace_value(x) for x in v]
    return v


def _workspace_state_snapshot() -> dict:
    excluded_prefixes = ("FormSubmitter:",)
    excluded_keys = {
        "sidebar_upload",
        "main_upload",
        "upload_workspace_bundle",
        "ahp_upload_matrix",
    }
    snapshot = {}
    for k in list(st.session_state.keys()):
        if k in excluded_keys:
            continue
        if any(str(k).startswith(p) for p in excluded_prefixes):
            continue
        snapshot[k] = _serialize_workspace_value(st.session_state.get(k))
    return snapshot


def build_workspace_bundle() -> dict:
    return {
        "version": 1,
        "df_work": st.session_state.get("df_work", pd.DataFrame()).to_dict(orient="split"),
        "uploaded_file_name": st.session_state.get("uploaded_file_name", ""),
        "custom_criteria": st.session_state.get("custom_criteria", []),
        "weights_pct": st.session_state.get("weights_pct", {}),
        "standard_column_mapping": st.session_state.get("standard_column_mapping", {}),
        "project_type_source_col": st.session_state.get("project_type_source_col", "(No mapping)"),
        "project_type_mode": st.session_state.get("project_type_mode", "Map from source values"),
        "subdivision_keywords": st.session_state.get("subdivision_keywords", ""),
        "channel_keywords": st.session_state.get("channel_keywords", ""),
        "ahp_selected_criteria": st.session_state.get("ahp_selected_criteria", []),
        "ahp_pairs": st.session_state.get("ahp_pairs", []),
        "ahp_weights": st.session_state.get("ahp_weights", {}),
        "ahp_cr": st.session_state.get("ahp_cr", None),
        "state_snapshot": _workspace_state_snapshot(),
    }


def apply_workspace_bundle(bundle: dict) -> None:
    if not isinstance(bundle, dict):
        raise ValueError("Invalid workspace file.")
    df_blob = bundle.get("df_work", {})
    if not isinstance(df_blob, dict) or "data" not in df_blob or "columns" not in df_blob:
        raise ValueError("Workspace file is missing dataset content.")
    st.session_state["df_work"] = pd.DataFrame(**df_blob)
    st.session_state["uploaded_file_name"] = bundle.get("uploaded_file_name", "workspace_import")
    st.session_state["custom_criteria"] = bundle.get("custom_criteria", [])
    st.session_state["weights_pct"] = bundle.get("weights_pct", st.session_state.get("weights_pct", {}))
    st.session_state["standard_column_mapping"] = bundle.get("standard_column_mapping", {})
    st.session_state["project_type_source_col"] = bundle.get("project_type_source_col", "(No mapping)")
    st.session_state["project_type_mode"] = bundle.get("project_type_mode", "Map from source values")
    st.session_state["subdivision_keywords"] = bundle.get("subdivision_keywords", "subdivision, subdiv, sub, local drainage, street")
    st.session_state["channel_keywords"] = bundle.get("channel_keywords", "channel, detention, ch, det, regional")
    st.session_state["ahp_selected_criteria"] = bundle.get("ahp_selected_criteria", [])
    st.session_state["ahp_pairs"] = bundle.get("ahp_pairs", [])
    st.session_state["ahp_weights"] = bundle.get("ahp_weights", {})
    if bundle.get("ahp_cr") is not None:
        st.session_state["ahp_cr"] = bundle.get("ahp_cr")
    snapshot = bundle.get("state_snapshot", {})
    if isinstance(snapshot, dict):
        for k, v in snapshot.items():
            if k in {"sidebar_upload", "main_upload", "upload_workspace_bundle", "ahp_upload_matrix"}:
                continue
            st.session_state[k] = _deserialize_workspace_value(v)
    # Force Custom Criteria Mapping table to rebuild from loaded dataset.
    st.session_state["custom_criteria_table_df"] = None
    st.session_state["custom_criteria_table_cols"] = []
    st.session_state.pop("project_data_editor", None)


def sync_project_data_editor() -> None:
    edited_df = st.session_state.get("project_data_editor")
    if isinstance(edited_df, pd.DataFrame):
        st.session_state["df_work"] = edited_df

EFFICIENCY_CLASSES = {
    "10": "Very High (Score 10)",
    "8": "High (Score 8)",
    "6": "Medium (Score 6)",
    "4": "Low (Score 4)",
    "1": "Very Low (Score 1)"
}

def get_efficiency_tables_html() -> str:
                # Single 4-column table with two group headers and a thick divider between groups
                return """
                <table style="width:100%;border-collapse:collapse;">
                    <tr>
                        <th colspan="2" style="background:#f0f0f0;text-align:left;padding:8px;border-bottom:2px solid #ccc;">Project Efficiency using People Benefitted Scoring Criteria</th>
                        <th colspan="2" style="background:#f0f0f0;text-align:left;padding:8px;border-bottom:2px solid #ccc;">Project Efficiency using Structures Benefitted Scoring Criteria</th>
                    </tr>
                    <tr style="background:#fafafa;"><th style="text-align:left;padding:8px;">Criteria</th><th style="text-align:center;padding:8px;">Score</th><th style="text-align:left;padding:8px;">Criteria</th><th style="text-align:center;padding:8px;">Score</th></tr>

                    <tr style="background:#d4edda;"><td style="padding:8px;">Less than $6,000/person</td><td style="text-align:center;padding:8px;">10</td><td style="padding:8px;">Less than $23,000/structure</td><td style="text-align:center;padding:8px;">10</td></tr>
                    <tr style="background:#c3e6cb;"><td style="padding:8px;">$6,000 to $15,000/person</td><td style="text-align:center;padding:8px;">8</td><td style="padding:8px;">$23,000 to $60,000/structure</td><td style="text-align:center;padding:8px;">8</td></tr>
                    <tr style="background:#fff3cd;"><td style="padding:8px;">$15,001 to $28,000/person</td><td style="text-align:center;padding:8px;">6</td><td style="padding:8px;">$60,001 to $106,000/structure</td><td style="text-align:center;padding:8px;">6</td></tr>
                    <tr style="background:#f8d7da;"><td style="padding:8px;">$28,001 to $77,000/person</td><td style="text-align:center;padding:8px;">4</td><td style="padding:8px;">$106,001 to $261,000/structure</td><td style="text-align:center;padding:8px;">4</td></tr>
                    <tr style="background:#f5c6cb;"><td style="padding:8px;">Greater than $77,000/person</td><td style="text-align:center;padding:8px;">1</td><td style="padding:8px;">Greater than $261,000/structure</td><td style="text-align:center;padding:8px;">1</td></tr>

                </table>
                """

def get_svi_html() -> str:
    # Flip color mapping so higher vulnerability is red (use wording from CSV)
    return """
    <table style="width:100%;border-collapse:collapse;">
      <tr style="background:#f0f0f0;"><th style="text-align:left;padding:8px;">Criteria</th><th style="text-align:center;padding:8px;">Score</th></tr>
      <tr style="background:#d4edda;"><td style="padding:8px;">SVI indicates low level of vulnerability</td><td style="text-align:center;padding:8px;">1</td></tr>
      <tr style="background:#fff3cd;"><td style="padding:8px;">SVI indicates low to moderate level of vulnerability</td><td style="text-align:center;padding:8px;">4</td></tr>
      <tr style="background:#f8d7da;"><td style="padding:8px;">SVI indicates moderate to high level of vulnerability</td><td style="text-align:center;padding:8px;">7</td></tr>
      <tr style="background:#f5c6cb;"><td style="padding:8px;">SVI indicates high level of vulnerability</td><td style="text-align:center;padding:8px;">10</td></tr>
    </table>
    """

def get_existing_conditions_channel_html() -> str:
    # Use exact wording from RawHCFCDTables.csv (Channel)
    return """
    <table style="width:100%;border-collapse:collapse;">
      <tr style="background:#f0f0f0;"><th style="text-align:left;padding:8px;">Criteria</th><th style="text-align:center;padding:8px;">Score</th></tr>
      <tr style="background:#d4edda;"><td style="padding:8px;">System capacity is &lt; 50% AEP storm (2-year storm)</td><td style="text-align:center;padding:8px;">10</td></tr>
      <tr style="background:#c3e6cb;"><td style="padding:8px;">System capacity is &lt; 20% AEP storm (5-year storm)</td><td style="text-align:center;padding:8px;">8</td></tr>
      <tr style="background:#fff3cd;"><td style="padding:8px;">System capacity is &lt; 10% AEP storm (10-year storm)</td><td style="text-align:center;padding:8px;">6</td></tr>
      <tr style="background:#f8d7da;"><td style="padding:8px;">System capacity is &lt; 4% AEP storm (25-year storm)</td><td style="text-align:center;padding:8px;">4</td></tr>
      <tr style="background:#f5c6cb;"><td style="padding:8px;">System capacity is &lt; 2% AEP storm (50-year storm)</td><td style="text-align:center;padding:8px;">2</td></tr>
      <tr style="background:#f5c6cb;"><td style="padding:8px;">System capacity is &gt; 1% AEP storm (100-year storm)</td><td style="text-align:center;padding:8px;">0</td></tr>
    </table>
    """

def get_existing_conditions_subdivision_html() -> str:
    # Use exact wording from RawHCFCDTables.csv (Subdivision)
    return """
    <table style="width:100%;border-collapse:collapse;">
      <tr style="background:#f0f0f0;"><th style="text-align:left;padding:8px;">Criteria</th><th style="text-align:center;padding:8px;">Score</th></tr>
      <tr style="background:#d4edda;"><td style="padding:8px;">Low estimated excess rainfall AND high-quality drainage infrastructure</td><td style="text-align:center;padding:8px;">0</td></tr>
      <tr style="background:#fff3cd;"><td style="padding:8px;">Intermediate estimated excess rainfall OR medium-quality drainage infrastructure (but not both)</td><td style="text-align:center;padding:8px;">3</td></tr>
      <tr style="background:#c3e6cb;"><td style="padding:8px;">Intermediate estimated excess rainfall AND medium-quality drainage infrastructure</td><td style="text-align:center;padding:8px;">6</td></tr>
      <tr style="background:#f8d7da;"><td style="padding:8px;">High estimated excess rainfall OR low-quality drainage infrastructure (but not both)</td><td style="text-align:center;padding:8px;">9</td></tr>
      <tr style="background:#f5c6cb;"><td style="padding:8px;">High estimated excess rainfall AND low-quality drainage infrastructure</td><td style="text-align:center;padding:8px;">10</td></tr>
    </table>
    """

def get_environment_channel_html() -> str:
    # Use exact wording from RawHCFCDTables.csv (Channel environmental)
    return """
    <table style="width:100%;border-collapse:collapse;">
      <tr style="background:#f0f0f0;"><th style="text-align:left;padding:8px;">Criteria</th><th style="text-align:center;padding:8px;">Score</th></tr>
      <tr style="background:#f5c6cb;"><td style="padding:8px;">Project will have significant environmental impacts requiring a Corps of Engineers Individual Permit and mitigation bank credits</td><td style="text-align:center;padding:8px;">0</td></tr>
      <tr style="background:#f8d7da;"><td style="padding:8px;">Project will have significant environmental impacts requiring mitigation bank credits</td><td style="text-align:center;padding:8px;">2</td></tr>
      <tr style="background:#fff3cd;"><td style="padding:8px;">Project is able to significantly avoid environmental impacts</td><td style="text-align:center;padding:8px;">6</td></tr>
      <tr style="background:#d4edda;"><td style="padding:8px;">Project has minimal or no environmental impacts</td><td style="text-align:center;padding:8px;">10</td></tr>
    </table>
    """

def get_environment_subdivision_html() -> str:
    # Use exact wording from RawHCFCDTables.csv (Subdivision environmental)
    return """
    <table style="width:100%;border-collapse:collapse;">
      <tr style="background:#f0f0f0;"><th style="text-align:left;padding:8px;">Criteria</th><th style="text-align:center;padding:8px;">Score</th></tr>
      <tr style="background:#f8d7da;"><td style="padding:8px;">Project will require acquiring additional right-of-way</td><td style="text-align:center;padding:8px;">6</td></tr>
      <tr style="background:#d4edda;"><td style="padding:8px;">Project can be completed within the road's existing right-of-way</td><td style="text-align:center;padding:8px;">10</td></tr>
    </table>
    """

def get_multiple_benefits_channel_html() -> str:
    # Use exact wording from RawHCFCDTables.csv (Multiple benefits channel)
    return """
    <table style="width:100%;border-collapse:collapse;">
      <tr style="background:#f0f0f0;"><th style="text-align:left;padding:8px;">Criteria</th><th style="text-align:center;padding:8px;">Score</th></tr>
      <tr style="background:#f5c6cb;"><td style="padding:8px;">Project does not have multiple benefits</td><td style="text-align:center;padding:8px;">0</td></tr>
      <tr style="background:#fff3cd;"><td style="padding:8px;">Project has recreational benefits</td><td style="text-align:center;padding:8px;">4</td></tr>
      <tr style="background:#fff3cd;"><td style="padding:8px;">Project has environmental enhancement benefits</td><td style="text-align:center;padding:8px;">6</td></tr>
      <tr style="background:#d4edda;"><td style="padding:8px;">Project has recreational and environmental enhancement benefits</td><td style="text-align:center;padding:8px;">10</td></tr>
    </table>
    """

def get_multiple_benefits_subdivision_html() -> str:
    # Use exact wording from RawHCFCDTables.csv (Multiple benefits subdivision)
    return """
    <table style="width:100%;border-collapse:collapse;">
      <tr style="background:#f0f0f0;"><th style="text-align:left;padding:8px;">Criteria</th><th style="text-align:center;padding:8px;">Score</th></tr>
      <tr style="background:#f8d7da;"><td style="padding:8px;">Project area does not benefit from a District improvement such as a nearby channel improvement or detention basin project</td><td style="text-align:center;padding:8px;">6</td></tr>
      <tr style="background:#d4edda;"><td style="padding:8px;">Project area also benefits from a District improvement such as a nearby channel improvement or detention basin project</td><td style="text-align:center;padding:8px;">10</td></tr>
    </table>
    """

# ----------------------------
# Session defaults
# ----------------------------
if "weights_pct" not in st.session_state:
    st.session_state["weights_pct"] = {k: float(v) * 100.0 for k, v in config["weights"].items()}
if "show_add_project" not in st.session_state:
    st.session_state["show_add_project"] = False
if "custom_criteria" not in st.session_state:
    st.session_state["custom_criteria"] = []
if "custom_criteria_table_rows" not in st.session_state:
    st.session_state["custom_criteria_table_rows"] = None
if "custom_criteria_table_cols" not in st.session_state:
    st.session_state["custom_criteria_table_cols"] = []
if "custom_criteria_table_df" not in st.session_state:
    st.session_state["custom_criteria_table_df"] = None
if "ahp_pairs_editor_version" not in st.session_state:
    st.session_state["ahp_pairs_editor_version"] = 0

BASE_CRITERIA_META = [
    ("people_efficiency", "Resident Benefits Efficiency"),
    ("structures_efficiency", "Structure Benefit Efficiency"),
    ("existing_conditions", "Existing Conditions"),
    ("svi", "Social Vulnerability Index"),
    ("maintenance", "Long-Term Maintenance Costs"),
    ("environment", "Minimizes Environmental Impacts"),
    ("multiple_benefits", "Potential for Multiple Benefits"),
]

SCORE_COL_MAP = {
    "people_efficiency": "score_people_efficiency",
    "structures_efficiency": "score_structures_efficiency",
    "existing_conditions": "score_existing_conditions",
    "svi": "score_svi",
    "maintenance": "score_maintenance",
    "environment": "score_environment",
    "multiple_benefits": "score_multiple_benefits",
}

STANDARD_COLUMNS = [
    "project_id",
    "project_name",
    "project_type",
    "total_cost",
    "people_benefitted",
    "structures_benefitted",
    "channel_capacity_class",
    "excess_rainfall_class",
    "drainage_infra_quality",
    "svi_value",
    "svi_value_reclassified",
    "svi_class",
    "maintenance_class",
    "people_efficiency_class",
    "structures_efficiency_class",
    "environment_channel_class",
    "row_subdivision_class",
    "multiple_benefits_channel_class",
    "district_improvement_synergy",
    "notes",
]

HCFCD_PARAMETER_FIELDS = [
    "project_name",
    "project_type",
    "total_cost",
    "people_benefitted",
    "structures_benefitted",
    "channel_capacity_class",
    "excess_rainfall_class",
    "drainage_infra_quality",
    "svi_value",
    "svi_class",
    "maintenance_class",
    "people_efficiency_class",
    "structures_efficiency_class",
    "environment_channel_class",
    "row_subdivision_class",
    "multiple_benefits_channel_class",
    "district_improvement_synergy",
]

HCFCD_PARAMETER_LABELS = {
    "project_name": "Project Name",
    "project_type": "Project Type",
    "total_cost": "Total Cost",
    "people_benefitted": "People Benefitted",
    "structures_benefitted": "Structures Benefitted",
    "channel_capacity_class": "Channel Capacity Class",
    "excess_rainfall_class": "Excess Rainfall Class",
    "drainage_infra_quality": "Drainage Infrastructure Quality",
    "svi_value": "SVI Value",
    "svi_class": "SVI Class",
    "maintenance_class": "Maintenance Class",
    "people_efficiency_class": "People Efficiency Class",
    "structures_efficiency_class": "Structures Efficiency Class",
    "environment_channel_class": "Environment (Channel) Class",
    "row_subdivision_class": "ROW (Subdivision) Class",
    "multiple_benefits_channel_class": "Multiple Benefits (Channel) Class",
    "district_improvement_synergy": "District Improvement Synergy",
}


def normalize_project_type_value(v: str) -> str:
    s = str(v).strip().lower()
    if not s:
        return ""
    subdiv_tokens = ["subdivision", "subdiv", "sub ", "sub-", "sub_", "local drainage", "street"]
    channel_tokens = ["channel", "detention", "ch/det", "det ", "det-", "regional"]
    if any(t in s for t in subdiv_tokens):
        return "subdivision_drainage"
    if any(t in s for t in channel_tokens):
        return "channel_detention"
    if s in {"sub", "sd"}:
        return "subdivision_drainage"
    if s in {"ch", "det"}:
        return "channel_detention"
    return ""


def get_criteria_meta() -> list[tuple[str, str]]:
    custom = st.session_state.get("custom_criteria", [])
    custom_meta = []
    for item in custom:
        key = item.get("key", "")
        label = item.get("label", "")
        include = item.get("include", True)
        if key:
            if include:
                if label:
                    custom_meta.append((key, label))
                else:
                    custom_meta.append((key, key))
    return BASE_CRITERIA_META + custom_meta


def render_weights_inputs(context_key: str, show_reference: bool = True) -> float:
    def _w(key: str, label: str) -> float:
        val = st.number_input(
            label,
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["weights_pct"].get(key, 0.0)),
            step=0.1,
            format="%.1f",
            key=f"{context_key}_w_{key}",
        )
        val = round(float(val), 1)
        st.session_state["weights_pct"][key] = val
        return val

    vals = []
    for k, label in get_criteria_meta():
        vals.append(_w(k, f"{label} (%)"))

    total_w = round(sum(vals), 1)
    st.session_state["is_valid_weights"] = (total_w == 100.0)
    if st.session_state["is_valid_weights"]:
        st.success(f"Total: {total_w:.1f}% OK")
    else:
        diff = round(100.0 - total_w, 1)
        if diff > 0:
            st.error(f"Total: {total_w:.1f}% - add {diff:.1f}%")
        else:
            st.error(f"Total: {total_w:.1f}% - remove {abs(diff):.1f}%")

    if show_reference:
        st.markdown("**Reference HCFCD (2022) Weights for Prioritization:**")
        hcfcd_weights = config["weights"]
        hcfcd_weights_table = pd.DataFrame([
            {"Criterion": "People Benefits Efficiency", "Weight": f"{int(round(hcfcd_weights['people_efficiency'] * 100))}%"},
            {"Criterion": "Structure Benefits Efficiency", "Weight": f"{int(round(hcfcd_weights['structures_efficiency'] * 100))}%"},
            {"Criterion": "Existing Conditions", "Weight": f"{int(round(hcfcd_weights['existing_conditions'] * 100))}%"},
            {"Criterion": "Social Vulnerability Index", "Weight": f"{int(round(hcfcd_weights['svi'] * 100))}%"},
            {"Criterion": "Long-Term Maintenance Costs", "Weight": f"{int(round(hcfcd_weights['maintenance'] * 100))}%"},
            {"Criterion": "Minimizes Environmental Impacts", "Weight": f"{int(round(hcfcd_weights['environment'] * 100))}%"},
            {"Criterion": "Potential for Multiple Benefits", "Weight": f"{int(round(hcfcd_weights['multiple_benefits'] * 100))}%"},
        ])
        st.markdown(hcfcd_weights_table.to_html(index=False), unsafe_allow_html=True)

    return total_w


def render_weights_table(context_key: str) -> None:
    meta = get_criteria_meta()
    total_w = 0.0
    st.caption("Use the +/- controls to adjust each weight. The total must equal 100.")
    for key, label in meta:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(label)
        with col2:
            val = st.number_input(
                "Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state["weights_pct"].get(key, 0.0)),
                step=0.1,
                format="%.1f",
                key=f"{context_key}_weight_{key}",
                label_visibility="collapsed",
            )
            st.session_state["weights_pct"][key] = round(float(val), 1)
            total_w += float(val)
    st.session_state["is_valid_weights"] = (total_w == 100.0)
    if st.session_state["is_valid_weights"]:
        st.success(f"Total: {total_w:.1f}% OK")
    else:
        diff = round(100.0 - total_w, 1)
        if diff > 0:
            st.error(f"Total: {total_w:.1f}% - add {diff:.1f}%")
        else:
            st.error(f"Total: {total_w:.1f}% - remove {abs(diff):.1f}%")


def render_reference_weights_table() -> None:
    st.markdown("**Reference HCFCD (2022) Weights for Prioritization:**")
    hcfcd_weights = config["weights"]
    hcfcd_weights_table = pd.DataFrame([
        {"Criterion": "People Benefits Efficiency", "Weight": f"{int(round(hcfcd_weights['people_efficiency'] * 100))}%"},
        {"Criterion": "Structure Benefits Efficiency", "Weight": f"{int(round(hcfcd_weights['structures_efficiency'] * 100))}%"},
        {"Criterion": "Existing Conditions", "Weight": f"{int(round(hcfcd_weights['existing_conditions'] * 100))}%"},
        {"Criterion": "Social Vulnerability Index", "Weight": f"{int(round(hcfcd_weights['svi'] * 100))}%"},
        {"Criterion": "Long-Term Maintenance Costs", "Weight": f"{int(round(hcfcd_weights['maintenance'] * 100))}%"},
        {"Criterion": "Minimizes Environmental Impacts", "Weight": f"{int(round(hcfcd_weights['environment'] * 100))}%"},
        {"Criterion": "Potential for Multiple Benefits", "Weight": f"{int(round(hcfcd_weights['multiple_benefits'] * 100))}%"},
    ])
    st.markdown(hcfcd_weights_table.to_html(index=False), unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
col_left, col_right = st.columns([5, 1])
with col_left:
    st.title("Project Prioritization Tool")
    st.caption("HCFCD Framework - Internal Use at infraTECH")
with col_right:
    hc_logo_path = resource_path("HC_P1.jpg")
    ite_logo_path = resource_path("ITE_Logo.png")

    if os.path.exists(hc_logo_path) and os.path.exists(ite_logo_path):
        with open(hc_logo_path, "rb") as f:
            hc_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(ite_logo_path, "rb") as f:
            ite_b64 = base64.b64encode(f.read()).decode("utf-8")

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:flex-end; gap:12px;">
                <img src="data:image/jpeg;base64,{hc_b64}" style="height:90px; width:auto;" />
                <img src="data:image/png;base64,{ite_b64}" style="height:120px; width:auto;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if os.path.exists(hc_logo_path):
            st.image(hc_logo_path, width=100)
        if os.path.exists(ite_logo_path):
            st.image(ite_logo_path, width=200)

st.caption("Upload a CSV, edit data, add projects, run scoring, and download results.")
st.markdown(
    """
    <style>
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #dfe7dd;
        border-color: #a8b8a1;
        color: #2c3a2f;
        font-weight: 600;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #d2ddd0;
        border-color: #93a58d;
        color: #253228;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #1f6feb;
        border-color: #1f6feb;
        color: white;
        font-weight: 700;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #1558c0;
        border-color: #1558c0;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar: Data + Config + Weights
# ----------------------------
with st.sidebar:
    uploaded_sidebar = None
    st.markdown("### About This Tool")
    st.markdown(
        "The Project Prioritization Tool was developed by infraTECH Engineers and Innovators "
        "to support transparent, evidence-based ranking of drainage improvement projects and "
        "planning-level studies in Harris County Precinct 1.\n\n"
        "It operationalizes the HCFCD 2022 prioritization framework and integrates advanced "
        "multi-criteria decision methods, including AHP and TOPSIS, enabling project evaluation "
        "across alternative weighting scenarios.\n\n"
        "The platform provides structured data ingestion, automated scoring, configurable criteria "
        "selection, and sensitivity-testing workflows to support consistent, defensible, and "
        "well-documented investment decisions."
    )
    st.divider()
    st.markdown("### Workspace")

    if "sidebar_workspace_filename" not in st.session_state:
        st.session_state["sidebar_workspace_filename"] = f"Workspace_{datetime.now().strftime('%Y%m%d%H%M')}.json"

    export_name = st.text_input(
        "Workspace file name",
        key="sidebar_workspace_filename",
    ).strip()
    if not export_name:
        export_name = f"Workspace_{datetime.now().strftime('%Y%m%d%H%M')}.json"
    if not export_name.lower().endswith(".json"):
        export_name = f"{export_name}.json"

    if st.button("Export Entire Workspace", key="btn_sidebar_export_workspace"):
        try:
            bundle = build_workspace_bundle()
            export_dir = get_workspace_export_dir()
            if export_dir is None:
                st.error("Could not find/write the Importable Database folder.")
            else:
                out_path = os.path.join(export_dir, export_name)
                with open(out_path, "w", encoding="utf-8") as wf:
                    json.dump(bundle, wf, indent=2)
                st.success(f"Workspace exported: {out_path}")
        except Exception as ex:
            st.error(f"Workspace export failed: {ex}")

    importable_json = get_importable_workspace_json_files()
    selected_saved_workspace = None
    if importable_json:
        selected_saved_workspace = st.selectbox(
            "Saved workspace files",
            options=[n for n, _ in importable_json],
            key="sidebar_saved_workspace_name",
        )
    uploaded_saved_workspace = st.file_uploader("Or upload workspace JSON", type=["json"], key="sidebar_workspace_upload")

    if st.button("Load Saved Workspace", key="btn_sidebar_load_workspace"):
        try:
            if uploaded_saved_workspace is not None:
                loaded = json.load(uploaded_saved_workspace)
                apply_workspace_bundle(loaded)
                st.success("Workspace loaded from uploaded file.")
                st.rerun()
            elif selected_saved_workspace:
                chosen_path = dict(importable_json).get(selected_saved_workspace)
                if not chosen_path:
                    st.error("Selected workspace file could not be resolved.")
                else:
                    with open(chosen_path, "r", encoding="utf-8") as jf:
                        loaded = json.load(jf)
                    apply_workspace_bundle(loaded)
                    st.success(f"Workspace loaded: {selected_saved_workspace}")
                    st.rerun()
            else:
                st.warning("No saved workspace file found. Upload a JSON workspace file.")
        except Exception as ex:
            st.error(f"Workspace load failed: {ex}")

# ----------------------------
# Load or initialize dataframe
# ----------------------------
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = ""
if "loaded_sidebar_upload_id" not in st.session_state:
    st.session_state["loaded_sidebar_upload_id"] = None
if "loaded_main_upload_id" not in st.session_state:
    st.session_state["loaded_main_upload_id"] = None

if uploaded_sidebar is not None:
    uploaded_df, upload_id = read_uploaded_csv_with_id(uploaded_sidebar)
    if st.session_state.get("loaded_sidebar_upload_id") != upload_id:
        st.session_state["df_work"] = preprocess_uploaded_df(uploaded_df)
        st.session_state["uploaded_file_name"] = uploaded_sidebar.name
        st.session_state["loaded_sidebar_upload_id"] = upload_id
        st.session_state["svi_source_force_reset"] = True

if "df_work" not in st.session_state:
    st.session_state["df_work"] = preprocess_uploaded_df(pd.read_csv(resource_path("input_template.csv")))
    st.session_state["uploaded_file_name"] = "input_template.csv"
    st.info("Using included template. You can upload a CSV from the sidebar.")

df = st.session_state["df_work"]

tab_data, tab_tools, tab_analysis, tab_weights, tab_ahp, tab_topsis = st.tabs(
    ["Prioritization Database", "Data Tools", "Parameter Analysis", "Direct Weights", "AHP Weights", "Ranking"]
)


def render_data_tab():
    st.subheader("Data Source")
    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        uploaded_main = st.file_uploader("Upload input CSV", type=["csv"], key="main_upload")
        importable_main = get_importable_database_files()
        if importable_main:
            main_names = [n for n, _ in importable_main]
            selected_main_db = st.selectbox(
                "Or load from Importable Database",
                options=main_names,
                key="main_importable_db",
            )
            if st.button("Load selected database", key="btn_load_selected_main_db"):
                chosen_path = dict(importable_main).get(selected_main_db)
                if chosen_path:
                    st.session_state["df_work"] = preprocess_uploaded_df(pd.read_csv(chosen_path))
                    st.session_state["uploaded_file_name"] = selected_main_db
                    st.session_state["loaded_main_upload_id"] = None
                    st.session_state["loaded_sidebar_upload_id"] = None
                    st.session_state["svi_source_force_reset"] = True
                    st.rerun()
    with col_u2:
        if st.button("Load template dataset", key="btn_load_template_main"):
            st.session_state["df_work"] = preprocess_uploaded_df(pd.read_csv(resource_path("input_template.csv")))
            st.session_state["uploaded_file_name"] = "input_template.csv"
            st.session_state["loaded_main_upload_id"] = None
            st.session_state["svi_source_force_reset"] = True
        if st.button("Clear current dataset", key="btn_clear_dataset"):
            st.session_state["df_work"] = st.session_state["df_work"].head(0)
            st.session_state["uploaded_file_name"] = "cleared"
            st.session_state["loaded_main_upload_id"] = None
            st.session_state["svi_source_force_reset"] = True
    if uploaded_main is not None:
        uploaded_df, upload_id = read_uploaded_csv_with_id(uploaded_main)
        if st.session_state.get("loaded_main_upload_id") != upload_id:
            st.session_state["df_work"] = preprocess_uploaded_df(uploaded_df)
            st.session_state["uploaded_file_name"] = uploaded_main.name
            st.session_state["loaded_main_upload_id"] = upload_id
            st.session_state["svi_source_force_reset"] = True

    current_name = st.session_state.get("uploaded_file_name", "") or "session data"
    st.caption(f"Current dataset: {current_name}")

    st.divider()
    df_local = st.session_state["df_work"]

    # Editable grid
    st.subheader("Edit Project Data")
    st.write("Click a cell to edit. You can also add rows at the bottom of the table.")
    st.data_editor(
        df_local,
        use_container_width=True,
        num_rows="dynamic",
        key="project_data_editor",
        on_change=sync_project_data_editor,
    )

    st.divider()
    st.subheader("Add New Parameter")
    col_c1, col_c2, col_c3 = st.columns([2, 2, 1])
    with col_c1:
        new_col_name = st.text_input("New column header (short)", key="new_col_name")
    with col_c2:
        new_col_desc = st.text_input("Short description (full form)", key="new_col_desc")
    with col_c3:
        new_col_type = st.selectbox("Column type", ["Text", "Number"], key="new_col_type")

    if new_col_type == "Number":
        new_col_default = st.number_input("Default value", value=0.0, key="new_col_default_num")
    else:
        new_col_default = st.text_input("Default value", value="", key="new_col_default_text")

    if st.button("Add Column", key="btn_add_column"):
        if not new_col_name.strip():
            st.error("Column name is required.")
        else:
            if new_col_name in st.session_state["df_work"].columns:
                st.error("Column already exists.")
            else:
                st.session_state["df_work"][new_col_name] = new_col_default
                key_name = new_col_name.strip()
                if not any(c.get("key") == key_name for c in st.session_state["custom_criteria"]):
                    st.session_state["custom_criteria"].append({
                        "key": key_name,
                        "label": new_col_desc.strip(),
                        "include": True,
                        "type": new_col_type,
                    })
                st.session_state["weights_pct"][key_name] = 0.0
                st.success(f"Added column: {new_col_name}")

    # ----------------------------
    # Add Project (in-context, no form wrapper)
    # ----------------------------
    st.divider()
    st.subheader("Add Project")
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        if st.button("Add Project", key="btn_add_project"):
            st.session_state["show_add_project"] = True
    with col_btn2:
        if st.session_state["show_add_project"]:
            st.caption("Fill the fields below then click Save Project to append it to the table above.")

    project_type_options = maps.get("project_type", ["channel_detention", "subdivision_drainage"])
    svi_options = list(maps.get("svi_class", {}).keys()) or ["low", "low_moderate", "moderate_high", "high"]
    maint_options = list(maps.get("maintenance_class", {}).keys()) or ["extensive_specialized", "outside_regular", "regular"]
    channel_capacity_options = list(maps.get("existing_conditions_channel_capacity", {}).keys()) or ["gt_1_percent", "lt_1_percent", "lt_2_percent", "lt_4_percent", "lt_10_percent", "lt_20_percent", "lt_50_percent"]
    env_channel_options = list(maps.get("environment_channel", {}).keys()) or ["individual_permit_and_credits", "credits", "avoid_impacts", "minimal_none"]
    mb_channel_options = list(maps.get("multiple_benefits_channel", {}).keys()) or ["none", "recreation", "environment", "both"]
    rain_options = list(maps.get("existing_conditions_subdivision_matrix", {}).keys()) or ["high", "intermediate", "low"]
    infra_options = ["high", "intermediate", "low"]
    row_options = list(maps.get("row_subdivision", {}).keys()) or ["needs_additional_row", "within_existing_row"]
    syn_options = list(maps.get("multiple_benefits_subdivision", {}).keys()) or ["no", "yes"]

    if st.session_state["show_add_project"]:
        # 1) Project Type
        project_type = st.selectbox("1) Project Type*", project_type_options, index=0, key="add_project_type", format_func=lambda k: label_for("project_type", k))

        # 2) Project Name
        st.markdown("### 2) Project Name")
        project_name = st.text_input("Project Name*", value="", key="add_project_name")
        st.divider()

        # 3) Project Efficiency Weighting Factor (in-context)
        st.markdown("### 3) Project Efficiency Weighting Factor")
        with st.expander("Project Efficiency Tables", expanded=False):
            # Build DataFrames and render side-by-side using pandas Styler to avoid raw-HTML rendering issues
            people_rows = [
                ("Less than $6,000/person", 10),
                ("$6,000 to $15,000/person", 8),
                ("$15,001 to $28,000/person", 6),
                ("$28,001 to $77,000/person", 4),
                ("Greater than $77,000/person", 1),
            ]
            struct_rows = [
                ("Less than $23,000/structure", 10),
                ("$23,000 to $60,000/structure", 8),
                ("$60,001 to $106,000/structure", 6),
                ("$106,001 to $261,000/structure", 4),
                ("Greater than $261,000/structure", 1),
            ]
            df_people = pd.DataFrame(people_rows, columns=["Criteria", "Score"])
            df_struct = pd.DataFrame(struct_rows, columns=["Criteria", "Score"])

            colors = {10: "#d4edda", 8: "#c3e6cb", 6: "#fff3cd", 4: "#f8d7da", 1: "#f5c6cb"}

            # Create stylers
            sty_people = df_people.style
            sty_struct = df_struct.style

            # Apply row-wise colors
            sty_people = sty_people.apply(lambda row: [f'background-color: {colors[row[1]]}']*len(row), axis=1)
            sty_struct = sty_struct.apply(lambda row: [f'background-color: {colors[row[1]]}']*len(row), axis=1)

            # Set header style
            header_style = [{"selector": "th", "props": [("background-color", "#f0f0f0"), ("padding", "8px")]}]
            sty_people = sty_people.set_table_styles(header_style)
            sty_struct = sty_struct.set_table_styles(header_style)

            combined_html = f'<div style="display:flex; gap:12px; align-items:flex-start;">{sty_people.to_html()}</div>'
            # insert structure table after people table by simple concatenation
            combined_html = combined_html.replace("</div>", sty_struct.to_html() + "</div>")
            st.markdown(combined_html, unsafe_allow_html=True)
            efficiency_input_method = st.radio("How would you like to input project efficiency?", options=["Calculate from project costs", "Enter efficiency classes directly"], index=0, horizontal=True, key="add_efficiency_method")
        efficiency_input_method = st.session_state.get("add_efficiency_method", "Calculate from project costs")

        if efficiency_input_method == "Enter efficiency classes directly":
            ec1, ec2 = st.columns(2)
            with ec1:
                people_efficiency_class = st.selectbox("Resident Benefits Efficiency*", options=list(EFFICIENCY_CLASSES.keys()), format_func=lambda k: f"Score {k}: {EFFICIENCY_CLASSES[k]}", key="add_people_efficiency_class")
            with ec2:
                structures_efficiency_class = st.selectbox("Structure Benefit Efficiency*", options=list(EFFICIENCY_CLASSES.keys()), format_func=lambda k: f"Score {k}: {EFFICIENCY_CLASSES[k]}", key="add_structures_efficiency_class")
            total_cost = 0.0
            people_benefitted = 0.0
            structures_benefitted = 0.0
        else:
            st.info("Efficiency will be calculated from: Total Cost divided by Residents (or Structures) Benefitted")
            col_cost, col_people, col_structs = st.columns(3)
            with col_cost:
                total_cost = st.number_input("Total Cost*", min_value=0.0, value=0.0, step=1000.0, format="%.0f", key="add_total_cost")
            with col_people:
                people_benefitted = st.number_input("Residents Benefitted*", min_value=0.0, value=0.0, step=1.0, format="%.0f", key="add_people_benefitted")
            with col_structs:
                structures_benefitted = st.number_input("Structures Benefitted*", min_value=0.0, value=0.0, step=1.0, format="%.0f", key="add_structures_benefitted")
            people_efficiency_class = ""
            structures_efficiency_class = ""

        st.divider()

        # 4) Existing Conditions
        st.markdown("### 4) Existing Conditions Weighting Factor")
        if project_type == "channel_detention":
            with st.expander("View Channel Scoring Criteria", expanded=False):
                st.markdown(get_existing_conditions_channel_html(), unsafe_allow_html=True)
            channel_capacity_class = st.selectbox("Channel Capacity Class*", channel_capacity_options, index=0, key="add_channel_capacity_class", format_func=lambda k: label_for("existing_conditions_channel_capacity", k))
            excess_rainfall_class = ""
            drainage_infra_quality = ""
        else:
            with st.expander("View Subdivision Scoring Criteria", expanded=False):
                st.markdown(get_existing_conditions_subdivision_html(), unsafe_allow_html=True)
            ec1, ec2 = st.columns(2)
            with ec1:
                excess_rainfall_class = st.selectbox("Excess Rainfall Class*", rain_options, index=0, key="add_excess_rainfall_class")
            with ec2:
                drainage_infra_quality = st.selectbox("Drainage Infrastructure Quality*", infra_options, index=0, key="add_drainage_infra_quality")
            channel_capacity_class = ""

        st.divider()

        # 5) Social Vulnerability Index (SVI)
        st.markdown("### 5) Social Vulnerability Index (SVI)")
        with st.expander("Social Vulnerability Index (SVI)", expanded=False):
            st.markdown(get_svi_html(), unsafe_allow_html=True)
            svi_input_method = st.radio("How would you like to input SVI?", options=["Select from predefined class", "Enter SVI value (0-1)"], index=0, horizontal=True, key="add_svi_method")
        svi_input_method = st.session_state.get("add_svi_method", "Select from predefined class")
        if svi_input_method == "Enter SVI value (0-1)":
            col_slider, col_info = st.columns([3, 1])
            with col_slider:
                svi_value = st.slider("SVI Value", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f", key="add_svi_value")
            svi_class = classify_svi(svi_value)
            with col_info:
                st.markdown(f"**Auto-classified:**  \n**{label_for('svi_class', svi_class)}**")
        else:
            svi_value = None
            svi_class = st.selectbox("SVI Class*", svi_options, index=0, key="add_svi_class", format_func=lambda k: label_for("svi_class", k))

        st.divider()

        # 6) Minimizes Environmental Impact
        st.markdown("### 6) Minimizes Environmental Impact Weighting Factor")
        if project_type == "channel_detention":
            with st.expander("View Channel Environmental Criteria", expanded=False):
                st.markdown(get_environment_channel_html(), unsafe_allow_html=True)
            environment_channel_class = st.selectbox("Environmental Class (Channel)*", env_channel_options, index=0, key="add_env_channel", format_func=lambda k: label_for("environment_channel", k))
            row_subdivision_class = ""
        else:
            with st.expander("View Subdivision Environmental Criteria", expanded=False):
                st.markdown(get_environment_subdivision_html(), unsafe_allow_html=True)
            row_subdivision_class = st.selectbox("ROW Availability (Subdivision)*", row_options, index=0, key="add_row_subdivision", format_func=lambda k: label_for("row_subdivision", k))
            environment_channel_class = ""

        st.divider()

        # 7) Potential for Multiple Benefits
        st.markdown("### 7) Potential for Multiple Benefits Weighting Factor")
        if project_type == "channel_detention":
            with st.expander("View Channel Multiple Benefits Criteria", expanded=False):
                st.markdown(get_multiple_benefits_channel_html(), unsafe_allow_html=True)
            multiple_benefits_channel_class = st.selectbox("Multiple Benefits (Channel)*", mb_channel_options, index=0, key="add_mb_channel", format_func=lambda k: label_for("multiple_benefits_channel", k))
            district_improvement_synergy = ""
        else:
            with st.expander("View Subdivision Multiple Benefits Criteria", expanded=False):
                st.markdown(get_multiple_benefits_subdivision_html(), unsafe_allow_html=True)
            district_improvement_synergy = st.selectbox("District Improvement Synergy*", syn_options, index=0, key="add_synergy", format_func=lambda k: label_for("multiple_benefits_subdivision", k))
            multiple_benefits_channel_class = ""

        st.divider()

        # Other Details
        st.markdown("### Other Details")
        c_maint, c_notes = st.columns([1, 2])
        with c_maint:
            maintenance_class = st.selectbox("Maintenance Class*", maint_options, index=0, key="add_maint_class", format_func=lambda k: label_for("maintenance_class", k))
        with c_notes:
            notes = st.text_area("Notes (optional)", value="", key="add_notes", height=100)

        custom_inputs = {}
        custom_criteria = [c for c in st.session_state.get("custom_criteria", []) if c.get("include")]
        if custom_criteria:
            st.divider()
            st.markdown("### Custom Criteria Inputs")
            for c in custom_criteria:
                key = c.get("key", "")
                label = c.get("label") or key
                ctype = c.get("type", "Text")
                if not key:
                    continue
                if ctype == "Number":
                    custom_inputs[key] = st.number_input(label, value=0.0, key=f"custom_{key}")
                else:
                    custom_inputs[key] = st.text_input(label, value="", key=f"custom_{key}")

        st.divider()

        # Action buttons
        colb1, colb2 = st.columns(2)
        with colb1:
            if st.button("Save Project", key="btn_save_project"):
                # validation
                if not project_name.strip():
                    st.error("Project Name is required.")
                elif efficiency_input_method == "Calculate from project costs" and total_cost <= 0:
                    st.error("Total Cost must be greater than 0.")
                else:
                    df_current = st.session_state["df_work"].copy()
                    next_id = 1
                    if "project_id" in df_current.columns:
                        try:
                            mx = pd.to_numeric(df_current["project_id"], errors="coerce").max()
                            next_id = int(mx) + 1 if pd.notna(mx) else 1
                        except Exception:
                            next_id = 1

                    new_row = {
                        "project_id": next_id,
                        "project_name": project_name.strip(),
                        "project_type": project_type,
                        "total_cost": float(total_cost) if total_cost else "",
                        "people_benefitted": float(people_benefitted) if people_benefitted else "",
                        "structures_benefitted": float(structures_benefitted) if structures_benefitted else "",
                        "channel_capacity_class": channel_capacity_class,
                        "excess_rainfall_class": excess_rainfall_class,
                        "drainage_infra_quality": drainage_infra_quality,
                        "svi_value": svi_value if svi_value is not None else "",
                        "svi_value_reclassified": svi_value if svi_value is not None else "",
                        "svi_class": svi_class,
                        "maintenance_class": maintenance_class,
                        "people_efficiency_class": people_efficiency_class,
                        "structures_efficiency_class": structures_efficiency_class,
                        "environment_channel_class": environment_channel_class,
                        "row_subdivision_class": row_subdivision_class,
                        "multiple_benefits_channel_class": multiple_benefits_channel_class,
                        "district_improvement_synergy": district_improvement_synergy,
                        "notes": notes.strip() if notes else "",
                    }

                    for k in new_row.keys():
                        if k not in df_current.columns:
                            df_current[k] = ""

                    for k, v in custom_inputs.items():
                        if k not in df_current.columns:
                            df_current[k] = ""
                        new_row[k] = v

                    df_current = pd.concat([df_current, pd.DataFrame([new_row])], ignore_index=True)
                    st.session_state["df_work"] = df_current
                    st.session_state["show_add_project"] = False
                    st.success(f"Added project: {project_name.strip()} (ID {next_id})")
                    st.rerun()
        with colb2:
            if st.button("Cancel", key="btn_cancel_add"):
                st.session_state["show_add_project"] = False
                st.rerun()

    st.divider()
    st.info("Criteria Mapping is available in the Data Tools tab.")



def render_direct_weights_tab():
    st.subheader("Direct Weight Input")
    st.write("Enter weights as percentages. The total must equal 100.")
    render_weights_table("direct")
    render_reference_weights_table()


def render_data_tools_tab():
    st.subheader("Data Tools")
    st.caption("Utilities for data cleanup and reclassification. More tools can be added in this tab later.")

    df_local = st.session_state.get("df_work", pd.DataFrame()).copy()
    if df_local.empty:
        st.info("Dataset is empty. Upload or add data first.")
    with st.expander("Workspace Export / Import", expanded=False):
        st.caption("Export current work (data + mappings + weights + AHP setup) and import later to continue.")
        if "workspace_export_filename" not in st.session_state:
            st.session_state["workspace_export_filename"] = f"Workspace_{datetime.now().strftime('%Y%m%d%H%M')}"

        with st.expander("Workspace Export", expanded=False):
            bundle = build_workspace_bundle()
            export_name = st.text_input(
                "Workspace export filename",
                key="workspace_export_filename",
                help="Saved to the local 'Importable Database' folder.",
            ).strip()
            if not export_name:
                export_name = f"Workspace_{datetime.now().strftime('%Y%m%d%H%M')}"
            if not export_name.lower().endswith(".json"):
                export_name = f"{export_name}.json"

            if st.button("Save Workspace to Importable Database", key="btn_save_workspace_to_folder"):
                export_dir = get_workspace_export_dir()
                if export_dir is None:
                    st.error("Could not find/write the Importable Database folder.")
                else:
                    try:
                        out_path = os.path.join(export_dir, export_name)
                        with open(out_path, "w", encoding="utf-8") as wf:
                            json.dump(bundle, wf, indent=2)
                        st.success(f"Workspace saved: {out_path}")
                    except Exception as ex:
                        st.error(f"Could not save workspace file: {ex}")

            st.download_button(
                "Download Workspace File",
                data=json.dumps(bundle, indent=2).encode("utf-8"),
                file_name=export_name,
                mime="application/json",
                key="download_workspace_bundle",
            )

        with st.expander("Workspace Import", expanded=False):
            uploaded_bundle = st.file_uploader("Upload Workspace File", type=["json"], key="upload_workspace_bundle")
            if uploaded_bundle is not None and st.button("Load Workspace File", key="btn_load_workspace_bundle"):
                try:
                    loaded = json.load(uploaded_bundle)
                    apply_workspace_bundle(loaded)
                    st.success("Workspace loaded.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Could not load workspace file: {ex}")

            importable_json = get_importable_workspace_json_files()
            if importable_json:
                json_names = [n for n, _ in importable_json]
                selected_json = st.selectbox(
                    "Or load JSON from Importable Database",
                    options=json_names,
                    key="workspace_importable_json",
                )
                if st.button("Load Selected JSON", key="btn_load_selected_workspace_json"):
                    chosen_path = dict(importable_json).get(selected_json)
                    if chosen_path:
                        try:
                            with open(chosen_path, "r", encoding="utf-8") as jf:
                                loaded = json.load(jf)
                            apply_workspace_bundle(loaded)
                            st.success(f"Loaded workspace JSON: {selected_json}")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Could not load selected JSON as workspace: {ex}")
            else:
                st.caption("No JSON files found in Importable Database folder.")
    if df_local.empty:
        return

    numeric_like_cols = []
    for c in df_local.columns:
        if pd.api.types.is_numeric_dtype(df_local[c]) or str(c).strip().lower() in {"svi", "svi_value"}:
            numeric_like_cols.append(c)

    if not numeric_like_cols:
        st.info("No numeric columns available for SVI/Excess Rainfall reclassification.")
        numeric_like_cols = []

    with st.expander("Standard Parameter Mapping", expanded=False):
        st.caption("Map your imported CSV columns to HCFCD parameters from the 2022 Prioritization Framework for Allocation of Funds from the Harris County Flood Resilience Trust (April 26, 2022).")
        source_options = ["(No mapping)"] + list(df_local.columns)
        current_map = st.session_state.get("standard_column_mapping", {})
        map_rows = []
        mapping_targets = [t for t in HCFCD_PARAMETER_FIELDS if t != "project_type"]
        for target in mapping_targets:
            default_src = current_map.get(target)
            if not default_src:
                default_src = target if target in df_local.columns else "(No mapping)"
            if default_src not in source_options:
                default_src = "(No mapping)"
            map_rows.append({
                "Standard Parameter": HCFCD_PARAMETER_LABELS.get(target, target),
                "Target Column": target,
                "Source Column": default_src,
            })

        mapping_df = pd.DataFrame(map_rows)
        edited_mapping = st.data_editor(
            mapping_df,
            use_container_width=True,
            hide_index=True,
            key="standard_mapping_table",
            column_config={
                "Standard Parameter": st.column_config.TextColumn(disabled=True),
                "Target Column": st.column_config.TextColumn(disabled=True),
                "Source Column": st.column_config.SelectboxColumn(options=source_options),
            },
        )

        with st.expander("Project Type Mapping", expanded=False):
            pt_source_default = current_map.get("project_type", "project_type" if "project_type" in df_local.columns else "(No mapping)")
            if pt_source_default not in source_options:
                pt_source_default = "(No mapping)"
            pt_mode_default = st.session_state.get("project_type_mode", "Map from source values")
            if pt_mode_default not in ["Map from source values", "All Channel/Detention", "All Subdivision/Drainage"]:
                pt_mode_default = "Map from source values"
            c_pt1, c_pt2 = st.columns([1, 1])
            with c_pt1:
                st.selectbox(
                    "Project type source column",
                    options=source_options,
                    index=source_options.index(pt_source_default),
                    key="project_type_source_col",
                )
            with c_pt2:
                st.selectbox(
                    "Project type mode",
                    options=["Map from source values", "All Channel/Detention", "All Subdivision/Drainage"],
                    index=["Map from source values", "All Channel/Detention", "All Subdivision/Drainage"].index(pt_mode_default),
                    key="project_type_mode",
                )
            if st.session_state.get("project_type_mode") == "Map from source values":
                st.caption("Keywords used for matching source values to standard project types.")
                c_kw1, c_kw2 = st.columns(2)
                with c_kw1:
                    st.text_input(
                        "Subdivision keywords (comma-separated)",
                        value=st.session_state.get("subdivision_keywords", "subdivision, subdiv, sub, local drainage, street"),
                        key="subdivision_keywords",
                    )
                with c_kw2:
                    st.text_input(
                        "Channel/Detention keywords (comma-separated)",
                        value=st.session_state.get("channel_keywords", "channel, detention, ch, det, regional"),
                        key="channel_keywords",
                    )

        if st.button("Apply Standard Parameter Mapping", key="btn_apply_standard_mapping"):
            df_updated = st.session_state["df_work"].copy()
            saved_map = {}
            applied_count = 0
            for _, r in edited_mapping.iterrows():
                target = str(r.get("Target Column", "")).strip()
                source = str(r.get("Source Column", "")).strip()
                if target == "project_type":
                    continue
                if not target or source in {"", "(No mapping)"}:
                    continue
                if source in df_updated.columns:
                    df_updated[target] = df_updated[source]
                    saved_map[target] = source
                    applied_count += 1

            # Dedicated project_type mapping controls
            pt_source = st.session_state.get("project_type_source_col", "(No mapping)")
            pt_mode = st.session_state.get("project_type_mode", "Map from source values")
            if pt_mode == "All Channel/Detention":
                df_updated["project_type"] = "channel_detention"
                applied_count += 1
            elif pt_mode == "All Subdivision/Drainage":
                df_updated["project_type"] = "subdivision_drainage"
                applied_count += 1
            elif pt_source in df_updated.columns:
                subdiv_kw = [k.strip().lower() for k in str(st.session_state.get("subdivision_keywords", "")).split(",") if k.strip()]
                channel_kw = [k.strip().lower() for k in str(st.session_state.get("channel_keywords", "")).split(",") if k.strip()]
                src = df_updated[pt_source].astype(str).str.lower()
                mapped = pd.Series("", index=df_updated.index, dtype=object)
                if subdiv_kw:
                    mapped[src.apply(lambda x: any(k in x for k in subdiv_kw))] = "subdivision_drainage"
                if channel_kw:
                    mapped[src.apply(lambda x: any(k in x for k in channel_kw))] = "channel_detention"
                if "project_type" not in df_updated.columns:
                    df_updated["project_type"] = ""
                existing = df_updated["project_type"].astype(str).str.strip()
                fill_mask = existing.eq("")
                df_updated.loc[fill_mask, "project_type"] = mapped[fill_mask]
                saved_map["project_type"] = pt_source
                applied_count += 1

            # Project ID is required by scoring, auto-fill if not mapped/provided.
            if "project_id" not in df_updated.columns:
                df_updated["project_id"] = np.arange(1, len(df_updated) + 1)
            else:
                ids = pd.to_numeric(df_updated["project_id"], errors="coerce")
                next_id = int(np.nanmax(ids)) + 1 if np.isfinite(np.nanmax(ids)) else 1
                new_ids = []
                for v in ids:
                    if pd.isna(v):
                        new_ids.append(next_id)
                        next_id += 1
                    else:
                        new_ids.append(int(v))
                df_updated["project_id"] = new_ids

            # Keep project_name available even when missing in imported files.
            if "project_name" not in df_updated.columns:
                df_updated["project_name"] = df_updated["project_id"].apply(lambda x: f"Project {x}")
            else:
                name_series = df_updated["project_name"].astype(str)
                blank_mask = name_series.str.strip().eq("") | name_series.eq("nan")
                df_updated.loc[blank_mask, "project_name"] = df_updated.loc[blank_mask, "project_id"].apply(lambda x: f"Project {x}")

            df_updated, auto_changes = auto_fill_derived_classes(df_updated, overwrite=False)
            st.session_state["df_work"] = df_updated
            st.session_state["standard_column_mapping"] = saved_map
            st.session_state.pop("project_data_editor", None)
            st.success(
                f"Applied mapping for {applied_count} standard parameters. "
                f"Auto-filled classes -> SVI: {auto_changes.get('svi_class', 0)}, "
                f"People Eff.: {auto_changes.get('people_efficiency_class', 0)}, "
                f"Structures Eff.: {auto_changes.get('structures_efficiency_class', 0)}."
            )
            st.rerun()

    with st.expander("Custom Criteria Mapping", expanded=False):
        st.caption("Map extra columns from your dataset to criteria with clear descriptions.")
        df_current = st.session_state["df_work"]
        base_set = set(STANDARD_COLUMNS + list(SCORE_COL_MAP.values()))
        candidate_cols = [c for c in df_current.columns if c not in base_set]

        existing = {c.get("key"): c for c in st.session_state.get("custom_criteria", []) if c.get("key")}
        if st.session_state.get("custom_criteria_table_df") is None or st.session_state.get("custom_criteria_table_cols") != candidate_cols:
            rows = []
            for c in candidate_cols:
                row = existing.get(c, {"key": c, "label": "", "include": False})
                rows.append({
                    "Column": c,
                    "Description": row.get("label", ""),
                    "Include": bool(row.get("include", False)),
                })
            st.session_state["custom_criteria_table_df"] = pd.DataFrame(rows)
            st.session_state["custom_criteria_table_cols"] = list(candidate_cols)

        if candidate_cols:
            edited = st.data_editor(
                st.session_state["custom_criteria_table_df"],
                use_container_width=True,
                hide_index=True,
                key="custom_criteria_table",
                column_config={
                    "Description": st.column_config.TextColumn(),
                    "Include": st.column_config.CheckboxColumn(),
                },
            )
            st.session_state["custom_criteria_table_df"] = edited

            if st.button("Save Criteria Mapping", key="btn_save_custom_mapping"):
                custom_list = []
                for _, r in edited.iterrows():
                    if bool(r.get("Include")):
                        key = str(r.get("Column", "")).strip()
                        label = str(r.get("Description", "")).strip()
                        if key:
                            dtype = "Number" if pd.api.types.is_numeric_dtype(df_current[key]) else "Text"
                            custom_list.append({"key": key, "label": label, "include": True, "type": dtype})
                            if key not in st.session_state["weights_pct"]:
                                st.session_state["weights_pct"][key] = 0.0
                st.session_state["custom_criteria"] = custom_list
                st.success("Custom criteria mapping saved.")
                st.rerun()
        else:
            st.info("No extra columns found to map. Add columns in the Prioritization Database tab or upload a dataset with additional fields.")

    with st.expander("Efficiency Calculation", expanded=False):
        st.caption("Use these formulas to derive efficiency classes from mapped value columns. This is useful after changing Standard Parameter Mapping.")
        st.latex(r"\text{Project Efficiency using People Benefitted}=\frac{\text{Total Cost of Project (\$)}}{\#\ \text{of People Benefitted}}")
        st.latex(r"\text{Project Efficiency using Structures Benefitted}=\frac{\text{Total Cost of Project (\$)}}{\#\ \text{of Structures Benefitted}}")
        if st.button("Recalculate Efficiency Classes", key="btn_recalc_efficiency_classes"):
            df_updated, eff_changes = recalculate_efficiency_classes(st.session_state["df_work"])
            st.session_state["df_work"] = df_updated
            st.session_state.pop("project_data_editor", None)
            st.success(
                f"Efficiency classes recalculated -> "
                f"People: {eff_changes.get('people_efficiency_class', 0)} updated, "
                f"Structures: {eff_changes.get('structures_efficiency_class', 0)} updated."
            )
            st.rerun()

    if numeric_like_cols:
        preview_df = df_local.loc[:, ~df_local.columns.duplicated(keep="first")].copy()
        if df_local.columns.duplicated().any():
            st.warning("Duplicate column names detected in dataset. Preview is showing first occurrence of each duplicate column.")

        svi_col_exact = find_column_case_insensitive(df_local, "svi")
        svi_col_with_tag = find_column_by_contains(df_local, "(svi)")
        default_source = svi_col_exact or svi_col_with_tag or numeric_like_cols[0]
        default_index = numeric_like_cols.index(default_source) if default_source in numeric_like_cols else 0
        cols_sig = tuple(str(c) for c in df_local.columns)
        if st.session_state.get("svi_source_cols_sig") != cols_sig or st.session_state.get("svi_source_force_reset"):
            st.session_state["svi_reclass_source_col"] = default_source
            st.session_state["svi_source_cols_sig"] = cols_sig
            st.session_state["svi_source_force_reset"] = False
        if (
            "svi_reclass_source_col" not in st.session_state
            or st.session_state.get("svi_reclass_source_col") not in numeric_like_cols
        ):
            st.session_state["svi_reclass_source_col"] = default_source

        with st.expander("SVI Reclassification", expanded=False):
            source_col = st.selectbox(
                "SVI source column",
                options=numeric_like_cols,
                key="svi_reclass_source_col",
            )
            method_label = st.radio(
                "Method",
                options=[
                    "Normalize values to 0.01-0.99 (min-max), then classify at 0.25 increments",
                    "Use raw values as-is for classification",
                ],
                index=0,
                key="svi_reclass_method",
            )
            method = "normalized" if method_label.startswith("Normalize") else "raw"

            source_series = pd.to_numeric(df_local[source_col], errors="coerce")
            svi_value_new, svi_class_new = classify_svi_series(source_series, mode=method)
            counts_df = (
                svi_class_new.replace("", np.nan)
                .dropna()
                .value_counts()
                .rename_axis("SVI Class")
                .reset_index(name="Count")
            )
            if not counts_df.empty:
                counts_df["SVI Class"] = counts_df["SVI Class"].apply(lambda k: label_for("svi_class", k))
                st.markdown("Preview of resulting class counts:")
                st.dataframe(counts_df, use_container_width=True)

            if st.button("Apply SVI Reclassification", key="btn_apply_svi_reclass"):
                df_updated = st.session_state["df_work"].copy()
                df_updated[SVI_RECLASS_COL] = svi_value_new
                df_updated["svi_value"] = svi_value_new
                df_updated["svi_class"] = svi_class_new
                st.session_state["df_work"] = df_updated
                st.session_state.pop("project_data_editor", None)
                st.success("SVI values/classification updated in current dataset.")
                st.rerun()

            source_col_preview = st.session_state.get("svi_reclass_source_col", default_source)
            preview_cols = []
            for c in ["project_id", "project_name", source_col_preview, SVI_RECLASS_COL, "svi_class"]:
                if c in preview_df.columns and c not in preview_cols:
                    preview_cols.append(c)
            st.markdown("Current Working Dataset Preview (SVI Columns)")
            st.dataframe(preview_df[preview_cols] if preview_cols else preview_df, use_container_width=True, height=280)

        rain_default = find_column_by_aliases(df_local, ["excess_rainfall", "Exc_Rain"]) or numeric_like_cols[0]
        if st.session_state.get("rain_source_cols_sig") != cols_sig:
            st.session_state["rain_reclass_source_col"] = rain_default
            st.session_state["rain_source_cols_sig"] = cols_sig
        if (
            "rain_reclass_source_col" not in st.session_state
            or st.session_state.get("rain_reclass_source_col") not in numeric_like_cols
        ):
            st.session_state["rain_reclass_source_col"] = rain_default

        with st.expander("Existing Condition Excess Rainfall Classification", expanded=False):
            rain_source_col = st.selectbox(
                "Excess rainfall source column",
                options=numeric_like_cols,
                key="rain_reclass_source_col",
            )
            st.caption("Values are min-max normalized to 0-1, then classified: Low (<0.33), Intermediate (0.33-<0.66), High (>=0.66).")

            rain_norm, rain_class = classify_excess_rainfall_series(df_local[rain_source_col])
            rain_counts = (
                rain_class.replace("", np.nan)
                .dropna()
                .value_counts()
                .rename_axis("Excess Rainfall Class")
                .reset_index(name="Count")
            )
            if not rain_counts.empty:
                st.dataframe(rain_counts, use_container_width=True)

            rain_preview_cols = []
            for c in ["project_id", "project_name", rain_source_col, "excess_rainfall_class"]:
                if c in preview_df.columns and c not in rain_preview_cols:
                    rain_preview_cols.append(c)
            rain_preview = preview_df[rain_preview_cols].copy() if rain_preview_cols else preview_df.copy()
            rain_preview["excess_rainfall_class_new"] = rain_class.values
            rain_preview["excess_rainfall_norm"] = rain_norm.values
            st.markdown("Current Working Dataset Preview (Excess Rainfall Columns)")
            st.dataframe(rain_preview, use_container_width=True, height=280)

            if st.button("Apply Excess Rainfall Classification", key="btn_apply_rain_reclass"):
                df_updated = st.session_state["df_work"].copy()
                df_updated["excess_rainfall_class"] = rain_class
                st.session_state["df_work"] = df_updated
                st.session_state.pop("project_data_editor", None)
                st.success("Excess rainfall class updated in current dataset.")
                st.rerun()

    st.markdown("### Current Working Dataset Preview (Full)")
    preview_df = df_local.loc[:, ~df_local.columns.duplicated(keep="first")].copy()
    st.dataframe(preview_df, use_container_width=True, height=320)


def render_parameter_analysis_tab():
    st.subheader("Parameter Analysis")
    st.caption("Explore relationships, regressions, distributions, and data quality for selected parameters.")

    df_local = st.session_state.get("df_work", pd.DataFrame()).copy()
    if df_local.empty:
        st.info("Dataset is empty. Upload or add data first.")
        return

    # Include both true numeric dtypes and numeric-like text columns (e.g., coded 0-10 classes).
    numeric_cols = []
    for c in df_local.columns:
        s_num = pd.to_numeric(df_local[c], errors="coerce")
        if s_num.notna().sum() >= 2:
            numeric_cols.append(c)
    if not numeric_cols:
        st.info("No numeric parameters found for analysis.")
        return

    with st.expander("Correlation Matrix", expanded=True):
        corr_cols = st.multiselect(
            "Select parameters for correlation",
            options=numeric_cols,
            default=numeric_cols,
            key="analysis_corr_cols",
        )
        corr_method = st.selectbox(
            "Correlation method",
            options=["pearson", "spearman"],
            index=0,
            key="analysis_corr_method",
        )
        if len(corr_cols) < 2:
            st.warning("Select at least two parameters.")
        else:
            corr_df = df_local[corr_cols].apply(pd.to_numeric, errors="coerce").corr(method=corr_method)

            def _corr_style(v):
                if pd.isna(v):
                    return ""
                x = max(-1.0, min(1.0, float(v)))
                # Muted diverging scale: soft red (-1) -> light neutral (0) -> soft green (+1)
                neg = (196, 138, 132)   # muted brick
                mid = (245, 245, 242)   # soft near-white
                pos = (139, 166, 139)   # muted sage
                if x >= 0:
                    t = x
                    r = int(mid[0] + (pos[0] - mid[0]) * t)
                    g = int(mid[1] + (pos[1] - mid[1]) * t)
                    b = int(mid[2] + (pos[2] - mid[2]) * t)
                else:
                    t = abs(x)
                    r = int(mid[0] + (neg[0] - mid[0]) * t)
                    g = int(mid[1] + (neg[1] - mid[1]) * t)
                    b = int(mid[2] + (neg[2] - mid[2]) * t)
                return f"background-color: rgb({r},{g},{b});"

            st.dataframe(
                corr_df.style.applymap(_corr_style).format("{:.3f}"),
                use_container_width=True,
            )

    with st.expander("Regression Analysis", expanded=False):
        y_col = st.selectbox("Dependent variable (Y)", options=numeric_cols, key="analysis_reg_y")
        x_candidates = [c for c in numeric_cols if c != y_col]
        x_cols = st.multiselect(
            "Independent variable(s) (X)",
            options=x_candidates,
            default=x_candidates[:1] if x_candidates else [],
            key="analysis_reg_x",
        )
        force_origin = st.checkbox("Force regression through origin", value=False, key="analysis_reg_force_origin")
        run_reg = st.checkbox("Run regression", value=False, key="analysis_run_reg")

        if run_reg:
            if len(x_cols) == 0:
                st.warning("Select at least one independent variable.")
            else:
                work = df_local[[y_col] + x_cols].apply(pd.to_numeric, errors="coerce").dropna()
                if work.empty:
                    st.warning("No valid numeric rows for selected variables after removing missing values.")
                else:
                    y = work[y_col].to_numpy(dtype=float)
                    X = work[x_cols].to_numpy(dtype=float)
                    if not force_origin:
                        X_model = np.column_stack([np.ones(len(X)), X])
                        coef = np.linalg.lstsq(X_model, y, rcond=None)[0]
                        intercept = float(coef[0])
                        betas = coef[1:]
                        y_hat = X_model @ coef
                    else:
                        coef = np.linalg.lstsq(X, y, rcond=None)[0]
                        intercept = 0.0
                        betas = coef
                        y_hat = X @ coef

                    ss_res = float(np.sum((y - y_hat) ** 2))
                    ss_tot = float(np.sum((y - y.mean()) ** 2))
                    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

                    coeff_rows = []
                    if not force_origin:
                        coeff_rows.append({"Term": "Intercept", "Coefficient": intercept})
                    for i, xname in enumerate(x_cols):
                        coeff_rows.append({"Term": xname, "Coefficient": float(betas[i])})
                    coeff_df = pd.DataFrame(coeff_rows)

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Rows used", f"{len(work)}")
                    with m2:
                        st.metric("R²", f"{r2:.4f}" if pd.notna(r2) else "N/A")
                    with m3:
                        st.metric("RMSE", f"{rmse:.4f}")

                    st.markdown("**Model Coefficients**")
                    st.dataframe(coeff_df, use_container_width=True, hide_index=True)

                    pred_df = work.copy()
                    pred_df["predicted_y"] = y_hat
                    pred_df["residual"] = y - y_hat
                    st.markdown("**Observed vs Predicted (Preview)**")
                    st.dataframe(pred_df[[y_col, "predicted_y", "residual"] + x_cols].head(200), use_container_width=True)

                    st.markdown("**Regression Plots**")
                    c_plot1, c_plot2 = st.columns(2)
                    with c_plot1:
                        st.caption("Observed vs Predicted")
                        ovp_df = pred_df[[y_col, "predicted_y"]].rename(columns={y_col: "observed"})
                        st.scatter_chart(ovp_df, x="predicted_y", y="observed", use_container_width=True)
                    with c_plot2:
                        st.caption("Residual vs Predicted")
                        st.scatter_chart(pred_df[["predicted_y", "residual"]], x="predicted_y", y="residual", use_container_width=True)

                    if len(x_cols) == 1:
                        st.caption("Single-variable fit view")
                        single_x = x_cols[0]
                        fit_df = pred_df[[single_x, y_col, "predicted_y"]].sort_values(single_x)
                        st.line_chart(
                            fit_df.set_index(single_x)[["predicted_y", y_col]].rename(columns={y_col: "observed_y"}),
                            use_container_width=True,
                        )

    with st.expander("Distribution Explorer", expanded=False):
        dist_col = st.selectbox("Select parameter", options=numeric_cols, key="analysis_dist_col")
        dist_mode = st.radio(
            "Distribution display",
            options=["Histogram (Bins)", "Smooth Curve"],
            horizontal=True,
            key="analysis_dist_mode",
        )
        bins = st.slider("Histogram bins", min_value=5, max_value=60, value=20, step=1, key="analysis_dist_bins")

        s = pd.to_numeric(df_local[dist_col], errors="coerce").dropna()
        if s.empty:
            st.warning("Selected parameter has no valid numeric values.")
        else:
            stats_df = pd.DataFrame([{
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }])
            st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True, hide_index=True)

            if dist_mode == "Histogram (Bins)":
                hist_counts, hist_edges = np.histogram(s.to_numpy(dtype=float), bins=bins)
                hist_df = pd.DataFrame({
                    "bin_start": np.round(hist_edges[:-1], 4),
                    "bin_end": np.round(hist_edges[1:], 4),
                    "count": hist_counts,
                })
                st.bar_chart(hist_df.set_index("bin_start")["count"])
                st.caption("Histogram shown as bin counts.")
            else:
                values = s.to_numpy(dtype=float)
                n = len(values)
                sigma = float(np.std(values, ddof=1)) if n > 1 else 0.0
                if n < 2 or sigma == 0.0:
                    st.info("Not enough variation to compute smooth curve. Showing histogram instead.")
                    hist_counts, hist_edges = np.histogram(values, bins=bins)
                    hist_df = pd.DataFrame({
                        "bin_start": np.round(hist_edges[:-1], 4),
                        "count": hist_counts,
                    })
                    st.bar_chart(hist_df.set_index("bin_start")["count"])
                else:
                    # Silverman's rule of thumb bandwidth (no scipy dependency)
                    h = 1.06 * sigma * (n ** (-1 / 5))
                    x_grid = np.linspace(values.min(), values.max(), 200)
                    z = (x_grid[:, None] - values[None, :]) / h
                    density = np.exp(-0.5 * z**2).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))
                    kde_df = pd.DataFrame({
                        "x": np.round(x_grid, 4),
                        "density": density,
                    }).set_index("x")
                    st.line_chart(kde_df, use_container_width=True)
                    st.caption("Smooth density curve (Gaussian KDE approximation).")

    with st.expander("Data Quality Snapshot", expanded=False):
        total_rows = len(df_local)
        missing_pct = df_local.isna().mean() * 100.0
        quality_df = pd.DataFrame({
            "Parameter": df_local.columns,
            "Missing %": [float(missing_pct.get(c, 0.0)) for c in df_local.columns],
            "Unique Values": [int(df_local[c].nunique(dropna=True)) for c in df_local.columns],
            "Dtype": [str(df_local[c].dtype) for c in df_local.columns],
        }).sort_values("Missing %", ascending=False)

        q1, q2 = st.columns(2)
        with q1:
            st.metric("Total Rows", f"{total_rows}")
        with q2:
            st.metric("Total Parameters", f"{len(df_local.columns)}")
        st.dataframe(quality_df, use_container_width=True, hide_index=True)


def render_ahp_tab():
    st.subheader("AHP Weights")
    st.write("Build a pairwise comparison table using the Saaty scale. The value means Criterion A is preferred over Criterion B.")

    meta = get_criteria_meta()
    label_map = {k: v for k, v in meta}
    criteria_keys = [k for k, _ in meta]
    standard_criteria_keys = [k for k, _ in BASE_CRITERIA_META if k in criteria_keys]

    if st.button("Use Standard HCFCD Criteria", key="btn_ahp_use_standard_hcfcd"):
        st.session_state["ahp_selected_criteria"] = standard_criteria_keys
        st.rerun()

    selected = st.multiselect(
        "Select criteria for AHP",
        options=criteria_keys,
        default=criteria_keys,
        format_func=lambda k: label_map.get(k, k),
        key="ahp_selected_criteria",
    )

    def _build_ahp_template(keys: list[str]) -> pd.DataFrame:
        labels_local = [label_map[k] for k in keys]
        m = pd.DataFrame("", index=labels_local, columns=labels_local)
        for i in range(len(labels_local)):
            for j in range(len(labels_local)):
                if i == j:
                    m.iat[i, j] = "1"
                elif i < j:
                    m.iat[i, j] = ""
                else:
                    m.iat[i, j] = ""
        m.index.name = "Parameter"
        return m

    st.markdown("### Export / Import AHP Matrix")
    st.caption("CSV format: first row and first column are parameter names, diagonal values are 1. Fill only the upper triangular cells (above the diagonal).")
    d1, d2 = st.columns(2)
    with d1:
        all_template = _build_ahp_template(criteria_keys)
        st.download_button(
            "Download AHP Template (All Parameters)",
            data=all_template.to_csv(index=True).encode("utf-8"),
            file_name="ahp_template_all_parameters.csv",
            mime="text/csv",
            key="ahp_download_all",
        )
    with d2:
        selected_template = _build_ahp_template(selected if selected else criteria_keys)
        st.download_button(
            "Download AHP Template (Selected Parameters)",
            data=selected_template.to_csv(index=True).encode("utf-8"),
            file_name="ahp_template_selected_parameters.csv",
            mime="text/csv",
            key="ahp_download_selected",
        )

    uploaded_ahp = st.file_uploader("Import completed AHP matrix CSV", type=["csv"], key="ahp_upload_matrix")

    if len(selected) < 2:
        st.warning("Select at least two criteria to run AHP.")
        return

    saaty_options = [
        ("1/9 (Extreme)", 1/9),
        ("1/7 (Very strong)", 1/7),
        ("1/5 (Strong)", 1/5),
        ("1/3 (Moderate)", 1/3),
        ("1 (Equal)", 1.0),
        ("3 (Moderate)", 3.0),
        ("5 (Strong)", 5.0),
        ("7 (Very strong)", 7.0),
        ("9 (Extreme)", 9.0),
    ]
    option_labels = [o[0] for o in saaty_options]
    option_values = {o[0]: o[1] for o in saaty_options}
    def _parse_saaty_value(x) -> float:
        s = str(x).strip()
        if not s:
            return np.nan
        if "/" in s:
            parts = s.split("/", 1)
            try:
                num = float(parts[0].strip())
                den = float(parts[1].strip())
                if den != 0:
                    return num / den
            except Exception:
                return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan

    def _nearest_saaty_label(v: float) -> str:
        return min(option_labels, key=lambda lab: abs(option_values[lab] - v))
    scale_df = pd.DataFrame({
        "Saaty Scale": option_labels,
        "Meaning": [
            "Criterion A is extremely less important than B",
            "Criterion A is very strongly less important than B",
            "Criterion A is strongly less important than B",
            "Criterion A is moderately less important than B",
            "Criteria A and B are equally important",
            "Criterion A is moderately more important than B",
            "Criterion A is strongly more important than B",
            "Criterion A is very strongly more important than B",
            "Criterion A is extremely more important than B",
        ],
    })
    st.dataframe(scale_df, use_container_width=True)

    if st.session_state.get("ahp_pairs_selected") != selected:
        pairs = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                pairs.append({
                    "Criterion A": label_map[selected[i]],
                    "Criterion B": label_map[selected[j]],
                    "Preference": "1 (Equal)",
                })
        st.session_state["ahp_pairs"] = pairs
        st.session_state["ahp_pairs_selected"] = list(selected)
        st.session_state["ahp_pairs_editor_version"] = st.session_state.get("ahp_pairs_editor_version", 0) + 1

    if uploaded_ahp is not None and st.button("Load AHP Matrix from CSV", key="btn_load_ahp_csv"):
        try:
            m = pd.read_csv(uploaded_ahp, index_col=0)
            m.index = [str(x).strip() for x in m.index]
            m.columns = [str(x).strip() for x in m.columns]

            selected_labels = [label_map[k] for k in selected]
            selected_keys = list(selected)

            row_lookup_exact = {str(x).strip(): str(x).strip() for x in m.index}
            row_lookup_norm = {_norm_col_name(str(x)): str(x).strip() for x in m.index}
            col_lookup_exact = {str(x).strip(): str(x).strip() for x in m.columns}
            col_lookup_norm = {_norm_col_name(str(x)): str(x).strip() for x in m.columns}

            resolved_rows = {}
            resolved_cols = {}
            mapping_rows = []

            for k, lab in zip(selected_keys, selected_labels):
                aliases = [lab, k, SCORE_COL_MAP.get(k, "")]
                aliases = [a for a in aliases if a]

                row_match = next((row_lookup_exact[a] for a in aliases if a in row_lookup_exact), None)
                if row_match is None:
                    row_match = next((row_lookup_norm[_norm_col_name(a)] for a in aliases if _norm_col_name(a) in row_lookup_norm), None)

                col_match = next((col_lookup_exact[a] for a in aliases if a in col_lookup_exact), None)
                if col_match is None:
                    col_match = next((col_lookup_norm[_norm_col_name(a)] for a in aliases if _norm_col_name(a) in col_lookup_norm), None)

                resolved_rows[lab] = row_match
                resolved_cols[lab] = col_match
                mapping_rows.append({
                    "Parameter": lab,
                    "Matched Row": row_match or "",
                    "Matched Column": col_match or "",
                })

            mapping_df = pd.DataFrame(mapping_rows)
            missing_params = [r["Parameter"] for r in mapping_rows if not r["Matched Row"] or not r["Matched Column"]]
            if missing_params:
                st.error("Some selected AHP parameters could not be matched in uploaded CSV. Update CSV headers or selected parameters and try again.")
                with st.expander("Show Matched Row/Column Details", expanded=False):
                    st.dataframe(mapping_df, use_container_width=True)
            else:
                m_sel = pd.DataFrame(index=selected_labels, columns=selected_labels, dtype=object)
                for a_lab in selected_labels:
                    for b_lab in selected_labels:
                        m_sel.loc[a_lab, b_lab] = m.loc[resolved_rows[a_lab], resolved_cols[b_lab]]

                imported_pairs = []
                imported_count = 0
                defaulted_count = 0
                for i in range(len(selected_labels)):
                    for j in range(i + 1, len(selected_labels)):
                        raw_val = m_sel.iat[i, j]
                        v = _parse_saaty_value(raw_val)
                        if pd.isna(v) or v <= 0:
                            pref = "1 (Equal)"
                            defaulted_count += 1
                        else:
                            pref = _nearest_saaty_label(v)
                            imported_count += 1
                        imported_pairs.append({
                            "Criterion A": selected_labels[i],
                            "Criterion B": selected_labels[j],
                            "Preference": pref,
                        })
                st.session_state["ahp_pairs"] = imported_pairs
                st.session_state["ahp_pairs_selected"] = list(selected)
                st.session_state["ahp_pairs_editor_version"] = st.session_state.get("ahp_pairs_editor_version", 0) + 1
                st.session_state["ahp_import_summary"] = {
                    "imported_count": imported_count,
                    "defaulted_count": defaulted_count,
                    "mapping_df": mapping_df,
                }
                st.success(f"AHP matrix imported. Imported {imported_count} pair values; defaulted {defaulted_count} pairs to 1 (Equal).")
                st.rerun()
        except Exception as ex:
            st.error(f"Could not import AHP matrix: {ex}")

    if "ahp_import_summary" in st.session_state:
        summary = st.session_state["ahp_import_summary"]
        st.caption(f"Last import summary: {summary.get('imported_count', 0)} imported, {summary.get('defaulted_count', 0)} defaulted to 1 (Equal).")
        map_df = summary.get("mapping_df")
        if isinstance(map_df, pd.DataFrame):
            with st.expander("Show Last Import Matched Row/Column Table", expanded=False):
                st.dataframe(map_df, use_container_width=True)

    pairs_df = pd.DataFrame(st.session_state.get("ahp_pairs", []))
    pairs_editor_key = f"ahp_pairs_table_{st.session_state.get('ahp_pairs_editor_version', 0)}"
    edited = st.data_editor(
        pairs_df,
        use_container_width=True,
        hide_index=True,
        key=pairs_editor_key,
        column_config={
            "Preference": st.column_config.SelectboxColumn(options=option_labels)
        },
    )
    st.session_state["ahp_pairs"] = edited.to_dict("records")

    if st.button("Compute AHP Weights", key="btn_compute_ahp"):
        n = len(selected)
        matrix = np.ones((n, n), dtype=float)
        key_by_label = {label_map[k]: k for k in selected}
        for _, row in edited.iterrows():
            a_label = row.get("Criterion A", "")
            b_label = row.get("Criterion B", "")
            pref_label = row.get("Preference", "1 (Equal)")
            if a_label in key_by_label and b_label in key_by_label:
                i = selected.index(key_by_label[a_label])
                j = selected.index(key_by_label[b_label])
                val = option_values.get(pref_label, 1.0)
                matrix[i, j] = val
                matrix[j, i] = 1.0 / val if val != 0 else 1.0

        weights, cr = ahp_weights(matrix)
        st.session_state["ahp_weights"] = {selected[i]: float(weights[i]) for i in range(n)}
        st.session_state["ahp_cr"] = float(cr)
        st.session_state["ahp_selected_snapshot"] = list(selected)

    if "ahp_weights" in st.session_state and st.session_state.get("ahp_selected_snapshot") == list(selected):
        w = st.session_state["ahp_weights"]
        cr = st.session_state.get("ahp_cr", 0.0)
        w_df = pd.DataFrame([
            {"Criterion": label_map[k], "Weight": round(v, 6)} for k, v in w.items()
        ])
        st.markdown("### AHP Weights")
        w_style = w_df.style.set_properties(**{"text-align": "center"})
        w_style = w_style.set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
        st.dataframe(w_style, use_container_width=True)
        if cr > 0.1:
            st.warning(f"Consistency Ratio (CR) = {cr:.3f}. Consider revising pairwise comparisons (CR should be <= 0.10).")
        else:
            st.success(f"Consistency Ratio (CR) = {cr:.3f}.")

        if st.button("Use AHP weights as Direct Weights", key="btn_apply_ahp"):
            for k, v in w.items():
                st.session_state["weights_pct"][k] = round(float(v) * 100.0, 1)
            st.success("AHP weights applied to Direct Weights.")


def render_topsis_tab():
    st.subheader("Ranking")
    st.write("Select criteria, then choose a ranking method.")

    weights_pct = st.session_state.get("weights_pct", {k: float(v) * 100.0 for k, v in config["weights"].items()})
    weights_dec = {k: round(float(v) / 100.0, 6) for k, v in weights_pct.items()}
    config_run = copy.deepcopy(config)
    config_run["weights"] = weights_dec

    df_source = st.session_state["df_work"].copy()
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df_source.columns]
    scoring_available = len(missing_required) == 0

    # Create a fallback frame so ranking can still proceed even when framework fields are incomplete.
    df_fallback = df_source.copy()
    for c in REQUIRED_COLUMNS:
        if c not in df_fallback.columns:
            df_fallback[c] = ""
    if "project_id" in df_fallback.columns:
        ids = pd.to_numeric(df_fallback["project_id"], errors="coerce")
        if ids.isna().all():
            df_fallback["project_id"] = np.arange(1, len(df_fallback) + 1)
    if "project_name" in df_fallback.columns:
        names = df_fallback["project_name"].astype(str).str.strip()
        blank = names.eq("") | names.eq("nan")
        if blank.any():
            fallback_ids = pd.to_numeric(
                df_fallback.get("project_id", pd.Series(range(1, len(df_fallback) + 1))),
                errors="coerce",
            ).fillna(0).astype(int)
            df_fallback.loc[blank, "project_name"] = fallback_ids.loc[blank].apply(lambda x: f"Project {x}" if x > 0 else "Project")

    try:
        results, warnings = compute_scores(df_fallback, config_run)
    except Exception:
        results, warnings = df_source.copy(), []

    score_cols = [SCORE_COL_MAP[k] for k, _ in BASE_CRITERIA_META if SCORE_COL_MAP.get(k) in results.columns]
    custom_keys = [c.get("key") for c in st.session_state.get("custom_criteria", []) if c.get("key")]
    numeric_custom = [
        c for c in custom_keys
        if c in results.columns and pd.api.types.is_numeric_dtype(results[c])
    ]
    numeric_all = [c for c in results.columns if pd.api.types.is_numeric_dtype(results[c])]
    available_cols = list(dict.fromkeys(score_cols + numeric_custom + numeric_all))
    st.caption("Only numeric columns are available for TOPSIS. Add numeric columns in the Data tab if needed.")

    base_label_map = {k: label for k, label in BASE_CRITERIA_META}
    label_by_col = {v: base_label_map.get(k, v) for k, v in SCORE_COL_MAP.items() if v}
    for item in st.session_state.get("custom_criteria", []):
        key = item.get("key")
        label = item.get("label") or item.get("key")
        if key:
            label_by_col[key] = label

    btn_col_1, btn_col_2 = st.columns(2)
    with btn_col_1:
        if st.button("Use AHP-selected criteria order", key="btn_use_ahp_order"):
            ahp_order = st.session_state.get("ahp_selected_criteria", [])
            ordered_cols = []
            for k in ahp_order:
                c = SCORE_COL_MAP.get(k, k)
                if c in available_cols and c not in ordered_cols:
                    ordered_cols.append(c)
            if ordered_cols:
                st.session_state["topsis_selected_cols"] = ordered_cols
                st.success("Ranking criteria set from AHP selected criteria (same order).")
                st.rerun()
            else:
                st.warning("No overlap found between AHP-selected criteria and currently available ranking columns.")
    with btn_col_2:
        if st.button("Use Standard HCFCD Criteria", key="btn_topsis_use_standard_hcfcd"):
            standard_cols = [
                SCORE_COL_MAP[k]
                for k, _ in BASE_CRITERIA_META
                if SCORE_COL_MAP.get(k) in available_cols
            ]
            if standard_cols:
                st.session_state["topsis_selected_cols"] = standard_cols
                st.success("Ranking criteria set to standard HCFCD criteria.")
                st.rerun()
            else:
                st.warning("No standard HCFCD score columns are currently available.")

    selected_cols = st.multiselect(
        "Select criteria for ranking",
        options=available_cols,
        default=score_cols if score_cols else available_cols[:min(5, len(available_cols))],
        format_func=lambda c: label_by_col.get(c, c),
        key="topsis_selected_cols",
    )

    # Only warn for missing framework fields that are relevant to currently selected criteria.
    if missing_required:
        field_to_framework_key = {
            "project_type": {"existing_conditions", "environment", "multiple_benefits"},
            "total_cost": {"people_efficiency", "structures_efficiency"},
            "people_benefitted": {"people_efficiency"},
            "structures_benefitted": {"structures_efficiency"},
            "svi_class": {"svi"},
            "maintenance_class": {"maintenance"},
        }
        inv_score_map = {v: k for k, v in SCORE_COL_MAP.items()}
        selected_framework_keys = {inv_score_map[c] for c in selected_cols if c in inv_score_map}
        missing_relevant = []
        for f in missing_required:
            impacted = field_to_framework_key.get(f, set())
            if impacted & selected_framework_keys:
                missing_relevant.append(f)
        if missing_relevant:
            st.warning(
                "Framework-scored columns may be partial because these required fields are missing for selected criteria: "
                + ", ".join(missing_relevant)
            )

    if warnings:
        inv_score_map = {v: k for k, v in SCORE_COL_MAP.items()}
        selected_framework_keys = {inv_score_map[c] for c in selected_cols if c in inv_score_map}

        def _warning_relevant(msg: str) -> bool:
            marker = "missing scores for:"
            m = str(msg).lower()
            if marker not in m:
                return True
            tail = m.split(marker, 1)[1]
            tail = tail.split("(", 1)[0]
            crits = {x.strip() for x in tail.split(",") if x.strip()}
            return bool(crits & selected_framework_keys)

        filtered_warnings = [w for w in warnings if _warning_relevant(w)]
        if filtered_warnings:
            st.info("Some selected framework criteria have missing/invalid inputs; missing scores were treated as 0 during scoring.")
            with st.expander("Show relevant scoring warnings by project", expanded=False):
                for w in filtered_warnings:
                    st.write(f"- {w}")

    if len(selected_cols) < 2:
        st.warning("Select at least two criteria to rank.")
        return

    method_options = [
        "Weighted Sum (Direct Weights)",
        "Direct Weights + TOPSIS",
        "AHP Weights + TOPSIS",
        "Equal Weights + TOPSIS",
    ]
    if not scoring_available:
        st.caption(
            "You can still run AHP/TOPSIS with selected numeric columns. "
            "Only framework score columns that depend on missing required fields may be incomplete."
        )
    compare_methods = st.checkbox("Compare two methods side-by-side", key="ranking_compare_methods")
    if compare_methods:
        methods = st.multiselect(
            "Select up to two methods",
            options=method_options,
            default=["Weighted Sum (Direct Weights)", "Direct Weights + TOPSIS"],
            max_selections=2,
            key="ranking_methods",
        )
    else:
        method = st.radio(
            "Ranking method",
            options=method_options,
            index=0,
            horizontal=False,
            key="ranking_method",
        )
        methods = [method]

    if st.session_state.get("topsis_selected_snapshot") != selected_cols:
        rows = []
        for c in selected_cols:
            rows.append({
                "Criterion": c,
                "Type": "Benefit",
            })
        st.session_state["topsis_settings"] = rows
        st.session_state["topsis_selected_snapshot"] = list(selected_cols)

    settings_df = pd.DataFrame(st.session_state.get("topsis_settings", []))
    if "Criterion" not in settings_df.columns:
        settings_df = pd.DataFrame(columns=["Criterion", "Type"])
    settings_df = settings_df[settings_df["Criterion"].isin(selected_cols)]
    if "Better Value" in settings_df.columns and "Type" not in settings_df.columns:
        settings_df["Type"] = settings_df["Better Value"].apply(
            lambda v: "Cost" if str(v).lower() == "lower" else "Benefit"
        )
    if "Better Value" in settings_df.columns:
        settings_df = settings_df.drop(columns=["Better Value"])

    label_by_col = {v: f"Score - {label}" for k, label in BASE_CRITERIA_META for v in [SCORE_COL_MAP.get(k)] if v}
    for item in st.session_state.get("custom_criteria", []):
        key = item.get("key")
        label = item.get("label") or item.get("key")
        if key:
            label_by_col[key] = label
    settings_df["Criterion"] = settings_df["Criterion"].map(lambda c: label_by_col.get(c, c))

    edited = None
    if any(m.endswith("+ TOPSIS") for m in methods):
        st.caption("Type: Benefit = higher is better, Cost = lower is better.")
        edited = st.data_editor(
            settings_df,
            use_container_width=True,
            hide_index=True,
            key="topsis_settings_table",
            column_config={
                "Type": st.column_config.SelectboxColumn(options=["Benefit", "Cost"]),
            },
        )
        st.session_state["topsis_settings"] = edited.to_dict("records")

    if st.button("Run Ranking", key="btn_run_topsis", type="primary"):
        data = results[selected_cols].copy()
        for c in selected_cols:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        if data.isna().any().any():
            st.warning("Some selected columns have missing or non-numeric values. They are treated as 0.")
            data = data.fillna(0.0)

        results_by_source = {}
        for method in methods:
            if method == "Weighted Sum (Direct Weights)":
                inv_score_map = {v: k for k, v in SCORE_COL_MAP.items()}
                w = []
                missing_cols = []
                for c in selected_cols:
                    k = inv_score_map.get(c, c)
                    if k not in weights_pct:
                        missing_cols.append(c)
                        w.append(1.0)
                    else:
                        w.append(float(weights_pct.get(k, 0.0)) / 100.0)
                if missing_cols:
                    st.warning("Some selected columns have no direct weight. Using equal weight for those columns.")
                w = np.array(w, dtype=float)
                w_sum = w.sum()
                if w_sum == 0:
                    w = np.ones(len(selected_cols), dtype=float) / len(selected_cols)
                else:
                    w = w / w_sum

                decision = data.to_numpy(dtype=float)
                scores = (decision * w).sum(axis=1)

                out = results.copy()
                out["ranking_score"] = scores
                out["ranking_rank"] = out["ranking_score"].rank(ascending=False, method="min").astype(int)
                out = out.sort_values(["ranking_score", "project_name"], ascending=[False, True]).reset_index(drop=True)

                show_cols = ["ranking_rank", "project_id", "project_name", "ranking_score"] + selected_cols
                results_by_source[method] = out[show_cols]
                continue

            benefit_flags = []
            ideal_best = []
            ideal_worst = []

            label_to_col = {v: k for k, v in label_by_col.items()}
            missing_rows = 0
            for _, row in edited.iterrows():
                c = label_to_col.get(row["Criterion"], row["Criterion"])
                if c not in selected_cols:
                    missing_rows += 1
                    continue
                is_benefit = str(row.get("Type", "Benefit")).lower() != "cost"
                benefit_flags.append(is_benefit)
                if is_benefit:
                    best_val = data[c].max()
                    worst_val = data[c].min()
                else:
                    best_val = data[c].min()
                    worst_val = data[c].max()
                ideal_best.append(float(best_val))
                ideal_worst.append(float(worst_val))

            if missing_rows or len(benefit_flags) != len(selected_cols):
                for c in selected_cols:
                    if len(benefit_flags) >= len(selected_cols):
                        break
                    if c not in [label_to_col.get(r["Criterion"], r["Criterion"]) for _, r in edited.iterrows()]:
                        benefit_flags.append(True)
                        ideal_best.append(float(data[c].max()))
                        ideal_worst.append(float(data[c].min()))

            inv_score_map = {v: k for k, v in SCORE_COL_MAP.items()}

            def weights_for_source(source: str) -> np.ndarray:
                if source == "AHP weights":
                    ahp = st.session_state.get("ahp_weights")
                    if not ahp:
                        st.warning("AHP weights not found. Falling back to equal weights.")
                        return np.ones(len(selected_cols), dtype=float)
                    w = []
                    for c in selected_cols:
                        k = inv_score_map.get(c, c)
                        if k not in ahp:
                            st.warning("AHP weights do not cover all selected criteria. Falling back to equal weights.")
                            return np.ones(len(selected_cols), dtype=float)
                        w.append(float(ahp[k]))
                    return np.array(w, dtype=float)

                if source == "Direct weights":
                    w = []
                    missing_cols = []
                    for c in selected_cols:
                        k = inv_score_map.get(c, c)
                        if k not in weights_pct:
                            missing_cols.append(c)
                            w.append(1.0)
                        else:
                            w.append(float(weights_pct.get(k, 0.0)) / 100.0)
                    if missing_cols:
                        st.warning("Some selected columns have no direct weight. Using equal weight for those columns.")
                    return np.array(w, dtype=float)

                return np.ones(len(selected_cols), dtype=float)

            if method == "Direct Weights + TOPSIS":
                weight_sources = ["Direct weights"]
            elif method == "AHP Weights + TOPSIS":
                weight_sources = ["AHP weights"]
            else:
                weight_sources = ["Equal weights"]

            decision = data.to_numpy(dtype=float)
            for source in weight_sources:
                w = weights_for_source(source)
                scores = topsis_rank(
                    decision,
                    w,
                    benefit_flags=benefit_flags,
                    ideal_best=np.array(ideal_best),
                    ideal_worst=np.array(ideal_worst),
                )
                out = results.copy()
                out["topsis_score"] = scores
                out["topsis_rank"] = out["topsis_score"].rank(ascending=False, method="min").astype(int)
                out = out.sort_values(["topsis_score", "project_name"], ascending=[False, True]).reset_index(drop=True)
                show_cols = ["topsis_rank", "project_id", "project_name", "topsis_score"] + selected_cols
                results_by_source[method] = out[show_cols]

        st.session_state["topsis_results_multi"] = results_by_source

    if "topsis_results_multi" in st.session_state:
        results_by_source = st.session_state["topsis_results_multi"]
        if len(results_by_source) == 1:
            st.dataframe(next(iter(results_by_source.values())), use_container_width=True)
        else:
            col_a, col_b = st.columns(2)
            items = list(results_by_source.items())
            with col_a:
                st.markdown(f"**{items[0][0]}**")
                st.dataframe(items[0][1], use_container_width=True)
            with col_b:
                st.markdown(f"**{items[1][0]}**")
                st.dataframe(items[1][1], use_container_width=True)


with tab_data:
    render_data_tab()

with tab_tools:
    render_data_tools_tab()

with tab_analysis:
    render_parameter_analysis_tab()

with tab_weights:
    render_direct_weights_tab()

with tab_ahp:
    render_ahp_tab()

with tab_topsis:
    render_topsis_tab()
