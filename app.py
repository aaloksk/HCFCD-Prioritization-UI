import json
import os
import copy
import io
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
import base64
import sys

from engine import compute_scores, ahp_weights, topsis_rank

st.set_page_config(page_title="Project Prioritization UI", layout="wide")

# ----------------------------
# Helpers and config
# ----------------------------
def resource_path(rel_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.dirname(__file__), rel_path)

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
        return df

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

    return df


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


def read_uploaded_csv_with_id(uploaded_file) -> tuple[pd.DataFrame, str]:
    raw = uploaded_file.getvalue()
    file_hash = hashlib.md5(raw).hexdigest()
    upload_id = f"{uploaded_file.name}:{uploaded_file.size}:{file_hash}"
    df = pd.read_csv(io.BytesIO(raw))
    return df, upload_id


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

# ----------------------------
# Sidebar: Data + Config + Weights
# ----------------------------
with st.sidebar:
    st.header("Controls")
    st.subheader("Data Source")
    uploaded_sidebar = st.file_uploader("Upload input CSV", type=["csv"], key="sidebar_upload")
    if st.button("Load template dataset", key="btn_load_template_sidebar"):
        st.session_state["df_work"] = preprocess_uploaded_df(pd.read_csv(resource_path("input_template.csv")))
        st.session_state["uploaded_file_name"] = "input_template.csv"
        st.session_state["loaded_sidebar_upload_id"] = None
        st.session_state["svi_source_force_reset"] = True
    st.subheader("Scoring Config")
    if st.checkbox("Show config", value=False):
        st.json(config)
    st.subheader("Template")
    with open(resource_path("input_template.csv"), "r", encoding="utf-8") as tf:
        template_bytes = tf.read().encode("utf-8")
    st.download_button("Download input template CSV", data=template_bytes, file_name="input_template.csv", mime="text/csv")
    st.divider()
    st.header("Weights")
    with st.expander("Direct Weights (sidebar)", expanded=False):
        render_weights_inputs("sidebar", show_reference=False)

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

tab_data, tab_tools, tab_weights, tab_ahp, tab_topsis = st.tabs(
    ["Prioritization Database", "Data Tools", "Direct Weights", "AHP Weights", "Ranking"]
)


def render_data_tab():
    st.subheader("Data Source")
    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        uploaded_main = st.file_uploader("Upload input CSV", type=["csv"], key="main_upload")
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

    st.divider()
    st.subheader("Custom Criteria Mapping")
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
    else:
        st.info("No extra columns found to map. Add columns above or upload a dataset with additional fields.")



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
        return

    numeric_like_cols = []
    for c in df_local.columns:
        if pd.api.types.is_numeric_dtype(df_local[c]) or str(c).strip().lower() in {"svi", "svi_value"}:
            numeric_like_cols.append(c)

    if not numeric_like_cols:
        st.info("No numeric columns available for SVI reclassification.")
        return

    svi_col_exact = find_column_case_insensitive(df_local, "svi")
    default_source = svi_col_exact or numeric_like_cols[0]
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
            index=default_index,
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

        if df_local.columns.duplicated().any():
            st.warning("Duplicate column names detected in dataset. Preview is showing first occurrence of each duplicate column.")
        preview_df = df_local.loc[:, ~df_local.columns.duplicated(keep="first")].copy()
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
            index=numeric_like_cols.index(st.session_state["rain_reclass_source_col"]),
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

    with st.expander("Current Working Dataset Preview (Full)", expanded=False):
        preview_df = df_local.loc[:, ~df_local.columns.duplicated(keep="first")].copy()
        st.dataframe(preview_df, use_container_width=True, height=320)


def render_ahp_tab():
    st.subheader("AHP Weights")
    st.write("Build a pairwise comparison table using the Saaty scale. The value means Criterion A is preferred over Criterion B.")

    meta = get_criteria_meta()
    label_map = {k: v for k, v in meta}
    criteria_keys = [k for k, _ in meta]

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

    if uploaded_ahp is not None and st.button("Load AHP Matrix from CSV", key="btn_load_ahp_csv"):
        try:
            m = pd.read_csv(uploaded_ahp, index_col=0)
            m.index = [str(x).strip() for x in m.index]
            m.columns = [str(x).strip() for x in m.columns]

            selected_labels = [label_map[k] for k in selected]
            missing_rows = [lab for lab in selected_labels if lab not in m.index]
            missing_cols = [lab for lab in selected_labels if lab not in m.columns]
            if missing_rows or missing_cols:
                st.error("Uploaded matrix does not contain all currently selected AHP parameters.")
            else:
                m_sel = m.loc[selected_labels, selected_labels]
                imported_pairs = []
                for i in range(len(selected_labels)):
                    for j in range(i + 1, len(selected_labels)):
                        raw_val = m_sel.iat[i, j]
                        v = _parse_saaty_value(raw_val)
                        pref = "1 (Equal)" if pd.isna(v) or v <= 0 else _nearest_saaty_label(v)
                        imported_pairs.append({
                            "Criterion A": selected_labels[i],
                            "Criterion B": selected_labels[j],
                            "Preference": pref,
                        })
                st.session_state["ahp_pairs"] = imported_pairs
                st.session_state["ahp_pairs_selected"] = list(selected)
                st.success("AHP matrix imported into the pairwise table.")
                st.rerun()
        except Exception as ex:
            st.error(f"Could not import AHP matrix: {ex}")

    pairs_df = pd.DataFrame(st.session_state.get("ahp_pairs", []))
    edited = st.data_editor(
        pairs_df,
        use_container_width=True,
        hide_index=True,
        key="ahp_pairs_table",
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

    results, warnings = compute_scores(st.session_state["df_work"], config_run)
    if warnings:
        st.info("Some projects have missing or invalid inputs. Missing scores are treated as 0 for TOPSIS.")

    score_cols = [SCORE_COL_MAP[k] for k, _ in BASE_CRITERIA_META if SCORE_COL_MAP.get(k) in results.columns]
    custom_keys = [c.get("key") for c in st.session_state.get("custom_criteria", []) if c.get("key")]
    numeric_custom = [
        c for c in custom_keys
        if c in results.columns and pd.api.types.is_numeric_dtype(results[c])
    ]
    available_cols = score_cols + numeric_custom
    st.caption("Only numeric columns are available for TOPSIS. Add numeric columns in the Data tab if needed.")

    base_label_map = {k: label for k, label in BASE_CRITERIA_META}
    label_by_col = {v: base_label_map.get(k, v) for k, v in SCORE_COL_MAP.items() if v}
    for item in st.session_state.get("custom_criteria", []):
        key = item.get("key")
        label = item.get("label") or item.get("key")
        if key:
            label_by_col[key] = label

    selected_cols = st.multiselect(
        "Select criteria for ranking",
        options=available_cols,
        default=score_cols,
        format_func=lambda c: label_by_col.get(c, c),
        key="topsis_selected_cols",
    )

    if len(selected_cols) < 2:
        st.warning("Select at least two criteria to rank.")
        return

    method_options = [
        "Weighted Sum (Direct Weights)",
        "Direct Weights + TOPSIS",
        "AHP Weights + TOPSIS",
        "Equal Weights + TOPSIS",
    ]
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
                "Use Auto": True,
                "Ideal Best": 10.0,
                "Ideal Worst": 1.0,
            })
        st.session_state["topsis_settings"] = rows
        st.session_state["topsis_selected_snapshot"] = list(selected_cols)

    settings_df = pd.DataFrame(st.session_state.get("topsis_settings", []))
    if "Criterion" not in settings_df.columns:
        settings_df = pd.DataFrame(columns=["Criterion", "Type", "Use Auto", "Ideal Best", "Ideal Worst"])
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
                "Use Auto": st.column_config.CheckboxColumn(),
                "Ideal Best": st.column_config.NumberColumn(),
                "Ideal Worst": st.column_config.NumberColumn(),
            },
        )
        st.session_state["topsis_settings"] = edited.to_dict("records")

    if st.button("Run Ranking", key="btn_run_topsis"):
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

                auto_flag = bool(row.get("Use Auto", True))
                best_val = row.get("Ideal Best")
                worst_val = row.get("Ideal Worst")

                if auto_flag or pd.isna(best_val) or pd.isna(worst_val):
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
                        ideal_best.append(10.0)
                        ideal_worst.append(1.0)

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

with tab_weights:
    render_direct_weights_tab()

with tab_ahp:
    render_ahp_tab()

with tab_topsis:
    render_topsis_tab()
