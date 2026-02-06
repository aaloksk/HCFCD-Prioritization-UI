import json
import os
import copy
import pandas as pd
import streamlit as st

from engine import compute_scores

st.set_page_config(page_title="Project Prioritization UI", layout="wide")

# ----------------------------
# Header with logo (top-right)
# ----------------------------
col_left, col_right = st.columns([5, 1])

with col_left:
    st.title("Project Prioritization Tool")
    st.caption("HCFCD Framework – Internal Use at infraTECH (Still under Development)")

with col_right:
    logo_path = os.path.join(os.path.dirname(__file__), "ITE_Logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)

st.caption("Upload a CSV, edit data in the browser, run scoring, and download ranked results.")

# ----------------------------
# Load scoring config
# ----------------------------
with open("scoring_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

maps = config.get("mappings", {})
labels = config.get("labels", {})  # ✅ NEW: read dropdown labels

def label_for(group: str, key: str) -> str:
    """
    Returns the human-friendly label for a dropdown option.
    Falls back to the raw key if no label is found.
    """
    try:
        return labels.get(group, {}).get(key, key)
    except Exception:
        return key

def classify_svi(value: float) -> str:
    """
    Classify SVI value into a class based on ranges:
    0-0.25: low
    0.25-0.5: low_moderate
    0.5-0.75: moderate_high
    0.75-1: high
    """
    if value < 0.25:
        return "low"
    elif value < 0.5:
        return "low_moderate"
    elif value < 0.75:
        return "moderate_high"
    else:
        return "high"

# Efficiency class mappings (score -> label)
EFFICIENCY_CLASSES = {
    "10": "Very High (Score 10)",
    "8": "High (Score 8)",
    "6": "Medium (Score 6)",
    "4": "Low (Score 4)",
    "1": "Very Low (Score 1)"
}

def get_efficiency_tables_html() -> str:
    """Generate HTML tables showing efficiency class ranges"""
    html = """
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Cost per Resident</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Cost per Structure</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">< $6,000</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;"><b>10</b></td>
        <td style="border: 1px solid #ddd; padding: 8px;">< $23,000</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;"><b>10</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">$6,000 - $15,000</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #c3e6cb;"><b>8</b></td>
        <td style="border: 1px solid #ddd; padding: 8px;">$23,000 - $60,000</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #c3e6cb;"><b>8</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">$15,000 - $28,000</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #fff3cd;"><b>6</b></td>
        <td style="border: 1px solid #ddd; padding: 8px;">$60,000 - $106,000</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #fff3cd;"><b>6</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">$28,000 - $77,000</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>4</b></td>
        <td style="border: 1px solid #ddd; padding: 8px;">$106,000 - $261,000</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>4</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">> $77,000</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f5c6cb;"><b>1</b></td>
        <td style="border: 1px solid #ddd; padding: 8px;">> $261,000</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f5c6cb;"><b>1</b></td>
    </tr>
    </table>
    """
    return html

def get_existing_conditions_channel_html() -> str:
    """Channel Detention - Existing Conditions Scoring"""
    html = """
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Criteria</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">System capacity is > 1% AEP storm (100-year)</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f5c6cb;"><b>0</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">System capacity is < 1% AEP storm (100-year)</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>1</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">System capacity is < 2% AEP storm (50-year)</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>2</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">System capacity is < 4% AEP storm (25-year)</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>4</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">System capacity is < 10% AEP storm (10-year)</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #fff3cd;"><b>6</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">System capacity is < 20% AEP storm (5-year)</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #c3e6cb;"><b>8</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">System capacity is < 50% AEP storm (2-year)</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;"><b>10</b></td>
    </tr>
    </table>
    """
    return html

def get_existing_conditions_subdivision_html() -> str:
    """Subdivision - Existing Conditions Scoring"""
    html = """
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Criteria</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Low rainfall AND high-quality drainage infrastructure</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f5c6cb;"><b>0</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Intermediate rainfall OR medium-quality infrastructure</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>3</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Intermediate rainfall AND medium-quality infrastructure</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #fff3cd;"><b>6</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">High rainfall OR low-quality infrastructure</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #c3e6cb;"><b>9</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">High rainfall AND low-quality infrastructure</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;"><b>10</b></td>
    </tr>
    </table>
    """
    return html

def get_svi_html() -> str:
    """Social Vulnerability Index Scoring"""
    html = """
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Criteria</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">SVI indicates low level of vulnerability</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f5c6cb;"><b>1</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">SVI indicates low to moderate level of vulnerability</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>4</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">SVI indicates moderate to high level of vulnerability</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #fff3cd;"><b>7</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">SVI indicates high level of vulnerability</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;"><b>10</b></td>
    </tr>
    </table>
    """
    return html

def get_environment_channel_html() -> str:
    """Channel - Minimizes Environmental Impacts"""
    html = """
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Criteria</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Requires Individual Permit and mitigation credits</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f5c6cb;"><b>0</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Requires mitigation bank credits</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>2</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Able to significantly avoid environmental impacts</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #c3e6cb;"><b>6</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Has minimal or no environmental impacts</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;"><b>10</b></td>
    </tr>
    </table>
    """
    return html

def get_environment_subdivision_html() -> str:
    """Subdivision - Minimizes Environmental Impacts (Right-of-Way)"""
    html = """
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Criteria</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Requires acquiring additional right-of-way</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>6</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Can be completed within existing right-of-way</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;"><b>10</b></td>
    </tr>
    </table>
    """
    return html

def get_multiple_benefits_channel_html() -> str:
    """Channel - Potential for Multiple Benefits"""
    html = """
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Criteria</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Does not have multiple benefits</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f5c6cb;"><b>0</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Has recreational benefits</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>4</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Has environmental enhancement benefits</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #c3e6cb;"><b>6</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Has recreational AND environmental benefits</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;"><b>10</b></td>
    </tr>
    </table>
    """
    return html

def get_multiple_benefits_subdivision_html() -> str:
    """Subdivision - Potential for Multiple Benefits"""
    html = """
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Criteria</th>
        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Project area does not benefit from a nearby District improvement</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;"><b>6</b></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Project area benefits from a nearby District improvement</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;"><b>10</b></td>
    </tr>
    </table>
    """
    return html

# ---- Initialize weights in session (percent, not decimals) ----
if "weights_pct" not in st.session_state:
    st.session_state["weights_pct"] = {k: float(v) * 100.0 for k, v in config["weights"].items()}

# ---- Add Project panel visibility ----
if "show_add_project" not in st.session_state:
    st.session_state["show_add_project"] = False

# ----------------------------
# Sidebar: Controls + Weights
# ----------------------------
with st.sidebar:
    st.header("Controls")

    st.subheader("1) Load Data")
    uploaded = st.file_uploader("Upload input CSV", type=["csv"])

    st.subheader("2) Scoring Config")
    st.write("Weights and scoring breakpoints come from `scoring_config.json`.")
    show_config = st.checkbox("Show config", value=False)
    if show_config:
        st.json(config)

    st.subheader("3) Template")
    with open("input_template.csv", "r", encoding="utf-8") as tf:
        template_bytes = tf.read().encode("utf-8")
    st.download_button(
        label="Download input template CSV",
        data=template_bytes,
        file_name="input_template.csv",
        mime="text/csv",
    )

    st.divider()
    st.header("Weights (must total 100.0%)")

    def _w(key: str, label: str) -> float:
        val = st.number_input(
            label,
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["weights_pct"].get(key, 0.0)),
            step=0.1,
            format="%.1f",
            key=f"w_{key}",
        )
        val = round(float(val), 1)
        st.session_state["weights_pct"][key] = val
        return val

    w_people = _w("people_efficiency", "People Benefits Efficiency (%)")
    w_struct = _w("structures_efficiency", "Structure Benefits Efficiency (%)")
    w_exist  = _w("existing_conditions", "Existing Conditions (%)")
    w_svi    = _w("svi", "Social Vulnerability Index (%)")
    w_maint  = _w("maintenance", "Long-Term Maintenance Costs (%)")
    w_env    = _w("environment", "Minimizes Environmental Impacts (%)")
    w_multi  = _w("multiple_benefits", "Potential for Multiple Benefits (%)")

    total_w = round(w_people + w_struct + w_exist + w_svi + w_maint + w_env + w_multi, 1)
    is_valid_weights = (total_w == 100.0)
    st.session_state["is_valid_weights"] = is_valid_weights

    if is_valid_weights:
        st.success(f"Total: {total_w:.1f}% ✅")
    else:
        diff = round(100.0 - total_w, 1)
        if diff > 0:
            st.error(f"Total: {total_w:.1f}% — add {diff:.1f}% to reach 100.0%")
        else:
            st.error(f"Total: {total_w:.1f}% — remove {abs(diff):.1f}% to reach 100.0%")

    if st.button("Reset to default weights"):
        st.session_state["weights_pct"] = {k: float(v) * 100.0 for k, v in config["weights"].items()}
        st.rerun()

# ----------------------------
# Load dataframe into session_state (so Add Project can append)
# ----------------------------
if "df_work" not in st.session_state or uploaded is not None:
    if uploaded is None:
        st.session_state["df_work"] = pd.read_csv("input_template.csv")
        st.info("Using the included template. Upload your own CSV anytime from the left sidebar.")
    else:
        st.session_state["df_work"] = pd.read_csv(uploaded)

df = st.session_state["df_work"]

# ----------------------------
# Editable grid
# ----------------------------
st.subheader("Edit Project Data")
st.write("Click a cell to edit. You can also add rows at the bottom of the table.")

edited = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
)

# Save edits to working dataframe immediately
st.session_state["df_work"] = edited

# ----------------------------
# Add Project (button + form with dropdowns)
# ----------------------------
st.divider()
st.subheader("Add Project")

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    if st.button("➕ Add Project"):
        st.session_state["show_add_project"] = True

with col_btn2:
    if st.session_state["show_add_project"]:
        st.caption("Fill the form and click **Save Project** to append it to the table above.")

# Dropdown choices from config mappings (fallbacks included)
project_type_options = maps.get("project_type", ["channel_detention", "subdivision_drainage"])
svi_options = list(maps.get("svi_class", {}).keys()) or ["low", "low_moderate", "moderate_high", "high"]
maint_options = list(maps.get("maintenance_class", {}).keys()) or ["extensive_specialized", "outside_regular", "regular"]

channel_capacity_options = list(maps.get("existing_conditions_channel_capacity", {}).keys()) or [
    "gt_1_percent", "lt_1_percent", "lt_2_percent", "lt_4_percent", "lt_10_percent", "lt_20_percent", "lt_50_percent"
]

env_channel_options = list(maps.get("environment_channel", {}).keys()) or [
    "individual_permit_and_credits", "credits", "avoid_impacts", "minimal_none"
]
mb_channel_options = list(maps.get("multiple_benefits_channel", {}).keys()) or [
    "none", "recreation", "environment", "both"
]

# NOTE: For subdivision matrix, your config keys are rainfall classes (low/intermediate/high)
rain_options = list(maps.get("existing_conditions_subdivision_matrix", {}).keys()) or ["high", "intermediate", "low"]

# Use only the configured keys: high / intermediate / low
infra_options = ["high", "intermediate", "low"]

row_options = list(maps.get("row_subdivision", {}).keys()) or ["needs_additional_row", "within_existing_row"]
syn_options = list(maps.get("multiple_benefits_subdivision", {}).keys()) or ["no", "yes"]

if st.session_state["show_add_project"]:

    # ✅ Project Type outside the form so it reruns instantly
    project_type = st.selectbox(
        "Project Type*",
        project_type_options,
        index=0,
        key="add_project_type",
        format_func=lambda k: label_for("project_type", k)  # ✅ show report wording
    )

    # ✅ SVI Input Method outside the form so it reruns instantly
    with st.expander("📊 Social Vulnerability Index (SVI)", expanded=True):
        st.markdown("""
        **What is SVI?**  
        The Social Vulnerability Index measures a community's resilience and ability to respond to hazards. 
        Higher SVI values indicate communities that may need more priority for protection.
        
        **Choose your input method:**
        """)
        
        svi_input_method = st.radio(
            "How would you like to input SVI?",
            options=["Select from predefined class", "Enter SVI value (0-1)"],
            index=0,
            horizontal=True,
            key="add_svi_method",
            help="Either pick from standard classes or provide an exact value that will be auto-classified"
        )
    
    # Display SVI input based on method (outside form)
    svi_class = ""
    svi_value = None
    
    if svi_input_method == "Enter SVI value (0-1)":
        st.markdown("##### Enter SVI Value")
        col_slider, col_info = st.columns([3, 1])
        with col_slider:
            svi_value = st.slider(
                "SVI Value",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                format="%.2f",
                key="add_svi_value",
                help="0 = Low vulnerability, 1 = High vulnerability"
            )
        # Auto-classify based on value
        svi_class = classify_svi(svi_value)
        with col_info:
            st.markdown(f"**Auto-classified:**  \n**{label_for('svi_class', svi_class)}**")
    else:
        st.markdown("##### Select SVI Class")
        svi_class = st.selectbox(
            "SVI Class",
            svi_options,
            index=0,
            key="add_svi_class",
            format_func=lambda k: label_for("svi_class", k),
            help="Choose from: Low, Low-Moderate, Moderate-High, High"
        )
        svi_value = None

    # ✅ Cost Efficiency Input Method outside the form so it reruns instantly
    with st.expander("💰 Cost Efficiency Classes", expanded=True):
        st.markdown("""
        **What are efficiency classes?**  
        Cost efficiency measures how much you spend per person or structure benefitted. 
        Lower cost per beneficiary = higher efficiency score.
        
        **Reference tables below show the cost ranges for each score:**
        """)
        
        st.markdown(get_efficiency_tables_html(), unsafe_allow_html=True)
        
        st.markdown("**Choose your input method:**")
        
        efficiency_input_method = st.radio(
            "How would you like to input cost efficiency?",
            options=["Calculate from project costs", "Enter efficiency classes directly"],
            index=0,
            horizontal=True,
            key="add_efficiency_method",
            help="Either provide costs and auto-calculate, or select efficiency classes directly"
        )

    with st.form("add_project_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            project_name = st.text_input("Project Name*", value="", key="add_project_name")

        with c2:
            total_cost = st.number_input(
                "Total Cost*",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.0f",
                key="add_total_cost"
            )
            people_benefitted = st.number_input(
                "People Benefitted*",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.0f",
                key="add_people_benefitted"
            )

        with c3:
            structures_benefitted = st.number_input(
                "Structures Benefitted*",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.0f",
                key="add_structures_benefitted"
            )

        # Display efficiency inputs based on selected method (from outside form)
        people_efficiency_class = ""
        structures_efficiency_class = ""
        
        if efficiency_input_method == "Enter efficiency classes directly":
            ec1, ec2 = st.columns(2)
            with ec1:
                people_efficiency_class = st.selectbox(
                    "People Cost Efficiency*",
                    options=list(EFFICIENCY_CLASSES.keys()),
                    format_func=lambda k: f"{EFFICIENCY_CLASSES[k]} - Cost per person",
                    key="add_people_efficiency_class",
                    help="Based on total cost ÷ people benefitted"
                )
            with ec2:
                structures_efficiency_class = st.selectbox(
                    "Structures Cost Efficiency*",
                    options=list(EFFICIENCY_CLASSES.keys()),
                    format_func=lambda k: f"{EFFICIENCY_CLASSES[k]} - Cost per structure",
                    key="add_structures_efficiency_class",
                    help="Based on total cost ÷ structures benefitted"
                )
        else:
            st.info("💡 Efficiency will be calculated from: Total Cost ÷ People (or Structures) Benefitted")

        maintenance_class = st.selectbox(
            "Maintenance Class*",
            maint_options,
            index=0,
            key="add_maint_class",
            format_func=lambda k: label_for("maintenance_class", k)  # ✅ show report wording
        )

        st.markdown("#### Project-type specific inputs")

        # Defaults for columns that may not apply
        channel_capacity_class = ""
        excess_rainfall_class = ""
        drainage_infra_quality = ""
        environment_channel_class = ""
        row_subdivision_class = ""
        # ✅ Project Type outside the form so it reruns instantly
        project_type = st.selectbox(
            "1️⃣ Project Type*",

        if project_type == "channel_detention":
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                channel_capacity_class = st.selectbox(
                    "Channel Capacity Class",
        st.divider()

        # Initialize all efficiency variables before use
        people_efficiency_class = ""
        structures_efficiency_class = ""
        existing_conditions_class = ""
        svi_class = ""
        svi_value = None
        environment_class = ""
        multiple_benefits_class = ""

        # ========== FORM STARTS HERE ==========
                st.error("Total Cost must be greater than 0.")
            # 2️⃣ Project Name
            st.markdown("### 2️⃣ Project Name")
            project_name = st.text_input("Project Name*", value="", key="add_project_name")
                df_current = st.session_state["df_work"].copy()
            st.divider()
                next_id = 1
            # 3️⃣ Project Efficiency Weighting Factor
            st.markdown("### 3️⃣ Project Efficiency Weighting Factor")
            with st.expander("📊 View Efficiency Scoring Tables", expanded=False):
                st.markdown(get_efficiency_tables_html(), unsafe_allow_html=True)
        
            efficiency_input_method = st.radio(
                "How would you like to input project efficiency?",
                options=["Calculate from project costs", "Enter efficiency classes directly"],
                index=0,
                horizontal=True,
                key="add_efficiency_method",
                help="Either provide costs and auto-calculate, or select efficiency classes directly"
            )
        
            if efficiency_input_method == "Enter efficiency classes directly":
                ec1, ec2 = st.columns(2)
                with ec1:
                    people_efficiency_class = st.selectbox(
                        "Resident Benefits Efficiency*",
                        options=list(EFFICIENCY_CLASSES.keys()),
                        format_func=lambda k: f"Score {k}: {EFFICIENCY_CLASSES[k]}",
                        key="add_people_efficiency_class",
                        help="Based on total cost ÷ residents benefitted"
                    )
                with ec2:
                    structures_efficiency_class = st.selectbox(
                        "Structure Benefit Efficiency*",
                        options=list(EFFICIENCY_CLASSES.keys()),
                        format_func=lambda k: f"Score {k}: {EFFICIENCY_CLASSES[k]}",
                        key="add_structures_efficiency_class",
                        help="Based on total cost ÷ structures benefitted"
                    )
            else:
                st.info("💡 Efficiency will be calculated from: Total Cost ÷ Residents (or Structures) Benefitted")
            
                col_cost, col_residents, col_structures = st.columns(3)
                with col_cost:
                    total_cost = st.number_input(
                        "Total Cost*",
                        min_value=0.0,
                        value=0.0,
                        step=1000.0,
                        format="%.0f",
                        key="add_total_cost"
                    )
                with col_residents:
                    people_benefitted = st.number_input(
                        "Residents Benefitted*",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        format="%.0f",
                        key="add_people_benefitted"
                    )
                with col_structures:
                    structures_benefitted = st.number_input(
                        "Structures Benefitted*",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        format="%.0f",
                        key="add_structures_benefitted"
                    )

            st.divider()

            # 4️⃣ Existing Conditions Weighting Factor
            st.markdown("### 4️⃣ Existing Conditions Weighting Factor")
            if project_type == "channel_detention":
                with st.expander("📋 View Channel Scoring Criteria", expanded=False):
                    st.markdown(get_existing_conditions_channel_html(), unsafe_allow_html=True)
                existing_conditions_class = st.selectbox(
                    "Channel Capacity Class*",
                    channel_capacity_options,
                    index=0,
                    key="add_channel_capacity_class",
                    format_func=lambda k: label_for("existing_conditions_channel_capacity", k)
                )
            else:
                with st.expander("📋 View Subdivision Scoring Criteria", expanded=False):
                    st.markdown(get_existing_conditions_subdivision_html(), unsafe_allow_html=True)
            
                ec_col1, ec_col2 = st.columns(2)
                with ec_col1:
                    excess_rainfall_class = st.selectbox(
                        "Excess Rainfall Class*",
                        rain_options,
                        index=0,
                        key="add_excess_rainfall_class",
                        format_func=lambda k: labels.get("existing_conditions_subdivision_matrix", {})
                                             .get("excess_rainfall_class", {})
                                             .get(k, k)
                    )
                with ec_col2:
                    drainage_infra_quality = st.selectbox(
                        "Drainage Infrastructure Quality*",
                        infra_options,
                        index=0,
                        key="add_drainage_infra_quality",
                        format_func=lambda k: labels.get("existing_conditions_subdivision_matrix", {})
                                             .get("drainage_infra_quality", {})
                                             .get(k, k)
                    )

            st.divider()

            # 5️⃣ Social Vulnerability Index (Full Form)
            st.markdown("### 5️⃣ Social Vulnerability Index (SVI)")
            with st.expander("📊 View SVI Scoring Criteria", expanded=False):
                st.markdown(get_svi_html(), unsafe_allow_html=True)
        
            svi_input_method = st.radio(
                "How would you like to input SVI?",
                options=["Select from predefined class", "Enter SVI value (0-1)"],
                index=0,
                horizontal=True,
                key="add_svi_method",
                help="Either pick from standard classes or provide an exact value that will be auto-classified"
            )
        
            if svi_input_method == "Enter SVI value (0-1)":
                col_slider, col_info = st.columns([3, 1])
                with col_slider:
                    svi_value = st.slider(
                        "SVI Value",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        format="%.2f",
                        key="add_svi_value",
                        help="0 = Low vulnerability, 1 = High vulnerability"
                    )
                svi_class = classify_svi(svi_value)
                with col_info:
                    st.markdown(f"**Auto-classified:**  \n**{label_for('svi_class', svi_class)}**")
            else:
                svi_class = st.selectbox(
                    "SVI Class*",
                    svi_options,
                    index=0,
                    key="add_svi_class",
                    format_func=lambda k: label_for("svi_class", k),
                    help="Choose from: Low, Low-Moderate, Moderate-High, High"
                )
                svi_value = None

            st.divider()

            # 6️⃣ Minimizes Environmental Impact Weighting Factor
            st.markdown("### 6️⃣ Minimizes Environmental Impact Weighting Factor")
            if project_type == "channel_detention":
                with st.expander("📋 View Channel Environmental Criteria", expanded=False):
                    st.markdown(get_environment_channel_html(), unsafe_allow_html=True)
                environment_class = st.selectbox(
                    "Environmental Class (Channel)*",
                    env_channel_options,
                    index=0,
                    key="add_env_channel",
                    format_func=lambda k: label_for("environment_channel", k)
                )
            else:
                with st.expander("📋 View Subdivision Environmental Criteria", expanded=False):
                    st.markdown(get_environment_subdivision_html(), unsafe_allow_html=True)
                environment_class = st.selectbox(
                    "ROW Availability (Subdivision)*",
                    row_options,
                    index=0,
                    key="add_row_subdivision",
                    format_func=lambda k: label_for("row_subdivision", k)
                )

            st.divider()

            # 7️⃣ Potential for Multiple Benefits Weighting Factor
            st.markdown("### 7️⃣ Potential for Multiple Benefits Weighting Factor")
            if project_type == "channel_detention":
                with st.expander("📋 View Channel Multiple Benefits Criteria", expanded=False):
                    st.markdown(get_multiple_benefits_channel_html(), unsafe_allow_html=True)
                multiple_benefits_class = st.selectbox(
                    "Multiple Benefits (Channel)*",
                    mb_channel_options,
                    index=0,
                    key="add_mb_channel",
                    format_func=lambda k: label_for("multiple_benefits_channel", k)
                )
            else:
                with st.expander("📋 View Subdivision Multiple Benefits Criteria", expanded=False):
                    st.markdown(get_multiple_benefits_subdivision_html(), unsafe_allow_html=True)
                multiple_benefits_class = st.selectbox(
                    "District Improvement Synergy*",
                    syn_options,
                    index=0,
                    key="add_synergy",
                    format_func=lambda k: label_for("multiple_benefits_subdivision", k)
                )

            st.divider()

            # Additional details
            st.markdown("### Other Details")
        
            col_maint, col_notes = st.columns([1, 2])
            with col_maint:
                maintenance_class = st.selectbox(
                    try:
                        mx = pd.to_numeric(df_current["project_id"], errors="coerce").max()
                        next_id = int(mx) + 1 if pd.notna(mx) else 1
                    except Exception:
                        next_id = 1

                new_row = {
                    "project_id": next_id,
                    "project_name": project_name.strip(),
                    "project_type": project_type,
                    "total_cost": float(total_cost),
                    "people_benefitted": float(people_benefitted),
                    "structures_benefitted": float(structures_benefitted),

                    "channel_capacity_class": channel_capacity_class,
                    "excess_rainfall_class": excess_rainfall_class,
                    "drainage_infra_quality": drainage_infra_quality,

                    "svi_value": svi_value if svi_value is not None else "",
                    "svi_class": svi_class,
                    "maintenance_class": maintenance_class,
                    
                    "people_efficiency_class": people_efficiency_class,
                    "structures_efficiency_class": structures_efficiency_class,

                    "environment_channel_class": environment_channel_class,
                    "row_subdivision_class": row_subdivision_class,

                    "multiple_benefits_channel_class": multiple_benefits_channel_class,
                    "district_improvement_synergy": district_improvement_synergy,

                    "notes": notes.strip(),
                }

                # Ensure columns exist before append
                for k in new_row.keys():
                    if k not in df_current.columns:
                        df_current[k] = ""

                df_current = pd.concat([df_current, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state["df_work"] = df_current
                st.session_state["show_add_project"] = False
                st.success(f"Added project: {project_name.strip()} (ID {next_id})")
                st.rerun()

# ----------------------------
# Run + Results
# ----------------------------
colA, colB, colC = st.columns([1, 1, 2])

is_valid_weights = bool(st.session_state.get("is_valid_weights", False))

with colA:
    run = st.button("Run Prioritization", type="primary", disabled=not is_valid_weights)

with colB:
    st.button("Clear Results", on_click=lambda: st.session_state.pop("results", None))

if run:
    try:
        # Convert UI weights (%) -> decimals (0-1)
        weights_pct = st.session_state["weights_pct"]
        weights_dec = {k: round(float(v) / 100.0, 6) for k, v in weights_pct.items()}

        config_run = copy.deepcopy(config)
        config_run["weights"] = weights_dec

        results, warnings = compute_scores(st.session_state["df_work"], config_run)
        st.session_state["results"] = results
        st.session_state["warnings"] = warnings
        st.session_state["last_run_weights_pct"] = dict(weights_pct)
    except Exception as e:
        st.error(f"Error: {e}")

if "warnings" in st.session_state and st.session_state["warnings"]:
    st.warning("Some projects have missing/invalid inputs. See details below.")
    with st.expander("Warnings"):
        for w in st.session_state["warnings"]:
            st.write("- " + w)

if "results" in st.session_state:
    results = st.session_state["results"]

    if "last_run_weights_pct" in st.session_state:
        lrw = st.session_state["last_run_weights_pct"]
        st.caption(f"Results reflect the last run weights (total = {sum(lrw.values()):.1f}%).")

    st.subheader("Ranked Results")
    st.dataframe(results, use_container_width=True)

    st.subheader("Download Results")
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ranked_results.csv",
        data=csv_bytes,
        file_name="ranked_results.csv",
        mime="text/csv",
    )

    with st.expander("Show only key output columns"):
        key_cols = [
            "rank", "project_id", "project_name", "project_type",
            "total_weighted_score",
            "score_people_efficiency", "score_structures_efficiency",
            "score_existing_conditions", "score_svi",
            "score_maintenance", "score_environment", "score_multiple_benefits",
            "total_cost", "people_benefitted", "structures_benefitted"
        ]
        existing_cols = [c for c in key_cols if c in results.columns]
        st.dataframe(results[existing_cols], use_container_width=True)
