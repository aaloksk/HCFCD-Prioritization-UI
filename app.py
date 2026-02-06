import json
import os
import copy
import pandas as pd
import streamlit as st

from engine import compute_scores

st.set_page_config(page_title="Project Prioritization UI", layout="wide")
import json
import os
import copy
import pandas as pd
import streamlit as st

from engine import compute_scores

st.set_page_config(page_title="Project Prioritization UI", layout="wide")

# ----------------------------
# Helpers and config
# ----------------------------
with open("scoring_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

maps = config.get("mappings", {})
labels = config.get("labels", {})

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

# ----------------------------
# Header
# ----------------------------
col_left, col_right = st.columns([5, 1])
with col_left:
    st.title("Project Prioritization Tool")
    st.caption("HCFCD Framework – Internal Use at infraTECH")
with col_right:
    logo_path = os.path.join(os.path.dirname(__file__), "ITE_Logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)

st.caption("Upload a CSV, edit data, add projects, run scoring, and download results.")

# ----------------------------
# Sidebar: Data + Config + Weights
# ----------------------------
with st.sidebar:
    st.header("Controls")
    st.subheader("Load Data")
    uploaded = st.file_uploader("Upload input CSV", type=["csv"]) 
    st.subheader("Scoring Config")
    if st.checkbox("Show config", value=False):
        st.json(config)
    st.subheader("Template")
    with open("input_template.csv", "r", encoding="utf-8") as tf:
        template_bytes = tf.read().encode("utf-8")
    st.download_button("Download input template CSV", data=template_bytes, file_name="input_template.csv", mime="text/csv")
    st.divider()
    st.header("Weights (must total 100%)")

    def _w(key: str, label: str) -> float:
        val = st.number_input(label, min_value=0.0, max_value=100.0, value=float(st.session_state["weights_pct"].get(key, 0.0)), step=0.1, format="%.1f", key=f"w_{key}")
        val = round(float(val), 1)
        st.session_state["weights_pct"][key] = val
        return val

    w_people = _w("people_efficiency", "Resident Benefits Efficiency (%)")
    w_struct = _w("structures_efficiency", "Structure Benefit Efficiency (%)")
    w_exist  = _w("existing_conditions", "Existing Conditions (%)")
    w_svi    = _w("svi", "Social Vulnerability Index (%)")
    w_maint  = _w("maintenance", "Long-Term Maintenance Costs (%)")
    w_env    = _w("environment", "Minimizes Environmental Impacts (%)")
    w_multi  = _w("multiple_benefits", "Potential for Multiple Benefits (%)")

    total_w = round(w_people + w_struct + w_exist + w_svi + w_maint + w_env + w_multi, 1)
    st.session_state["is_valid_weights"] = (total_w == 100.0)
    if st.session_state["is_valid_weights"]:
        st.success(f"Total: {total_w:.1f}% ✅")
    else:
        diff = round(100.0 - total_w, 1)
        if diff > 0:
            st.error(f"Total: {total_w:.1f}% — add {diff:.1f}%")
        else:
            st.error(f"Total: {total_w:.1f}% — remove {abs(diff):.1f}%")

    if st.button("Reset to default weights", key="reset_default_weights"):
        st.session_state["weights_pct"] = {k: float(v) * 100.0 for k, v in config["weights"].items()}
        st.rerun()

# ----------------------------
# Load or initialize dataframe
# ----------------------------
if "df_work" not in st.session_state or uploaded is not None:
    if uploaded is None:
        st.session_state["df_work"] = pd.read_csv("input_template.csv")
        st.info("Using included template. You can upload a CSV from the sidebar.")
    else:
        st.session_state["df_work"] = pd.read_csv(uploaded)

df = st.session_state["df_work"]

# Editable grid
st.subheader("Edit Project Data")
st.write("Click a cell to edit. You can also add rows at the bottom of the table.")
edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")
st.session_state["df_work"] = edited

# ----------------------------
# Add Project (in-context, no form wrapper)
# ----------------------------
st.divider()
st.subheader("Add Project")
col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    if st.button("➕ Add Project", key="btn_add_project"):
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
    project_type = st.selectbox("1️⃣ Project Type*", project_type_options, index=0, key="add_project_type", format_func=lambda k: label_for("project_type", k))

    # 2) Project Name
    st.markdown("### 2️⃣ Project Name")
    project_name = st.text_input("Project Name*", value="", key="add_project_name")
    st.divider()

    # 3) Project Efficiency Weighting Factor (in-context)
    st.markdown("### 3️⃣ Project Efficiency Weighting Factor")
    with st.expander("💰 Project Efficiency Tables", expanded=False):
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

        def row_color(s):
            return [f'background-color: {colors.get(s.name_val, "transparent")}' for _ in s.index]

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
        st.info("💡 Efficiency will be calculated from: Total Cost ÷ Residents (or Structures) Benefitted")
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
    st.markdown("### 4️⃣ Existing Conditions Weighting Factor")
    if project_type == "channel_detention":
        with st.expander("📋 View Channel Scoring Criteria", expanded=False):
            st.markdown(get_existing_conditions_channel_html(), unsafe_allow_html=True)
        channel_capacity_class = st.selectbox("Channel Capacity Class*", channel_capacity_options, index=0, key="add_channel_capacity_class", format_func=lambda k: label_for("existing_conditions_channel_capacity", k))
        excess_rainfall_class = ""
        drainage_infra_quality = ""
    else:
        with st.expander("📋 View Subdivision Scoring Criteria", expanded=False):
            st.markdown(get_existing_conditions_subdivision_html(), unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            excess_rainfall_class = st.selectbox("Excess Rainfall Class*", rain_options, index=0, key="add_excess_rainfall_class")
        with ec2:
            drainage_infra_quality = st.selectbox("Drainage Infrastructure Quality*", infra_options, index=0, key="add_drainage_infra_quality")
        channel_capacity_class = ""

    st.divider()

    # 5) Social Vulnerability Index (SVI)
    st.markdown("### 5️⃣ Social Vulnerability Index (SVI)")
    with st.expander("📊 Social Vulnerability Index (SVI)", expanded=False):
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
    st.markdown("### 6️⃣ Minimizes Environmental Impact Weighting Factor")
    if project_type == "channel_detention":
        with st.expander("📋 View Channel Environmental Criteria", expanded=False):
            st.markdown(get_environment_channel_html(), unsafe_allow_html=True)
        environment_channel_class = st.selectbox("Environmental Class (Channel)*", env_channel_options, index=0, key="add_env_channel", format_func=lambda k: label_for("environment_channel", k))
        row_subdivision_class = ""
    else:
        with st.expander("📋 View Subdivision Environmental Criteria", expanded=False):
            st.markdown(get_environment_subdivision_html(), unsafe_allow_html=True)
        row_subdivision_class = st.selectbox("ROW Availability (Subdivision)*", row_options, index=0, key="add_row_subdivision", format_func=lambda k: label_for("row_subdivision", k))
        environment_channel_class = ""

    st.divider()

    # 7) Potential for Multiple Benefits
    st.markdown("### 7️⃣ Potential for Multiple Benefits Weighting Factor")
    if project_type == "channel_detention":
        with st.expander("📋 View Channel Multiple Benefits Criteria", expanded=False):
            st.markdown(get_multiple_benefits_channel_html(), unsafe_allow_html=True)
        multiple_benefits_channel_class = st.selectbox("Multiple Benefits (Channel)*", mb_channel_options, index=0, key="add_mb_channel", format_func=lambda k: label_for("multiple_benefits_channel", k))
        district_improvement_synergy = ""
    else:
        with st.expander("📋 View Subdivision Multiple Benefits Criteria", expanded=False):
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

    st.divider()

    # Action buttons
    colb1, colb2 = st.columns(2)
    with colb1:
        if st.button("✅ Save Project", key="btn_save_project"):
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

                df_current = pd.concat([df_current, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state["df_work"] = df_current
                st.session_state["show_add_project"] = False
                st.success(f"Added project: {project_name.strip()} (ID {next_id})")
                st.rerun()
    with colb2:
        if st.button("❌ Cancel", key="btn_cancel_add"):
            st.session_state["show_add_project"] = False
            st.rerun()

# ----------------------------
# Run + Results
# ----------------------------
colA, colB, colC = st.columns([1, 1, 2])
is_valid_weights = bool(st.session_state.get("is_valid_weights", False))

with colA:
    run = st.button("Run Prioritization", type="primary", disabled=not is_valid_weights, key="btn_run_prioritization")
with colB:
    st.button("Clear Results", on_click=lambda: st.session_state.pop("results", None), key="btn_clear_results")

if run:
    try:
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
    st.download_button(label="Download ranked_results.csv", data=csv_bytes, file_name="ranked_results.csv", mime="text/csv")

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
