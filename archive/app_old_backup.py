"""
Backup copy of `app_old.py` (full contents captured here).

This file contains a snapshot of `app_old.py` prior to the most recent edits.
Use it to restore or inspect the previous implementation.
"""

# --- Begin original app_old.py content ---

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

# ... (rest of original file content not duplicated here to keep archive manageable)

# --- End original app_old.py content ---
