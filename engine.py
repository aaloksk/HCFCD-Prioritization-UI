from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "project_id",
    "project_name",
    "project_type",
    "total_cost",
    "people_benefitted",
    "structures_benefitted",
    "svi_class",
    "maintenance_class",
]


@dataclass
class ScoreBreakdown:
    people_efficiency: float
    structures_efficiency: float
    existing_conditions: float
    svi: float
    maintenance: float
    environment: float
    multiple_benefits: float
    total_weighted: float


def _as_float(x: Any) -> float:
    if x is None or (isinstance(x, str) and x.strip() == ""):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def score_from_bins(value: float, bins: List[Dict[str, Any]]) -> float:
    """
    bins: list like [{"max": 500, "score": 10}, ...] sorted ascending by max
    """
    if np.isnan(value):
        return np.nan
    for b in bins:
        if value <= float(b["max"]):
            return float(b["score"])
    return float(bins[-1]["score"])


def _get(mapping: Dict[str, Any], key: str, default=np.nan):
    if key is None or (isinstance(key, str) and key.strip() == ""):
        return default
    return mapping.get(str(key).strip(), default)


def compute_scores(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns (results_df, warnings)
    results_df includes criterion scores + weighted total + rank.
    """
    warnings: List[str] = []

    # Basic validation
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    weights = config["weights"]
    maps = config["mappings"]
    bins = config["efficiency_bins"]

    out_rows = []

    for _, r in df.iterrows():
        project_type = str(r.get("project_type", "")).strip()

        cost = _as_float(r.get("total_cost"))
        people = _as_float(r.get("people_benefitted"))
        structs = _as_float(r.get("structures_benefitted"))

        # --- Efficiency (cost per unit)
        cpp = (cost / people) if (not np.isnan(cost) and not np.isnan(people) and people > 0) else np.nan
        cps = (cost / structs) if (not np.isnan(cost) and not np.isnan(structs) and structs > 0) else np.nan

        people_eff = score_from_bins(cpp, bins["people_cost_per_person"])
        struct_eff = score_from_bins(cps, bins["structures_cost_per_structure"])

        # --- Existing conditions
        existing = np.nan
        if project_type == "channel_detention":
            cap_class = str(r.get("channel_capacity_class", "")).strip()
            existing = _get(maps["existing_conditions_channel_capacity"], cap_class)
        elif project_type == "subdivision_drainage":
            rain = str(r.get("excess_rainfall_class", "")).strip()   # high/intermediate/low
            infra = str(r.get("drainage_infra_quality", "")).strip() # high/intermediate/low OR high/medium/low
            # normalize a common "medium" -> "intermediate"
            if infra == "medium":
                infra = "intermediate"
            matrix = maps["existing_conditions_subdivision_matrix"]
            existing = matrix.get(rain, {}).get(infra, np.nan)
        else:
            warnings.append(f"Unknown project_type '{project_type}' for project '{r.get('project_name')}'.")

        # --- SVI
        svi = _get(maps["svi_class"], str(r.get("svi_class", "")).strip())

        # --- Maintenance
        maint = _get(maps["maintenance_class"], str(r.get("maintenance_class", "")).strip())

        # --- Environment
        env = np.nan
        if project_type == "channel_detention":
            env_class = str(r.get("environment_channel_class", "")).strip()
            env = _get(maps["environment_channel"], env_class)
        elif project_type == "subdivision_drainage":
            row_class = str(r.get("row_subdivision_class", "")).strip()
            env = _get(maps["row_subdivision"], row_class)

        # --- Multiple benefits
        mult = np.nan
        if project_type == "channel_detention":
            mb = str(r.get("multiple_benefits_channel_class", "")).strip()
            mult = _get(maps["multiple_benefits_channel"], mb)
        elif project_type == "subdivision_drainage":
            syn = str(r.get("district_improvement_synergy", "")).strip()
            mult = _get(maps["multiple_benefits_subdivision"], syn)

        # Weighted total (skip NaNs by treating them as 0, but warn)
        crit = {
            "people_efficiency": people_eff,
            "structures_efficiency": struct_eff,
            "existing_conditions": existing,
            "svi": svi,
            "maintenance": maint,
            "environment": env,
            "multiple_benefits": mult,
        }

        nan_crit = [k for k, v in crit.items() if np.isnan(v)]
        if nan_crit:
            warnings.append(
                f"Project '{r.get('project_name')}' has missing scores for: {', '.join(nan_crit)} "
                f"(check inputs / config). Missing treated as 0 in total."
            )

        total = 0.0
        for k, v in crit.items():
            w = float(weights[k])
            total += (0.0 if np.isnan(v) else float(v)) * w

        out_rows.append({
            **{c: r.get(c) for c in df.columns},
            "cost_per_person": cpp,
            "cost_per_structure": cps,
            "score_people_efficiency": people_eff,
            "score_structures_efficiency": struct_eff,
            "score_existing_conditions": existing,
            "score_svi": svi,
            "score_maintenance": maint,
            "score_environment": env,
            "score_multiple_benefits": mult,
            "total_weighted_score": total
        })

    out = pd.DataFrame(out_rows)

    # Rank (higher score = better)
    out["rank"] = out["total_weighted_score"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values(["total_weighted_score", "project_name"], ascending=[False, True]).reset_index(drop=True)

    return out, warnings