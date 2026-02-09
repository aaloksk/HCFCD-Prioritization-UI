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

    if out.empty:
        if "total_weighted_score" not in out.columns:
            out["total_weighted_score"] = []
        out["rank"] = []
        return out, warnings

    # Rank (higher score = better)
    out["rank"] = out["total_weighted_score"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values(["total_weighted_score", "project_name"], ascending=[False, True]).reset_index(drop=True)

    return out, warnings


def ahp_weights(pairwise_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Returns (weights, consistency_ratio).
    pairwise_matrix must be a square reciprocal matrix.
    """
    if pairwise_matrix.ndim != 2 or pairwise_matrix.shape[0] != pairwise_matrix.shape[1]:
        raise ValueError("pairwise_matrix must be a square matrix.")

    n = pairwise_matrix.shape[0]
    if n == 0:
        raise ValueError("pairwise_matrix must be non-empty.")

    # Principal eigenvector method
    eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)
    max_index = int(np.argmax(eigenvalues.real))
    max_eigval = float(eigenvalues.real[max_index])
    weights = np.abs(eigenvectors[:, max_index].real)
    weights = weights / weights.sum() if weights.sum() != 0 else np.ones(n) / n

    # Consistency ratio
    if n == 1:
        return weights, 0.0

    ci = (max_eigval - n) / (n - 1)
    ri_table = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }
    ri = ri_table.get(n, 1.49)
    cr = (ci / ri) if ri != 0 else 0.0
    return weights, float(cr)


def topsis_rank(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    benefit_flags: List[bool] | None = None,
    ideal_best: np.ndarray | None = None,
    ideal_worst: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns TOPSIS scores (higher is better) for each alternative (row).
    """
    if decision_matrix.ndim != 2:
        raise ValueError("decision_matrix must be 2D.")
    if decision_matrix.shape[1] != weights.shape[0]:
        raise ValueError("weights size must match number of criteria.")

    m, n = decision_matrix.shape
    if benefit_flags is None:
        benefit_flags = [True] * n
    if len(benefit_flags) != n:
        raise ValueError("benefit_flags size must match number of criteria.")

    # Normalize
    denom = np.sqrt((decision_matrix ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    norm = decision_matrix / denom

    # Weighted normalized matrix
    w = weights / weights.sum() if weights.sum() != 0 else np.ones(n) / n
    weighted = norm * w

    # Ideal best/worst
    if ideal_best is not None or ideal_worst is not None:
        if ideal_best is None or ideal_worst is None:
            best_raw = np.zeros(n)
            worst_raw = np.zeros(n)
            for j in range(n):
                if benefit_flags[j]:
                    best_raw[j] = np.max(decision_matrix[:, j])
                    worst_raw[j] = np.min(decision_matrix[:, j])
                else:
                    best_raw[j] = np.min(decision_matrix[:, j])
                    worst_raw[j] = np.max(decision_matrix[:, j])
            if ideal_best is None:
                ideal_best = best_raw
            if ideal_worst is None:
                ideal_worst = worst_raw

        ideal_best = np.array(ideal_best, dtype=float)
        ideal_worst = np.array(ideal_worst, dtype=float)
        if ideal_best.shape[0] != n or ideal_worst.shape[0] != n:
            raise ValueError("ideal_best and ideal_worst must match number of criteria.")

        ideal_best = (ideal_best / denom) * w
        ideal_worst = (ideal_worst / denom) * w
    else:
        ideal_best = np.zeros(n)
        ideal_worst = np.zeros(n)
        for j in range(n):
            if benefit_flags[j]:
                ideal_best[j] = np.max(weighted[:, j])
                ideal_worst[j] = np.min(weighted[:, j])
            else:
                ideal_best[j] = np.min(weighted[:, j])
                ideal_worst[j] = np.max(weighted[:, j])

    # Distances
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    denom = dist_best + dist_worst
    denom[denom == 0] = 1.0
    scores = dist_worst / denom
    return scores
