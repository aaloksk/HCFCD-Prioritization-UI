import json
import os
import copy
import io
import hashlib
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk
import base64
import sys
import shapefile
from pyproj import CRS, Transformer
try:
    import folium
    from streamlit_folium import st_folium
    try:
        from folium.plugins import GroupedLayerControl
    except Exception:
        GroupedLayerControl = None
except Exception:
    folium = None
    st_folium = None
    GroupedLayerControl = None

from engine import compute_scores, ahp_weights, topsis_rank, REQUIRED_COLUMNS

st.set_page_config(page_title="Project Prioritization UI", layout="wide", initial_sidebar_state="collapsed")


def _collapse_sidebar_once() -> None:
    if st.session_state.get("_sidebar_collapsed_once", False):
        return
    components.html(
        """
        <script>
          const d = window.parent.document;
          const closeBtn = d.querySelector('button[aria-label="Close sidebar"]');
          if (closeBtn) { closeBtn.click(); }
        </script>
        """,
        height=0,
    )
    st.session_state["_sidebar_collapsed_once"] = True

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
    workspace_files = []
    for name, path in get_importable_database_files(extensions=(".json",)):
        try:
            with open(path, "r", encoding="utf-8") as jf:
                payload = json.load(jf)
            df_blob = payload.get("df_work", {}) if isinstance(payload, dict) else {}
            if isinstance(df_blob, dict) and "data" in df_blob and "columns" in df_blob:
                workspace_files.append((name, path))
        except Exception:
            continue
    return workspace_files


def get_database_csv_files() -> list[tuple[str, str]]:
    return [
        (name, path)
        for name, path in get_importable_database_files(extensions=(".csv",))
        if str(name).strip().lower().endswith("database.csv")
    ]


def get_ahp_csv_files() -> list[tuple[str, str]]:
    return [
        (name, path)
        for name, path in get_importable_database_files(extensions=(".csv",))
        if "ahp" in str(name).lower() and "final" in str(name).lower()
    ]


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


AUTH_USERS = {
    "aaloksk": {"passcode": "9841", "role": "main"},
    "ite": {"passcode": "8164", "role": "standard"},
    "hce": {"passcode": "8164", "role": "standard"},
}


def _history_file_path() -> str:
    export_dir = get_workspace_export_dir()
    base_dir = export_dir if export_dir else os.path.dirname(__file__)
    return os.path.join(base_dir, "access_history.json")


def _load_access_history() -> list[dict]:
    p = _history_file_path()
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_access_history(records: list[dict]) -> None:
    p = _history_file_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except Exception:
        pass


def _start_access_session(user_key: str, person_name: str) -> None:
    canonical_user = user_key
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    sid = uuid.uuid4().hex
    st.session_state["auth_user"] = canonical_user
    st.session_state["auth_person_name"] = person_name.strip()
    st.session_state["auth_session_id"] = sid
    st.session_state["auth_started_at"] = now

    records = _load_access_history()
    records.append(
        {
            "session_id": sid,
            "username": canonical_user,
            "person_name": person_name.strip(),
            "started_at_utc": now,
            "last_seen_utc": now,
            "runtime_minutes": 0.0,
        }
    )
    _save_access_history(records)


def _update_access_session_runtime() -> None:
    sid = st.session_state.get("auth_session_id")
    started_at = st.session_state.get("auth_started_at")
    if not sid or not started_at:
        return
    try:
        start_dt = datetime.fromisoformat(str(started_at).replace("Z", ""))
        now_dt = datetime.utcnow()
        runtime_min = max(0.0, (now_dt - start_dt).total_seconds() / 60.0)
        now = now_dt.isoformat(timespec="seconds") + "Z"
    except Exception:
        return

    records = _load_access_history()
    for rec in records:
        if str(rec.get("session_id", "")) == str(sid):
            rec["last_seen_utc"] = now
            rec["runtime_minutes"] = round(runtime_min, 2)
            break
    _save_access_history(records)


def get_landing_background_path() -> str | None:
    candidates = [
        "landing_background.png",
        "landing_background.jpg",
        "landing_background.jpeg",
        "hero_background.png",
        "hero_background.jpg",
        "infra_background.png",
        "infra_background.jpg",
    ]
    for name in candidates:
        path = resource_path(name)
        if os.path.exists(path):
            return path
    return None


def _default_database_name(db_files: list[tuple[str, str]], token: str | None = None) -> str | None:
    if not db_files:
        return None
    if token:
        for name, _ in db_files:
            if token.lower() in name.lower():
                return name
    return db_files[0][0]


def _strip_z_from_coords(coords):
    if not isinstance(coords, (list, tuple)):
        return coords
    if not coords:
        return coords
    first = coords[0]
    if isinstance(first, (list, tuple)):
        if first and isinstance(first[0], (list, tuple)):
            return [_strip_z_from_coords(x) for x in coords]
        return [[pt[0], pt[1]] for pt in coords]
    return coords


def _shape_to_geojson_geometry(shape_obj) -> dict:
    geom = shape_obj.__geo_interface__
    out = dict(geom)
    if "coordinates" in out:
        out["coordinates"] = _strip_z_from_coords(out["coordinates"])
    return out


def _transform_coords_to_wgs84(coords, transformer: Transformer | None):
    if transformer is None:
        return coords
    if not isinstance(coords, (list, tuple)):
        return coords
    if not coords:
        return coords
    first = coords[0]
    if isinstance(first, (int, float)) and len(coords) >= 2:
        lon, lat = transformer.transform(float(coords[0]), float(coords[1]))
        return [lon, lat]
    return [_transform_coords_to_wgs84(item, transformer) for item in coords]


def _get_shapefile_transformer(shp_path: str) -> Transformer | None:
    prj_path = os.path.splitext(shp_path)[0] + ".prj"
    if not os.path.exists(prj_path):
        return None
    try:
        with open(prj_path, "r", encoding="utf-8", errors="ignore") as f:
            prj_text = f.read().strip()
        if not prj_text:
            return None
        source_crs = CRS.from_wkt(prj_text)
        target_crs = CRS.from_epsg(4326)
        if source_crs == target_crs:
            return None
        return Transformer.from_crs(source_crs, target_crs, always_xy=True)
    except Exception:
        return None


def _collect_lon_lat_pairs(coords, out: list[tuple[float, float]]):
    if not isinstance(coords, (list, tuple)) or not coords:
        return
    first = coords[0]
    if isinstance(first, (int, float)) and len(coords) >= 2:
        out.append((float(coords[0]), float(coords[1])))
        return
    for item in coords:
        _collect_lon_lat_pairs(item, out)


def _bbox_from_features(features: list[dict]) -> list[float] | None:
    points: list[tuple[float, float]] = []
    for feature in features:
        geometry = feature.get("geometry", {})
        _collect_lon_lat_pairs(geometry.get("coordinates"), points)
    if not points:
        return None
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return [min(lons), min(lats), max(lons), max(lats)]


def _point_bbox(points: list[dict], lon_key: str = "lon", lat_key: str = "lat") -> list[float] | None:
    valid = [p for p in points if lon_key in p and lat_key in p]
    if not valid:
        return None
    lons = [float(p[lon_key]) for p in valid]
    lats = [float(p[lat_key]) for p in valid]
    return [min(lons), min(lats), max(lons), max(lats)]


def _json_safe_value(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, (str, bytes)):
        try:
            return value.isoformat()
        except Exception:
            pass
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _json_safe_obj(value):
    if isinstance(value, dict):
        return {str(k): _json_safe_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_obj(v) for v in value]
    return _json_safe_value(value)


def _feature_label_position(feature: dict) -> list[float] | None:
    geometry = feature.get("geometry", {})
    points: list[tuple[float, float]] = []
    _collect_lon_lat_pairs(geometry.get("coordinates"), points)
    if not points:
        return None
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return [(min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2]


def _feature_body_positions(feature: dict) -> list[list[float]]:
    geometry = feature.get("geometry", {}) or {}
    geom_type = str(geometry.get("type", "")).strip()
    coords = geometry.get("coordinates")
    positions: list[list[float]] = []

    def _ring_center(ring_coords) -> list[float] | None:
        pts: list[tuple[float, float]] = []
        _collect_lon_lat_pairs(ring_coords, pts)
        if not pts:
            return None
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        return [(min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2]

    if geom_type == "Polygon" and isinstance(coords, (list, tuple)) and coords:
        center = _ring_center(coords[0])
        if center:
            positions.append(center)
    elif geom_type == "MultiPolygon" and isinstance(coords, (list, tuple)):
        for polygon_coords in coords:
            if not isinstance(polygon_coords, (list, tuple)) or not polygon_coords:
                continue
            center = _ring_center(polygon_coords[0])
            if center:
                positions.append(center)

    if positions:
        return positions

    fallback = _feature_label_position(feature)
    return [fallback] if fallback else []


def _outer_rings_from_geometry(geometry: dict) -> list[list[tuple[float, float]]]:
    geom_type = str((geometry or {}).get("type", "")).strip()
    coords = (geometry or {}).get("coordinates")
    rings: list[list[tuple[float, float]]] = []

    def _norm_ring(r):
        out = []
        for pt in (r or []):
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                out.append((float(pt[0]), float(pt[1])))
        if len(out) >= 3:
            return out
        return []

    if geom_type == "Polygon" and isinstance(coords, (list, tuple)) and coords:
        ring = _norm_ring(coords[0])
        if ring:
            rings.append(ring)
    elif geom_type == "MultiPolygon" and isinstance(coords, (list, tuple)):
        for poly in coords:
            if not isinstance(poly, (list, tuple)) or not poly:
                continue
            ring = _norm_ring(poly[0])
            if ring:
                rings.append(ring)
    return rings


def _ring_bbox(ring: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not ring:
        return None
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _bboxes_overlap(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> bool:
    if not a or not b:
        return False
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _point_on_segment(px, py, ax, ay, bx, by, eps: float = 1e-12) -> bool:
    cross = (py - ay) * (bx - ax) - (px - ax) * (by - ay)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -eps:
        return False
    sq_len = (bx - ax) ** 2 + (by - ay) ** 2
    if dot - sq_len > eps:
        return False
    return True


def _point_in_ring(point: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(ring)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / ((y2 - y1) if (y2 - y1) != 0 else 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def _orientation(a, b, c) -> float:
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def _segments_intersect(a1, a2, b1, b2) -> bool:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    if abs(o1) < 1e-12 and _point_on_segment(b1[0], b1[1], a1[0], a1[1], a2[0], a2[1]):
        return True
    if abs(o2) < 1e-12 and _point_on_segment(b2[0], b2[1], a1[0], a1[1], a2[0], a2[1]):
        return True
    if abs(o3) < 1e-12 and _point_on_segment(a1[0], a1[1], b1[0], b1[1], b2[0], b2[1]):
        return True
    if abs(o4) < 1e-12 and _point_on_segment(a2[0], a2[1], b1[0], b1[1], b2[0], b2[1]):
        return True
    return False


def _rings_intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    if not _bboxes_overlap(_ring_bbox(a), _ring_bbox(b)):
        return False
    for i in range(len(a)):
        a1 = a[i]
        a2 = a[(i + 1) % len(a)]
        for j in range(len(b)):
            b1 = b[j]
            b2 = b[(j + 1) % len(b)]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _feature_intersects_region_rings(feature: dict, region_rings: list[list[tuple[float, float]]]) -> bool:
    if not region_rings:
        return True
    cand_rings = _outer_rings_from_geometry(feature.get("geometry", {}))
    if not cand_rings:
        return False
    for cr in cand_rings:
        cr_bbox = _ring_bbox(cr)
        for rr in region_rings:
            rr_bbox = _ring_bbox(rr)
            if not _bboxes_overlap(cr_bbox, rr_bbox):
                continue
            if any(_point_in_ring(pt, rr) for pt in cr):
                return True
            if any(_point_in_ring(pt, cr) for pt in rr):
                return True
            if _rings_intersect(cr, rr):
                return True
    return False


def _region_outer_rings(feature_collection: dict) -> list[list[tuple[float, float]]]:
    rings: list[list[tuple[float, float]]] = []
    for feature in feature_collection.get("features", []):
        rings.extend(_outer_rings_from_geometry(feature.get("geometry", {})))
    return rings


def _build_label_points(layer_name: str, geojson: dict) -> list[dict]:
    label_points = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        label = str(props.get("Name", props.get("name", ""))).strip()
        if not label:
            continue
        position = _feature_label_position(feature)
        if not position:
            continue
        label_points.append(
            {
                "name": label,
                "position": position,
                "size": 180 if layer_name == "Selected_Feature" else 120,
            }
        )
    return label_points


def _hover_label_html() -> str:
    return """
    <div style="
        background: rgba(255,255,255,0.92);
        color: #2f3a34;
        padding: 6px 10px;
        border-radius: 8px;
        border: 1px solid rgba(92, 105, 98, 0.28);
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 6px 16px rgba(0,0,0,0.10);
    ">
      {hover_label}
    </div>
    """


@st.cache_data(show_spinner=False)
def load_shapefile_feature_collection(shp_path: str) -> tuple[dict, pd.DataFrame, list[float] | None]:
    reader = shapefile.Reader(shp_path)
    transformer = _get_shapefile_transformer(shp_path)
    rows = []
    features = []
    for sr in reader.iterShapeRecords():
        props = _json_safe_obj(sr.record.as_dict())
        geometry = _shape_to_geojson_geometry(sr.shape)
        geometry["coordinates"] = _transform_coords_to_wgs84(geometry.get("coordinates"), transformer)
        hover_label = str(props.get("Name", props.get("name", ""))).strip()
        rows.append(props)
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {**dict(props), "hover_label": hover_label},
            }
        )
    return {"type": "FeatureCollection", "features": features}, pd.DataFrame(rows), _bbox_from_features(features)


@st.cache_data(show_spinner=False)
def load_hcfcd_project_boundaries(shp_path: str, precinct_boundary_path: str | None = None) -> tuple[list[dict], dict[str, dict], pd.DataFrame, list[float] | None]:
    geojson, attr_df, bbox = load_shapefile_feature_collection(shp_path)
    region_rings: list[list[tuple[float, float]]] = []
    if precinct_boundary_path and os.path.exists(precinct_boundary_path):
        pc1_geojson, _, _ = load_shapefile_feature_collection(precinct_boundary_path)
        region_rings = _region_outer_rings(pc1_geojson)

    point_rows = []
    outlines_by_project: dict[str, list[dict]] = {}

    for idx, feature in enumerate(geojson.get("features", [])):
        if region_rings and not _feature_intersects_region_rings(feature, region_rings):
            continue
        props = dict(feature.get("properties", {}))
        project_id = str(props.get("ProjectID", "")).strip()
        project_name = str(props.get("PROJECT_NA", "")).strip() or str(props.get("PROJECT_NMAE", "")).strip() or project_id
        lifecycle_stage = str(props.get("LATESTACTI", "")).strip() or str(props.get("PROJECT_ST", "")).strip()
        watershed = str(props.get("WATERSHED", "")).strip()
        body_positions = _feature_body_positions(feature)
        for body_index, centroid in enumerate(body_positions):
            point_rows.append(
                {
                    "project_id": project_id,
                    "project_name": project_name or project_id,
                    "lifecycle_stage": lifecycle_stage,
                    "watershed": watershed,
                    "lon": centroid[0],
                    "lat": centroid[1],
                    "source_index": idx,
                    "body_index": body_index,
                    "hover_label": project_name or project_id,
                }
            )
        outlines_by_project.setdefault(project_id, []).append(feature)

    outline_fc_by_project = {
        pid: {"type": "FeatureCollection", "features": feats}
        for pid, feats in outlines_by_project.items()
    }
    filtered_bbox = _bbox_from_features([f for fc in outline_fc_by_project.values() for f in fc.get("features", [])])
    return point_rows, outline_fc_by_project, pd.DataFrame(point_rows), filtered_bbox or bbox or _point_bbox(point_rows)


def get_spatial_shapefile_paths() -> dict[str, str]:
    shp_dir = resource_path("SHPs")
    if not os.path.isdir(shp_dir):
        return {}
    return {
        os.path.splitext(name)[0]: os.path.join(shp_dir, name)
        for name in sorted(os.listdir(shp_dir))
        if name.lower().endswith(".shp")
    }


def _merge_bboxes(bboxes: list[list[float] | None]) -> list[float] | None:
    valid = [b for b in bboxes if b and len(b) == 4]
    if not valid:
        return None
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


def _get_feature_name_column(df: pd.DataFrame) -> str | None:
    candidates = ["Name", "name", "project_name", "Project_Name", "PROJECT_NAME"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_named_database_matches(layer_name: str, feature_name: str) -> pd.DataFrame:
    feature_name_norm = str(feature_name).strip().lower()
    if not feature_name_norm:
        return pd.DataFrame()

    sources = []
    db_files = get_database_csv_files()
    token = "candidate" if layer_name == "Candidate_Projects" else "planning" if layer_name == "PlanningLevel_Projects" else None
    if token:
        for file_name, file_path in db_files:
            if token in file_name.lower():
                try:
                    sources.append((file_name, pd.read_csv(file_path)))
                except Exception:
                    continue

    matches = []
    for source_name, df_source in sources:
        if df_source.empty:
            continue
        for col in df_source.columns:
            if "name" not in str(col).strip().lower():
                continue
            series = df_source[col].astype(str).str.strip()
            matched = df_source.loc[series.str.lower() == feature_name_norm].copy()
            if not matched.empty:
                matched.insert(0, "Source", source_name)
                matched.insert(1, "Matched On", str(col))
                matches.append(matched)
                break
    if not matches:
        return pd.DataFrame()
    return pd.concat(matches, ignore_index=True)


def _render_record_details(record: pd.Series, title: str | None = None) -> None:
    if title:
        st.markdown(f"**{title}**")
    display_items = []
    for key, value in record.items():
        value_str = str(value).strip()
        if value_str in {"", "nan", "None"}:
            continue
        display_items.append((str(key), value_str))
    if not display_items:
        st.caption("No details available.")
        return
    for key, value in display_items:
        st.markdown(f"**{key}:** {value}")


def _resolve_pydeck_map_style(style_label: str) -> str:
    label = str(style_label or "").strip().lower()
    if label == "satellite imagery":
        return getattr(pdk.map_styles, "SATELLITE", pdk.map_styles.CARTO_LIGHT)
    if label == "street basemap":
        return getattr(pdk.map_styles, "ROAD", pdk.map_styles.CARTO_LIGHT)
    if label == "dark basemap":
        return getattr(pdk.map_styles, "CARTO_DARK", pdk.map_styles.CARTO_LIGHT)
    return pdk.map_styles.CARTO_LIGHT


def build_spatial_deck(layer_specs: list[dict], bbox: list[float] | None, map_style: str | None = None):
    layers = []
    layer_colors = {
        "Candidate_Projects": ([30, 138, 72, 120], [16, 99, 49, 240]),
        "PlanningLevel_Projects": ([255, 145, 26, 120], [225, 112, 0, 245]),
        "PC1_Boundary": ([169, 98, 55, 0], [124, 78, 45, 235]),
        "PC1_UnincorporatedRegion": ([122, 122, 122, 42], [122, 122, 122, 0]),
        "Selected_Feature": ([196, 120, 56, 75], [196, 120, 56, 255]),
        "HCFCD_ProjectBoundaries_Outline": ([196, 120, 56, 10], [184, 82, 32, 255]),
    }
    for spec in layer_specs:
        layer_name = spec.get("name", "")
        layer_kind = spec.get("kind", "geojson")
        if layer_kind == "scatter":
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    spec.get("data", []),
                    id=f"layer::{layer_name}",
                    get_position='[lon, lat]',
                    get_radius=spec.get("radius", 65),
                    radius_units=spec.get("radius_units", "meters"),
                    radius_min_pixels=spec.get("radius_min_pixels", 4),
                    radius_max_pixels=spec.get("radius_max_pixels", 12),
                    get_fill_color=spec.get("fill_color", [204, 95, 54, 190]),
                    get_line_color=spec.get("line_color", [124, 78, 45, 255]),
                    line_width_min_pixels=spec.get("line_width_min_pixels", 1),
                    stroked=True,
                    filled=True,
                    pickable=bool(spec.get("pickable", True)),
                    auto_highlight=False,
                )
            )
            continue

        geojson = spec.get("data")
        fill_color, line_color = layer_colors.get(layer_name, ([120, 120, 120, 28], [90, 90, 90, 160]))
        is_pickable = bool(spec.get("pickable", layer_name not in {"PC1_Boundary", "PC1_UnincorporatedRegion", "Selected_Feature", "HCFCD_ProjectBoundaries_Outline"}))
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                geojson,
                id=f"layer::{layer_name}",
                opacity=spec.get("opacity", 0.45 if layer_name in {"Candidate_Projects", "PlanningLevel_Projects"} else 0.30),
                stroked=bool(spec.get("stroked", True)),
                filled=bool(spec.get("filled", True)),
                wireframe=False,
                get_fill_color=spec.get("fill_color", fill_color),
                get_line_color=spec.get("line_color", line_color),
                line_width_min_pixels=spec.get("line_width_min_pixels", 3 if layer_name in {"Candidate_Projects", "PlanningLevel_Projects", "Selected_Feature"} else 2),
                pickable=is_pickable,
                auto_highlight=False,
            )
        )
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        center_lon = (x0 + x1) / 2
        center_lat = (y0 + y1) / 2
        lon_span = max(abs(x1 - x0), 0.01)
        lat_span = max(abs(y1 - y0), 0.01)
        span = max(lon_span, lat_span)
        zoom = 12.0 if span < 0.03 else 10.8 if span < 0.08 else 9.8 if span < 0.20 else 8.9 if span < 0.45 else 8.0
    else:
        center_lon, center_lat, zoom = -95.45, 29.82, 8.5
    return pdk.Deck(
        map_style=map_style or pdk.map_styles.CARTO_LIGHT,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0),
        layers=layers,
        tooltip={"html": _hover_label_html()},
    )


def _spatial_layer_colors(layer_name: str) -> tuple[list[int], list[int]]:
    color_map = {
        "Candidate_Projects": ([30, 138, 72, 120], [16, 99, 49, 240]),
        "PlanningLevel_Projects": ([255, 145, 26, 120], [225, 112, 0, 245]),
        "PC1_Boundary": ([169, 98, 55, 0], [124, 78, 45, 235]),
        "PC1_UnincorporatedRegion": ([122, 122, 122, 42], [122, 122, 122, 0]),
        "Selected_Feature": ([196, 120, 56, 75], [196, 120, 56, 255]),
        "HCFCD_ProjectBoundaries_Outline": ([196, 120, 56, 10], [184, 82, 32, 255]),
    }
    return color_map.get(layer_name, ([120, 120, 120, 28], [90, 90, 90, 160]))


def _zoom_from_bbox(bbox: list[float] | None) -> tuple[float, float, float]:
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        center_lon = (x0 + x1) / 2
        center_lat = (y0 + y1) / 2
        lon_span = max(abs(x1 - x0), 0.01)
        lat_span = max(abs(y1 - y0), 0.01)
        span = max(lon_span, lat_span)
        zoom = 12.0 if span < 0.03 else 10.8 if span < 0.08 else 9.8 if span < 0.20 else 8.9 if span < 0.45 else 8.0
        return center_lat, center_lon, zoom
    return 29.82, -95.45, 8.5


def load_ahp_matrix_pairs(csv_source, selected: list[str], label_map: dict[str, str]):
    m = pd.read_csv(csv_source, index_col=0)
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
        return None, mapping_df, 0, 0, "Some selected AHP parameters could not be matched in uploaded CSV. Update CSV headers or selected parameters and try again."

    m_sel = pd.DataFrame(index=selected_labels, columns=selected_labels, dtype=object)
    for a_lab in selected_labels:
        for b_lab in selected_labels:
            m_sel.loc[a_lab, b_lab] = m.loc[resolved_rows[a_lab], resolved_cols[b_lab]]

    imported_pairs = []
    imported_count = 0
    defaulted_count = 0
    saaty_options = [
        ("1/9 (Extreme)", 1/9),
        ("1/7 (Very strong)", 1/7),
        ("1/5 (Strong)", 1/5),
        ("1/4 (Between moderate and strong)", 1/4),
        ("1/3 (Moderate)", 1/3),
        ("1/2 (Between equal and moderate)", 1/2),
        ("1 (Equal)", 1.0),
        ("2 (Between equal and moderate)", 2.0),
        ("3 (Moderate)", 3.0),
        ("4 (Between moderate and strong)", 4.0),
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

    return imported_pairs, mapping_df, imported_count, defaulted_count, None


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


def _ensure_string_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = pd.Series([""] * len(df), index=df.index, dtype="string")
    else:
        df[col] = df[col].astype("string").fillna("")


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

        _ensure_string_column(df, "svi_class")
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
        _ensure_string_column(df, "people_efficiency_class")
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
        _ensure_string_column(df, "structures_efficiency_class")
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
        _ensure_string_column(df, "people_efficiency_class")
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
        _ensure_string_column(df, "structures_efficiency_class")
        old_vals = df["structures_efficiency_class"].astype(str)
        new_vals = struct_class.astype(str)
        df["structures_efficiency_class"] = new_vals
        changes["structures_efficiency_class"] = int((old_vals != new_vals).sum())

    return df, changes


def auto_prepare_dataset(df_in: pd.DataFrame, overwrite: bool = True) -> tuple[pd.DataFrame, dict]:
    """Run the common one-click preparation steps used before scoring/ranking."""
    df = preprocess_uploaded_df(df_in)
    changes = {
        "svi_class": 0,
        "people_efficiency_class": 0,
        "structures_efficiency_class": 0,
        "excess_rainfall_class": 0,
        "excess_rainfall_source": "",
    }

    rain_source_col = find_column_by_aliases(
        df,
        [
            "excess_rainfall",
            "Excess Rainfall",
            "excess rainfall",
            "Exc_Rain",
            "EXC_RAIN",
            "maximum flood depth",
            "Maximum Flood Depth (ft)",
            "Maximum Flooding Depth",
        ],
    )
    if rain_source_col is None:
        rain_source_col = find_column_by_contains(df, "excess rainfall")

    if rain_source_col is not None:
        _, rain_class = classify_excess_rainfall_series(df[rain_source_col])
        if "excess_rainfall_class" not in df.columns:
            df["excess_rainfall_class"] = ""
        if overwrite:
            target_mask = pd.Series(True, index=df.index)
        else:
            target_mask = df["excess_rainfall_class"].apply(_is_missing_series_value)
        fill_mask = target_mask & rain_class.ne("")
        df.loc[fill_mask, "excess_rainfall_class"] = rain_class[fill_mask]
        changes["excess_rainfall_class"] = int(fill_mask.sum())
        changes["excess_rainfall_source"] = rain_source_col

    df, derived_changes = auto_fill_derived_classes(df, overwrite=overwrite)
    for k, v in derived_changes.items():
        changes[k] = changes.get(k, 0) + int(v)

    df, eff_changes = recalculate_efficiency_classes(df)
    for k, v in eff_changes.items():
        changes[k] = int(v)

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
if "direct_weights_editor_version" not in st.session_state:
    st.session_state["direct_weights_editor_version"] = 0


def get_reference_hcfcd_weights_pct() -> dict[str, float]:
    return {k: float(v) * 100.0 for k, v in config["weights"].items()}


def reset_direct_weights_to_reference_hcfcd() -> None:
    custom_zero_weights = {
        item.get("key"): 0.0
        for item in st.session_state.get("custom_criteria", [])
        if item.get("key")
    }
    st.session_state["weights_pct"] = {
        **get_reference_hcfcd_weights_pct(),
        **custom_zero_weights,
    }
    st.session_state["direct_weights_editor_version"] = st.session_state.get("direct_weights_editor_version", 0) + 1
if "app_started" not in st.session_state:
    st.session_state["app_started"] = False

# Keep app-started state stable across URL-based navigation reruns.
qp_started = st.query_params.get("started")
if isinstance(qp_started, list):
    qp_started = qp_started[0] if qp_started else None
qp_tab_boot = st.query_params.get("tab")
if isinstance(qp_tab_boot, list):
    qp_tab_boot = qp_tab_boot[0] if qp_tab_boot else None
if (
    ((isinstance(qp_started, str) and qp_started.strip() in {"1", "true", "yes"}) or isinstance(qp_tab_boot, str))
    and bool(st.session_state.get("auth_user"))
):
    st.session_state["app_started"] = True

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
    editor_version = st.session_state.get(f"{context_key}_weights_editor_version", st.session_state.get("direct_weights_editor_version", 0))
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
                key=f"{context_key}_weight_{editor_version}_{key}",
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
_collapse_sidebar_once()

if not st.session_state.get("app_started", False):
    landing_bg_css = ""
    landing_bg_path = get_landing_background_path()
    if landing_bg_path and os.path.exists(landing_bg_path):
        ext = os.path.splitext(landing_bg_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(landing_bg_path, "rb") as f:
            landing_bg_b64 = base64.b64encode(f.read()).decode("utf-8")
        landing_bg_css = f"""
        .landing-wrap::before {{
            content: "";
            position: fixed;
            inset: 0;
            background-image: url("data:{mime};base64,{landing_bg_b64}");
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
            background-color: #edf2eb;
            opacity: 0.28;
            filter: saturate(0.92) contrast(0.94);
            z-index: 0;
        }}
        .landing-wrap::after {{
            content: "";
            position: fixed;
            inset: 0;
            background: linear-gradient(180deg, rgba(248, 250, 247, 0.58) 0%, rgba(238, 243, 236, 0.68) 100%);
            z-index: 0;
        }}
        .landing-wrap > * {{
            position: relative;
            z-index: 1;
        }}
        """
    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            overflow-x: hidden !important;
            height: 100% !important;
        }}
        div.block-container {{
            padding-top: 0.75rem;
            padding-bottom: 0rem;
            margin-top: 0 !important;
        }}
        .landing-wrap {{
            min-height: auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            gap: 0;
            padding: 28.5vh 24px 0 24px;
        }}
        .landing-card {{
            position: relative;
            width: min(760px, 92%);
            margin: 0 auto;
            padding: 24px 24px;
            border: 1px solid rgba(215, 221, 213, 0.9);
            border-radius: 24px;
            background: rgba(248, 250, 247, 0.84);
            backdrop-filter: blur(6px);
            text-align: center;
            box-shadow: 0 18px 40px rgba(57, 74, 60, 0.12);
        }}
        .st-key-btn_access_tool button[kind="primary"] {{
            background: #2d5f53 !important;
            border: 1px solid #2d5f53 !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            min-height: 2.6rem !important;
            border-radius: 0.65rem !important;
            box-shadow: 0 6px 14px rgba(45, 95, 83, 0.22) !important;
        }}
        .st-key-btn_access_tool button[kind="primary"] p,
        .st-key-btn_access_tool button[kind="primary"] span {{
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            line-height: 1.1 !important;
            font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif !important;
            letter-spacing: 0.01em !important;
        }}
        .st-key-btn_access_tool button[kind="primary"]:hover {{
            background: #244c43 !important;
            border-color: #244c43 !important;
            color: #ffffff !important;
        }}
        .landing-title {{
            font-size: 2.2rem;
            font-weight: 700;
            line-height: 1.05;
            letter-spacing: -0.02em;
            color: #243127;
            margin-bottom: 8px;
        }}
        .landing-subtitle {{
            font-size: 1.1rem;
            font-weight: 500;
            color: #5b6a5d;
            margin-bottom: 12px;
        }}
        .landing-copy {{
            max-width: 580px;
            margin: 0 auto 4px auto;
            font-size: 0.96rem;
            line-height: 1.5;
            color: #425045;
        }}
        {landing_bg_css}
        </style>
        <div class="landing-wrap">
          <div class="landing-card">
            <div class="landing-title">Prioritization Tool</div>
            <div class="landing-subtitle">Built by InfraTECH</div>
            <div class="landing-copy">
              A flexible project prioritization platform for data intake, scoring, weighting,
              analysis, ranking, and framework customization under the HCFCD approach or
              user-defined criteria.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True)
    auth_c1, auth_c2, auth_c3 = st.columns([3.2, 3.6, 3.2])
    with auth_c2:
        username = st.text_input("Username", key="landing_username", placeholder="Enter username").strip().lower()
        passcode = st.text_input("Passcode", key="landing_passcode", placeholder="Enter passcode", type="password").strip()
        person_name = st.text_input("Name of Person Accessing", key="landing_person_name", placeholder="Enter full name")
        if st.button("Access Tool", key="btn_access_tool", type="primary", use_container_width=True):
            if username not in AUTH_USERS:
                st.error("Invalid username.")
            elif passcode != str(AUTH_USERS[username]["passcode"]):
                st.error("Invalid passcode.")
            else:
                _start_access_session(username, person_name or "")
                st.session_state["app_started"] = True
                st.query_params["started"] = "1"
                st.rerun()
    st.stop()

_update_access_session_runtime()

col_left, col_right = st.columns([5, 1])
with col_left:
    st.title("Project Prioritization Tool")
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
        background-color: #4f77a8;
        border-color: #4f77a8;
        color: white;
        font-weight: 700;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #41658f;
        border-color: #41658f;
        color: white;
    }
    /* Main navigation (st.segmented_control) */
    div[data-testid="stSegmentedControl"] {
        margin-bottom: 0.6rem;
    }
    div[data-testid="stSegmentedControl"] [role="radiogroup"],
    div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
        background: #0b1220 !important;
        border: 1px solid #223045 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        padding: 0 !important;
    }
    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stSegmentedControl"] [role="radio"] {
        background: #0b0f17 !important;
        color: #ffffff !important;
        border: 0 !important;
        border-right: 1px solid #2a3344 !important;
        border-radius: 0 !important;
        min-height: 58px !important;
        padding: 0.6rem 0.8rem !important;
    }
    div[data-testid="stSegmentedControl"] button[kind="segmented_control"],
    div[data-testid="stSegmentedControl"] button[data-testid="stBaseButton-segmented_control"] {
        background: #0b0f17 !important;
        color: #ffffff !important;
        border-color: #2a3344 !important;
    }
    div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"],
    div[data-testid="stSegmentedControl"] button[data-testid="stBaseButton-segmented_controlActive"] {
        background: #2563eb !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
    }
    div[data-testid="stSegmentedControl"] button:last-child,
    div[data-testid="stSegmentedControl"] [role="radio"]:last-child {
        border-right: 0 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] button[aria-selected="true"],
    div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
        background: #2563eb !important;
        color: #ffffff !important;
    }
    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stSegmentedControl"] [role="radio"],
    div[data-testid="stSegmentedControl"] button p,
    div[data-testid="stSegmentedControl"] [role="radio"] p,
    div[data-testid="stSegmentedControl"] button span,
    div[data-testid="stSegmentedControl"] [role="radio"] span {
        font-size: 1.22rem !important;
        font-weight: 800 !important;
        color: inherit !important;
        letter-spacing: 0.01em !important;
    }
    div[data-testid="stSegmentedControl"] button[kind="segmented_control"] p,
    div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] p,
    div[data-testid="stSegmentedControl"] button[data-testid="stBaseButton-segmented_control"] p,
    div[data-testid="stSegmentedControl"] button[data-testid="stBaseButton-segmented_controlActive"] p {
        font-size: 1.22rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        line-height: 1.05 !important;
    }
    .spatial-layer-panel-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #26384f;
        margin-bottom: 8px;
    }
    div[data-testid="stCheckbox"] label p {
        font-size: 1.02rem;
        font-weight: 600;
        color: #27374a;
    }
    .spatial-find-box {
        border: 1px solid #c9d8e8;
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 12px;
        margin-bottom: 8px;
        background: #eef5fc;
    }
    .spatial-find-box.study {
        border-color: #d6cce5;
        background: #f5f0fa;
    }
    .spatial-find-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #2a3d57;
        margin-bottom: 2px;
    }
    .spatial-find-caption {
        font-size: 0.92rem;
        color: #536b89;
        margin-bottom: 6px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.spatial-layer-panel-title) {
        background: #f6f9fc;
    }
    .pydeck-legend-overlay {
        position: relative;
        z-index: 30;
        margin-top: -220px;
        margin-left: 12px;
        margin-right: auto;
        width: 320px;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 8px 10px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    }
    .pydeck-legend-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 4px;
    }
    .pydeck-layer-controls-marker {
        font-size: 0.92rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 6px;
    }
    .st-key-spatial_layer_controls div[data-testid="stCheckbox"] {
        margin-bottom: 0rem;
    }
    .st-key-spatial_layer_controls div[data-testid="stCheckbox"] label {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .st-key-spatial_layer_controls div[data-testid="stRadio"] {
        margin-bottom: 0.1rem;
    }
    .st-key-spatial_layer_controls div[data-testid="stRadio"] label {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .st-key-spatial_layer_controls p {
        margin-bottom: 0.12rem;
    }
    .st-key-spatial_layer_controls [data-testid="column"] > div {
        row-gap: 0rem;
    }
    .st-key-spatial_layer_controls [data-testid="stMarkdownContainer"] p {
        margin-top: 0rem;
        margin-bottom: 0.16rem;
    }
    .st-key-ahp_importable_matrix {
        background: #edf6ff;
        border: 2px solid #3f7fb5;
        border-radius: 10px;
        padding: 12px 14px 14px 14px;
        margin: 10px 0 12px 0;
        box-shadow: 0 2px 8px rgba(63, 127, 181, 0.12);
    }
    .st-key-ahp_importable_matrix label p {
        color: #1d4f78;
        font-size: 1.03rem;
        font-weight: 800;
    }
    .st-key-ahp_importable_matrix [data-baseweb="select"] > div {
        background: #ffffff;
        border-color: #3f7fb5;
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
    if st.button("Return to Landing", key="btn_return_to_landing", use_container_width=True):
        st.session_state["app_started"] = False
        st.session_state["_sidebar_collapsed_once"] = False
        st.query_params.clear()
        st.rerun()

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
        st.session_state.pop("project_data_editor", None)

if "df_work" not in st.session_state:
    default_db_files = get_database_csv_files()
    default_db_name = _default_database_name(default_db_files)
    default_db_path = dict(default_db_files).get(default_db_name) if default_db_name else None
    if default_db_path:
        st.session_state["df_work"] = preprocess_uploaded_df(pd.read_csv(default_db_path))
        st.session_state["uploaded_file_name"] = default_db_name
        st.info(f"Loaded default database: {default_db_name}")
    else:
        st.session_state["df_work"] = preprocess_uploaded_df(pd.read_csv(resource_path("input_template.csv")))
        st.session_state["uploaded_file_name"] = "input_template.csv"
        st.info("Using included template. You can upload a CSV from the main data section.")

df = st.session_state["df_work"]

TAB_OPTIONS = [
    "Spatial View",
    "Prioritization Database",
    "Data Tools",
    "Parameter Analysis",
    "Direct Weights",
    "Pairwise Comparision (AHP)",
    "Ranking",
]
if "active_main_tab" not in st.session_state:
    st.session_state["active_main_tab"] = TAB_OPTIONS[0]
else:
    legacy_tab_map = {
        "AHP Weights": "Pairwise Comparision (AHP)",
        "Spatial View (Pydeck)": "Spatial View",
        "Spatial View (Folium)": "Spatial View",
    }
    current_tab = st.session_state.get("active_main_tab")
    if current_tab in legacy_tab_map:
        st.session_state["active_main_tab"] = legacy_tab_map[current_tab]
    elif current_tab not in TAB_OPTIONS:
        st.session_state["active_main_tab"] = TAB_OPTIONS[0]

active_main_tab = st.segmented_control(
    "",
    options=TAB_OPTIONS,
    default=st.session_state.get("active_main_tab", TAB_OPTIONS[0]),
    key="active_main_tab",
    width="stretch",
)
active_main_tab = active_main_tab or st.session_state.get("active_main_tab", TAB_OPTIONS[0])


def render_data_tab():
    st.subheader("Data Source")
    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        uploaded_main = st.file_uploader("Upload input CSV", type=["csv"], key="main_upload")
        importable_main = get_database_csv_files()
        if importable_main:
            main_names = [n for n, _ in importable_main]
            current_name = st.session_state.get("uploaded_file_name", "")
            default_name = current_name if current_name in main_names else _default_database_name(importable_main)
            selected_main_db = st.selectbox(
                "Or load from Importable Database",
                options=main_names,
                index=main_names.index(default_name) if default_name in main_names else 0,
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
                    st.session_state.pop("project_data_editor", None)
                    st.rerun()
    with col_u2:
        if st.button("Load template dataset", key="btn_load_template_main"):
            st.session_state["df_work"] = preprocess_uploaded_df(pd.read_csv(resource_path("input_template.csv")))
            st.session_state["uploaded_file_name"] = "input_template.csv"
            st.session_state["loaded_main_upload_id"] = None
            st.session_state["svi_source_force_reset"] = True
            st.session_state.pop("project_data_editor", None)
        if st.button("Clear current dataset", key="btn_clear_dataset"):
            st.session_state["df_work"] = st.session_state["df_work"].head(0)
            st.session_state["uploaded_file_name"] = "cleared"
            st.session_state["loaded_main_upload_id"] = None
            st.session_state["svi_source_force_reset"] = True
            st.session_state.pop("project_data_editor", None)
    if uploaded_main is not None:
        uploaded_df, upload_id = read_uploaded_csv_with_id(uploaded_main)
        if st.session_state.get("loaded_main_upload_id") != upload_id:
            st.session_state["df_work"] = preprocess_uploaded_df(uploaded_df)
            st.session_state["uploaded_file_name"] = uploaded_main.name
            st.session_state["loaded_main_upload_id"] = upload_id
            st.session_state["svi_source_force_reset"] = True
            st.session_state.pop("project_data_editor", None)

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
    c_reset, _ = st.columns([1.7, 4])
    with c_reset:
        if st.button("Reset to Reference HCFCD Weights", key="btn_reset_direct_hcfcd_weights", use_container_width=True):
            reset_direct_weights_to_reference_hcfcd()
            st.success("Direct weights reset to Reference HCFCD (2022) weights.")
            st.rerun()
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

    if not df_local.empty:
        st.markdown("#### One-Click Dataset Preparation")
        st.caption(
            "Runs SVI reclassification, excess-rainfall classification when a source column is found, "
            "and recalculates people/structure efficiency classes."
        )
        if st.button("Prepare Dataset for Scoring", key="btn_prepare_dataset_for_scoring"):
            df_prepared, prep_changes = auto_prepare_dataset(df_local, overwrite=True)
            st.session_state["df_work"] = df_prepared
            st.session_state.pop("project_data_editor", None)
            rain_source = prep_changes.get("excess_rainfall_source") or "not found"
            st.success(
                "Dataset prepared. "
                f"SVI classes updated: {prep_changes.get('svi_class', 0)}; "
                f"people efficiency updates: {prep_changes.get('people_efficiency_class', 0)}; "
                f"structure efficiency updates: {prep_changes.get('structures_efficiency_class', 0)}; "
                f"excess rainfall classes updated: {prep_changes.get('excess_rainfall_class', 0)} "
                f"(source: {rain_source})."
            )
            st.rerun()

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


def render_spatial_tab_pydeck():
    st.subheader("Spatial View")

    shp_paths = get_spatial_shapefile_paths()
    if not shp_paths:
        st.info("No shapefiles found in the `Support Files/SHPs` folder.")
        return

    base_layers = [name for name in ["PC1_Boundary", "PC1_UnincorporatedRegion"] if name in shp_paths]
    core_layers = [name for name in ["Candidate_Projects", "PlanningLevel_Projects", "HCFCD_ProjectBoundaries"] if name in shp_paths]
    other_layers = [name for name in shp_paths if name not in set(base_layers + core_layers)]

    if "spatial_show_candidate" not in st.session_state:
        st.session_state["spatial_show_candidate"] = "Candidate_Projects" in core_layers
    if "spatial_show_planning" not in st.session_state:
        st.session_state["spatial_show_planning"] = "PlanningLevel_Projects" in core_layers
    if "spatial_show_hcfcd" not in st.session_state:
        st.session_state["spatial_show_hcfcd"] = False
    if "spatial_show_pc1_boundary" not in st.session_state:
        st.session_state["spatial_show_pc1_boundary"] = "PC1_Boundary" in base_layers
    if "spatial_show_pc1_uninc" not in st.session_state:
        st.session_state["spatial_show_pc1_uninc"] = "PC1_UnincorporatedRegion" in base_layers
    if "spatial_selected_other_layers" not in st.session_state:
        st.session_state["spatial_selected_other_layers"] = []
    if "spatial_pydeck_basemap" not in st.session_state:
        st.session_state["spatial_pydeck_basemap"] = "Original Basemap"
    if "spatial_map_version" not in st.session_state:
        st.session_state["spatial_map_version"] = 0

    show_candidate = bool(st.session_state.get("spatial_show_candidate", False))
    show_planning = bool(st.session_state.get("spatial_show_planning", False))
    show_hcfcd = bool(st.session_state.get("spatial_show_hcfcd", False))
    show_pc1_boundary = bool(st.session_state.get("spatial_show_pc1_boundary", False))
    show_pc1_uninc = bool(st.session_state.get("spatial_show_pc1_uninc", False))
    selected_other_layers = []

    selected_base_layers = []
    if show_pc1_boundary and "PC1_Boundary" in base_layers:
        selected_base_layers.append("PC1_Boundary")
    if show_pc1_uninc and "PC1_UnincorporatedRegion" in base_layers:
        selected_base_layers.append("PC1_UnincorporatedRegion")

    selected_layers = []
    if show_candidate and "Candidate_Projects" in core_layers:
        selected_layers.append("Candidate_Projects")
    if show_planning and "PlanningLevel_Projects" in core_layers:
        selected_layers.append("PlanningLevel_Projects")
    if show_hcfcd and "HCFCD_ProjectBoundaries" in core_layers:
        selected_layers.append("HCFCD_ProjectBoundaries")

    legend_items = []
    if "PC1_Boundary" in selected_base_layers:
        legend_items.append(("Precinct 1 Boundary", "#7C4E2D", "outline"))
    if "PC1_UnincorporatedRegion" in selected_base_layers:
        legend_items.append(("Precinct 1 Unincorporated Region", "#7A7A7A", "fill"))
    if "Candidate_Projects" in selected_layers:
        legend_items.append(("Candidate Projects", "#1E8A48", "fill"))
    if "PlanningLevel_Projects" in selected_layers:
        legend_items.append(("Planning Level Study Areas", "#FF911A", "fill"))
    if "HCFCD_ProjectBoundaries" in selected_layers:
        legend_items.append(("HCFCD Project Boundaries (Points)", "#396CB1", "point"))

    legend_html_rows = []
    for name, color, style_type in legend_items:
        if style_type == "outline":
            swatch = (
                f"<span style='display:inline-block;width:18px;height:12px;border:2px solid {color};"
                "background:transparent;border-radius:2px;'></span>"
            )
        elif style_type == "point":
            swatch = (
                f"<span style='display:inline-block;width:12px;height:12px;background:{color};"
                "border:1px solid #1f2937;border-radius:50%;'></span>"
            )
        else:
            swatch = (
                f"<span style='display:inline-block;width:18px;height:12px;background:{color};"
                "border:1px solid #4b5563;border-radius:2px;opacity:0.85;'></span>"
            )
        legend_html_rows.append(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>{swatch}<span>{name}</span></div>"
        )

    pydeck_legend_html = ""
    if legend_html_rows:
        pydeck_legend_html = (
            "<div class='pydeck-legend-overlay'><div class='pydeck-legend-title'>Map Legend</div>"
            + "".join(legend_html_rows)
            + "</div>"
        )

    active_layers = selected_base_layers + selected_layers
    if not active_layers:
        st.info("No spatial layers are available to display.")
        return

    layer_geojsons = []
    map_layer_specs = []
    layer_tables: dict[str, pd.DataFrame] = {}
    layer_geojson_lookup: dict[str, dict] = {}
    layer_bboxes = []
    load_errors = []
    for layer_name in active_layers:
        try:
            if layer_name == "HCFCD_ProjectBoundaries":
                point_rows, outline_fc_by_project, attr_df, bbox = load_hcfcd_project_boundaries(
                    shp_paths[layer_name],
                    shp_paths.get("PC1_Boundary"),
                )
                layer_tables[layer_name] = attr_df
                layer_geojson_lookup[f"{layer_name}__outlines"] = outline_fc_by_project
                map_layer_specs.append(
                    {
                        "name": layer_name,
                        "kind": "scatter",
                        "data": point_rows,
                        "radius": 3,
                        "radius_units": "pixels",
                        "radius_min_pixels": 2,
                        "radius_max_pixels": 5,
                        "fill_color": [57, 108, 177, 200],
                        "line_color": [34, 78, 138, 255],
                        "pickable": True,
                    }
                )
                layer_bboxes.append(bbox or _point_bbox(point_rows))
            else:
                geojson, attr_df, bbox = load_shapefile_feature_collection(shp_paths[layer_name])
                layer_geojsons.append((layer_name, geojson))
                map_layer_specs.append(
                    {
                        "name": layer_name,
                        "kind": "geojson",
                        "data": geojson,
                        "pickable": layer_name not in {"PC1_Boundary", "PC1_UnincorporatedRegion"},
                        "filled": False if layer_name == "PC1_Boundary" else True,
                        "stroked": True,
                    }
                )
                layer_geojson_lookup[layer_name] = geojson
                layer_tables[layer_name] = attr_df
                layer_bboxes.append(bbox)
        except Exception as ex:
            load_errors.append(f"{layer_name}: {ex}")

    if load_errors:
        st.warning("Some layers could not be loaded.")
        for msg in load_errors:
            st.write(f"- {msg}")

    combined_bbox = _merge_bboxes(layer_bboxes)
    if not layer_geojsons:
        st.info("No spatial layers were successfully loaded.")
        return

    selected_layers_signature = "|".join(selected_layers)
    stored_selected_layer = st.session_state.get("spatial_selected_layer_name", "")
    stored_selected_feature_name = st.session_state.get("spatial_selected_feature_name", "")
    stored_selected_feature_geojson = st.session_state.get("spatial_selected_feature_geojson")
    stored_hcfcd_project_id = st.session_state.get("spatial_selected_hcfcd_project_id", "")
    stored_forced_bbox = st.session_state.get("spatial_forced_bbox")
    stored_view_bbox = st.session_state.get("spatial_view_bbox")
    pc1_bbox = _bbox_from_features(layer_geojson_lookup["PC1_Boundary"].get("features", [])) if "PC1_Boundary" in layer_geojson_lookup else None
    base_bbox = pc1_bbox or combined_bbox
    if not combined_bbox:
        combined_bbox = base_bbox
    previous_layers_signature = st.session_state.get("spatial_layers_signature", "")
    if previous_layers_signature != selected_layers_signature:
        st.session_state["spatial_layers_signature"] = selected_layers_signature
        st.session_state["spatial_view_bbox"] = base_bbox
        st.session_state.pop("spatial_forced_bbox", None)
        stored_view_bbox = base_bbox
        stored_forced_bbox = None

    selected_outline_bbox = None
    if stored_hcfcd_project_id and "HCFCD_ProjectBoundaries" in selected_layers:
        hcfcd_outlines = layer_geojson_lookup.get("HCFCD_ProjectBoundaries__outlines", {})
        selected_outline_fc = hcfcd_outlines.get(stored_hcfcd_project_id)
        selected_outline_bbox = _bbox_from_features(selected_outline_fc.get("features", [])) if selected_outline_fc else None

    map_layers = []
    if isinstance(stored_forced_bbox, list) and len(stored_forced_bbox) == 4:
        effective_bbox = stored_forced_bbox
    elif isinstance(stored_view_bbox, list) and len(stored_view_bbox) == 4:
        effective_bbox = stored_view_bbox
    else:
        effective_bbox = base_bbox
        st.session_state["spatial_view_bbox"] = base_bbox
    if not (isinstance(effective_bbox, list) and len(effective_bbox) == 4):
        effective_bbox = base_bbox
    for spec in map_layer_specs:
        if spec.get("name") == "HCFCD_ProjectBoundaries" and stored_hcfcd_project_id:
            continue
        map_layers.append(spec)
    map_layers.sort(key=lambda spec: 1 if spec.get("name") == "HCFCD_ProjectBoundaries" else 0)
    if "HCFCD_ProjectBoundaries" in selected_layers and stored_hcfcd_project_id:
        hcfcd_outlines = layer_geojson_lookup.get("HCFCD_ProjectBoundaries__outlines", {})
        if stored_hcfcd_project_id in hcfcd_outlines:
            selected_outline_fc = _json_safe_obj(hcfcd_outlines[stored_hcfcd_project_id])
            map_layers.append(
                {
                    "name": "HCFCD_ProjectBoundaries_Outline",
                    "kind": "geojson",
                    "data": selected_outline_fc,
                    "pickable": False,
                    "filled": True,
                    "opacity": 1.0,
                    "fill_color": [196, 120, 56, 36],
                    "line_color": [184, 82, 32, 255],
                    "line_width_min_pixels": 8,
                }
            )
    elif stored_selected_feature_geojson and stored_selected_layer in selected_layers:
        outline_name = "HCFCD_ProjectBoundaries_Outline" if stored_selected_layer == "HCFCD_ProjectBoundaries" else "Selected_Feature"
        map_layers.append(
            {
                "name": outline_name,
                "kind": "geojson",
                "data": _json_safe_obj(stored_selected_feature_geojson),
                "pickable": False,
                "filled": True,
                "opacity": 0.55 if outline_name == "Selected_Feature" else 1.0,
                "fill_color": [196, 120, 56, 36] if outline_name == "HCFCD_ProjectBoundaries_Outline" else [196, 120, 56, 75],
                "line_color": [184, 82, 32, 255] if outline_name == "HCFCD_ProjectBoundaries_Outline" else [196, 120, 56, 255],
                "line_width_min_pixels": 8 if outline_name == "HCFCD_ProjectBoundaries_Outline" else 4,
            }
        )
    with st.container(border=True, key="spatial_layer_controls"):
        st.markdown("**Layer Visibility**")
        col_base, col_layers = st.columns(2)
        with col_base:
            st.markdown("**Base Maps**")
            st.radio(
                "Base Map",
                options=["Original Basemap", "Street Basemap", "Dark Basemap", "Satellite Imagery"],
                key="spatial_pydeck_basemap",
                horizontal=False,
                label_visibility="collapsed",
            )
            st.checkbox(
                "Precinct 1 Boundary",
                key="spatial_show_pc1_boundary",
                disabled="PC1_Boundary" not in base_layers,
            )
            st.checkbox(
                "Precinct 1 Unincorporated Region",
                key="spatial_show_pc1_uninc",
                disabled="PC1_UnincorporatedRegion" not in base_layers,
            )
        with col_layers:
            st.markdown("**Other Layers**")
            st.checkbox(
                "Candidate Projects",
                key="spatial_show_candidate",
                disabled="Candidate_Projects" not in core_layers,
            )
            st.checkbox(
                "Planning Level Study Areas",
                key="spatial_show_planning",
                disabled="PlanningLevel_Projects" not in core_layers,
            )
            st.checkbox(
                "HCFCD Project Boundaries",
                key="spatial_show_hcfcd",
                disabled="HCFCD_ProjectBoundaries" not in core_layers,
            )

    map_container = st.container(border=True)
    with map_container:
        selected_basemap = st.session_state.get("spatial_pydeck_basemap", "Original Basemap")
        chart_state = st.pydeck_chart(
            build_spatial_deck(map_layers, effective_bbox, map_style=_resolve_pydeck_map_style(selected_basemap)),
            width="stretch",
            height=700,
            selection_mode="single-object",
            on_select="rerun",
            key=f"spatial_map_{st.session_state.get('spatial_map_version', 0)}",
        )
        if pydeck_legend_html:
            st.markdown(pydeck_legend_html, unsafe_allow_html=True)

    if stored_hcfcd_project_id and "HCFCD_ProjectBoundaries" in selected_layers:
        _, c1, c2, _ = st.columns([2.4, 2.2, 3.0, 2.4])
        with c1:
            if st.button("Clear selected HCFCD project", key="btn_clear_hcfcd_project", type="primary"):
                st.session_state.pop("spatial_selected_hcfcd_project_id", None)
                st.session_state.pop("spatial_selected_layer_name", None)
                st.session_state.pop("spatial_selected_feature_name", None)
                st.session_state.pop("spatial_selected_feature_geojson", None)
                st.session_state.pop("spatial_forced_bbox", None)
                st.session_state["spatial_skip_selection_once"] = True
                st.session_state["spatial_map_version"] = int(st.session_state.get("spatial_map_version", 0)) + 1
                stored_hcfcd_project_id = ""
                stored_selected_layer = ""
                stored_selected_feature_name = ""
                stored_selected_feature_geojson = None
                stored_forced_bbox = None
                st.rerun()
        with c2:
            if st.button("Zoom to extent of selected HCFCD Project", key="btn_zoom_hcfcd_project", type="primary", disabled=not bool(selected_outline_bbox)):
                if selected_outline_bbox:
                    st.session_state["spatial_forced_bbox"] = selected_outline_bbox
                    st.session_state["spatial_view_bbox"] = selected_outline_bbox
                    stored_forced_bbox = selected_outline_bbox
                    stored_view_bbox = selected_outline_bbox

    selected_layer_name = ""
    selected_feature_name = ""
    selected_feature_row = pd.DataFrame()
    selected_feature_geojson = None

    try:
        if st.session_state.pop("spatial_skip_selection_once", False):
            selection = {}
        else:
            selection = chart_state.selection if chart_state else {}
        indices_by_layer = selection.get("indices", {}) if isinstance(selection, dict) else {}
        for layer_id, indices in indices_by_layer.items():
            if not indices:
                continue
            if not str(layer_id).startswith("layer::"):
                continue
            selected_layer_name = str(layer_id).split("layer::", 1)[1]
            selected_index = int(indices[0])
            if selected_layer_name == "HCFCD_ProjectBoundaries" and selected_layer_name in layer_tables:
                df_sel = layer_tables[selected_layer_name]
                if 0 <= selected_index < len(df_sel):
                    selected_feature_row = df_sel.iloc[[selected_index]].copy()
                    selected_feature_name = str(selected_feature_row.iloc[0].get("project_id", "")).strip()
                    st.session_state["spatial_selected_hcfcd_project_id"] = selected_feature_name
            elif selected_layer_name in layer_geojson_lookup:
                features = layer_geojson_lookup[selected_layer_name].get("features", [])
                if 0 <= selected_index < len(features):
                    selected_feature_geojson = _json_safe_obj({"type": "FeatureCollection", "features": [features[selected_index]]})
            if selected_layer_name in layer_tables and selected_feature_row.empty:
                df_sel = layer_tables[selected_layer_name]
                if 0 <= selected_index < len(df_sel):
                    selected_feature_row = df_sel.iloc[[selected_index]].copy()
                    name_col = _get_feature_name_column(df_sel)
                    if name_col and name_col in selected_feature_row.columns:
                        selected_feature_name = str(selected_feature_row.iloc[0][name_col]).strip()
            break
    except Exception:
        selected_layer_name = ""

    if selected_feature_geojson:
        if (
            st.session_state.get("spatial_selected_layer_name") != selected_layer_name
            or st.session_state.get("spatial_selected_feature_name") != selected_feature_name
        ):
            st.session_state["spatial_view_bbox"] = effective_bbox
            st.session_state["spatial_selected_layer_name"] = selected_layer_name
            st.session_state["spatial_selected_feature_name"] = selected_feature_name
            st.session_state["spatial_selected_feature_geojson"] = _json_safe_obj(selected_feature_geojson)
            st.rerun()
    elif selected_layer_name == "HCFCD_ProjectBoundaries" and selected_feature_name:
        if (
            st.session_state.get("spatial_selected_layer_name") != selected_layer_name
            or st.session_state.get("spatial_selected_feature_name") != selected_feature_name
        ):
            # On the first HCFCD click after page load, avoid snapping back to global extent.
            hcfcd_outlines = layer_geojson_lookup.get("HCFCD_ProjectBoundaries__outlines", {})
            selected_outline_fc = hcfcd_outlines.get(selected_feature_name)
            selected_click_bbox = _bbox_from_features(selected_outline_fc.get("features", [])) if selected_outline_fc else None
            st.session_state["spatial_view_bbox"] = selected_click_bbox if selected_click_bbox else effective_bbox
            st.session_state.pop("spatial_forced_bbox", None)
            st.session_state["spatial_selected_layer_name"] = selected_layer_name
            st.session_state["spatial_selected_feature_name"] = selected_feature_name
            st.session_state["spatial_selected_feature_geojson"] = None
            st.rerun()
    elif stored_selected_layer not in selected_layers:
        st.session_state.pop("spatial_selected_layer_name", None)
        st.session_state.pop("spatial_selected_feature_name", None)
        st.session_state.pop("spatial_selected_feature_geojson", None)
        st.session_state.pop("spatial_selected_hcfcd_project_id", None)
        st.session_state.pop("spatial_forced_bbox", None)

    if selected_feature_row.empty and stored_selected_layer in layer_tables and stored_selected_feature_name:
        df_sel = layer_tables[stored_selected_layer]
        if stored_selected_layer == "HCFCD_ProjectBoundaries":
            matched = df_sel.loc[df_sel["project_id"].astype(str).str.strip() == stored_selected_feature_name].head(1)
        else:
            name_col = _get_feature_name_column(df_sel)
            matched = df_sel.loc[df_sel[name_col].astype(str).str.strip() == stored_selected_feature_name].head(1) if name_col else pd.DataFrame()
        if not matched.empty:
            selected_feature_row = matched.copy()
            selected_layer_name = stored_selected_layer
            selected_feature_name = stored_selected_feature_name
    elif "HCFCD_ProjectBoundaries" not in selected_layers:
        st.session_state.pop("spatial_selected_hcfcd_project_id", None)
        st.session_state.pop("spatial_forced_bbox", None)
        st.session_state["spatial_view_bbox"] = base_bbox

    if selected_layer_name and not selected_feature_row.empty:
        if selected_layer_name == "HCFCD_ProjectBoundaries":
            row = selected_feature_row.iloc[0]
            project_name = str(row.get("project_name", "")).strip() or str(row.get("Project Name", "")).strip() or selected_feature_name
            project_id = str(row.get("project_id", "")).strip() or str(row.get("ProjectID", "")).strip() or selected_feature_name
            lifecycle_stage = str(row.get("lifecycle_stage", "")).strip() or str(row.get("Lifecycle Stage", "")).strip()
            watershed = str(row.get("watershed", "")).strip() or str(row.get("Watershed", "")).strip()
            st.markdown(f"<div style='font-size:1.35rem;font-weight:700;margin-bottom:0.35rem;'>Project Name: {project_name}</div>", unsafe_allow_html=True)
            st.markdown(f"**Project ID:** {project_id or 'N/A'}")
            st.markdown(f"**Lifecycle Stage:** {lifecycle_stage or 'N/A'}")
            st.markdown(f"**Watershed:** {watershed or 'N/A'}")
        else:
            matches_df = _load_named_database_matches(selected_layer_name, selected_feature_name)
            title_text = selected_feature_name or f"Feature {selected_feature_row.index[0] + 1}"
            st.markdown(f"**Selected Feature:** {title_text}")
            if not matches_df.empty:
                st.caption("Matched project information found by name.")
                for idx, (_, row) in enumerate(matches_df.iterrows(), start=1):
                    label = f"Matched Database Record {idx}"
                    if "Source" in row and str(row["Source"]).strip():
                        label = f"{label} - {row['Source']}"
                    with st.expander(label, expanded=(idx == 1)):
                        _render_record_details(row.drop(labels=[c for c in ["Source", "Matched On"] if c in row.index]))
            else:
                st.caption("No matching project information was found for this feature name.")

        if "spatial_feature_notes" not in st.session_state:
            st.session_state["spatial_feature_notes"] = {}
        notes_key = f"{selected_layer_name}::{selected_feature_name or selected_feature_row.index[0]}"
        current_note = st.session_state["spatial_feature_notes"].get(notes_key, "")
        note_value = st.text_area(
            "Notes for selected feature",
            value=current_note,
            key=f"spatial_note_editor_{notes_key}",
            height=120,
            placeholder="Add project-specific notes here.",
        )
        st.session_state["spatial_feature_notes"][notes_key] = note_value
    else:
        st.caption("Click a project or thematic feature on the map to inspect it and add notes.")

    with st.expander("Find Specific HCFCD Project", expanded=False):
        st.markdown(
            """
            <div class="spatial-find-box">
              <div class="spatial-find-caption">Type 3+ characters from project name to get matching suggestions.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "HCFCD_ProjectBoundaries" in layer_tables and not layer_tables["HCFCD_ProjectBoundaries"].empty:
            hcfcd_df = layer_tables["HCFCD_ProjectBoundaries"].copy()
            name_series = hcfcd_df.get("project_name", pd.Series(dtype=str)).astype(str).str.strip()
            id_series = hcfcd_df.get("project_id", pd.Series(dtype=str)).astype(str).str.strip()
            project_lookup = (
                pd.DataFrame({"project_name": name_series, "project_id": id_series})
                .replace({"project_name": {"": np.nan}, "project_id": {"": np.nan}})
                .dropna(subset=["project_name", "project_id"])
                .drop_duplicates()
                .sort_values(["project_name", "project_id"])
            )
            search_text = st.text_input(
                "HCFCD project name contains",
                key="hcfcd_project_search_text",
                placeholder="e.g., green",
            ).strip()
            if len(search_text) >= 3:
                matches = project_lookup.loc[project_lookup["project_name"].str.contains(search_text, case=False, na=False)].copy()
                if matches.empty:
                    st.caption("No HCFCD project name matches found.")
                else:
                    options = [f"{r.project_name} ({r.project_id})" for r in matches.itertuples(index=False)]
                    selected_label = st.selectbox(
                        "Matching HCFCD projects",
                        options=options,
                        key="hcfcd_project_search_match",
                    )
                    if st.button("Select HCFCD Project", key="btn_select_hcfcd_from_search", type="primary"):
                        selected_row = matches.iloc[options.index(selected_label)]
                        selected_project_id = str(selected_row["project_id"]).strip()
                        if selected_project_id:
                            st.session_state["spatial_selected_layer_name"] = "HCFCD_ProjectBoundaries"
                            st.session_state["spatial_selected_feature_name"] = selected_project_id
                            st.session_state["spatial_selected_hcfcd_project_id"] = selected_project_id
                            st.session_state["spatial_selected_feature_geojson"] = None
                            st.session_state.pop("spatial_forced_bbox", None)
                            st.rerun()
            else:
                st.caption("Start typing 3 or more characters to get suggestions.")
        else:
            st.caption("HCFCD Project Boundaries layer is not available in this map view.")

    with st.expander("Find Specific Candidate Project / Study Area", expanded=False):
        st.markdown(
            """
            <div class="spatial-find-box study">
              <div class="spatial-find-caption">Search Candidate Projects and Planning Level Study Areas by name.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        candidate_search_rows = []
        for layer_name in ["Candidate_Projects", "PlanningLevel_Projects"]:
            if layer_name not in layer_tables or layer_name not in layer_geojson_lookup:
                continue
            df_layer = layer_tables[layer_name]
            if df_layer.empty:
                continue
            name_col = _get_feature_name_column(df_layer)
            if not name_col or name_col not in df_layer.columns:
                continue
            for idx, val in df_layer[name_col].astype(str).items():
                name_val = str(val).strip()
                if not name_val:
                    continue
                candidate_search_rows.append(
                    {
                        "layer_name": layer_name,
                        "feature_index": int(idx),
                        "feature_name": name_val,
                    }
                )
        if candidate_search_rows:
            candidate_lookup = (
                pd.DataFrame(candidate_search_rows)
                .drop_duplicates(subset=["layer_name", "feature_index", "feature_name"])
                .sort_values(["feature_name", "layer_name"])
            )
            study_search_text = st.text_input(
                "Candidate / Study Area name contains",
                key="candidate_study_search_text",
                placeholder="e.g., creek",
            ).strip()
            if len(study_search_text) >= 3:
                study_matches = candidate_lookup.loc[candidate_lookup["feature_name"].str.contains(study_search_text, case=False, na=False)].copy()
                if study_matches.empty:
                    st.caption("No Candidate Project / Study Area matches found.")
                else:
                    study_options = [f"{r.feature_name} [{r.layer_name}]" for r in study_matches.itertuples(index=False)]
                    selected_study_label = st.selectbox(
                        "Matching Candidate / Study Area",
                        options=study_options,
                        key="candidate_study_search_match",
                    )
                    if st.button("Select Candidate / Study Area", key="btn_select_candidate_study", type="primary"):
                        selected_study = study_matches.iloc[study_options.index(selected_study_label)]
                        layer_name = str(selected_study["layer_name"])
                        feature_index = int(selected_study["feature_index"])
                        feature_name = str(selected_study["feature_name"])
                        features = layer_geojson_lookup.get(layer_name, {}).get("features", [])
                        if 0 <= feature_index < len(features):
                            st.session_state["spatial_selected_layer_name"] = layer_name
                            st.session_state["spatial_selected_feature_name"] = feature_name
                            st.session_state["spatial_selected_feature_geojson"] = _json_safe_obj(
                                {"type": "FeatureCollection", "features": [features[feature_index]]}
                            )
                            st.session_state.pop("spatial_selected_hcfcd_project_id", None)
                            st.session_state.pop("spatial_forced_bbox", None)
                            st.rerun()
            else:
                st.caption("Start typing 3 or more characters to get suggestions.")
        else:
            st.caption("Candidate/Planning layers are not available in this map view.")


def render_spatial_tab_folium():
    st.subheader("Spatial View")
    st.markdown("<div id='folium_map_anchor'></div>", unsafe_allow_html=True)
    components.html(
        """
        <script>
          const anchor = window.parent.document.getElementById("folium_map_anchor");
          if (anchor) {
            anchor.scrollIntoView({behavior: "auto", block: "start"});
          }
        </script>
        """,
        height=0,
    )
    if folium is None or st_folium is None:
        st.error("`folium` and `streamlit-folium` are required for this tab. Install them from `requirements.txt`.")
        return

    shp_paths = get_spatial_shapefile_paths()
    if not shp_paths:
        st.info("No shapefiles found in the `Support Files/SHPs` folder.")
        return

    base_layers = [name for name in ["PC1_Boundary", "PC1_UnincorporatedRegion"] if name in shp_paths]
    core_layers = [name for name in ["Candidate_Projects", "PlanningLevel_Projects", "HCFCD_ProjectBoundaries"] if name in shp_paths]
    other_layers = [name for name in shp_paths if name not in set(base_layers + core_layers)]

    # Folium LayerControl (inside map) handles visibility toggles.
    active_layers = base_layers + core_layers + other_layers
    if not active_layers:
        st.info("No spatial layers are available to display.")
        return

    layer_tables: dict[str, pd.DataFrame] = {}
    layer_geojson_lookup: dict[str, dict] = {}
    layer_bboxes = []
    load_errors = []
    hcfcd_points = []
    hcfcd_outlines_by_project: dict[str, dict] = {}
    for layer_name in active_layers:
        try:
            if layer_name == "HCFCD_ProjectBoundaries":
                pts, outline_fc, attr_df, bbox = load_hcfcd_project_boundaries(
                    shp_paths[layer_name],
                    shp_paths.get("PC1_Boundary"),
                )
                layer_tables[layer_name] = attr_df
                hcfcd_points = pts
                hcfcd_outlines_by_project = outline_fc
                layer_bboxes.append(bbox or _point_bbox(pts))
            else:
                geojson, attr_df, bbox = load_shapefile_feature_collection(shp_paths[layer_name])
                layer_geojson_lookup[layer_name] = geojson
                layer_tables[layer_name] = attr_df
                layer_bboxes.append(bbox)
        except Exception as ex:
            load_errors.append(f"{layer_name}: {ex}")

    if load_errors:
        st.warning("Some layers could not be loaded.")
        for msg in load_errors:
            st.write(f"- {msg}")

    combined_bbox = _merge_bboxes(layer_bboxes)
    pc1_bbox = _bbox_from_features(layer_geojson_lookup["PC1_Boundary"].get("features", [])) if "PC1_Boundary" in layer_geojson_lookup else None
    base_bbox = pc1_bbox or combined_bbox

    selected_hcfcd_project_id = str(st.session_state.get("folium_selected_hcfcd_project_id", "")).strip()
    selected_layer_name = str(st.session_state.get("folium_selected_layer_name", "")).strip()
    selected_feature_name = str(st.session_state.get("folium_selected_feature_name", "")).strip()
    selected_feature_geojson = st.session_state.get("folium_selected_feature_geojson")
    forced_bbox = st.session_state.get("folium_forced_bbox")

    # Read the previous component event before drawing this run, so selection updates immediately.
    component_state = st.session_state.get("spatial_map_folium")
    component_popup = ""
    if isinstance(component_state, dict):
        component_popup = str(component_state.get("last_object_clicked_popup") or "").strip()
        center_obj = component_state.get("center")
        zoom_obj = component_state.get("zoom")
        if isinstance(center_obj, dict) and {"lat", "lng"} <= set(center_obj.keys()):
            st.session_state["folium_center"] = {
                "lat": float(center_obj["lat"]),
                "lng": float(center_obj["lng"]),
            }
        if isinstance(zoom_obj, (int, float)):
            st.session_state["folium_zoom"] = float(zoom_obj)

    # Ignore stale popup once after clear to prevent immediate re-select.
    ignored_popup = str(st.session_state.get("folium_ignored_popup", "")).strip()
    if component_popup and component_popup == ignored_popup:
        pass
    elif component_popup and component_popup.startswith("HCFCD::"):
        pid = component_popup.split("HCFCD::", 1)[1].strip()
        if pid and pid != selected_hcfcd_project_id:
            selected_hcfcd_project_id = pid
            selected_layer_name = "HCFCD_ProjectBoundaries"
            selected_feature_name = pid
            selected_feature_geojson = None
            st.session_state["folium_selected_hcfcd_project_id"] = pid
            st.session_state["folium_selected_layer_name"] = "HCFCD_ProjectBoundaries"
            st.session_state["folium_selected_feature_name"] = pid
            st.session_state["folium_selected_feature_geojson"] = None
            st.session_state.pop("folium_forced_bbox", None)
    elif ignored_popup and component_popup != ignored_popup:
        st.session_state.pop("folium_ignored_popup", None)
    selected_outline_bbox = None
    if selected_hcfcd_project_id and selected_hcfcd_project_id in hcfcd_outlines_by_project:
        selected_outline_bbox = _bbox_from_features(hcfcd_outlines_by_project[selected_hcfcd_project_id].get("features", []))

    if "folium_center" not in st.session_state or "folium_zoom" not in st.session_state:
        init_lat, init_lon, init_zoom = _zoom_from_bbox(base_bbox)
        st.session_state["folium_center"] = {"lat": float(init_lat), "lng": float(init_lon)}
        st.session_state["folium_zoom"] = float(init_zoom)

    center_lat, center_lon, zoom = _zoom_from_bbox(base_bbox)
    saved_center = st.session_state.get("folium_center")
    saved_zoom = st.session_state.get("folium_zoom")
    if isinstance(saved_center, dict) and {"lat", "lng"} <= set(saved_center.keys()):
        center_lat = float(saved_center["lat"])
        center_lon = float(saved_center["lng"])
    if isinstance(saved_zoom, (int, float)):
        zoom = float(saved_zoom)
    if isinstance(forced_bbox, list) and len(forced_bbox) == 4:
        center_lat, center_lon, zoom = _zoom_from_bbox(forced_bbox)

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)
    base_layer_objects = []
    overlay_layer_objects = []

    original_tile = folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Original Basemap",
        overlay=False,
        control=False,
        show=True,
    )
    original_tile.add_to(fmap)
    base_layer_objects.append(original_tile)

    satellite_tile = folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Satellite Imagery",
        overlay=False,
        control=False,
        show=False,
    )
    satellite_tile.add_to(fmap)
    base_layer_objects.append(satellite_tile)

    if "PC1_UnincorporatedRegion" in layer_geojson_lookup:
        pc1_uninc_fg = folium.FeatureGroup(name="Precinct 1 Unincorporated Region", show=True, overlay=True)
        folium.GeoJson(
            layer_geojson_lookup["PC1_UnincorporatedRegion"],
            name="Precinct 1 Unincorporated Region",
            style_function=lambda _: {"fillColor": "#7A7A7A", "color": "#7A7A7A", "weight": 0, "fillOpacity": 0.16},
            tooltip=None,
            highlight_function=None,
        ).add_to(pc1_uninc_fg)
        pc1_uninc_fg.add_to(fmap)
        base_layer_objects.append(pc1_uninc_fg)

    if "PC1_Boundary" in layer_geojson_lookup:
        pc1_boundary_fg = folium.FeatureGroup(name="Precinct 1 Boundary", show=True, overlay=True)
        folium.GeoJson(
            layer_geojson_lookup["PC1_Boundary"],
            name="Precinct 1 Boundary",
            style_function=lambda _: {"fillColor": "#00000000", "color": "#7C4E2D", "weight": 3, "fillOpacity": 0.0},
            tooltip=None,
            highlight_function=None,
        ).add_to(pc1_boundary_fg)
        pc1_boundary_fg.add_to(fmap)
        base_layer_objects.append(pc1_boundary_fg)

    for lname in ["Candidate_Projects", "PlanningLevel_Projects"]:
        if lname in layer_geojson_lookup:
            fill, line = _spatial_layer_colors(lname)
            tooltip_cols = ["Name"] if "Name" in layer_tables.get(lname, pd.DataFrame()).columns else None
            fg_name = "Candidate Projects" if lname == "Candidate_Projects" else "Planning Level Study Areas"
            thematic_fg = folium.FeatureGroup(name=fg_name, show=True, overlay=True)
            folium.GeoJson(
                layer_geojson_lookup[lname],
                name=fg_name,
                style_function=lambda _, fc=fill, lc=line: {
                    "fillColor": f"rgba({fc[0]},{fc[1]},{fc[2]},0.45)",
                    "color": f"rgba({lc[0]},{lc[1]},{lc[2]},1.0)",
                    "weight": 2,
                    "fillOpacity": 0.45,
                },
                tooltip=folium.GeoJsonTooltip(fields=tooltip_cols) if tooltip_cols else None,
            ).add_to(thematic_fg)
            thematic_fg.add_to(fmap)
            overlay_layer_objects.append(thematic_fg)

    for lname in other_layers:
        if lname in layer_geojson_lookup:
            other_fg = folium.FeatureGroup(name=lname.replace("_", " "), show=False, overlay=True)
            folium.GeoJson(
                layer_geojson_lookup[lname],
                name=lname.replace("_", " "),
                style_function=lambda _: {"fillColor": "#8A8A8A", "color": "#4B4B4B", "weight": 1.8, "fillOpacity": 0.28},
                tooltip=None,
            ).add_to(other_fg)
            other_fg.add_to(fmap)
            overlay_layer_objects.append(other_fg)

    if "HCFCD_ProjectBoundaries" in core_layers:
        hcfcd_fg = folium.FeatureGroup(name="HCFCD Project Boundaries (Points)", show=False)
        points_to_draw = hcfcd_points
        if selected_hcfcd_project_id:
            points_to_draw = [r for r in hcfcd_points if str(r.get("project_id", "")).strip() == selected_hcfcd_project_id]
        for row in points_to_draw:
            pid = str(row.get("project_id", "")).strip()
            pname = str(row.get("project_name", "")).strip()
            popup_val = f"HCFCD::{pid}"
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lon"])],
                radius=4,
                color="#1E4A8A",
                weight=1,
                fill=True,
                fill_color="#396CB1",
                fill_opacity=0.9,
                tooltip=pname or pid,
                popup=popup_val,
            ).add_to(hcfcd_fg)
        hcfcd_fg.add_to(fmap)
        overlay_layer_objects.append(hcfcd_fg)

    if selected_hcfcd_project_id and selected_hcfcd_project_id in hcfcd_outlines_by_project:
        folium.GeoJson(
            hcfcd_outlines_by_project[selected_hcfcd_project_id],
            name="Selected HCFCD Project",
            style_function=lambda _: {"fillColor": "#C47838", "color": "#B85220", "weight": 5, "fillOpacity": 0.08},
            tooltip=None,
            highlight_function=None,
        ).add_to(fmap)
    elif selected_feature_geojson and selected_layer_name in core_layers + other_layers:
        folium.GeoJson(
            selected_feature_geojson,
            name="Selected Feature",
            style_function=lambda _: {"fillColor": "#C47838", "color": "#B85220", "weight": 4, "fillOpacity": 0.18},
            tooltip=None,
            highlight_function=None,
        ).add_to(fmap)

    if isinstance(forced_bbox, list) and len(forced_bbox) == 4:
        fmap.fit_bounds([[forced_bbox[1], forced_bbox[0]], [forced_bbox[3], forced_bbox[2]]])
        st.session_state.pop("folium_forced_bbox", None)
    elif base_bbox and len(base_bbox) == 4 and "folium_view_initialized" not in st.session_state:
        fmap.fit_bounds([[base_bbox[1], base_bbox[0]], [base_bbox[3], base_bbox[2]]])
        st.session_state["folium_view_initialized"] = True

    legend_html = """
    <div style="
      position: fixed; bottom: 24px; left: 22px; z-index: 9999;
      background: rgba(255,255,255,0.92); border:1px solid #cbd5e1; border-radius:8px;
      padding:8px 10px; font-size:12px; color:#111827; min-width:310px;">
      <div style="font-weight:700; margin-bottom:6px;">Map Legend</div>
      <div style="white-space:nowrap;"><span style="display:inline-block;width:16px;height:2px;background:#7C4E2D;vertical-align:middle;margin-right:6px;"></span>Precinct 1 Boundary</div>
      <div style="white-space:nowrap;"><span style="display:inline-block;width:12px;height:12px;background:#7A7A7A;opacity:0.35;margin-right:6px;"></span>Precinct 1 Unincorporated Region</div>
      <div style="white-space:nowrap;"><span style="display:inline-block;width:12px;height:12px;background:#1E8A48;opacity:0.8;margin-right:6px;"></span>Candidate Projects</div>
      <div style="white-space:nowrap;"><span style="display:inline-block;width:12px;height:12px;background:#FF911A;opacity:0.85;margin-right:6px;"></span>Planning Level Study Areas</div>
      <div style="white-space:nowrap;"><span style="display:inline-block;width:10px;height:10px;background:#396CB1;border-radius:50%;margin-right:6px;"></span>HCFCD Points</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    if GroupedLayerControl is not None:
        GroupedLayerControl(
            groups={
                "Base Map": base_layer_objects,
                "Layer Visibility": overlay_layer_objects,
            },
            exclusive_groups=False,
            collapsed=False,
        ).add_to(fmap)
    else:
        folium.LayerControl(collapsed=False).add_to(fmap)

    with st.container(border=True):
        map_state = st_folium(
            fmap,
            width=None,
            height=700,
            key="spatial_map_folium",
            returned_objects=["center", "zoom", "last_object_clicked_popup"],
        )

    # Selection is handled at the top of the function from prior component state.
    _ = map_state

    if selected_hcfcd_project_id and "HCFCD_ProjectBoundaries" in core_layers:
        _, c1, c2, _ = st.columns([2.4, 2.2, 3.0, 2.4])
        with c1:
            if st.button("Clear selected HCFCD project", key="btn_clear_hcfcd_project_folium", type="primary"):
                if component_popup:
                    st.session_state["folium_ignored_popup"] = component_popup
                st.session_state.pop("folium_selected_hcfcd_project_id", None)
                st.session_state.pop("folium_selected_layer_name", None)
                st.session_state.pop("folium_selected_feature_name", None)
                st.session_state.pop("folium_selected_feature_geojson", None)
                st.session_state.pop("folium_forced_bbox", None)
        with c2:
            if st.button("Zoom to extent of selected HCFCD Project", key="btn_zoom_hcfcd_project_folium", type="primary", disabled=not bool(selected_outline_bbox)):
                if selected_outline_bbox:
                    st.session_state["folium_forced_bbox"] = selected_outline_bbox

    if selected_hcfcd_project_id and "HCFCD_ProjectBoundaries" in layer_tables:
        df_sel = layer_tables["HCFCD_ProjectBoundaries"]
        selected_feature_row = df_sel.loc[df_sel["project_id"].astype(str).str.strip() == selected_hcfcd_project_id].head(1)
        if not selected_feature_row.empty:
            row = selected_feature_row.iloc[0]
            project_name = str(row.get("project_name", "")).strip() or selected_hcfcd_project_id
            project_id = str(row.get("project_id", "")).strip() or selected_hcfcd_project_id
            lifecycle_stage = str(row.get("lifecycle_stage", "")).strip()
            watershed = str(row.get("watershed", "")).strip()
            st.markdown(f"<div style='font-size:1.35rem;font-weight:700;margin-bottom:0.35rem;'>Project Name: {project_name}</div>", unsafe_allow_html=True)
            st.markdown(f"**Project ID:** {project_id or 'N/A'}")
            st.markdown(f"**Lifecycle Stage:** {lifecycle_stage or 'N/A'}")
            st.markdown(f"**Watershed:** {watershed or 'N/A'}")
    else:
        st.caption("Click an HCFCD point or use search below to inspect a project.")

    if "folium_feature_notes" not in st.session_state:
        st.session_state["folium_feature_notes"] = {}
    note_key = f"HCFCD_ProjectBoundaries::{selected_hcfcd_project_id}" if selected_hcfcd_project_id else "none"
    current_note = st.session_state["folium_feature_notes"].get(note_key, "")
    note_value = st.text_area(
        "Notes for selected feature",
        value=current_note,
        key=f"folium_note_editor_{note_key}",
        height=120,
        placeholder="Add project-specific notes here.",
    )
    st.session_state["folium_feature_notes"][note_key] = note_value

    with st.expander("Find Specific HCFCD Project", expanded=False):
        st.markdown(
            "<div class='spatial-find-box'><div class='spatial-find-caption'>Type 3+ characters from project name to get matching suggestions.</div></div>",
            unsafe_allow_html=True,
        )
        if "HCFCD_ProjectBoundaries" in layer_tables and not layer_tables["HCFCD_ProjectBoundaries"].empty:
            hcfcd_df = layer_tables["HCFCD_ProjectBoundaries"].copy()
            project_lookup = (
                hcfcd_df[["project_name", "project_id"]]
                .astype(str)
                .replace({"project_name": {"": np.nan}, "project_id": {"": np.nan}})
                .dropna(subset=["project_name", "project_id"])
                .drop_duplicates()
                .sort_values(["project_name", "project_id"])
            )
            search_text = st.text_input(
                "HCFCD project name contains",
                key="hcfcd_project_search_text_folium",
                placeholder="e.g., green",
            ).strip()
            if len(search_text) >= 3:
                matches = project_lookup.loc[project_lookup["project_name"].str.contains(search_text, case=False, na=False)].copy()
                if matches.empty:
                    st.caption("No HCFCD project name matches found.")
                else:
                    options = [f"{r.project_name} ({r.project_id})" for r in matches.itertuples(index=False)]
                    selected_label = st.selectbox("Matching HCFCD projects", options=options, key="hcfcd_project_search_match_folium")
                    if st.button("Select HCFCD Project", key="btn_select_hcfcd_from_search_folium", type="primary"):
                        selected_row = matches.iloc[options.index(selected_label)]
                        selected_project_id = str(selected_row["project_id"]).strip()
                        st.session_state["folium_selected_hcfcd_project_id"] = selected_project_id
                        st.session_state["folium_selected_layer_name"] = "HCFCD_ProjectBoundaries"
                        st.session_state["folium_selected_feature_name"] = selected_project_id
                        st.session_state["folium_selected_feature_geojson"] = None
                        st.session_state.pop("folium_forced_bbox", None)
            else:
                st.caption("Start typing 3 or more characters to get suggestions.")
        else:
            st.caption("HCFCD Project Boundaries layer is not available in this map view.")

    with st.expander("Find Specific Candidate Project / Study Area", expanded=False):
        st.markdown(
            "<div class='spatial-find-box study'><div class='spatial-find-caption'>Search Candidate Projects and Planning Level Study Areas by name.</div></div>",
            unsafe_allow_html=True,
        )
        search_rows = []
        for layer_name in ["Candidate_Projects", "PlanningLevel_Projects"]:
            df_layer = layer_tables.get(layer_name, pd.DataFrame())
            if df_layer.empty:
                continue
            name_col = _get_feature_name_column(df_layer)
            if not name_col:
                continue
            for idx, val in df_layer[name_col].astype(str).items():
                name_val = str(val).strip()
                if name_val:
                    search_rows.append({"layer_name": layer_name, "feature_index": int(idx), "feature_name": name_val})
        if search_rows:
            lookup = pd.DataFrame(search_rows).drop_duplicates().sort_values(["feature_name", "layer_name"])
            text = st.text_input("Candidate / Study Area name contains", key="candidate_study_search_text_folium", placeholder="e.g., creek").strip()
            if len(text) >= 3:
                matches = lookup.loc[lookup["feature_name"].str.contains(text, case=False, na=False)].copy()
                if matches.empty:
                    st.caption("No Candidate Project / Study Area matches found.")
                else:
                    options = [f"{r.feature_name} [{r.layer_name}]" for r in matches.itertuples(index=False)]
                    selected = st.selectbox("Matching Candidate / Study Area", options=options, key="candidate_study_search_match_folium")
                    if st.button("Select Candidate / Study Area", key="btn_select_candidate_study_folium", type="primary"):
                        row = matches.iloc[options.index(selected)]
                        lname = str(row["layer_name"])
                        idx = int(row["feature_index"])
                        features = layer_geojson_lookup.get(lname, {}).get("features", [])
                        if 0 <= idx < len(features):
                            st.session_state["folium_selected_layer_name"] = lname
                            st.session_state["folium_selected_feature_name"] = str(row["feature_name"])
                            st.session_state["folium_selected_feature_geojson"] = _json_safe_obj({"type": "FeatureCollection", "features": [features[idx]]})
                            st.session_state.pop("folium_selected_hcfcd_project_id", None)
                            st.session_state.pop("folium_forced_bbox", None)
                            feature_bbox = _bbox_from_features([features[idx]])
                            if feature_bbox:
                                st.session_state["folium_forced_bbox"] = feature_bbox
            else:
                st.caption("Start typing 3 or more characters to get suggestions.")
        else:
            st.caption("Candidate/Planning layers are not available in this map view.")


def render_spatial_tab():
    engine = str(st.session_state.get("spatial_engine", "pydeck")).strip().lower()
    if engine not in {"folium", "pydeck"}:
        engine = "pydeck"

    if engine == "pydeck":
        render_spatial_tab_pydeck()
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Switch to Folium", key="btn_switch_to_folium_bottom"):
            st.session_state["spatial_engine"] = "folium"
            st.rerun()
    else:
        render_spatial_tab_folium()
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Switch to Pydeck", key="btn_switch_to_pydeck_bottom"):
            st.session_state["spatial_engine"] = "pydeck"
            st.rerun()


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
                corr_df.style.map(_corr_style).format("{:.3f}"),
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
    st.subheader("Pairwise Comparision (AHP)")
    st.write("Build a pairwise comparison table using the Saaty scale. The value means Criterion A is preferred over Criterion B.")

    meta = get_criteria_meta()
    label_map = {k: v for k, v in meta}
    criteria_keys = [k for k, _ in meta]
    standard_criteria_keys = [k for k, _ in BASE_CRITERIA_META if k in criteria_keys]

    def _criteria_key_from_label(target_label: str, label_map_local: dict[str, str], criteria_keys_local: list[str]) -> str | None:
        target_norm = _norm_col_name(target_label)
        for key_local in criteria_keys_local:
            aliases = [key_local, label_map_local.get(key_local, "")]
            if SCORE_COL_MAP.get(key_local):
                aliases.append(SCORE_COL_MAP[key_local])
            if any(_norm_col_name(a) == target_norm for a in aliases if a):
                return key_local
        return None

    def _ensure_custom_criterion_for_label(target_label: str) -> str | None:
        alias_map = {
            "10YR Flood Depth": [
                "Mean_10YR_Flood_Depth_Mean",
                "Mean 10 YR Flood Depth Mean",
                "10YR Flood Depth",
                "10 YR Flood Depth",
            ],
            "Total Benefiting Population": ["Total Benefiting Population"],
            "Maximum Flooding Depth": ["Maximum Flooding Depth"],
            "LMI": ["LMI"],
        }
        aliases = alias_map.get(target_label, [target_label])
        df_current = st.session_state.get("df_work", pd.DataFrame())
        has_df = isinstance(df_current, pd.DataFrame) and not df_current.empty
        matched_col = find_column_by_aliases(df_current, aliases) if has_df else None

        # Known preset fields should remain selectable even if the current session
        # still has an older dataset loaded. Ranking will use the column once the
        # updated candidate database is loaded.
        if not matched_col and target_label in alias_map:
            matched_col = aliases[0]
        if not matched_col:
            return None

        custom_criteria = list(st.session_state.get("custom_criteria", []))
        existing = {c.get("key"): c for c in custom_criteria if c.get("key")}
        if matched_col not in existing:
            dtype = (
                "Number"
                if has_df and matched_col in df_current.columns and pd.api.types.is_numeric_dtype(df_current[matched_col])
                else "Number"
            )
            custom_criteria.append({
                "key": matched_col,
                "label": target_label,
                "include": True,
                "type": dtype,
            })
            st.session_state["custom_criteria"] = custom_criteria
            if matched_col not in st.session_state["weights_pct"]:
                st.session_state["weights_pct"][matched_col] = 0.0
        return matched_col

    def _preset_keys_from_ahp_file(file_name: str) -> list[str]:
        db_lookup = dict(get_ahp_csv_files())
        file_path = db_lookup.get(file_name)
        if not file_path:
            return []

        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            header = f.readline().strip().split(",")
        preset_labels = [h.strip() for h in header[1:] if h.strip()]

        resolved_keys = []
        local_meta = get_criteria_meta()
        local_label_map = {k: v for k, v in local_meta}
        local_keys = [k for k, _ in local_meta]
        for preset_label in preset_labels:
            resolved = _criteria_key_from_label(preset_label, local_label_map, local_keys)
            if resolved is None:
                resolved = _ensure_custom_criterion_for_label(preset_label)
                if resolved:
                    local_meta = get_criteria_meta()
                    local_label_map = {k: v for k, v in local_meta}
                    local_keys = [k for k, _ in local_meta]
            if resolved and resolved not in resolved_keys:
                resolved_keys.append(resolved)
        return resolved_keys

    p1, p2, _ = st.columns([1.35, 1.6, 3.0])
    with p1:
        if st.button("Use HCFCD Parameters", key="btn_ahp_use_standard_hcfcd"):
            st.session_state["ahp_selected_criteria"] = standard_criteria_keys
            st.session_state["ahp_importable_matrix"] = "HCFCD Parameters AHP Final.csv"
            st.session_state["ahp_matrix_criteria_source"] = "HCFCD Parameters AHP Final.csv"
            reset_direct_weights_to_reference_hcfcd()
            st.session_state.pop("ahp_importable_loaded_sig", None)
            st.session_state["topsis_sync_from_ahp"] = True
            st.rerun()
    with p2:
        if st.button("Use Preset Selected Parameters", key="btn_ahp_use_selected_parameters"):
            preset_keys = _preset_keys_from_ahp_file("Selected Parameters AHP Final.csv")
            if preset_keys:
                st.session_state["ahp_selected_criteria"] = preset_keys
                st.session_state["ahp_importable_matrix"] = "Selected Parameters AHP Final.csv"
                st.session_state["ahp_matrix_criteria_source"] = "Selected Parameters AHP Final.csv"
                st.session_state.pop("ahp_importable_loaded_sig", None)
                st.session_state["topsis_sync_from_ahp"] = True
                st.rerun()
            else:
                st.warning("Could not match the preset selected parameters to current criteria.")

    meta = get_criteria_meta()
    label_map = {k: v for k, v in meta}
    criteria_keys = [k for k, _ in meta]
    pending_ahp_selection = st.session_state.pop("pending_ahp_selected_criteria", None)
    if pending_ahp_selection:
        st.session_state["ahp_selected_criteria"] = [
            k for k in pending_ahp_selection
            if k in criteria_keys
        ]
    current_ahp_selection = [
        k for k in st.session_state.get("ahp_selected_criteria", [])
        if k in criteria_keys
    ]
    if not current_ahp_selection:
        current_ahp_selection = criteria_keys
    st.session_state["ahp_selected_criteria"] = current_ahp_selection

    selected = st.multiselect(
        "Select criteria for AHP",
        options=criteria_keys,
        default=current_ahp_selection,
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

    importable_ahp = get_ahp_csv_files()
    if importable_ahp:
        ahp_names = [n for n, _ in importable_ahp]
        default_ahp_name = _default_database_name(importable_ahp, token="ahp")
        if st.session_state.get("ahp_importable_matrix") not in ahp_names:
            st.session_state["ahp_importable_matrix"] = default_ahp_name if default_ahp_name in ahp_names else ahp_names[0]
        selected_ahp_db = st.selectbox(
            "Load AHP matrix from Importable Database",
            options=ahp_names,
            index=ahp_names.index(default_ahp_name) if default_ahp_name in ahp_names else 0,
            key="ahp_importable_matrix",
        )
        if st.session_state.get("ahp_matrix_criteria_source") != selected_ahp_db:
            preset_keys_for_db = _preset_keys_from_ahp_file(selected_ahp_db)
            if preset_keys_for_db:
                st.session_state["pending_ahp_selected_criteria"] = preset_keys_for_db
                st.session_state["ahp_matrix_criteria_source"] = selected_ahp_db
                st.session_state.pop("ahp_importable_loaded_sig", None)
                st.session_state["topsis_sync_from_ahp"] = True
                st.rerun()
    uploaded_ahp = st.file_uploader("Or import completed AHP matrix CSV", type=["csv"], key="ahp_upload_matrix")

    if len(selected) < 2:
        st.warning("Select at least two criteria to run AHP.")
        return

    def _compute_ahp_from_pair_rows(pair_rows: list[dict], selected_keys: list[str], option_values_local: dict[str, float], label_map_local: dict[str, str]) -> tuple[dict, float]:
        n_local = len(selected_keys)
        matrix_local = np.ones((n_local, n_local), dtype=float)
        key_by_label_local = {label_map_local[k]: k for k in selected_keys}
        for row_local in pair_rows:
            a_label = row_local.get("Criterion A", "")
            b_label = row_local.get("Criterion B", "")
            pref_label = row_local.get("Preference", "1 (Equal)")
            if a_label in key_by_label_local and b_label in key_by_label_local:
                i_local = selected_keys.index(key_by_label_local[a_label])
                j_local = selected_keys.index(key_by_label_local[b_label])
                val_local = option_values_local.get(pref_label, 1.0)
                matrix_local[i_local, j_local] = val_local
                matrix_local[j_local, i_local] = 1.0 / val_local if val_local != 0 else 1.0
        weights_local, cr_local = ahp_weights(matrix_local)
        return {selected_keys[i]: float(weights_local[i]) for i in range(n_local)}, float(cr_local)

    saaty_options = [
        ("1/9 (Extreme)", 1/9),
        ("1/7 (Very strong)", 1/7),
        ("1/5 (Strong)", 1/5),
        ("1/4 (Between moderate and strong)", 1/4),
        ("1/3 (Moderate)", 1/3),
        ("1/2 (Between equal and moderate)", 1/2),
        ("1 (Equal)", 1.0),
        ("2 (Between equal and moderate)", 2.0),
        ("3 (Moderate)", 3.0),
        ("4 (Between moderate and strong)", 4.0),
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
            "Criterion A is between strongly and moderately less important than B",
            "Criterion A is moderately less important than B",
            "Criterion A is between equally and moderately less important than B",
            "Criteria A and B are equally important",
            "Criterion A is between equally and moderately more important than B",
            "Criterion A is moderately more important than B",
            "Criterion A is between moderately and strongly more important than B",
            "Criterion A is strongly more important than B",
            "Criterion A is very strongly more important than B",
            "Criterion A is extremely more important than B",
        ],
    })

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

    if importable_ahp:
        selected_path = dict(importable_ahp).get(selected_ahp_db)
        import_sig = f"{'|'.join(selected)}::{str(selected_ahp_db).strip().lower()}"
        if selected_path and st.session_state.get("ahp_importable_loaded_sig") != import_sig:
            try:
                imported_pairs, mapping_df, imported_count, defaulted_count, error_msg = load_ahp_matrix_pairs(selected_path, selected, label_map)
                st.session_state["ahp_importable_loaded_sig"] = import_sig
                if error_msg:
                    st.error(error_msg)
                    with st.expander("Show Matched Row/Column Details", expanded=False):
                        st.dataframe(mapping_df, use_container_width=True)
                else:
                    st.session_state["ahp_pairs"] = imported_pairs
                    st.session_state["ahp_pairs_selected"] = list(selected)
                    st.session_state["ahp_pairs_editor_version"] = st.session_state.get("ahp_pairs_editor_version", 0) + 1
                    imported_weights, imported_cr = _compute_ahp_from_pair_rows(imported_pairs, selected, option_values, label_map)
                    st.session_state["ahp_weights"] = imported_weights
                    st.session_state["ahp_cr"] = imported_cr
                    st.session_state["ahp_selected_snapshot"] = list(selected)
                    st.session_state["ahp_import_summary"] = {
                        "imported_count": imported_count,
                        "defaulted_count": defaulted_count,
                        "mapping_df": mapping_df,
                    }
                    st.success(
                        f"AHP matrix loaded from Importable Database. Imported {imported_count} pair values; "
                        f"defaulted {defaulted_count} pairs to 1 (Equal)."
                    )
                    st.rerun()
            except Exception as ex:
                st.session_state["ahp_importable_loaded_sig"] = import_sig
                st.error(f"Could not import AHP matrix from folder: {ex}")

    if uploaded_ahp is not None and st.button("Load AHP Matrix from CSV", key="btn_load_ahp_csv"):
        try:
            imported_pairs, mapping_df, imported_count, defaulted_count, error_msg = load_ahp_matrix_pairs(uploaded_ahp, selected, label_map)
            if error_msg:
                st.error(error_msg)
                with st.expander("Show Matched Row/Column Details", expanded=False):
                    st.dataframe(mapping_df, use_container_width=True)
            else:
                st.session_state["ahp_pairs"] = imported_pairs
                st.session_state["ahp_pairs_selected"] = list(selected)
                st.session_state["ahp_pairs_editor_version"] = st.session_state.get("ahp_pairs_editor_version", 0) + 1
                imported_weights, imported_cr = _compute_ahp_from_pair_rows(imported_pairs, selected, option_values, label_map)
                st.session_state["ahp_weights"] = imported_weights
                st.session_state["ahp_cr"] = imported_cr
                st.session_state["ahp_selected_snapshot"] = list(selected)
                st.session_state["ahp_import_summary"] = {
                    "imported_count": imported_count,
                    "defaulted_count": defaulted_count,
                    "mapping_df": mapping_df,
                }
                st.success(f"AHP matrix imported. Imported {imported_count} pair values; defaulted {defaulted_count} pairs to 1 (Equal).")
                st.rerun()
        except Exception as ex:
            st.error(f"Could not import AHP matrix: {ex}")

    if st.button("Compute AHP Weights", key="btn_compute_ahp"):
        pair_rows_current = st.session_state.get("ahp_pairs", [])
        manual_weights, manual_cr = _compute_ahp_from_pair_rows(pair_rows_current, selected, option_values, label_map)
        st.session_state["ahp_weights"] = manual_weights
        st.session_state["ahp_cr"] = manual_cr
        st.session_state["ahp_selected_snapshot"] = list(selected)

    with st.expander("Show / Hide Saaty Scale", expanded=False):
        st.dataframe(scale_df, use_container_width=True)

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

    default_weights_pct = {k: float(v) * 100.0 for k, v in config["weights"].items()}
    weights_pct = {
        **default_weights_pct,
        **st.session_state.get("weights_pct", {}),
    }
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

    custom_label_to_key = {}
    for item in st.session_state.get("custom_criteria", []):
        key = item.get("key")
        label = item.get("label") or item.get("key")
        if key and label:
            custom_label_to_key[_norm_col_name(label)] = key

    def _ranking_col_from_ahp_key(k: str) -> str | None:
        score_col = SCORE_COL_MAP.get(k)
        if score_col in available_cols:
            return score_col
        if k in available_cols:
            return k
        custom_key = custom_label_to_key.get(_norm_col_name(k))
        if custom_key in available_cols:
            return custom_key

        # Fall back to matching labels and column names. This keeps AHP presets usable
        # when criteria were created from dataset columns instead of framework scores.
        norm_target = _norm_col_name(k)
        criteria_label_map = {key: label for key, label in get_criteria_meta()}
        if k in criteria_label_map:
            norm_target = _norm_col_name(criteria_label_map[k])
        for col in available_cols:
            aliases = [col, label_by_col.get(col, "")]
            if any(_norm_col_name(alias) == norm_target for alias in aliases if alias):
                return col
        return None

    if st.session_state.get("topsis_sync_from_ahp"):
        ahp_order = st.session_state.get("ahp_selected_criteria", [])
        ordered_cols = []
        for k in ahp_order:
            c = _ranking_col_from_ahp_key(k)
            if c in available_cols and c not in ordered_cols:
                ordered_cols.append(c)
        if ordered_cols:
            st.session_state["topsis_selected_cols"] = ordered_cols
        st.session_state["topsis_sync_from_ahp"] = False

    btn_col_1, btn_col_2 = st.columns(2)
    with btn_col_1:
        if st.button("Use AHP-selected criteria order", key="btn_use_ahp_order"):
            ahp_order = st.session_state.get("ahp_selected_criteria", [])
            ordered_cols = []
            for k in ahp_order:
                c = _ranking_col_from_ahp_key(k)
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

    display_label_by_col = {v: f"Score - {label}" for k, label in BASE_CRITERIA_META for v in [SCORE_COL_MAP.get(k)] if v}
    for item in st.session_state.get("custom_criteria", []):
        key = item.get("key")
        label = item.get("label") or item.get("key")
        if key:
            display_label_by_col[key] = label
    col_by_display_label = {v: k for k, v in display_label_by_col.items()}

    # Older session state may have stored display labels instead of internal
    # column names. Convert back before filtering, otherwise TOPSIS settings
    # can lose rows and misalign ideal best/worst arrays.
    settings_df["Criterion"] = settings_df["Criterion"].map(lambda c: col_by_display_label.get(c, c))
    settings_df = settings_df[settings_df["Criterion"].isin(selected_cols)]
    if "Better Value" in settings_df.columns and "Type" not in settings_df.columns:
        settings_df["Type"] = settings_df["Better Value"].apply(
            lambda v: "Cost" if str(v).lower() == "lower" else "Benefit"
        )
    if "Better Value" in settings_df.columns:
        settings_df = settings_df.drop(columns=["Better Value"])

    if len(settings_df) != len(selected_cols):
        existing_settings = {
            str(row.get("Criterion", "")): str(row.get("Type", "Benefit"))
            for _, row in settings_df.iterrows()
        }
        settings_df = pd.DataFrame([
            {
                "Criterion": c,
                "Type": existing_settings.get(c, "Benefit"),
            }
            for c in selected_cols
        ])
        st.session_state["topsis_settings"] = settings_df.to_dict("records")
        st.session_state["topsis_settings_editor_version"] = st.session_state.get("topsis_settings_editor_version", 0) + 1
        st.session_state.pop("topsis_results_multi", None)

    label_by_col = display_label_by_col

    edited = None
    if any(m.endswith("+ TOPSIS") for m in methods):
        st.caption("Type: Benefit = higher is better, Cost = lower is better.")
        settings_display_df = settings_df.copy()
        settings_display_df["Criterion"] = settings_display_df["Criterion"].map(lambda c: label_by_col.get(c, c))
        settings_editor_key = f"topsis_settings_table_{st.session_state.get('topsis_settings_editor_version', 0)}"
        edited_display = st.data_editor(
            settings_display_df,
            use_container_width=True,
            hide_index=True,
            key=settings_editor_key,
            column_config={
                "Criterion": st.column_config.TextColumn(disabled=True),
                "Type": st.column_config.SelectboxColumn(options=["Benefit", "Cost"]),
            },
        )
        edited = edited_display.copy()
        edited["Criterion"] = edited["Criterion"].map(lambda c: col_by_display_label.get(c, c))
        st.session_state["topsis_settings"] = edited.to_dict("records")

    project_name_options = []
    if "project_name" in results.columns:
        project_name_options = [
            str(x)
            for x in results["project_name"].dropna().astype(str).drop_duplicates().tolist()
            if str(x).strip()
        ]
    excluded_project_names = st.multiselect(
        "Exclude projects from this ranking run",
        options=project_name_options,
        default=[
            x for x in st.session_state.get("ranking_excluded_project_names", [])
            if x in project_name_options
        ],
        key="ranking_excluded_project_names",
        help="Selected projects remain in the database but are not included when Run Ranking is pressed.",
    )
    if excluded_project_names:
        st.caption(f"{len(excluded_project_names)} project name(s) will be excluded from the next ranking run.")
    if st.session_state.get("ranking_exclusion_snapshot") != list(excluded_project_names):
        st.session_state["ranking_exclusion_snapshot"] = list(excluded_project_names)
        st.session_state.pop("topsis_results_multi", None)

    if st.button("Run Ranking", key="btn_run_topsis", type="primary"):
        ranking_results = results.copy()
        if excluded_project_names and "project_name" in ranking_results.columns:
            exclude_mask = ranking_results["project_name"].astype(str).isin(excluded_project_names)
            ranking_results = ranking_results.loc[~exclude_mask].copy()
        if ranking_results.empty:
            st.warning("All projects were excluded. Leave at least one project in the ranking run.")
            return

        data = ranking_results[selected_cols].copy()
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

                out = ranking_results.copy()
                out["ranking_score"] = scores
                out["ranking_rank"] = out["ranking_score"].rank(ascending=False, method="min").astype(int)
                out = out.sort_values(["ranking_score", "project_name"], ascending=[False, True]).reset_index(drop=True)

                show_cols = ["ranking_rank", "project_id", "project_name", "ranking_score"] + selected_cols
                results_by_source[method] = out[show_cols]
                continue

            benefit_flags = []
            ideal_best = []
            ideal_worst = []

            missing_rows = 0
            for _, row in edited.iterrows():
                c = row["Criterion"]
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
                edited_cols = set(edited["Criterion"].tolist()) if isinstance(edited, pd.DataFrame) and "Criterion" in edited.columns else set()
                for c in selected_cols:
                    if len(benefit_flags) >= len(selected_cols):
                        break
                    if c not in edited_cols:
                        benefit_flags.append(True)
                        ideal_best.append(float(data[c].max()))
                        ideal_worst.append(float(data[c].min()))

            inv_score_map = {v: k for k, v in SCORE_COL_MAP.items()}

            def _ahp_key_for_ranking_col(col: str, ahp_weights_local: dict) -> str | None:
                direct_key = inv_score_map.get(col, col)
                if direct_key in ahp_weights_local:
                    return direct_key
                for ahp_key in ahp_weights_local.keys():
                    mapped_col = _ranking_col_from_ahp_key(ahp_key)
                    if mapped_col == col:
                        return ahp_key
                return None

            def weights_for_source(source: str) -> np.ndarray:
                if source == "AHP weights":
                    ahp = st.session_state.get("ahp_weights")
                    if not ahp:
                        st.warning("AHP weights not found. Falling back to equal weights.")
                        return np.ones(len(selected_cols), dtype=float)
                    w = []
                    missing_cols = []
                    for c in selected_cols:
                        k = _ahp_key_for_ranking_col(c, ahp)
                        if k not in ahp:
                            missing_cols.append(label_by_col.get(c, c))
                            w.append(0.0)
                            continue
                        w.append(float(ahp[k]))
                    if missing_cols:
                        st.warning(
                            "AHP weights do not cover these selected ranking criteria, so they were given 0 weight: "
                            + ", ".join(missing_cols)
                        )
                    if np.array(w, dtype=float).sum() == 0:
                        st.warning("AHP weights do not overlap the selected ranking criteria. Falling back to equal weights.")
                        return np.ones(len(selected_cols), dtype=float)
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
                out = ranking_results.copy()
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


if active_main_tab == "Prioritization Database":
    render_data_tab()
elif active_main_tab == "Data Tools":
    render_data_tools_tab()
elif active_main_tab == "Spatial View":
    render_spatial_tab()
elif active_main_tab == "Parameter Analysis":
    render_parameter_analysis_tab()
elif active_main_tab == "Direct Weights":
    render_direct_weights_tab()
elif active_main_tab == "Pairwise Comparision (AHP)":
    render_ahp_tab()
elif active_main_tab == "Ranking":
    render_topsis_tab()

if str(st.session_state.get("auth_user", "")).strip().lower() == "aaloksk":
    if "show_access_history" not in st.session_state:
        st.session_state["show_access_history"] = False

    _, col_hist = st.columns([8, 2])
    with col_hist:
        if st.button("Access History", key="btn_access_history_main", use_container_width=True):
            st.session_state["show_access_history"] = not bool(st.session_state.get("show_access_history", False))

    if st.session_state.get("show_access_history", False):
        csp1, csp2, csp3 = st.columns([6, 2, 2])
        with csp3:
            if st.button("Clear Access History", key="btn_clear_access_history_main", use_container_width=True):
                _save_access_history([])
                st.session_state["show_access_history"] = False
                st.success("Access history permanently cleared.")
        records = _load_access_history()
        if records:
            hist_df = pd.DataFrame(records)
            preferred = ["username", "person_name", "started_at_utc", "last_seen_utc", "runtime_minutes", "session_id"]
            cols = [c for c in preferred if c in hist_df.columns] + [c for c in hist_df.columns if c not in preferred]
            hist_df = hist_df[cols]
            if "started_at_utc" in hist_df.columns:
                hist_df = hist_df.sort_values("started_at_utc", ascending=False)
            st.dataframe(hist_df, use_container_width=True, height=260)
        else:
            st.info("No access history found yet.")
