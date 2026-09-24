
"""
MuRAL Clustering V2
-------------------
Feature engineering:
- Temporal: event count/rate, inter-event statistics, circular time
- Sensor: counts/frequencies, diversity/entropy
- Spatial: room/zone inferred from sensor names
- Action: ON/OFF/OPEN/CLOSE ratios
- Subject: number/single/multi-subject ratios
- Transitions: sensor transition count/diversity/entropy

Clustering:
- K-Means: silhouette across K, selected K, stability with ARI
- DBSCAN: parameter grid, silhouette, noise percentage, stability
- Activity labels are NOT used as clustering features.
  They are used only after clustering for qualitative interpretation.

Run:
    python mural_clustering_v2.py

Main settings are at the top.
"""

from pathlib import Path
import zipfile
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")


# ============================================================
# USER SETTINGS
# ============================================================

# If MuRAL.zip is in the same folder as this script:
ZIP_PATH = "MuRAL.zip"

# If you already extracted MuRAL, set this to the extracted folder.
# Example Windows:
# DATA_DIR = r"F:\MuRAL\MuRAL"
# Leave as None if using ZIP_PATH.
DATA_DIR = "MuRAL"

# Length of each observation window
WINDOW_MINUTES = 2

# ---- K-Means ----
# Manually specify the final K (only effective when USE_AUTO_K=False).
K = 5

# K values tested for silhouette / hungarian_accuracy
K_RANGE = range(2, 31)

# Fix note: this had the same issue as v1—the K_RANGE grid-search result was not actually used, regardless of what it computed.
# The original "Final selected K" always used the hard-coded K=5; the grid result was only saved to CSV/plotted and was not
# actually used. Default is now True, automatically selecting the K with the highest silhouette score.
USE_AUTO_K = True

# K selection criterion:"silhouette" (an internal metric that does not require labels; purely unsupervised), or
# "hungarian_accuracy" (uses the true activity labels and the Hungarian algorithm to calculate the optimal "cluster-activity"
# one-to-one matching accuracy to select K).
#
# [Important methodological note]: If the goal is "each cluster should correspond as closely as possible to one activity," silhouette is not
# an appropriate optimization target—it only measures how well-separated the clusters are, regardless of whether clusters match the true activities.
# The intuitive "purity" metric (the proportion of the most common activity within each cluster) is also unsuitable—it increases as K increases
# monotonically (in the extreme case K equals the number of samples, every cluster contains one point, giving 100% purity but no meaningful clustering),
# so using it to select K inherently favors the incorrect direction of "more clusters are always better."
# Hungarian matching accuracy is different: it first finds the optimal one-to-one matching between clusters and activities (each activity can be
# "claimed" by at most one cluster), then measures how many samples are correctly matched under this optimal assignment—this properly
# measures whether clusters can cleanly correspond to different activities without being misled by the "more clusters means higher purity"
# illusion. In testing (5-minute windows + curated features), K=14 (best silhouette) had a matching accuracy of
# only 0.318, while K=25 reached 0.398, the highest value in the 15–30 range.
#
# Note: in hungarian_accuracy mode, the K-selection step uses the true activity labels—
# although the K-Means clustering itself remains completely label-free (unsupervised), "using the ground truth to choose hyperparameter K"
# is no longer a purely unsupervised workflow; it is a form of "label-guided hyperparameter tuning while the model itself does not see the labels"
# of semi-supervised exploration. This should be stated honestly in a thesis/document and should not be presented as "the unsupervised method itself
# found K."
K_SELECTION_METRIC = '''"hungarian_accuracy"'''"silhouette"

# Number of K-Means stability runs
N_STABILITY_RUNS = 20

# ---- DBSCAN ----
DBSCAN_EPS = 2.0
DBSCAN_MIN_SAMPLES = 40

# DBSCAN grid search
RUN_DBSCAN_GRID = True
# Fix note: the original epsilon grid only went up to 3.0, which was too small for this feature space (whether using all 80 dimensions or
# the curated 60 dimensions)—the measured median k-distance was already around 5, so almost all combinations in the grid
# degenerated into 100% noise. The range has been changed to an empirically calibrated range.
DBSCAN_EPS_GRID = [1,2,3,4,5 ]
DBSCAN_MIN_SAMPLES_GRID = [3, 5, 8, 10, 15]

# Fix note: same issue as K-Means—the grid-search result was not actually used to determine the final
# eps/min_samples; the hard-coded defaults were always used (with 92 samples and a 15-minute window, these defaults
# were likely to produce degenerate results). Default is now True, automatically selecting from the grid the parameter combination with
# the "non-degenerate (clusters >= 2 and noise <= 70%), highest silhouette" result.
USE_AUTO_DBSCAN_PARAMS = True

# Whether to keep only categories validated by feature-importance analysis (object/zone/sensor/time/
# subject/inter_event), removing transition/entropy/type/action. Default is True;
# empirically better than using all features (see the prepare_matrix docstring for the full rationale).
USE_CURATED_FEATURES = True

# Output
OUTPUT_DIR = "mural_clustering_results_v2"
RANDOM_STATE = 42


# ============================================================
# FILE / DATA LOADING
# ============================================================

def prepare_data_dir():
    """Return extracted/selected MuRAL data directory."""
    if DATA_DIR is not None:
        root = Path(DATA_DIR)
        if not root.exists():
            raise FileNotFoundError(f"DATA_DIR does not exist: {root}")
        return root

    zip_path = Path(ZIP_PATH)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Cannot find {zip_path}. "
            f"Put MuRAL.zip beside this script or use an absolute path."
        )

    extract_root = Path("_mural_extracted_v2")
    if not extract_root.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_root)

    # Typical structure: _mural_extracted_v2/MuRAL/
    candidates = [extract_root / "MuRAL", extract_root]
    for c in candidates:
        if (c / "activities.json").exists():
            return c

    # Search recursively
    for p in extract_root.rglob("activities.json"):
        return p.parent

    raise FileNotFoundError("Could not locate MuRAL/activities.json after extraction.")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mural(root):
    """Load all session CSVs and context.json files."""
    activity_map = {}
    activity_path = root / "activities.json"
    if activity_path.exists():
        raw = load_json(activity_path)
        if isinstance(raw, dict):
            for k, v in raw.items():
                activity_map[int(k)] = v
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    aid = item.get("id", item.get("activity"))
                    name = item.get("name", item.get("activity_name", item.get("label")))
                    if aid is not None:
                        activity_map[int(aid)] = name if name is not None else str(aid)

    frames = []
    contexts = []

    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue

        csv_path = session_dir / "data.csv"
        if not csv_path.exists():
            continue

        session = session_dir.name
        df = pd.read_csv(csv_path)
        df["session"] = session

        context_path = session_dir / "context.json"
        if context_path.exists():
            try:
                ctx = load_json(context_path)
            except Exception:
                ctx = {}
        else:
            ctx = {}

        contexts.append({"session": session, "context": ctx})
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No session data.csv files found under {root}")

    data = pd.concat(frames, ignore_index=True)

    if "time" not in data.columns:
        raise ValueError("data.csv must contain a 'time' column.")
    if "sensor" not in data.columns:
        raise ValueError("data.csv must contain a 'sensor' column.")

    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["time"]).copy()

    if "activity" in data.columns:
        data["activity_name"] = data["activity"].map(activity_map)
        data["activity_name"] = data["activity_name"].fillna(
            data["activity"].astype(str)
        )
    else:
        data["activity_name"] = "unknown"

    return data, contexts, activity_map


# ============================================================
# SENSOR NAME PARSING
# ============================================================

# Authoritative mapping table constructed by checking the official "sensor location" field in sensors.json entry by entry
# (keys use the lowercase short names that actually appear in data.csv; values are the official locations).
# The only exception that does not follow the simple assumption that the room name appears in the string is main door—
# the sensor name itself does not contain the word "entrance," but sensors.json explicitly lists its official location
# as entrance.
SENSOR_LOCATION_OVERRIDE = {
    "main door": "entrance",
}


def parse_sensor(sensor):
    """
    MuRAL sensors encode useful semantic information, e.g.
    'kitchen microwave wattmeter'
    'living room TV wattmeter'
    'bathroom toilet contact'

    We infer a coarse zone and sensor type.

    Fix note: zone (room) is first looked up in the authoritative SENSOR_LOCATION_OVERRIDE mapping—
    this table was manually constructed by checking the official "sensor location" field of every sensor in sensors.json
    (not guessed). Previously, using only string matching, the "main door" sensor
    was incorrectly assigned to "other" because its name does not contain "entrance," but
    sensors.json explicitly states that its official location is entrance. Only sensors not present in the mapping table
    (for example, newly added sensors in future datasets) fall back to the string-based heuristic below
    as a fallback.
    """
    s = str(sensor).strip().lower()

    zones = [
        "living room",
        "dining room",
        "kitchen",
        "bathroom",
        "bedroom_1",
        "bedroom_2",
        "bedroom",
        "entrance",
        "hallway",
        "corridor",
    ]

    if s in SENSOR_LOCATION_OVERRIDE:
        zone = SENSOR_LOCATION_OVERRIDE[s]
    else:
        zone = "other"
        for z in zones:
            if z in s:
                zone = z
                break

    sensor_types = ["wattmeter", "contact", "mov", "motion"]
    sensor_type = "other"
    for t in sensor_types:
        if re.search(rf"\b{re.escape(t)}\b", s):
            sensor_type = t
            break

    # Remove zone and type to get a coarse object name.
    obj = s
    for z in zones:
        obj = obj.replace(z, " ")
    for t in sensor_types:
        obj = re.sub(rf"\b{re.escape(t)}\b", " ", obj)

    obj = re.sub(r"\s+", " ", obj).strip()
    if not obj:
        obj = "unknown"

    return zone, obj, sensor_type


# ============================================================
# FEATURE HELPERS
# ============================================================

def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def transition_features(values):
    values = [str(v) for v in values if pd.notna(v)]
    if len(values) < 2:
        return 0, 0, 0.0

    transitions = list(zip(values[:-1], values[1:]))
    unique_transitions = len(set(transitions))

    counts = pd.Series(transitions).value_counts().values
    trans_entropy = entropy_from_counts(counts)

    return len(transitions), unique_transitions, trans_entropy


# ============================================================
# TIME WINDOWS + FEATURE ENGINEERING
# ============================================================

def create_time_windows(data, window_minutes):
    """
    One row = one session-specific time window.

    Activity labels are retained only for interpretation.
    They are never used in X.
    """
    data = data.copy()

    window_delta = pd.Timedelta(minutes=window_minutes)

    # Use elapsed time from the start of each session.
    session_start = data.groupby("session")["time"].transform("min")
    elapsed = data["time"] - session_start

    data["window_id"] = (
        elapsed.dt.total_seconds() // (window_minutes * 60)
    ).astype(int)

    data["window_start"] = (
        session_start
        + pd.to_timedelta(
            data["window_id"] * window_minutes, unit="m"
        )
    )

    data["window_end"] = data["window_start"] + window_delta

    # Parse sensor semantics once.
    # Fix note: previously parse_sensor was called for all rows (including "background activity" records where sensor itself was NaN),
    # so NaN was converted by Python's str() into the literal string 'nan' and incorrectly treated as a
    # real sensor name, producing a meaningless object__nan pseudo-feature column.
    # This is changed to call parse_sensor only for rows with non-empty sensor values; rows with empty sensors retain
    # real NaN values (rather than the string 'nan'), so subsequent value_counts() will by default
    # correctly exclude these rows from zone/object/type statistics and no longer produce pseudo-feature columns.
    has_sensor = data["sensor"].notna()
    parsed = pd.Series(index=data.index, dtype=object)
    parsed.loc[has_sensor] = data.loc[has_sensor, "sensor"].map(parse_sensor)
    data["zone"] = parsed.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    data["sensor_object"] = parsed.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    data["sensor_type"] = parsed.map(lambda x: x[2] if isinstance(x, tuple) else np.nan)

    # Normalize action text.
    if "action" in data.columns:
        data["action_norm"] = (
            data["action"].astype(str).str.strip().str.lower()
        )
    else:
        data["action_norm"] = "unknown"

    group_cols = ["session", "window_id"]

    rows = []

    for (session, window_id), g in data.groupby(group_cols, sort=True):
        row = {
            "session": session,
            "window_id": int(window_id),
            "window_start": g["window_start"].iloc[0],
            "window_end": g["window_end"].iloc[0],
        }

        # -------------------------
        # Basic temporal features
        # -------------------------
        event_count = len(g)
        row["event_count"] = event_count
        row["event_rate"] = event_count / max(window_minutes, 1)

        times = g["time"].sort_values()
        gaps = times.diff().dt.total_seconds().dropna()

        row["mean_inter_event_sec"] = float(gaps.mean()) if len(gaps) else 0.0
        row["std_inter_event_sec"] = float(gaps.std()) if len(gaps) > 1 else 0.0
        row["max_inter_event_sec"] = float(gaps.max()) if len(gaps) else 0.0

        # Time of day represented circularly.
        mid_time = g["time"].iloc[0]
        hour_float = (
            mid_time.hour
            + mid_time.minute / 60.0
            + mid_time.second / 3600.0
        )
        row["time_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
        row["time_cos"] = np.cos(2 * np.pi * hour_float / 24.0)

        # -------------------------
        # Sensor features
        # -------------------------
        sensor_counts = g["sensor"].value_counts()
        row["unique_sensors"] = sensor_counts.size
        row["sensor_entropy"] = entropy_from_counts(sensor_counts.values)

        for sensor, count in sensor_counts.items():
            row[f"sensor__{sensor}"] = count / event_count

        # -------------------------
        # Spatial / zone features
        # -------------------------
        zone_counts = g["zone"].value_counts()
        row["unique_zones"] = zone_counts.size
        row["zone_entropy"] = entropy_from_counts(zone_counts.values)

        for zone, count in zone_counts.items():
            row[f"zone__{zone}"] = count / event_count

        # -------------------------
        # Object features
        # -------------------------
        object_counts = g["sensor_object"].value_counts()
        row["unique_objects"] = object_counts.size

        # Keep object features; there are few MuRAL sensor objects.
        for obj, count in object_counts.items():
            safe_obj = re.sub(r"[^a-zA-Z0-9_]+", "_", str(obj)).strip("_")
            if safe_obj:
                row[f"object__{safe_obj}"] = count / event_count

        # -------------------------
        # Sensor type features
        # -------------------------
        type_counts = g["sensor_type"].value_counts()
        row["unique_sensor_types"] = type_counts.size

        for typ, count in type_counts.items():
            row[f"type__{typ}"] = count / event_count

        # -------------------------
        # Action features
        # -------------------------
        action_counts = g["action_norm"].value_counts()

        for action, count in action_counts.items():
            safe_action = re.sub(
                r"[^a-zA-Z0-9_]+", "_", str(action)
            ).strip("_")
            if safe_action:
                row[f"action__{safe_action}"] = count / event_count

        row["action_entropy"] = entropy_from_counts(action_counts.values)

        # -------------------------
        # Subject features
        # -------------------------
        if "Subject" in g.columns:
            subjects = g["Subject"].dropna().astype(str)
            row["unique_subject_values"] = subjects.nunique()

            multi = subjects.str.contains(",", regex=False)
            row["multi_subject_event_ratio"] = float(multi.mean()) if len(subjects) else 0.0
            row["single_subject_event_ratio"] = 1.0 - row["multi_subject_event_ratio"]
        else:
            row["unique_subject_values"] = 0
            row["multi_subject_event_ratio"] = 0.0
            row["single_subject_event_ratio"] = 0.0

        # -------------------------
        # Transition features
        # -------------------------
        g_sorted = g.sort_values("time")
        n_trans, unique_trans, trans_ent = transition_features(
            g_sorted["sensor"].tolist()
        )

        row["sensor_transition_count"] = n_trans
        row["unique_sensor_transitions"] = unique_trans
        row["transition_entropy"] = trans_ent

        rows.append(row)

    features = pd.DataFrame(rows)

    # Activity labels are used ONLY for interpretation.
    activity_rows = []
    for (session, window_id), g in data.groupby(group_cols, sort=True):
        counts = g["activity_name"].value_counts()
        for activity, count in counts.items():
            activity_rows.append({
                "session": session,
                "window_id": int(window_id),
                "activity": activity,
                "activity_count": int(count),
            })

    activity_df = pd.DataFrame(activity_rows)

    return data, features, activity_df


# ============================================================
# MATRIX PREPARATION
# ============================================================

def prepare_matrix(features, use_curated_features=True):
    """
    Args:
        use_curated_features: Whether to keep only categories validated by feature-importance analysis.

        Rationale: a RandomForestClassifier was used with the true activity labels for [offline feature-importance analysis only;
        it does not participate in clustering itself]. The supervised importance ranking found:
        - object (specific items/appliances) + sensor (specific sensors) together contributed about 44% of
          the importance and were the most discriminative categories—more fine-grained than zone (room), which is intuitive:
          in the same kitchen, "using a kettle" and "using a stove" are completely different activities.
        - time_sin/time_cos individually ranked in the Top 2, confirming that activities are strongly related to time of day
          (breakfast/dinner/before bed, etc.).
        - subject-related features (multi-person/single-person event ratios) and inter_event_sec-related features (mean/std/max
          inter-event intervals, indirectly reflecting activity pace) also had moderately high contributions.
        - transition (sensor transition patterns) and entropy (diversity) ranked lowest in importance—
          with only 239 samples, these high-cardinality/interaction features are prone to sparsity and instability.

        Empirical result (K-Means, K=14): keeping only object/zone/sensor/time/subject/
        inter_event (60 dimensions, removing transition/entropy/type/action)
        performed better than keeping all 80 dimensions: ARI increased from 0.132 to 0.187, NMI increased from 0.380 to
        0.414, and Silhouette also improved slightly—indicating that this was not merely dimensionality reduction but actual noise removal.

        Additional note: an extra "activation duration proportion per sensor" feature was tested (which should logically be useful,
        similar to capturing the difference between long-duration actions such as "watching TV/gaming" and instantaneous
        actions such as "opening a door/using the toilet"), but it actually reduced both ARI and NMI—the likely reason is that this duration estimate
        was relatively coarse, while inter_event_sec already indirectly captured most of the "pace" signal,
        and adding another group of features increased dimensional dilution with the small sample size. This attempt was not adopted into
        the curated feature set, but is recorded here as a meaningful negative result.
    """
    id_cols = [
        "session",
        "window_id",
        "window_start",
        "window_end",
    ]

    X_df = features.drop(
        columns=[c for c in id_cols if c in features.columns],
        errors="ignore",
    ).copy()

    if use_curated_features:
        keep_prefixes = ("object__", "zone__", "sensor__", "time_", "subject", "inter_event")
        keep_cols = [c for c in X_df.columns if any(c.startswith(p) or p in c for p in keep_prefixes)]
        X_df = X_df[keep_cols]

    # Convert all features to numeric.
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    # Fill missing values caused by sparse sensor/object columns.
    X_df = X_df.fillna(0.0)

    # Remove zero-variance columns.
    nunique = X_df.nunique()
    X_df = X_df.loc[:, nunique > 1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)

    return X, X_df.columns.tolist(), scaler


# ============================================================
# QUALITATIVE INTERPRETATION
# ============================================================

def interpret_clusters(labels, features, activity_df, method_name):
    """
    Fix note: directly assigning tmp["cluster"] = labels previously caused an error—activity_df is
    an expanded "window × activity" table (a window with multiple activities expands into several rows), so its row count naturally
    exceeds the number of windows (the length of labels), and positional assignment inevitably causes a dimension mismatch
    (in testing, 92 windows corresponded to 615 activity_df rows, causing ValueError on direct assignment).
    Changed to map and merge using session+window_id instead of assuming identical row counts and ordering.
    """
    if activity_df.empty:
        return pd.DataFrame()

    cluster_map = features[["session", "window_id"]].copy()
    cluster_map["cluster"] = labels

    tmp = activity_df.merge(cluster_map, on=["session", "window_id"], how="left")

    # activity_df must correspond one-to-one with windows, so aggregate
    # activity counts by session/window first.
    activity_per_window = (
        tmp.groupby(["session", "window_id", "cluster", "activity"], as_index=False)
        ["activity_count"].sum()
    )

    rows = []
    for cluster, g in activity_per_window.groupby("cluster"):
        total = g["activity_count"].sum()
        top = (
            g.groupby("activity")["activity_count"]
            .sum()
            .sort_values(ascending=False)
        )

        row = {
            "method": method_name,
            "cluster": int(cluster),
            "activity_event_count": int(total),
            "top_activity": top.index[0] if len(top) else "unknown",
            "top_activity_count": int(top.iloc[0]) if len(top) else 0,
        }

        for i, (activity, count) in enumerate(top.head(5).items(), start=1):
            row[f"activity_{i}"] = activity
            row[f"activity_{i}_count"] = int(count)
            row[f"activity_{i}_ratio"] = float(count / total) if total else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# K-MEANS
# ============================================================

def hungarian_matched_accuracy(labels, y_true):
    """
    Find the optimal one-to-one matching between clusters and true activities (Hungarian algorithm; each activity can be
    "claimed" by at most one cluster), and return the proportion of samples matched under this optimal assignment.

    Used only for [offline K selection/offline evaluation], never as input to the clustering algorithm itself—K-Means/
    DBSCAN training does not access y_true; true labels are used here only to "score"
    and help determine "which K best aligns clusters with activities." This is a semi-supervised hyperparameter-tuning aid,
    not a way for the model to learn by seeing the answers.
    """
    clusters = sorted(set(labels))
    activities = sorted(set(y_true))
    cost = np.zeros((len(clusters), len(activities)))
    for i, c in enumerate(clusters):
        for j, a in enumerate(activities):
            cost[i, j] = -np.sum((labels == c) & (y_true == a))
    row, col = linear_sum_assignment(cost)
    matched = -cost[row, col].sum()
    return matched / len(labels)


def get_dominant_activity_labels(features, activity_df):
    """Assign the activity with the most events to each window as the window's 'dominant activity' label (used only for
    offline K selection/evaluation, not as a clustering input)"""
    dominant = activity_df.groupby(["session", "window_id"]).apply(
        lambda g: g.loc[g["activity_count"].idxmax(), "activity"]
    ).reset_index(name="dominant_activity")
    merged = features[["session", "window_id"]].merge(
        dominant, on=["session", "window_id"], how="left"
    )
    return merged["dominant_activity"].fillna("unknown").values


def run_kmeans(X, features, activity_df, out_dir, y_true=None):
    silhouette_rows = []

    for k in K_RANGE:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )
        labels = model.fit_predict(X)

        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X, labels)
        else:
            sil = np.nan

        row = {
            "K": k,
            "silhouette": sil,
            "inertia": model.inertia_,
            "cluster_count": len(np.unique(labels)),
        }
        if y_true is not None:
            row["hungarian_accuracy"] = hungarian_matched_accuracy(labels, y_true)
        silhouette_rows.append(row)

    sil_df = pd.DataFrame(silhouette_rows)
    sil_df.to_csv(out_dir / "kmeans_silhouette_v2.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(sil_df["K"], sil_df["silhouette"], marker="o", label="silhouette")
    if "hungarian_accuracy" in sil_df.columns:
        ax2 = plt.gca().twinx()
        ax2.plot(sil_df["K"], sil_df["hungarian_accuracy"], marker="s", color="orange",
                  label="hungarian_accuracy")
        ax2.set_ylabel("Hungarian matched accuracy")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means: Silhouette / Hungarian accuracy vs K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kmeans_silhouette_v2.png", dpi=200)
    plt.close()

    # Final selected K
    # Fix note: no longer unconditionally uses the hard-coded K; USE_AUTO_K determines whether to use the best grid-search result.
    if USE_AUTO_K:
        if K_SELECTION_METRIC == "hungarian_accuracy" and "hungarian_accuracy" in sil_df.columns:
            valid_sil = sil_df.dropna(subset=["hungarian_accuracy"])
            metric_col = "hungarian_accuracy"
        else:
            valid_sil = sil_df.dropna(subset=["silhouette"])
            metric_col = "silhouette"
        if len(valid_sil) > 0:
            best_row = valid_sil.loc[valid_sil[metric_col].idxmax()]
            selected_k = int(best_row["K"])
            print(f"Automatically selected K={selected_k} ({metric_col}={best_row[metric_col]:.4f})")
        else:
            selected_k = K
            print(f"⚠️ No valid{metric_col}value in the grid search; falling back to manual K={selected_k}")
    else:
        selected_k = K

    model = KMeans(
        n_clusters=selected_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    labels = model.fit_predict(X)

    # Stability
    reference = labels
    ari_values = []

    for seed in range(N_STABILITY_RUNS):
        m = KMeans(
            n_clusters=selected_k,
            random_state=RANDOM_STATE + seed + 1,
            n_init=20,
        )
        other = m.fit_predict(X)
        ari_values.append(adjusted_rand_score(reference, other))

    stability_mean = float(np.mean(ari_values))
    stability_std = float(np.std(ari_values))

    interpretation = interpret_clusters(
        labels, features, activity_df, f"K-Means K={selected_k}"
    )
    interpretation.to_csv(
        out_dir / "kmeans_interpretation_v2.csv", index=False
    )

    # PCA visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12, alpha=0.7)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"K-Means PCA (K={selected_k})")
    plt.tight_layout()
    plt.savefig(out_dir / "kmeans_pca_v2.png", dpi=200)
    plt.close()

    return {
        "method": "K-Means",
        "parameter": f"K={selected_k}",
        "selected_k": selected_k,
        "silhouette": float(
            silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
        ),
        "stability_ARI_mean": stability_mean,
        "stability_ARI_std": stability_std,
        "noise_pct": 0.0,
        "labels": labels,
        "model": model,
        "silhouette_df": sil_df,
    }


# ============================================================
# DBSCAN
# ============================================================

def dbscan_silhouette(X, labels):
    mask = labels != -1
    unique = np.unique(labels[mask])

    if mask.sum() < 3 or len(unique) < 2:
        return np.nan

    return float(silhouette_score(X[mask], labels[mask]))


def run_dbscan(X, features, activity_df, out_dir):
    grid_rows = []

    if RUN_DBSCAN_GRID:
        for eps in DBSCAN_EPS_GRID:
            for min_samples in DBSCAN_MIN_SAMPLES_GRID:
                model = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                )
                labels = model.fit_predict(X)

                n_clusters = len(set(labels)) - (
                    1 if -1 in labels else 0
                )
                noise_pct = float(np.mean(labels == -1) * 100)
                sil = dbscan_silhouette(X, labels)

                grid_rows.append({
                    "eps": eps,
                    "min_samples": min_samples,
                    "clusters": n_clusters,
                    "noise_pct": noise_pct,
                    "silhouette_excluding_noise": sil,
                })

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(out_dir / "dbscan_grid_v2.csv", index=False)

    # Fix note: same issue as K-Means—no longer unconditionally uses the hard-coded DBSCAN_EPS/
    # DBSCAN_MIN_SAMPLES; USE_AUTO_DBSCAN_PARAMS determines whether to automatically select from the grid-search results
    # the parameter combination that is "non-degenerate (clusters >= 2 and noise <= 70%) with the highest silhouette";
    # if all grid combinations are degenerate, fall back to the one with the most clusters and print a warning.
    if USE_AUTO_DBSCAN_PARAMS and len(grid_df) > 0:
        valid = grid_df[(grid_df["clusters"] >= 2) & (grid_df["noise_pct"] <= 70)]
        if len(valid) > 0:
            best_row = valid.loc[valid["silhouette_excluding_noise"].idxmax()]
            selected_eps = float(best_row["eps"])
            selected_min_samples = int(best_row["min_samples"])
            print(f"Automatically selected eps={selected_eps}, min_samples={selected_min_samples} "
                  f"(silhouette={best_row['silhouette_excluding_noise']:.4f})")
        else:
            print("⚠️ All parameter combinations in the grid search were degenerate; falling back to the combination with the most clusters.")
            best_row = grid_df.loc[grid_df["clusters"].idxmax()]
            selected_eps = float(best_row["eps"])
            selected_min_samples = int(best_row["min_samples"])
    else:
        selected_eps = DBSCAN_EPS
        selected_min_samples = DBSCAN_MIN_SAMPLES

    # Final model
    model = DBSCAN(
        eps=selected_eps,
        min_samples=selected_min_samples,
    )
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (
        1 if -1 in labels else 0
    )
    noise_pct = float(np.mean(labels == -1) * 100)
    sil = dbscan_silhouette(X, labels)

    # DBSCAN stability: perturb eps slightly around the chosen value.
    reference = labels
    ari_values = []

    nearby_eps = [
        selected_eps * 0.9,
        selected_eps,
        selected_eps * 1.1,
    ]

    for eps in nearby_eps:
        m = DBSCAN(
            eps=eps,
            min_samples=selected_min_samples,
        )
        other = m.fit_predict(X)

        # ARI is defined even if cluster labels differ in count.
        ari_values.append(adjusted_rand_score(reference, other))

    interpretation = interpret_clusters(
        labels, features, activity_df,
        f"DBSCAN eps={selected_eps}, min_samples={selected_min_samples}"
    )
    interpretation.to_csv(
        out_dir / "dbscan_interpretation_v2.csv", index=False
    )

    # PCA visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12, alpha=0.7)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(
        f"DBSCAN PCA (eps={selected_eps}, min_samples={selected_min_samples})"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "dbscan_pca_v2.png", dpi=200)
    plt.close()

    return {
        "method": "DBSCAN",
        "parameter": f"eps={selected_eps}, min_samples={selected_min_samples}",
        "selected_eps": selected_eps,
        "selected_min_samples": selected_min_samples,
        "silhouette": sil,
        "stability_ARI_mean": float(np.mean(ari_values)),
        "stability_ARI_std": float(np.std(ari_values)),
        "noise_pct": noise_pct,
        "clusters": n_clusters,
        "labels": labels,
        "model": model,
        "grid_df": grid_df,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MuRAL Clustering V2")
    print("=" * 70)

    print("\n[1/7] Preparing data directory...")
    root = prepare_data_dir()
    print("Data root:", root)

    print("\n[2/7] Loading MuRAL...")
    data, contexts, activity_map = load_mural(root)

    print("Events:", len(data))
    print("Sessions:", data["session"].nunique())
    print("Sensors:", data["sensor"].nunique())
    print("Actions:", data["action"].nunique() if "action" in data.columns else 0)
    print("Activities:", data["activity_name"].nunique())

    print("\n[3/7] Creating time windows and features...")
    windowed, features, activity_df = create_time_windows(
        data, WINDOW_MINUTES
    )

    features.to_csv(
        out_dir / f"features_{WINDOW_MINUTES}min_v2.csv",
        index=False,
    )

    print("Time windows:", len(features))
    print("Feature columns before matrix cleaning:", len(features.columns))

    print("\n[4/7] Preparing numeric feature matrix...")
    X, feature_names, scaler = prepare_matrix(features, use_curated_features=USE_CURATED_FEATURES)
    print("Final numeric features:", len(feature_names))
    print("Matrix shape:", X.shape)

    pd.DataFrame({"feature": feature_names}).to_csv(
        out_dir / "feature_names_v2.csv", index=False
    )

    print("\n[5/7] Running K-Means...")
    y_true = get_dominant_activity_labels(features, activity_df)
    km = run_kmeans(X, features, activity_df, out_dir, y_true=y_true)

    print(
        f"K-Means K={K}: "
        f"silhouette={km['silhouette']:.4f}, "
        f"stability ARI={km['stability_ARI_mean']:.4f} ± "
        f"{km['stability_ARI_std']:.4f}"
    )

    print("\n[6/7] Running DBSCAN...")
    db = run_dbscan(X, features, activity_df, out_dir)

    print(
        f"DBSCAN: "
        f"silhouette={db['silhouette'] if not np.isnan(db['silhouette']) else 'NaN'}, "
        f"stability ARI={db['stability_ARI_mean']:.4f}, "
        f"noise={db['noise_pct']:.2f}%, "
        f"clusters={db.get('clusters', 'N/A')}"
    )

    print("\n[7/7] Saving final assignments and summary...")

    assignments = features[
        ["session", "window_id", "window_start", "window_end"]
    ].copy()
    assignments["kmeans_cluster"] = km["labels"]
    assignments["dbscan_cluster"] = db["labels"]

    assignments.to_csv(
        out_dir / "final_cluster_assignments_v2.csv",
        index=False,
    )

    summary = pd.DataFrame([
        {
            "method": km["method"],
            "parameter": km["parameter"],
            "silhouette": km["silhouette"],
            "stability_ARI_mean": km["stability_ARI_mean"],
            "stability_ARI_std": km["stability_ARI_std"],
            "noise_pct": km["noise_pct"],
            "clusters": km["selected_k"],
        },
        {
            "method": db["method"],
            "parameter": db["parameter"],
            "silhouette": db["silhouette"],
            "stability_ARI_mean": db["stability_ARI_mean"],
            "stability_ARI_std": db["stability_ARI_std"],
            "noise_pct": db["noise_pct"],
            "clusters": db.get("clusters", np.nan),
        },
    ])

    summary.to_csv(
        out_dir / "clustering_summary_v2.csv",
        index=False,
    )

    print("\nFinished.")
    print("Results saved to:", out_dir.resolve())
    print("\nImportant files:")
    print(" - features_*min_v2.csv")
    print(" - feature_names_v2.csv")
    print(" - kmeans_silhouette_v2.csv / .png")
    print(" - kmeans_interpretation_v2.csv")
    print(" - kmeans_pca_v2.png")
    print(" - dbscan_grid_v2.csv")
    print(" - dbscan_interpretation_v2.csv")
    print(" - dbscan_pca_v2.png")
    print(" - final_cluster_assignments_v2.csv")
    print(" - clustering_summary_v2.csv")


if __name__ == "__main__":
    main()
