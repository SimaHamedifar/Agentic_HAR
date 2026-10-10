"""MuRAL loading, feature engineering, and windowed sequence datasets for HAR.

The rename/remap/feature logic below is ported from
``data_preprocessing/MuRAL_preprocessing.ipynb`` (Sima's exploratory notebook)
so this module reuses the project's existing preprocessing decisions instead
of inventing new ones, with one deliberate correction (see
``SENSOR_RENAME_MAP``'s ``"living room couch mov"`` entry below) and one
addition (the inter-event time-delta feature) made after the first baseline
run showed they were needed. What is new here, because the notebook never
built it, is turning the flat per-house event table into fixed-length
sliding windows for sequence classification.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# --- Ported verbatim from MuRAL_preprocessing.ipynb -------------------------

SENSOR_RENAME_MAP = {
    "kitchen range hood wattmeter": "hood",
    "kitchen induction stove wattmeter": "stove",
    "kitchen coffee machine wattmeter": "coffee_machine",
    "kitchen microwave wattmeter": "microwave",
    "kitchen kettle wattmeter": "kettle",
    "kitchen counter mov": "kitchen",
    "kitchen cabinet door": "kitchen",
    "kitchen fridge door": "fridge_door",
    "kitchen sink mov": "kitchen",
    "main door": "main door",
    "dining room chair_1 contact": "dining_room",
    "dining room chair_2 contact": "dining_room",
    "dining room chair_3 contact": "dining_room",
    "dining room food cabinet door": "dining_room",
    "living room bookshelf mov": "living_room",
    "living room TV wattmeter": "TV",
    "living room game console wattmeter": "game_console",
    # The notebook maps this to "dining_room", which merges the single most
    # frequent sensor in the whole dataset (~45% of all events) into the same
    # location bucket as the three actual dining chairs and the food cabinet
    # door. That collapses watching TV / playing video games / resting /
    # chatting - all couch-triggered - into one indistinguishable feature.
    # Giving the couch its own location category fixes that conflation.
    "living room couch mov": "couch",
    "bathroom toilet contact": "bathroom",
    "bathroom global mov": "bathroom",
    "bathroom sink mov": "bathroom",
    "bedroom_1 door": "bedroom_1_door",
    "bedroom_2 door sensor": "bedroom_2_door",
    "nan": "unknown",
}

APPLIANCES = {
    "hood", "stove", "coffee_machine", "microwave", "kettle", "TV", "game_console", "unknown",
}
ROOMS = {"kitchen", "dining_room", "living_room", "bathroom", "couch"}
DOOR_SENSORS = {"fridge_door", "main door", "bedroom_1_door", "bedroom_2_door"}

VALID_SENSOR_STATUS = {"OPENED", "CLOSED", "turned ON", "turned OFF"}
SENSOR_STATUS_RENAME_MAP = {
    "OPENED": "ON",
    "CLOSED": "OFF",
    "turned ON": "ON",
    "turned OFF": "OFF",
    "nan": "unknown",
}

ACTIVITY_RENAME_MAP = {
    "using toilet": "using bathroom",
    "personal washing": "using bathroom",
    "preparing drinks": "grabing food or drinks",
    "grabing tableware": "grabing food or drinks",
    "clearing table": "grabing food or drinks",
    "treating fruits": "grabing food or drinks",
    "chatting": "chatting/using the cell-phone",
    "playing board games": "playing board games or puzzles",
}

# Fixed, known one-hot vocabularies for the categorical input features. These
# come from the sensor inventory above (a physical property of the MuRAL
# apartment, not something to be fit from a particular data split), so they
# are constants rather than dynamically discovered.
APPLIANCE_VALUES: tuple[str, ...] = tuple(sorted(APPLIANCES))
LOCATION_VALUES: tuple[str, ...] = tuple(sorted(ROOMS | {"unknown"}))
DOOR_SENSOR_VALUES: tuple[str, ...] = tuple(sorted(DOOR_SENSORS | {"unknown"}))
SENSOR_STATUS_VALUES: tuple[str, ...] = ("ON", "OFF", "unknown")

FEATURE_NAMES: tuple[str, ...] = (
    ("Hour_X", "Hour_Y", "time_delta")
    + tuple(f"appliance={v}" for v in APPLIANCE_VALUES)
    + tuple(f"location={v}" for v in LOCATION_VALUES)
    + tuple(f"door_sensor={v}" for v in DOOR_SENSOR_VALUES)
    + tuple(f"sensor_status={v}" for v in SENSOR_STATUS_VALUES)
)


def add_cyclical_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[timestamp_col], format="%H:%M:%S")
    fractional_hour = ts.dt.hour + (ts.dt.minute / 60.0) + (ts.dt.second / 3600.0)
    radians_24h = 2 * np.pi * (fractional_hour / 24.0)
    df["Hour_X"] = np.cos(radians_24h)
    df["Hour_Y"] = np.sin(radians_24h)
    return df


def add_time_delta_feature(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Add ``time_delta``: a scaled log of seconds since the previous event.

    Windows are built over a fixed *count* of events rather than a fixed
    *duration* (event arrivals are irregular, unlike NILM's fixed-rate
    samples), so without this feature the model cannot tell a burst of
    events a few seconds apart from the same count of events spread over
    many minutes. Assumes ``df`` holds one house's rows in chronological
    order (true for every MuRAL house.csv). The first row of a house has no
    predecessor, so its delta is 0.
    """
    ts = pd.to_datetime(df[timestamp_col], format="%H:%M:%S")
    seconds = ts.diff().dt.total_seconds().fillna(0.0)
    seconds = seconds.clip(lower=0.0)  # a negative delta would indicate a data error, not a real gap
    df["time_delta"] = np.log1p(seconds) / 10.0
    return df


def rename_sensor_column(df: pd.DataFrame, sensor_col: str = "sensor") -> pd.DataFrame:
    cleaned_sensors = df[sensor_col].astype(str).str.strip()
    df[sensor_col] = cleaned_sensors.replace(SENSOR_RENAME_MAP)
    return df


def split_sensor_into_three_columns(df: pd.DataFrame, sensor_col: str = "sensor") -> pd.DataFrame:
    s = df[sensor_col].astype(str).str.strip()
    df["appliance"] = s.apply(lambda x: x if x in APPLIANCES else "unknown")
    df["location"] = s.apply(lambda x: x if x in ROOMS else "unknown")
    df["door_sensor"] = s.apply(lambda x: x if x in DOOR_SENSORS else "unknown")
    df = df.drop(columns=[sensor_col])
    return df


def remap_activities(df: pd.DataFrame, activity_col: str = "activity") -> pd.DataFrame:
    df[activity_col] = df[activity_col].astype(str).str.strip().replace(ACTIVITY_RENAME_MAP)
    return df


def remap_sensor_status(df: pd.DataFrame, sensor_status_col: str = "sensor_status") -> pd.DataFrame:
    df[sensor_status_col] = (
        df[sensor_status_col].astype(str).str.strip().replace(SENSOR_STATUS_RENAME_MAP)
    )
    return df


# --- New: loading raw MuRAL houses and building the final feature table -----


def _house_ids(mural_dir: str | Path) -> list[str]:
    return sorted(
        name
        for name in os.listdir(mural_dir)
        if (Path(mural_dir) / name).is_dir()
    )


def load_activity_lookup(mural_dir: str | Path) -> dict[int, str]:
    activities = json.load(open(Path(mural_dir) / "activities.json", "r", encoding="utf-8"))
    return {int(entry["id"]): str(entry["name"]) for entry in activities}


def load_house_frame(house_dir: str | Path, activity_lookup: dict[int, str], house_id: str) -> pd.DataFrame:
    """Load one house's ``data.csv`` and apply the notebook's exact feature pipeline."""
    csv_path = Path(house_dir) / "data.csv"
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"action": "sensor_status"})

    # A small number of MuRAL houses contain a malformed trailing row with
    # every field blank (e.g. house 10). ``time`` is required to compute the
    # cyclical time features and is never legitimately missing, so rows
    # without it are dropped rather than silently propagating NaN.
    df = df[df["time"].notna() & (df["time"].astype(str).str.strip() != "")].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{csv_path} has no rows with a valid time value")

    df["activity"] = df["activity"].map(activity_lookup)
    if df["activity"].isna().any():
        raise ValueError(f"{csv_path} contains activity ids missing from activities.json")

    df = add_cyclical_time_features(df, timestamp_col="time")
    df = add_time_delta_feature(df, timestamp_col="time")
    df = rename_sensor_column(df, sensor_col="sensor")
    df = split_sensor_into_three_columns(df, sensor_col="sensor")

    known_status = df["sensor_status"].astype(str).str.strip()
    df["sensor_status"] = known_status.where(known_status.isin(VALID_SENSOR_STATUS), "nan")
    df = remap_sensor_status(df, sensor_status_col="sensor_status")
    df = remap_activities(df, activity_col="activity")

    df = df[["Hour_X", "Hour_Y", "time_delta", "appliance", "location", "door_sensor", "sensor_status", "activity"]].copy()
    df["house"] = house_id
    df["event_index"] = np.arange(len(df))
    return df.reset_index(drop=True)


def load_mural_dataset(mural_dir: str | Path) -> pd.DataFrame:
    """Load and feature-engineer every house under ``mural_dir`` into one table."""
    mural_dir = Path(mural_dir)
    activity_lookup = load_activity_lookup(mural_dir)
    frames = [
        load_house_frame(mural_dir / house_id, activity_lookup, house_id)
        for house_id in _house_ids(mural_dir)
    ]
    if not frames:
        raise ValueError(f"no MuRAL house folders found under {mural_dir}")
    return pd.concat(frames, ignore_index=True)


# --- Encoding categorical features and activity labels -----------------------


def _one_hot(value: str, vocabulary: Sequence[str]) -> np.ndarray:
    encoded = np.zeros(len(vocabulary), dtype=np.float32)
    encoded[vocabulary.index(value) if value in vocabulary else vocabulary.index("unknown")] = 1.0
    return encoded


def encode_features(df: pd.DataFrame) -> np.ndarray:
    """Encode the categorical + cyclical feature columns into a dense float matrix."""
    parts = [
        df[["Hour_X", "Hour_Y", "time_delta"]].to_numpy(dtype=np.float32),
        np.stack([_one_hot(v, APPLIANCE_VALUES) for v in df["appliance"]]),
        np.stack([_one_hot(v, LOCATION_VALUES) for v in df["location"]]),
        np.stack([_one_hot(v, DOOR_SENSOR_VALUES) for v in df["door_sensor"]]),
        np.stack([_one_hot(v, SENSOR_STATUS_VALUES) for v in df["sensor_status"]]),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


@dataclass(frozen=True)
class LabelEncoder:
    """Deterministic activity-name <-> integer-id mapping fit from observed data."""

    classes: tuple[str, ...]

    @classmethod
    def fit(cls, labels: Sequence[str]) -> "LabelEncoder":
        classes = tuple(sorted(set(labels)))
        if not classes:
            raise ValueError("cannot fit a label encoder on zero labels")
        return cls(classes)

    def transform(self, labels: Sequence[str]) -> np.ndarray:
        index = {name: i for i, name in enumerate(self.classes)}
        missing = set(labels) - index.keys()
        if missing:
            raise ValueError(f"labels not seen during fitting: {sorted(missing)}")
        return np.asarray([index[label] for label in labels], dtype=np.int64)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def build_windows(
    features: np.ndarray, labels: np.ndarray, house_ids: np.ndarray, window_size: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window (start, end) index pairs, one per house, never crossing houses."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if not (len(features) == len(labels) == len(house_ids)):
        raise ValueError("features, labels, and house_ids must have equal length")
    starts: list[int] = []
    for house in pd.unique(house_ids):
        house_positions = np.flatnonzero(house_ids == house)
        if house_positions.size < window_size:
            continue
        first, last = house_positions[0], house_positions[-1]
        if not np.array_equal(house_positions, np.arange(first, last + 1)):
            raise ValueError("house rows must be contiguous to build windows")
        last_start = last - window_size + 1
        starts.extend(range(first, last_start + 1, stride))
    return np.asarray(starts, dtype=np.int64), np.full(len(starts), window_size, dtype=np.int64)


class HARWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding windows of encoded MuRAL events, labeled by the most recent event's activity."""

    def __init__(self, features: np.ndarray, label_ids: np.ndarray, window_starts: np.ndarray, window_size: int) -> None:
        if features.ndim != 2:
            raise ValueError("features must be two-dimensional [num_events, num_feature_dims]")
        if window_starts.size == 0:
            raise ValueError("no windows were produced for this split")
        self.features = features
        self.label_ids = label_ids
        self.window_starts = window_starts
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.window_starts[index])
        end = start + self.window_size
        window = self.features[start:end]
        label = self.label_ids[end - 1]
        return torch.from_numpy(window.copy()), torch.tensor(label, dtype=torch.long)

    def class_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        last_labels = self.label_ids[self.window_starts + self.window_size - 1]
        for class_id in last_labels:
            counts[class_id] += 1
        return counts
