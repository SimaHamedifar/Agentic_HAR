"""Build and load per-house memory maps from Sima's merged REFIT CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from nilm.config import NILMConfig
from nilm.preprocessing import HouseSeries


MANIFEST_NAME = "manifest.json"
_BIN_COLUMN = "__time_bin"
_TIMESTAMP_SECONDS = "__timestamp_seconds"


def _required_columns(config: NILMConfig) -> list[str]:
    return [config.timestamp_column, config.house_column, *config.feature_columns, *config.appliance_columns]


def _clean_chunk(frame: pd.DataFrame, config: NILMConfig) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned[config.timestamp_column] = pd.to_datetime(cleaned[config.timestamp_column], errors="coerce", utc=True)
    cleaned[config.house_column] = pd.to_numeric(cleaned[config.house_column], errors="coerce")
    for column in (*config.feature_columns, *config.appliance_columns):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    valid = cleaned[config.timestamp_column].notna() & cleaned[config.house_column].notna()
    valid &= cleaned[list(config.feature_columns)].notna().all(axis=1)
    valid &= np.isfinite(cleaned[list(config.feature_columns)].to_numpy(dtype=np.float64)).all(axis=1)
    cleaned = cleaned.loc[valid]
    cleaned = cleaned[cleaned[config.house_column].isin(config.all_houses)]
    return cleaned


def _iter_source_chunks(config: NILMConfig, chunk_size: int) -> Iterable[pd.DataFrame]:
    header = pd.read_csv(config.data_file, nrows=0).columns
    missing = [name for name in _required_columns(config) if name not in header]
    if missing:
        raise ValueError(f"merged CSV is missing required columns: {missing}")
    for frame in pd.read_csv(config.data_file, usecols=_required_columns(config), chunksize=chunk_size):
        cleaned = _clean_chunk(frame, config)
        if not cleaned.empty:
            yield cleaned


def _aggregate_bins(frame: pd.DataFrame, config: NILMConfig) -> pd.DataFrame:
    interval_ns = int(config.expected_interval_seconds * 1_000_000_000)
    working = frame.copy()
    working[_BIN_COLUMN] = working[config.timestamp_column].astype("int64") // interval_ns
    values = [*config.feature_columns, *config.appliance_columns]
    grouped = working.groupby([config.house_column, _BIN_COLUMN], as_index=False, sort=True)[values].mean()
    grouped[_TIMESTAMP_SECONDS] = grouped[_BIN_COLUMN].to_numpy(dtype=np.float64) * config.expected_interval_seconds
    return grouped


def _iter_resampled_chunks(config: NILMConfig, chunk_size: int) -> Iterable[pd.DataFrame]:
    """Yield complete fixed-width bins while retaining chunk-edge bins as carry."""
    pending: pd.DataFrame | None = None
    interval_ns = int(config.expected_interval_seconds * 1_000_000_000)
    for chunk in _iter_source_chunks(config, chunk_size):
        if pending is not None and not pending.empty:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        bins = chunk[config.timestamp_column].astype("int64") // interval_ns
        last_bin = bins.groupby(chunk[config.house_column]).transform("max")
        pending = chunk.loc[bins == last_bin].copy()
        complete = chunk.loc[bins != last_bin]
        if not complete.empty:
            yield _aggregate_bins(complete, config)
    if pending is not None and not pending.empty:
        yield _aggregate_bins(pending, config)


def build_cache(config: NILMConfig, chunk_size: int = 250_000, overwrite: bool = False) -> dict:
    """Two-pass conversion keeps peak RAM bounded by one CSV chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source_stat = config.data_file.stat()
    cache_dir = config.cache_dir
    manifest_path = cache_dir / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"cache already exists: {manifest_path}; pass --overwrite to rebuild")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in cache_dir.glob("house_*_*.npy"):
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    counts = {house: 0 for house in config.all_houses}
    for chunk in _iter_resampled_chunks(config, chunk_size):
        chunk_counts = chunk[config.house_column].astype(int).value_counts()
        for house, value in chunk_counts.items():
            limit = config.max_rows_per_house
            available = int(value) if limit is None else max(0, min(int(value), limit - counts[int(house)]))
            counts[int(house)] += available
    missing_houses = [house for house, count in counts.items() if count == 0]
    if missing_houses:
        raise ValueError(f"no valid rows found for configured houses: {missing_houses}")

    arrays: dict[int, tuple[np.memmap, np.memmap, np.memmap]] = {}
    for house, count in counts.items():
        prefix = cache_dir / f"house_{house}"
        arrays[house] = (
            np.lib.format.open_memmap(f"{prefix}_features.npy", mode="w+", dtype=np.float32, shape=(count, len(config.feature_columns))),
            np.lib.format.open_memmap(f"{prefix}_targets.npy", mode="w+", dtype=np.float32, shape=(count, len(config.appliance_columns))),
            np.lib.format.open_memmap(f"{prefix}_timestamps.npy", mode="w+", dtype=np.float64, shape=(count,)),
        )
    positions = {house: 0 for house in counts}
    for chunk in _iter_resampled_chunks(config, chunk_size):
        houses = chunk[config.house_column].astype(int)
        for house in counts:
            subset = chunk.loc[houses == house]
            if subset.empty:
                continue
            remaining = counts[house] - positions[house]
            subset = subset.iloc[:remaining]
            if subset.empty:
                continue
            start, stop = positions[house], positions[house] + len(subset)
            features, targets, timestamps = arrays[house]
            features[start:stop] = subset[list(config.feature_columns)].to_numpy(dtype=np.float32)
            targets[start:stop] = subset[list(config.appliance_columns)].to_numpy(dtype=np.float32)
            timestamps[start:stop] = subset[_TIMESTAMP_SECONDS].to_numpy(dtype=np.float64)
            positions[house] = stop
    for group in arrays.values():
        for array in group:
            array.flush()
    incomplete = {house: {"expected": counts[house], "written": positions[house]} for house in counts if positions[house] != counts[house]}
    if incomplete:
        raise RuntimeError(f"cache row counts changed between passes: {incomplete}")
    final_stat = config.data_file.stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise RuntimeError("source CSV changed while the cache was being built; rebuild from a stable file")
    manifest = {
        "format_version": 1,
        "source": str(config.data_file),
        "source_size_bytes": final_stat.st_size,
        "source_mtime_ns": final_stat.st_mtime_ns,
        "feature_columns": list(config.feature_columns),
        "appliance_columns": list(config.appliance_columns),
        "timestamp_column": config.timestamp_column,
        "house_column": config.house_column,
        "resample_interval_seconds": config.expected_interval_seconds,
        "row_counts": {str(house): count for house, count in counts.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_cached_houses(cache_dir: str | Path, houses: Sequence[int], mmap_mode: str | None = "r", expected_config: NILMConfig | None = None) -> list[HouseSeries]:
    directory = Path(cache_dir)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"NILM cache is missing {manifest_path}; run python -m nilm.cache first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_config is not None:
        expected = {
            "feature_columns": list(expected_config.feature_columns),
            "appliance_columns": list(expected_config.appliance_columns),
            "resample_interval_seconds": expected_config.expected_interval_seconds,
            "source": str(expected_config.data_file),
            "source_size_bytes": expected_config.data_file.stat().st_size,
            "source_mtime_ns": expected_config.data_file.stat().st_mtime_ns,
        }
        mismatched = [name for name, value in expected.items() if manifest.get(name) != value]
        if mismatched:
            raise ValueError(f"cache contract does not match config/source for: {mismatched}; rebuild the cache")
    result = []
    for house in houses:
        prefix = directory / f"house_{int(house)}"
        paths = [Path(f"{prefix}_features.npy"), Path(f"{prefix}_targets.npy"), Path(f"{prefix}_timestamps.npy")]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"cache files are missing for house {house}")
        result.append(HouseSeries(np.load(paths[0], mmap_mode=mmap_mode), np.load(paths[1], mmap_mode=mmap_mode), np.load(paths[2], mmap_mode=mmap_mode), f"H{house}", str(manifest.get("source", "cache"))))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build per-house NILM cache from the merged REFIT CSV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_cache(NILMConfig.from_json(args.config), args.chunk_size, args.overwrite)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
