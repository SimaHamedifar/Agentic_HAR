"""Construct labeled, house-bounded replay examples from MuRAL."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .schemas import HistoricalEvent, RecognitionRequest, SensorEvent, WINDOW_SIZE


def replay_examples(
    csv_path: str | Path, house: str, household_id: str, *, labeled_history: bool = False,
) -> Iterator[tuple[RecognitionRequest, str]]:
    frame = pd.read_csv(csv_path, dtype={"house": str})
    rows = frame.loc[frame["house"] == house].sort_values("event_index")
    if len(rows) <= WINDOW_SIZE:
        raise ValueError(f"House {house!r} needs more than {WINDOW_SIZE} events")
    sensor_fields = list(SensorEvent.model_fields)
    history_fields = [*sensor_fields, "time_delta"]
    if labeled_history:
        history_fields.append("activity")
    events = [HistoricalEvent.model_validate(row) for row in rows[history_fields].to_dict("records")]
    targets = rows[[*sensor_fields, "activity"]].to_dict("records")
    for i in range(WINDOW_SIZE, len(events)):
        target = targets[i]
        yield RecognitionRequest(
            household_id=household_id,
            history=events[i - WINDOW_SIZE:i],
            event=SensorEvent.model_validate({key: target[key] for key in sensor_fields}),
        ), target["activity"]
