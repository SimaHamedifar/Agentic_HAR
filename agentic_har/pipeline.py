"""Convert MuRAL-format sensor observations into recognition evidence."""

import pandas as pd

from .activity_ml.preprocessing import (
    FEATURE_NAMES, add_cyclical_time_features, add_time_delta_feature, encode_features,
)
from .labels import ACTIVITY_MAP
from .schemas import HistoricalEvent, RecognitionRequest, RecognitionResponse, SensorEvent, WINDOW_SIZE


def describe_event(event: SensorEvent) -> str:
    sensor = "unknown sensor"
    for field in ("appliance", "location", "door_sensor"):
        if (value := getattr(event, field)) != "unknown":
            sensor = value
    text = f"At time {event.time}, {sensor} activated with status '{event.sensor_status}'."
    if isinstance(event, HistoricalEvent) and event.activity is not None:
        text += f" Detected activity: {event.activity}."
    return text


def encode_history(history: list[HistoricalEvent]):
    frame = pd.DataFrame([item.model_dump() for item in history])
    supplied_deltas = frame["time_delta"].copy()
    add_cyclical_time_features(frame, "time")
    add_time_delta_feature(frame, "time")
    frame["time_delta"] = supplied_deltas.fillna(frame["time_delta"])
    return encode_features(frame)


class RecognitionPipeline:
    def __init__(self, predictor, graph):
        if predictor.window_size != WINDOW_SIZE or predictor.feature_names != FEATURE_NAMES:
            raise ValueError("Checkpoint window size or feature ordering is incompatible with the API")
        self.predictor = predictor
        self.graph = graph

    def recognize(self, request: RecognitionRequest) -> RecognitionResponse:
        model_result = self.predictor.predict_activity(encode_history(request.history))
        scores = model_result["probabilities"]
        result = self.graph.invoke({
            "household_id": request.household_id,
            "event_sequence": "\n".join(map(describe_event, request.history)),
            "event": describe_event(request.event),
            "har_output": scores,
        })
        activity = result["activity"]
        return RecognitionResponse(activity=activity, label=ACTIVITY_MAP[str(activity)], model_top_3=scores)
