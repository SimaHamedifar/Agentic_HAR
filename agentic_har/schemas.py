"""Structured activity output and the shared API input contract."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SensorValue = Annotated[str, Field(min_length=1, max_length=100)]
ClockTime = Annotated[str, Field(pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$")]
WINDOW_SIZE = 10


class SensorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: ClockTime
    sensor_status: SensorValue
    appliance: SensorValue = "unknown"
    location: SensorValue = "unknown"
    door_sensor: SensorValue = "unknown"


class HistoricalEvent(SensorEvent):
    activity: SensorValue | None = None
    time_delta: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class RecognitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    household_id: str = Field(min_length=1, max_length=80)
    history: list[HistoricalEvent] = Field(min_length=WINDOW_SIZE, max_length=WINDOW_SIZE)
    event: SensorEvent


class ActivityOutput(BaseModel):
    activity: int = Field(ge=0, le=21)


class RecognitionResponse(ActivityOutput):
    label: str
    model_top_3: dict[str, float]
