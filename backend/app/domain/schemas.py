from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class NormalizedRoi(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class RoiSettings(BaseModel):
    processing: NormalizedRoi | None = None
    apriltag: NormalizedRoi | None = None


class ItemClassificationConfirmation(BaseModel):
    index: int = Field(ge=1)
    label: str = Field(min_length=1, max_length=120)
    learn: bool = False


class ClassificationConfirmation(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=120)
    learn: bool = False
    items: list[ItemClassificationConfirmation] | None = None


class TareCaptureRequest(BaseModel):
    mode: Literal["pledge", "release"]


class JobState(BaseModel):
    status: Literal["idle", "queued", "running", "complete", "failed"] = "idle"
    stage: str = ""
    message: str = ""
    percent: int = Field(default=0, ge=0, le=100)
    seconds_remaining: int = Field(default=0, ge=0)
    error: str | None = None


class CaptureSummary(BaseModel):
    id: str
    captured_at: str
    weight_g: float | None
    classification: dict[str, Any] = {}
    result: dict[str, Any] = {}
    job: JobState = JobState()
