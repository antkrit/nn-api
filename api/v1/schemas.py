"""Contains pydantic schemas implementations."""
# pylint: disable=no-name-in-module
from typing import List

from pydantic import BaseModel, Field


class Task(BaseModel):
    """Task schema."""

    task_id: str
    status: str


class PredictionEntry(BaseModel):
    """Single :class:`Prediction` result."""

    class_: int = Field(alias="class")
    probability: str = Field(alias="probability")


class Prediction(BaseModel):
    """Model prediction schema."""

    task_id: str
    status: str
    result: List[PredictionEntry]
