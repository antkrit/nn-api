"""Contains pydantic schemas implementations."""
# pylint: disable=no-name-in-module
from pydantic import BaseModel


class Task(BaseModel):
    """Task schema."""

    task_id: str
    status: str


class InputData(BaseModel):
    """Model input schema."""

    x: int
    y: int


class Prediction(BaseModel):
    """Model prediction schema."""

    task_id: str
    status: str
    result: list
