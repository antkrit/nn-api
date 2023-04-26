"""Contains pydantic schemas implementations."""
# pylint: disable=no-name-in-module
from typing import List, Union

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


class BaseJSONLog(BaseModel):
    """Main JSON log format."""

    thread: Union[int, str]
    level_name: str
    message: str
    source: str
    timestamp: str = Field(..., alias="@timestamp")
    app_version: str
    app_env: str
    duration: int
    exceptions: Union[list[str], str] = None

    class Config:  # pylint: disable=missing-class-docstring
        allow_population_by_field_name = True


class APIRequestJSONLogSchema(BaseModel):
    """JSON log of request (response)."""

    request_uri: str
    request_protocol: str
    request_method: str
    request_path: str
    request_host: str
    request_size: int
    request_content_type: str
    request_headers: str
    request_body: str
    remote_ip: str
    remote_port: str
    response_status_code: int
    response_size: int
    response_headers: str
    response_body: str
    duration: int
