"""Utilities to work with logging."""
import datetime
import json
import logging
import traceback

from api import __version__
from api.v1.config import settings
from api.v1.schemas import BaseJSONLog

EMPTY_VALUE = b""
JSON_INDENT = 2


class JSONLogFormatter(logging.Formatter):
    """Formatter for JSON-like logs."""

    def format(self, record):
        """Convert log object to json."""
        log_object = self._format_log_object(record)
        return json.dumps(log_object, ensure_ascii=False, indent=JSON_INDENT)

    @staticmethod
    def _format_log_object(record):
        """Фdd the required fields to the log."""
        now = (
            datetime.datetime.fromtimestamp(record.created)
            .astimezone()
            .replace(microsecond=0)
            .isoformat()
        )

        message = record.getMessage()

        if hasattr(record, "duration"):
            duration = record.duration
        else:
            duration = record.msecs

        json_log_fields = BaseJSONLog(
            thread=record.process,
            timestamp=now,
            level_name=logging.getLevelName(record.levelno),
            message=message,
            source=record.name,
            duration=duration,
            app_version=__version__,
            app_env=settings.get("ENVIRONMENT", "UNKNOWN"),
        )

        if record.exc_info:
            json_log_fields.exceptions = traceback.format_exception(
                *record.exc_info
            )
        elif record.exc_text:
            json_log_fields.exceptions = record.exc_text

        json_log_object = json_log_fields.dict(
            exclude_unset=True, by_alias=True
        )

        if hasattr(record, "api_request_log"):
            json_log_object.update(record.api_request_log)

        return json_log_object


def write_log(msg):
    """Asynclog helper."""
    print(msg)
