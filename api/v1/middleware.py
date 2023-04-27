"""Contains FastAPI middlewares"""
import http
import json
import math
import time

from fastapi import Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from api.utils.logging import EMPTY_VALUE
from api.v1.schemas import APIRequestJSONLogSchema


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to process logs."""

    def __init__(self, app, *, logger) -> None:
        self._logger = logger
        super().__init__(app)

    async def dispatch(self, request, call_next):
        # pylint: disable=too-many-locals
        start_time = time.time()
        exception_object = None

        # REQUEST
        try:
            raw_request_body = await request.body()

            await self.set_body(request, raw_request_body)
            raw_request_body = await self.get_body(request)

            request_body = raw_request_body.decode()
        except Exception:
            request_body = EMPTY_VALUE

        server = request.get("server", ("localhost", 8000))
        request_headers = dict(request.headers.items())

        # RESPONSE
        try:
            response = await call_next(request)
        except Exception as exc:
            response_body = bytes(status.HTTP_500_INTERNAL_SERVER_ERROR)

            response = Response(
                content=response_body,
                status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.real,
            )

            exception_object = exc
            response_headers = {}
        else:
            response_headers = dict(response.headers.items())
            response_body = EMPTY_VALUE

            async for chunk in response.body_iterator:
                response_body += chunk

            response = Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response_headers),
                media_type=response.media_type,
            )

        duration = math.ceil((time.time() - start_time) * 1000)

        api_request_log = APIRequestJSONLogSchema(
            request_uri=str(request.url),
            request_protocol=await self.get_protocol(request),
            request_method=request.method,
            request_path=request.url.path,
            request_host=f"{server[0]}:{server[1]}",
            request_size=int(request_headers.get("content-length", 0)),
            request_content_type=request_headers.get(
                "content-type", EMPTY_VALUE
            ),
            request_headers=json.dumps(request_headers),
            request_body=request_body,
            remote_ip=request.client[0],
            remote_port=request.client[1],
            response_status_code=response.status_code,
            response_size=int(response_headers.get("content-length", 0)),
            response_headers=json.dumps(response_headers),
            response_body=response_body,
            duration=duration,
        ).dict()

        message = (
            f'{"Error" if exception_object else "Response"} '
            f"with code {response.status_code} "
            f'on request {request.method} "{str(request.url)}". '
            f"Duration: {duration}ms"
        )

        self._logger.info(
            message,
            extra={"api_request_log": api_request_log},
            exc_info=exception_object,
        )

        return response

    @staticmethod
    async def get_protocol(request):
        """Get request protocol."""
        protocol = str(request.scope.get("type", ""))
        http_version = str(request.scope.get("http_version", ""))

        if protocol.lower() == "http" and http_version:
            return f"{protocol.upper()}/{http_version}"

    @staticmethod
    async def set_body(request, body):
        """Set request body."""

        async def receive():
            nonlocal body
            return {"type": "http.request", "body": body}

        # pylint: disable=protected-access
        request._receive = receive

    async def get_body(self, request):
        """Get request body."""
        if request.headers["Content-Type"] == "application/json":
            body = await request.body()
            await self.set_body(request, body)
        else:
            body = EMPTY_VALUE

        return body
