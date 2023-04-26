"""App router implementation.

Routes:
- (POST) '/predict': predict result for the input data
- (POST) '/{task_id}': get task status/result
"""
import logging

from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from api.utils.data import decode_mnist_model_output
from api.utils.image import ALLOWED_IMAGE_EXT, read_imagefile
from api.v1.schemas import Prediction, Task
from api.v1.tasks import predict_task

logger = logging.getLogger(__name__)

model_router = APIRouter()


@model_router.post(
    "/predict",
    response_model=Task,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create task",
    description="""
    Read and pre-process the uploaded image, schedule the prediction
    task with Celery.
    """,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "detail": "Unable to delay `predict` task."
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "detail": "Extension of the provided file is not allowed."
        },
    },
)
async def predict(file: UploadFile = File(...)):
    """Predict result for the input data."""
    extension = file.filename.split(".")[-1]
    ext_allowed = extension in ALLOWED_IMAGE_EXT

    if not ext_allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Extension of the provided file is not allowed:"
            f" `{extension}`.",
        )

    try:
        data = read_imagefile(await file.read())
        task_id = predict_task.delay(data)

        return {"task_id": str(task_id), "status": "Processing"}
    except ValueError as exc:
        detail = f"Unable to delay `predict` task: {str(exc)}"
        logger.error(msg=detail, exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        ) from exc


@model_router.get(
    "/{task_id}",
    response_model=Prediction,
    status_code=status.HTTP_200_OK,
    summary="Get task by id",
    description="""
    Get task output by its unique identifier: if ready - return the result,
    otherwise return the status of the task.
    """,
    responses={
        status.HTTP_202_ACCEPTED: {
            "model": Task,
            "description": "Accepted: Not Ready",
        }
    },
)
async def result(task_id: str, limit: int = 1):
    """Get task result."""
    task = AsyncResult(task_id)

    if not task.ready():
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"task_id": task_id, "status": task.status},
        )

    prediction = decode_mnist_model_output(task.result, n_entries=limit)

    response = []
    for idx, probability in prediction.items():
        response_entry = {
            "class": idx,
            "probability": f"{probability*100:0.2f}%",
        }
        response.append(response_entry)

    return {"task_id": task_id, "status": task.status, "result": response}
