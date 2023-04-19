"""App router implementation.

Routes:
- (POST) '/predict': predict result for the input data
- (POST) '/{task_id}': get task status/result
"""
from celery.result import AsyncResult
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.celery_service.tasks import predict_task
from api.v1.schemas import InputData, Prediction, Task

model_router = APIRouter(
    prefix="/model", tags=["model"]  # prefix can be changed to match your model
)


@model_router.post("/predict", response_model=Task, status_code=202)
async def predict(input_data: InputData):
    """Predict result for the input data."""
    task_id = predict_task.delay(input_data.dict())
    return {"task_id": str(task_id), "status": "Processing"}


@model_router.post(
    "/{task_id}",
    response_model=Prediction,
    status_code=200,
    responses={202: {"model": Task, "description": "Accepted: Not Ready"}},
)
async def result(task_id):
    """Get task result."""
    task = AsyncResult(task_id)

    if not task.ready():
        return JSONResponse(
            status_code=202,
            content={"task_id": str(task_id), "status": "Processing"},
        )

    output = task.get()
    return {"task_id": task_id, "status": "Success", "result": str(output)}
