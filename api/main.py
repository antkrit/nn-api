"""Contains main app."""
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.v1 import app as apiv1

app = FastAPI()

API_V1_PATH = "/api/v1"
app.mount(API_V1_PATH, apiv1)


@app.get("/")
async def root():
    """Get root."""
    return RedirectResponse(url=API_V1_PATH + "/docs")
