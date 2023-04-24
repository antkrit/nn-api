"""Contains main app."""
from fastapi import FastAPI

from api import __version__
from api.v1 import app as api_v1

app = FastAPI()

API_V1_PATH = "/api/v1"
app.mount(API_V1_PATH, api_v1)


@app.get("/")
async def root():
    """Get root."""
    return {"message": f"get API v{__version__} docs on '{API_V1_PATH}/docs'"}
