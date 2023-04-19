"""Contains main app."""
from fastapi import FastAPI

from api.v1 import app as api_v1

app = FastAPI()
app.mount("/api/v1", api_v1)


@app.get("/")
async def root():
    """Get root."""
    return {"message": "get an API v1.0 docs on '/api/v1/docs'"}
