from .config import config
from .api.routes import model_router
from .models.model import model

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model.load(config.best_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(model_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
