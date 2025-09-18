from fastapi import APIRouter, UploadFile, File, HTTPException
from ..models.model import model
from .schemas import Prediction

model_router = APIRouter()

@model_router.post("/predict")
async def predict(image_file: UploadFile = File()):
    if not image_file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    image_bytes = await image_file.read()
    
    if not image_bytes:
        raise HTTPException(
            status_code=400, 
            detail="Empty file received"
        )
    
    score = model.predict(image_bytes)
    
    return Prediction(prediction=round(score, 4))