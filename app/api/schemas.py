from pydantic import BaseModel


class Prediction(BaseModel):
    prediction: float
