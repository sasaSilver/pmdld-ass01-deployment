from fastapi.testclient import TestClient

from app.config import config
from app.__main__ import app
from app.models.dataset import get_data_loaders
from app.models.model import model

client = TestClient(app)


def load_model():
    global model
    model.load(config.best_model_path)
    assert model._model is not None, "Model could not be loaded"


def evaluate_model():
    global model
    if model._model is None:
        load_model()
    train_loader, test_loader = get_data_loaders()
    model.evaluate(test_loader)


def test_predict_endpoint():
    global model
    if model._model is None:
        load_model()
    image_file = ("test.jpg", b"test", "image/jpeg")
    response = client.post("/predict", files={"image_file": image_file})
    response.raise_for_status()
    assert response.json().get("prediction", None) is not None, "Prediction is None"


if __name__ == "__main__":
    load_model()
    print("Model loaded")
    test_predict_endpoint()
    print("Prediction endpoint tested")
    evaluate_model()
    print("Model evaluated")
