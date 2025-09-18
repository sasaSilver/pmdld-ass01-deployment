import requests
from config import config

class ApiClient:
    def __init__(self, api_host: str, api_port: int):
        self.base_url = f"http://{api_host}:{api_port}"

    def predict(self, image: bytes) -> float:
        files = {
            "image_file": ("image.jpg", image, "image/jpeg")
        }
        
        response = requests.post(self.base_url + "/predict", files=files)
        response.raise_for_status()
        return response.json()["prediction"]


api_client = ApiClient(config.host, config.port)
