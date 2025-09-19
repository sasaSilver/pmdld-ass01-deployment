import requests
from config import config


class ApiClient:
    def __init__(self, api_host: str, api_port: int):
        self.base_url = f"http://{api_host}:{api_port}"

    def predict(self, image_file: tuple[str, bytes, str]) -> float:
        response = requests.post(
            f"{self.base_url}/predict", files={"image_file": image_file}
        )
        response.raise_for_status()
        return response.json()["prediction"]


api_client = ApiClient(config.host, config.port)
