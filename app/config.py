from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    host: str = Field(validation_alias="API_HOST")
    port: int = Field(validation_alias="API_PORT")
    data_base_path: str = "dataset/data"
    model_path: str = "app/models/saved/model.pth"

config = Config()
