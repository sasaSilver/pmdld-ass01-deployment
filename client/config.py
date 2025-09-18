from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    host: str = Field(validation_alias="API_HOST")
    port: int = Field(validation_alias="API_PORT")


config = Config()
