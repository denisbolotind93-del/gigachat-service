import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class Settings(BaseSettings):
    app_env: Literal["dev", "prod", "test"] = "dev"
    gigachat_credentials: str = Field(..., min_length=1)
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat"
    gigachat_verify_ssl_certs: bool = True
    gigachat_ca_bundle_file: str | None = None

    llm_timeout: float = Field(default=15.0, gt=0, le=120.0)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_retry_delay: float = Field(default=0.5, ge=0.0, le=10.0)

    cache_ttl: int = Field(default=300, gt=0)
    cache_maxsize: int = Field(default=200, gt=0)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("gigachat_ca_bundle_file", mode="before")
    @classmethod
    def empty_str_to_none(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    @property
    def log_level(self) -> int:
        return logging.INFO if self.is_prod else logging.DEBUG


settings = Settings()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "env": settings.app_env,
            "request_id": request_id_ctx.get(),
            "message": record.getMessage(),
        }

        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            log_payload.update(extra_fields)

        if record.exc_info and record.exc_info is not None:
            exc_type, exc_value, _ = record.exc_info
            log_payload["error_type"] = exc_type.__name__
            if not settings.is_prod:
                log_payload["traceback"] = self.formatException(record.exc_info)
            else:
                log_payload["error_detail"] = str(exc_value)

        return json.dumps(log_payload, ensure_ascii=False)


def get_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
