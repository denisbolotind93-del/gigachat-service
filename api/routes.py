from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, StrictStr, field_validator

from config.settings import get_logger
from llm.client import LLMConfigurationError
from services.processor import TextProcessor, normalize_text

router = APIRouter()
logger = get_logger("api")


@lru_cache
def get_text_processor() -> TextProcessor:
    return TextProcessor()


class TextRequest(BaseModel):
    text: StrictStr = Field(
        ...,
        min_length=15,
        max_length=5000,
        description="Исходный текст для суммаризации (от 15 до 5000 символов после очистки пробелов).",
    )

    @field_validator("text")
    @classmethod
    def validate_and_normalize_text(cls, value: str) -> str:
        cleaned = normalize_text(value)
        if len(cleaned) < 15:
            raise ValueError("Текст после удаления лишних пробелов должен содержать минимум 15 символов.")
        if len(cleaned) > 5000:
            raise ValueError("Текст не должен превышать 5000 символов.")
        return cleaned


class TextResponse(BaseModel):
    result: str
    cached: bool
    fallback: bool


@router.post("/process", response_model=TextResponse)
def process_endpoint(
    req: TextRequest,
    processor: TextProcessor = Depends(get_text_processor),
) -> TextResponse:
    logger.info(
        "Получен запрос POST /process",
        extra={"extra_fields": {"text_length": len(req.text)}},
    )
    try:
        data = processor.process(req.text)
    except LLMConfigurationError as exc:
        logger.error("Сервис не может обработать запрос из-за ошибки конфигурации LLM")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис суммаризации временно недоступен из-за ошибки конфигурации провайдера.",
        ) from exc

    return TextResponse(**data)


@router.get("/health")
def health_live() -> dict[str, str]:
    """Liveness-проверка: подтверждает, что HTTP-сервер запущен и отвечает."""
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready(processor: TextProcessor = Depends(get_text_processor)) -> dict[str, str]:
    """Readiness-проверка: оценивает готовность интеграции с GigaChat."""
    if not processor.llm.check_readiness():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис не готов: проверьте GIGACHAT_CREDENTIALS или доступность GigaChat API.",
        )
    return {"status": "ready"}
