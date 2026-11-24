import hashlib
import threading
import time

from cachetools import TTLCache

from config.settings import get_logger, settings
from llm.client import FALLBACK_TEXT, GigaChatClient

logger = get_logger("service")

MIN_SUMMARY_LENGTH = 5
MAX_SUMMARY_LENGTH = 2000
FORBIDDEN_ARTIFACTS = ("```", "<|im_start|>", "<|im_end|>", "[INST]", "[/INST]")


def normalize_text(text: str) -> str:
    """Единая нормализация пробельных символов для ключа кеша и запроса в LLM."""
    return " ".join(text.strip().split())


class TextProcessor:
    def __init__(self, llm_client: GigaChatClient | None = None) -> None:
        self.llm = llm_client or GigaChatClient()
        self.cache: TTLCache = TTLCache(maxsize=settings.cache_maxsize, ttl=settings.cache_ttl)
        self._lock = threading.Lock()

    def _get_key(self, normalized_text: str) -> str:
        model = getattr(self.llm, "model", settings.gigachat_model)
        temperature = getattr(self.llm, "temperature", settings.llm_temperature)
        system_prompt = getattr(self.llm, "system_prompt", "")

        cache_material = f"{model}|{temperature}|{system_prompt}|{normalized_text}"
        return hashlib.sha256(cache_material.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_summary(summary: str, source_text: str) -> bool:
        if not summary or len(summary) < MIN_SUMMARY_LENGTH:
            return False
        if len(summary) > MAX_SUMMARY_LENGTH:
            return False
        if any(artifact in summary for artifact in FORBIDDEN_ARTIFACTS):
            return False
        if len(source_text) > 300 and len(summary) >= len(source_text) * 1.2:
            return False
        return True

    def process(self, text: str) -> dict:
        start_time = time.perf_counter()
        normalized_text = normalize_text(text)
        key = self._get_key(normalized_text)

        with self._lock:
            if key in self.cache:
                cached_result = self.cache[key]
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.info(
                    "Cache hit",
                    extra={"extra_fields": {"cached": True, "fallback": False, "duration_ms": duration_ms}},
                )
                return {"result": cached_result, "cached": True, "fallback": False}

        logger.info("Cache miss", extra={"extra_fields": {"cached": False}})
        raw_text, is_fallback = self.llm.summarize(normalized_text)
        cleaned_text = normalize_text(raw_text)

        if not is_fallback and not self._validate_summary(cleaned_text, normalized_text):
            logger.warning(
                "Ответ модели не прошёл валидацию качества, перевод в fallback",
                extra={"extra_fields": {"output_length": len(cleaned_text)}},
            )
            cleaned_text = FALLBACK_TEXT
            is_fallback = True

        if not is_fallback:
            with self._lock:
                self.cache[key] = cleaned_text

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Обработка текста завершена",
            extra={
                "extra_fields": {
                    "cached": False,
                    "fallback": is_fallback,
                    "duration_ms": duration_ms,
                }
            },
        )
        return {"result": cleaned_text, "cached": False, "fallback": is_fallback}
