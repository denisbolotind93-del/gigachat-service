import hashlib
from cachetools import TTLCache
from config.settings import get_logger, settings
from llm.client import GigaChatClient

logger = get_logger("service")


class TextProcessor:
    def __init__(self, llm_client: GigaChatClient | None = None) -> None:
        self.llm = llm_client or GigaChatClient()
        self.cache = TTLCache(maxsize=200, ttl=settings.cache_ttl)

    def _get_key(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def process(self, text: str) -> dict:
        key = self._get_key(text)

        if key in self.cache:
            logger.info("cache hit")
            return {"result": self.cache[key], "cached": True, "fallback": False}

        logger.info("cache miss")
        raw_text, is_fallback = self.llm.summarize(text)
        cleaned_text = " ".join(raw_text.strip().split())

        if not is_fallback:
            self.cache[key] = cleaned_text

        return {"result": cleaned_text, "cached": False, "fallback": is_fallback}
