import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from gigachat.exceptions import AuthenticationError

from api.routes import get_text_processor
from llm.client import FALLBACK_TEXT, GigaChatClient
from main import app
from services.processor import TextProcessor


@pytest.fixture
def isolated_processor():
    llm_client = GigaChatClient()
    processor = TextProcessor(llm_client=llm_client)
    app.dependency_overrides[get_text_processor] = lambda: processor
    yield processor
    app.dependency_overrides.clear()


@pytest.fixture
def client(isolated_processor):
    return TestClient(app)


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "Привет"},
        {"text": "                    "},
        {"text": "\n\t   \n   \t  "},
        {"text": "а" * 5001},
        {"text": 12345678901234567890},
        {},
    ],
)
def test_validation_errors(client, payload):
    res = client.post("/process", json=payload)
    assert res.status_code == 422


def test_success_normalization_and_cache(client, isolated_processor):
    with patch.object(
        isolated_processor.llm, "summarize", return_value=("Краткий   итог.", False)
    ) as mock_summarize:
        r1 = client.post(
            "/process",
            json={"text": "   Длинный текст   для проверки работы суммаризатора и кэширования.  "},
        )
        assert r1.status_code == 200
        assert r1.json() == {"result": "Краткий итог.", "cached": False, "fallback": False}
        assert "X-Request-ID" in r1.headers
        assert mock_summarize.call_count == 1
        mock_summarize.assert_called_with("Длинный текст для проверки работы суммаризатора и кэширования.")

        r2 = client.post(
            "/process",
            json={"text": "Длинный текст для проверки работы суммаризатора и кэширования."},
        )
        assert r2.status_code == 200
        assert r2.json() == {"result": "Краткий итог.", "cached": True, "fallback": False}
        assert mock_summarize.call_count == 1


def test_different_texts_do_not_collide_in_cache(client, isolated_processor):
    with patch.object(
        isolated_processor.llm,
        "summarize",
        side_effect=[("Пересказ первого.", False), ("Пересказ второго.", False)],
    ) as mock_summarize:
        r1 = client.post("/process", json={"text": "Первый уникальный текст достаточной длины для теста."})
        r2 = client.post("/process", json={"text": "Второй уникальный текст достаточной длины для теста."})

        assert r1.json()["result"] == "Пересказ первого."
        assert r2.json()["result"] == "Пересказ второго."
        assert r2.json()["cached"] is False
        assert mock_summarize.call_count == 2


def test_cache_ttl_expiration(client, isolated_processor):
    from cachetools import TTLCache

    isolated_processor.cache = TTLCache(maxsize=10, ttl=0.1)

    with patch.object(
        isolated_processor.llm,
        "summarize",
        side_effect=[("Первый ответ.", False), ("Обновлённый ответ.", False)],
    ) as mock_summarize:
        payload = {"text": "Текст для проверки истечения времени жизни записи в кеше."}

        r1 = client.post("/process", json=payload)
        assert r1.json()["cached"] is False

        time.sleep(0.15)

        r2 = client.post("/process", json=payload)
        assert r2.json()["cached"] is False
        assert r2.json()["result"] == "Обновлённый ответ."
        assert mock_summarize.call_count == 2


def test_fallback_on_timeout_and_not_cached(client, isolated_processor):
    payload = {"text": "Текст для проверки сбоя сети и отсутствия кеширования fallback."}

    with patch("llm.client.GigaChat", side_effect=httpx.TimeoutException("Request timed out")):
        r1 = client.post("/process", json=payload)
        assert r1.status_code == 200
        assert r1.json() == {"result": FALLBACK_TEXT, "cached": False, "fallback": True}

    with patch.object(isolated_processor.llm, "summarize", return_value=("Успешный пересказ.", False)):
        r2 = client.post("/process", json=payload)
        assert r2.status_code == 200
        assert r2.json() == {"result": "Успешный пересказ.", "cached": False, "fallback": False}


def test_auth_error_returns_503(client):
    with patch("llm.client.GigaChat", side_effect=AuthenticationError("Invalid token")):
        res = client.post("/process", json={"text": "Текст при ошибке авторизации во внешнем сервисе."})
        assert res.status_code == 503


@pytest.mark.parametrize("bad_model_output", ["", "   ", "ок", "```json\n{}\n```"])
def test_invalid_model_output_triggers_fallback_and_not_cached(client, isolated_processor, bad_model_output):
    mock_giga = MagicMock()
    mock_giga.__enter__.return_value.chat.return_value.choices = [
        MagicMock(message=MagicMock(content=bad_model_output))
    ]

    with patch("llm.client.GigaChat", return_value=mock_giga):
        res = client.post("/process", json={"text": "Текст для проверки некорректного ответа от модели."})
        assert res.status_code == 200
        assert res.json()["fallback"] is True
        assert res.json()["result"] == FALLBACK_TEXT
        assert len(isolated_processor.cache) == 0


def test_concurrent_requests_thread_safety(client, isolated_processor):
    def slow_summarize(text: str):
        time.sleep(0.05)
        return "Параллельный результат.", False

    with patch.object(isolated_processor.llm, "summarize", side_effect=slow_summarize):
        payload = {"text": "Текст для проверки потокобезопасности кеша при нагрузке."}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(client.post, "/process", json=payload) for _ in range(6)]
            responses = [f.result() for f in futures]

        assert all(r.status_code == 200 for r in responses)
        assert all(r.json()["result"] == "Параллельный результат." for r in responses)


def test_health_endpoints(client, isolated_processor):
    live = client.get("/health")
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}

    with patch.object(isolated_processor.llm, "check_readiness", return_value=True):
        ready_ok = client.get("/health/ready")
        assert ready_ok.status_code == 200
        assert ready_ok.json() == {"status": "ready"}

    with patch.object(isolated_processor.llm, "check_readiness", return_value=False):
        ready_fail = client.get("/health/ready")
        assert ready_fail.status_code == 503
