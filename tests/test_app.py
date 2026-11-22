from unittest.mock import patch
from fastapi.testclient import TestClient
from api.routes import processor
from main import app

client = TestClient(app)


def test_validation_error():
    res = client.post("/process", json={"text": "Привет"})
    assert res.status_code == 422


def test_success_and_cache():
    processor.cache.clear()
    with patch.object(processor.llm, "summarize", return_value=("Краткий итог.", False)) as mock:
        payload = {"text": "Длинный текст для проверки работы суммаризатора и кэширования."}

        r1 = client.post("/process", json=payload)
        assert r1.status_code == 200
        assert r1.json() == {"result": "Краткий итог.", "cached": False, "fallback": False}
        assert mock.call_count == 1

        r2 = client.post("/process", json=payload)
        assert r2.status_code == 200
        assert r2.json()["cached"] is True
        assert mock.call_count == 1


def test_fallback_on_error():
    processor.cache.clear()
    with patch("llm.client.GigaChat", side_effect=Exception("Network down")):
        res = client.post("/process", json={"text": "Текст для проверки сбоя сети и fallback."})
        assert res.status_code == 200
        assert res.json()["fallback"] is True
