# GigaChat Text Summarizer Service

Сервис на базе FastAPI для краткого пересказа (суммаризации) текста с помощью **GigaChat API** (Сбер).

## Установка и запуск

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

Документация Swagger: `http://127.0.0`
