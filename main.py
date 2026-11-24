import time
import uuid

from fastapi import FastAPI, Request

from api.routes import router
from config.settings import get_logger, request_id_ctx

logger = get_logger("api")

app = FastAPI(
    title="GigaChat Summarizer Service",
    description="Сервис суммаризации текста на базе FastAPI и GigaChat API.",
    version="1.1.0",
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_ctx.set(req_id)
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        logger.info(
            "HTTP запрос обработан",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                }
            },
        )
        return response
    finally:
        request_id_ctx.reset(token)


app.include_router(router)
