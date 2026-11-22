from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="GigaChat Summarizer Service")
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
