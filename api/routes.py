from fastapi import APIRouter
from pydantic import BaseModel, Field
from config.settings import get_logger
from services.processor import TextProcessor

router = APIRouter()
logger = get_logger("api")
processor = TextProcessor()


class TextRequest(BaseModel):
    text: str = Field(..., min_length=15, max_length=5000)


class TextResponse(BaseModel):
    result: str
    cached: bool
    fallback: bool


@router.post("/process", response_model=TextResponse)
def process_endpoint(req: TextRequest) -> TextResponse:
    logger.info("Получен запрос POST /process")
    data = processor.process(req.text)
    return TextResponse(**data)
