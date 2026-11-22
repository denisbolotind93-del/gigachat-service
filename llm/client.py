from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from config.settings import get_logger, settings

logger = get_logger("llm")
FALLBACK_TEXT = "GigaChat временно недоступен. Попробуйте повторить запрос позже."


class GigaChatClient:
    def summarize(self, text: str) -> tuple[str, bool]:
        logger.info("Формирование промпта и отправка запроса в GigaChat")
        payload = Chat(
            messages=[
                Messages(
                    role=MessagesRole.SYSTEM,
                    content="Ты редактор. Кратко перескажи текст пользователя в 2-3 предложениях.",
                ),
                Messages(role=MessagesRole.USER, content=text),
            ],
            temperature=0.3,
        )

        try:
            with GigaChat(
                credentials=settings.gigachat_credentials,
                scope=settings.gigachat_scope,
                model=settings.gigachat_model,
                timeout=settings.llm_timeout,
                verify_ssl_certs=False,
            ) as giga:
                response = giga.chat(payload)
                answer = response.choices[0].message.content
                logger.info("Ответ от GigaChat успешно получен")
                return answer, False
        except Exception as e:
            logger.error(f"Ошибка при обращении к GigaChat: {e}")
            return FALLBACK_TEXT, True
