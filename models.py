# models.py
"""
Структурированные модели данных для умного ассистента.

Три модели описывают полный жизненный цикл запроса:
1. RequestType — ЧТО за запрос (классификация)
2. Classification — результат анализа запроса (тип + уверенность + обоснование)
3. AssistantResponse — итоговый ответ пользователю (контент + метаданные)
"""

from enum import Enum
from pydantic import BaseModel, Field


class RequestType(str, Enum):
    """
    Перечисление возможных типов пользовательского запроса.
    
    Наследуем от str, чтобы значения легко сериализовались в JSON
    и сравнивались со строками: RequestType.QUESTION == "question" → True
    """
    QUESTION = "question"       # Вопрос, требующий информации
    TASK = "task"               # Просьба выполнить задачу
    SMALL_TALK = "small_talk"   # Приветствие, болтовня
    COMPLAINT = "complaint"     # Жалоба, недовольство
    UNKNOWN = "unknown"         # Нераспознанный запрос


class Classification(BaseModel):
    """
    Результат работы классификатора запросов.
    
    Эту модель возвращает цепочка классификации (Часть 2).
    PydanticOutputParser будет использовать её схему для генерации
    format_instructions — инструкций для LLM, в каком JSON-формате отвечать.
    
    Пример валидного объекта:
        Classification(
            request_type=RequestType.QUESTION,
            confidence=0.92,
            reasoning="Пользователь задаёт вопрос 'Что такое...'"
        )
    """
    request_type: RequestType = Field(
        description="Тип пользовательского запроса"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Уверенность классификации от 0 до 1"
    )
    reasoning: str = Field(
        description="Краткое обоснование выбранного типа (для отладки)"
    )


class AssistantResponse(BaseModel):
    """
    Итоговый ответ ассистента, включающий контент и метаданные.
    
    Метаданные (тип запроса, уверенность, токены) нужны для:
    - Отображения в CLI: [question] Ответ... confidence: 0.92
    - Мониторинга: сколько токенов тратим на разные типы запросов
    - Отладки: если бот отвечает странно, видим, как он классифицировал запрос
    """
    content: str = Field(
        description="Текст ответа ассистента"
    )
    request_type: RequestType = Field(
        description="Определённый тип исходного запроса"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Уверенность классификации"
    )
    tokens_used: int = Field(
        ge=0,
        description="Приблизительное количество использованных токенов"
    )