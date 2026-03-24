# classifier.py
"""
Классификатор пользовательских запросов.

Принимает текст → определяет тип (question/task/small_talk/complaint/unknown)
с уверенностью и обоснованием.

Используемые концепции LangChain:
- ChatPromptTemplate с few-shot примерами
- PydanticOutputParser для структурированного вывода
- RunnablePassthrough / RunnableLambda для формирования входных данных
- LCEL-цепочка через оператор |
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from models import Classification, RequestType


# ── 1. Парсер ────────────────────────────────────────────────────────
# PydanticOutputParser автоматически генерирует инструкции для LLM:
# "Верни JSON с полями request_type, confidence, reasoning..."
# Он же парсит ответ LLM обратно в объект Classification.

parser = PydanticOutputParser(pydantic_object=Classification)


# ── 2. Промпт классификатора ─────────────────────────────────────────
# Few-shot примеры помогают модели понять, что мы ожидаем.
# format_instructions подставляются автоматически из парсера.

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — классификатор пользовательских запросов. 
Твоя задача — определить тип запроса и вернуть результат в формате JSON.

## Типы запросов:

- **question** — вопрос, требующий информации или объяснения
  Примеры: "Что такое Python?", "Как работает GIL?", "Сколько планет в солнечной системе?"

- **task** — просьба выполнить конкретное действие, создать что-то
  Примеры: "Напиши стихотворение о весне", "Переведи на английский", "Сгенерируй список идей"

- **small_talk** — приветствие, прощание, болтовня, личные вопросы к боту
  Примеры: "Привет!", "Как дела?", "Меня зовут Алексей", "Спасибо, пока!"

- **complaint** — жалоба, недовольство, критика
  Примеры: "Это ужасный ответ!", "Почему так долго?", "Ты вообще не помогаешь"

- **unknown** — бессмыслица, случайные символы, нераспознаваемый ввод
  Примеры: "asdfghjkl", "123456", "ыаупкен"

## Примеры классификации:

Запрос: "Привет! Меня зовут Даша"
Ответ: {{"request_type": "small_talk", "confidence": 0.95, "reasoning": "Пользователь приветствует и представляется"}}

Запрос: "Что такое LCEL в LangChain?"
Ответ: {{"request_type": "question", "confidence": 0.97, "reasoning": "Пользователь задаёт вопрос о технологии"}}

Запрос: "Напиши анекдот про программистов"
Ответ: {{"request_type": "task", "confidence": 0.93, "reasoning": "Пользователь просит создать контент"}}

Запрос: "Ты работаешь отвратительно, ничего полезного!"
Ответ: {{"request_type": "complaint", "confidence": 0.91, "reasoning": "Пользователь выражает недовольство работой"}}

Запрос: "фывапролд"
Ответ: {{"request_type": "unknown", "confidence": 0.85, "reasoning": "Набор случайных символов без смысла"}}

{format_instructions}"""),
    ("human", "Запрос: {query}")
])


# ── 3. LCEL-цепочка ─────────────────────────────────────────────────

def build_classifier_chain(model: ChatOpenAI):
    """
    Собирает цепочку классификации.
    
    Архитектура цепочки:
        {"query": ..., "format_instructions": ...}  ← RunnableParallel
        → ChatPromptTemplate                         ← подстановка в промпт
        → ChatOpenAI                                 ← вызов LLM
        → PydanticOutputParser                       ← парсинг JSON → Classification
    
    Args:
        model: инициализированная языковая модель
    
    Returns:
        LCEL-цепочка, принимающая строку и возвращающая Classification
    """
    chain = (
        {
            "query": RunnablePassthrough(),
            "format_instructions": lambda _: parser.get_format_instructions()
        }
        | CLASSIFIER_PROMPT
        | model
        | parser
    )
    return chain


def classify_request(chain, query: str) -> Classification:
    """
    Классифицирует запрос с обработкой ошибок.
    
    Зачем обёртка: LLM иногда возвращает невалидный JSON 
    (лишние символы, markdown-обрамление, неправильные типы).
    В этом случае возвращаем безопасный fallback вместо краша.
    
    Args:
        chain: собранная цепочка классификатора
        query: текст пользователя
    
    Returns:
        Classification — всегда валидный объект
    """
    try:
        result = chain.invoke(query)
        return result
    except Exception as e:
        print(f"  ⚠ Ошибка классификации: {e}")
        return Classification(
            request_type=RequestType.UNKNOWN,
            confidence=0.5,
            reasoning=f"Ошибка парсинга ответа модели: {str(e)[:100]}"
        )
