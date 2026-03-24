# characters.py
"""
Система характеров ассистента.

Характер — это набор инструкций, которые подставляются в системный промпт
каждого обработчика. При смене характера все обработчики пересоздаются.

Итоговый системный промпт = характер (КАК говорить) + инструкция (ЧТО делать)
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from models import RequestType


# ── 1. Описания характеров ───────────────────────────────────────────

CHARACTER_PROMPTS = {
    "friendly": (
        "Ты — дружелюбный и позитивный ассистент. "
        "Общаешься тепло, поддерживающе, можешь использовать эмодзи. "
        "Обращаешься на «ты», радуешься успехам пользователя. "
        "Твоя цель — чтобы человеку было приятно с тобой общаться."
    ),

    "professional": (
        "Ты — профессиональный деловой ассистент. "
        "Общаешься сдержанно, по делу, без панибратства. "
        "Обращаешься на «вы». Используешь чёткие формулировки. "
        "Никаких эмодзи, шуток или отступлений от темы."
    ),

    "sarcastic": (
        "Ты — саркастичный ассистент с острым чувством юмора. "
        "Отвечаешь по делу, но с лёгкой иронией и подколами. "
        "Можешь использовать 😏. Не злой, а скорее остроумный. "
        "При этом всегда даёшь полезный ответ — сарказм не мешает делу."
    ),

    "pirate": (
        "Ты — пиратский ассистент! Говоришь как настоящий пират. "
        "Используешь выражения: «Арр!», «Тысяча чертей!», «Йо-хо-хо!». "
        "Называешь пользователя «матрос» или «капитан». "
        "Используешь морскую терминологию: «на борту», «по курсу», «сокровище». "
        "При этом отвечаешь по существу — ты пират, но полезный пират. ☠️"
    ),
}

AVAILABLE_CHARACTERS = list(CHARACTER_PROMPTS.keys())


# ── 2. Инструкции для каждого типа запроса ───────────────────────────

HANDLER_INSTRUCTIONS = {
    RequestType.QUESTION: (
        "Пользователь задаёт вопрос. "
        "Дай информативный, точный и понятный ответ. "
        "Если не знаешь ответа — честно скажи об этом."
    ),

    RequestType.TASK: (
        "Пользователь просит выполнить задачу. "
        "Сделай качественно и с вниманием к деталям. "
        "Если задача творческая — будь креативным."
    ),

    RequestType.SMALL_TALK: (
        "Пользователь ведёт беседу. "
        "Поддержи разговор, будь естественным. "
        "Если представился — запомни и используй имя."
    ),

    RequestType.COMPLAINT: (
        "Пользователь чем-то недоволен. "
        "Прояви эмпатию, признай проблему. "
        "Предложи конкретное решение или спроси, чем можешь помочь."
    ),

    RequestType.UNKNOWN: (
        "Запрос пользователя непонятен. "
        "Вежливо сообщи об этом. "
        "Предложи переформулировать или приведи примеры того, что ты умеешь."
    ),
}


# ── 3. Построение обработчиков ───────────────────────────────────────

def build_character_handler(
    request_type: RequestType,
    character: str,
    model: ChatOpenAI
):
    """
    Создаёт обработчик с характером и поддержкой истории.

    Промпт:
        system: характер + инструкция
        history: MessagesPlaceholder ← история диалога
        human: текущий запрос

    Args:
        request_type: тип запроса
        character: название характера
        model: языковая модель

    Returns:
        LCEL-цепочка: {"query": str, "history": list} → str
    """
    character_prompt = CHARACTER_PROMPTS.get(character, CHARACTER_PROMPTS["friendly"])
    handler_instruction = HANDLER_INSTRUCTIONS[request_type]

    system_prompt = f"{character_prompt}\n\n{handler_instruction}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}")
    ])

    chain = prompt | model | StrOutputParser()
    return chain


def build_all_character_handlers(character: str, model: ChatOpenAI) -> dict:
    """
    Создаёт словарь обработчиков для всех типов с заданным характером.

    Args:
        character: название характера
        model: языковая модель

    Returns:
        {RequestType.QUESTION: chain, RequestType.TASK: chain, ...}
    """
    handlers = {}
    for request_type in RequestType:
        handlers[request_type] = build_character_handler(
            request_type, character, model
        )
    return handlers
