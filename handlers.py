# handlers.py (финальная версия)
"""
Обработчики запросов и роутинг.
Финальная версия: характеры + память.
"""

from langchain_openai import ChatOpenAI

from models import RequestType, Classification, AssistantResponse
from classifier import build_classifier_chain, classify_request
from characters import build_all_character_handlers, AVAILABLE_CHARACTERS
from memory import MemoryManager


class SmartRouter:
    """
    Роутер с характерами и памятью.
    
    Поток данных:
        query → классификатор → Classification
                                    │
                handlers[type] ← history из MemoryManager
                        │
                        ▼
                AssistantResponse
                        │
                memory.add(query, response)  ← обновление истории
    """

    def process_stream(self, query: str) -> AssistantResponse:
        """
        Обработка запроса с потоковым выводом.
        
        Текст печатается по мере генерации,
        затем собирается в AssistantResponse.
        """
        # Шаг 1: Классификация (без стриминга — нужен полный результат)
        classification = classify_request(self.classifier_chain, query)

        # Шаг 2: История
        history = self.memory.get_history()

        # Шаг 3: Выбор обработчика
        handler = self.handlers.get(
            classification.request_type,
            self.handlers[RequestType.UNKNOWN]
        )

        # Шаг 4: Стриминг ответа
        print(f"[{classification.request_type.value}] ", end="", flush=True)

        full_response = ""
        try:
            for chunk in handler.stream({"query": query, "history": history}):
                print(chunk, end="", flush=True)
                full_response += chunk
        except Exception as e:
            full_response = f"Произошла ошибка: {e}"
            print(full_response, end="")

        print()  # Перевод строки после ответа

        # Шаг 5: Сохраняем в память
        self.memory.add_user_message(query)
        self.memory.add_ai_message(full_response)

        # Шаг 6: Упаковка
        estimated_tokens = len(query + full_response) // 3

        return AssistantResponse(
            content=full_response,
            request_type=classification.request_type,
            confidence=classification.confidence,
            tokens_used=estimated_tokens
        )

    def __init__(
        self,
        model: ChatOpenAI,
        character: str = "friendly",
        memory_strategy: str = "buffer"
    ):
        """
        Args:
            model: языковая модель
            character: начальный характер
            memory_strategy: стратегия памяти ("buffer" или "summary")
        """
        self.model = model
        self.character = character
        self.classifier_chain = build_classifier_chain(model)
        self.handlers = build_all_character_handlers(character, model)
        self.memory = MemoryManager(
            strategy=memory_strategy,
            max_messages=20,
            model=model
        )

    def set_character(self, character: str) -> bool:
        """Переключает характер. Память сохраняется!"""
        if character not in AVAILABLE_CHARACTERS:
            return False
        self.character = character
        self.handlers = build_all_character_handlers(character, self.model)
        return True

    def set_memory_strategy(self, strategy: str) -> bool:
        """Переключает стратегию памяти. История сохраняется!"""
        return self.memory.set_strategy(strategy)

    def clear_memory(self):
        """Очищает историю диалога."""
        self.memory.clear()

    def get_status(self) -> dict:
        """Текущее состояние для команды /status."""
        memory_stats = self.memory.get_stats()
        return {
            "character": self.character,
            **memory_stats
        }

    def process(self, query: str) -> AssistantResponse:
        """
        Полный цикл обработки запроса с памятью.
        
        1. Классификация
        2. Получение истории из памяти
        3. Роутинг + генерация (с историей в промпте)
        4. Сохранение в память
        5. Упаковка в AssistantResponse
        """
        # Шаг 1: Классификация
        classification = classify_request(self.classifier_chain, query)

        # Шаг 2: Получаем историю
        history = self.memory.get_history()

        # Шаг 3: Выбор обработчика и генерация
        handler = self.handlers.get(
            classification.request_type,
            self.handlers[RequestType.UNKNOWN]
        )

        try:
            # Теперь передаём И запрос, И историю
            response_text = handler.invoke({
                "query": query,
                "history": history
            })
        except Exception as e:
            response_text = f"Произошла ошибка: {e}"

        # Шаг 4: Сохраняем в память
        self.memory.add_user_message(query)
        self.memory.add_ai_message(response_text)

        # Шаг 5: Упаковка
        estimated_tokens = len(query + response_text) // 3

        return AssistantResponse(
            content=response_text,
            request_type=classification.request_type,
            confidence=classification.confidence,
            tokens_used=estimated_tokens
        )
