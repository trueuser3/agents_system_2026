# memory.py
"""
Менеджер памяти диалога.

Две стратегии:
- buffer: хранит последние N сообщений, старые отбрасывает
- summary: когда сообщений слишком много, сжимает старые в краткое содержание

Оба варианта возвращают список сообщений, который подставляется
в MessagesPlaceholder(variable_name="history") внутри промпта.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class MemoryManager:
    """
    Управляет историей диалога.
    
    Использование:
        memory = MemoryManager(strategy="buffer", max_messages=20)
        memory.add_user_message("Привет!")
        memory.add_ai_message("Привет! Как дела?")
        history = memory.get_history()  # → список сообщений для промпта
    """

    def __init__(
        self,
        strategy: str = "buffer",
        max_messages: int = 20,
        model: ChatOpenAI = None
    ):
        """
        Args:
            strategy: "buffer" или "summary"
            max_messages: максимум сообщений до обрезки/суммаризации
            model: языковая модель (нужна только для strategy="summary")
        """
        self.strategy = strategy
        self.max_messages = max_messages
        self.model = model
        self.messages = []       # полная история сообщений
        self.summary = ""        # краткое содержание (для strategy="summary")

    def add_user_message(self, content: str):
        """Добавляет сообщение пользователя в историю."""
        self.messages.append(HumanMessage(content=content))
        self._trim_if_needed()

    def add_ai_message(self, content: str):
        """Добавляет ответ ассистента в историю."""
        self.messages.append(AIMessage(content=content))
        self._trim_if_needed()

    def get_history(self) -> list:
        """
        Возвращает историю для подстановки в промпт.
        
        Для buffer: последние N сообщений как есть
        Для summary: SystemMessage с саммари + последние сообщения
        
        Returns:
            Список объектов Message для MessagesPlaceholder
        """
        if self.strategy == "buffer":
            return self._get_buffer_history()
        elif self.strategy == "summary":
            return self._get_summary_history()
        else:
            return self._get_buffer_history()

    def _get_buffer_history(self) -> list:
        """
        Стратегия buffer: просто возвращает последние max_messages сообщений.
        
        Плюс: простота, нет дополнительных вызовов LLM
        Минус: теряется контекст из начала разговора
        """
        return self.messages[-self.max_messages:]

    def _get_summary_history(self) -> list:
        """
        Стратегия summary: саммари старых сообщений + последние сообщения.
        
        Если есть саммари, оно идёт первым как SystemMessage,
        затем последние сообщения для актуального контекста.
        
        Плюс: помнит ключевые факты из длинного разговора
        Минус: тратит токены на суммаризацию
        """
        history = []

        # Добавляем саммари, если есть
        if self.summary:
            history.append(
                SystemMessage(content=f"Краткое содержание предыдущего разговора:\n{self.summary}")
            )

        # Последние сообщения (половина от max, чтобы оставить место для саммари)
        recent_count = self.max_messages // 2
        history.extend(self.messages[-recent_count:])

        return history

    def _trim_if_needed(self):
        """
        Проверяет, не превышен ли лимит сообщений.
        
        Для buffer: просто обрезает старые
        Для summary: суммаризирует старые, потом обрезает
        """
        if len(self.messages) <= self.max_messages:
            return

        if self.strategy == "buffer":
            # Просто отбрасываем старые
            self.messages = self.messages[-self.max_messages:]

        elif self.strategy == "summary":
            # Суммаризируем старые сообщения
            self._summarize_old_messages()

    def _summarize_old_messages(self):
        """
        Сжимает старые сообщения в краткое содержание.
        
        Берёт первую половину сообщений, отправляет в LLM
        с просьбой сделать саммари, сохраняет результат,
        оставляет только вторую половину сообщений.
        """
        if not self.model:
            # Нет модели для суммаризации — fallback на buffer
            self.messages = self.messages[-self.max_messages:]
            return

        # Делим: старые (для суммаризации) и новые (оставляем)
        split_point = len(self.messages) // 2
        old_messages = self.messages[:split_point]
        recent_messages = self.messages[split_point:]

        # Формируем текст старых сообщений
        old_text = self._messages_to_text(old_messages)

        # Промпт для суммаризации
        summarize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Ты — ассистент для создания кратких содержаний диалогов. "
             "Сделай краткое содержание разговора ниже. "
             "Обязательно сохрани: имена, ключевые факты, предпочтения пользователя, "
             "важные договорённости. Будь лаконичен, но не теряй важное."),
            ("human",
             "Предыдущее краткое содержание:\n{previous_summary}\n\n"
             "Новые сообщения:\n{old_text}\n\n"
             "Обновлённое краткое содержание:")
        ])

        summarize_chain = summarize_prompt | self.model | StrOutputParser()

        try:
            self.summary = summarize_chain.invoke({
                "previous_summary": self.summary or "(нет)",
                "old_text": old_text
            })
        except Exception as e:
            print(f"  ⚠ Ошибка суммаризации: {e}")
            # Если суммаризация не удалась — не теряем старое саммари

        # Оставляем только недавние сообщения
        self.messages = recent_messages

    @staticmethod
    def _messages_to_text(messages: list) -> str:
        """Конвертирует список сообщений в читаемый текст."""
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"Пользователь: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Ассистент: {msg.content}")
            elif isinstance(msg, SystemMessage):
                lines.append(f"Система: {msg.content}")
        return "\n".join(lines)

    def clear(self):
        """Полная очистка истории и саммари."""
        self.messages = []
        self.summary = ""

    def set_strategy(self, strategy: str) -> bool:
        """
        Переключает стратегию памяти.
        
        Args:
            strategy: "buffer" или "summary"
        
        Returns:
            True если стратегия валидна
        """
        if strategy not in ("buffer", "summary"):
            return False
        self.strategy = strategy
        return True

    def get_stats(self) -> dict:
        """Статистика для команды /status."""
        return {
            "strategy": self.strategy,
            "messages_count": len(self.messages),
            "max_messages": self.max_messages,
            "has_summary": bool(self.summary),
            "summary_preview": self.summary[:100] + "..." if len(self.summary) > 100 else self.summary
        }
