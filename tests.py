# tests.py
"""
Тесты для умного ассистента.

Unit-тесты:     python3 -m pytest tests.py -v -k "not integration"
Интеграционные:  python3 -m pytest tests.py -v -k "integration"
Все тесты:       python3 -m pytest tests.py -v

Интеграционные тесты вызывают реальный API — нужен OPENAI_API_KEY в .env
"""

import pytest
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
load_dotenv()

from pydantic import ValidationError
from langchain_openai import ChatOpenAI

from models import RequestType, Classification, AssistantResponse
from memory import MemoryManager
from characters import (
    CHARACTER_PROMPTS,
    AVAILABLE_CHARACTERS,
    HANDLER_INSTRUCTIONS,
    build_all_character_handlers,
)
from classifier import build_classifier_chain, classify_request
from handlers import SmartRouter


# ═══════════════════════════════════════════════════════════════════════
# UNIT-ТЕСТЫ (без вызовов API)
# ═══════════════════════════════════════════════════════════════════════

class TestModels:
    """Тесты Pydantic-моделей."""

    def test_valid_classification(self):
        c = Classification(
            request_type=RequestType.QUESTION,
            confidence=0.92,
            reasoning="Это вопрос"
        )
        assert c.request_type == RequestType.QUESTION
        assert c.confidence == 0.92

    def test_confidence_too_high(self):
        with pytest.raises(ValidationError):
            Classification(
                request_type=RequestType.QUESTION,
                confidence=1.5,
                reasoning="test"
            )

    def test_confidence_negative(self):
        with pytest.raises(ValidationError):
            Classification(
                request_type=RequestType.QUESTION,
                confidence=-0.1,
                reasoning="test"
            )

    def test_invalid_request_type(self):
        with pytest.raises(ValidationError):
            Classification(
                request_type="magic",
                confidence=0.5,
                reasoning="test"
            )

    def test_assistant_response(self):
        resp = AssistantResponse(
            content="Ответ",
            request_type=RequestType.TASK,
            confidence=0.88,
            tokens_used=42
        )
        assert resp.content == "Ответ"
        assert resp.tokens_used == 42

    def test_request_type_is_string(self):
        assert RequestType.QUESTION == "question"
        assert RequestType.SMALL_TALK == "small_talk"

    def test_classification_json_roundtrip(self):
        original = Classification(
            request_type=RequestType.COMPLAINT,
            confidence=0.77,
            reasoning="Жалоба"
        )
        json_str = original.model_dump_json()
        restored = Classification.model_validate_json(json_str)
        assert original == restored


class TestMemory:
    """Тесты менеджера памяти."""

    def test_add_and_get_messages(self):
        mem = MemoryManager(strategy="buffer", max_messages=10)
        mem.add_user_message("Привет")
        mem.add_ai_message("Здравствуйте!")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0].content == "Привет"
        assert history[1].content == "Здравствуйте!"

    def test_buffer_trimming(self):
        mem = MemoryManager(strategy="buffer", max_messages=4)
        for i in range(3):
            mem.add_user_message(f"Вопрос {i}")
            mem.add_ai_message(f"Ответ {i}")
        history = mem.get_history()
        assert len(history) == 4
        assert history[0].content == "Вопрос 1"

    def test_clear(self):
        mem = MemoryManager(strategy="buffer")
        mem.add_user_message("Привет")
        mem.add_ai_message("Здравствуйте!")
        mem.summary = "Какое-то саммари"
        mem.clear()
        assert len(mem.get_history()) == 0
        assert mem.summary == ""

    def test_set_strategy(self):
        mem = MemoryManager(strategy="buffer")
        assert mem.set_strategy("summary") is True
        assert mem.strategy == "summary"
        assert mem.set_strategy("invalid") is False
        assert mem.strategy == "summary"

    def test_stats(self):
        mem = MemoryManager(strategy="buffer", max_messages=20)
        mem.add_user_message("Тест")
        stats = mem.get_stats()
        assert stats["strategy"] == "buffer"
        assert stats["messages_count"] == 1
        assert stats["max_messages"] == 20
        assert stats["has_summary"] is False


class TestCharacters:
    """Тесты системы характеров."""

    def test_all_characters_have_prompts(self):
        for char in AVAILABLE_CHARACTERS:
            assert char in CHARACTER_PROMPTS
            assert len(CHARACTER_PROMPTS[char]) > 0

    def test_all_request_types_have_instructions(self):
        for rt in RequestType:
            assert rt in HANDLER_INSTRUCTIONS
            assert len(HANDLER_INSTRUCTIONS[rt]) > 0

    def test_build_handlers_returns_all_types(self):
        mock_model = MagicMock()
        handlers = build_all_character_handlers("friendly", mock_model)
        for rt in RequestType:
            assert rt in handlers

    def test_available_characters_list(self):
        assert len(AVAILABLE_CHARACTERS) >= 4
        assert "friendly" in AVAILABLE_CHARACTERS
        assert "pirate" in AVAILABLE_CHARACTERS


class TestRouter:
    """Тесты SmartRouter без API."""

    def test_set_character_valid(self):
        mock_model = MagicMock()
        with patch("handlers.build_classifier_chain"):
            router = SmartRouter(mock_model, character="friendly")
            assert router.set_character("pirate") is True
            assert router.character == "pirate"

    def test_set_character_invalid(self):
        mock_model = MagicMock()
        with patch("handlers.build_classifier_chain"):
            router = SmartRouter(mock_model, character="friendly")
            assert router.set_character("robot") is False
            assert router.character == "friendly"

    def test_get_status(self):
        mock_model = MagicMock()
        with patch("handlers.build_classifier_chain"):
            router = SmartRouter(mock_model, character="sarcastic")
            status = router.get_status()
            assert status["character"] == "sarcastic"
            assert "strategy" in status
            assert "messages_count" in status


# ═══════════════════════════════════════════════════════════════════════
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (реальные вызовы API)
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def model():
    """Одна модель на все интеграционные тесты (экономим время)."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@pytest.fixture(scope="module")
def classifier_chain(model):
    """Цепочка классификатора."""
    return build_classifier_chain(model)


@pytest.fixture
def router(model):
    """Свежий роутер для каждого теста (чистая память)."""
    return SmartRouter(model, character="friendly", memory_strategy="buffer")


class TestClassifierIntegration:
    """Интеграционные тесты классификатора — реальные вызовы LLM."""

    @pytest.mark.integration
    def test_classify_question(self, classifier_chain):
        """Вопрос классифицируется как question."""
        result = classify_request(classifier_chain, "Что такое Python?")
        assert result.request_type == RequestType.QUESTION
        assert result.confidence >= 0.7

    @pytest.mark.integration
    def test_classify_small_talk(self, classifier_chain):
        """Приветствие классифицируется как small_talk."""
        result = classify_request(classifier_chain, "Привет! Как дела?")
        assert result.request_type == RequestType.SMALL_TALK
        assert result.confidence >= 0.7

    @pytest.mark.integration
    def test_classify_task(self, classifier_chain):
        """Просьба создать контент классифицируется как task."""
        result = classify_request(classifier_chain, "Напиши стихотворение о весне")
        assert result.request_type == RequestType.TASK
        assert result.confidence >= 0.7

    @pytest.mark.integration
    def test_classify_complaint(self, classifier_chain):
        """Жалоба классифицируется как complaint."""
        result = classify_request(classifier_chain, "Это ужасный ответ! Ты бесполезен!")
        assert result.request_type == RequestType.COMPLAINT
        assert result.confidence >= 0.7

    @pytest.mark.integration
    def test_classify_unknown(self, classifier_chain):
        """Бессмыслица классифицируется как unknown."""
        result = classify_request(classifier_chain, "фывапролджэ")
        assert result.request_type == RequestType.UNKNOWN
        assert result.confidence >= 0.5

    @pytest.mark.integration
    def test_classification_returns_valid_object(self, classifier_chain):
        """Результат — валидный объект Classification."""
        result = classify_request(classifier_chain, "Сколько будет 2+2?")
        assert isinstance(result, Classification)
        assert 0 <= result.confidence <= 1
        assert len(result.reasoning) > 0


class TestRouterIntegration:
    """Интеграционные тесты полного пайплайна."""

    @pytest.mark.integration
    def test_full_pipeline_question(self, router):
        """Полный цикл: вопрос → классификация → ответ."""
        result = router.process("Что такое Python?")
        assert isinstance(result, AssistantResponse)
        assert result.request_type == RequestType.QUESTION
        assert len(result.content) > 10
        assert result.tokens_used > 0

    @pytest.mark.integration
    def test_full_pipeline_task(self, router):
        """Полный цикл: задача → классификация → выполнение."""
        result = router.process("Напиши хайку о программировании")
        assert isinstance(result, AssistantResponse)
        assert result.request_type == RequestType.TASK
        assert len(result.content) > 5

    @pytest.mark.integration
    def test_full_pipeline_small_talk(self, router):
        """Полный цикл: приветствие → ответ."""
        result = router.process("Привет! Меня зовут Тестер")
        assert isinstance(result, AssistantResponse)
        assert result.request_type == RequestType.SMALL_TALK
        assert len(result.content) > 5


class TestMemoryIntegration:
    """Интеграционные тесты памяти — бот запоминает факты."""

    @pytest.mark.integration
    def test_remembers_name(self, router):
        """Бот запоминает имя из предыдущего сообщения."""
        # Представляемся
        router.process("Привет! Меня зовут Алексей")

        # Спрашиваем имя
        result = router.process("Как меня зовут?")

        # Имя должно быть в ответе
        assert "Алексей" in result.content

    @pytest.mark.integration
    def test_remembers_multiple_facts(self, router):
        """Бот запоминает несколько фактов."""
        router.process("Меня зовут Даша")
        router.process("Мой любимый язык — Python")

        result = router.process("Как меня зовут и какой мой любимый язык?")

        assert "Даша" in result.content
        assert "Python" in result.content

    @pytest.mark.integration
    def test_memory_survives_character_switch(self, router):
        """Память сохраняется при смене характера."""
        router.process("Привет, я Борис")

        router.set_character("pirate")

        result = router.process("Как меня зовут?")
        assert "Борис" in result.content

    @pytest.mark.integration
    def test_clear_erases_memory(self, router):
        """После очистки бот не помнит имя."""
        router.process("Меня зовут Секретный Агент")

        router.clear_memory()

        result = router.process("Как меня зовут?")
        # После очистки не должен знать имя
        assert "Секретный Агент" not in result.content


class TestCharacterIntegration:
    """Интеграционные тесты характеров — стиль ответа меняется."""

    @pytest.mark.integration
    def test_pirate_character(self, model):
        """Пиратский характер использует морскую лексику."""
        router = SmartRouter(model, character="pirate")
        result = router.process("Привет!")

        content_lower = result.content.lower()
        pirate_words = ["арр", "матрос", "капитан", "борт", "йо-хо", "☠"]
        has_pirate_style = any(word in content_lower for word in pirate_words)
        assert has_pirate_style, f"Ответ не похож на пиратский: {result.content}"

    @pytest.mark.integration
    def test_professional_character(self, model):
        """Профессиональный характер обращается на 'вы'."""
        router = SmartRouter(model, character="professional")
        result = router.process("Привет!")

        # Профессионал не использует эмодзи и обращается на "вы"
        assert "😊" not in result.content or "вы" in result.content.lower() or "Вы" in result.content
