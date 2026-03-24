# smart_assistant.py
"""
Умный ассистент с характером — CLI-интерфейс.

Запуск:
    python smart_assistant.py
    python smart_assistant.py --character sarcastic
    python smart_assistant.py --stream --character pirate
    python smart_assistant.py --model gpt-4o --memory summary
"""

import argparse

from dotenv import load_dotenv
load_dotenv()

from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_openai import ChatOpenAI

from handlers import SmartRouter
from characters import AVAILABLE_CHARACTERS

# Кэширование — повторные вопросы не тратят токены
set_llm_cache(InMemoryCache())


def parse_args():
    parser = argparse.ArgumentParser(
        description="🤖 Умный ассистент с характером"
    )
    parser.add_argument(
        "--character",
        type=str,
        default="friendly",
        choices=AVAILABLE_CHARACTERS,
        help=f"Характер: {', '.join(AVAILABLE_CHARACTERS)}"
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="buffer",
        choices=["buffer", "summary"],
        help="Стратегия памяти: buffer или summary"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Модель (по умолчанию: gpt-4o-mini)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Потоковый вывод (текст печатается по токенам)"
    )
    return parser.parse_args()


def handle_command(command: str, router: SmartRouter) -> bool:
    parts = command.strip().split()
    cmd = parts[0].lower()

    if cmd == "/quit":
        print("\n👋 До встречи!")
        return False

    elif cmd == "/help":
        print_help()

    elif cmd == "/clear":
        router.clear_memory()
        print("🗑️  История очищена")

    elif cmd == "/status":
        print_status(router)

    elif cmd == "/character":
        if len(parts) < 2:
            print(f"❌ Укажите характер: {', '.join(AVAILABLE_CHARACTERS)}")
        else:
            name = parts[1].lower()
            if router.set_character(name):
                print(f"✓ Характер изменён на: {name}")
            else:
                print(f"❌ Неизвестный характер: {name}")
                print(f"  Доступные: {', '.join(AVAILABLE_CHARACTERS)}")

    elif cmd == "/memory":
        if len(parts) < 2:
            print("❌ Укажите стратегию: buffer или summary")
        else:
            strategy = parts[1].lower()
            if router.set_memory_strategy(strategy):
                print(f"✓ Стратегия памяти: {strategy}")
            else:
                print(f"❌ Неизвестная стратегия: {strategy}")

    else:
        print(f"❌ Неизвестная команда: {cmd}. Введите /help")

    return True


def print_banner(character: str, memory: str, stream: bool):
    print()
    print("🤖 Умный ассистент с характером")
    stream_label = " | Стриминг: вкл" if stream else ""
    print(f"Характер: {character} | Память: {memory}{stream_label}")
    print("─" * 40)
    print("Введите сообщение или /help для справки")
    print()


def print_help():
    print()
    print("📖 Доступные команды:")
    print("─" * 40)
    print("  /character <name>  — сменить характер")
    print(f"    Варианты: {', '.join(AVAILABLE_CHARACTERS)}")
    print("  /memory <strategy> — сменить стратегию памяти")
    print("    Варианты: buffer, summary")
    print("  /clear             — очистить историю")
    print("  /status            — текущие настройки")
    print("  /help              — эта справка")
    print("  /quit              — выход")
    print()


def print_status(router: SmartRouter):
    status = router.get_status()
    print()
    print("📊 Текущий статус:")
    print("─" * 40)
    print(f"  Характер:    {status['character']}")
    print(f"  Память:      {status['strategy']}")
    print(f"  Сообщений:   {status['messages_count']} / {status['max_messages']}")
    if status['has_summary']:
        print(f"  Саммари:     {status['summary_preview']}")
    print()


def print_response(result):
    print(f"[{result.request_type.value}] {result.content}")
    print(f"confidence: {result.confidence} | tokens: ~{result.tokens_used}")


def main():
    args = parse_args()

    # Основная модель + fallback
    main_model = ChatOpenAI(model=args.model, temperature=0.7)
    fallback_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    model = main_model.with_fallbacks([fallback_model])

    router = SmartRouter(
        model=model,
        character=args.character,
        memory_strategy=args.memory
    )

    print_banner(args.character, args.memory, args.stream)

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 До встречи!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, router):
                break
            continue

        # Обработка сообщения
        if args.stream:
            result = router.process_stream(user_input)
            print(f"confidence: {result.confidence} | tokens: ~{result.tokens_used}")
        else:
            result = router.process(user_input)
            print_response(result)
        print()


if __name__ == "__main__":
    main()
