Egor Shakhonin

# Quick Start

## 1. Клонируйте репозиторий

```
git clone https://github.com/trueuser3/agents_system_2026.git
cd agents
```

## 2. Установите зависимости

```pip install langchain langchain-openai langchain-core python-dotenv pydantic```


## 3. Создайте файл .env в папке проекта

```echo "OPENAI_API_KEY=sk-ваш-ключ-сюда" > .env```





# Запуск

### Базовый запуск (характер: friendly, память: buffer)

```python smart_assistant.py```


### С выбранным характером

```python smart_assistant.py --character pirate```


### С потоковым выводом (текст печатается по токенам)

```python smart_assistant.py --stream```


### С суммаризацией памяти

```python smart_assistant.py --memory summary```


### Комбинация параметров

```python smart_assistant.py --character sarcastic --memory summary --stream```


### С конкретной моделью

```python smart_assistant.py --model gpt-4o```


## Пример сессии

```
🤖 Умный ассистент с характером
Характер: friendly | Память: buffer
────────────────────────────────────────
Введите сообщение или /help для справки

> Привет! Меня зовут Алексей
[small_talk] Привет, Алексей! Рад знакомству! 😊 Чем могу помочь?
confidence: 0.95 | tokens: ~32

> Что такое Python?
[question] Python — это высокоуровневый язык программирования,
известный своей простотой и читаемостью...
confidence: 0.96 | tokens: ~78

> /character pirate
✓ Характер изменён на: pirate

> Как меня зовут?
[question] Арр, матрос Алексей! Память у меня крепкая, как якорь! ☠️
confidence: 0.93 | tokens: ~35

> /quit
👋 До встречи!
```


## Тесты

### Установить pytest (если не установлен)
```pip install pytest```


### Все тесты (unit + интеграционные с API)
```python -m pytest tests.py -v```


### Только unit-тесты 
```python -m pytest tests.py -v -k "not integration"```

### Только интеграционные 
```python -m pytest tests.py -v -k "integration"```
