# Whisper Distillation Backend

Бекенд для распознавания речи с использованием дистиллированной модели Whisper.

## Структура

```
backends/distillation/backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # Точка входа FastAPI приложения
│   └── model.py         # Класс WhisperModel для инференса
├── Dockerfile           # Docker образ
├── docker-compose.yml   # Docker Compose конфигурация
├── requirements.txt     # Python зависимости
├── start.sh            # Скрипт для запуска контейнера
└── README.md           # Эта документация
```

## Требования

- Docker и Docker Compose
- `.env` файл в корне проекта с необходимыми переменными окружения

## Конфигурация

Создайте или обновите файл `.env` в корне проекта со следующими переменными:

```bash
# Порт для бекенда (по умолчанию: 8001)
WHISPER_BACKEND_PORT=8001

# Путь к чекпоинту модели внутри контейнера (по умолчанию: /app/checkpoints/checkpoint-6100)
WHISPER_MODEL_CHECKPOINT_PATH=/app/checkpoints/checkpoint-6100

# Локальная директория с чекпоинтами для монтирования (по умолчанию: ./checkpoints)
LOCAL_WHISPER_CHECKPOINT_DIR=/home/user/datasets/itmo/whisper_acc/distill/exps

# Устройство для инференса: cpu, cuda:0, cuda:1 и т.д. (по умолчанию: cpu)
WHISPER_DEVICE=cpu

# Тип данных: float32, float16, bfloat16 (по умолчанию: float32)
WHISPER_DTYPE=float32

# Язык распознавания (по умолчанию: ru)
WHISPER_LANGUAGE=ru

# Задача: transcribe или translate (по умолчанию: transcribe)
WHISPER_TASK=transcribe
```

## Запуск

### Вариант 1: Использование start.sh

```bash
cd backends/distillation/backend
./start.sh
```

### Вариант 2: Напрямую через Docker Compose

```bash
cd backends/distillation/backend
docker compose up --build -d
```

## Использование API

После запуска бекенд будет доступен на порту, указанном в `WHISPER_BACKEND_PORT` (по умолчанию 8001).

### Транскрипция аудио

```bash
curl -X POST "http://localhost:8001/process_audio" \
  -F "file=@/path/to/audio.opus" \
  -H "Content-Type: multipart/form-data"
```

Ответ:
```json
{
  "text": "распознанный текст из аудио"
}
```

### Поддерживаемые форматы аудио

- audio/wav (WAV)
- audio/x-wav
- audio/mpeg (MP3)
- audio/mp3
- audio/ogg (OGG)
- audio/opus (OPUS)

## Остановка

```bash
cd backends/distillation/backend
docker compose down
```

## Логи

Просмотр логов контейнера:

```bash
docker compose logs -f whisper-backend-distillation
```

## Отладка

### Проверка статуса контейнера

```bash
docker ps | grep whisper-backend-distillation
```

### Вход в контейнер

```bash
docker compose exec whisper-backend-distillation bash
```

### Проверка доступности API

```bash
curl http://localhost:8001/docs
```

Откроется Swagger UI с документацией API.

## Примечания

- Для использования GPU укажите `WHISPER_DEVICE=cuda:0` и убедитесь, что Docker настроен для работы с NVIDIA GPU
- Для float16 и bfloat16 рекомендуется использовать GPU, на CPU лучше использовать float32
- Модель загружается при запуске контейнера, что может занять некоторое время
- Чекпоинт модели должен содержать все необходимые файлы (config.json, pytorch_model.bin, processor и т.д.)

