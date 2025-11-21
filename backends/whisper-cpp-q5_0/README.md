# Whisper.cpp Q5_0 Backend

Backend для Whisper используя whisper.cpp с GGML квантизацией Q5_0.

## Быстрый старт

### 1. Настройте .env

```bash
# В корне проекта добавьте/обновите:
WHISPER_MODEL_WEIGHTS_PATH=/app/weights/whisper-large-v3-ggml/ggml-large-v3-q5_0.bin
WHISPER_BACKEND_NAME=whisper-cpp-q5_0
WHISPER_BACKEND_PORT=7000
WHISPER_CPP_THREADS=12
```

### 2. Запустите бэкенд

```bash
cd backends/whisper-cpp-q5_0
./start.sh
```

### 3. Протестируйте

```bash
# Подождите ~30 секунд для загрузки
sleep 30

# Тест
cd ../..
python3 debug_api.py benchmark/data/golos_10_debug/files/0aaa7210f5c32b7f4d06c71b24e8689d.opus
```

### 4. Запустите бенчмарк

```bash
# Убедитесь, что в .env:
# BENCH_NAME=golos_1k
# WHISPER_BACKEND_NAME=whisper-cpp-q5_0

./benchmark/start.sh
```

## Особенности

- Использует whisper.cpp для максимальной скорости на CPU
- GGML квантизация Q5_0 (~1.24 GB, хороший баланс скорости и качества)
- Автоматическая компиляция whisper.cpp в Docker образе
- Поддержка настройки количества потоков через `WHISPER_CPP_THREADS`

## Ожидаемые результаты

Для Whisper Large v3 на Intel Xeon Gold 6348:
- **Latency**: ~5500-6000ms
- **WER**: ~0.40-0.42
- **CER**: ~0.18
- **Throughput**: ~0.18 req/s

## Troubleshooting

### Ошибка: whisper-cli not found

Убедитесь, что whisper.cpp скомпилирован в Docker образе. Проверьте логи:

```bash
docker logs whisper-backend-cpp-q5_0
```

### Медленная работа

Увеличьте количество потоков в .env:
```bash
WHISPER_CPP_THREADS=16
```

### Проблемы с аудио форматом

whisper.cpp ожидает WAV 16kHz. Код автоматически конвертирует через librosa.

