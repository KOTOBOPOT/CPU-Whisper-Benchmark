# PyTorch Whisper Dynamic Quantization Backend

Backend для Whisper используя PyTorch с Dynamic Quantization (int8).

## Что такое Dynamic Quantization?

**Dynamic Quantization** — это метод квантизации, который:
- Квантизирует веса Linear слоев в int8 **во время выполнения**
- Не требует калибровочного датасета
- Работает с оригинальной PyTorch моделью
- Остальные слои остаются в float32

## Быстрый старт

### 1. Настройте .env

```bash
# В корне проекта добавьте/обновите:
WHISPER_MODEL_WEIGHTS_PATH=openai/whisper-large-v3  # HuggingFace модель
WHISPER_BACKEND_NAME=whisper-pytorch-dynamic
WHISPER_BACKEND_PORT=7000
WHISPER_WORKERS=4
```

### 2. Запустите бэкенд

```bash
cd backends/whisper-pytorch-dynamic
./start.sh
```

### 3. Протестируйте

```bash
# Подождите ~60-90 секунд для загрузки модели
sleep 90

# Тест
cd ../..
python3 debug_api.py benchmark/data/golos_10_debug/files/0aaa7210f5c32b7f4d06c71b24e8689d.opus
```

### 4. Запустите бенчмарк

```bash
# Убедитесь, что в .env:
# BENCH_NAME=golos_1k
# WHISPER_BACKEND_NAME=whisper-pytorch-dynamic

./benchmark/start.sh
```

## Особенности

- **Dynamic Quantization**: Квантизация на лету без предварительной подготовки
- **Только Linear слои**: Квантизируются только Linear слои, остальное float32
- **CPU-only**: Dynamic quantization работает только на CPU
- **HuggingFace модели**: Поддержка любых HuggingFace Whisper моделей

## Ожидаемые результаты

Для Whisper Large v3 на Intel Xeon Gold 6348:
- **Latency**: ~8000-9000ms (между float32 и CTranslate2)
- **WER**: ~0.41-0.42 (близко к CTranslate2)
- **CER**: ~0.18
- **Throughput**: ~0.11-0.12 req/s

**Сравнение:**
- CTranslate2 int8: ~6600ms
- PyTorch Dynamic: ~8500ms (медленнее на ~30%)
- Float32 baseline: ~10000ms

## Преимущества

✅ Не требует конвертации модели  
✅ Работает с оригинальными HuggingFace моделями  
✅ Простая интеграция  
✅ Хорошее качество (лучше чем некоторые квантизации)

## Недостатки

⚠️ Медленнее чем CTranslate2 (на ~30%)  
⚠️ Больше памяти чем CTranslate2  
⚠️ Только CPU (не работает на GPU)

## Troubleshooting

### Медленная загрузка модели

Первая загрузка может занять 1-2 минуты (скачивание модели из HuggingFace).

### Out of Memory

Уменьшите количество воркеров:
```bash
WHISPER_WORKERS=2
```

### Плохое качество

Проверьте, что используется правильная модель:
```bash
WHISPER_MODEL_WEIGHTS_PATH=openai/whisper-large-v3
```

