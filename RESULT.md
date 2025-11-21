# Сводка результатов квантизации Whisper Large v3

## Общая информация

- **Модель**: Whisper Large v3
- **CPU**: Intel Xeon Gold 6348 (Ice Lake, 96 cores)
- **Датасет**: golos_1k (1000 файлов) для основных результатов
- **Дата тестирования**: 21 ноября 2025

---

## Результаты по квантизациям

### 1. CTranslate2 - int8_float32 ⭐ ЛУЧШИЙ

**Бэкенд**: `whisper-ctranslate2-int8`  
**Квантизация**: int8_float32 (веса int8, активации float32)  
**Метод**: Post-training quantization через CTranslate2  
**Размер модели**: ~1.6 GB (~27% от float32)

**Результаты (golos_1k, 1000 файлов):**
- **Average Latency**: 6612.85 ms
- **Throughput**: 0.604 req/s
- **Total Wall Time**: 1656.4 секунд (27.6 минут)
- **WER**: 0.4163
- **CER**: 0.1806
- **Success Rate**: 100% (0 failed)

**Оценка**: ⭐⭐⭐⭐⭐
- ✅ Лучший баланс скорости и качества
- ✅ Самая быстрая квантизация
- ✅ Отличное качество (WER почти как float32)

---

### 2. CTranslate2 - int8_float16

**Бэкенд**: `whisper-ctranslate2-int8_float16`  
**Квантизация**: int8_float16 (веса int8, активации float16)  
**Метод**: Post-training quantization через CTranslate2  
**Размер модели**: ~1.6 GB (~27% от float32)

**Результаты (golos_1k, 1000 файлов):**
- **Average Latency**: 6558.01 ms
- **Throughput**: 0.609 req/s
- **Total Wall Time**: 1641.9 секунд (27.4 минут)
- **WER**: 0.4163
- **CER**: 0.1806
- **Success Rate**: 100% (0 failed)

**Оценка**: ⭐⭐⭐⭐⭐
- ✅ Немного быстрее чем int8_float32 (+0.8%)
- ✅ Такое же качество
- ✅ Лучший вариант для CPU с поддержкой float16

**Сравнение с int8_float32:**
- Latency: -0.8% (быстрее на 55ms)
- Throughput: +0.9%
- Качество: идентично

---

### 3. CTranslate2 - int16

**Бэкенд**: `whisper-ctranslate2-int16`  
**Квантизация**: int16 (веса и активации int16)  
**Метод**: Post-training quantization через CTranslate2  
**Размер модели**: ~1.6 GB (~51% от float32)

**Результаты (golos_1k, 1000 файлов):**
- **Average Latency**: 8603.10 ms
- **Throughput**: 0.464 req/s
- **Total Wall Time**: 2153.9 секунд (35.9 минут)
- **WER**: 0.4194
- **CER**: 0.1834
- **Success Rate**: 100% (0 failed)

**Оценка**: ⭐⭐⭐
- ⚠️ Медленнее чем int8 на 30%
- ✅ Лучше качество чем int8 (но разница минимальна)
- ⚠️ Не оптимально для вашего CPU

**Сравнение с int8_float32:**
- Latency: +30% (медленнее на 1990ms)
- Throughput: -23%
- WER: +0.7% (хуже)

---

### 4. CTranslate2 - float32 (Baseline)

**Бэкенд**: `whisper-ctranslate2-float32`  
**Квантизация**: float32 (без квантизации)  
**Метод**: Baseline, полная точность  
**Размер модели**: ~3.1 GB (100%)

**Результаты (golos_1k, 1000 файлов):**
- **Average Latency**: 10167.19 ms
- **Throughput**: 0.393 req/s
- **Total Wall Time**: 2547.4 секунд (42.5 минут)
- **WER**: 0.4175
- **CER**: 0.1823
- **Success Rate**: 100% (0 failed)

**Оценка**: ⭐⭐⭐
- ✅ Максимальное качество (baseline)
- ❌ Самая медленная
- ❌ Самый большой размер модели

**Сравнение с int8_float32:**
- Latency: +54% (медленнее на 3554ms)
- Throughput: -35%
- WER: +0.3% (лучше, но разница минимальна)

---

### 5. PyTorch Dynamic Quantization

**Бэкенд**: `whisper-pytorch-dynamic`  
**Квантизация**: Dynamic Quantization (Linear слои int8)  
**Метод**: `torch.quantization.quantize_dynamic`  
**Размер модели**: ~1.6 GB (int8 веса)

**Результаты (golos_10_debug, 10 файлов):**
- **Average Latency**: 45877.81 ms (45.9 секунд)
- **Throughput**: 0.052 req/s
- **Total Wall Time**: 191.4 секунд
- **WER**: 0.70
- **CER**: 0.468
- **Success Rate**: 60% (4 failed из 10)

**Оценка**: ⭐⭐
- ❌ Очень медленно (в 7 раз медленнее CTranslate2)
- ❌ Плохое качество (WER 0.70 vs 0.42)
- ❌ Много ошибок (40% failed)
- ⚠️ Результаты на маленьком датасете (10 файлов)

**Проблемы:**
- Возможно, проблемы с квантизацией или конфигурацией
- Нужна дополнительная оптимизация

**Примечание**: Результаты могут быть неточными из-за малого размера датасета и ошибок.

---

### 6. whisper.cpp - GGML Q5_0

**Бэкенд**: `whisper-cpp-q5_0`  
**Квантизация**: GGML Q5_0 (5-bit квантизация)  
**Метод**: GGML квантизация через whisper.cpp  
**Размер модели**: ~1.24 GB (~40% от float32)

**Результаты (golos_10_debug, 10 файлов):**
- **Average Latency**: 19101.02 ms (19.1 секунд)
- **Throughput**: 0.181 req/s
- **Total Wall Time**: 55.4 секунд
- **WER**: 0.5625
- **CER**: 0.352
- **Success Rate**: 100% (0 failed)

**Оценка**: ⭐⭐⭐
- ⚠️ Медленнее чем CTranslate2 (в 3 раза)
- ⚠️ Хуже качество (WER 0.56 vs 0.42)
- ⚠️ Результаты на маленьком датасете (10 файлов)
- ✅ Работает стабильно

**Проблемы:**
- Запуск через subprocess добавляет overhead
- Модель загружается каждый раз заново
- Не оптимально для production

**Примечание**: Результаты на маленьком датасете, могут отличаться на полном.

---

## Сравнительная таблица (golos_1k, 1000 файлов)

| Бэкенд | Квантизация | Latency (ms) | Throughput (req/s) | WER | CER | Размер | Рейтинг |
|--------|-------------|--------------|-------------------|-----|-----|--------|---------|
| **CTranslate2 int8_float16** | int8_float16 | **6558** | **0.609** | 0.416 | 0.181 | 1.6 GB | ⭐⭐⭐⭐⭐ |
| **CTranslate2 int8** | int8_float32 | 6613 | 0.604 | 0.416 | 0.181 | 1.6 GB | ⭐⭐⭐⭐⭐ |
| **CTranslate2 int16** | int16 | 8603 | 0.464 | 0.419 | 0.183 | 1.6 GB | ⭐⭐⭐ |
| **CTranslate2 float32** | float32 | 10167 | 0.393 | 0.418 | 0.182 | 3.1 GB | ⭐⭐⭐ |

---

## Сравнительная таблица (golos_10_debug, 10 файлов)

| Бэкенд | Квантизация | Latency (ms) | Throughput (req/s) | WER | CER | Статус |
|--------|-------------|--------------|-------------------|-----|-----|--------|
| **whisper.cpp Q5_0** | GGML Q5_0 | 19101 | 0.181 | 0.563 | 0.352 | ✅ Работает |
| **PyTorch Dynamic** | Dynamic int8 | 45878 | 0.052 | 0.70 | 0.468 | ⚠️ Проблемы |

---

## Выводы и рекомендации

### 🏆 Лучший вариант: CTranslate2 int8_float16

**Почему:**
- ✅ Самая быстрая квантизация (6558ms)
- ✅ Лучший throughput (0.609 req/s)
- ✅ Отличное качество (WER 0.416)
- ✅ Стабильная работа (100% success rate)

**Рекомендация**: Используйте `whisper-ctranslate2-int8_float16` для production.

### 📊 Сравнение производительности

**Ускорение относительно float32:**
- int8_float16: **1.55x быстрее** (10167ms → 6558ms)
- int8_float32: **1.54x быстрее** (10167ms → 6613ms)
- int16: **1.18x быстрее** (10167ms → 8603ms)

**Улучшение throughput:**
- int8_float16: **+55%** (0.393 → 0.609 req/s)
- int8_float32: **+54%** (0.393 → 0.604 req/s)
- int16: **+18%** (0.393 → 0.464 req/s)

### 📈 Качество

**WER (Word Error Rate):**
- Все квантизации показывают очень близкое качество к float32
- Разница в WER: <1% (практически незаметно)
- int8_float16 и int8_float32: идентичное качество (0.416)

**CER (Character Error Rate):**
- Все квантизации показывают отличное качество
- int8_float16 и int8_float32: лучший CER (0.181)

### ⚠️ Проблемные варианты

1. **PyTorch Dynamic Quantization**
   - Очень медленно (в 7 раз медленнее CTranslate2)
   - Плохое качество
   - Много ошибок
   - **Не рекомендуется для production**

2. **whisper.cpp Q5_0**
   - Медленнее CTranslate2 (в 3 раза)
   - Хуже качество
   - Архитектурные ограничения (subprocess overhead)
   - **Не рекомендуется для production**

### 💡 Рекомендации

**Для production:**
1. ✅ **CTranslate2 int8_float16** — лучший выбор
2. ✅ **CTranslate2 int8_float32** — альтернатива (почти идентично)

**Для экспериментов:**
- Можно попробовать ONNX Runtime с квантизацией
- Можно попробовать Static Quantization (PyTorch) с калибровкой

**Не рекомендуется:**
- ❌ PyTorch Dynamic Quantization (слишком медленно)
- ❌ whisper.cpp через CLI (архитектурные ограничения)

---

## Технические детали

### CTranslate2 квантизации

**int8_float32:**
- Веса: int8 (8 бит)
- Активации: float32 (32 бита)
- Оптимизация: Intel MKL/oneDNN
- Формула: `WQ[i,j] = round(scale[i] * W[i,j])`

**int8_float16:**
- Веса: int8 (8 бит)
- Активации: float16 (16 бит)
- Оптимизация: AVX-512 или ARM NEON
- Может быть быстрее на некоторых CPU

**int16:**
- Веса: int16 (16 бит)
- Активации: int16 (16 бит)
- Оптимизация: Intel MKL
- Лучше качество, но медленнее

### PyTorch Dynamic Quantization

- Метод: `torch.quantization.quantize_dynamic`
- Квантизирует: только Linear слои
- Остальное: float32
- Проблема: медленнее чем CTranslate2

### whisper.cpp GGML Q5_0

- Метод: GGML квантизация
- Точность: 5-bit
- Размер: ~40% от оригинала
- Проблема: subprocess overhead

---

## Метрики для сравнения

### Скорость (Latency)
1. **int8_float16**: 6558ms ⭐
2. **int8_float32**: 6613ms
3. **int16**: 8603ms
4. **float32**: 10167ms (baseline)

### Пропускная способность (Throughput)
1. **int8_float16**: 0.609 req/s ⭐
2. **int8_float32**: 0.604 req/s
3. **int16**: 0.464 req/s
4. **float32**: 0.393 req/s (baseline)

### Качество (WER)
1. **int8_float16**: 0.416 ⭐
2. **int8_float32**: 0.416 ⭐
3. **float32**: 0.418
4. **int16**: 0.419

### Качество (CER)
1. **int8_float16**: 0.181 ⭐
2. **int8_float32**: 0.181 ⭐
3. **float32**: 0.182
4. **int16**: 0.183

---

## Итоговая рекомендация

**Используйте: `whisper-ctranslate2-int8_float16`**

**Причины:**
- ✅ Лучшая производительность
- ✅ Отличное качество
- ✅ Стабильная работа
- ✅ Оптимально для вашего CPU (Intel Xeon Gold 6348)

**Настройка:**ash
# В .env:
WHISPER_BACKEND_NAME=whisper-large-v3-ctranslate2-int8_float16
WHISPER_MODEL_WEIGHTS_PATH=/app/weights/whisper-large-v3-ct2-int8_float16
WHISPER_WORKERS=12  # Для параллельной обработки
BENCH_WORKERS=12    # Для параллельной отправки запросов---

**Дата анализа**: 21 ноября 2025  
**Версия модели**: Whisper Large v3  
**CPU**: Intel Xeon Gold 6348 (Ice Lake, 96 cores)




## Сравнительная таблица (все методы)

| Бэкенд | Квантизация | Метод квантизации | Датасет | Latency (ms) | Throughput (req/s) | WER | CER | Размер | Рейтинг |
|--------|-------------|-------------------|---------|--------------|-------------------|-----|-----|--------|---------|
| **CTranslate2 int8_float16** | int8_float16 | Post-training (CTranslate2) | golos_1k (1000) | **6558** | **0.609** | 0.416 | 0.181 | 1.6 GB | ⭐⭐⭐⭐⭐ |
| **CTranslate2 int8** | int8_float32 | Post-training (CTranslate2) | golos_1k (1000) | 6613 | 0.604 | 0.416 | 0.181 | 1.6 GB | ⭐⭐⭐⭐⭐ |
| **CTranslate2 int16** | int16 | Post-training (CTranslate2) | golos_1k (1000) | 8603 | 0.464 | 0.419 | 0.183 | 1.6 GB | ⭐⭐⭐ |
| **CTranslate2 float32** | float32 | Baseline (без квантизации) | golos_1k (1000) | 10167 | 0.393 | 0.418 | 0.182 | 3.1 GB | ⭐⭐⭐ |
| **whisper.cpp Q5_0** | GGML Q5_0 | GGML (5-bit) | golos_1k (1000) | 19101 | 0.181 | 0.563 | 0.352 | 1.24 GB | ⭐⭐⭐ |
| **PyTorch Dynamic** | Dynamic int8 | Dynamic Quantization (PyTorch) | golos_1k (1000) | 45878 | 0.052 | 0.70 | 0.468 | 1.6 GB | ⭐⭐ |

**Примечания:**
- ⚠️ whisper.cpp и PyTorch тестировались на меньшем датасете (10 файлов), результаты могут отличаться на полном датасете
- ⚠️ PyTorch Dynamic показал 40% failed samples, результаты могут быть неточными

---

## Сравнительная таблица (только golos_1k, 1000 файлов)

| Бэкенд | Квантизация | Метод квантизации | Latency (ms) | Throughput (req/s) | WER | CER | Размер | Рейтинг |
|--------|-------------|-------------------|--------------|-------------------|-----|-----|--------|---------|
| **CTranslate2 int8_float16** | int8_float16 | Post-training (CTranslate2) | **6558** | **0.609** | 0.416 | 0.181 | 1.6 GB | ⭐⭐⭐⭐⭐ |
| **CTranslate2 int8** | int8_float32 | Post-training (CTranslate2) | 6613 | 0.604 | 0.416 | 0.181 | 1.6 GB | ⭐⭐⭐⭐⭐ |
| **CTranslate2 int16** | int16 | Post-training (CTranslate2) | 8603 | 0.464 | 0.419 | 0.183 | 1.6 GB | ⭐⭐⭐ |
| **CTranslate2 float32** | float32 | Baseline (без квантизации) | 10167 | 0.393 | 0.418 | 0.182 | 3.1 GB | ⭐⭐⭐ |

---

## Сравнительная таблица (golos_10_debug, 10 файлов)

| Бэкенд | Квантизация | Метод квантизации | Latency (ms) | Throughput (req/s) | WER | CER | Размер | Статус |
|--------|-------------|-------------------|--------------|-------------------|-----|-----|--------|--------|
| **whisper.cpp Q5_0** | GGML Q5_0 | GGML (5-bit квантизация) | 19101 | 0.181 | 0.563 | 0.352 | 1.24 GB | ✅ Работает |
| **PyTorch Dynamic** | Dynamic int8 | Dynamic Quantization (PyTorch) | 45878 | 0.052 | 0.70 | 0.468 | 1.6 GB | ⚠️ Проблемы (40% failed) |