# CTranslate2 Whisper Conversion

Скрипт для конвертации Whisper моделей в формат CTranslate2 с различными типами квантизации для CPU инференса.

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Использование

### Базовое использование

Конвертировать openai/whisper-tiny со всеми типами квантизации (float32, int8_float32, int16):

```bash
python convert_whisper.py --output-dir ../../weights/ctranslate2
```

### Выбор конкретных типов квантизации

```bash
python convert_whisper.py \
  --output-dir ../../weights/ctranslate2 \
  --quantizations int8_float32 int16
```

### Конвертация другой модели

```bash
python convert_whisper.py \
  --model openai/whisper-base \
  --output-dir ../../weights/ctranslate2 \
  --quantizations float32 int8_float32
```

## Поддерживаемые типы квантизации для CPU

### int8_float32
- **Рекомендуется для**: x86-64 CPU с Intel MKL или oneDNN
- **Размер модели**: ~100MB (для базовой модели)
- **Производительность**: Значительно быстрее float32 на поддерживаемых CPU
- **Качество**: Минимальное снижение (обычно незаметное)

Согласно [документации CTranslate2](https://opennmt.net/CTranslate2/quantization.html), int8 квантизация использует симметричную квантизацию весов embedding и linear слоев.

### int16
- **Рекомендуется для**: Intel CPU с Intel MKL backend
- **Размер модели**: ~187MB (для базовой модели)
- **Производительность**: Умеренное ускорение на Intel CPU
- **Качество**: Очень высокое (близко к float32)

### float32
- **Baseline**: Полная точность без квантизации
- **Размер модели**: ~364MB (для базовой модели)
- **Производительность**: Базовая (для сравнения)
- **Качество**: Максимальное

## Структура выходных файлов

После конвертации будут созданы следующие директории:

```
weights/ctranslate2/
├── whisper-tiny-ct2-float32/
│   ├── config.json
│   ├── model.bin
│   └── tokenizer.json
├── whisper-tiny-ct2-int8/
│   ├── config.json
│   ├── model.bin
│   └── tokenizer.json
└── whisper-tiny-ct2-int16/
    ├── config.json
    ├── model.bin
    └── tokenizer.json
```

## Использование сконвертированных моделей

После конвертации используйте соответствующие backends для запуска моделей:

### Float32
```bash
WHISPER_MODEL_WEIGHTS_PATH=/app/weights/whisper-tiny-ct2-float32 \
  ./backends/whisper-ctranslate2-float32/start.sh
```

### Int8
```bash
WHISPER_MODEL_WEIGHTS_PATH=/app/weights/whisper-tiny-ct2-int8 \
  ./backends/whisper-ctranslate2-int8/start.sh
```

### Int16
```bash
WHISPER_MODEL_WEIGHTS_PATH=/app/weights/whisper-tiny-ct2-int16 \
  ./backends/whisper-ctranslate2-int16/start.sh
```

## Бенчмаркинг

Для запуска бенчмарка на всех квантизациях:

1. Запустите нужный backend
2. Выполните бенчмарк:
```bash
./benchmark/start.sh
```

## Примечания

- **Неявная конвертация типов**: CTranslate2 может автоматически конвертировать типы при загрузке, если текущая платформа не поддерживает оптимизированное выполнение для выбранного типа. См. [документацию](https://opennmt.net/CTranslate2/quantization.html#implicit-type-conversion-on-load).

- **Проверка поддержки**: Для проверки поддерживаемых типов на вашей системе включите info логи:
  ```bash
  CT2_VERBOSE=1 python convert_whisper.py --output-dir ./weights
  ```

- **Размер модели на диске**: Квантизация значительно уменьшает размер:
  - float32: 100%
  - int16: ~51%
  - int8_float32: ~27%

## Справка

Для получения полной справки:

```bash
python convert_whisper.py --help
```

