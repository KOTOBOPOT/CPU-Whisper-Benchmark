# Setup Guide для C++ ONNX Backend

Эта директория содержит C++ бэкенд для Whisper с ONNX Runtime. ONNX модели не включены в git из-за их размера (2.9 GB).

## 🚀 Быстрый старт

### 1. Конвертация моделей

Используйте `convert_to_onnx.py` для конвертации моделей из HuggingFace в ONNX:

```bash
# Установка зависимостей
conda activate fish-speech  # или ваше окружение
pip install -r requirements_convert.txt

# Конвертация whisper-base (по умолчанию)
python convert_to_onnx.py \
  --model-id openai/whisper-base \
  --output models/whisper-base-onnx \
  --use-optimum

# Конвертация whisper-small
python convert_to_onnx.py \
  --model-id openai/whisper-small \
  --output models/whisper-small-onnx \
  --use-optimum

# Конвертация whisper-tiny
python convert_to_onnx.py \
  --model-id openai/whisper-tiny \
  --output models/whisper-tiny-onnx \
  --use-optimum
```

### 2. Создание suppress_tokens.json

После конвертации создайте файл с suppress tokens:

```bash
python -c "
import json
from pathlib import Path

model_dir = Path('models/whisper-small-onnx')  # измените на нужную модель
config_path = model_dir / 'generation_config.json'

with open(config_path) as f:
    config = json.load(f)

output = {
    'suppress_tokens': config.get('suppress_tokens', []),
    'begin_suppress_tokens': config.get('begin_suppress_tokens', [])
}

with open(model_dir / 'suppress_tokens.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'Created {model_dir}/suppress_tokens.json')
"
```

### 3. Загрузка датасета (опционально)

Для бенчмарка нужен датасет Golos:

```bash
# Скачивание через wget или curl
wget https://path-to-golos-dataset/golos_1k.zip -O data/golos_1k.zip

# Извлечение
unzip data/golos_1k.zip -d data/golos_1k_extracted/
```

## 📦 Результат конвертации

После конвертации в `models/whisper-small-onnx/` будет:

```
whisper-small-onnx/
├── encoder_model.onnx              # 336 MB - обработка mel-спектрограмм
├── decoder_model.onnx              # 738 MB - первый токен (создает KV-cache)
├── decoder_with_past_model.onnx   # 684 MB - последующие токены (использует cache)
├── config.json
├── generation_config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── vocab.json
├── tokenizer.json
├── merges.txt
├── special_tokens_map.json
├── added_tokens.json
├── normalizer.json
└── suppress_tokens.json           # создается отдельно
```

## 🐳 Запуск Docker

```bash
# Сборка и запуск
WHISPER_MODEL_PATH=/app/models/whisper-small-onnx docker compose up -d --build

# Проверка логов
docker compose logs -f

# Быстрый тест
python quick_test.py
```

## 📊 Размеры моделей

| Модель | Encoder | Decoder | Decoder w/past | Total |
|--------|---------|---------|----------------|-------|
| **tiny**  | 31 MB   | 189 MB  | 184 MB         | **404 MB** |
| **base**  | 108 MB  | 270 MB  | 254 MB         | **632 MB** |
| **small** | 336 MB  | 738 MB  | 684 MB         | **1.76 GB** |
| **medium**| 648 MB  | 1.5 GB  | 1.4 GB         | **3.5 GB** ⚠️ |
| **large** | 1.0 GB  | 2.3 GB  | 2.2 GB         | **5.5 GB** ⚠️ |

⚠️ Модели medium и large требуют >16GB RAM для конвертации.

## 🔧 Системные требования

### Для конвертации:
- Python 3.10+
- RAM: 8GB (tiny/base), 16GB (small), 32GB+ (medium/large)
- Disk: 2-6 GB на модель

### Для инференса:
- ONNX Runtime 1.17+
- C++17 compiler
- CMake 3.15+
- RAM: 2-4 GB

## 📚 Документация

- [C++ ONNX Architecture](../../docs/cpp_onnx_architecture.md) - Mermaid схемы
- [Model API](../../docs/model_api.md) - API документация
- [Benchmark Results](../../results/) - Результаты бенчмарков

## ❓ FAQ

**Q: Почему модели не в git?**  
A: ONNX модели занимают 2.9 GB и должны быть сконвертированы локально.

**Q: Можно ли скачать готовые ONNX модели?**  
A: Да, можно попробовать `onnx-community` на HuggingFace, но рекомендуется конвертировать самостоятельно для полного контроля.

**Q: Почему три ONNX файла?**  
A: Для оптимизации:
- `encoder_model.onnx` - обрабатывает аудио один раз
- `decoder_model.onnx` - первый токен, создает KV-cache
- `decoder_with_past_model.onnx` - последующие токены, использует cache (40% быстрее!)

**Q: Out of Memory при конвертации large/medium?**  
A: Используйте меньшую модель (small/base) или добавьте swap памяти.

## 🤝 Контрибьюция

При добавлении новых моделей:
1. Конвертируйте модель локально
2. НЕ коммитьте `.onnx` файлы
3. Обновите эту инструкцию
4. Добавьте пример конвертации

