# Whisper-Benchmark

Репозиторий предназначен для удобного оценивания различных whisper-ов. В частности он используется для оценки качества и скорости разных whisper, ускоренных на cpu.


## Quickstart

```bash
cp .env.example .env
```
todo: finish quickstart

## Структура репозитория:
- backends/ - папка с бекендами whisper. Для запуска бекенда должен быть файл backends/<exp_name>/start.sh, который соберет докер образ и стартует контейнер с бекендом модели. Порт для развертывания - переменная WHISPER_BACKEND_PORT в  глобальном .env файле. 
- benchmark/ - папка с кодом бенчмарка. Для старта бенчмарка должен быть benchmark/start.sh, который будет слать запросы к api развернутой модели. Результат будет записан в папку results/<exp_name_bench_name>. В глобальном .env файле нужно указать название бенчмарка (BENCH_NAME). Датасет должен лежать в benchmark/data/<bench_name>/. 

## Система оценивания whisper
Для оценки одной реализации whisper будет развернут бекенд этой модели. 

## Структура датасетов
Датасет должен лежать в benchmark/data/<bench_name>/. 
Аудиофайлы должны лежать в папке benchmark/data/<bench_name>/files. Разметка - в файле benchmark/data/<bench_name>/annotation.csv со следующей структурой:
filename, text


