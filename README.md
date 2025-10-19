# Whisper-Benchmark

Репозиторий предназначен для удобного оценивания различных whisper-ов. В частности он используется для оценки качества и скорости разных whisper, ускоренных на cpu.


## Quickstart

```bash
cp .env.example .env
```
todo: finish quickstart

### Отправка тестового аудио на API
Запустите нужный бэкенд. Пример:
```bash
./backends/template/start.sh
```
и выполните:
```bash
python debug_api.py <path to audio file>
```

## Структура репозитория:
- backends/ - папка с бекендами whisper. Для запуска бекенда должен быть файл backends/<exp_name>/start.sh, который соберет докер образ и стартует контейнер с бекендом модели. Порт для развертывания - переменная WHISPER_BACKEND_PORT в  глобальном .env файле. 
- benchmark/ - папка с кодом бенчмарка. Для старта бенчмарка должен быть benchmark/start.sh, который будет слать запросы к api развернутой модели. Результат будет записан в папку results/<exp_name_bench_name>. В глобальном .env файле нужно указать название бенчмарка (BENCH_NAME). Датасет должен лежать в benchmark/data/<bench_name>/. 
- weights_utils/ - папка, где будет лежать код экспериментов для получения весов. Подробнее смотри weights_utils/README.md

## Система оценивания whisper
Для оценки одной реализации whisper будет развернут бекенд этой модели. 

## Добавление новой модели
Код получения весов модели должен быть реализован в weights_utils/your_experiment_name. 

<br> Создадим папку с бекендом. 
```bash
cp -r backends/template backends/your_experiment_name
```
Там напишем код инференса модели. Достаточно поменять код model.py
Веса можно класть в любую папку, а затем указать ее в .env (WHISPER_MODEL_WEIGHTS_PATH)

Теперь стартуем модель
```bash
./backends/your_experiment_name/start.sh
```
Чтобы удостовериться, что все отработало корретно можно кинуть debug запрос через
```
python3 debug_api.py <audio file path>
```


## Структура датасетов
Датасет должен лежать в benchmark/data/<bench_name>/. 
Аудиофайлы должны лежать в папке benchmark/data/<bench_name>/files. Разметка - в файле benchmark/data/<bench_name>/annotation.csv со следующей структурой:
filename, text

<br> Бенчмарк (bench_name) указывается в .env (переменная BENCH_NAME)

