📊 ЗАПУСК И МОНИТОРИНГ БЕНЧМАРКА
=================================

🚀 ЗАПУСК БЕНЧМАРКА В SCREEN
-----------------------------
# Запустить бенчмарк в фоне (не блокирует терминал)
screen -dmS whisper_benchmark bash -c "cd /opt/whisper_test/CPU-Whisper-Benchmark && ./benchmark/start.sh 2>&1 | tee benchmark_large_v3.log"

# Проверить, что сессия создана
screen -ls

📊 МОНИТОРИНГ
-------------

🔥 Смотреть логи в реальном времени:
./watch_benchmark.sh

или просто:
tail -f benchmark_large_v3.log

📈 Быстрая проверка статуса:
./check_benchmark_progress.sh

🎯 Подключиться к screen (видеть прогресс-бар):
screen -r whisper_benchmark
Выйти: Ctrl+A, затем D

🛑 ОСТАНОВИТЬ БЕНЧМАРК (если нужно):
screen -S whisper_benchmark -X quit

📁 РЕЗУЛЬТАТЫ
-------------
# Когда закончится, смотреть результаты
cat results/whisper-large-v3_golos_1k/*/summary.txt

# Сравнить с whisper-small
cat results/whisper-small_golos_1k/*/summary.txt
