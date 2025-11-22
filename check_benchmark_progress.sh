#!/bin/bash
echo "=== Проверка прогресса бенчмарка ==="
echo ""
echo "📊 Screen сессия:"
screen -ls | grep whisper_benchmark || echo "❌ Screen сессия не найдена"
echo ""
echo "🔥 Процессы Python:"
ps aux | grep "run_benchmark.py" | grep -v grep || echo "❌ Процесс не найден"
echo ""
echo "📁 Папка с результатами:"
LATEST_DIR=$(ls -td results/whisper-large-v3_golos_1k/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_DIR" ]; then
    echo "✅ Найдена: $LATEST_DIR"
    if [ -f "${LATEST_DIR}summary.txt" ]; then
        echo ""
        echo "📄 Результаты (готово!):"
        cat "${LATEST_DIR}summary.txt"
    else
        echo "⏳ Бенчмарк еще выполняется..."
    fi
else
    echo "⏳ Результаты еще не созданы, бенчмарк выполняется..."
fi
echo ""
echo "📝 Последние 5 строк лога:"
tail -5 benchmark_large_v3.log 2>/dev/null || echo "Лог пока пуст"
