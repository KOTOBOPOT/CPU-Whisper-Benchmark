#!/usr/bin/env python3
"""
Сравнение результатов C++ ONNX vs Python Whisper Backend
Анализ на датасете Golos-1k (1000 samples)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

def load_data():
    """Загрузка данных из результатов бенчмарков"""
    print("📂 Загрузка данных...")
    
    # Пути к результатам
    cpp_metrics_path = Path('results/whisper-small-cpp_golos_1k/20251122T101016Z/metrics.json')
    python_metrics_path = Path('results/whisper-small_golos_1k/20251121T172439Z/metrics.json')
    
    cpp_predictions_path = Path('results/whisper-small-cpp_golos_1k/20251122T101016Z/predictions.csv')
    python_predictions_path = Path('results/whisper-small_golos_1k/20251121T172439Z/predictions.csv')
    
    # Загрузка метрик
    with open(cpp_metrics_path) as f:
        cpp_metrics = json.load(f)
    
    with open(python_metrics_path) as f:
        python_metrics = json.load(f)
    
    # Загрузка детальных предсказаний
    cpp_df = pd.read_csv(cpp_predictions_path)
    python_df = pd.read_csv(python_predictions_path)
    
    print(f"✅ C++ Backend: {len(cpp_df)} samples")
    print(f"✅ Python Backend: {len(python_df)} samples")
    
    return cpp_metrics, python_metrics, cpp_df, python_df

def print_summary(cpp_metrics, python_metrics):
    """Вывод сводной таблицы"""
    print("\n" + "="*80)
    print("📊 СРАВНЕНИЕ МЕТРИК")
    print("="*80)
    
    print(f"\n{'Метрика':<25} {'Python':<20} {'C++ ONNX':<20} {'Улучшение':<15}")
    print("-"*80)
    
    # Processed Samples
    print(f"{'Processed Samples':<25} {python_metrics['processed_samples']:<20} {cpp_metrics['processed_samples']:<20} {'':<15}")
    
    # Failed Samples
    print(f"{'Failed Samples':<25} {python_metrics['failed_samples']:<20} {cpp_metrics['failed_samples']:<20} {'':<15}")
    
    # Latency
    latency_improvement = (1 - cpp_metrics['average_latency_ms']/python_metrics['average_latency_ms'])*100
    print(f"{'Avg Latency (ms)':<25} {python_metrics['average_latency_ms']:<20.1f} {cpp_metrics['average_latency_ms']:<20.1f} {latency_improvement:>12.1f}% ⚡")
    
    # WER
    wer_improvement = (1 - cpp_metrics['wer']/python_metrics['wer'])*100
    print(f"{'WER':<25} {python_metrics['wer']:<20.4f} {cpp_metrics['wer']:<20.4f} {wer_improvement:>12.2f}% ✅")
    
    # CER
    cer_improvement = (1 - cpp_metrics['cer']/python_metrics['cer'])*100
    print(f"{'CER':<25} {python_metrics['cer']:<20.4f} {cpp_metrics['cer']:<20.4f} {cer_improvement:>12.2f}% ✅")
    
    # Throughput
    python_throughput = 60000 / python_metrics['average_latency_ms']
    cpp_throughput = 60000 / cpp_metrics['average_latency_ms']
    throughput_improvement = (cpp_throughput/python_throughput - 1)*100
    print(f"{'Throughput (req/min)':<25} {python_throughput:<20.1f} {cpp_throughput:<20.1f} {throughput_improvement:>12.1f}% 📈")
    
    print("="*80)

def plot_latency_comparison(cpp_metrics, python_metrics, cpp_df, python_df):
    """Визуализация сравнения задержек"""
    print("\n📊 Создание графиков задержки...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    backends = ['Python\n(transformers)', 'C++\n(ONNX Runtime)']
    latencies = [python_metrics['average_latency_ms'], cpp_metrics['average_latency_ms']]
    colors = ['#3776ab', '#f34b7d']
    
    # Барчарт средней задержки
    bars = axes[0].bar(backends, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Средняя задержка (ms)', fontsize=12, fontweight='bold')
    axes[0].set_title('Сравнение средней задержки', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{latency:.0f} ms',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    improvement = (1 - cpp_metrics['average_latency_ms']/python_metrics['average_latency_ms'])*100
    axes[0].text(0.5, max(latencies)*0.5, f'⚡ {improvement:.1f}% быстрее',
                ha='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Гистограмма распределения
    cpp_latencies = cpp_df[cpp_df['status'] == 'ok']['latency_ms']
    python_latencies = python_df[python_df['status'] == 'ok']['latency_ms']
    
    axes[1].hist(python_latencies, bins=50, alpha=0.6, label='Python', color='#3776ab', edgecolor='black')
    axes[1].hist(cpp_latencies, bins=50, alpha=0.6, label='C++ ONNX', color='#f34b7d', edgecolor='black')
    axes[1].set_xlabel('Задержка (ms)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Количество', fontsize=12, fontweight='bold')
    axes[1].set_title('Распределение задержек', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/latency_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Сохранено: results/latency_comparison.png")
    plt.close()

def plot_accuracy_comparison(cpp_metrics, python_metrics):
    """Визуализация сравнения точности"""
    print("\n📊 Создание графиков точности...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    backends = ['Python\n(transformers)', 'C++\n(ONNX Runtime)']
    colors = ['#3776ab', '#f34b7d']
    
    # WER
    wer_data = [python_metrics['wer'], cpp_metrics['wer']]
    bars1 = axes[0].bar(backends, wer_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Word Error Rate (WER)', fontsize=12, fontweight='bold')
    axes[0].set_title('Сравнение точности (WER)', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, max(wer_data) * 1.2])
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar, wer in zip(bars1, wer_data):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{wer:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    wer_improvement = (1 - cpp_metrics['wer']/python_metrics['wer'])*100
    axes[0].text(0.5, max(wer_data)*0.5, f'✅ {wer_improvement:.2f}% лучше',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # CER
    cer_data = [python_metrics['cer'], cpp_metrics['cer']]
    bars2 = axes[1].bar(backends, cer_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Character Error Rate (CER)', fontsize=12, fontweight='bold')
    axes[1].set_title('Сравнение точности (CER)', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, max(cer_data) * 1.2])
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, cer in zip(bars2, cer_data):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{cer:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    cer_improvement = (1 - cpp_metrics['cer']/python_metrics['cer'])*100
    axes[1].text(0.5, max(cer_data)*0.5, f'✅ {cer_improvement:.2f}% лучше',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Сохранено: results/accuracy_comparison.png")
    plt.close()

def plot_comprehensive(cpp_metrics, python_metrics, cpp_df, python_df):
    """Комплексная визуализация"""
    print("\n📊 Создание комплексной визуализации...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    backends = ['Python\n(transformers)', 'C++\n(ONNX Runtime)']
    latencies = [python_metrics['average_latency_ms'], cpp_metrics['average_latency_ms']]
    wer_data = [python_metrics['wer'], cpp_metrics['wer']]
    cer_data = [python_metrics['cer'], cpp_metrics['cer']]
    colors = ['#3776ab', '#f34b7d']
    
    # 1. Latency
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(backends, latencies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Средняя задержка', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ms', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    for i, latency in enumerate(latencies):
        ax1.text(i, latency + 50, f'{latency:.0f}', ha='center', fontweight='bold')
    
    # 2. WER
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(backends, wer_data, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Word Error Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('WER', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    for i, wer in enumerate(wer_data):
        ax2.text(i, wer + 0.01, f'{wer:.4f}', ha='center', fontweight='bold', fontsize=9)
    
    # 3. CER
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(backends, cer_data, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title('Character Error Rate', fontsize=12, fontweight='bold')
    ax3.set_ylabel('CER', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    for i, cer in enumerate(cer_data):
        ax3.text(i, cer + 0.005, f'{cer:.4f}', ha='center', fontweight='bold', fontsize=9)
    
    # 4. Распределение latency
    ax4 = fig.add_subplot(gs[1, :])
    cpp_latencies = cpp_df[cpp_df['status'] == 'ok']['latency_ms']
    python_latencies = python_df[python_df['status'] == 'ok']['latency_ms']
    
    ax4.hist(python_latencies, bins=50, alpha=0.5, label='Python', color='#3776ab', edgecolor='black')
    ax4.hist(cpp_latencies, bins=50, alpha=0.5, label='C++ ONNX', color='#f34b7d', edgecolor='black')
    ax4.set_xlabel('Задержка (ms)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Количество', fontsize=11, fontweight='bold')
    ax4.set_title('Распределение задержек на 1000 samples', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Box plot
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.boxplot([python_latencies, cpp_latencies], labels=['Python', 'C++ ONNX'],
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax5.set_ylabel('Задержка (ms)', fontsize=10, fontweight='bold')
    ax5.set_title('Box Plot: Latency', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Throughput
    ax6 = fig.add_subplot(gs[2, 1])
    python_throughput = 60000 / python_metrics['average_latency_ms']
    cpp_throughput = 60000 / cpp_metrics['average_latency_ms']
    throughputs = [python_throughput, cpp_throughput]
    ax6.bar(backends, throughputs, color=colors, alpha=0.8, edgecolor='black')
    ax6.set_title('Throughput', fontsize=12, fontweight='bold')
    ax6.set_ylabel('requests/min', fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    for i, thr in enumerate(throughputs):
        ax6.text(i, thr + 0.5, f'{thr:.1f}', ha='center', fontweight='bold')
    
    # 7. Summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    improvement = (1 - cpp_metrics['average_latency_ms']/python_metrics['average_latency_ms'])*100
    wer_improvement = (1 - cpp_metrics['wer']/python_metrics['wer'])*100
    cer_improvement = (1 - cpp_metrics['cer']/python_metrics['cer'])*100
    
    summary_text = f"""
📊 ИТОГИ СРАВНЕНИЯ
━━━━━━━━━━━━━━━━━━━━━

Latency:
  {improvement:.1f}% быстрее ⚡

WER:
  {wer_improvement:.2f}% лучше ✅

CER:
  {cer_improvement:.2f}% лучше ✅

Throughput:
  {(cpp_throughput/python_throughput - 1)*100:.1f}% выше 📈

Успешность:
  Python: {python_metrics['processed_samples']}/{python_metrics['requested_samples']}
  C++: {cpp_metrics['processed_samples']}/{cpp_metrics['requested_samples']}
"""
    ax7.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Whisper-Small: C++ ONNX Runtime vs Python Transformers\nGolos-1k Dataset (1000 samples)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('results/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Сохранено: results/comprehensive_comparison.png")
    plt.close()

def save_summary(cpp_metrics, python_metrics, cpp_df, python_df):
    """Сохранение итогового summary"""
    print("\n📝 Сохранение summary...")
    
    python_throughput = 60000 / python_metrics['average_latency_ms']
    cpp_throughput = 60000 / cpp_metrics['average_latency_ms']
    
    improvement = (1 - cpp_metrics['average_latency_ms']/python_metrics['average_latency_ms'])*100
    wer_improvement = (1 - cpp_metrics['wer']/python_metrics['wer'])*100
    cer_improvement = (1 - cpp_metrics['cer']/python_metrics['cer'])*100
    
    # Расчет speedup
    cpp_df_clean = cpp_df[cpp_df['status'] == 'ok'].copy()
    python_df_clean = python_df[python_df['status'] == 'ok'].copy()
    merged_df = pd.merge(
        cpp_df_clean[['filename', 'latency_ms']],
        python_df_clean[['filename', 'latency_ms']],
        on='filename',
        suffixes=('_cpp', '_python')
    )
    merged_df['speedup'] = merged_df['latency_ms_python'] / merged_df['latency_ms_cpp']
    
    summary = {
        'Experiment': 'Whisper-Small на Golos-1k (1000 samples)',
        'Python Backend': {
            'Average Latency (ms)': round(python_metrics['average_latency_ms'], 1),
            'WER': round(python_metrics['wer'], 4),
            'CER': round(python_metrics['cer'], 4),
            'Throughput (req/min)': round(python_throughput, 1),
            'Processed': python_metrics['processed_samples'],
            'Failed': python_metrics['failed_samples']
        },
        'C++ ONNX Backend': {
            'Average Latency (ms)': round(cpp_metrics['average_latency_ms'], 1),
            'WER': round(cpp_metrics['wer'], 4),
            'CER': round(cpp_metrics['cer'], 4),
            'Throughput (req/min)': round(cpp_throughput, 1),
            'Processed': cpp_metrics['processed_samples'],
            'Failed': cpp_metrics['failed_samples']
        },
        'Improvements': {
            'Latency': f"{improvement:.1f}% faster",
            'WER': f"{wer_improvement:.2f}% better",
            'CER': f"{cer_improvement:.2f}% better",
            'Throughput': f"{(cpp_throughput/python_throughput - 1)*100:.1f}% higher",
            'Average Speedup': f"{merged_df['speedup'].mean():.2f}x"
        }
    }
    
    with open('results/benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("   ✅ Сохранено: results/benchmark_summary.json")

def main():
    """Главная функция"""
    print("\n" + "="*80)
    print("🚀 АНАЛИЗ РЕЗУЛЬТАТОВ: C++ ONNX vs Python Whisper Backend")
    print("="*80)
    
    # Загрузка данных
    cpp_metrics, python_metrics, cpp_df, python_df = load_data()
    
    # Вывод сводки
    print_summary(cpp_metrics, python_metrics)
    
    # Создание графиков
    plot_latency_comparison(cpp_metrics, python_metrics, cpp_df, python_df)
    plot_accuracy_comparison(cpp_metrics, python_metrics)
    plot_comprehensive(cpp_metrics, python_metrics, cpp_df, python_df)
    
    # Сохранение summary
    save_summary(cpp_metrics, python_metrics, cpp_df, python_df)
    
    print("\n" + "="*80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН!")
    print("="*80)
    print("\n📁 Созданные файлы:")
    print("   - results/latency_comparison.png")
    print("   - results/accuracy_comparison.png")
    print("   - results/comprehensive_comparison.png")
    print("   - results/benchmark_summary.json")
    print("\n📄 Документация:")
    print("   - docs/cpp_onnx_architecture.md (Mermaid схемы)")
    print("\n")

if __name__ == "__main__":
    main()

