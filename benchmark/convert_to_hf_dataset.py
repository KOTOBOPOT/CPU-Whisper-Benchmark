#!/usr/bin/env python
# coding=utf-8
"""
Скрипт для конвертации CSV + audio files в формат HuggingFace Dataset
для использования с distil-whisper training scripts.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import datasets
from datasets import Audio, Dataset, DatasetDict, Features, Value


def load_csv_annotation(csv_path: Path) -> List[Dict[str, str]]:
    """Загрузить CSV аннотацию."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'filename': row['filename'],
                'text': row['text']
            })
    return rows


def create_dataset_from_csv(
    csv_path: Path,
    audio_dir: Path,
    output_dir: Path,
    sampling_rate: int = 16000,
    split_name: str = "train"
) -> None:
    """
    Создать HuggingFace Dataset из CSV и аудио файлов.
    
    Args:
        csv_path: путь к CSV файлу с аннотацией
        audio_dir: директория с аудио файлами
        output_dir: директория для сохранения датасета
        sampling_rate: частота дискретизации для аудио
        split_name: название сплита (train/validation/test)
    """
    print(f"Загрузка аннотации из {csv_path}...")
    rows = load_csv_annotation(csv_path)
    print(f"Найдено {len(rows)} записей")
    
    # Подготовка данных для датасета
    data = {
        'audio': [],
        'text': [],
        'id': []
    }
    
    missing_files = []
    for idx, row in enumerate(rows):
        filename = row['filename']
        # Попробуем найти файл с разными расширениями
        audio_path = None
        for ext in ['.opus', '.wav', '.mp3', '.flac']:
            potential_path = audio_dir / f"{filename}{ext}" if not filename.endswith(ext) else audio_dir / filename
            if potential_path.exists():
                audio_path = str(potential_path)
                break
        
        if audio_path is None:
            missing_files.append(filename)
            continue
        
        data['audio'].append(audio_path)
        data['text'].append(row['text'])
        data['id'].append(filename)
    
    if missing_files:
        print(f"ВНИМАНИЕ: Не найдено {len(missing_files)} аудио файлов:")
        for f in missing_files[:10]:  # Показываем первые 10
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... и ещё {len(missing_files) - 10} файлов")
    
    print(f"Обработано {len(data['audio'])} записей")
    
    # Создание датасета
    features = Features({
        'audio': Audio(sampling_rate=sampling_rate),
        'text': Value('string'),
        'id': Value('string')
    })
    
    print("Создание HuggingFace Dataset...")
    dataset = Dataset.from_dict(data, features=features)
    
    # Создание DatasetDict для поддержки сплитов
    dataset_dict = DatasetDict({split_name: dataset})
    
    # Сохранение
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Сохранение датасета в {output_path}...")
    dataset_dict.save_to_disk(output_path)
    
    print(f"✓ Датасет успешно создан!")
    print(f"  - Количество записей: {len(dataset)}")
    print(f"  - Сплит: {split_name}")
    print(f"  - Колонки: {dataset.column_names}")
    print(f"\nТеперь вы можете использовать этот датасет с distil-whisper скриптами:")
    print(f"  --train_dataset_name '{output_path}'")
    print(f"  --train_split_name '{split_name}'")
    print(f"  --text_column_name 'text'")
    print(f"  --streaming False")


def create_multi_split_dataset(
    data_dir: Path,
    output_dir: Path,
    train_csv: str = None,
    val_csv: str = None,
    test_csv: str = None,
    audio_subdir: str = "files",
    sampling_rate: int = 16000
) -> None:
    """
    Создать датасет с несколькими сплитами из одной директории.
    
    Args:
        data_dir: директория с данными
        output_dir: директория для сохранения датасета
        train_csv: имя CSV файла для train сплита
        val_csv: имя CSV файла для validation сплита
        test_csv: имя CSV файла для test сплита
        audio_subdir: поддиректория с аудио файлами
        sampling_rate: частота дискретизации
    """
    audio_dir = data_dir / audio_subdir
    
    if not audio_dir.exists():
        raise ValueError(f"Директория с аудио не найдена: {audio_dir}")
    
    dataset_dict = {}
    
    splits_config = {
        'train': train_csv,
        'validation': val_csv,
        'test': test_csv
    }
    
    for split_name, csv_name in splits_config.items():
        if csv_name is None:
            continue
        
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            print(f"Предупреждение: CSV файл не найден для {split_name}: {csv_path}")
            continue
        
        print(f"\n=== Обработка сплита '{split_name}' ===")
        rows = load_csv_annotation(csv_path)
        print(f"Найдено {len(rows)} записей")
        
        data = {
            'audio': [],
            'text': [],
            'id': []
        }
        
        missing_files = []
        for row in rows:
            filename = row['filename']
            audio_path = None
            for ext in ['.opus', '.wav', '.mp3', '.flac']:
                potential_path = audio_dir / f"{filename}{ext}" if not filename.endswith(ext) else audio_dir / filename
                if potential_path.exists():
                    audio_path = str(potential_path)
                    break
            
            if audio_path is None:
                missing_files.append(filename)
                continue
            
            data['audio'].append(audio_path)
            data['text'].append(row['text'])
            data['id'].append(filename)
        
        if missing_files:
            print(f"ВНИМАНИЕ: Не найдено {len(missing_files)} аудио файлов для {split_name}")
        
        features = Features({
            'audio': Audio(sampling_rate=sampling_rate),
            'text': Value('string'),
            'id': Value('string')
        })
        
        dataset_dict[split_name] = Dataset.from_dict(data, features=features)
        print(f"✓ Создан сплит '{split_name}': {len(dataset_dict[split_name])} записей")
    
    if not dataset_dict:
        raise ValueError("Не создано ни одного сплита! Проверьте пути к CSV файлам.")
    
    # Сохранение
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    final_dataset = DatasetDict(dataset_dict)
    print(f"\nСохранение датасета в {output_path}...")
    final_dataset.save_to_disk(output_path)
    
    print(f"\n✓ Датасет успешно создан!")
    print(f"  - Сплиты: {list(final_dataset.keys())}")
    for split_name, split_dataset in final_dataset.items():
        print(f"    - {split_name}: {len(split_dataset)} записей")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация CSV + аудио файлов в HuggingFace Dataset для distil-whisper"
    )
    
    # Общие аргументы
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Директория с CSV и аудио файлами (например, benchmark/data/golos_10_debug)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Директория для сохранения HuggingFace датасета"
    )
    parser.add_argument(
        "--audio_subdir",
        type=str,
        default="files",
        help="Поддиректория с аудио файлами (по умолчанию: 'files')"
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="Частота дискретизации для аудио (по умолчанию: 16000)"
    )
    
    # Режимы работы
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--single_split",
        action="store_true",
        help="Создать датасет с одним сплитом"
    )
    mode_group.add_argument(
        "--multi_split",
        action="store_true",
        help="Создать датасет с несколькими сплитами"
    )
    
    # Аргументы для single split
    parser.add_argument(
        "--csv_file",
        type=str,
        default="annotation.csv",
        help="Имя CSV файла (для single_split режима)"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Название сплита (для single_split режима)"
    )
    
    # Аргументы для multi split
    parser.add_argument(
        "--train_csv",
        type=str,
        help="Имя CSV файла для train сплита"
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        help="Имя CSV файла для validation сплита"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        help="Имя CSV файла для test сплита"
    )
    
    args = parser.parse_args()
    
    if args.single_split:
        csv_path = args.data_dir / args.csv_file
        audio_dir = args.data_dir / args.audio_subdir
        
        if not csv_path.exists():
            raise ValueError(f"CSV файл не найден: {csv_path}")
        if not audio_dir.exists():
            raise ValueError(f"Директория с аудио не найдена: {audio_dir}")
        
        create_dataset_from_csv(
            csv_path=csv_path,
            audio_dir=audio_dir,
            output_dir=args.output_dir,
            sampling_rate=args.sampling_rate,
            split_name=args.split_name
        )
    else:  # multi_split
        create_multi_split_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            audio_subdir=args.audio_subdir,
            sampling_rate=args.sampling_rate
        )


if __name__ == "__main__":
    main()

