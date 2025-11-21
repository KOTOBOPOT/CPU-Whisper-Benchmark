"""Whisper model implementation using whisper.cpp with GGML quantization."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import soundfile as sf


class WhisperModel:
    """Whisper model using whisper.cpp with GGML quantization."""

    def __init__(self, weights_path: Optional[Path], whisper_cpp_bin: Optional[Path] = None) -> None:
        self.weights_path = weights_path
        
        # Путь к whisper-cli бинарнику
        if whisper_cpp_bin is None:
            # Попробуем найти в стандартных местах
            possible_paths = [
                Path(os.getenv("WHISPER_CPP_BIN", "")),
                Path("/usr/local/bin/whisper-cli"),
                Path("/app/whisper-cli"),
                Path(__file__).parent.parent.parent.parent / "whisper.cpp" / "build" / "bin" / "whisper-cli",
                Path(__file__).parent.parent.parent.parent / "whisper.cpp" / "whisper-cli",
            ]
            
            for path in possible_paths:
                if path and path.exists():
                    whisper_cpp_bin = path
                    break
            else:
                # Попробуем найти в PATH
                try:
                    result = subprocess.run(
                        ["which", "whisper-cli"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        whisper_cpp_bin = Path(result.stdout.strip())
                except Exception:
                    pass
                
                if not whisper_cpp_bin or not whisper_cpp_bin.exists():
                    raise ValueError(
                        "whisper-cli not found. Please set WHISPER_CPP_BIN environment variable "
                        "or ensure whisper-cli is in PATH"
                    )
        
        self.whisper_cpp_bin = whisper_cpp_bin
        
        # Validate paths
        if weights_path is None or not weights_path.exists():
            raise ValueError(f"Model weights not found at {weights_path}")
        
        if not self.whisper_cpp_bin.exists():
            raise ValueError(f"whisper-cli not found at {self.whisper_cpp_bin}")
        
        logging.info("Using whisper.cpp model: %s", weights_path)
        logging.info("whisper-cli binary: %s", self.whisper_cpp_bin)
        
        # Проверка доступности бинарника
        try:
            result = subprocess.run(
                [str(self.whisper_cpp_bin), "-h"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                logging.warning("whisper-cli may not be working correctly")
        except Exception as e:
            logging.warning("Could not verify whisper-cli: %s", e)

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe audio using whisper.cpp.
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename for logging
            
        Returns:
            Transcribed text as string
        """
        tmp_path = None
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            # Convert to 16kHz WAV if needed using librosa
            try:
                audio_array, sr = librosa.load(tmp_path, sr=16000)
                # Сохраняем обратно как WAV
                sf.write(tmp_path, audio_array, sr)
            except Exception as e:
                logging.warning("Could not preprocess audio with librosa: %s", e)
                # Продолжаем с оригинальным файлом
            
            # Определяем количество потоков (можно настроить через переменную окружения)
            num_threads = int(os.getenv("WHISPER_CPP_THREADS", "12"))
            
            # Run whisper-cli
            # Оптимизация для скорости: используем greedy decoding (beam-size=1, best-of=1)
            # Это соответствует CTranslate2 с beam_size=1
            # Дополнительные оптимизации:
            # - -nf: no fallback (не использовать temperature fallback)
            # - -ac 0: audio context = 0 (использовать весь контекст, но быстрее)
            cmd = [
                str(self.whisper_cpp_bin),
                "-m", str(self.weights_path),
                "-f", tmp_path,
                "-l", "ru",  # Russian language
                "-t", str(num_threads),  # Threads
                "-bs", "1",  # beam-size=1 (greedy decoding для скорости)
                "-bo", "1",  # best-of=1 (только один кандидат)
                "-nf",  # no-fallback (не использовать temperature fallback)
            ]
            
            logging.info("Running command: %s", " ".join(cmd))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 минут таймаут
                check=False
            )
            
            if result.returncode != 0:
                error_msg = f"whisper-cli failed with code {result.returncode}: {result.stderr}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Parse output
            # whisper-cli выводит транскрипцию в stdout
            # Формат может быть разным, попробуем несколько способов парсинга
            stdout_text = result.stdout.strip()
            stderr_text = result.stderr.strip()
            
            transcription = ""
            
            # Способ 1: Ищем текст после временных меток [XX:XX.XXX --> XX:XX.XXX]
            # Whisper-cli может выводить: [00:00.000 --> 00:05.000]  текст
            # Паттерн для временных меток
            timestamp_pattern = r'\[\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}\.\d{3}\]\s*(.+)'
            matches = re.findall(timestamp_pattern, stdout_text, re.MULTILINE)
            if matches:
                # Берем весь текст из всех сегментов
                transcription = " ".join(m.strip() for m in matches if m.strip())
            
            # Способ 2: Если не нашли по паттерну, берем все строки, исключая служебные
            if not transcription:
                lines = stdout_text.split('\n')
                valid_lines = []
                for line in lines:
                    line = line.strip()
                    # Пропускаем служебные строки
                    if (line and 
                        not line.startswith('[') and 
                        not line.startswith('whisper') and
                        not line.startswith('system_info') and
                        not line.startswith('whisper_model_') and
                        not '-->' in line or '-->' in line):  # Включаем строки с временными метками
                        # Если есть временная метка, извлекаем текст после неё
                        if '-->' in line:
                            parts = line.split(']', 1)
                            if len(parts) > 1:
                                valid_lines.append(parts[1].strip())
                        else:
                            valid_lines.append(line)
                
                if valid_lines:
                    transcription = " ".join(valid_lines)
            
            # Способ 3: Если всё ещё пусто, пробуем взять всё кроме первых служебных строк
            if not transcription and stdout_text:
                lines = stdout_text.split('\n')
                # Пропускаем первые несколько строк (обычно служебная информация)
                for line in lines[3:]:  # Пропускаем первые 3 строки
                    line = line.strip()
                    if line and not line.startswith('whisper') and not line.startswith('system'):
                        # Извлекаем текст после временной метки если есть
                        if ']' in line:
                            parts = line.split(']', 1)
                            if len(parts) > 1:
                                transcription += " " + parts[1].strip()
                        else:
                            transcription += " " + line
                transcription = transcription.strip()
            
            # Очистка от служебных символов
            transcription = transcription.replace('[BLANK_AUDIO]', '').strip()
            transcription = re.sub(r'\[\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}\.\d{3}\]', '', transcription).strip()
            
            if not transcription:
                # Всегда логируем при пустой транскрипции для отладки
                logging.warning("Empty transcription for %s", filename)
                logging.warning("whisper-cli return code: %d", result.returncode)
                logging.warning("whisper-cli stdout (first 1000 chars): %s", stdout_text[:1000] if stdout_text else "(empty)")
                logging.warning("whisper-cli stderr (first 1000 chars): %s", stderr_text[:1000] if stderr_text else "(empty)")
                return ""
            
            logging.info("Successfully transcribed %s (length: %d chars)", filename, len(transcription))
            return transcription
            
        except subprocess.TimeoutExpired:
            error_msg = f"whisper-cli timeout for {filename}"
            logging.error(error_msg)
            return f"[Transcription failed: timeout]"
        except Exception as e:
            error_msg = f"Error transcribing {filename}: {str(e)}"
            logging.error(error_msg)
            return f"[Transcription failed: {str(e)}]"
        finally:
            # Cleanup
            if tmp_path and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                except Exception as e:
                    logging.warning("Could not delete temp file %s: %s", tmp_path, e)
