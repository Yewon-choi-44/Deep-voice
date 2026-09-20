"""
baseline_submit/script.py 의 오디오 처리 함수를 그대로 옮겨온 공용 모듈.

일부러 baseline 코드를 복붙 수준으로 재현합니다 — 학습 데이터를 만들 때와
실제 대회 추론 시점에 오디오가 잘리는 방식(세그먼트 분할 규칙)이 한 글자라도
다르면, 그 자체가 train-test distribution shift의 원인이 되기 때문입니다.
"""

from pathlib import Path

import librosa
import numpy as np

import config


def load_audio(audio_path: Path) -> np.ndarray:
    """baseline script.py의 load_audio와 동일. 16kHz mono float32로 로드."""
    audio, _ = librosa.load(
        audio_path, sr=config.AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32
    )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {audio_path}")
    return audio


def get_segment_starts(audio_length: int) -> list[int]:
    """baseline script.py의 get_segment_starts와 동일."""
    if audio_length <= config.SEGMENT_SAMPLES:
        return [0]

    last_start = audio_length - config.SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, config.SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio: np.ndarray, start: int) -> np.ndarray:
    """baseline script.py의 extract_segment와 동일."""
    if audio.size < config.SEGMENT_SAMPLES:
        repeat_count = config.SEGMENT_SAMPLES // audio.size + 1
        audio = np.tile(audio, repeat_count)
        return audio[: config.SEGMENT_SAMPLES].astype(np.float32)

    end = start + config.SEGMENT_SAMPLES
    return audio[start:end].astype(np.float32, copy=False)


def calculate_rms(audio: np.ndarray) -> float:
    """baseline script.py의 calculate_rms와 동일."""
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def is_silent(audio: np.ndarray) -> bool:
    return calculate_rms(audio) < config.SILENCE_RMS
