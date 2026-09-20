"""
파형(waveform) -> log-mel spectrogram 변환.

신규 음악 FAKE 분류기의 입력 표현. baseline의 DF-Arena(wav2vec2, 파형 직접
입력)와 달리, 이 분류기는 고전적 신호처리 방식(STFT 기반 mel-spectrogram)을
씁니다 — 음악은 화성/리듬 등 wav2vec2가 음성 특화로 학습한 표현과는 다른
음향 구조를 가지므로, 이미지 분류에 가까운 접근이 더 안전한 선택입니다.
"""

import numpy as np
import torch
import torchaudio

import config

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.AUDIO_SAMPLE_RATE,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS,
    f_min=config.F_MIN,
    f_max=config.F_MAX,
    power=2.0,
)
_db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)


def waveform_to_logmel(waveform: np.ndarray) -> torch.Tensor:
    """
    Parameters
    ----------
    waveform : np.ndarray, shape (SEGMENT_SAMPLES,), 16kHz mono float32

    Returns
    -------
    torch.Tensor, shape (1, N_MELS, N_FRAMES) — CNN 입력용 1채널 "이미지"
    """
    tensor = torch.from_numpy(np.ascontiguousarray(waveform)).float()
    mel = _mel_transform(tensor)          # (N_MELS, time)
    log_mel = _db_transform(mel)          # 데시벨 스케일 변환

    # 세그먼트 길이가 SEGMENT_SAMPLES로 고정되어 있으므로 프레임 수도 고정이어야
    # 하지만, 부동소수점 경계에서 ±1 프레임 오차가 날 수 있어 방어적으로 맞춤.
    n_frames = config.N_FRAMES
    if log_mel.shape[-1] < n_frames:
        pad = n_frames - log_mel.shape[-1]
        log_mel = torch.nn.functional.pad(log_mel, (0, pad))
    elif log_mel.shape[-1] > n_frames:
        log_mel = log_mel[:, :n_frames]

    # 정규화: 배치/데이터셋 전체 통계 대신 샘플별 표준화 사용 (간단하고 안정적)
    mean = log_mel.mean()
    std = log_mel.std().clamp_min(1e-6)
    log_mel = (log_mel - mean) / std

    return log_mel.unsqueeze(0)  # (1, N_MELS, N_FRAMES)
