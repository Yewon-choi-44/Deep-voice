"""
음악 REAL/FAKE 분류기 추론 전용 모듈 (자기완결, 외부 프로젝트 파일에 의존하지 않음).

model_fake_music_pipeline/{model.py, features.py, audio_utils.py} 에서
필요한 부분만 복제해 하나의 파일로 만들었습니다. submit.zip 안에서
model/music_classifier/ 폴더에 weights.pt 와 함께 배치되어,
script.py 에서 import 되어 쓰입니다.

원본 개발 프로젝트: music_fake_pipeline/ (baseline_submit/script.py의
music_fake 계산 부분만 교체하기 위해 별도로 학습한 소형 CNN)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio

# --- 오디오/피처 상수 (music_fake_pipeline/config.py 와 반드시 동일해야 함) ---
AUDIO_SAMPLE_RATE = 16000
SEGMENT_SAMPLES = 64600
SILENCE_RMS = 1e-05

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
F_MIN = 20
F_MAX = 8000
N_FRAMES = 253


# -----------------------------------------------------------------------------
# 오디오 세그먼트 분할 (baseline script.py의 get_segment_starts/extract_segment와 동일)
# -----------------------------------------------------------------------------

def get_segment_starts(audio_length: int):
    if audio_length <= SEGMENT_SAMPLES:
        return [0]
    last_start = audio_length - SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio: np.ndarray, start: int) -> np.ndarray:
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // audio.size + 1
        audio = np.tile(audio, repeat_count)
        return audio[:SEGMENT_SAMPLES].astype(np.float32)
    end = start + SEGMENT_SAMPLES
    return audio[start:end].astype(np.float32, copy=False)


def calculate_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


# -----------------------------------------------------------------------------
# log-mel spectrogram
# -----------------------------------------------------------------------------

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=AUDIO_SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    power=2.0,
)
_db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)


def waveform_to_logmel(waveform: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(waveform)).float()
    mel = _mel_transform(tensor)
    log_mel = _db_transform(mel)

    if log_mel.shape[-1] < N_FRAMES:
        pad = N_FRAMES - log_mel.shape[-1]
        log_mel = torch.nn.functional.pad(log_mel, (0, pad))
    elif log_mel.shape[-1] > N_FRAMES:
        log_mel = log_mel[:, :N_FRAMES]

    mean = log_mel.mean()
    std = log_mel.std().clamp_min(1e-6)
    log_mel = (log_mel - mean) / std
    return log_mel.unsqueeze(0)  # (1, N_MELS, N_FRAMES)


# -----------------------------------------------------------------------------
# 모델 구조 (model.py의 MusicFakeClassifier와 동일)
# -----------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class MusicFakeClassifier(nn.Module):
    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.blocks = nn.Sequential(
            _ConvBlock(1, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 128),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        features_map = self.blocks(x)
        pooled = self.global_pool(features_map)
        return self.classifier(pooled).squeeze(-1)

    @torch.inference_mode()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


# -----------------------------------------------------------------------------
# 로딩 및 추론 (baseline script.py의 predict_fake()와 동일한 인터페이스/동작)
# -----------------------------------------------------------------------------

def load_music_fake_model(weights_path: Path, device: torch.device) -> MusicFakeClassifier:
    model = MusicFakeClassifier().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_music_fake(model: MusicFakeClassifier, audio: np.ndarray, device: torch.device) -> float:
    """baseline script.py의 predict_fake()와 동일한 시그니처/집계 방식(세그먼트 max)."""
    if calculate_rms(audio) < SILENCE_RMS:
        return 0.0

    segment_scores = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        mel = waveform_to_logmel(segment).unsqueeze(0).to(device)  # (1,1,N_MELS,N_FRAMES)
        with torch.inference_mode():
            probability = model.predict_proba(mel)
        segment_scores.append(float(probability.item()))

    return max(segment_scores)
