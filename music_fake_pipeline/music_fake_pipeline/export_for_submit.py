#!/usr/bin/env python3
"""
학습된 체크포인트를 submit.zip에 그대로 넣을 수 있는 형태로 내보낸다.

산출물 (export/music_classifier/):
    weights.pt            state_dict만 저장 (optimizer 상태 등 제외, 용량 최소화)
    music_fake_infer.py   추론 전용 자기완결(self-contained) 모듈
                           (submit.zip 안에서는 이 프로젝트의 다른 파일에 의존할 수
                            없으므로, model.py/features.py/audio_utils.py의 필요한
                            부분을 하나의 파일로 복제해 넣음)

이후 사용법 (SUBMIT_INTEGRATION.md 참고):
    1. export/music_classifier/ 를 통째로 submit.zip 압축 해제 폴더의
       model/music_classifier/ 로 복사
    2. patched_script.py 의 내용으로 submit.zip의 script.py를 교체
    3. requirements.txt는 수정 불필요 (torch/torchaudio/librosa 등 기존 항목으로 충분)
"""

import argparse
import shutil
from pathlib import Path

import torch

import config
from model import MusicFakeClassifier

INFER_MODULE_TEMPLATE = '''\
"""
음악 REAL/FAKE 분류기 추론 전용 모듈 (자기완결, 외부 프로젝트 파일에 의존하지 않음).

model_fake_music_pipeline/{{model.py, features.py, audio_utils.py}} 에서
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
AUDIO_SAMPLE_RATE = {audio_sample_rate}
SEGMENT_SAMPLES = {segment_samples}
SILENCE_RMS = {silence_rms}

N_FFT = {n_fft}
HOP_LENGTH = {hop_length}
N_MELS = {n_mels}
F_MIN = {f_min}
F_MAX = {f_max}
N_FRAMES = {n_frames}


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
'''


def build_infer_module_text() -> str:
    return INFER_MODULE_TEMPLATE.format(
        audio_sample_rate=config.AUDIO_SAMPLE_RATE,
        segment_samples=config.SEGMENT_SAMPLES,
        silence_rms=config.SILENCE_RMS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        f_min=config.F_MIN,
        f_max=config.F_MAX,
        n_frames=config.N_FRAMES,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=config.BEST_CHECKPOINT)
    parser.add_argument(
        "--out-dir", type=Path, default=config.EXPORT_DIR / "music_classifier"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {args.checkpoint}\n"
            f"먼저 train.py로 학습을 완료해야 합니다."
        )

    device = torch.device("cpu")
    model = MusicFakeClassifier().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = args.out_dir / "weights.pt"
    torch.save(model.state_dict(), weights_path)

    infer_module_path = args.out_dir / "music_fake_infer.py"
    infer_module_path.write_text(build_infer_module_text(), encoding="utf-8")

    size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"저장 완료: {args.out_dir}")
    print(f"  - {weights_path.name}  ({size_mb:.2f} MB)")
    print(f"  - {infer_module_path.name}")
    print(
        "\n다음 단계: 이 폴더 전체를 submit.zip 압축 해제 폴더의 "
        "model/music_classifier/ 로 복사한 뒤, "
        "patched_script.py 내용으로 script.py를 교체하세요. "
        "(SUBMIT_INTEGRATION.md 참고)"
    )


if __name__ == "__main__":
    main()
