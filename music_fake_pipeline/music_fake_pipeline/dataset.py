"""
manifest(csv) -> PyTorch Dataset.

두 가지 모드를 분리한 이유:
  - "train" 모드: 매 epoch마다 파일에서 무작위 위치의 세그먼트 하나를 뽑음
    (일종의 random-crop augmentation. 같은 파일이라도 epoch마다 다른 구간을 봄)
  - "eval" 모드: 파일 전체의 모든 세그먼트를 다 반환함. 검증 시 baseline과
    동일하게 "세그먼트별 점수의 max"를 파일 단위 점수로 집계해야 하기 때문
    (baseline script.py의 predict_fake() 로직과 동일한 집계 방식).
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import audio_utils
import features


def _read_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest이 비어 있습니다: {manifest_path}")
    return rows


class MusicManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, mode: str = "train"):
        if mode not in ("train", "eval"):
            raise ValueError("mode는 'train' 또는 'eval' 이어야 합니다")
        self.mode = mode
        self.rows = _read_manifest(Path(manifest_path))

    def __len__(self):
        return len(self.rows)

    def _load_waveform(self, row: dict) -> np.ndarray:
        return np.load(row["path"]).astype(np.float32)

    def __getitem__(self, index: int):
        row = self.rows[index]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        audio = self._load_waveform(row)

        if self.mode == "train":
            starts = audio_utils.get_segment_starts(audio.size)
            start = random.choice(starts)
            segment = audio_utils.extract_segment(audio, start)
            mel = features.waveform_to_logmel(segment)  # (1, N_MELS, N_FRAMES)
            return mel, label

        # eval 모드: 모든 세그먼트를 스택해서 반환 (배치 크기는 항상 1로 사용할 것)
        # 각 세그먼트는 (1, N_MELS, N_FRAMES) — 채널 차원을 squeeze하지 않고 그대로
        # 쌓아야 model(Conv2d)이 기대하는 (num_segments, 1, N_MELS, N_FRAMES) 형태가 됨.
        # squeeze(0) 후 stack하면 채널 차원이 사라져 num_segments가 채널로 오인식되는
        # 버그가 생기므로 주의.
        starts = audio_utils.get_segment_starts(audio.size)
        segments = [audio_utils.extract_segment(audio, s) for s in starts]
        mels = torch.stack(
            [features.waveform_to_logmel(seg) for seg in segments]
        )  # (num_segments, 1, N_MELS, N_FRAMES)
        return mels, label, row["source"]
