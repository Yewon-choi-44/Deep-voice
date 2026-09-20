#!/usr/bin/env python3
"""
대회 평가 산식과 동일한 방식으로 Music EER을 계산한다.

평가 산식 원문(제공받은 문서 그대로):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    EER = (fpr[idx] + fnr[idx]) / 2
    (FAKE = 1을 양성 클래스로 정의)

파일 단위 점수는 baseline의 predict_fake()와 동일하게, 한 파일에 속한
모든 세그먼트 FAKE 확률의 max를 사용한다 (국소 구간 조작 탐지 목적).

사용 예:
    uv run evaluate.py --checkpoint checkpoints/music_classifier_best.pt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve
from tqdm import tqdm

import config
from dataset import MusicManifestDataset
from model import MusicFakeClassifier


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """대회 평가 산식과 완전히 동일한 EER 계산."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


@torch.inference_mode()
def evaluate_manifest(model, manifest_path: Path, device: torch.device, save_csv: Path = None):
    """
    manifest(eval 모드)를 순회하며 파일 단위 max-aggregated 확률을 계산하고
    EER을 산출한다.

    Returns
    -------
    eer : float
    records : list[dict]  (source, label, score) — 디버깅/분석용
    """
    dataset = MusicManifestDataset(manifest_path, mode="eval")
    model.eval()

    y_true, y_score, records = [], [], []
    for segments, label, source in tqdm(dataset, desc="Evaluating", total=len(dataset)):
        segments = segments.to(device)  # (num_segments, 1, N_MELS, N_FRAMES)
        probs = model.predict_proba(segments)  # (num_segments,)
        file_score = float(probs.max().item())

        y_true.append(float(label))
        y_score.append(file_score)
        records.append({"source": source, "label": int(label), "score": file_score})

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    eer = compute_eer(y_true, y_score)

    if save_csv is not None:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with save_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["source", "label", "score"])
            writer.writeheader()
            writer.writerows(records)

    return eer, records


def load_model(checkpoint_path: Path, device: torch.device) -> MusicFakeClassifier:
    model = MusicFakeClassifier().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=config.BEST_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=config.VAL_MANIFEST)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    model = load_model(args.checkpoint, device)
    eer, records = evaluate_manifest(model, args.manifest, device, save_csv=args.save_csv)

    n_real = sum(1 for r in records if r["label"] == 0)
    n_fake = sum(1 for r in records if r["label"] == 1)
    print(f"\n검증 파일 수: {len(records)} (REAL={n_real}, FAKE={n_fake})")
    print(f"Music EER: {eer:.4f}  ({eer * 100:.2f}%)")
    print(f"참고 — 대회 배점 기여: 0.9 * 0.3 * (1 - EER) = {0.9 * 0.3 * (1 - eer):.4f} / 0.27")


if __name__ == "__main__":
    main()
