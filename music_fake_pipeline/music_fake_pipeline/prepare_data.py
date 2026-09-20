#!/usr/bin/env python3
"""
원본 REAL/FAKE 음악 오디오를 HTDemucs로 분리하고, train/val manifest를 생성한다.

입력 (직접 다운로드해서 채워 넣어야 함):
    data/raw/real_music/*.{wav,mp3,flac,...}   REAL 음악 (예: FMA, MUSDB18 등)
    data/raw/fake_music/*.{wav,mp3,flac,...}   AI 생성 음악 (예: FakeMusicCaps, Echoes 등)

수행 과정:
    1. 각 파일을 16kHz mono로 로드 (baseline과 동일 규칙)
    2. HTDemucs로 vocals/accompaniment 분리 → accompaniment(=음악 성분)만 사용
       (--skip-demucs 옵션으로 이 단계를 건너뛸 수 있지만, 실제 추론 파이프라인과
        분포가 달라지므로 프로토타입 검증 용도로만 사용할 것을 권장)
    3. 분리된 파형을 .npy로 저장
    4. 원본 "파일(트랙)" 단위로 train/val을 분할 (세그먼트 단위 분할 금지 —
       같은 트랙의 다른 구간이 train/val에 동시에 들어가면 검증 점수가
       부풀려지는 data leakage가 발생함)
    5. data/manifests/train.csv, val.csv 생성 (columns: path,label,source)

사용 예:
    uv run prepare_data.py --htdemucs-dir /path/to/baseline_submit/model/htdemucs
    uv run prepare_data.py --skip-demucs --limit 50   # 빠른 파이프라인 점검용
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import audio_utils
import config


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--htdemucs-dir",
        type=Path,
        default=config.HTDEMUCS_MODEL_DIR,
        help="baseline_submit/model/htdemucs 경로",
    )
    parser.add_argument(
        "--skip-demucs",
        action="store_true",
        help="HTDemucs 분리를 건너뛰고 원본 오디오를 그대로 사용 (빠른 점검용, 비권장)",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="클래스별 처리 파일 수 제한 (디버그용)"
    )
    parser.add_argument("--val-ratio", type=float, default=config.VAL_RATIO)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def find_raw_files(raw_dir: Path, limit):
    files = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_AUDIO_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"{raw_dir} 안에서 오디오 파일을 찾지 못했습니다. "
            f"지원 확장자: {sorted(config.SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    if limit is not None:
        files = files[:limit]
    return files


def process_class(
    raw_dir: Path,
    out_dir: Path,
    label: int,
    label_name: str,
    htdemucs_model,
    device,
    skip_demucs: bool,
    limit,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_files = find_raw_files(raw_dir, limit)

    records = []
    skipped = []
    for raw_path in tqdm(raw_files, desc=f"{label_name} (label={label})"):
        out_path = out_dir / f"{raw_path.stem}.npy"
        try:
            if skip_demucs:
                music_audio = audio_utils.load_audio(raw_path)
            else:
                import demucs_separate  # 지연 임포트: --skip-demucs 시 demucs 불필요

                _voice_audio, music_audio = demucs_separate.separate_voice_and_music(
                    raw_path, htdemucs_model, device
                )

            if audio_utils.is_silent(music_audio):
                skipped.append((raw_path.name, "silent_after_separation"))
                continue

            np.save(out_path, music_audio.astype(np.float32))
            records.append(
                {"path": str(out_path), "label": label, "source": raw_path.name}
            )
        except Exception as exc:  # noqa: BLE001 - 개별 파일 실패는 건너뛰고 계속 진행
            skipped.append((raw_path.name, str(exc)))

    if skipped:
        print(f"[{label_name}] {len(skipped)}개 파일 건너뜀 (예: {skipped[:3]})")

    return records


def split_train_val(records: list[dict], val_ratio: float, seed: int):
    """파일(트랙) 단위로 shuffle 후 분할. 클래스별로 비율을 맞춰 층화 분할."""
    rng = random.Random(seed)

    by_label: dict[int, list[dict]] = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)

    train_records, val_records = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio)) if len(group) > 1 else 0
        val_records.extend(group[:n_val])
        train_records.extend(group[n_val:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def write_manifest(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write("path,label,source\n")
        for record in records:
            file.write(f"{record['path']},{record['label']},{record['source']}\n")
    print(f"저장: {path} ({len(records)}개)")


def main():
    args = parse_arguments()

    device_name = args.device
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    htdemucs_model = None
    if not args.skip_demucs:
        import demucs_separate

        if not args.htdemucs_dir.is_dir():
            raise FileNotFoundError(
                f"HTDemucs 모델 경로를 찾을 수 없습니다: {args.htdemucs_dir}\n"
                f"--htdemucs-dir 로 baseline_submit/model/htdemucs 경로를 지정하거나, "
                f"프로토타입 검증만 원한다면 --skip-demucs 를 사용하세요."
            )
        htdemucs_model = demucs_separate.load_htdemucs_model(args.htdemucs_dir)
        print(f"HTDemucs 로드 완료 (device={device})")

    real_records = process_class(
        config.RAW_REAL_DIR,
        config.SEPARATED_REAL_DIR,
        label=0,
        label_name="REAL",
        htdemucs_model=htdemucs_model,
        device=device,
        skip_demucs=args.skip_demucs,
        limit=args.limit,
    )
    fake_records = process_class(
        config.RAW_FAKE_DIR,
        config.SEPARATED_FAKE_DIR,
        label=1,
        label_name="FAKE",
        htdemucs_model=htdemucs_model,
        device=device,
        skip_demucs=args.skip_demucs,
        limit=args.limit,
    )

    all_records = real_records + fake_records
    if not all_records:
        raise RuntimeError("처리된 파일이 없습니다. data/raw/ 아래 데이터를 먼저 채워주세요.")

    print(f"\n총 REAL={len(real_records)}, FAKE={len(fake_records)}")

    train_records, val_records = split_train_val(all_records, args.val_ratio, args.seed)
    write_manifest(config.TRAIN_MANIFEST, train_records)
    write_manifest(config.VAL_MANIFEST, val_records)

    def label_balance(records):
        n_real = sum(1 for r in records if r["label"] == 0)
        n_fake = sum(1 for r in records if r["label"] == 1)
        return n_real, n_fake

    tr_real, tr_fake = label_balance(train_records)
    va_real, va_fake = label_balance(val_records)
    print(f"train: REAL={tr_real}, FAKE={tr_fake}")
    print(f"val  : REAL={va_real}, FAKE={va_fake}")


if __name__ == "__main__":
    main()
