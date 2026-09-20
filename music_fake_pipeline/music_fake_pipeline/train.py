#!/usr/bin/env python3
"""
음악 REAL/FAKE 분류기 학습.

매 epoch:
  1. train manifest에서 무작위 크롭 세그먼트로 학습 (BCEWithLogitsLoss)
  2. val manifest에서 파일 단위 max-aggregated EER을 계산 (evaluate.py 재사용,
     대회 평가 산식과 동일한 방식)
  3. val EER이 개선되면 checkpoints/music_classifier_best.pt 갱신
  4. EARLY_STOP_PATIENCE epoch 동안 개선이 없으면 조기 종료

사용 예:
    uv run train.py
    uv run train.py --epochs 15 --batch-size 16 --device cpu
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from dataset import MusicManifestDataset
from evaluate import evaluate_manifest
from model import MusicFakeClassifier, count_parameters


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--train-manifest", type=Path, default=config.TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=config.VAL_MANIFEST)
    parser.add_argument(
        "--resume", type=Path, default=None, help="이어서 학습할 체크포인트 경로"
    )
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss, n_samples = 0.0, 0

    for mels, labels in loader:
        mels = mels.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(mels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / max(1, n_samples)


def save_checkpoint(path: Path, model, optimizer, epoch: int, val_eer: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_eer": val_eer,
        },
        path,
    )


def main():
    args = parse_arguments()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"device: {device}")

    train_dataset = MusicManifestDataset(args.train_manifest, mode="train")
    print(f"train samples(파일): {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    model = MusicFakeClassifier().to(device)
    print(f"model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss()

    start_epoch = 1
    best_eer = float("inf")
    epochs_without_improvement = 0

    if args.resume is not None and args.resume.is_file():
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state.get("epoch", 0) + 1
        best_eer = state.get("val_eer", float("inf"))
        print(f"체크포인트에서 재개: epoch={start_epoch}, best_eer={best_eer:.4f}")

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_DIR / "train_log.csv"
    log_is_new = not log_path.exists()
    log_file = log_path.open("a", encoding="utf-8", newline="")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(["epoch", "train_loss", "val_eer", "elapsed_sec"])

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_eer, _ = evaluate_manifest(model, args.val_manifest, device)
        scheduler.step()
        elapsed = time.time() - epoch_start

        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  val_eer={val_eer:.4f}  "
            f"({elapsed:.1f}s)"
        )
        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_eer:.6f}", f"{elapsed:.1f}"])
        log_file.flush()

        save_checkpoint(config.LAST_CHECKPOINT, model, optimizer, epoch, val_eer)

        if val_eer < best_eer:
            best_eer = val_eer
            epochs_without_improvement = 0
            save_checkpoint(config.BEST_CHECKPOINT, model, optimizer, epoch, val_eer)
            print(f"  -> best 갱신 (val_eer={best_eer:.4f}), {config.BEST_CHECKPOINT} 저장")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"{args.patience} epoch 동안 개선 없음 — 조기 종료 "
                    f"(best_val_eer={best_eer:.4f})"
                )
                break

    log_file.close()
    print(f"\n학습 종료. best val EER = {best_eer:.4f}")
    print(f"체크포인트: {config.BEST_CHECKPOINT}")


if __name__ == "__main__":
    main()
