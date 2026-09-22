"""
학습 1회 실행(run)마다 별도의 CSV 로그 파일을 만든다.

기존에는 logs/train_log.csv 하나에 모든 실행 기록이 이어 붙는 방식이라,
실행이 여러 번 반복되면 어디부터 어디까지가 한 번의 실행인지 epoch 번호가
1로 리셋되는지만 보고 추측해야 했다. 이 모듈은 실행 시작 시각을 파일명으로
써서(logs/20260922_172239.csv) 실행 단위를 명확히 구분한다.

사용 예 (train.py 안에서):
    from run_logger import RunLogger
    logger = RunLogger()
    ...
    logger.log_epoch(epoch, train_loss, val_eer, elapsed_sec)
    ...
    logger.close()
"""

import csv
from datetime import datetime
from pathlib import Path

import config


class RunLogger:
    def __init__(self, log_dir: Path = None, start_time: datetime = None):
        log_dir = log_dir if log_dir is not None else config.LOG_DIR
        start_time = start_time or datetime.now()

        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{start_time:%Y%m%d_%H%M%S}.csv"

        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["epoch", "train_loss", "val_eer", "elapsed_sec"])

    def log_epoch(self, epoch: int, train_loss: float, val_eer: float, elapsed_sec: float):
        self._writer.writerow(
            [epoch, f"{train_loss:.6f}", f"{val_eer:.6f}", f"{elapsed_sec:.1f}"]
        )
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
