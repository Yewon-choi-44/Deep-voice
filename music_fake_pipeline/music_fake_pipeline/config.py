"""
프로젝트 전역 설정.

baseline_submit/script.py 와 동일한 상수(AUDIO_SAMPLE_RATE, SEGMENT_SAMPLES)를
그대로 재사용합니다. 학습 데이터와 실제 추론 시점의 입력 분포를 최대한
일치시키기 위해서입니다 (train-test distribution shift 방지).
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# 경로
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_REAL_DIR = PROJECT_ROOT / "data" / "raw" / "real_music"
RAW_FAKE_DIR = PROJECT_ROOT / "data" / "raw" / "fake_music"

SEPARATED_REAL_DIR = PROJECT_ROOT / "data" / "separated" / "real"
SEPARATED_FAKE_DIR = PROJECT_ROOT / "data" / "separated" / "fake"

MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "train.csv"
VAL_MANIFEST = MANIFEST_DIR / "val.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "music_classifier_best.pt"
LAST_CHECKPOINT = CHECKPOINT_DIR / "music_classifier_last.pt"

EXPORT_DIR = PROJECT_ROOT / "export"  # submit.zip의 model/music_classifier/ 에 그대로 복사할 산출물
LOG_DIR = PROJECT_ROOT / "logs"

# baseline_submit.zip 압축 해제 위치를 사용자가 알려주면 여기 채워서 씀.
# prepare_data.py --htdemucs-dir 인자로도 덮어쓸 수 있음.
BASELINE_SUBMIT_DIR = PROJECT_ROOT.parent / "baseline_submit"
HTDEMUCS_MODEL_DIR = BASELINE_SUBMIT_DIR / "model" / "htdemucs"

# -----------------------------------------------------------------------------
# 오디오 처리 (baseline script.py 값과 동일하게 고정)
# -----------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 64_600          # 16kHz 기준 약 4.04초. DF-Arena 1B와 동일 윈도우.
SILENCE_RMS = 1e-5

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}

# -----------------------------------------------------------------------------
# mel-spectrogram (신규 음악 분류기 입력)
# -----------------------------------------------------------------------------
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
F_MIN = 20
F_MAX = 8000  # AUDIO_SAMPLE_RATE / 2

# SEGMENT_SAMPLES, HOP_LENGTH 기준 프레임 수 (모델 입력 고정 크기 계산용)
# frames = floor(SEGMENT_SAMPLES / HOP_LENGTH) + 1
N_FRAMES = SEGMENT_SAMPLES // HOP_LENGTH + 1

# -----------------------------------------------------------------------------
# 학습
# -----------------------------------------------------------------------------
SEED = 42
VAL_RATIO = 0.15          # 원본 파일(트랙) 단위로 분할 — 세그먼트 단위 분할 금지 (leakage 방지)
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 6     # val EER 기준

DEVICE = "cuda"  # train.py에서 미사용 가능 시 자동으로 cpu로 폴백
