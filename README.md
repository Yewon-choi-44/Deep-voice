# 딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회

> 주최: 행정안전부 · 한국지능정보사회진흥원 / 주관: 국립과학수사연구원 / 운영: DACON  
> **리더보드 제출 마감: 2026-09-29 (화) 10:00 KST**

---

## 목차

1. [대회 개요](#1-대회-개요)
2. [태스크 정의](#2-태스크-정의)
3. [평가 산식](#3-평가-산식)
4. [프로젝트 전략](#4-프로젝트-전략)
5. [파이프라인 아키텍처](#5-파이프라인-아키텍처)
6. [디렉토리 구조](#6-디렉토리-구조)
7. [환경 설정](#7-환경-설정)
8. [개발 워크플로우](#8-개발-워크플로우)
9. [실행 방법](#9-실행-방법)
10. [제출 규격 및 주의사항](#10-제출-규격-및-주의사항)
11. [데이터](#11-데이터)
12. [검증 현황](#12-검증-현황)
13. [참고 자료](#13-참고-자료)

---

## 1. 대회 개요

생성형 AI·음성합성 기술의 발전으로 실제 음성과 구분하기 어려운 딥보이스가 보이스피싱·음성 사칭·허위정보 생성 등에 악용되고 있다. 이 대회는 음성뿐 아니라 음악 등 다양한 오디오 유형에서 AI 생성 여부를 탐지하는 모델을 개발하는 것이 목적이다.

### 대회 방식

| 단계 | 내용 |
|------|------|
| 1차 평가 | Private 리더보드 상위 **15팀** 선정 |
| 2차 평가 | 모델 개발 보고서 + 학습데이터 구성 보고서(HWP) 제출 → 최종 상위 **7팀** 수상 |

### 일정

| 항목 | 일시 |
|------|------|
| 팀 병합 마감 | 2026-09-23 (수) 23:59 |
| **리더보드 제출 마감** | **2026-09-29 (화) 10:00** |
| 대회 종료 | 2026-09-30 (수) 10:00 |
| 2차 평가 자료 제출 | 2026-09-30 ~ 2026-10-05 |
| 최종 결과 발표 | 2026-10-16 (금) |

---

## 2. 태스크 정의

오디오 파일 하나를 입력받아 **5개의 확률값(0~1)**을 동시에 예측하는 멀티태스크 문제다.

| 컬럼명 | 의미 |
|--------|------|
| `FILE_FAKE_PROB` | 파일 전체가 AI 생성(Fake)일 확률 |
| `VOICE_FAKE_PROB` | 음성 성분이 AI 생성일 확률 |
| `MUSIC_FAKE_PROB` | 음악 성분이 AI 생성일 확률 |
| `VOICE_PRESENT_PROB` | 파일에 음성이 존재할 확률 |
| `MUSIC_PRESENT_PROB` | 파일에 음악이 존재할 확률 |

### 오디오 유형 및 Real/Fake 기준

- **음성**: 발화 또는 보컬만 포함
- **음악**: 보컬 없는 반주·악기음만 포함
- **혼합**: 음성+음악 동시/순차 포함
- 음성·음악 중 **하나라도 Fake이면 파일 전체 Fake**
- 품질 개선·잡음 제거·음량 조정 등 성분 자체를 새로 생성하지 않는 후처리는 Real 유지

### 평가 데이터 특성

- 총 **1,200개** 오디오 파일 (비공개)
- 길이: 4초 이상 1분 이하
- 샘플링레이트: 16kHz (채널은 모노/스테레오 혼재)
- 지원 포맷: `.aac` `.flac` `.m4a` `.mp3` `.ogg` `.opus` `.wav` `.wma`
- 일부 샘플에 전화채널 오디오 포함

---

## 3. 평가 산식

```
Score (↑) = 0.9 × ADS + 0.1 × CPS
```

### ADS — AI-Generated Audio Detection Score (전체의 90%)

```
ADS = 0.5 × (1 - File EER) + 0.2 × (1 - Voice EER) + 0.3 × (1 - Music EER)
```

- EER(Equal Error Rate): FPR = FNR이 되는 지점의 오류율. 낮을수록 좋음 → `(1 - EER)`로 변환하여 높을수록 유리
- Voice EER: **음성이 존재하는 샘플에서만** 계산
- Music EER: **음악이 존재하는 샘플에서만** 계산
- FAKE = 양성 클래스 (1)

### CPS — Component Presence Score (전체의 10%)

```
CPS = 0.5 × Voice Presence ROC-AUC + 0.5 × Music Presence ROC-AUC
```

### 배점 가중치 요약

| 항목 | 전체 기여도 |
|------|------------|
| File EER | 45% (0.9 × 0.5) |
| Music EER | 27% (0.9 × 0.3) ← **핵심 개선 타겟** |
| Voice EER | 18% (0.9 × 0.2) |
| CPS | 10% (0.1 × 1.0) |

---

## 4. 프로젝트 전략

### 베이스라인의 취약점

베이스라인은 음성 스푸핑(deepfake voice) 탐지용으로 설계된 **DF-Arena 1B**를 음악 Fake 판별에도 그대로 적용하고 있다. 이 모델은 음성 24개 벤치마크에서 평균 EER 3.66%를 기록했지만, **음악 도메인에서는 검증된 적이 없다.**

### 개선 방향

**MUSIC_FAKE_PROB 계산 경로만 교체한다.** 나머지는 모두 baseline 그대로 유지한다.

| 컴포넌트 | 베이스라인 | 개선안 |
|---------|-----------|--------|
| 음원 분리 | HTDemucs | 유지 |
| 음성/음악 존재 판별 | PANNs Cnn14 | 유지 |
| 음성 Fake 판별 | DF-Arena 1B | 유지 |
| **음악 Fake 판별** | **DF-Arena 1B (음성 전용 모델 전용)** | **`MusicFakeClassifier` (소형 CNN)으로 교체** |
| FILE_FAKE_PROB 결합 | MAX Fusion | 유지 |

### `MusicFakeClassifier` 개요

- 입력: log-mel spectrogram `(1, 128, 253)` — 4.04초 세그먼트, 128 mel bins
- 구조: ConvBlock(Conv2d-BN-ReLU-MaxPool2d) × 4 → GlobalAvgPool → Dropout → FC
- 파라미터 수: **약 25만** (DF-Arena 1B 대비 1/4,000 수준)
- 손실 함수: `BCEWithLogitsLoss`
- 설계 의도: 60분/1,200파일 추론 시간 제약 내에서 DF-Arena보다 오히려 연산량 감소

### 전략적 근거

- ADS 내 Music 가중치(0.3) > Voice 가중치(0.2)
- 베이스라인에서 Music 경로만 유일하게 도메인 검증이 이루어지지 않은 지점
- Voice/Presence/HTDemucs는 원래 용도와 일치하여 교체 효과 기대 낮음

> ⚠️ **09-21 리더보드 결과로 이 가설이 아직 검증되지 않았음이 드러남.** DF-Arena를 그대로 쓴 09-20 제출(총점 0.6909)보다, `music_classifier`로 교체한 09-21 제출(총점 0.6722)의 점수가 더 낮았다 — 즉 현재 버전의 `music_classifier`는 DF-Arena zero-shot보다 실제 대회 데이터에서 더 나쁘게 동작한다. Real 150개(MusicCaps) · Fake 5개 생성기(FakeMusicCaps)라는 좁은 데이터로 학습한 탓에 로컬 검증 세트에는 잘 맞아도(§12 참고) 실제 평가 데이터로는 일반화되지 못했을 가능성이 높다. 상세는 `dev_log.md` 2026-09-22 항목 참고.

---

## 5. 파이프라인 아키텍처

### 세그먼트 처리 방식 (baseline과 공통)

- 세그먼트 크기: `SEGMENT_SAMPLES = 64,600` (16kHz 기준 약 **4.04초**)
- 파일이 세그먼트보다 짧으면: tile 반복으로 패딩
- 파일이 세그먼트보다 길면: 64,600 샘플 단위 슬라이딩, 각 세그먼트 독립 예측 후 **MAX** 집계
- 침묵 처리: RMS < 1e-5이면 Fake 확률 **0.0** 반환 (HTDemucs로 분리 후 침묵 성분은 Fake 판정하지 않음)

### 베이스라인 (Zero-shot)

```
INPUT AUDIO
│
├── PANNs Cnn14 (32kHz, AudioSet 527클래스)
│   ├── voice 그룹 (18개 레이블) → max → VOICE_PRESENT_PROB
│   └── music 그룹 (112개 레이블) → max → MUSIC_PRESENT_PROB
│
└── HTDemucs (z-score 정규화 후 분리)
     ├── vocals  → 16kHz 리샘플 → DF-Arena 1B → 세그먼트 MAX → VOICE_FAKE_PROB
     └── accompaniment → 16kHz 리샘플 → DF-Arena 1B → 세그먼트 MAX → MUSIC_FAKE_PROB (⚠️ 취약)

FILE_FAKE_PROB = max(VOICE_PRESENT × VOICE_FAKE, MUSIC_PRESENT × MUSIC_FAKE)
```

### 개선안

```
INPUT AUDIO
│
├── PANNs Cnn14 ─────────────────── VOICE_PRESENT_PROB / MUSIC_PRESENT_PROB  (유지)
│
└── HTDemucs
     ├── vocals  → DF-Arena 1B ──── VOICE_FAKE_PROB                          (유지)
     └── accompaniment
           → log-mel spectrogram
           → MusicFakeClassifier ── MUSIC_FAKE_PROB                           (✅ 교체)

FILE_FAKE_PROB = max(VOICE_PRESENT × VOICE_FAKE, MUSIC_PRESENT × MUSIC_FAKE) (유지)
```

---

## 6. 디렉토리 구조

```
Deep_voice/
├── README.md                         # 이 문서
├── deepvoice_competition_summary.md  # 대회 규정·평가·일정 정리 노트
├── main.py                           # placeholder (미사용)
├── .python-version                   # Python 3.11
├── .venv/                            # 가상환경
│
├── baseline_submit.zip               # 베이스라인 원본 (zip 내부에 script.py 포함)
├── baseline_submit/                  # zip 압축 해제본 (모델 가중치만 추출된 상태)
│   ├── requirements.txt
│   └── model/
│       ├── MODEL_INFO.txt
│       ├── SHA256SUMS.txt
│       ├── df_arena_1b/              # DF-Arena 1B 가중치 및 소스
│       ├── htdemucs/                 # HTDemucs 가중치
│       └── panns/                    # PANNs Cnn14 가중치 + 레이블 설정
│
├── music_fake_pipeline/
│   └── music_fake_pipeline/          # MUSIC_FAKE_PROB 교체 파이프라인 (상세: 내부 README 참고)
│       ├── config.py
│       ├── audio_utils.py
│       ├── demucs_separate.py
│       ├── features.py
│       ├── model.py                  # MusicFakeClassifier
│       ├── dataset.py
│       ├── prepare_data.py
│       ├── train.py
│       ├── evaluate.py
│       ├── export_for_submit.py
│       ├── patched_script.py         # 최종 제출용 script.py
│       ├── requirements-dev.txt
│       └── data/
│           ├── raw/real_music/       # ← REAL 음악 원본 (직접 채울 것)
│           ├── raw/fake_music/       # ← Fake 음악 원본 (직접 채울 것)
│           ├── separated/{real,fake}/
│           └── manifests/
│
└── data/                             # 형식 확인용 더미 데이터
    ├── sample_submission.csv
    └── test/
        ├── TEST_0000.wav
        ├── TEST_0001.wav
        └── TEST_0002.wav
```

> `data/test/`는 형식 확인용 더미(3개)다. 실제 평가 시 서버가 1,200개 파일로 자동 교체한다.  
> `script.py`는 `baseline_submit.zip` 내부에만 있다. 실행 전 별도로 추출 필요.

---

## 7. 환경 설정

### 로컬 학습 환경 (uv 권장)

```bash
uv venv --python 3.11
uv pip install -r music_fake_pipeline/music_fake_pipeline/requirements-dev.txt

# CUDA 12.8 GPU 사용 시 torch/torchaudio는 cu128 wheel로 별도 설치
uv pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

### 평가 서버 환경 (제출 기준)

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA L4 (VRAM 22.4GiB) |
| CPU | 6 vCPU / RAM 28GB |
| Python | 3.11.15 |
| CUDA | 12.8 |

**서버에 사전 설치된 주요 패키지** (requirements.txt 중복 기재 불필요):

```
torch==2.7.1+cu128      torchaudio==2.7.1+cu128
transformers==4.57.6    demucs==4.0.1
panns-inference==0.1.1  librosa==0.10.2.post1
pandas==2.0.3           numpy==1.26.4
scikit-learn==1.8.0     scipy==1.15.3
```

### ⚠️ 반드시 알아야 할 버전 함정 (직접 확인한 사실)

`demucs 4.0.1`과 `4.1.0` 사이에 `demucs.separate.load_track` 함수가 제거되는 **호환성 단절**이 있다. baseline `script.py`와 이 프로젝트의 `demucs_separate.py` 모두 이 함수를 직접 import한다. `pip install demucs`만 실행하면 최신 버전이 설치되어 `ImportError`가 발생한다. **반드시 `demucs==4.0.1`로 고정해야 한다.**

### ⚠️ 위 환경 설정은 평가 서버 기준 — 실제 로컬 개발 환경은 다름

로컬 학습은 Windows 11 + Python 3.13.2(시스템) + NVIDIA Quadro RTX 5000(16GB, 드라이버 CUDA 13.2) + `torch==2.11.0+cu128` 환경에서 진행 중이다. `pip install torch`를 옵션 없이 실행하면 CPU 전용 빌드가 잡히는 문제 등 실제로 겪은 삽질과 해결 과정은 `music_fake_pipeline/music_fake_pipeline/README.md` 9장에 날것으로 기록되어 있다. **제출 전에는 반드시 위 표의 평가 서버 스펙(Python 3.11.15 / CUDA 12.8 / torch 2.7.1+cu128)에서 한 번 더 검증**해야 한다 — 로컬 환경과 버전이 달라 체크포인트는 로드되어도 추론 결과가 미묘하게 달라질 위험이 있다.

---

## 8. 개발 워크플로우

`music_fake_pipeline/music_fake_pipeline/` 디렉토리 안에서 실행한다.

### Step 1 — 데이터 준비

`data/raw/real_music/`과 `data/raw/fake_music/`에 오디오 파일을 채워 넣은 뒤:

```bash
# HTDemucs로 accompaniment 분리 + train/val manifest 생성
python prepare_data.py --htdemucs-dir ../../baseline_submit/model/htdemucs

# 파이프라인 빠른 점검만 할 경우 (HTDemucs 분리 생략, 비권장)
python prepare_data.py --skip-demucs --limit 20
```

- train/val 분할은 **원본 파일(트랙) 단위** — 세그먼트 단위 분할 시 data leakage 발생
- `data/manifests/train.csv`, `val.csv` 생성됨

### Step 2 — 학습

```bash
python train.py
# 또는 하이퍼파라미터 직접 지정
python train.py --epochs 30 --batch-size 32 --device cuda
```

- val EER 기준으로 best checkpoint(`checkpoints/music_classifier_best.pt`) 갱신
- `EARLY_STOP_PATIENCE = 6` epoch 동안 개선 없으면 자동 종료

### Step 3 — 평가

```bash
python evaluate.py \
    --checkpoint checkpoints/music_classifier_best.pt \
    --save-csv logs/val_predictions.csv
```

- 대회 평가 산식(`fpr/fnr` 교차점 EER)과 동일한 계산식 사용
- 출력에 `0.9 × 0.3 × (1 - EER)` 값 포함 → 총점(Score) 중 이 분류기의 기여분

### Step 4 — 제출용 패키지 생성

> ⚠️ **주의**: `music_fake_pipeline/music_fake_pipeline/.venv`에는 `torch`가 설치되어 있지 않다. 반드시 **프로젝트 루트의 `.venv`**(torch+CUDA 포함, 학습 때 쓴 환경)를 사용해야 한다.

```bash
cd music_fake_pipeline/music_fake_pipeline

# 프로젝트 루트 .venv의 python을 직접 지정 (WSL 기준 예시)
/mnt/d/wiki/wiki/wiki/projects/Deep_voice/.venv/bin/python3 export_for_submit.py \
    --checkpoint checkpoints/music_classifier_best.pt
```

- `export/music_classifier/weights.pt` + `export/music_classifier/music_fake_infer.py` 생성
- `music_fake_infer.py`는 이 프로젝트의 다른 파일에 의존하지 않는 **자기완결 추론 모듈**
- 새로 만든 `weights.pt`가 실제로 갱신됐는지는 파일 크기/수정시각만으로 확신하지 말 것 — 여러 체크포인트가 파라미터 수는 같아서 파일 크기가 우연히 같을 수 있다. 필요하면 아래로 이전 것과 다른 파일인지 확인한다:
  ```bash
  python3 -c "import zlib; print(hex(zlib.crc32(open('export/music_classifier/weights.pt','rb').read())))"
  ```

### Step 5 — submit.zip 구성 (직접 진행하는 절차)

`baseline_submit/` 디렉토리(이미 `baseline_submit.zip`이 한 번 압축 해제되어 있고, `model/df_arena_1b`·`model/htdemucs`·`model/panns`가 들어있는 상태)를 기준으로 진행한다. 매번 `baseline_submit.zip`을 새로 풀 필요는 없다 — 이미 풀려있는 폴더를 계속 재사용하면 된다.

1. **최신 체크포인트 export** (Step 4). `export/music_classifier/`에 `weights.pt`, `music_fake_infer.py`가 생겼는지 확인.

2. **`music_classifier` 갱신** — export 결과를 `baseline_submit/model/music_classifier/`로 복사(덮어쓰기):
   ```bash
   cp music_fake_pipeline/music_fake_pipeline/export/music_classifier/weights.pt \
      baseline_submit/model/music_classifier/weights.pt
   cp music_fake_pipeline/music_fake_pipeline/export/music_classifier/music_fake_infer.py \
      baseline_submit/model/music_classifier/music_fake_infer.py
   ```

3. **`script.py` 확인** — `baseline_submit/script.py`가 최신 `patched_script.py`와 같은 내용인지 diff로 확인하고, 다르면 통째로 교체한다:
   ```bash
   diff baseline_submit/script.py music_fake_pipeline/music_fake_pipeline/patched_script.py
   # 다르면:
   cp music_fake_pipeline/music_fake_pipeline/patched_script.py baseline_submit/script.py
   ```
   > ⚠️ **알려진 함정 (2026-09-22 수정됨)**: 한때 `main()`이 `args = parse_arguments([])`로 빈 리스트를 넘겨 `--test-dir`/`--sample-submission`/`--output`/`--device` 등 **모든 CLI 인자를 무시하고 하드코딩된 기본값만 쓰는 버그**가 있었다. 지금은 `args = parse_arguments()`로 고쳐져 있다. `patched_script.py`를 손으로 다시 수정할 일이 있으면 이 줄이 되돌아가지 않았는지 반드시 확인할 것.

4. **`requirements.txt`** — 수정 불필요 (음악 분류기가 요구하는 패키지는 서버에 이미 있는 torch/torchaudio뿐).

5. **로컬 스모크 테스트** — 압축하기 전에 더미 3개 파일로 전체 파이프라인이 에러 없이 도는지, 그리고 **CLI 인자가 실제로 반영되는지** 확인한다 (출력 경로를 기본값이 아닌 곳으로 지정해서, 그 경로에 파일이 생기는지 보면 인자 무시 버그도 함께 잡을 수 있다):
   ```bash
   cd baseline_submit
   /mnt/d/wiki/wiki/wiki/projects/Deep_voice/.venv/bin/python3 script.py \
       --test-dir ../data/test \
       --sample-submission ../data/sample_submission.csv \
       --output /tmp/submission_smoketest.csv \
       --device cuda
   cat /tmp/submission_smoketest.csv
   ```
   - `TEST_0000`~`TEST_0002` 3행이 `sample_submission.csv`와 동일한 컬럼으로 출력되면 정상.
   - 에러가 나거나, `/tmp/submission_smoketest.csv`가 생기지 않고 대신 `baseline_submit/output/submission.csv`에 생겼다면 인자가 무시되고 있다는 뜻 — 3번의 함정을 다시 확인한다.

6. **압축** — `baseline_submit/` **폴더 자체가 아니라 그 안의 내용물**이 zip 최상위에 오도록 압축한다:
   ```
   submit.zip
   ├── model/            (df_arena_1b, htdemucs, panns, music_classifier)
   ├── script.py
   └── requirements.txt
   ```
   `baseline_submit/`이라는 폴더가 통째로 최상위에 들어가면(예: `submit.zip/baseline_submit/model/...`) **설치 오류**로 처리된다(§10 참고). `baseline_submit/` 디렉토리 안에서 그 내용물만 선택해 압축하면 된다.

7. **압축 후 확인**: zip 용량 ≤ 10GB, 압축 해제 후 ≤ 32GB (§10 제약 조건 참고). 가능하면 방금 만든 zip을 임시 폴더에 풀어서 최상위 항목이 `model/`, `script.py`, `requirements.txt` 3개뿐인지 한 번 더 확인한다.

---

## 9. 실행 방법

### baseline script.py 실행 (로컬 검증용)

> `script.py`는 `baseline_submit.zip` 내부에만 있다. 먼저 추출해야 한다.

```bash
# zip에서 script.py 추출
python -c "import zipfile; zipfile.ZipFile('baseline_submit.zip').extract('script.py')"

# 실행
python script.py \
    --test-dir data/test \
    --sample-submission data/sample_submission.csv \
    --output output/submission.csv \
    --device cuda
```

### 입출력 규격

- **입력**: `data/test/` 내 오디오 파일 (`.aac` `.flac` `.m4a` `.mp3` `.ogg` `.opus` `.wav` `.wma`)
- **출력**: `output/submission.csv`

```csv
ID,FILE_FAKE_PROB,VOICE_FAKE_PROB,MUSIC_FAKE_PROB,VOICE_PRESENT_PROB,MUSIC_PRESENT_PROB
TEST_0000,0.5,0.5,0.5,0.5,0.5
```

---

## 10. 제출 규격 및 주의사항

### submit.zip 구조 (통합 완료 후)

```
submit.zip
├── model/
│   ├── df_arena_1b/           # 기존, 그대로
│   ├── htdemucs/              # 기존, 그대로
│   ├── panns/                 # 기존, 그대로
│   └── music_classifier/      # 신규 추가
│       ├── weights.pt
│       └── music_fake_infer.py
├── script.py                  # patched_script.py로 교체
└── requirements.txt           # 수정 불필요
```

최상위에 불필요한 폴더가 추가되면 **설치 오류** 발생. 최상위 3개 항목만 존재해야 한다.

### 제약 조건

| 항목 | 제한 |
|------|------|
| zip 용량 | ≤ 10GB (압축 해제 후 ≤ 32GB) |
| 패키지 설치 시간 | ≤ 10분 |
| 추론 실행 시간 | ≤ 60분 (1,200개 파일) |
| 인터넷 접속 | 비활성화 |
| **일일 제출 횟수** | **최대 3회** |

### 오류 유형별 제출 횟수 차감

| 오류 종류 | 차감 여부 |
|----------|----------|
| 설치 오류 (zip 구조 불일치, 패키지 설치 실패) | ❌ 차감 안 됨 |
| 제출 오류 (script.py 실행 중 에러) | ✅ 차감됨 |

### 핵심 금지 규칙 (위반 시 실격)

1. **테스트 데이터를 활용한 추가 학습·튜닝·Pseudo-Labeling 금지**
2. **파일 단위 독립 예측 원칙**: 다른 파일의 정보·예측값·통계를 활용해 예측값을 생성하거나 보정하는 것 금지

### 제출 전 검증 체크리스트

- [ ] `data/test/`의 더미 3개 파일로 `patched_script.py` 전체 실행이 에러 없이 완료되는지
- [ ] `output/submission.csv`가 `sample_submission.csv`와 동일한 컬럼/행 수로 생성되는지
- [ ] (가능하면) Python 3.11.15 + CUDA 12.8 + torch 2.7.1+cu128 환경에서 한 번 더 실행
- [ ] `submit.zip` 압축 시 최상위에 불필요한 폴더가 끼지 않았는지
- [ ] zip 용량 10GB 이하, 압축 해제 후 32GB 이하
- [ ] 60분 추론 제한 — `music_classifier`는 DF-Arena 대비 훨씬 가벼우나, 실측 확인 필요

---

## 11. 데이터

### 대회 제공 데이터

별도 학습 데이터셋 미제공. 참가자가 직접 구성해야 한다.

### 사용 데이터셋 (2026-09-22 기준)

| 용도 | 데이터셋 | 특징 | 라이선스 | 상태 |
|------|---------|------|----------|------|
| FAKE 음악 | FakeMusicCaps (Zenodo 15063698) | 생성기 5종(audioldm2/MusicGen_medium/musicldm/mustango/stable_audio_open) | CC-BY-NC-4.0 | ✅ 확보 완료 — **55,216개** |
| REAL 음악 | MusicCaps | FakeMusicCaps가 원래 이 데이터셋 기반으로 생성됨 → 가장 자연스러운 Real 짝으로 채택 (FMA 계획은 폐기) | 확인 필요 | ✅ 확보 완료 — **150개** (`download_musiccaps.py`, yt-dlp 기반) |
| REAL 음악 (추가) | mtg-jamendo-dataset | Real 규모 확대용 후보 | 확인 필요 | 🔄 진행 중 — 클론 완료, `venv`(Python 3.12) `pip install`이 `setuptools`/`distutils` 빌드 오류로 실패 (해결책: `pip install --upgrade setuptools`, 아직 실행 미확인) |
| FAKE 음악 | Echoes | 생성기 12종, 일반화 검증용 | 확인 필요 | 미확보 |
| REAL/FAKE | SONICS | 대용량, 서브샘플링 권장 | 확인 필요 | 미확보 |

> ⚠️ 각 데이터셋의 라이선스 조항(특히 대회 수상 시 비상업적 범위 해당 여부)은 실제 학습 전 대회 규칙 탭 및 각 데이터셋 원문에서 직접 확인·준수해야 한다. **아직 최종 확인 전.**

### 학습 데이터 구성 원칙

- FakeMusicCaps(Fake) + MusicCaps(Real)로 이진 분류 학습 데이터 구성
- 실제 추론 시 HTDemucs로 분리된 accompaniment가 모델 입력 → 학습 데이터도 동일하게 HTDemucs 분리 적용 (`prepare_data.py`로 완료)
- train/val 분할은 **트랙(파일) 단위** — 세그먼트 단위 분할은 data leakage 발생
- **클래스 불균형**: Real 150개 vs Fake 55,216개로 극단적 불균형 → 현재는 Fake 서브샘플링(`--limit`)으로 대응 중, `train.py`에 `pos_weight` 등 정식 클래스 가중치는 아직 미구현

---

## 12. 검증 현황

| 항목 | 상태 |
|------|------|
| 전체 파이프라인 합성 데이터 end-to-end 실행 | ✅ 정상 동작 확인 |
| `MusicFakeClassifier` 텐서 채널 차원 버그 발견·수정 | ✅ 수정 완료 |
| `music_fake_infer.py` 독립 실행(다른 파일 의존 없음) 확인 | ✅ 확인 |
| `patched_script.py` vs baseline diff — 음악 분기만 교체 확인 | ✅ 확인 |
| `demucs 4.0.1` vs `4.1.0` API 차이 직접 재현 | ✅ 확인 |
| **실제 HTDemucs 분리 결과물(MusicCaps/FakeMusicCaps)에 대한 학습** | ✅ 실행 완료 — 학습 로그 기준 val EER **최저 4.67%** (30 epoch, 기존 하이퍼파라미터) |
| 하이퍼파라미터 재조정 후 재학습(batch 16 / lr 1e-4 / epoch 100) | 🔄 진행 중 — epoch 9 시점 val EER 7.0%, 정상 종료 여부 미확인 |
| **실제 대회 평가 데이터(리더보드) 결과** | ⚠️ 확인됨, 그러나 예상과 반대 — **09-20 제출은 baseline 그대로(Music도 DF-Arena 사용)**로 총점 0.6909(ADS 0.6578/CPS 0.9893), **09-21 제출이 이 프로젝트의 `music_classifier`를 처음 적용**한 것으로 총점 0.6722(ADS 0.6370/CPS 0.9893). 즉 **DF-Arena zero-shot을 이 프로젝트의 소형 CNN으로 교체한 것 자체가 실제 대회 데이터에서는 오히려 손해**였다는 뜻. 원인은 Real 150개/Fake 5개 생성기라는 좁은 학습 데이터로 인한 과적합·분포 불일치로 추정(§4, §11 참고) |
| 검증 세트 규모 | ⚠️ 초기 실행 기준 44개로 작음 — EER 신뢰구간 넓음, 데이터 확대(mtg-jamendo) 시도 중 |
| 평가 서버에서의 실행 시간 및 zip 용량 | ❌ 미확인 |

---

## 13. 참고 자료

### 모델

- **DF-Arena 1B**: [Speech-Arena-2025/DF_Arena_1B_V_1 (Hugging Face)](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1)
- **HTDemucs**: [facebookresearch/demucs (GitHub)](https://github.com/facebookresearch/demucs)
- **PANNs Cnn14**: [qiuqiangkong/panns_inference (GitHub)](https://github.com/qiuqiangkong/panns_inference)

### 데이터셋

- **FakeMusicCaps**: [Zenodo 15063698](https://zenodo.org/records/15063698)

### 논문

- Kulkarni et al. (2026). *Do Compact SSL Backbones Matter for Audio Deepfake Detection? A Controlled Study with RAPTOR.* arXiv:2603.06164
