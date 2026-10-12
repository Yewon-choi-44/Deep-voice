# 딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회

> 주최: 행정안전부 · 한국지능정보사회진흥원 / 주관: 국립과학수사연구원 / 운영: DACON  
> 리더보드 제출 마감 2026-09-29 10:00 KST · 대회 종료 2026-09-30 10:00 KST — **프로젝트 종료, 2026-10-12 최종 정리**

> **브랜치 안내**: `v1` = 2026-09-22 파이프라인 자동화 이전(각 스크립트를 하나씩 수동 실행). `v2` = 학습 진행 게이지, 실행별 로그 분리(`run_logger.py`), 원샷 파이프라인(`run_pipeline.py`), `prepare_data.py` 증분 처리, 경로 버그 수정까지 반영된 최종 버전 — 이 문서는 `v2` 기준으로 작성됨. 상세 변경 이력은 `dev_log.md` 참고.

---

## 프로젝트 요약

**무엇을**: 공개된 베이스라인(음원 분리 HTDemucs → 음성/음악 존재 판별 PANNs → 가짜 판별 DF-Arena 1B)에서 **음악 Fake 판별 경로(`MUSIC_FAKE_PROB`)만** 직접 학습한 소형 CNN으로 교체하고, 데이터 수집부터 학습·평가·제출 패키징까지를 재현 가능한 파이프라인으로 만들었다.

**왜 이 경로인가**: 평가식에서 Music EER 가중치(0.3)가 Voice(0.2)보다 크고, 베이스라인의 DF-Arena는 음성용 모델인데 음악 판별에도 검증 없이 쓰이고 있었다.

### 결과

| 제출 | 내용 | 총점 | ADS | CPS |
|------|------|------|-----|-----|
| 2026-09-20 | 순수 baseline (Music도 DF-Arena) | 0.6909 | 0.6578 | 0.9893 |
| 2026-09-21 | `music_classifier` 1차 (Real 150개로 학습) | 0.6722 | 0.6370 | 0.9893 |
| **2026-09-28 확인** | **`music_classifier` 3차 (학습 Real/Fake 각 2,550개, 검증 각 450개)** | **0.70871** | **0.67753** | 0.98932 |

- 공개 리더보드 최고 점수 **0.70871**로 baseline(0.6909)을 **+0.0178** 넘어섬. CPS(음성·음악 존재 판별)는 세 제출에서 사실상 같으므로 개선분은 File/Music EER 경로에서만 나온 것
- 추론 시간은 `music_classifier`를 넣은 버전이 15분 51초로 60분 제한에 여유가 있었고, 제출 패키지는 4.25 GiB로 10GB 제한 이내
- **최종 리더보드 순위: 480 중 308위**

### 만든 것

- `MusicFakeClassifier`: log-mel 입력, Conv-BN-ReLU-MaxPool 4블록 + GAP + FC, 약 25만 파라미터 (DF-Arena 1B의 약 1/4,000)
- 데이터 파이프라인: HTDemucs로 반주만 분리해 학습(추론 때 모델이 보는 입력과 같은 분포), 트랙 단위 분할 설계
- `run_pipeline.py` 원샷 실행, 실행별 loss 로그(`run_logger.py`), 평가식과 동일한 EER 계산(`evaluate.py`)
- 제출 패키징·스모크 테스트 절차 문서화 (`SUBMIT_INTEGRATION.md`)

### 배운 것 (핵심만, 상세는 §14)

- **로컬 val 지표와 리더보드가 반대로 움직일 수 있다.** 09-21에 로컬 val EER은 13.6%→4.67%로 좋아졌는데 리더보드는 오히려 하락했다(0.6909→0.6722). val이 44개뿐이고 학습 데이터와 같은 좁은 분포(Real 150개)였던 탓으로 추정했고, 데이터를 넓힌 뒤 점수가 올랐다. 다만 그 사이 스케줄러·patience·배치·학습률도 바뀌어서 데이터 확대가 단독 원인이라고 확정하지는 못했다.
- **스케줄러와 조기 종료는 서로 맞물려야 한다.** `CosineAnnealingLR`(100 epoch 전제)과 patience 6 조기 종료가 충돌해 LR이 거의 안 내려간 채 학습이 끝나고 있었고, `ReduceLROnPlateau`로 바꿔 해결했다. patience는 로그에서 "5 epoch 정체 후 개선" 사례를 세어 근거를 만들었다.
- **종료 후 스스로 점검했을 때 학습 데이터에 결함이 있었다.** FakeMusicCaps 5개 생성기의 파일명이 같아 분리 결과가 덮어쓰기/스킵되며 Fake가 사실상 MusicGen_medium **한 종류**였고, 마지막 학습의 검증 세트는 학습 도중 바뀌었다. 그래서 마지막 학습의 val EER 0.75%는 성과로 쓰지 않았고 대표 성과는 리더보드 점수 하나로 한정했다.

### 한계

- 리더보드 0.70871은 **여러 생성기에 대한 일반화를 입증하지 않는다.** 학습에 쓰인 Fake는 단일 생성기였고, 개선 원인이 Fake 학습 때문인지 Real 음악 도메인 학습 때문인지 분리하는 실험은 하지 않았다.
- 마지막 재학습(87 epoch)은 마감 이후에 끝나 제출되지 않았다. 리더보드로 검증된 모델은 3차 체크포인트뿐이다.
- 최종 리더보드 순위는 **480 중 308위**였다. baseline을 넘는 개선이 상위권 진입으로 이어지지는 않았다.

> 근거가 되는 수치·파일 대조 과정은 [`dev_log.md`](dev_log.md)의 "2026-10-12 — 마무리 사후 점검"에 있다.

---

## 목차

0. [프로젝트 요약](#프로젝트-요약)
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
14. [최종 결과 및 회고](#14-최종-결과-및-회고)

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

> ⚠️ **09-21 리더보드 결과로 이 가설이 한때 검증되지 않은 것처럼 보였음.** DF-Arena를 그대로 쓴 09-20 제출(총점 0.6909)보다, 좁은 데이터(Real 150개)로 학습한 09-21 제출(총점 0.6722)의 점수가 더 낮았다 — 당시엔 "Music 경로 교체" 자체가 손해로 보였다.
>
> ✅ **2026-09-28 갱신 — 이번 제출(데이터 확대 + `ReduceLROnPlateau`/patience=10 재조정 버전)에서 처음으로 baseline을 돌파함.** 총점 **0.70871**(ADS 0.67753 / CPS 0.98932) — baseline(0.6909)과 09-21(0.6722) 모두를 상회. CPS가 세 제출 모두에서 거의 동일(0.9893 → 0.9893)한 것으로 보아 이번 개선분은 전부 File/Music EER 경로에서 나온 것이며, "Music 경로 교체"가 baseline보다 나을 수 있다는 가설은 데이터를 넓힌 뒤 리더보드에서 뒷받침됨. 상세는 `dev_log.md` 2026-09-23~2026-09-28 항목 참고.
>
> ⚠️ **2026-10-12 단서**: 종료 후 점검에서 이 모델이 본 Fake가 **MusicGen_medium 한 종류뿐**이었음이 확인됐다(§11, §14). 따라서 이 결과는 "다양한 생성기에 일반화된다"는 뜻이 아니라 "이 데이터로 학습한 소형 CNN이 DF-Arena zero-shot보다 숨겨진 테스트셋에서 나았다"까지만 말한다. 개선 원인이 Fake 학습인지 Real 음악 도메인 학습인지는 분리 실험을 하지 않아 알 수 없다.

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

아래 트리에서 **(제외)** 표시는 로컬 작업 환경에는 있지만 이 저장소에는 포함하지 않은 대회 제공물·파생물이다(§14.5).

```
Deep_voice/
├── README.md                         # 이 문서
├── deepvoice_competition_summary.md  # (제외) 대회 규정·평가·일정 정리 노트
├── main.py                           # placeholder (미사용)
├── .python-version                   # Python 3.11
├── .venv/                            # 가상환경
│
├── dev_log.md                        # 날짜별 개발 일지 + 종료 후 사후 점검
├── 제출용_기본파일/                  # (제외) 제출 zip 백업 (각 4.5GB) — baseline_submit.zip 원본 포함
├── baseline_submit/                  # baseline_submit.zip 압축 해제본 + music_classifier (제출 패키지의 작업 디렉토리)
│   ├── requirements.txt              # (제외)
│   ├── script.py                     # (제외) baseline script.py에 patched_script.diff를 적용한 제출용 스크립트
│   └── model/
│       ├── MODEL_INFO.txt            # (제외)
│       ├── SHA256SUMS.txt            # (제외)
│       ├── df_arena_1b/              # (제외) DF-Arena 1B 가중치 및 소스
│       ├── htdemucs/                 # (제외) HTDemucs 가중치
│       ├── panns/                    # (제외) PANNs Cnn14 가중치 + 레이블 설정
│       └── music_classifier/         # 직접 학습한 분류기 (3차 체크포인트, 리더보드 0.70871)
│           ├── weights.pt            # 약 1MB, CRC32 f6e9e676
│           └── music_fake_infer.py   # 다른 파일에 의존하지 않는 자기완결 추론 모듈
│
├── music_fake_pipeline/
│   └── music_fake_pipeline/          # MUSIC_FAKE_PROB 교체 파이프라인 (상세: 내부 README 참고)
│       ├── config.py
│       ├── audio_utils.py
│       ├── demucs_separate.py
│       ├── features.py
│       ├── model.py                  # MusicFakeClassifier
│       ├── dataset.py
│       ├── prepare_data.py           # v2: 이미 분리 끝난 파일은 재실행 시 건너뜀
│       ├── train.py                  # v2: 학습 진행 게이지, run_logger 사용
│       ├── evaluate.py
│       ├── export_for_submit.py
│       ├── run_logger.py             # v2 신규: 실행별 logs/{시각}.csv 로그 생성
│       ├── run_pipeline.py           # v2 신규: prepare→train→evaluate→export 원샷 실행
│       ├── SUBMIT_INTEGRATION.md     # 제출 파일 구성 상세 절차
│       ├── patched_script.diff       # baseline script.py 대비 내가 바꾼 부분만 (patched_script.py 자체는 제외)
│       ├── requirements-dev.txt
│       └── data/
│           ├── raw/real_music/       # ← REAL 음악 원본 (직접 채울 것)
│           ├── raw/fake_music/       # ← Fake 음악 원본 (직접 채울 것)
│           ├── separated/{real,fake}/
│           └── manifests/
│
└── data/                             # (제외) 형식 확인용 더미 데이터
    ├── sample_submission.csv
    └── test/
        ├── TEST_0000.wav
        ├── TEST_0001.wav
        └── TEST_0002.wav
```

> `data/test/`는 형식 확인용 더미(3개)다. 실제 평가 시 서버가 1,200개 파일로 자동 교체한다.  
> 제출용 `script.py`는 대회 제공 baseline `script.py`(CRC32 `bf8e3455`)에 `patched_script.diff`를 적용한 결과(CRC32 `84248641`)다. 원본은 대회 페이지의 baseline 패키지(`baseline_submit.zip`)에 들어 있다.

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

로컬 학습은 Windows 11 + Python 3.13.2(시스템) + NVIDIA Quadro RTX 5000(16GB, 드라이버 CUDA 13.2) + `torch==2.11.0+cu128` 환경에서 진행했다. `pip install torch`를 옵션 없이 실행하면 CPU 전용 빌드가 잡히는 문제 등 실제로 겪은 삽질과 해결 과정은 `music_fake_pipeline/music_fake_pipeline/README.md` 9장에 날것으로 기록되어 있다. **제출 전에는 반드시 위 표의 평가 서버 스펙(Python 3.11.15 / CUDA 12.8 / torch 2.7.1+cu128)에서 한 번 더 검증**해야 한다 — 로컬 환경과 버전이 달라 체크포인트는 로드되어도 추론 결과가 미묘하게 달라질 위험이 있다.

---

## 8. 개발 워크플로우

`music_fake_pipeline/music_fake_pipeline/` 디렉토리 안에서 실행한다.

### Step 0 — 원샷 실행 (v2, 권장)

아래 Step 1~4를 매번 하나씩 따로 실행하기 번거롭다면, `run_pipeline.py` 하나로
기본값 그대로 전체를 순서대로 실행할 수 있다:

```bash
# 반드시 프로젝트 루트 .venv(torch 포함)의 python으로 실행할 것
../../.venv/bin/python3 run_pipeline.py

# data/manifests가 이미 최신이라 데이터 준비를 건너뛰고 싶을 때
../../.venv/bin/python3 run_pipeline.py --skip-prepare
```

`prepare_data.py`(증분 처리 — 이미 끝난 파일은 건너뜀) → `train.py` → `evaluate.py`
→ `export_for_submit.py`를 이어서 실행하고, 한 단계라도 실패하면 즉시 멈춘다.
세부 옵션(`--epochs`, `--limit` 등)을 조정하고 싶으면 아래 Step 1~4처럼 해당
스크립트를 개별 실행한다.

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
- v2부터는 **이미 HTDemucs 분리가 끝난 파일은 재실행 시 건너뛴다** — 원본 데이터를 나중에 추가했을 때, 전체를 다시 돌려도 새로 추가된 파일만 실제로 처리됨

> ⚠️ **알려진 결함 (2026-10-12 사후 점검에서 발견, 코드는 결과 재현을 위해 수정하지 않음)**
>
> 1. **파일명 충돌**: 분리 결과를 `{원본 파일명 stem}.npy`로 저장하는데, FakeMusicCaps는 5개 생성기 폴더가 같은 파일명을 쓴다. 먼저 처리된 파일만 남고 같은 이름의 나머지는 "이미 있음"으로 건너뛰어진다. 실측으로는 분리된 Fake `.npy` 5,521개가 전부 `MusicGen_medium`의 길이와 일치한다(정렬상 가장 먼저 처리돼서라는 것은 해석). 현재 남아 있는 분리 산출물과 이를 쓴 3·4차 재학습이 이 상태다(앞선 학습의 manifest는 남아 있지 않아 확인 불가). 다른 데이터에 쓰려면 저장 파일명에 하위 폴더 이름을 포함시킬 것.
> 2. **manifest 중복·누수**: 위 충돌 때문에 서로 다른 행이 같은 `.npy`를 가리키고, 행 단위로 train/val을 나누므로 같은 오디오가 양쪽에 들어갈 수 있다.
> 3. **`--limit`은 정렬 순서 앞쪽 N개**를 고른다. 클래스 안에서 무작위 추출이 아니므로 폴더 구성에 따라 한 생성기로 쏠린다.
> 4. **학습 중 manifest를 덮어쓰지 말 것**: `train.py`는 val manifest를 매 epoch 다시 읽는다(train은 시작 때 한 번). 학습이 도는 동안 `prepare_data.py`를 다시 돌리면 val만 바뀐다.

### Step 2 — 학습

```bash
python train.py
# 또는 하이퍼파라미터 직접 지정
python train.py --epochs 30 --batch-size 32 --device cuda
```

- val EER 기준으로 best checkpoint(`checkpoints/music_classifier_best.pt`) 갱신
- `EARLY_STOP_PATIENCE` epoch 동안 개선 없으면 자동 종료. 3차 재학습(리더보드 0.70871)까지는 **10** (실측 데이터 기반 산정 근거는 `dev_log.md` 참고), 현재 `config.py` 기본값은 4차 재학습에 쓴 **20**(`WEIGHT_DECAY`도 1e-5 → 1e-3으로 같이 바뀐 상태)
- LR 스케줄러는 `ReduceLROnPlateau` — val EER이 정체되면 LR을 자동으로 절반씩 낮춤 (`CosineAnnealingLR`은 조기 종료와 구조적으로 맞지 않아 v2에서 교체함)
- v2부터는 학습 진행 게이지(tqdm)가 매 epoch의 배치 진행률과 실시간 평균 loss를 보여주고, 실행마다 `logs/{실행 시각}.csv`에 별도로 기록됨 (epoch, train_loss, val_eer, elapsed_sec)

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

3. **`script.py` 구성** — 대회 제공 baseline `script.py`(CRC32 `bf8e3455`)에 `patched_script.diff`를 적용해 `baseline_submit/script.py`를 만든다. (`patched_script.py` 전체는 대회 제공 코드와 거의 같아 저장소에 두지 않고 변경분만 diff로 보관한다.)
   ```bash
   # <원본>은 baseline_submit.zip 안의 script.py를 따로 풀어 둔 경로
   patch -o baseline_submit/script.py <원본>/script.py \
         < music_fake_pipeline/music_fake_pipeline/patched_script.diff
   # 결과 확인: CRC32가 84248641 이어야 함
   python3 -c "import zlib; print('%08x' % (zlib.crc32(open('baseline_submit/script.py','rb').read()) & 0xffffffff))"
   ```
   > ⚠️ **알려진 함정 (2026-09-22 수정됨)**: 한때 `main()`이 `args = parse_arguments([])`로 빈 리스트를 넘겨 `--test-dir`/`--sample-submission`/`--output`/`--device` 등 **모든 CLI 인자를 무시하고 하드코딩된 기본값만 쓰는 버그**가 있었다. 지금은 `args = parse_arguments()`로 고쳐져 있다(diff에 포함). 스크립트를 손으로 다시 수정할 일이 있으면 이 줄이 되돌아가지 않았는지 반드시 확인할 것.

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

> 원본 baseline `script.py`는 대회 페이지의 baseline 패키지(`baseline_submit.zip`) 안에만 있다(이 저장소에는 포함하지 않음). 먼저 추출해야 한다. 아래 경로는 원본 zip을 `제출용_기본파일/`에 보관한 로컬 환경 기준이다.

```bash
# zip에서 원본 script.py 추출
python -c "import zipfile; zipfile.ZipFile('제출용_기본파일/baseline_submit.zip').extract('script.py')"

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
├── script.py                  # baseline에 patched_script.diff를 적용한 것
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

- [ ] `data/test/`의 더미 3개 파일로 패치 적용된 `script.py` 전체 실행이 에러 없이 완료되는지
- [ ] `output/submission.csv`가 `sample_submission.csv`와 동일한 컬럼/행 수로 생성되는지
- [ ] (가능하면) Python 3.11.15 + CUDA 12.8 + torch 2.7.1+cu128 환경에서 한 번 더 실행
- [ ] `submit.zip` 압축 시 최상위에 불필요한 폴더가 끼지 않았는지
- [ ] zip 용량 10GB 이하, 압축 해제 후 32GB 이하
- [ ] 60분 추론 제한 — `music_classifier`는 DF-Arena 대비 훨씬 가벼우나, 실측 확인 필요

---

## 11. 데이터

### 대회 제공 데이터

별도 학습 데이터셋 미제공. 참가자가 직접 구성해야 한다.

### 사용 데이터셋 (2026-10-12 실측 기준)

| 용도 | 데이터셋 | 특징 | 라이선스 | 상태 |
|------|---------|------|----------|------|
| FAKE 음악 | FakeMusicCaps (Zenodo 15063698) | 생성기 5종(audioldm2/MusicGen_medium/musicldm/mustango/stable_audio_open) 폴더 구성. **그러나 학습에 실제로 쓰인 것은 MusicGen_medium 한 종류** (아래 경고 참고) | CC-BY-NC-4.0 | raw **27,604개**(5종 × 약 5,521). HTDemucs 분리 `.npy` 고유 **5,521개**, 전부 MusicGen_medium 길이(162,880 samples)와 일치 |
| REAL 음악 | MusicCaps + mtg-jamendo-dataset | FakeMusicCaps가 원래 MusicCaps 기반으로 생성됨 → 가장 자연스러운 Real 짝. mtg-jamendo는 `setuptools` 빌드 오류(09-21) 해결 후 대량 확보 | 확인 필요 | raw **18,635개**(MusicCaps 150 + mtg-jamendo 다수), HTDemucs 분리 `.npy` **8,016개** |
| FAKE 음악 | Echoes | 생성기 12종, 일반화 검증용 | 확인 필요 | 미확보 |
| REAL/FAKE | SONICS | 대용량, 서브샘플링 권장 | 확인 필요 | 미확보 |

> ⚠️ 각 데이터셋의 라이선스 조항(특히 대회 수상 시 비상업적 범위 해당 여부)은 실제 학습 전 대회 규칙 탭 및 각 데이터셋 원문에서 직접 확인·준수해야 한다. **아직 최종 확인 전.**

> ⚠️ **Fake 데이터 구성에 대한 정정 (2026-10-12)**: 이전 버전의 이 문서와 제출 메모는 Fake를 "FakeMusicCaps 5개 생성기 중 서브샘플링"이라고 적었으나 사실이 아니다. 5개 생성기 폴더가 **같은 파일명**을 쓰고 분리 결과가 `{stem}.npy`로 저장되어 충돌했기 때문이다. 그 결과 고유 Fake 오디오는 5,521개가 상한이고 전부 MusicGen_medium이다. 같은 이유로 09-28에 실행한 `run_pipeline.py --limit 6000`(Fake 분리 확대)은 새 오디오를 늘리지 못하고 중복 행만 늘렸다. 상세는 `dev_log.md` "2026-10-12 — 마무리 사후 점검".

### 학습 데이터 구성 원칙

- FakeMusicCaps(Fake) + MusicCaps/mtg-jamendo(Real)로 이진 분류 학습 데이터 구성
- 실제 추론 시 HTDemucs로 분리된 accompaniment가 모델 입력 → 학습 데이터도 동일하게 HTDemucs 분리 적용 (`prepare_data.py`로 완료, v2부터 이미 처리된 파일은 재실행 시 건너뜀)
- train/val 분할은 **트랙(파일) 단위** — 세그먼트 단위 분할은 data leakage 발생
- **클래스 균형**: 두 클래스를 1:1로 맞춰 샘플링. 리더보드에 제출한 3차 재학습(2026-09-23)은 학습셋 5,100개(Real 2,550 : Fake 2,550) / 검증셋 900개(Real 450 : Fake 450). `train.py`에 `pos_weight` 등 정식 클래스 가중치는 필요 없는 상태(이미 balanced sampling)
- **4차(최종) 재학습의 데이터는 정확히 복원할 수 없음**: 학습 도중(09-29 00:52) manifest가 16,000행(8,000 : 8,000)짜리로 덮어써졌고 시작 시점의 train.csv는 남아 있지 않다. 현재 manifest의 Fake 8,000행은 고유 5,521개를 가리키며 val의 고유 Fake 1,144개 중 640개가 train과 겹친다.

---

## 12. 검증 현황

| 항목 | 상태 |
|------|------|
| 전체 파이프라인 합성 데이터 end-to-end 실행 | ✅ 정상 동작 확인 |
| `MusicFakeClassifier` 텐서 채널 차원 버그 발견·수정 | ✅ 수정 완료 |
| `music_fake_infer.py` 독립 실행(다른 파일 의존 없음) 확인 | ✅ 확인 |
| `patched_script.diff`(baseline 대비 변경분) — 음악 분기만 교체 확인, 원본에 적용하면 제출본(CRC32 `84248641`)이 그대로 재현됨 | ✅ 확인 |
| `demucs 4.0.1` vs `4.1.0` API 차이 직접 재현 | ✅ 확인 |
| 실제 HTDemucs 분리 결과물(MusicCaps/FakeMusicCaps)에 대한 학습 (1차, 좁은 데이터) | ✅ 실행 완료 — val EER **4.67%** (30 epoch, `CosineAnnealingLR`+patience 6) |
| 데이터 확대(mtg-jamendo) + 하이퍼파라미터 재조정 후 재학습 (2차) | ✅ 실행 완료 — val EER **5.33%**(epoch 17, `CosineAnnealingLR`+patience 6, epoch 23 조기 종료) |
| `ReduceLROnPlateau`+patience 10 교체 후 재학습 (3차, 학습 5,100/검증 900) | ✅ 실행 완료 — **best val EER 3.11%**(epoch 31, train loss 0.0607), 10 epoch 미개선으로 epoch 41 조기 종료(최종 train loss 0.0457). 배치 16 / lr 1e-4 / weight decay 1e-5 |
| 마지막 재학습 (4차, `ReduceLROnPlateau`+patience 20+weight decay 1e-3, 데이터 확대) | ⚠️ 실행 완료, **성과로 쓰지 않음** — epoch 67에서 val EER 0.75%(train loss 0.0219), epoch 87 조기 종료(train loss 0.0166), 약 69시간. 학습 도중 val manifest가 바뀌었고 train/val 중복·단일 생성기 문제가 있어 3차와 비교 불가. 마감 후 종료되어 export·제출하지 않음 |
| **실제 대회 평가 데이터(리더보드) 결과** | ✅ **3차 재학습 체크포인트에서 baseline 최초 돌파.** 09-20 baseline 총점 0.6909(ADS 0.6578/CPS 0.9893) → 09-21 좁은 데이터 `music_classifier` 총점 0.6722(ADS 0.6370/CPS 0.9893, baseline보다 낮았음) → **2026-09-28 확인: 총점 0.70871(ADS 0.67753/CPS 0.98932)**, baseline·09-21 모두 상회. CPS가 세 제출에서 거의 동일해 개선분은 File/Music EER에서만 발생한 것으로 판단됨 |
| 검증 세트 규모 | 초기 44개 → 3차 900개(Real 450 : Fake 450)로 개선. 3차 val 방향은 리더보드와 일치했으나, 평가 대상이 **단일 생성기(MusicGen_medium)** 라는 한계가 있음 |
| 평가 서버에서의 실행 시간 및 zip 용량 | ✅ 리더보드 소요 시간 15분 51초(`music_classifier` 포함, 09-21) / 21분 41초(baseline) — 60분 제한 대비 여유. `submit.zip` 4.25 GiB (≤ 10GB). 09-28 제출의 소요 시간은 기록 없음 |
| 미생성기(unseen generator) 일반화 | ❌ **측정하지 않음.** 생성기 hold-out 검증을 한 적 없음 |

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

---

## 14. 최종 결과 및 회고

### 14.1 진행 요약

| 날짜 | 일어난 일 |
|------|-----------|
| 09-20 | 규정·baseline 코드 분석, `music_fake_pipeline` 검토, 첫 제출 (사후 확인: 이 제출은 `music_classifier`가 빠진 **순수 baseline**이었음 → 0.6909) |
| 09-21 | 로컬 환경(CUDA 휠 문제) 구축, MusicCaps(Real 150개) + FakeMusicCaps로 1차 학습(val EER 4.67%), 제출 → **0.6722로 하락** |
| 09-22 | mtg-jamendo로 Real 대폭 확대, 학습 균형화(2차, 5.33%), `CosineAnnealingLR`↔조기 종료 충돌 발견 → `ReduceLROnPlateau`, patience 6→10 근거 산정, 원샷 파이프라인(v2), `script.py` CLI 인자 무시 버그 수정 |
| 09-23~24 | 3차 재학습 (학습 5,100 / 검증 900): epoch 31, **val EER 3.11%** |
| 09-28 | 3차 체크포인트 제출 → **0.70871, baseline 최초 돌파.** 같은 날 4차 재학습 착수 (weight decay 1e-3, patience 20) |
| 10-01 | 4차 재학습 종료 (87 epoch, 약 69시간) — **마감 이후라 제출하지 못함** |
| 대회 종료 | 최종 리더보드 순위 **480 중 308위** |
| 10-12 | 문서 수치를 로그·산출물과 대조하는 사후 점검 → 데이터 결함 3건 발견·기록 |

### 14.2 잘한 판단과 그 근거

- **교체 범위를 한 경로로 좁힘.** 평가식 가중치(Music 0.3 > Voice 0.2)와 "베이스라인에서 음악만 도메인 검증이 없다"는 사실로 `MUSIC_FAKE_PROB` 한 곳만 바꿨고, 나머지 컴포넌트는 그대로 두어 변화의 원인을 한 곳으로 한정할 수 있었다.
- **추론 때 모델이 보는 입력과 같은 분포로 학습.** 학습 데이터도 HTDemucs로 반주를 분리해 쓰고, 세그먼트 길이(64,600 샘플)·샘플링레이트·침묵 기준을 baseline `script.py`와 같은 상수로 고정했다.
- **하이퍼파라미터를 로그에서 근거를 찾아 정했다.** patience 6은 기본값일 뿐 근거가 없었다. 실행별 로그를 모아 "5 epoch 정체 후 개선된 사례가 반복된다"는 것을 확인하고 10으로 올렸다. 비용도 따졌다(epoch당 약 3분, patience를 올리는 비용은 작고 너무 일찍 멈추면 학습 전체를 잃는 비대칭).
- **스케줄러와 조기 종료의 구조적 충돌을 찾아냄.** `CosineAnnealingLR(T_max=100)`은 100 epoch을 다 돈다는 전제인데 실제로는 23 epoch에서 끝나 LR이 초기값 근처에 머물렀다. 정체 신호를 공유하는 `ReduceLROnPlateau`로 바꿨다.
- **제출 결과를 믿지 않고 패키지 내용을 직접 열어 검증.** zip 내부를 직접 열어 09-20 제출에 `music_classifier`가 없었음(순수 baseline)을, 09-20·09-21 패키지의 나머지 가중치 28개 파일이 byte-identical임을 확인했다. 점수 하락이 "재학습이 나빠서"가 아니라는 근거를 이렇게 얻었다.
- **조용한 버그를 잡음.** `script.py`가 `parse_arguments([])`로 모든 CLI 인자를 무시하고 있었고(수정 후 출력 경로를 바꾼 스모크 테스트로 확인), `config.py`의 `BASELINE_SUBMIT_DIR` 경로 계산이 틀려 있었으며, `export_for_submit.py`가 가리키던 문서가 실제로는 없었다.

### 14.3 실패와 한계 (사후 점검에서 확인)

| 문제 | 무슨 일이었나 | 배운 점 |
|------|--------------|---------|
| Fake가 단일 생성기 | 5개 생성기 폴더의 같은 파일명이 `{stem}.npy`로 충돌. 문서에는 "5종"이라 적었으나 실제 학습 데이터는 MusicGen_medium뿐 | **데이터를 눈으로(길이·개수·출처) 확인한 뒤 학습.** 요청한 `--limit 6000`보다 실제 고유 Fake 산출물(5,521개)이 적다는 신호가 있었다 |
| train/val 중복 | 서로 다른 행이 같은 `.npy`를 가리키는데 행 단위로 분할 → val 고유 Fake의 56%가 train에도 존재 | "트랙 단위 분할"은 설계 선언이 아니라 **고유 ID 기준으로 겹침을 직접 세어** 검증해야 한다 |
| 학습 중 val 교체 | `evaluate_manifest`가 매 epoch val.csv를 다시 읽는데, 학습 도중 manifest가 새로 만들어져(09-29 00:52, 병렬 `prepare_data.py` 실행으로 추정) epoch 18부터 다른 val로 평가됨 | **실행 중인 학습이 읽는 파일은 건드리지 않는다.** 실행마다 manifest를 스냅샷으로 저장하면 사고 후에도 복원할 수 있었다 |
| 변수 동시 변경 | 4차에서 데이터·weight decay·patience를 한꺼번에 바꿈 | 한 번에 하나씩 바꿔야 점수 변화를 해석할 수 있다 |
| 마감 이후 완료 | 약 69시간 걸린 학습이 마감 후 종료 | 긴 학습은 마감 역산으로 시작 시점을 정하거나 중간 체크포인트로 제출 후보를 확보한다 |

이 문제들을 알고도 **코드는 고치지 않고 기록만** 남겼다. 고치면 리더보드 점수를 만든 코드와 저장소 상태가 달라져 결과의 근거가 흐려지고, 제대로 고치려면 재분리·재학습이 필요한데 대회는 이미 끝났다.

### 14.4 다음에 한다면

1. 분리 결과 파일명에 생성기 폴더를 포함하고 FakeMusicCaps 5종 전체로 재분리
2. 원곡(MusicCaps ID) 단위 그룹 분할 — 같은 ID에서 파생된 Real·Fake가 갈라지지 않게
3. 생성기 hold-out 검증(한 생성기를 빼고 학습)으로 미지 생성기 일반화를 직접 측정
4. Real 도메인 효과와 Fake 데이터 효과를 분리하는 ablation, 변수는 한 번에 하나씩
5. 실행별 manifest 스냅샷과 학습률 로그 저장

### 14.5 저장소 사용 안내

- 학습 데이터, 체크포인트, 제출 zip, 대화 로그(`chat_log/`)는 용량·라이선스 문제로 git에서 제외되어 있다. 예외로 **리더보드 0.70871을 낸 가중치** `baseline_submit/model/music_classifier/weights.pt`(약 1MB, CRC32 `f6e9e676`, 3차 재학습 epoch 31)는 추적한다. 4차 재학습 체크포인트(`checkpoints/`)는 포함하지 않는다.
- 이 저장소의 코드는 **결과를 만든 상태 그대로**이며 §8의 "알려진 결함"을 포함한다. 새로 학습하려면 먼저 그 결함들을 고칠 것.
- **대회 제공물은 저장소에 포함하지 않는다.** 재배포 가능 여부를 확인하지 못했기 때문이다. 제외한 것: baseline `script.py`와 이를 그대로 옮긴 `patched_script.py`, `baseline_submit/`의 `requirements.txt`·`MODEL_INFO.txt`·`SHA256SUMS.txt`·`df_arena_1b/`·`htdemucs/`·`panns/`, 형식 확인용 `data/test/`·`data/sample_submission.csv`, 대회 규정 정리 노트 `deepvoice_competition_summary.md`. 로컬 작업 디렉토리에는 그대로 있고 `.gitignore`로 추적만 제외했다. 이 파일들은 대회 페이지의 baseline 패키지에서 받아 `baseline_submit/`에 풀어야 한다. 제출용 `script.py`는 내가 바꾼 부분만 `patched_script.diff`로 보관한다(§8 Step 5).
- 라이선스: FakeMusicCaps는 CC-BY-NC-4.0이며, 데이터셋·대회 제공물의 재사용·재배포 가능 여부는 최종 확인하지 않았다(`dev_log.md` 참고). README §1~3·§10에 정리한 대회 개요·평가식·제출 규격은 대회 페이지 내용을 요약한 것이다.
