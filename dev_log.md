---
type: Project
title: 딥보이스 탐지 대회 — 개발 일지
description: 대회 분석부터 제출까지의 작업 이력과 진행 계획
tags: [project, deepvoice, competition]
timestamp: 2026-09-20
status: active
---

# 딥보이스 탐지 대회 — 개발 일지

> **리더보드 제출 마감: 2026-09-29 (화) 10:00 KST** — 최신 갱신 시점(2026-09-22) 기준 **D-7**

---

## 진행 이력

### 2026-09-20

#### ✅ 대회 규정 및 환경 분석

- `deepvoice_competition_summary.md` 전체 정독
- 평가 산식 확인: `Score = 0.9 × ADS + 0.1 × CPS`
  - ADS 내 가중치: File(0.5) > Music(0.3) > Voice(0.2)
  - **Music EER이 전체 점수의 27%를 차지하며, baseline에서 유일하게 검증되지 않은 경로**
- 제출 규격 확인: zip 구조, 60분 추론 제한, 일일 3회 제출, 오류 유형별 차감 여부
- 평가 서버 사양 확인: NVIDIA L4 (22.4GiB), Python 3.11.15, CUDA 12.8

#### ✅ baseline 코드 전체 분석

- `baseline_submit.zip` 내부의 `script.py` 추출·정독
  - `script.py`는 zip 내부에만 존재 (프로젝트 디렉토리에는 미추출 상태)
  - 세그먼트 처리: 64,600 샘플(약 4.04초) 단위 슬라이딩, 세그먼트 MAX 집계
  - 침묵 처리: RMS < 1e-5이면 Fake 확률 0.0 반환
  - FILE_FAKE_PROB 결합: `max(VP × VF, MP × MF)`
  - HTDemucs: 파일 단위 z-score 정규화 → 분리 → 역정규화 (파일 내 통계만 사용, 규정 적합)
  - PANNs: voice 18개 레이블 / music 112개 레이블 (AudioSet 기반), 세그먼트 MAX
  - 오프라인 모드: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` 자동 설정
- demucs 4.0.1 vs 4.1.0 API 단절 (`load_track` 제거) 직접 확인

#### ✅ music_fake_pipeline 코드 전체 검토

- `music_fake_pipeline/music_fake_pipeline/` 내 전체 파일 검토
- `MusicFakeClassifier` 구조 확인
  - 입력: log-mel spectrogram `(1, 128, 253)` — 128 mel bins, 약 4.04초
  - 구조: ConvBlock × 4 (Conv2d-BN-ReLU-MaxPool2d) → GlobalAvgPool → Dropout → FC
  - 파라미터: 약 25만 (DF-Arena 1B 대비 1/4,000)
- 파이프라인 흐름 확인: prepare_data → train → evaluate → export → submit.zip 통합
- `patched_script.py` diff 확인: 음악 분기 한 곳만 교체, 나머지 baseline 동일
- 합성 데이터 end-to-end 검증 완료, 텐서 채널 차원 버그 수정 이력 확인

#### ✅ 데이터 전략 수립

- FakeMusicCaps (Zenodo 15063698) 분석: 생성기 5종, 27,605개 10초 클립, CC-BY-NC-4.0
  - Fake 샘플로 적합, **단독 사용 불가 — Real 샘플 별도 확보 필요**
  - FakeMusicCaps는 현재 다운로드 중
- Real 음악 후보: FMA (CC0/CC-BY), MusicCaps
- 추가 Fake 후보: Echoes (생성기 12종), SONICS
- 라이선스 최종 확인 미완료 (대회 규칙 탭 재확인 필요)

#### ✅ 문서 작업

- `README.md` 초안 작성 후 전면 업데이트
  - music_fake_pipeline 섹션 추가
  - 개발 워크플로우(Step 1~5) 추가
  - submit.zip 구조 업데이트
  - 검증 현황 섹션 추가
  - 지원 포맷 8개 명시, 세그먼트 처리·MAX Fusion·침묵 처리 상세 추가

### 2026-09-21

#### ✅ 실제 로컬 환경 구축 (문서 기준 환경과 차이 발생)

- 실제 개발 머신: Windows 11, Python 3.13.2 (시스템), GPU NVIDIA Quadro RTX 5000 (16GB), 드라이버 CUDA 13.2
- `pip install torch` 기본 실행 시 CPU 전용 빌드가 설치되는 문제 확인 → `torch==2.11.0+cu128` cu128 휠을 `download.pytorch.org/whl/cu128`에서 직접 설치해 해결
- `pyproject.toml` + `uv` 기반 환경으로 정리, `cuda_check.py` 추가
- **주의**: README 7장의 `Python 3.11` / `torch==2.7.1+cu128`은 평가 서버 기준이며, 로컬 개발 환경과는 다름 (상세 트러블슈팅은 `music_fake_pipeline/music_fake_pipeline/README.md` 9장에 날것으로 기록)

#### ✅ Real 음악 데이터 확보 방침 확정: FMA 대신 MusicCaps

- FakeMusicCaps가 원래 MusicCaps를 기반으로 생성되었다는 점에 착안, 가장 자연스러운 Real 짝으로 **MusicCaps**를 채택 (FMA 계획 폐기)
- `download_musiccaps.py` 작성 (yt-dlp 기반) → 경로·오류 수정을 거쳐 **Real 음악 150개 확보**
- FakeMusicCaps 다운로드 완료 — **Fake 55,216개** (생성기 5종: audioldm2, MusicGen_medium, musicldm, mustango, stable_audio_open)

#### ✅ 클래스 불균형 대응

- Real 150 : Fake 55,216은 극단적 불균형 → 이대로 학습하면 모델이 전부 "fake"로만 예측
- `train.py`에 class weight(`pos_weight`) 부재 확인 → Fake 쪽 서브샘플링(`--limit`)으로 1차 대응하기로 결정

#### ✅ 1차 학습 및 제출

- 소규모 데이터(Real 149 + Fake 150, 생성기 5종 균등 추출)로 `MusicFakeClassifier` 학습
- **Best val EER: 13.6%** (22 epoch, early stopping) — 단, 검증 세트 44개로 매우 작아 신뢰구간 넓음
- 제출 메모 작성 후 **첫 리더보드 제출** 진행 (`제출용_기본파일/0920_submit.zip`)
  - **리더보드 결과 (09-20)**: 총점 **0.6909110688** / ADS **0.6577539683** / CPS **0.9893249735** / 소요 시간 21분 41초
  - **[2026-09-22 사후 확인] 이 제출물은 사실 `music_classifier`가 반영되지 않은 순수 baseline이었다.** `0920_submit.zip`을 직접 열어보면 `model/music_classifier/` 폴더 자체가 없고 `script.py`도 baseline 원본과 동일한 로직(Music도 DF-Arena 1B 사용)이다. 즉 13.6% val EER짜리 1차 모델은 로컬에서만 검증됐고 실제로 제출되지는 않았다 — **09-20 리더보드 점수는 곧 "손대지 않은 baseline"의 실측 점수**다.
- 추가 데이터로 재학습한 2차 실행에서 **Best val EER 4.67%** (epoch 30, 기존 하이퍼파라미터 유지)까지 개선 확인, 이 체크포인트로 `music_classifier`를 처음으로 통합해 **두 번째 제출** 진행 (`제출용_기본파일/0921_submit.zip`)
  - **리더보드 결과 (09-21)**: 총점 **0.6722039259** / ADS **0.636968254** / CPS **0.9893249735** / 소요 시간 15분 51초
  - ⚠️ **09-20 대비 총점이 하락(0.6909 → 0.6722)**. `0920_submit.zip`과 `0921_submit.zip`의 `model/` 폴더를 CRC 단위로 직접 비교한 결과 **df_arena_1b·htdemucs·panns 가중치는 28개 파일 전부 byte-identical**했고, 유일한 차이는 `music_classifier/` 추가와 그에 맞춘 `script.py` 패치뿐이었다. CPS도 두 제출에서 완전히 동일(0.9893249735)해 PANNs 경로가 안 바뀌었다는 것과도 일치한다.
  - **결론(사후 원인 분석)**: 이 두 점수 차이는 "재학습이 오히려 나빠진 것"이 아니라, **"DF-Arena zero-shot을 이 프로젝트의 소형 CNN(`music_classifier`)으로 교체한 것 자체가 실제 대회 데이터 기준으로는 손해였다"**는 뜻이다. Voice EER은 두 제출에서 동일하므로 ADS 하락분(0.0208)은 전부 File EER·Music EER 쪽에서 나왔고, `FILE_FAKE_PROB = max(VP×VF, MP×MF)`이라 Music 쪽이 나빠지면 File도 같이 나빠질 수 있는 구조다. File/Music EER이 비슷한 폭으로 움직였다고 가정하면 각각 약 **+2.6%p** 악화된 셈이다(0.5x + 0.3x = 0.0208 → x ≈ 0.026).
  - **원인 추정**: `music_classifier`는 Real 150개(MusicCaps)·Fake 5개 생성기(FakeMusicCaps)라는 매우 좁은 데이터로만 학습됐다. 로컬 val EER이 크게 개선된 것은 **학습 데이터와 같은 좁은 분포를 공유하는 val 44개 세트에 대한 적합**이었을 뿐, 실제 대회 데이터(더 다양한 생성기·장르·녹음 환경)에 대한 일반화는 오히려 DF-Arena 1B(비록 음악 도메인 미검증이지만 훨씬 크고 다양하게 사전학습된 모델) zero-shot보다 못한 것으로 보인다. 이는 §4(프로젝트 전략)의 핵심 전제 — "Music 경로 교체가 이득일 것" — 가 아직 실증되지 않았다는 뜻이므로 최우선으로 재검토가 필요하다.

#### 🔄 Real 데이터 추가 확보 시도: mtg-jamendo-dataset (미해결)

- Real 150개로는 여전히 부족 판단 → MTG 공식 저장소 `mtg-jamendo-dataset`을 클론하여 추가 Real 음악 소스로 검토
- 저장소 내 `venv`(Python 3.12)에서 `pip install` 시 `setuptools`/`distutils` 빌드 실패 (sdist 빌드 단계에서 에러)
- 원인: venv의 `setuptools`가 너무 오래되어 `distutils` 대체 역할을 못함
- 해결책 제시: `mtg-jamendo-dataset/venv/Scripts/pip install --upgrade setuptools` 후 재시도 — **아직 실행 여부 미확인, 다음 세션에서 이어서 처리 필요**

---

### 2026-09-22

#### ✅ 하이퍼파라미터 재조정 및 재학습

- `config.py` 변경: `BATCH_SIZE 32→16`, `NUM_EPOCHS 30→100`, `LEARNING_RATE 3e-4→1e-4`, `WEIGHT_DECAY 1e-4→1e-5`
- manifest 재생성 (`train.csv`/`val.csv`, 18:11) 후 재학습 실행
- 학습 로그(`logs/train_log.csv`) 기준 **epoch 9에서 val EER 7.0%**까지 확인된 이후 10 epoch 시점 기록에서 로그가 멈춰 있음 — `EARLY_STOP_PATIENCE=6` 조건(6 epoch 연속 미개선)을 아직 채우지 못한 상태라 **정상 종료인지 중단된 것인지 다음 세션에서 확인 필요**
- 현재 `checkpoints/music_classifier_best.pt`·`music_classifier_last.pt`는 이 실행 기준 (2026-09-22 19:12~19:15 갱신)

#### ✅ `.gitignore` 정비

- `venv/`(최상위, `.venv/`와 별개), `mtg-jamendo-dataset/`(서드파티 클론), `제출용_기본파일/`(제출 zip 백업 폴더, 각 파일 최대 4.5GB) 3개 항목 추가
- 해당 zip들이 GitHub 100MB 제한을 초과해 그대로 두면 push가 실패하는 상태였음

---

## 현재 상태

| 항목 | 상태 | 비고 |
|------|------|------|
| 대회 규정 분석 | ✅ 완료 | |
| baseline 코드 분석 | ✅ 완료 | script.py는 zip에서 별도 추출 필요 |
| music_fake_pipeline 구현 | ✅ 완료 (실제 데이터 학습까지 진행) | |
| README.md | ✅ 완료 (로컬 환경은 pipeline README 9장 참고) | |
| **FakeMusicCaps (Fake 샘플)** | ✅ 확보 완료 | 55,216개, 생성기 5종 |
| **Real 음악 데이터** | ✅ 확보 완료 (MusicCaps) | FMA 대신 MusicCaps 채택, 150개 |
| **Real 음악 추가 확보 (mtg-jamendo-dataset)** | 🔄 진행 중 (블로킹 이슈 있음) | venv `setuptools` 빌드 오류, 해결책 제시했으나 실행 미확인 |
| HTDemucs 분리 (prepare_data) | ✅ 완료 | manifest 재생성까지 반영 |
| 모델 학습 (train) | 🔄 진행 중 | 최신 실행 best val EER 7.0%(epoch 9), 정상 종료 여부 미확인 |
| 평가 (evaluate) | ⚠️ 부분 완료 | val EER 기준 확인, 리더보드 실측 결과는 아직 없음 |
| submit.zip 통합 및 제출 | ✅ 2회 제출 완료 (09-20 baseline 그대로 0.6909, 09-21 `music_classifier` 첫 적용 0.6722) | **원인 확인 완료**: `music_classifier`가 DF-Arena zero-shot보다 나쁨(원인: 좁은 학습 데이터 추정). 데이터 확대 없이 09-22 체크포인트를 그대로 제출하는 것은 비권장 |

---

## 앞으로의 작업 계획

### 우선순위 순서

> **09-22 진단 결과 반영**: `music_classifier`가 DF-Arena zero-shot보다 실제 리더보드에서 더 나쁘다는 것이 확인됐다(위 09-21 항목 참고). 원인으로 유력한 "학습 데이터가 너무 좁음(Real 150 · Fake 5종 생성기)"을 해결하기 전까지는, 09-22에 새 하이퍼파라미터로 재학습한 체크포인트를 그대로 제출해도 같은 문제가 반복될 가능성이 높다. **데이터 다양성 확보를 재제출보다 먼저 처리한다.**

#### 1단계 — 데이터 다양성 확보 (최우선, 재제출 전에 선행)

- [ ] `mtg-jamendo-dataset/venv`의 `setuptools` 업그레이드 후 설치 재시도 → Real 데이터 추가 확보
  ```bash
  mtg-jamendo-dataset/venv/Scripts/pip install --upgrade setuptools
  ```
- [ ] Real 음악을 MusicCaps 150개보다 훨씬 다양한 규모로 확대 (mtg-jamendo 등) — 장르·언어·녹음 환경 다양성이 핵심
- [ ] Fake 음악도 FakeMusicCaps 5개 생성기 외 추가 확보 검토 (Echoes 등 — 생성기 다양성 확대가 목적, 단순 개수 증가는 아님)
- [ ] 각 데이터셋 라이선스 조항 최종 확인 (FakeMusicCaps CC-BY-NC-4.0 등, 대회 "규칙" 탭 기준) — 아직 미완료

#### 2단계 — 재학습 및 재제출 판단

- [ ] `logs/train_log.csv`/프로세스 상태 확인 — 09-22 실행(epoch 9 val EER 7.0%)이 정상 종료됐는지 확인
- [ ] 확장된 데이터로 재학습 후 `evaluate.py`로 val EER 확인
  ```bash
  python evaluate.py --checkpoint checkpoints/music_classifier_best.pt --save-csv logs/val_predictions.csv
  ```
- [ ] **val EER 개선만으로 제출을 결정하지 않는다** — 09-21의 교훈(로컬 개선이 리더보드 악화로 이어짐)을 감안해, 데이터 다양성이 실질적으로 늘었는지를 먼저 확인
- [ ] 데이터 확대 없이 하이퍼파라미터만 바꾼 09-22 체크포인트는 **09-20 baseline 점수(0.6909)를 넘지 못할 가능성이 높으므로 그대로 제출하지 않는 것을 권장** — 일일 3회 제한 중 09-20/09-21에 이미 1회씩 사용했으므로 남은 제출 기회를 신중히 사용
- [ ] `export_for_submit.py` → `submit.zip` 재통합 → 세 번째 제출은 데이터 확대·재학습 이후로 미룸

#### 3단계 — 개선 (시간 여유 시)

- [ ] 하이퍼파라미터 추가 튜닝 (현재 BATCH_SIZE=16, LR=1e-4, WEIGHT_DECAY=1e-5, NUM_EPOCHS=100 기준) — 단, 데이터 확대가 우선
- [ ] 추론 시간 실측 후 여유 있으면 더 큰 모델 실험
- [ ] (최후 수단) 데이터 확대로도 DF-Arena를 못 넘으면, Music 경로를 baseline(DF-Arena)으로 되돌리는 것도 옵션으로 열어둘 것

---

## 주요 결정 사항 기록

| 날짜 | 결정 | 근거 |
|------|------|------|
| 2026-09-20 | MUSIC_FAKE_PROB 경로만 교체 | Music EER 가중치(0.3) > Voice(0.2), baseline의 유일한 미검증 경로 |
| 2026-09-20 | 소형 CNN 선택 (25만 파라미터) | 60분 추론 제한 내 DF-Arena보다 연산량 감소 목표 |
| 2026-09-20 | train/val 분할은 트랙 단위 | 세그먼트 단위 분할 시 data leakage 발생 |
| 2026-09-20 | demucs==4.0.1 고정 | 4.1.0에서 load_track API 제거 — 직접 재현 확인 |
| 2026-09-21 | Real 데이터는 FMA 대신 MusicCaps 채택 | FakeMusicCaps가 원래 MusicCaps 기반으로 생성됨 — 가장 자연스러운 Real/Fake 짝 |
| 2026-09-21 | Fake 데이터는 우선 서브샘플링으로 균형 맞춤 | Real 150 : Fake 55,216 극단적 불균형, `train.py`에 pos_weight 미구현 상태였음 |
| 2026-09-22 | 하이퍼파라미터 변경 (batch 16 / lr 1e-4 / wd 1e-5 / epoch 100) | 이전 설정(batch 32 / lr 3e-4) 대비 val EER 추가 개선 시도 |

---

## 미해결 사항 및 리스크

| 항목 | 내용 | 영향도 |
|------|------|--------|
| mtg-jamendo-dataset 설치 실패 | venv(Python 3.12)의 setuptools가 오래되어 distutils 대체 불가, pip install 시 빌드 오류 — 해결책만 제시, 실행 미확인 | 중간 (Real 데이터 확대 지연) |
| 라이선스 미확인 | FakeMusicCaps(CC-BY-NC-4.0) 등 대회 적용 가능 여부 불명확 — 대회 규칙 탭 재확인 필요 | 높음 |
| 최신 학습 실행 종료 상태 불명확 | 2026-09-22 실행이 epoch 10에서 기록이 멈춤 — early stop 조건 미충족, 정상 종료/중단 여부 미확인 | 중간 |
| **`music_classifier`가 DF-Arena zero-shot보다 실제로 더 나쁨** | 09-20(순수 baseline) 총점 0.6909 vs 09-21(`music_classifier` 첫 적용) 총점 0.6722. zip 내부 CRC 비교로 두 제출의 유일한 차이가 music 분기뿐임을 확인함 — 이제 "원인 미상"이 아니라 **"현재 버전의 자체 학습 모델이 baseline보다 열등하다"는 확정된 사실**. Real 150개/Fake 5종 생성기라는 좁은 학습 데이터의 과적합·분포 불일치가 유력한 원인. 프로젝트의 핵심 전제(§4) 자체를 재검토해야 함 | 매우 높음 |
| 검증 세트 규모가 작음 | 초기 실행 기준 val 44개 — EER 수치의 신뢰구간이 넓어 실제 리더보드 성능과 괴리 가능 | 중간 |
| 평가 서버 추론 시간 미확인 | music_classifier 추가 후 60분 제한 내 완료 여부 | 낮음 (모델이 매우 가벼움) |
