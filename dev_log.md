---
type: Project
title: 딥보이스 탐지 대회 — 개발 일지
description: 대회 분석부터 제출, 마지막 재학습, 종료 후 사후 점검까지의 작업 이력과 최종 상태
tags: [project, deepvoice, competition]
timestamp: 2026-10-12
status: completed
---

# 딥보이스 탐지 대회 — 개발 일지

> **대회 종료 (리더보드 제출 마감 2026-09-29 10:00 KST / 대회 종료 2026-09-30 10:00 KST).** 이 문서는 2026-10-12에 마지막 상태까지 갱신하고 마무리함.

## 최종 요약 (2026-10-12 기준)

| 항목 | 내용 |
|------|------|
| 한 일 | 베이스라인(DF-Arena 1B)의 `MUSIC_FAKE_PROB` 경로만 소형 CNN(`MusicFakeClassifier`, 약 25만 파라미터)으로 교체하고, 데이터 수집→HTDemucs 분리→학습→평가→제출 패키징까지 파이프라인화 |
| 공개 리더보드 최고 점수 | **총점 0.70871** (ADS 0.67753 / CPS 0.98932) — 순수 baseline 0.6909 대비 **+0.0178**. 확인일 2026-09-28 |
| 이 점수를 만든 체크포인트 | 3차 재학습 (`logs/20260923_134529.csv`): epoch 31, 로컬 val EER 3.11%, 학습 5,100 / 검증 900 |
| 마지막 재학습 (87 epoch, val EER 0.75%) | **마감 이후(10-01 11:13)에 끝나 export·제출되지 않음.** 리더보드 검증 없음 |
| 종료 후 사후 점검에서 발견한 한계 | ① Fake 학습 데이터가 사실상 MusicGen_medium **단일 생성기**(파일명 충돌) ② manifest 내 중복으로 train/val 일부 겹침 ③ 마지막 학습 도중 val 세트가 바뀜 → 마지막 학습의 val EER은 신뢰 불가. 상세는 [§2026-10-12 사후 점검](#2026-10-12--마무리-사후-점검) |
| 최종 순위 | **480 중 308위** (사용자 제공, 2026-10-12 기록. 공개/Private 구분은 제공받지 못함) |
| 기록에 없는 것 | 2차 평가(보고서) 참여 여부, 09-22 제출 2건의 점수. 1차 평가 선정 기준이 상위 15팀이라 이 순위로는 2차 평가 대상이 아닌 것으로 보이나 확인하지 않음 |

> **브랜치 안내**: `v1` = 2026-09-22 파이프라인 자동화 이전까지의 모든 내용(수동으로 `prepare_data.py`→`train.py`→`evaluate.py`→`export_for_submit.py`를 하나씩 실행하던 시절, `CosineAnnealingLR`+`patience=6`). `v2` = 학습 진행 게이지, 실행별 로그 분리(`run_logger.py`), 원샷 파이프라인(`run_pipeline.py`), `ReduceLROnPlateau`+`patience=10`, `prepare_data.py` 증분 처리, `config.py` 경로 버그 수정까지 반영된 현재 버전.

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
- 해결책 제시: `mtg-jamendo-dataset/venv/Scripts/pip install --upgrade setuptools` 후 재시도

---

### 2026-09-22

#### ✅ mtg-jamendo-dataset 연동 성공 — Real 데이터 대폭 확대

- 전날의 `setuptools` 빌드 오류가 해결되어 mtg-jamendo 트랙 다운로드가 정상 진행됨
- `data/raw/real_music/` 원본 파일 수가 **150개 → 18,636개**로 증가 (MusicCaps 150개 + Jamendo 트랙 다수, 파일명이 Jamendo 특유의 숫자 트랙ID `{id}.low.mp3` 형태)
- 이 중 HTDemucs 분리(`data/separated/real/`)까지 끝낸 것은 2,016개 (Jamendo 유래 1,867개 + MusicCaps 유래 149개) — 나머지 원본은 아직 미처리 상태로 남아 있음
- **이것으로 프로젝트의 최우선 리스크였던 "Real 데이터가 150개뿐이라 너무 좁다"는 문제가 실질적으로 해결 국면에 들어감**

#### ✅ 하이퍼파라미터 재조정 및 확대된 데이터로 재학습 완료 (18:44 시작 → 약 19:57 조기 종료)

- `config.py` 변경: `BATCH_SIZE 32→16`, `NUM_EPOCHS 30→100`, `LEARNING_RATE 3e-4→1e-4`, `WEIGHT_DECAY 1e-4→1e-5`
- manifest 재생성(18:11) — **Real(label 0) 1,700개 : Fake(label 1) 1,700개로 완전히 균형 맞춤** (기존 150 : 55,216의 극단적 불균형에서 개선). Fake는 FakeMusicCaps 55,216개 중 1,700개 서브샘플링
- WSL에서 `python train.py --device cuda`로 재학습 실행 (PID 22133, 처음부터 새로 학습·`--resume` 아님). 옵티마이저 AdamW + `CosineAnnealingLR`(T_max=100), `BCEWithLogitsLoss`(데이터가 이미 균형 잡혀 `pos_weight` 불필요)
- GPU(Quadro RTX 5000) VRAM은 학습 중 15.8/16GB로 거의 가득 찼는데 GPU 사용률은 1%에 불과 — dataloader worker 4개가 각 13%대 CPU를 쓰는 것으로 보아 **병목이 GPU 연산이 아니라 CPU 쪽 오디오 전처리(멜스펙트로그램 계산)에 있는 것으로 추정**. 다음 개선 여지로 기록
- **최종 결과**: epoch 17에서 **best val EER 5.33%** 달성 후 6 epoch 연속 개선 없어 `EARLY_STOP_PATIENCE=6` 조건으로 **epoch 23에서 정상 조기 종료** (`train.py`의 `if epochs_without_improvement >= args.patience: break` 로직 확인, 크래시 아님)
- `checkpoints/music_classifier_best.pt`(epoch 17 기준, 19:37 저장) / `checkpoints/music_classifier_last.pt`(epoch 23 기준, 19:56 저장) 갱신 완료, 프로세스는 정상 종료되어 GPU 메모리 해제됨(1GB 미만으로 복귀)

#### ✅ `.gitignore` 정비

- `venv/`(최상위, `.venv/`와 별개), `mtg-jamendo-dataset/`(서드파티 클론), `제출용_기본파일/`(제출 zip 백업 폴더, 각 파일 최대 4.5GB) 3개 항목 추가
- 해당 zip들이 GitHub 100MB 제한을 초과해 그대로 두면 push가 실패하는 상태였음

#### ✅ 세 번째 제출 패키지 구성 — 치명적 버그 발견·수정

- `export_for_submit.py`로 오늘 학습한 best 체크포인트(epoch 17, val EER 5.33%) export → `baseline_submit/model/music_classifier/`에 반영 (CRC32로 이전 09-21 체크포인트와 다른 파일임을 확인: `0xbaaa06f5` → `0x7901d53e`)
- **버그 발견**: `script.py`의 `main()`이 `args = parse_arguments([])`로 빈 리스트를 넘겨 `--test-dir`/`--sample-submission`/`--output`/`--device` 등 **모든 CLI 인자를 무시**하고 하드코딩된 기본값만 쓰고 있었음. `baseline_submit/script.py`와 원본 템플릿 `patched_script.py` 양쪽 다 `parse_arguments()`로 수정
- 더미 3개 파일로 스모크 테스트 — 수정 후 CLI 인자 정상 반영, 출력 컬럼도 `sample_submission.csv`와 일치함을 확인 (exit code 0)
- `export_for_submit.py:247`이 안내하던 `SUBMIT_INTEGRATION.md`가 실제로는 한 번도 만들어진 적 없는 "유령 파일"이었음을 발견 → 루트 `README.md` §8 Step 5 절차를 옮겨 `music_fake_pipeline/music_fake_pipeline/SUBMIT_INTEGRATION.md`로 신규 작성, 파이프라인 README §6도 이를 가리키도록 갱신

#### ✅ 세 번째 제출 메모 (제출 시점 기록용)

> [MUSIC_FAKE_PROB 경로 교체 제출]
>
> 베이스라인은 음성 딥페이크 탐지용 DF-Arena 1B를 음악 판별에도 그대로 씀. Music EER 가중치(0.3)가 Voice(0.2)보다 크고 baseline에서 유일하게 도메인 검증 안 된 지점이라 이 경로만 교체함. 나머지(PANNs, HTDemucs, Voice의 DF-Arena, MAX Fusion)는 baseline과 동일함.
>
> 모델(MusicFakeClassifier): log-mel spectrogram 입력, Conv-BN-ReLU-MaxPool 4블록+GAP+FC, 약 25만 파라미터.
>
> 데이터: 총 4,000개(Real 2,000+Fake 2,000) 중 학습셋 3,400개(Real 1,700:Fake 1,700), 검증셋 600개(Real 300:Fake 300)로 완전 균형. Real은 MusicCaps+mtg-jamendo, Fake는 FakeMusicCaps 5개 생성기 중 서브샘플링.
>
> 학습 조건: 배치 16, 학습률 1e-4(AdamW+CosineAnnealingLR), 최대 100 epoch, patience 6. epoch 17에서 best val EER 5.33%(train loss 0.1342) 달성, 이후 6 epoch 미개선으로 epoch 23 조기 종료(최종 train loss 0.1061).
>
> 참고: 직전 제출(Real 150개만 사용, val EER 4.67%)은 리더보드에서 baseline(0.6909) 대비 낮은 점수(0.6722)를 기록함. 데이터 다양성 확대가 이 격차를 해소하는지 확인하는 것이 이번 제출 목적임.

- 리더보드 결과는 다음 세션에서 확인 후 이 문서에 추가할 것 (§"09-21 항목"과 동일한 형식으로 기록) → **[10-12 확인] 끝내 기록되지 않았음.** 이 메모의 패키지는 `0922_2_submit.zip`(가중치 CRC32 `7901d53e`)이며 점수는 어디에도 남아 있지 않음 ("마지막 재학습" 항목의 제출 이력 표 참고)

#### ✅ patience(조기 종료) 값의 근거 점검 및 LR 스케줄러 수정

- 사용자 질문("patience=6은 어떤 기준으로 정해졌나")을 계기로 조사 — `EARLY_STOP_PATIENCE=6`은 09-20에 `config.py`를 처음 만들 때부터의 기본값일 뿐, 실험이나 분석으로 도출된 숫자가 아님을 확인 (근거 문서/커밋 없음)
- 조사 중 **`CosineAnnealingLR`과 early stopping의 구조적 불일치**를 발견: 스케줄러가 `T_max=args.epochs`(=config.NUM_EPOCHS=100)로 "100 epoch를 다 돈다"는 전제로 LR을 서서히 낮추도록 짜여 있는데, patience=6 조기 종료 때문에 실제로는 epoch 23(오늘 실행 기준, 전체의 23%)에서 멈춰버려 **LR이 초기값(1e-4) 근처에서 거의 안 내려간 채로 학습이 끝나는 상태**였음
- **수정**: `train.py`의 스케줄러를 `CosineAnnealingLR` → `ReduceLROnPlateau(mode="min", factor=0.5, patience=max(1, args.patience//2))`로 교체. Early stopping과 똑같이 val EER 정체를 신호로 삼으므로, 학습이 언제 멈추든 "정체되면 LR을 절반으로 낮춘다"는 동작이 항상 보장됨. `scheduler.step()` → `scheduler.step(val_eer)`로 변경, 매 epoch 로그에 현재 LR도 출력하도록 추가(`train_log.csv` 스키마는 기존 4컬럼 유지, LR은 콘솔 출력에만 추가)
- **참고**: 지금까지의 체크포인트(09-21 val EER 4.67%, 09-22 val EER 5.33%)는 모두 옛 `CosineAnnealingLR` 방식으로 학습된 것 — 이 수정 이후 재학습하면 다른(잠재적으로 더 나은) 결과가 나올 수 있음.

#### ✅ patience 값을 실측 데이터 기반으로 6 → 10으로 재산정

- `train_log.csv`에 남아있는 4번의 학습 실행 기록을 분석 — "val EER이 최고 기록을 갱신하기까지 실제로 몇 epoch을 기다려야 했는지"를 직접 집계함
- **핵심 증거(Run 2, 30 epoch를 조기종료 없이 끝까지 완주한 유일한 런)**: 개선 없이 **5 epoch 연속 정체됐다가 6번째 epoch에 다시 개선되는 패턴이 2번 반복 관찰됨**, 최종 best(val EER 4.67%)도 마지막 epoch(30)에 갱신됨. Run 4(오늘 실행)에서도 동일하게 5 epoch 정체 후 개선되는 사례 확인
- **결론**: `patience=5`였다면 지금 가진 최고 성능 체크포인트들을 애초에 발견하지 못하고 더 일찍 멈췄을 것 — `patience=6`은 최소 하한선이지 여유값이 아니었음
- **비용-이득 비교**: epoch당 약 3분. patience를 늘리는 비용(6→10 시 최대 +4 epoch ≈ 12분)은 작은데, patience가 부족해 조기종료되면 학습 런 전체를 날리는 손실이 훨씬 큼 — 비대칭 구조상 낮추기보다 올리는 쪽이 안전
- **결정**: `config.py`의 `EARLY_STOP_PATIENCE`를 6 → **10**으로 상향 (관측된 최대 정체 구간 5의 2배 여유). 연동되어 있는 `ReduceLROnPlateau`의 patience(`args.patience//2`)도 자동으로 3→5로 조정됨

#### ✅ 파이프라인 자동화 (v1 → v2) — 사용자 요청 3건 일괄 반영

> **브랜치 안내**: 여기서부터의 변경은 `v2` 브랜치에 있음. 그 이전까지의 모든 내용(§09-20~09-22 위 항목 전체, `CosineAnnealingLR`+patience=6 시절의 체크포인트 등)은 `v1` 브랜치에 고정되어 있음.

일일이 스크립트를 하나씩 확인하며 실행하기 번거롭다는 문제 제기에 따라 3가지를 구현:

1. **`run_logger.py` (신규)** — 학습 1회 실행마다 `logs/{YYYYMMDD_HHMMSS}.csv`로 별도 로그 파일 생성(epoch, train_loss, val_eer, elapsed_sec). 기존에는 `logs/train_log.csv` 하나에 모든 실행이 계속 이어 붙어서, 이번 대화에서 patience 근거를 분석할 때도 "epoch 번호가 1로 리셋되는 지점"을 보고 실행 경계를 추측해야 했음 — 이제는 파일명 자체로 실행이 구분됨.
2. **`train.py` 학습 진행 게이지** — `evaluate.py`의 "Evaluating" tqdm 게이지처럼, `train_one_epoch()`의 배치 루프에도 tqdm을 추가해 현재 epoch 진행률과 실시간 평균 loss를 보여줌.
3. **`run_pipeline.py` (신규)** — `prepare_data.py` → `train.py` → `evaluate.py` → `export_for_submit.py`를 기본값으로 순서대로 실행하는 원샷 스크립트. `sys.executable`을 그대로 하위 프로세스에 넘겨써서, 이 스크립트만 올바른 venv(프로젝트 루트 `.venv`, torch 포함)의 python으로 실행하면 나머지 단계도 같은 환경을 씀. torch 미설치 시 명확한 안내 메시지 출력, 한 단계라도 실패하면 즉시 중단.

**원샷 파이프라인을 실제로 굴리면서 드러난 두 가지 기존 버그도 함께 수정**:

- `prepare_data.py`가 재실행 시 **이미 HTDemucs 분리가 끝난 파일까지 매번 처음부터 다시 처리**하고 있었음 (skip 로직 없음). Real 원본이 18,636개까지 늘었는데 이 중 2,016개만 분리된 상태라, 나머지를 마저 처리하려면 이 비효율이 치명적 — `out_path.exists()`면 재처리 없이 manifest에만 포함하도록 수정
- 사용자가 `run_pipeline.py`를 직접 실행하다 `FileNotFoundError: HTDemucs 모델 경로를 찾을 수 없습니다`를 마주침 — 원인은 `config.py`의 `BASELINE_SUBMIT_DIR = PROJECT_ROOT.parent / "baseline_submit"`가 애초에 잘못 계산되고 있었던 것(`baseline_submit/`은 `music_fake_pipeline/`의 형제가 아니라 `Deep_voice/`(한 단계 더 위) 바로 아래에 있음). 지금까지는 모든 문서·실행 예시가 `--htdemucs-dir`을 매번 명시적으로 지정해서 이 기본값 버그가 드러나지 않았음 — `PROJECT_ROOT.parent.parent`로 수정

---

### 2026-09-23 ~ 2026-09-24

#### ✅ `ReduceLROnPlateau` + `patience=10` 설정으로 재학습 (3차, 데이터 추가 확대)

- 사용자가 직접 `train.py --device cuda`를 실행(13:45:29 시작, 로그 `logs/20260923_134529.csv`) — v2의 `ReduceLROnPlateau`/`patience=10` 반영 이후 첫 재학습
- 학습 직전 manifest가 재생성되어(13:42) 데이터가 다시 한 번 확대됨: **학습셋 5,100개(Real 2,550:Fake 2,550), 검증셋 900개(Real 450:Fake 450)** — 2차(3,400/600) 대비 약 1.5배
- 배치 16 / 학습률 1e-4 / **weight decay 1e-5**(당시 커밋 상태) / 최대 100 epoch
- **epoch 31에서 best val EER 3.11%(train loss 0.0607) 달성**, 이후 10 epoch(patience) 연속 미개선으로 **epoch 41에서 정상 조기 종료**(최종 train loss 0.0457) — 03:48:07(09-24) 종료, `checkpoints/music_classifier_best.pt`/`last.pt` 갱신
- 2차(val EER 5.33%, epoch 23) 대비 로컬 val EER이 5.33%→3.11%로 개선 — `ReduceLROnPlateau`+patience 상향+데이터 확대가 함께 작용한 결과로 추정, 개별 기여도는 분리 확인 안 됨

#### ✅ 제출 패키지 갱신 (3차 재학습 체크포인트 반영)

- 위 체크포인트(epoch 31, val EER 3.11%)를 `export_for_submit.py`로 export(03:56:22, 09-24) → `baseline_submit/model/music_classifier/` 갱신
- 이전 제출들과 동일한 절차(§8 Step 5)로 스모크 테스트 후 제출

---

### 2026-09-28

#### ✅ 이번 제출 리더보드 결과 확인 — 처음으로 baseline 돌파

- **총점 0.70871 / ADS 0.67753 / CPS 0.98932**
- 지금까지의 전체 제출 비교:

  | 제출 | 총점 | ADS | CPS | 비고 |
  |------|------|-----|-----|------|
  | 09-20 (순수 baseline, Music도 DF-Arena) | 0.6909110688 | 0.6577539683 | 0.9893249735 | — |
  | 09-21 (`music_classifier` 1차, Real 150개) | 0.6722039259 | 0.636968254 | 0.9893249735 | baseline보다 낮음 |
  | 09-28 확인 (`music_classifier` 3차 체크포인트, 학습 5,100/검증 900) | **0.70871** | **0.67753** | 0.98932 | **baseline·09-21 모두 상회** |

- CPS가 세 제출에서 거의 동일(0.9893 → 0.98932)한 것으로 보아, 이번 개선분은 전부 File/Music EER 쪽에서 나온 것 — PANNs 존재 판별 경로는 그대로이므로 예상과 일치
- **결론**: "Music 경로를 소형 CNN으로 교체한다"는 §4의 핵심 전략 가설이, 09-21에는 데이터가 너무 좁아 기각된 것처럼 보였지만 **데이터를 충분히 확대하자(학습 150→5,100) 실제로 유효한 것으로 확인됨**. 미해결 사항 표의 "music_classifier가 DF-Arena zero-shot보다 나쁨" 리스크는 이것으로 해소됨

#### ✅ `WEIGHT_DECAY` 1e-3 변경 보류 결정

- 다음 실험 후보로 `config.py`의 `WEIGHT_DECAY`를 1e-5 → 1e-3으로 워킹 트리에서 수정해뒀으나, 아직 이 값으로 학습을 돌린 적은 없음(위 3차 재학습은 1e-5로 진행됨)
- 사용자 확인 결과 **현재는 커밋하지 않고 그대로 보류** — 다음 재학습에서 실제로 시도할지는 추후 결정
- **[후속, 2026-10-12 확인] 이 보류는 같은 날 번복됨.** `config.py`가 11:27에 다시 수정되어 `WEIGHT_DECAY=1e-3`에 더해 `EARLY_STOP_PATIENCE`도 10→20으로 바뀐 상태로 14:00 재학습이 시작됐고, 20:24에 `2bfd41e`로 커밋됨. 번복 경위를 담은 대화 기록은 없음. 상세는 아래 "마지막 재학습" 항목.

#### 🔄 다음 라운드 개선 방향 논의 및 실행 착수

- 이번 제출 성공을 바탕으로 다음 재학습에서 시도해볼 항목을 우선순위로 논의함:
  1. **Fake 데이터 볼륨/다양성 확장** (최우선) — 확인 결과 HTDemucs 분리 완료분이 Real 6,884개 vs Fake 3,000개로 불균형해서, 균형을 맞추려 Real도 3,000개로 깎아 쓰고 있었음. Fake 분리를 늘리면 이미 확보된 Real 6,884개를 그대로 다 쓸 수 있어 총 데이터 규모가 거의 2배로 늘어남
  2. 데이터 증강 (SpecAugment 등, `dataset.py`에 현재 random-crop 외 augmentation 없음) — FakeMusicCaps가 생성기 5종뿐이라 미지 생성기에 대한 일반화력 강화 목적
  3. `WEIGHT_DECAY` 1e-3 실험 (가장 저비용, 위 보류 항목)
- **사용자가 1번 방향으로 `python run_pipeline.py --limit 6000` 실행 착수** (클래스별 최대 6,000개까지 raw→HTDemucs 분리 확대 → 재학습 → 평가 → export까지 원샷). 이 문서를 처음 갱신한 시점(09-28 오전)에는 `data/manifests/`, `checkpoints/`, `logs/`에 새 결과물이 반영되기 전이었음 — **결과는 아래 "마지막 재학습" 항목에 기록**

---

### 2026-09-28 (오후) ~ 2026-10-01 — 마지막 재학습 (4차, `run_pipeline.py --limit 6000`)

> 이 구간은 `chat_log/`에 대화 기록이 없고, 아래는 전부 `logs/`·`checkpoints/`·`data/` 파일의 내용과 수정 시각(mtime)에서 복원한 것임.

#### ✅ 학습 조건 (실측)

- 실행: WSL, 로그 `logs/20260928_140015.csv` (`RunLogger`가 만들어진 09-28 14:00:15 = 학습 루프 시작 시각)
- `config.py` 값 그대로(09-28 11:27 수정 → 20:24 `2bfd41e`로 커밋, 학습 시작 전이므로 이 값이 적용됨): 배치 16 / 학습률 1e-4 / **weight decay 1e-3**(3차는 1e-5) / **patience 20**(3차는 10) / 최대 100 epoch / AdamW + `ReduceLROnPlateau`(factor 0.5, patience = 20//2 = 10) / `BCEWithLogitsLoss`
- 3차 대비 **데이터·weight decay·patience 세 가지가 동시에 바뀜** → 결과 차이를 어느 한 요인으로 귀속할 수 없음

#### ✅ 학습 결과 (실측)

| 항목 | 값 |
|------|-----|
| 총 epoch | **87** (최대 100, patience 20 소진으로 조기 종료) |
| best | **epoch 67 — val EER 0.0075 (0.75%), train loss 0.0219** (epoch 75도 0.0075였으나 엄격한 `<` 개선이 아니라 best로 갱신되지 않음) |
| 종료 시점 | epoch 87 — val EER 0.0083, train loss 0.0166 |
| 시간 | 로그상 epoch 합계 69.2시간 (09-28 14:00 → 10-01 11:13). epoch 47 한 번이 49,292초(13.7시간)였으며 원인 기록 없음 — 제외하면 55.5시간, 평균 2,324초/epoch (3차 평균은 1,232초) |
| 체크포인트 | `music_classifier_best.pt` mtime 09-30 22:23 (로그 누적 시간으로 역산한 epoch 67 종료 시각과 일치), `music_classifier_last.pt` mtime 10-01 11:13 (epoch 87) |

- 숫자만 보면 3차(val EER 3.11%)보다 크게 좋아 보이나, **아래 사후 점검 결과 이 val EER은 3차와 직접 비교할 수 없고 일반화 지표로도 쓸 수 없음.**

#### ❌ 이 체크포인트는 export·제출되지 않음

- 학습 종료(10-01 11:13)는 리더보드 마감(09-29 10:00)과 대회 종료(09-30 10:00) **이후**였음
- 확인: `baseline_submit/model/music_classifier/weights.pt`의 CRC32 `f6e9e676`은 `submit.zip` 내부 weights와 동일하며, 이는 09-24 03:56에 export한 **3차 재학습(epoch 31)** 체크포인트임. 4차(epoch 67)는 한 번도 `baseline_submit/`에 반영되지 않았음
- 따라서 **공개 리더보드 0.70871은 3차 재학습 체크포인트의 점수**이고, 4차는 리더보드로 검증되지 않았음

#### 제출 이력 (패키지 내부 `weights.pt` CRC32 대조로 복원)

| 패키지 | `music_classifier` 가중치 | 리더보드 결과 |
|--------|--------------------------|---------------|
| `0920_submit.zip` | 없음 (순수 baseline) | 총점 0.6909110688 / ADS 0.6577539683 / CPS 0.9893249735 / 21분 41초 |
| `0921_submit.zip` | `baaa06f5` (1차, Real 150개) | 총점 0.6722039259 / ADS 0.636968254 / CPS 0.9893249735 / 15분 51초 |
| `0922_submit.zip` | `baaa06f5` (0921과 동일 가중치·동일 script, 09-22 10:49 패키징) | 기록 없음 |
| `0922_2_submit.zip` | `7901d53e` (2차, val EER 5.33%) + CLI 인자 무시 버그 수정 script | 기록 없음 |
| `submit.zip` (09-28, 현재 `baseline_submit/`와 동일) | `f6e9e676` (3차, val EER 3.11%) | **총점 0.70871 / ADS 0.67753 / CPS 0.98932** (소요 시간 기록 없음) |

- 점수 산식 재계산으로 표의 총점·ADS·CPS 정합성 확인: `0.9 × 0.67753 + 0.1 × 0.98932 = 0.70871`, `0.9 × 0.6577539683 + 0.1 × 0.9893249735 = 0.6909110688`
- 09-28 제출 메모는 "4차"라고 적었으나 이 표 기준으로는 5번째 패키지임 (제출 횟수와 패키지 수가 같다는 보장은 없음 — 기록 없음)

---

### 2026-10-12 — 마무리 사후 점검

프로젝트를 포트폴리오로 정리하면서 문서의 수치를 로그·manifest·산출물과 하나씩 대조했고, 그 과정에서 이전 서술과 어긋나는 사실을 발견함. 아래는 **확인된 사실 / 해석 / 추측**을 구분해 적음.

#### 발견 1 — Fake 학습 데이터는 사실상 MusicGen_medium 단일 생성기였음 (파일명 충돌)

**확인된 사실**

- FakeMusicCaps의 5개 생성기 폴더(`MusicGen_medium` 5,520개 / `audioldm2`·`musicldm`·`mustango`·`stable_audio_open` 각 5,521개, 합계 27,604개)는 **같은 파일명(MusicCaps ID)** 을 쓴다. 예: `OnL-e9X0oEc.wav`가 5개 폴더에 모두 있음
- `prepare_data.py`는 분리 결과를 `{raw_path.stem}.npy`로 저장하고, 이미 있으면 건너뛰면서도 manifest에는 그 경로를 기록함 (`prepare_data.py`의 `process_class`). 파일명만 같으면 생성기가 달라도 같은 `.npy`로 취급됨
- `data/separated/fake/`의 `.npy` **5,521개 전부**의 길이가 162,880 samples(10.180초)임. 생성기별 원본 길이(각 40개 표본 측정)는 MusicGen_medium 162,880 / audioldm2·musicldm·stable_audio_open 160,000 / mustango 163,872 samples → 길이가 **MusicGen_medium과만 일치**
- 고유 stem이 5,521개뿐이므로 `--limit`을 5,521보다 크게 줘도 새 Fake 오디오는 늘지 않고 중복 행만 늘어남

**해석**: `rglob` 결과를 정렬해 앞에서부터 N개를 쓰는데, POSIX(WSL) 경로 정렬은 대소문자를 구분해 대문자로 시작하는 `MusicGen_medium`이 가장 앞에 옴. 그래서 앞쪽 N개가 MusicGen_medium 파일이었고, 같은 stem의 다른 생성기 파일은 `.npy`가 이미 있어 처리되지 않았음. 이 정렬 순서 설명은 길이 측정 결과와 모순이 없는 해석이며, 정렬 순서 자체를 직접 재현해 확인하지는 않음.

**영향**: 3차 재학습의 Fake 3,000개도 이 5,521개의 부분집합이므로(`.npy`는 덮어쓰지 않음) MusicGen_medium 단일 생성기였을 것으로 **추론**됨. 이전 문서와 제출 메모의 "FakeMusicCaps 5개 생성기 중 서브샘플링" 표현은 사실과 다름.

#### 발견 2 — manifest에 같은 파일이 여러 행으로 들어가 train/val이 겹침

**확인된 사실** (현재 `data/manifests/`, 09-29 00:52 생성)

- train 13,600행(Real 6,800 : Fake 6,800), val 2,400행(1,200 : 1,200)
- Real은 행마다 서로 다른 파일(train 6,800 / val 1,200 고유), train·val 겹침 0개
- Fake는 8,000행이지만 고유 `.npy`는 5,521개. train 6,800행 중 고유 5,017개, val 1,200행 중 고유 1,144개, **val의 고유 Fake 1,144개 중 640개(56%)가 train에도 들어 있음**

**해석**: 설계 의도("파일(트랙) 단위로 분할해 leakage 방지", 2026-09-20 결정)는 레코드 단위로만 지켜졌음. 서로 다른 레코드가 같은 `.npy`를 가리키는 상태에서 레코드를 섞어 나눴기 때문에 같은 오디오가 양쪽에 들어감. Fake 쪽 val은 학습에 쓰인 파일을 포함하므로 val EER이 낙관적으로 나옴.

#### 발견 3 — 학습 도중 val manifest가 바뀌어 epoch 18부터 다른 val로 평가됨

**확인된 사실**

- `evaluate_manifest()`는 호출될 때마다 `MusicManifestDataset(manifest_path)`로 val.csv를 **다시 읽음**(`evaluate.py`). 반면 train 데이터셋은 학습 시작 때 한 번만 읽음(`train.py`의 `main`)
- manifest 파일 수정 시각은 09-29 00:52:52. 로그 누적 시간으로 역산하면 epoch 17 종료 00:33, epoch 18 종료 01:13이므로 덮어쓰기는 epoch 18 학습 중이었음
- val EER은 `(FPR+FNR)/2`라서 클래스당 n개일 때 1/(2n) 단위의 값만 나옴. **epoch 1~17은 전부 1/1500 격자(클래스당 750개), epoch 18~87은 전부 1/2400 격자(클래스당 1,200개)** 로 바뀜
- `data/separated/real/`의 가장 최근 파일 생성 시각이 09-29 00:21 — 학습과 병행해 `prepare_data.py`가 한 번 더 실행된 흔적

**해석**: 학습 프로세스의 train 데이터는 시작 시점의 옛 train.csv 그대로인데, val만 새 val.csv(풀 16,000개에서 다시 섞어 뽑은 것)로 교체됨. 새 val은 옛 train에 쓰인 파일을 상당수 포함했을 가능성이 높음(옛 train이 새 풀의 부분집합이므로, 대략 절반 이상이 겹친다고 계산되나 **옛 train.csv가 덮어써져 소실**되어 정확한 겹침은 계산 불가). 그래서 epoch 18 이후 val EER 곡선은 epoch 1~17과 이어지지 않으며, best epoch 67(0.75%)도 이 오염된 val 기준으로 고른 값임.

**추측**: 병렬로 `prepare_data.py`를 돌린 의도는 기록이 없음(Real을 더 쓰려던 것으로 짐작되나 확인 불가). 옛 manifest 규모는 val 1,500개(750:750) 격자로부터 풀 약 10,000개 / train 약 8,500개로 **추정**되지만 기록된 `--limit 6000`과 맞지 않아 확정하지 못함.

#### 종합 판단

| 질문 | 판단 |
|------|------|
| 4차 재학습의 val EER 0.75%는 믿을 수 있는가 | **아니오.** 학습 도중 val이 바뀌었고(발견 3), 일부 Fake가 train과 겹치며(발견 2), 평가 대상이 단일 생성기임(발견 1). 3차의 3.11%와 비교할 수도 없음 |
| 3차 재학습의 val EER 3.11%는 어떤가 | 당시 manifest가 남아 있지 않아 확인 불가. 다만 Fake 3,000개가 서로 다른 stem이었을 것으로 추론되므로(`--limit`이 앞쪽 N개를 고름) 중복 누수는 없었을 가능성이 높음. 단 이는 코드 논리에 따른 추론이고, 평가 대상이 MusicGen_medium 단일 생성기라는 한계는 동일 |
| 리더보드 0.70871은 "여러 생성기에 대한 일반화"를 의미하는가 | **그렇게 말할 수 없음.** 개선 자체는 숨겨진 테스트셋에서 확인된 사실이지만, 그 원인이 Fake 학습 때문인지 Real 음악(MusicCaps·mtg-jamendo) 도메인 학습 때문인지 가르는 실험은 하지 않았음 |
| 09-21의 하락(0.6722)을 "데이터가 좁아서"라고 한 진단은 | 데이터 확대 후 점수가 올랐다는 사실과는 부합하지만, 그 사이 `ReduceLROnPlateau`·patience·배치·학습률도 함께 바뀌었으므로 데이터 확대가 단독 원인이라고 확정할 수는 없음 (추측 수준) |

#### 문서 수치 정정

| 이전 서술 | 실측 (2026-10-12) |
|-----------|-------------------|
| FakeMusicCaps raw 55,216개 | **27,604개** (5개 생성기 × 약 5,521, 논문 기준 27,605). 55,216이 어디서 나온 숫자인지는 재현되지 않음 (중복 집계로 추측) |
| Real raw 18,636개 | 18,635개 (`.gitkeep` 제외). 분리 완료 `.npy`는 8,016개 |
| 분리 완료 Fake 3,000개 → 6,000개로 확장 | 분리 산출물은 고유 5,521개가 상한(전부 MusicGen_medium). 확장분은 새 오디오가 아니라 중복 행 |
| "평가 서버 추론 시간 미확인" | 리더보드 소요 시간이 기록돼 있음: baseline 21분 41초, `music_classifier` 포함 15분 51초 → 60분 제한 대비 여유. 패키지 크기는 `submit.zip` 4.25 GiB로 10GB 제한 이내 |
| 제출 3건 | 위 "제출 이력" 표의 5개 패키지 (09-22 두 패키지는 점수 기록 없음) |

#### 이어서 한다면 (이 프로젝트에서는 중단)

1. **파일명 충돌 수정**: 분리 결과 파일명에 생성기 폴더를 포함(예: `{generator}__{stem}.npy`)하고, 5종 전체(27,604개)로 재분리
2. **분할 단위를 원곡(MusicCaps ID) 그룹으로**: 같은 ID에서 파생된 Real·Fake가 train/val로 갈라지지 않게 함
3. **val을 학습 시작 시 한 번만 읽거나, 실행마다 manifest를 스냅샷으로 저장**(`logs/{run}/`)해 학습 도중 파일이 바뀌어도 평가가 흔들리지 않게 함
4. **생성기 hold-out 검증**: 한 생성기를 통째로 빼고 학습해 미지 생성기 일반화를 직접 측정
5. **ablation**: Real 도메인 효과 vs Fake 데이터 효과를 분리하고, weight decay·patience·데이터 규모를 한 번에 하나씩만 바꿈
6. 데이터 증강(SpecAugment 등) — 09-28에 논의했으나 미착수

---

## 현재 상태 (프로젝트 종료 시점)

| 항목 | 상태 | 비고 |
|------|------|------|
| 대회 규정 분석 | ✅ 완료 | |
| baseline 코드 분석 | ✅ 완료 | script.py는 zip에서 별도 추출 필요 |
| music_fake_pipeline 구현 | ✅ 완료 (v2: 원샷 파이프라인 `run_pipeline.py`까지 구축) | |
| README.md | ✅ 최종 갱신 (2026-10-12) | 로컬 환경은 pipeline README 9장 참고. 이 README 상단에 프로젝트 요약, §14에 회고 추가 |
| **FakeMusicCaps (Fake 샘플)** | ✅ 확보 완료 | raw **27,604개**(5개 생성기 폴더). 분리 완료 `.npy` 고유 **5,521개 — 전부 MusicGen_medium** (파일명 충돌, 사후 점검 발견 1) |
| **Real 음악 데이터** | ✅ 확보 완료 (MusicCaps 150 + mtg-jamendo 다수) | 원본 18,635개, 분리 완료 `.npy` **8,016개** |
| **Real 음악 추가 확보 (mtg-jamendo-dataset)** | ✅ 해결 완료 | `setuptools` 빌드 오류 해결, 트랙 다운로드 완료 |
| HTDemucs 분리 (prepare_data) | ✅ 완료 (Fake는 고유 stem 상한 5,521에 도달) | 이후 `--limit`을 올려도 Fake는 중복 행만 늘어남 |
| 모델 학습 (train) | ✅ 4회 재학습 완료 | 3차: epoch 31 best val EER 3.11% (제출·리더보드 검증됨). **4차(최종): epoch 67 best val EER 0.75%, epoch 87 조기 종료 — val 오염으로 신뢰 불가, 마감 후 종료되어 미제출** |
| 평가 (evaluate) | ✅ 완료 | 3차 val EER 3.11%는 리더보드 방향과 일치. 4차 val EER은 사후 점검에서 비교 불가로 판정 |
| submit.zip 통합 및 제출 | ✅ 종료 | 공개 리더보드 최고 **0.70871**(3차 체크포인트, 09-28 확인). baseline 0.6909, 1차 0.6722. 제출 이력 전체는 "마지막 재학습" 항목의 표 참고 |
| 대회 종료 후 사후 점검 | ✅ 완료 (2026-10-12) | 단일 생성기 / manifest 중복 / 학습 중 val 교체 3건 발견·기록. **코드는 수정하지 않음** (결과를 만든 코드 상태를 그대로 보존) |

---

## 계획 항목의 최종 처리 결과

> 대회 중(09-28까지) 세워둔 작업 계획을 종료 시점 기준으로 정리한 것. 체크 표시와 `→` 주석이 최종 결과이며, 원문 계획은 이력으로 남겨둠. 앞으로 이어간다면 할 일은 "2026-10-12 — 마무리 사후 점검" 끝의 목록을 볼 것.

> **09-28 당시 진단**: 이번 제출에서 처음으로 baseline(0.6909)을 돌파(0.70871)함 — "데이터가 좁아서 음악 경로 교체가 손해였다"는 09-21의 진단과 부합하는 결과. (단, 사후 점검에서 학습에 쓰인 Fake가 단일 생성기였음이 드러나 이 해석에는 단서가 붙음 — 데이터 확대 외에 스케줄러·patience 변경이 함께 있었고 효과를 분리하지 않았음.)

#### 1단계 — Fake 데이터 볼륨 확장 (최우선이었음)

- [x] `mtg-jamendo-dataset/venv`의 `setuptools` 업그레이드 → 설치 성공, 트랙 다운로드 완료
- [x] Real 음악을 MusicCaps 150개보다 다양하게 확대 (mtg-jamendo, 원본 18,636개 확보, 분리 완료 6,884개)
- [x] Real:Fake 균형 데이터로 재학습 3회 반복(4.67%→5.33%→3.11%) — 상세는 위 09-22·09-23~24 항목 참고
- [x] **Fake HTDemucs 분리를 3,000→6,000개로 확장** — 실행했으나 **새 오디오가 늘지 않았음.** 고유 stem이 5,521개뿐이고 전부 MusicGen_medium이라, 확장분은 중복 행이었음 (사후 점검 발견 1·2)
- [ ] Fake 음악도 FakeMusicCaps 5개 생성기 외 추가 확보 검토 (Echoes 등 — 생성기 다양성 확대 목적) → **미착수.** 그 전에 FakeMusicCaps 5종 자체를 온전히 쓰는 것이 선행 과제로 드러남
- [ ] 각 데이터셋 라이선스 조항 최종 확인 (FakeMusicCaps CC-BY-NC-4.0, mtg-jamendo 등, 대회 "규칙" 탭 기준) → **미완료.** 코드 공개·포트폴리오 게시 전에 확인 필요

#### 2단계 — `--limit 6000` 재학습 결과 검증 및 5차 제출 판단

- [x] `run_pipeline.py --limit 6000` 완료 후 새 `data/manifests/` 크기·클래스 균형 확인 → 현재 manifest는 16,000행(8,000 : 8,000)이나 Fake는 고유 5,521개 (manifest는 학습 도중 09-29 00:52에 덮어써진 것)
- [x] 새 체크포인트의 val EER이 3.11%(3차)보다 개선되는지 확인 → 0.75%로 수치상 개선, 그러나 **비교 불가 판정**. 마감(09-29 10:00) 후인 10-01에 종료되어 export·제출하지 않았고, `baseline_submit/`은 3차 체크포인트 그대로임
- [x] (선택) `WEIGHT_DECAY` 1e-3 실험 → 4차 재학습에서 patience 20, 데이터 변경과 **동시에** 적용됨. 단독 효과는 알 수 없음
- [x] **일일 제출 3회 제한** → 종료 시점에서는 의미 없어짐. 09-28 이후 추가 제출의 흔적(zip 갱신·점수)은 확인되지 않음

#### 3단계 — 개선 (완료됨, 참고용 기록)

- [x] `CosineAnnealingLR`(T_max=100) vs `EARLY_STOP_PATIENCE=6` 조기 종료 간 불일치 수정 → `ReduceLROnPlateau`로 교체 (`train.py`)
- [x] `EARLY_STOP_PATIENCE` 근거 재검토 — `train_log.csv` 실측 기반으로 6 → 10 상향
- [x] 새 스케줄러(`ReduceLROnPlateau`) + 새 patience(10)로 재학습 완료 — best val EER 3.11%(epoch 31), 기존 5.33% 대비 개선. 리더보드로도 개선 확인(0.70871)
- [x] 원샷 파이프라인(`run_pipeline.py`) + 실행별 로그 분리(`run_logger.py`) + 학습 진행 게이지 구축, `prepare_data.py` 증분 처리·`config.py` 경로 버그 수정
- [x] 추론 시간 실측 → 리더보드 소요 시간으로 확인: baseline 21분 41초, `music_classifier` 포함 15분 51초 (60분 제한 대비 여유)

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
| 2026-09-22 | LR 스케줄러를 `CosineAnnealingLR` → `ReduceLROnPlateau`로 교체 | `T_max=100` 전제가 patience 기반 조기종료(실제 23 epoch)와 구조적으로 불일치 — early stopping과 같은 신호(val EER 정체)로 LR도 낮추도록 통일 |
| 2026-09-22 | `EARLY_STOP_PATIENCE` 6 → 10 상향 | `train_log.csv` 실측: 5 epoch 연속 정체 후 개선되는 패턴이 반복 관찰됨 — 6은 최소 하한선이었고, epoch당 비용(~3분)이 낮아 여유를 더 둠 |
| 2026-09-28 | `WEIGHT_DECAY` 1e-3 변경은 커밋하지 않고 보류 | 이번 제출이 이미 baseline을 돌파해 급하게 정규화를 더 강화할 필요가 낮아짐 — D-1 상황에서 검증 안 된 변경보다는 Fake 데이터 확장(1단계)에 시간을 쓰는 게 우선이라 판단, 사용자 확인 |
| 2026-09-28 | 다음 라운드 우선순위를 "Fake 데이터 볼륨 확장 > augmentation > WEIGHT_DECAY 실험" 순으로 결정 | HTDemucs 분리 완료분이 Real 6,884개 vs Fake 3,000개로 Fake가 병목 — Fake만 늘리면 이미 확보된 Real을 그대로 활용 가능해 비용 대비 효과가 가장 큼. **→ 사후 점검: Fake 고유 stem이 5,521개뿐이고 단일 생성기여서 이 전제(Fake를 늘리면 규모가 커진다)가 성립하지 않았음** |
| 2026-09-28 | (번복) `WEIGHT_DECAY` 1e-3·`EARLY_STOP_PATIENCE` 20을 적용해 4차 재학습 시작 | 위 "보류" 결정 후 몇 시간 안에 번복됨. 번복 근거를 담은 기록은 없음. 데이터·weight decay·patience를 동시에 바꿔 효과 분리 불가 |
| 2026-10-12 | 사후 점검에서 발견한 결함(파일명 충돌·manifest 중복·학습 중 val 교체)을 **코드에서 고치지 않고 문서에만 기록** | 수정하면 리더보드 점수(0.70871)를 만든 코드와 저장소 상태가 달라져 결과의 재현 근거가 흐려짐. 고치려면 재분리·재학습이 필요하며 대회가 이미 끝남 |
| 2026-10-12 | 4차 재학습 val EER(0.75%)을 성과 지표로 쓰지 않음 | val 도중 교체 + train/val 중복 + 단일 생성기 평가. 대표 성과는 리더보드 0.70871만 사용 |

---

## 미해결 사항 및 한계 (프로젝트 종료 시점)

| 항목 | 내용 | 영향도 |
|------|------|--------|
| **Fake 학습 데이터가 단일 생성기** | 5개 생성기 폴더의 파일명이 같아 `{stem}.npy` 충돌 → 분리된 Fake 5,521개 전부 MusicGen_medium. 다른 생성기에 대한 일반화는 한 번도 측정되지 않음 (사후 점검 발견 1) | 높음 |
| **manifest 중복으로 인한 train/val 겹침** | Fake 8,000행이 고유 5,521개를 가리키며 val 고유 Fake의 56%가 train에도 있음 (발견 2) | 높음 |
| **학습 중 val manifest 교체** | 4차 재학습의 epoch 18부터 val이 바뀌어 val EER 곡선이 연속되지 않음 (발견 3) | 높음 (4차 val EER 신뢰 불가) |
| **효과 분리 안 됨** | 3차→4차에서 데이터·weight decay·patience를 동시에 변경. 09-21→09-28 점수 상승도 데이터·스케줄러·patience·배치·학습률 변경이 겹침 | 중간 |
| 최종 순위 | 480 중 308위. baseline 돌파(+0.0178)가 상위권으로 이어지지 못함 — 단일 생성기 학습 등 위 한계와 연결되는지는 순위만으로 알 수 없음 | — |
| 2차 평가 | 참여 여부 기록 없음 | 정보 부족 |
| 09-22 제출 2건의 점수 | 패키지는 남아 있으나 점수 기록 없음 | 낮음 |
| ~~mtg-jamendo-dataset 설치 실패~~ | **해결됨(09-22)** — setuptools 업그레이드로 pip install 성공, Real 원본 150→18,636개로 확대 | — |
| 라이선스 미확인 | FakeMusicCaps(CC-BY-NC-4.0), mtg-jamendo 등 대회 적용 가능 여부 불명확 — 대회 규칙 탭 재확인 필요. 포트폴리오로 코드를 공개하기 전에도, 저장소에 포함된 대회 제공물(`baseline_submit/script.py`, `df_arena_1b/` 소스, `data/test/` 더미, `deepvoice_competition_summary.md`)의 재배포 가능 여부를 확인해야 함 | 높음 |
| ~~`music_classifier`가 DF-Arena zero-shot보다 실제로 더 나쁨~~ | **해결됨(09-28)** — 09-20(순수 baseline) 총점 0.6909, 09-21(`music_classifier` 1차, 좁은 데이터) 총점 0.6722로 한때 baseline보다 열등했으나, 데이터 확대(학습 150→2,550~2,700)+`ReduceLROnPlateau`/patience 10 재학습 후 이번 제출에서 총점 **0.70871**로 처음 baseline을 돌파함. Real 150개/Fake 5종뿐이던 좁은 데이터가 원인이었다는 진단이 맞았던 것으로 확정 | — |
| ~~Fake 데이터가 현재 학습 규모의 병목~~ | **전제가 틀렸음(10-12).** Fake를 늘려도 고유 오디오가 5,521개(전부 MusicGen_medium)에서 더 늘지 않음 — 위 첫 행으로 대체 | — |
| GPU 활용률 저조 | 학습 중 VRAM은 15.8/16GB 점유되나 GPU 사용률 1% — CPU 쪽 오디오 전처리(멜스펙트로그램 계산)가 병목으로 추정(미검증), 개선 시 학습 속도 향상 가능. 4차 재학습은 epoch당 평균 2,324초로 오래 걸렸음 | 낮음 |
| 검증 세트 규모 | 3차: val 900개(450:450)로 리더보드 방향과 일치. 4차: 위 "val manifest 교체"로 신뢰 불가. 어느 쪽이든 **단일 생성기** 평가라는 한계는 동일 | 중간 |
| ~~평가 서버 추론 시간 미확인~~ | **해결.** 리더보드 소요 시간 15분 51초(`music_classifier` 포함, 09-21) — 60분 제한 대비 여유. 패키지 4.25 GiB로 용량 제한 이내 | — |
| ~~일일 제출 횟수 소진 관리~~ | 대회 종료로 의미 없어짐 | — |
