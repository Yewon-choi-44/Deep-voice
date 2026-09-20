---
type: Project
title: 딥보이스 탐지 대회 — 개발 일지
description: 대회 분석부터 제출까지의 작업 이력과 진행 계획
tags: [project, deepvoice, competition]
timestamp: 2026-09-20
status: active
---

# 딥보이스 탐지 대회 — 개발 일지

> **리더보드 제출 마감: 2026-09-29 (화) 10:00 KST** — 작성 시점 기준 **D-9**

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

---

## 현재 상태

| 항목 | 상태 | 비고 |
|------|------|------|
| 대회 규정 분석 | ✅ 완료 | |
| baseline 코드 분석 | ✅ 완료 | script.py는 zip에서 별도 추출 필요 |
| music_fake_pipeline 구현 | ✅ 완료 (합성 데이터 검증) | 실제 데이터 학습은 미시작 |
| README.md | ✅ 완료 | |
| **FakeMusicCaps (Fake 샘플)** | 🔄 다운로드 중 | |
| **Real 음악 데이터** | ❌ 미확보 | FMA 등 확보 필요 |
| HTDemucs 분리 (prepare_data) | ❌ 미시작 | 데이터 확보 후 실행 |
| 모델 학습 (train) | ❌ 미시작 | |
| 평가 (evaluate) | ❌ 미시작 | |
| submit.zip 통합 및 제출 | ❌ 미시작 | |

---

## 앞으로의 작업 계획

### 우선순위 순서

#### 1단계 — 데이터 확보 (즉시)

- [ ] FakeMusicCaps 다운로드 완료 확인 → `data/raw/fake_music/`에 배치
- [ ] Real 음악 데이터셋 확보
  - FMA (Free Music Archive) CC0/CC-BY 트랙 우선 검토
  - 라이선스 대회 규정 적합 여부 확인 후 다운로드
- [ ] 각 데이터셋 라이선스 조항 최종 확인 (대회 "규칙" 탭 기준)

#### 2단계 — 데이터 준비 및 학습

- [ ] `prepare_data.py` 실행 — HTDemucs 분리 + manifest 생성
  ```bash
  uv run prepare_data.py --htdemucs-dir ../../baseline_submit/model/htdemucs
  ```
- [ ] `train.py` 실행
  ```bash
  uv run train.py --epochs 30 --batch-size 32 --device cuda
  ```
- [ ] `evaluate.py`로 val EER 확인 — 리더보드 Music EER 사전 추정
  ```bash
  uv run evaluate.py --checkpoint checkpoints/music_classifier_best.pt
  ```

#### 3단계 — 통합 및 제출

- [ ] `export_for_submit.py` 실행 → `export/music_classifier/` 생성
- [ ] `baseline_submit.zip` 압축 해제 → `music_classifier/` 추가 → `patched_script.py`로 교체 → `submit.zip` 재압축
- [ ] 로컬에서 더미 3개 파일로 `patched_script.py` 전체 실행 검증
- [ ] 리더보드 제출 (일일 3회 제한 유의)

#### 4단계 — 개선 (시간 여유 시)

- [ ] Echoes 데이터셋 추가하여 재학습 (생성기 다양성 확대)
- [ ] 하이퍼파라미터 튜닝 (batch size, learning rate 등)
- [ ] 추론 시간 실측 후 여유 있으면 더 큰 모델 실험

---

## 주요 결정 사항 기록

| 날짜 | 결정 | 근거 |
|------|------|------|
| 2026-09-20 | MUSIC_FAKE_PROB 경로만 교체 | Music EER 가중치(0.3) > Voice(0.2), baseline의 유일한 미검증 경로 |
| 2026-09-20 | 소형 CNN 선택 (25만 파라미터) | 60분 추론 제한 내 DF-Arena보다 연산량 감소 목표 |
| 2026-09-20 | train/val 분할은 트랙 단위 | 세그먼트 단위 분할 시 data leakage 발생 |
| 2026-09-20 | demucs==4.0.1 고정 | 4.1.0에서 load_track API 제거 — 직접 재현 확인 |

---

## 미해결 사항 및 리스크

| 항목 | 내용 | 영향도 |
|------|------|--------|
| Real 음악 데이터 미확보 | FMA 등 확보 전까지 학습 불가 | 높음 |
| 라이선스 미확인 | FakeMusicCaps(CC-BY-NC-4.0) 등 대회 적용 가능 여부 불명확 | 높음 |
| 실제 Music EER 개선 폭 미확인 | 합성 데이터만 검증 — 실제 분리 음악에서 성능 보장 없음 | 중간 |
| 평가 서버 추론 시간 미확인 | music_classifier 추가 후 60분 제한 내 완료 여부 | 낮음 (모델이 매우 가벼움) |
