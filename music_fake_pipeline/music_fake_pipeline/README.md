# Music FAKE 분류기 — 개발 파이프라인

딥보이스 탐지 대회 baseline의 `MUSIC_FAKE_PROB` 계산 부분만 교체하는 프로젝트입니다.
왜 이 부분만 손대는지, 전체 설계 배경은 대화에서 이미 다룬 내용이라 여기서는
반복하지 않고 **실행 방법**에 집중합니다.

## 0. 이 프로젝트가 건드리지 않는 것

- PANNs 존재 판별 (VOICE_PRESENT_PROB, MUSIC_PRESENT_PROB)
- HTDemucs 음성/음악 분리
- DF-Arena 1B 기반 VOICE_FAKE_PROB
- MAX Fusion 결합 로직

`patched_script.py`와 baseline 원본을 줄 단위로 diff해서 확인했고, 실제로 바뀐 곳은
**음악 분기의 모델 교체 한 곳**뿐입니다 (아래 6번 섹션 참고).

## 1. 디렉토리 구조

```
music_fake_pipeline/
├── config.py                # 전역 설정 (경로, 오디오 상수, 학습 하이퍼파라미터)
├── audio_utils.py           # baseline과 동일한 세그먼트 분할/로딩 로직
├── demucs_separate.py       # HTDemucs 분리 (baseline 코드 재사용)
├── features.py               # waveform -> log-mel spectrogram
├── model.py                  # MusicFakeClassifier (소형 CNN, 약 25만 파라미터)
├── dataset.py                 # manifest -> PyTorch Dataset (train/eval 모드)
├── prepare_data.py            # 원본 오디오 -> HTDemucs 분리 -> manifest 생성
├── train.py                   # 학습 루프 (val EER 기준 best checkpoint 저장)
├── evaluate.py                 # 대회 평가 산식과 동일한 EER 계산
├── export_for_submit.py       # 체크포인트 -> submit.zip에 넣을 자기완결 패키지
├── patched_script.py          # baseline script.py에 music_classifier를 통합한 전체 코드
├── requirements-dev.txt        # 로컬 학습 환경용 패키지 버전 고정
├── data/
│   ├── raw/real_music/        # (직접 채워 넣을 것) REAL 음악 원본
│   ├── raw/fake_music/        # (직접 채워 넣을 것) AI 생성 음악 원본
│   ├── separated/{real,fake}/ # HTDemucs로 분리된 음악 성분 (.npy)
│   └── manifests/{train,val}.csv
├── checkpoints/                # 학습된 가중치
└── export/music_classifier/    # submit.zip에 그대로 복사할 최종 산출물
```

## 2. 로컬 환경 설정

대회 평가 서버 사양 문서에 명시된 버전을 그대로 씁니다. `requirements-dev.txt`에
고정해뒀습니다.

```bash
uv venv --python 3.11
uv pip install -r requirements-dev.txt
# CUDA 12.8 GPU를 쓴다면 torch/torchaudio는 아래처럼 cu128 wheel로 설치
uv pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

### ⚠️ 반드시 알아야 할 버전 함정 (직접 확인한 사실)

`demucs`는 **4.0.1과 4.1.0 사이에 호환되지 않는 API 변경**이 있습니다.
`demucs.separate.load_track` 함수가 4.0.1에는 있지만 **4.1.0에서는 제거되었습니다.**
baseline script.py와 이 프로젝트의 `demucs_separate.py` 둘 다 이 함수를 직접
import해서 씁니다. `pip install demucs`만 실행하면 최신 버전(4.1.0)이 깔려서
`ImportError: cannot import name 'load_track'`가 발생합니다. **반드시
`demucs==4.0.1`로 버전을 고정하세요.** (평가 서버에도 4.0.1이 고정 설치되어
있으므로, 이 핀을 지키지 않으면 로컬 검증 자체가 무의미해집니다.)

## 3. 데이터 준비

`data/raw/real_music/`, `data/raw/fake_music/`에 오디오 파일을 직접 받아서
채워 넣으세요. 대화에서 조사한 후보:

| 용도 | 데이터셋 | 비고 |
|---|---|---|
| FAKE 음악 | FakeMusicCaps | 생성기 5종(MusicGen 등), 10초 클립 |
| FAKE 음악 | Echoes | 생성기 12종, 일반화 검증용 |
| REAL 음악 | FMA (Free Music Archive) | CC0/CC-BY 라이선스 트랙만 선별 |
| REAL/FAKE | SONICS | 대용량, 일부만 서브샘플링 권장 |

**라이선스와 대회 자체의 외부 데이터 사용 규정은 아직 확인되지 않은 상태이니,
실제로 학습에 쓰기 전에 대회 "규칙" 탭에서 재확인하세요.**

파일을 채워 넣은 뒤:

```bash
# baseline_submit.zip 압축 해제 위치의 model/htdemucs 경로를 지정
uv run prepare_data.py --htdemucs-dir /path/to/baseline_submit/model/htdemucs

# 빠른 파이프라인 점검만 하고 싶다면 (HTDemucs 분리 생략, 비권장):
uv run prepare_data.py --skip-demucs --limit 20
```

`data/manifests/train.csv`, `val.csv`가 생성됩니다. **분할은 세그먼트가 아니라
원본 파일(트랙) 단위**로 이루어집니다 — 같은 트랙의 다른 구간이 train/val에
동시에 들어가면 검증 EER이 실제보다 낙관적으로 나오는 data leakage가
생기기 때문입니다.

## 4. 학습

```bash
uv run train.py
# 또는
uv run train.py --epochs 15 --batch-size 16 --device cuda
```

매 epoch마다 `evaluate.py`의 EER 계산 로직으로 검증하고, val EER이 개선될 때만
`checkpoints/music_classifier_best.pt`를 갱신합니다. `EARLY_STOP_PATIENCE`
(기본 6 epoch) 동안 개선이 없으면 자동 종료합니다.

## 5. 평가

```bash
uv run evaluate.py --checkpoint checkpoints/music_classifier_best.pt --save-csv logs/val_predictions.csv
```

대회 평가 산식(`fpr/fnr` 교차점 EER)과 완전히 동일한 계산식을 사용하므로,
여기서 나온 숫자가 실제 리더보드 Music EER의 신뢰할 만한 사전 추정치입니다.
출력에 `0.9 * 0.3 * (1 - EER)` 값도 함께 표시되는데, 이게 총점(Score) 중
이 분류기가 기여하는 정확한 몫입니다(배점 구조: ADS 내 Music 가중치 0.3,
전체 Score 내 ADS 가중치 0.9).

## 6. submit.zip 통합

```bash
uv run export_for_submit.py --checkpoint checkpoints/music_classifier_best.pt
```

`export/music_classifier/`에 `weights.pt`와 `music_fake_infer.py`(자기완결
추론 모듈, 이 프로젝트의 다른 파일에 의존하지 않음)가 생성됩니다.

**통합 절차:**
1. `baseline_submit.zip`을 압축 해제
2. `export/music_classifier/` 폴더 전체를 `model/music_classifier/`로 복사
   ```
   baseline_submit/
   ├── model/
   │   ├── df_arena_1b/      (기존, 그대로)
   │   ├── htdemucs/         (기존, 그대로)
   │   ├── panns/            (기존, 그대로)
   │   └── music_classifier/ (신규)
   │       ├── weights.pt
   │       └── music_fake_infer.py
   ├── script.py
   └── requirements.txt
   ```
3. `patched_script.py`의 내용으로 `script.py`를 통째로 교체
4. `requirements.txt`는 **수정할 필요 없음** — torch/torchaudio는 이미 포함되어 있고,
   신규 모듈이 추가로 요구하는 패키지가 없음

## 7. 제출 전 검증 체크리스트

- [ ] `data/test/`의 더미 3개 파일로 로컬에서 `script.py` 전체 실행이 에러 없이 끝나는지
- [ ] `output/submission.csv`가 `sample_submission.csv`와 동일한 컬럼/행 수로 나오는지
- [ ] (가능하면) 평가 서버와 동일한 Python 3.11.15 + CUDA 12.8 + torch 2.7.1+cu128
      환경에서 한 번 더 실행 — 로컬 학습 환경과 미묘하게 버전이 다르면 체크포인트가
      로드는 되어도 추론 결과가 달라질 위험이 있음
- [ ] `submit.zip` 압축 시 최상위에 불필요한 폴더가 끼지 않았는지 (구조 불일치는
      "설치 오류"로 처리되어 일일 제출 횟수에 반영되진 않지만, 시간 낭비이므로 미리 확인)
- [ ] 전체 zip 용량 10GB 이하, 압축 해제 후 32GB 이하
- [ ] 추론 60분 / 설치 10분 제한 — `music_classifier` 추가로 인한 지연은 미미할
      것으로 예상되지만(DF-Arena 대비 훨씬 가벼운 모델), 실측 확인 필요

## 8. 이 프로젝트에서 검증/확인한 사실 vs 추정

**직접 실행해서 확인한 것:**
- `model.py`/`features.py`/`dataset.py`/`train.py`/`evaluate.py`/
  `export_for_submit.py` 전체 파이프라인을 합성(synthetic) REAL/FAKE 데이터로
  end-to-end 실행 — 정상 동작 확인 (초기 버전에 있던 텐서 채널 차원 버그를
  이 과정에서 발견·수정함)
- `export_for_submit.py`가 만든 `music_fake_infer.py`가 프로젝트의 다른 파일 없이
  완전히 독립적으로 로드·추론되는지 확인
- `demucs==4.0.1` vs `4.1.0` 간의 `load_track` API 차이를 직접 재현·확인
- `patched_script.py`가 baseline 원본과 의도한 부분만 다른지 diff로 확인

**아직 확인되지 않은 것 (실측 필요):**
- 실제 HTDemucs 분리 결과물에 대한 학습 성능 (이 환경에는 실제 HTDemucs
  가중치가 없어 synthetic 데이터로만 검증함)
- 실제 대회 데이터에서의 Music EER 개선 폭
- 평가 서버 환경에서의 실행 시간/용량
