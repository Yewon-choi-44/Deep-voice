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

---

## 9. 트러블슈팅 및 실제 실행 기록 (2026-09-20 ~ 21, 추후 정리 예정)

> 아래는 실제 환경에서 파이프라인을 처음 돌리면서 마주친 문제와 수정 사항을 날것으로 기록한 섹션입니다.
> 위 섹션들의 내용을 수정하지 않고 누적식으로 추가합니다.

---

### 9-1. 실제 환경 사양

| 항목 | 내용 |
|---|---|
| OS | Windows 11 Pro for Workstations |
| Python | 3.13.2 (시스템 설치, `C:\Users\AI-00\AppData\Local\Programs\Python\Python313\`) |
| GPU | NVIDIA Quadro RTX 5000 (VRAM 16GB) |
| NVIDIA 드라이버 | 596.71 |
| CUDA 드라이버 버전 | 13.2 (드라이버가 지원하는 최대 CUDA 버전) |
| PyTorch 빌드 | `2.11.0+cu128` (CUDA 12.8 휠) |

→ 2번 섹션의 `torch==2.7.1+cu128`, `uv venv --python 3.11` 내용은 실제 환경 기준으로 아직 업데이트되지 않음.

---

### 9-2. PyTorch CUDA 설치 삽질

**문제:** `pip install torch` 또는 `uv pip install torch`를 아무 옵션 없이 실행하면
PyPI에서 CPU 전용 빌드(`2.11.0+cpu`)가 설치된다. `nvidia-smi`가 정상이어도
`torch.cuda.is_available()`이 `False`를 반환한다.

**원인:** PyTorch CUDA 빌드는 2~3GB라 PyPI 100MB 제한을 초과한다.
PyTorch 팀이 별도 휠 서버(`download.pytorch.org/whl/`)를 운영하며, PyPI에는 CPU 버전만 올라와 있다.

**해결:**
```bash
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

uv를 사용한다면 `--index-url` 대신 `pyproject.toml`에 인덱스를 등록해두면
이후 `uv sync` 한 번으로 끝난다 (9-3 참고).

**CUDA 버전 선택 기준:** `nvidia-smi` 우측 상단의 `CUDA Version`은 드라이버가
지원하는 최대 버전이다. 그보다 낮은 CUDA 빌드는 전부 호환된다.
(이 환경: 드라이버 13.2 → `cu128` 정상 동작 확인)

**확인 방법:** `python cuda_check.py` (프로젝트 루트 `Deep_voice/`에 작성됨)

---

### 9-3. pyproject.toml 생성 및 uv 설정

`requirements-dev.txt`와 병행하여 `pyproject.toml`을 생성했다.
`torch`/`torchaudio`의 CUDA 인덱스를 여기에 등록해두면 `uv sync` 한 번으로
전체 환경이 재현된다.

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }
```

**새 환경 세팅 순서 (현재 권장):**
```bash
# 1. nvidia-smi 로 드라이버/CUDA 버전 확인
nvidia-smi

# 2. 가상환경 생성
uv venv

# 3. 전체 의존성 설치 (torch는 자동으로 cu128 버전으로 설치됨)
uv sync

# 4. GPU 연결 확인
python ../../cuda_check.py
```

---

### 9-4. yt-dlp 및 ffmpeg 미설치 문제

`download_musiccaps.py`의 docstring에 필수 조건으로 명시되어 있지만,
`requirements-dev.txt`에 빠져 있었다. 또한 `ffmpeg`은 Python 패키지가 아니라
시스템 바이너리라 별도 설치가 필요하다.

**설치 방법:**
```bash
# yt-dlp (Python 패키지)
pip install yt-dlp

# ffmpeg (시스템 바이너리, Windows)
winget install Gyan.FFmpeg
# 설치 후 터미널 재시작 필요 (PATH 반영)
```

`pyproject.toml`과 `requirements-dev.txt`에 `yt-dlp`, `datasets[audio]`는 이미 추가됨.
ffmpeg은 시스템 설치라 패키지 목록에는 포함할 수 없으므로 이 문서에만 기록.

---

### 9-5. download_musiccaps.py 수정 이력

**수정 1 — data_dir 절대 경로 고정**

원본의 `'./music_data'` (CWD 기준 상대 경로)를 스크립트 위치 기준으로 고정:
```python
# 수정 전
'./music_data'

# 수정 후
Path(__file__).parent / 'data' / 'raw' / 'real_music'
```
이전에는 실행 디렉토리에 따라 저장 위치가 달라져서 `Deep_voice/music_data/`에 빈 디렉토리만 생긴 적 있음.

**수정 2 — cast_column 제거**

```python
# 제거된 코드
.cast_column('audio', Audio(sampling_rate=sampling_rate))
```
`datasets` 최신 버전(4.8.5)에서 `Audio` 디코딩 시 `torchcodec`이 필요한데,
파일을 디스크에 저장하는 게 목적인 이 스크립트에서는 불필요한 단계였다.
`from datasets import load_dataset, Audio` → `from datasets import load_dataset`로 수정.

---

### 9-6. 실제 데이터 현황 및 클래스 불균형

**MusicCaps (REAL):**
- 전체 5,521개 클립이지만 YouTube에서 삭제된 영상이 많아 실제 다운로드 가능한 수: **약 150개**
- 파일명은 YouTube 영상 ID (`-0Gj8-vB1q4.wav` 형태)

**FakeMusicCaps (FAKE):**
- 5개 생성기, 총 55,216개 클립 (`data/raw/fake_music/FakeMusicCaps/` 하위에 생성기별 폴더로 구성)

**클래스 불균형 대응:**
- `train.py`에 `pos_weight` 등 class weight 처리가 없음
- 이 상태에서 fake 55,216개를 그대로 쓰면 모델이 무조건 "fake"만 예측하도록 학습됨
- **현재 권장**: `prepare_data.py` 실행 시 `--limit 150`으로 클래스 균형 맞추기
  (real 150개 + fake 150개 = 총 300개)
- 추후 개선 방향: `train.py`에 `pos_weight` 추가 후 더 많은 fake 데이터 활용

---

### 9-7. 실제 실행 명령 (2026-09-21 기준)

**준비:**
```bash
# 실행 위치
cd D:\wiki\wiki\wiki\projects\Deep_voice\music_fake_pipeline\music_fake_pipeline
```

**데이터 다운로드 (MusicCaps REAL):**
```bash
python download_musiccaps.py
# 이미 받은 파일은 자동으로 건너뜀 (재실행 안전)
```

**데이터 준비 (HTDemucs 분리 + manifest 생성):**
```bash
python prepare_data.py \
    --htdemucs-dir ../../baseline_submit/model/htdemucs \
    --device cuda \
    --limit 150
```

**학습:**
```bash
python train.py --device cuda
```

**평가:**
```bash
python evaluate.py --checkpoint checkpoints/music_classifier_best.pt
```
