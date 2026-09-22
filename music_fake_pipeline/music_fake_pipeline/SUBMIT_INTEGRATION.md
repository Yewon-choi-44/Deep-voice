# SUBMIT_INTEGRATION — 학습된 체크포인트를 submit.zip으로 통합하기

`export_for_submit.py`가 안내하는 문서가 바로 이 파일입니다. 아래 순서를 그대로 따르면
학습된 `music_classifier` 체크포인트를 대회 제출용 zip 내용물로 구성할 수 있습니다.

이 문서의 모든 명령어는 **이 파일이 있는 디렉토리**
(`music_fake_pipeline/music_fake_pipeline/`)를 기준(CWD)으로 작성되어 있습니다.

전제: `../../baseline_submit/`에 `baseline_submit.zip`이 이미 한 번 압축 해제되어 있고,
`model/df_arena_1b`·`model/htdemucs`·`model/panns`가 들어있는 상태. 매번
`baseline_submit.zip`을 새로 풀 필요는 없습니다 — 이미 풀려있는 폴더를 계속 재사용하면
됩니다.

---

## 1. 체크포인트 export

> ⚠️ **주의**: 이 디렉토리 자체의 `.venv`에는 `torch`가 설치되어 있지 않습니다.
> 반드시 **프로젝트 루트(`../../`)의 `.venv`**(torch+CUDA 포함, 학습 때 쓴 환경)를
> 사용해야 합니다.

```bash
# 프로젝트 루트 .venv의 python을 직접 지정 (WSL 기준 예시)
../../.venv/bin/python3 export_for_submit.py \
    --checkpoint checkpoints/music_classifier_best.pt
```

- `export/music_classifier/weights.pt` + `export/music_classifier/music_fake_infer.py`가 생성됩니다.
- `music_fake_infer.py`는 이 프로젝트의 다른 파일에 의존하지 않는 **자기완결 추론 모듈**입니다.
- 새로 만든 `weights.pt`가 실제로 갱신됐는지는 파일 크기·수정시각만으로 확신하지 마세요 —
  체크포인트마다 파라미터 수가 같아 파일 크기가 우연히 같을 수 있습니다. 필요하면 아래로
  이전 것과 다른 파일인지 확인합니다:
  ```bash
  python3 -c "import zlib; print(hex(zlib.crc32(open('export/music_classifier/weights.pt','rb').read())))"
  ```

## 2. `baseline_submit/model/music_classifier/` 갱신

export 결과를 `baseline_submit/model/music_classifier/`로 복사(덮어쓰기)합니다:

```bash
cp export/music_classifier/weights.pt \
   ../../baseline_submit/model/music_classifier/weights.pt
cp export/music_classifier/music_fake_infer.py \
   ../../baseline_submit/model/music_classifier/music_fake_infer.py
```

## 3. `script.py` 동기화

`baseline_submit/script.py`가 최신 `patched_script.py`와 같은 내용인지 diff로 확인하고,
다르면 통째로 교체합니다:

```bash
diff ../../baseline_submit/script.py patched_script.py
# 다르면:
cp patched_script.py ../../baseline_submit/script.py
```

> ⚠️ **알려진 함정 (2026-09-22 수정됨)**: 한때 `main()`이 `args = parse_arguments([])`로
> 빈 리스트를 넘겨 `--test-dir`/`--sample-submission`/`--output`/`--device` 등
> **모든 CLI 인자를 무시하고 하드코딩된 기본값만 쓰는 버그**가 있었습니다. 지금은
> `args = parse_arguments()`로 고쳐져 있습니다. `patched_script.py`를 손으로 다시 수정할
> 일이 있으면 이 줄이 되돌아가지 않았는지 반드시 확인하세요.

## 4. `requirements.txt`

수정 불필요합니다. 음악 분류기가 요구하는 패키지는 평가 서버에 이미 있는
torch/torchaudio뿐입니다.

## 5. 로컬 스모크 테스트

압축하기 전에 더미 3개 파일로 전체 파이프라인이 에러 없이 도는지, 그리고 **CLI 인자가
실제로 반영되는지** 확인합니다 (출력 경로를 기본값이 아닌 곳으로 지정해서, 그 경로에
파일이 생기는지 보면 3번의 인자 무시 버그도 함께 잡을 수 있습니다):

```bash
cd ../../baseline_submit
../.venv/bin/python3 script.py \
    --test-dir ../data/test \
    --sample-submission ../data/sample_submission.csv \
    --output /tmp/submission_smoketest.csv \
    --device cuda
cat /tmp/submission_smoketest.csv
```

- `TEST_0000`~`TEST_0002` 3행이 `sample_submission.csv`와 동일한 컬럼으로 출력되면 정상입니다.
- 에러가 나거나, `/tmp/submission_smoketest.csv`가 생기지 않고 대신
  `baseline_submit/output/submission.csv`에 생겼다면 인자가 무시되고 있다는 뜻이니
  3번의 함정을 다시 확인하세요.

## 6. 압축

`baseline_submit/` **폴더 자체가 아니라 그 안의 내용물**이 zip 최상위에 오도록
압축합니다:

```
submit.zip
├── model/            (df_arena_1b, htdemucs, panns, music_classifier)
├── script.py
└── requirements.txt
```

`baseline_submit/`이라는 폴더가 통째로 최상위에 들어가면(예:
`submit.zip/baseline_submit/model/...`) **설치 오류**로 처리됩니다. `baseline_submit/`
디렉토리 안에서 그 내용물만 선택해 압축하세요.

## 7. 압축 후 확인

- zip 용량 ≤ 10GB, 압축 해제 후 ≤ 32GB
- 가능하면 방금 만든 zip을 임시 폴더에 풀어서 최상위 항목이 `model/`, `script.py`,
  `requirements.txt` 3개뿐인지 한 번 더 확인

---

이 절차의 배경(왜 music_classifier로 교체하는지, 평가 산식 등)은 프로젝트 루트의
`README.md`(§4, §10)를 참고하세요. 이 문서는 그중 "제출 파일 만들기" 실행 절차만
독립적으로 뽑아둔 것입니다.
