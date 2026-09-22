#!/usr/bin/env python3
"""
prepare_data.py -> train.py -> evaluate.py -> export_for_submit.py 를
기본 옵션 그대로 한 번에 순서대로 실행하는 파이프라인 스크립트.

각 단계는 독립된 서브프로세스로, "이 스크립트를 실행한 파이썬 인터프리터"
그대로 실행한다 — 즉 torch가 설치된 올바른 venv의 python으로 이 스크립트
자체를 실행하기만 하면, 나머지 단계도 자동으로 같은 환경을 쓴다.

하나라도 실패(0이 아닌 종료 코드)하면 그 자리에서 즉시 멈춘다. 각 단계에
세부 옵션을 주고 싶으면(예: --limit, --epochs) 이 스크립트 대신 해당 스크립트를
개별 실행할 것 — 이 스크립트는 "그냥 기본값으로 전체를 한 번에 돌리는" 용도다.

사용 예:
    python run_pipeline.py
    python run_pipeline.py --skip-prepare   # data/manifests가 이미 최신이면 prepare_data 생략
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def check_torch_available():
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "torch를 찾을 수 없습니다. 이 디렉토리 자체의 .venv에는 torch가 없으니,\n"
            "프로젝트 루트 .venv의 python으로 이 스크립트를 실행하세요. 예(WSL):\n"
            "    ../../.venv/bin/python3 run_pipeline.py\n"
            "(자세한 내용은 SUBMIT_INTEGRATION.md 1번 항목 참고)"
        )
        sys.exit(1)


def run_step(name: str, script: str, extra_args: list[str] = None):
    extra_args = extra_args or []
    print(f"\n{'=' * 70}\n[{name}] 시작 ({script} {' '.join(extra_args)})\n{'=' * 70}")
    result = subprocess.run([sys.executable, script, *extra_args], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n[{name}] 실패 (exit code {result.returncode}) — 파이프라인 중단")
        sys.exit(result.returncode)
    print(f"[{name}] 완료")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="prepare_data.py 단계를 건너뛴다 (data/manifests가 이미 최신일 때)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    check_torch_available()

    if not args.skip_prepare:
        run_step("1/4 데이터 준비 (prepare_data.py)", "prepare_data.py")
    else:
        print("\n[1/4 데이터 준비] --skip-prepare 옵션으로 건너뜀")

    run_step("2/4 학습 (train.py)", "train.py")
    run_step("3/4 평가 (evaluate.py)", "evaluate.py")
    run_step("4/4 제출용 export (export_for_submit.py)", "export_for_submit.py")

    print(
        "\n파이프라인 전체 완료. 다음 단계는 SUBMIT_INTEGRATION.md 2번 항목부터"
        " (baseline_submit/model/music_classifier/ 갱신) 참고."
    )


if __name__ == "__main__":
    main()
