#!/usr/bin/env python3
"""delete_log.py — SLURM job ID로 흩어진 로그 파일을 찾아 일괄 삭제.

사용법:
    python delete_log.py --id 128747
    python delete_log.py --id 128747 128748     # 여러 ID
    python delete_log.py --id 128747 --dry-run  # 미리보기(삭제 안 함)
"""
import argparse
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def find_logs(job_id: str):
    """
    logs/ 하위에서 *-<id>.out 에 매칭되는 파일을 모두 찾는다.
    """
    return sorted(LOGS_DIR.rglob(f"*-{job_id}.out"))


def main():
    parser = argparse.ArgumentParser(
        description="SLURM job ID의 로그(slurm/vllm/output)를 logs/ 전체에서 찾아 삭제"
    )
    parser.add_argument(
        "--id", dest="ids", nargs="+", required=True,
        help="삭제할 SLURM job ID (공백으로 여러 개 지정 가능)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="삭제하지 않고 대상 파일만 출력",
    )
    args = parser.parse_args()

    if not LOGS_DIR.exists():
        print(f"[delete_log] logs 디렉토리가 없음: {LOGS_DIR}")
        return

    deleted = 0
    for job_id in args.ids:
        matches = find_logs(job_id)
        if not matches:
            print(f"[delete_log] id={job_id}: 일치하는 로그 없음")
            continue
        print(f"[delete_log] id={job_id}: {len(matches)}개 발견")
        for f in matches:
            rel = f.relative_to(LOGS_DIR.parent)
            if args.dry_run:
                print(f"  [dry-run] {rel}")
            else:
                f.unlink()
                print(f"  deleted  {rel}")
                deleted += 1

    if args.dry_run:
        print("[delete_log] dry-run: 삭제하지 않음")
    else:
        print(f"[delete_log] 총 {deleted}개 삭제 완료")


if __name__ == "__main__":
    main()
