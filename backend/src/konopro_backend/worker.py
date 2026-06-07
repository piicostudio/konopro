import argparse
import sys

from konopro_backend.config import BackendSettings
from konopro_backend.jobs import run_next_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Konopro backend jobs.")
    parser.add_argument("--once", action="store_true", help="Run at most one queued job and exit.")
    args = parser.parse_args(argv)

    if not args.once:
        parser.error("Only --once is supported in Phase 04.")

    job = run_next_job(BackendSettings())
    if job is None:
        print("No queued jobs found.")
        return 0

    print(f"Processed job {job.id}: {job.status.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

