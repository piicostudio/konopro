import argparse
import sys
import time

from konopro_backend.config import BackendSettings
from konopro_backend.jobs import run_next_job


def run_once(settings: BackendSettings) -> bool:
    job = run_next_job(settings)
    if job is None:
        print("No queued jobs found.")
        return False

    print(f"Processed job {job.id}: {job.status.value}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Konopro backend jobs.")
    parser.add_argument("--once", action="store_true", help="Run at most one queued job and exit.")
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=5.0,
        help="Seconds to wait before checking again when no queued job exists.",
    )
    parser.add_argument(
        "--max-empty-polls",
        type=int,
        default=None,
        help="Exit after this many empty polls. Intended for tests and short demos.",
    )
    args = parser.parse_args(argv)

    settings = BackendSettings()
    if args.once:
        run_once(settings)
        return 0

    empty_polls = 0
    print("Worker polling for queued jobs. Press Ctrl+C to stop.")
    try:
        while True:
            processed_job = run_once(settings)
            if processed_job:
                empty_polls = 0
                continue

            empty_polls += 1
            if args.max_empty_polls is not None and empty_polls >= args.max_empty_polls:
                return 0

            time.sleep(args.poll_interval_s)
    except KeyboardInterrupt:
        print("Worker stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
