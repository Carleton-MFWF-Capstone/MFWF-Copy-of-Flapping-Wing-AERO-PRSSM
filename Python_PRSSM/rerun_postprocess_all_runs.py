from __future__ import annotations

import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_OUTPUT_ROOT = REPO_ROOT.parent / "Data" / "Output"
CURRENT_RUN_STATUS_FILE = DATA_OUTPUT_ROOT / "current_automation_run.json"
STOP_REQUEST_FILE = DATA_OUTPUT_ROOT / "stop_automation_after_current_run.txt"


def iter_completed_run_dirs(output_root: Path):
    for path in sorted(output_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("R") and path.name[1:].isdigit():
            # Skip active temporary directories; completed runs should have been renamed.
            continue
        if not (path / "training_config.json").exists():
            continue
        if not (path / "matfiles" / "predict_train_n_test.mat").exists():
            continue
        yield path


def wait_for_loop_to_finish(poll_seconds: int = 60) -> None:
    print("Waiting for the automation loop to finish the current run...")
    while CURRENT_RUN_STATUS_FILE.exists():
        time.sleep(poll_seconds)
    print("No active automation status file found. Continuing with postprocessing rerun.")


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "Python_PRSSM"))
    from outputs.postprocess_model_outputs import run_for_output_dir

    if STOP_REQUEST_FILE.exists():
        wait_for_loop_to_finish()

    processed = 0
    skipped = 0
    for out_dir in iter_completed_run_dirs(DATA_OUTPUT_ROOT):
        try:
            print(f"Reprocessing: {out_dir}")
            run_for_output_dir(out_dir, show_plots=False)
            processed += 1
        except Exception as exc:  # pragma: no cover
            skipped += 1
            print(f"Skipping {out_dir}: {exc}")

    print(f"Completed rerun for {processed} output folders. Skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
