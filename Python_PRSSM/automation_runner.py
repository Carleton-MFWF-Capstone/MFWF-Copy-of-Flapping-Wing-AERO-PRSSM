from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from experiment_runner import (
    DEFAULT_AUTOMATION_SHEET,
    CURRENT_RUN_STATUS_FILE,
    DEFAULT_EXCEL_PATH,
    DEFAULT_TUNING_RANDOM_SEED,
    STOP_REQUEST_FILE,
    propose_next_run,
    run_automation_cycle,
)


def main() -> None:
    stop_state = {"requested": False}

    def handle_interrupt(signum, frame):  # noqa: ARG001
        if not stop_state["requested"]:
            stop_state["requested"] = True
            print(
                "\nStop requested. The automation will finish the current run, "
                "save it, and then exit before starting a new one."
            )
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_interrupt)

    parser = argparse.ArgumentParser(
        description=(
            "Run surrogate-guided PRSSM tuning experiments. "
            "Each cycle proposes a configuration, launches training/postprocessing, "
            "and appends the results to the comparison workbook automation sheet. "
            "The currently active run configuration is also written beside the workbook."
        )
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=DEFAULT_EXCEL_PATH,
        help="Path to the comparison workbook used for recommendations and logging.",
    )
    parser.add_argument(
        "--sheet-name",
        default=None,
        help="Workbook sheet name to analyze. Defaults to the workbook's first sheet.",
    )
    parser.add_argument(
        "--strategy",
        default="best_balanced_accuracy_recommendation",
        choices=[
            "best_accuracy_recommendation",
            "best_conservative_recommendation",
            "best_balanced_accuracy_recommendation",
        ],
        help="Which recommendation from the optimizer to execute. The default prioritizes the highest predicted accuracy within the low-risk subset.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1,
        help="How many recommendation-run-log cycles to execute.",
    )
    parser.add_argument(
        "--exclude-overtraining",
        action="store_true",
        help="Ignore previously logged overtraining rows when generating recommendations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the next recommended configuration without launching training.",
    )
    parser.add_argument(
        "--allowed-risks",
        nargs="+",
        default=["low"],
        help="Only launch runs whose predicted overtraining risk is in this list. The default keeps the loop in the low-risk region.",
    )
    parser.add_argument(
        "--exploratory-runs",
        type=int,
        default=5,
        help="How many opening runs should use an exploration-focused recommendation strategy. Default is 5 so a new fixed-seed campaign gathers broader evidence before narrowing in on the highest low-risk accuracy recommendation.",
    )
    parser.add_argument(
        "--exploratory-strategy",
        default="best_accuracy_recommendation",
        choices=[
            "best_accuracy_recommendation",
            "best_conservative_recommendation",
            "best_balanced_accuracy_recommendation",
        ],
        help="Recommendation strategy to use during the exploratory opening phase.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_TUNING_RANDOM_SEED,
        help=(
            "Fixed random seed to use during hyperparameter tuning runs. "
            "Use --fresh-randomness to disable fixed seeding."
        ),
    )
    parser.add_argument(
        "--fresh-randomness",
        action="store_true",
        help="Disable fixed seeding so each run uses fresh randomness.",
    )
    args = parser.parse_args()

    base_overrides = {}
    if not args.fresh_randomness:
        base_overrides["random_seed"] = args.random_seed

    if args.dry_run:
        requested_strategy = (
            args.exploratory_strategy
            if args.exploratory_runs > 0
            else args.strategy
        )
        proposal = propose_next_run(
            excel_path=args.excel_path,
            sheet_name=args.sheet_name,
            strategy=requested_strategy,
            exclude_overtraining=args.exclude_overtraining,
            base_overrides=base_overrides or None,
        )
        payload = {
            "strategy": requested_strategy,
            "default_strategy": args.strategy,
            "exploratory_runs": args.exploratory_runs,
            "exploratory_strategy": args.exploratory_strategy,
            "fixed_random_seed": None if args.fresh_randomness else args.random_seed,
            "automation_sheet": DEFAULT_AUTOMATION_SHEET,
            "resolved_logging_sheet": proposal["resolved_sheet_name"],
            "current_run_status_file": CURRENT_RUN_STATUS_FILE,
            "stop_request_file": STOP_REQUEST_FILE,
            "recommendation": proposal["recommendation"],
            "overrides": proposal["overrides"],
        }
        print(json.dumps(payload, indent=2))
        return

    print("automation_runner: starting run_automation_cycle", flush=True)
    results = run_automation_cycle(
        max_runs=args.max_runs,
        excel_path=args.excel_path,
        sheet_name=args.sheet_name,
        strategy=args.strategy,
        exploratory_runs=args.exploratory_runs,
        exploratory_strategy=args.exploratory_strategy,
        exclude_overtraining=args.exclude_overtraining,
        base_overrides=base_overrides or None,
        stop_requested=lambda: stop_state["requested"],
        allowed_risks=args.allowed_risks,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
