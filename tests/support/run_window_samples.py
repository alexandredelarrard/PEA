"""Direct complete-sample worker; the same source checks without pytest collection.

The existing timed launcher includes interpreter startup. This sole worker clock
starts before project imports and is passed unchanged through Context and every
check. There is no budget, sample-size or preparation override.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

_PROCESS_STARTED = time.perf_counter()

# These are module-level bootstrap imports: their cost belongs to this deadline.
from src.validate.checks.window_sample import timing_sample_tables  # noqa: E402
from tests import live_guard  # noqa: E402
from tests.support.window_lineage import measure_window_lineage  # noqa: E402
from tests.support.window_samples import assert_lineage_results, window_sample_session  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-baseline", required=True, type=Path)
    parser.add_argument("--window-out", required=True, type=Path)
    args = parser.parse_args()
    live_guard.install()
    errors = []
    with window_sample_session(args.window_baseline, args.window_out, False, _PROCESS_STARTED) as run:
        for table in timing_sample_tables():
            result = run.measure(table)
            inactive = run.manifest["tables"].get(table.name, {}).get("applicability") == "inactive"
            if result.status != "pass" and not inactive:
                errors.append(result.summary())
            print(f"SANITY: scalar {table.name}: {result.status}; {'declared inactive' if inactive else 'original pinned-source checks'}.")
        results = measure_window_lineage(run)
        try:
            assert_lineage_results(results, run.planning)
        except AssertionError as error:
            errors.append(str(error))
        if live_guard.TRIPS:
            errors.append(f"Live tripwire attempts: {len(live_guard.TRIPS)}")
        failed = any(result["status"] == "fail" for group in (run.results, run.lineage, run.planning) for result in group.values())
        status = 1 if failed or live_guard.TRIPS else 3 if errors else 0
        print(
            f"SANITY: complete source/selector/assembly/planning checks finished together; tripwire attempts={len(live_guard.TRIPS)}; exit={status}."
        )
        for error in errors:
            print(error)
        return status


if __name__ == "__main__":
    raise SystemExit(main())
