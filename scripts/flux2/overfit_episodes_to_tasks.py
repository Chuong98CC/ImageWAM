#!/usr/bin/env python
"""Resolve which LIBERO benchmark tasks a set of dataset episodes covers.

`run_libero_manager.py` selects what to evaluate from a task file of
`suite,task_id` lines (`MULTIRUN.task_file`). Getting those ids right is the
whole problem: a LeRobot dataset's `meta/tasks.jsonl` `task_index` is NOT the
LIBERO benchmark `task_id`. The two are a permutation of each other -- for
libero_spatial, `tasks.jsonl` order maps to benchmark ids
`[0, 2, 4, 6, 8, 1, 3, 5, 7, 9]`. Resolving by index therefore yields a task
file that points at the wrong task, which evaluates cleanly and reports a
plausible number. The instruction string is the only field both sides store
verbatim, so that is the join key used here.

Usage:
    python overfit_episodes_to_tasks.py --suite libero_spatial --episodes 4
    python overfit_episodes_to_tasks.py --suite libero_spatial --episodes 4 --out tasks.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def read_episode_instructions(dataset_dir: Path, episodes: Sequence[int]) -> list[str]:
    """Instruction text for each requested episode, in the order requested."""
    episodes_path = Path(dataset_dir) / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(
            f"No episode metadata at {episodes_path}. Is {dataset_dir} a LeRobot dataset?"
        )

    by_index: dict[int, str] = {}
    with episodes_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks = row.get("tasks") or []
            if not tasks:
                raise ValueError(f"Episode {row.get('episode_index')} carries no task string")
            by_index[int(row["episode_index"])] = str(tasks[0])

    missing = [ep for ep in episodes if ep not in by_index]
    if missing:
        raise KeyError(
            f"Episode(s) {missing} not present in {episodes_path} "
            f"(it holds {len(by_index)} episodes)."
        )
    return [by_index[int(ep)] for ep in episodes]


def map_instructions_to_task_ids(
    instructions: Iterable[str],
    language_to_task_id: Mapping[str, int],
) -> list[int]:
    """Unique benchmark task ids covering `instructions`, ascending.

    Both sides are stripped before comparison: the language strings come from
    JSON on one side and from Python literals in the LIBERO suite on the other,
    and a trailing space is not a meaningful difference. An instruction with no
    counterpart raises rather than being dropped -- a silent drop would evaluate
    a strict subset of what was trained and still look successful.
    """
    normalized = {str(text).strip(): int(task_id) for text, task_id in language_to_task_id.items()}
    resolved: set[int] = set()
    for instruction in instructions:
        key = str(instruction).strip()
        if key not in normalized:
            raise ValueError(
                f"No LIBERO task in this suite has the instruction {instruction!r}. "
                "The dataset and the benchmark disagree, so any task id derived here "
                "would be wrong."
            )
        resolved.add(normalized[key])
    return sorted(resolved)


def load_benchmark_languages(suite_name: str) -> dict[str, int]:
    """`{instruction: task_id}` straight from the LIBERO benchmark suite."""
    try:
        from libero.libero import benchmark
    except ImportError as exc:  # pragma: no cover - depends on the venv
        raise RuntimeError(
            "The `libero` package is required to resolve task ids. Run this with "
            "the ImageWAM venv python (see .venv/bin/python)."
        ) from exc

    suites = benchmark.get_benchmark_dict()
    if suite_name not in suites:
        raise ValueError(
            f"Unknown LIBERO suite {suite_name!r}. Available: {sorted(suites)}"
        )
    suite = suites[suite_name]()
    return {str(suite.get_task(i).language).strip(): i for i in range(suite.n_tasks)}


def write_task_file(path: Path, suite: str, task_ids: Sequence[int]) -> Path:
    """Write the `suite,task_id` lines `MULTIRUN.task_file` expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{suite},{int(task_id)}\n" for task_id in task_ids))
    return path


def resolve_task_ids(suite: str, dataset_dir: Path, episodes: Sequence[int]) -> list[int]:
    """Episode indices -> the LIBERO benchmark tasks they belong to."""
    instructions = read_episode_instructions(dataset_dir, episodes)
    return map_instructions_to_task_ids(instructions, load_benchmark_languages(suite))


def default_dataset_dir(suite: str) -> Path:
    data_root = os.environ.get("DATA_ROOT")
    if not data_root:
        raise RuntimeError("DATA_ROOT is unset; pass --dataset-dir or export DATA_ROOT.")
    return Path(data_root) / f"{suite}_no_noops_lerobot"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True, help="e.g. libero_spatial")
    parser.add_argument(
        "--episodes", type=int, default=4,
        help="Keep the first N episodes of the suite (default: 4).",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="LeRobot dataset directory (default: $DATA_ROOT/<suite>_no_noops_lerobot).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the task file here.")
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir or default_dataset_dir(args.suite)
    episodes = list(range(args.episodes))
    task_ids = resolve_task_ids(args.suite, dataset_dir, episodes)

    lines = "".join(f"{args.suite},{task_id}\n" for task_id in task_ids)
    if args.out is not None:
        write_task_file(args.out, args.suite, task_ids)
        print(f"Wrote {args.out}:")
    print(lines, end="")
    print(
        f"# episodes {episodes} of {dataset_dir.name} -> "
        f"{len(task_ids)} task(s) in {args.suite}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
