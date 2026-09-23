"""Tests for the overfit check's episode -> LIBERO task resolver.

The invariant that matters: a LeRobot dataset's `meta/tasks.jsonl` index is NOT
the LIBERO benchmark `task_id`. For libero_spatial the two orderings are a
permutation (`[0, 2, 4, 6, 8, 1, 3, 5, 7, 9]`), so resolving by index produces a
task file that points at the wrong task -- one that still evaluates cleanly and
returns a plausible number, which is exactly the kind of failure this script
exists to catch. These tests pin the instruction-string join.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "flux2"))

from overfit_episodes_to_tasks import (  # noqa: E402
    map_instructions_to_task_ids,
    read_episode_instructions,
    write_task_file,
)

SPATIAL_LANGUAGES = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]

# The order `meta/tasks.jsonl` stores them in, mapped to benchmark task_id.
# Verified against the real libero_spatial suite.
TASKS_JSONL_ORDER = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]


def _benchmark_index():
    return {text: task_id for task_id, text in enumerate(SPATIAL_LANGUAGES)}


def _write_dataset(root: Path, episode_task_ids: list[int]) -> Path:
    """A minimal LeRobot meta/ dir.

    `episode_task_ids[i]` is the *benchmark* task id of episode `i`. The
    instructions are therefore written into `tasks.jsonl` in the scrambled
    `TASKS_JSONL_ORDER`, so the row position of an instruction is generally not
    its benchmark task id -- which is the whole hazard under test.
    """
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "episodes.jsonl").open("w") as fh:
        for ep_idx, task_id in enumerate(episode_task_ids):
            fh.write(json.dumps({
                "episode_index": ep_idx,
                "tasks": [SPATIAL_LANGUAGES[task_id]],
                "length": 100,
            }) + "\n")
    with (meta / "tasks.jsonl").open("w") as fh:
        for position, task_id in enumerate(TASKS_JSONL_ORDER):
            fh.write(json.dumps({
                "task_index": position,
                "task": SPATIAL_LANGUAGES[task_id],
            }) + "\n")
    return root


def tasks_jsonl_position_of(task_id: int) -> int:
    """Row that an index-based resolver would mistake for `task_id`."""
    return TASKS_JSONL_ORDER.index(task_id)


# --------------------------------------------------------------- reading
def test_reads_the_requested_episodes_in_order(tmp_path):
    root = _write_dataset(tmp_path / "ds", [0, 1, 1])
    got = read_episode_instructions(root, [2, 0])
    assert got == [SPATIAL_LANGUAGES[1], SPATIAL_LANGUAGES[0]]


def test_missing_episode_raises(tmp_path):
    root = _write_dataset(tmp_path / "ds", [0, 0])
    with pytest.raises(KeyError):
        read_episode_instructions(root, [5])


# --------------------------------------------------------------- resolution
def test_resolves_by_instruction_not_by_tasks_jsonl_index(tmp_path):
    """Benchmark task 2 sits at row 1 of tasks.jsonl.

    This is the regression the module exists for: an index-based resolver would
    answer 1 here, and 1 is a real task in the suite, so the mistake would
    evaluate cleanly and report a plausible number.
    """
    assert tasks_jsonl_position_of(2) == 1  # the trap
    root = _write_dataset(tmp_path / "ds", [0, 2])
    instructions = read_episode_instructions(root, [1])
    assert map_instructions_to_task_ids(instructions, _benchmark_index()) == [2]


def test_episodes_sharing_a_task_collapse_to_one_id(tmp_path):
    """The four overfit episodes are all task 0, so the task file has one line."""
    root = _write_dataset(tmp_path / "ds", [0, 0, 0, 0])
    instructions = read_episode_instructions(root, [0, 1, 2, 3])
    assert map_instructions_to_task_ids(instructions, _benchmark_index()) == [0]


def test_multiple_tasks_come_back_sorted_and_unique(tmp_path):
    root = _write_dataset(tmp_path / "ds", [2, 0, 2, 0])
    instructions = read_episode_instructions(root, [0, 1, 2, 3])
    assert map_instructions_to_task_ids(instructions, _benchmark_index()) == [0, 2]


def test_unmatched_instruction_raises_and_names_it():
    """A silent drop would evaluate a subset of the trained tasks and look fine."""
    with pytest.raises(ValueError) as excinfo:
        map_instructions_to_task_ids(["a task that is not in the suite"], _benchmark_index())
    assert "a task that is not in the suite" in str(excinfo.value)


def test_whitespace_differences_still_match():
    with pytest.raises(ValueError):
        map_instructions_to_task_ids(["  unrelated  "], _benchmark_index())


# ------------------------------------------------- the driver's episode filter
def _prefix_selection(episode_count: int, keep: int) -> list[int]:
    """Reproduce the override `run_overfit_check_*.sh` builds, against the real filter.

    The driver sets `period = episode_count + 1` so that `ep % period == ep` for
    every real episode and the periodic rule collapses to a plain prefix.
    """
    from imagewam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset

    stub = object.__new__(BaseLerobotDataset)  # only episode_index_filter is read
    stub.episode_index_filter = {
        "mode": "periodic_prefix",
        "period": episode_count + 1,
        "keep_first": keep,
    }
    return stub._filter_episode_indices(list(range(episode_count)), repo_id="stub")


@pytest.mark.parametrize(
    "episode_count,keep",
    [(434, 4), (434, 1), (388, 3), (457, 4), (10, 10), (1, 1)],
)
def test_prefix_selection_is_exactly_the_first_n(episode_count, keep):
    assert _prefix_selection(episode_count, keep) == list(range(keep))


def test_a_small_period_wraps_and_leaks_extra_episodes():
    """Why the driver derives `period = count + 1` instead of reusing keep_first.

    The filter is periodic, not first-N: any period <= the episode count keeps
    re-selecting episodes further down the suite. A driver that set period to N
    would quietly train on far more than N episodes -- the exact opposite of an
    overfit check -- while still looking like it worked.
    """
    from imagewam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset

    stub = object.__new__(BaseLerobotDataset)
    stub.episode_index_filter = {"mode": "periodic_prefix", "period": 10, "keep_first": 2}
    assert stub._filter_episode_indices(list(range(30)), repo_id="stub") == [0, 1, 10, 11, 20, 21]

    # With the driver's period the same keep_first is a true prefix.
    assert _prefix_selection(episode_count=30, keep=2) == [0, 1]


# --------------------------------------------------------------- output
def test_task_file_format(tmp_path):
    out = write_task_file(tmp_path / "tasks.txt", "libero_spatial", [0, 2])
    assert out.read_text() == "libero_spatial,0\nlibero_spatial,2\n"


def test_task_file_single_task(tmp_path):
    out = write_task_file(tmp_path / "tasks.txt", "libero_spatial", [0])
    assert out.read_text() == "libero_spatial,0\n"
