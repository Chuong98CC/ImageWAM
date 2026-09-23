"""The LIBERO rollout MP4 is opt-in (`EVALUATION.save_rollout_video`).

Saving a replay used to be unconditional: every episode held ~730 two-camera
frames in RAM and encoded them to a 256x512 MP4, which is the dominant
non-inference cost of a LIBERO eval and is not read back by
`summarize_results.py`. The flag therefore defaults off in every config the
eval entrypoints can select, and turning it off must also skip the per-step
frame accumulation -- otherwise the RAM cost survives the flag.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "experiments" / "libero"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import eval_libero_single as ev  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"
EVAL_CONFIGS = ["sim_libero.yaml", "sim_libero_omnigen2.yaml"]


def _obs():
    return {
        "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
    }


class _FakeEnv:
    """Stands in for OffScreenRenderEnv; finishes after `steps_until_done` steps."""

    def __init__(self, steps_until_done=2):
        self._left = steps_until_done

    def reset(self):
        pass

    def set_init_state(self, state):
        return _obs()

    def step(self, action):
        self._left -= 1
        return _obs(), 0.0, self._left <= 0, {}


def _cfg(**evaluation):
    base = {
        "task_suite_name": "libero_goal",
        "task_id": 0,
        "replan_steps": 2,
        "num_steps_wait": 0,
        "use_action_ensembler": False,
        "visualize_future_video": False,
        "save_rollout_video": False,
        "num_trials": 1,
    }
    base.update(evaluation)
    return OmegaConf.create(
        {
            "seed": 0,
            "data": {"train": {"num_frames": 6, "action_video_freq_ratio": 1}},
            "EVALUATION": base,
        }
    )


def _fake_action_chunk(**_kwargs):
    # `_predict_action_chunk` is replaced wholesale, so this skips denorm/gripper
    # handling and only has to satisfy the `.tolist()` slicing downstream. The
    # images still go through `get_libero_image` so frames match the real shape.
    return np.zeros((1, 7), dtype=np.float32), ev.get_libero_image(_obs()), None


@pytest.mark.parametrize("config_name", EVAL_CONFIGS)
def test_configs_default_to_no_rollout_video(config_name):
    cfg = OmegaConf.load(CONFIG_DIR / config_name)
    assert cfg.EVALUATION.save_rollout_video is False
    assert cfg.EVALUATION.visualize_future_video is False


def test_episode_skips_frame_accumulation_when_disabled(monkeypatch):
    monkeypatch.setattr(ev, "_predict_action_chunk", _fake_action_chunk)
    done, replay_images, _, _ = ev.run_single_episode(
        env=_FakeEnv(),
        initial_state=None,
        task_description="pick up the bowl",
        model=None,
        processor=None,
        cfg=_cfg(save_rollout_video=False),
        episode_idx=0,
        action_horizon=1,
        input_w=4,
        input_h=4,
        model_device="cpu",
    )
    assert done is True
    assert replay_images == []


def test_episode_accumulates_frames_when_enabled(monkeypatch):
    monkeypatch.setattr(ev, "_predict_action_chunk", _fake_action_chunk)
    _, replay_images, _, _ = ev.run_single_episode(
        env=_FakeEnv(),
        initial_state=None,
        task_description="pick up the bowl",
        model=None,
        processor=None,
        cfg=_cfg(save_rollout_video=True),
        episode_idx=0,
        action_horizon=1,
        input_w=4,
        input_h=4,
        model_device="cpu",
    )
    # One frame per executed step, both from the replan branch and the
    # mid-chunk branch that reads the observation directly.
    assert len(replay_images) == 2
    assert all(set(frame) == {"image", "wrist_image"} for frame in replay_images)


@pytest.mark.parametrize("save_rollout_video", [False, True])
def test_task_writes_video_only_when_enabled(monkeypatch, tmp_path, save_rollout_video):
    written = []
    monkeypatch.setattr(ev, "save_rollout_video", lambda *a, **kw: written.append(a))
    monkeypatch.setattr(ev, "get_libero_env", lambda *a, **kw: (_FakeEnv(), "task"))
    monkeypatch.setattr(
        ev, "run_single_episode", lambda **kw: (True, [{"image": _obs()["agentview_image"]}], [], None)
    )
    video_dir = tmp_path / "videos"

    results = ev.run_single_task(
        task=None,
        initial_states=[None],
        model=None,
        processor=None,
        cfg=_cfg(save_rollout_video=save_rollout_video),
        video_dir=video_dir,
        predicted_video_dir=tmp_path / "predicted_videos",
        action_horizon=1,
        input_w=4,
        input_h=4,
        model_device="cpu",
    )

    assert results["successes"] == 1
    assert len(written) == int(save_rollout_video)
