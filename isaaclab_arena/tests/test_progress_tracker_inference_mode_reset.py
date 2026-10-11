# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A tracker and the rest-pose recorder used under torch.inference_mode reset outside it."""

from types import SimpleNamespace


def test_reset_after_an_inference_mode_step():
    import torch

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = SimpleNamespace(done=torch.tensor([True, True]))
    criteria = CompletionCriteria(name="done", predicate_sequence=[lambda env: env.done.clone()])
    tracker = ProgressTracker([criteria], num_envs=2, device="cpu", env=env)
    with torch.inference_mode():
        tracker.step(env)
    assert tracker.is_complete().tolist() == [True, True]

    tracker.reset([0])  # a reset event or a script's env.reset(), outside inference mode

    assert tracker.is_complete().tolist() == [False, True]
    assert tracker.get_state()[0].criteria_by_name["done"].completed_sequences == 0
    env.done = torch.tensor([False, True])
    tracker.step(env)
    assert tracker.is_complete().tolist() == [False, True]


def test_rest_pose_recorder_reset_after_an_inference_mode_step():
    import math

    import torch

    from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder

    recorder = ObjectInitialRestPoseRecorder(num_envs=2, device="cpu")
    with torch.inference_mode():
        recorder.get("lid")  # created inside the step, as objects_settled's first call does
        recorder.record("box", torch.ones(2, 3), torch.tensor([True, True]))

    recorder.reset([0])  # the env's _reset_idx, outside inference mode

    position, settled = recorder.get("box")
    assert settled.tolist() == [False, True]
    assert math.isnan(position[0, 0].item()) and position[1].tolist() == [1.0, 1.0, 1.0]
    recorder.record("box", torch.full((2, 3), 2.0), torch.tensor([True, True]))
    assert recorder.get("box")[0][0].tolist() == [2.0, 2.0, 2.0]
