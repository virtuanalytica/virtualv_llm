"""Tool experts: the loop corrects with tool feedback and never sees an answer key."""

import json
import math
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
import specialist_experts as experts  # noqa: E402
import specialist_suite as suite  # noqa: E402


def test_differential_drive_matches_closed_form():
    assert suite.differential_drive(4, 4, 5, 0.05, 0.30) == pytest.approx((1.0, 0.0, 0.0))
    # Quarter circle of radius 0.45 m: both wheels forward, right faster.
    omega = math.pi / 2 / 5
    left, right = (0.45 - 0.15) * omega / 0.05, (0.45 + 0.15) * omega / 0.05
    assert suite.differential_drive(left, right, 5, 0.05, 0.30) == pytest.approx((0.45, 0.45, 90.0))


def scripted(answers):
    prompts = []

    def complete(prompt, max_tokens):
        prompts.append(prompt)
        return answers[min(len(prompts), len(answers)) - 1]
    return complete, prompts


def command(left, right, duration):
    return json.dumps({"left_rad_s": left, "right_rad_s": right, "duration_s": duration,
                       "wheel_radius_m": 0.05, "wheelbase_m": 0.30})


def test_fq_expert_feeds_back_the_simulated_pose_and_keeps_the_correction():
    complete, prompts = scripted([command(2, 2, 5), command(4, 4, 5), command(4, 4, 5)])
    rounds = []
    answer = experts.fq_expert(complete, rounds)("Reach (1.00 0.00) at heading 0 degrees.", 256)
    assert json.loads(answer)["left_rad_s"] == 4
    assert "x=0.500 m" in prompts[1] and "x=1.000 m" in prompts[2]
    assert rounds == [3]


def test_fq_expert_stops_when_the_model_confirms_its_command():
    complete, prompts = scripted([command(4, 4, 5), command(4, 4, 5)])
    rounds = []
    experts.fq_expert(complete, rounds)("Reach (1.00 0.00).", 256)
    assert rounds == [2] and len(prompts) == 2


def test_fq_expert_reports_unparseable_output_instead_of_guessing():
    complete, prompts = scripted(["I think 4 rad/s", command(4, 4, 5), command(4, 4, 5)])
    experts.fq_expert(complete)("Reach (1.00 0.00).", 256)
    assert "not the requested JSON" in prompts[1]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_video_expert_repairs_a_graph_from_ffmpeg_error():
    good = "drawbox=x=100:y=100:w=200:h=200:color=red:t=fill,noise=alls=20:allf=t"
    complete, prompts = scripted(["nosuchfilter=1", good])
    rounds = []
    answer = experts.video_expert(complete, rounds)("Return only a filter_complex expression.", 512)
    assert answer == good and rounds == [2]
    assert "nosuchfilter" in prompts[1] and "did not render" in prompts[1]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_video_expert_gives_up_after_the_round_limit():
    complete, prompts = scripted(["nosuchfilter=1"])
    rounds = []
    experts.video_expert(complete, rounds)("Return only a filter_complex expression.", 512)
    assert rounds == [experts.MAX_ROUNDS] and len(prompts) == experts.MAX_ROUNDS
