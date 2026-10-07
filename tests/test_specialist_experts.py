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


def test_parametric_fq_pack_is_fixed_and_every_target_is_reachable():
    rows = experts.fq_parametric_rows()
    assert len(rows) == 73 and rows == experts.fq_parametric_rows()
    assert len({(r["target_x_m"], r["target_y_m"], r["target_heading_deg"], r["wheel_radius_m"],
                 r["wheelbase_m"]) for r in rows}) == 73
    committed = experts.read_pack(experts.FQ_PACK_V2)
    assert committed == rows

    def solver(prompt, max_tokens):
        # Closed-form command for the single arc or segment the task describes.
        row = next(r for r in rows if r["prompt"] in prompt)
        radius, base = float(row["wheel_radius_m"]), float(row["wheelbase_m"])
        x, y, heading = float(row["target_x_m"]), float(row["target_y_m"]), math.radians(float(row["target_heading_deg"]))
        duration = 5.0
        if heading == 0:
            speed, omega = x / duration, 0.0
        else:
            omega = heading / duration
            speed = omega * x / math.sin(heading)
        return json.dumps({"left_rad_s": (speed - omega * base / 2) / radius,
                           "right_rad_s": (speed + omega * base / 2) / radius, "duration_s": duration})

    assert suite._run_fq(solver, rows)["accuracy"] == 1.0


def test_http_complete_retries_a_transient_error_and_raises_a_persistent_one(monkeypatch):
    import io
    from urllib.error import HTTPError

    calls = []

    def flaky(request, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise HTTPError(request.full_url, 502, "Bad Gateway", None, None)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

    monkeypatch.setattr(experts, "urlopen", flaky)
    assert experts.http_complete("http://x", "m", backoff_sec=0)("q", 8) == "ok" and len(calls) == 3

    def down(request, timeout):
        raise HTTPError(request.full_url, 502, "Bad Gateway", None, None)

    monkeypatch.setattr(experts, "urlopen", down)
    with pytest.raises(HTTPError):
        experts.http_complete("http://x", "m", attempts=2, backoff_sec=0)("q", 8)


def test_v3_pack_states_the_duration_limit_and_keeps_the_v2_targets():
    v2, v3 = experts.fq_parametric_rows(), experts.fq_parametric_rows_v3()
    assert experts.read_pack(experts.FQ_PACK_V3) == v3
    for old, new in zip(v2, v3):
        assert new["prompt"] == old["prompt"] + experts.FQ_LIMIT_SENTENCE
        assert "at most 10 seconds" in new["prompt"] and new["max_duration_s"] == "10"
        assert {k: v for k, v in new.items() if k not in ("id", "prompt")} == \
               {k: v for k, v in old.items() if k not in ("id", "prompt")}
