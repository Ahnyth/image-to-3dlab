"""Tests for the Hero chain's bookkeeping (`image_to_3dlab/hero_chain.py`, `scripts/hero_chain.py`).

A stage of the chain changes every model of a figure and is kept only when its check passes. These tests hold the parts
that decide that without running a stage: the finer models found in and swapped into the game-ready options, the region
a stage changed, the colour outside it, the verdicts and the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab import colour_compare as cc
from image_to_3dlab import figure_check as fc
from image_to_3dlab import hero_chain as ch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

GAME = ["--height", "2.2", "--colour-from", "C:/a/paint.glb", "--normal-from=C:/a/raw.glb", "--decimate-to", "14000",
        "--mesh-from", "C:/a/fine.glb", "--feet", "cut", "--drop-hidden"]


def test_the_finer_models_come_from_the_game_options_and_are_swapped_there():
    found = ch.game_models(GAME)
    assert found == {"--colour-from": Path("C:/a/paint.glb"), "--normal-from": Path("C:/a/raw.glb"),
                     "--mesh-from": Path("C:/a/fine.glb")}
    models = ch.Models(Path("C:/a/rig.glb"), found)
    assert models.colour() == Path("C:/a/paint.glb") and len(models.also()) == 3
    after = models.swapped({"C:/a/paint.glb": "C:/b/paint_hands.glb", "C:/a/rig.glb": "C:/b/rig_hands.glb"})
    assert after.rig == Path("C:/b/rig_hands.glb") and after.finer["--normal-from"] == Path("C:/a/raw.glb")
    args = after.game_args(GAME)
    assert ch.option_value(args, "--colour-from") == str(Path("C:/b/paint_hands.glb"))
    assert ch.option_value(args, "--normal-from") == str(Path("C:/a/raw.glb"))
    assert ch.option_value(args, "--decimate-to") == "14000" and "--drop-hidden" in args
    assert ch.option_value(ch.with_option(args, "--feet", "off"), "--feet") == "off"
    assert ch.option_value(ch.with_option(args, "--feet", None), "--feet") is None


def test_without_a_colour_source_the_colour_comes_from_the_rig():
    assert ch.Models(Path("rig.glb")).colour() == Path("rig.glb")


def _boxes(*corners):
    """Points on the surface grid of boxes (lo, hi)."""
    out = []
    for lo, hi in corners:
        g = np.stack(np.meshgrid(*[np.linspace(a, b, 12) for a, b in zip(lo, hi)]), -1).reshape(-1, 3)
        out.append(g)
    return np.concatenate(out)


def test_the_region_a_stage_changed_is_where_points_came_or_went_and_no_farther():
    body = ((-0.2, 0.0, -0.1), (0.2, 1.8, 0.1))
    old_hand = ((0.5, 0.9, -0.05), (0.6, 1.0, 0.05))
    new_hand = ((0.5, 0.85, -0.05), (0.62, 1.0, 0.05))
    inside, count = ch.changed_region(_boxes(body, old_hand), _boxes(body, new_hand))
    assert count > 0
    assert inside(np.array([[0.55, 0.95, 0.0], [0.55, 0.86, 0.0]])).all()          # the hand, old and new
    assert not inside(np.array([[0.0, 1.5, 0.1], [0.0, 0.2, 0.1]])).any()           # the body, far from it
    same, none = ch.changed_region(_boxes(body), _boxes(body))
    assert none == 0 and not same(np.array([[0.0, 1.0, 0.0]])).any()


def test_a_colour_change_inside_the_stages_region_passes_and_one_outside_fails():
    from test_colour_compare import model

    def back(p):                       # the stage changed the back
        return p[:, 2] < -0.4

    old, inside_change, outside_change = model(), model(dark_side=1), model(dark_side=0)
    patches_in = cc.compare(old, inside_change, leave_out=back)
    patches_out = cc.compare(old, outside_change, leave_out=back)
    assert ch.colour_verdict(patches_in, "the hands")[0] is True
    ok, why = ch.colour_verdict(patches_out, "the hands")
    assert ok is False and "outside the hands" in why and "front" in why


def test_fingers_pass_only_with_both_hands_counted_right():
    assert ch.fingers_verdict(fc.Result("fingers", "pass", "left hand 4, right hand 4", "", ""))[0] is True
    assert ch.fingers_verdict(fc.Result("fingers", "fail", "left hand 5, right hand 4", "", ""))[0] is False
    assert ch.fingers_verdict(fc.Result("fingers", "look", "left hand 3, right hand 4", "", ""))[0] is False
    assert ch.fingers_verdict(fc.Result("fingers", "pass", "left hand 4", "", ""))[0] is False
    assert ch.fingers_verdict(fc.Result("fingers", "info", "-", "", "not counted"))[0] is False


def test_a_stage_may_not_open_holes_in_motion_that_were_not_there_before():
    def report(left):
        return {"pose_check": {"hole_px_left": left}}

    assert ch.pose_verdict(report(0), report(0))[0] is True
    assert ch.pose_verdict(report(3), report(0))[0] is False
    assert ch.pose_verdict(report(3), None)[0] is False
    assert ch.pose_verdict(report(3), report(5))[0] is True
    assert ch.pose_verdict({}, report(0))[0] is None


def test_the_report_names_every_stage_and_what_became_of_it(tmp_path):
    stages = [ch.Stage("hands", "kept", "left hand 4, right hand 4 per cut", {"left": 4, "right": 4}, 12.0),
              ch.Stage("head", "dropped", "2 patch(es) changed colour outside the head", {"patches": 2}),
              ch.Stage("boots", "skipped", "no boots in the picture: shoes"),
              ch.Stage("band", "off", "not asked for")]
    html = ch.preface(stages, tmp_path)
    for word in ("hands", "kept", "dropped", "skipped", "not asked for", "shoes"):
        assert word in html
    ch.write_summary(tmp_path / "chain.json", stages, {"glb": "x.glb"})
    data = json.loads((tmp_path / "chain.json").read_text())
    assert [s["status"] for s in data["stages"]] == ["kept", "dropped", "skipped", "off"]


def test_the_script_plans_each_stages_command_from_the_models():
    import hero_chain

    models = ch.Models(Path("C:/f/rig.glb"), ch.game_models(GAME))
    hands = hero_chain.stage_command("hands", models, Path("C:/f/front.png"), Path("C:/o/1_hands"), {})
    assert Path(hands[1]).name == "hero_hands.py" and "--also" in hands
    assert {str(p) for p in models.also()} <= set(hands)
    head = hero_chain.stage_command("head", models, Path("C:/f/front.png"), Path("C:/o/2_head"), {})
    assert head[head.index("--region") + 1] == "auto"
    boots = hero_chain.stage_command("boots", models, Path("C:/f/front.png"), Path("C:/o/3_boots"), {})
    assert boots[boots.index("--top") + 1] == "auto" and "--boots" in boots
    given = hero_chain.stage_command("boots", models, Path("C:/f/front.png"), Path("C:/o/3_boots"), {"top": 1052.0})
    assert given[given.index("--top") + 1] == "1052.0"
    band = hero_chain.stage_command("band", models, Path("C:/f/front.png"), Path("C:/o/4_band"),
                                    {"views": Path("C:/f/views"), "boxes": [[1, 2, 3, 4]]})
    assert str(models.colour()) in band and band[band.index("--box") + 1:band.index("--box") + 5] == ["1", "2", "3",
                                                                                                    "4"]
    assert hero_chain.stage_results("hands", models, Path("C:/o/1_hands"))[models.rig] == Path(
        "C:/o/1_hands/rig_hands.glb")
    assert hero_chain.stage_results("band", models, Path("C:/o/4_band")) == {
        models.colour(): Path("C:/o/4_band/paint_band.glb")}


def test_the_script_splits_its_own_options_from_the_game_ready_ones():
    import hero_chain

    own, game = hero_chain.split_args(["rig.glb", "front.png", "--out", "o", "--", *GAME])
    assert own == ["rig.glb", "front.png", "--out", "o"] and game == GAME
    assert hero_chain.split_args(["rig.glb", "front.png"]) == (["rig.glb", "front.png"], [])


def test_a_gpu_lock_is_a_folder_taken_and_given_back(tmp_path):
    import hero_chain

    lock = tmp_path / "gpu.lock"
    with hero_chain.gpu_locks([lock], "test", wait=1):
        assert lock.is_dir() and "test" in (lock / "who").read_text()
        with pytest.raises(SystemExit):
            with hero_chain.gpu_locks([lock], "second", wait=0.2, step=0.05):
                pass
        assert lock.is_dir()                       # the second never took it, so it never removes it
    assert not lock.exists()


def test_the_models_kept_by_a_stage_are_read_back_as_they_were_written():
    models = ch.Models(Path("C:/f/rig.glb"), ch.game_models(GAME))
    again = ch.Models.from_dict(json.loads(json.dumps(models.as_dict())))
    assert again.rig == models.rig and again.finer == models.finer
