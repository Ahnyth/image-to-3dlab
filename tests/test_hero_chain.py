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


def test_a_rerun_keeps_a_stages_builds_when_only_the_stages_before_it_changed_the_models():
    origin = {"rig.glb": [1, 1], "front.png": [5, 1]}
    before = {"origin": origin, "files": {"rig.glb": [1, 1]}, "options": {"stage": "head"}}
    assert ch.rerun(None, before) == "fresh"
    assert ch.rerun(before, json.loads(json.dumps(before))) == "same"
    hands_kept = {"origin": origin, "files": {"out/1_hands/rig_hands.glb": [3, 2]}, "options": {"stage": "head"}}
    assert ch.rerun(before, hands_kept) == "rebuild"
    new_picture = {**hands_kept, "origin": {"rig.glb": [1, 1], "front.png": [6, 2]}}
    assert ch.rerun(before, new_picture) == "fresh"
    assert ch.rerun(before, {**before, "options": {"stage": "head", "given": {"top": 1}}}) == "fresh"


def test_the_game_step_runs_again_when_a_stage_rewrote_its_models(tmp_path, monkeypatch):
    # a rerun of the boots stage wrote new models under the same names, and the game step took its old figure with
    # the old boots, because its key held the paths only (skin_sorcerer_c, 2026-11-24)
    import hero_chain

    rig, paint = tmp_path / "rig.glb", tmp_path / "paint.glb"
    rig.write_bytes(b"rig")
    paint.write_bytes(b"old paint")
    models = ch.Models(rig, {"--colour-from": paint})
    calls = []

    def fake_call(command, log, what, gpu=False):
        calls.append(what)
        glb = Path(command[3])
        glb.write_bytes(b"figure")
        glb.with_suffix(".report.json").write_text("{}")
        return 0

    chain = hero_chain.Chain.__new__(hero_chain.Chain)
    chain.out, chain.games, chain.call = tmp_path, {}, fake_call
    chain.game_run(models, "boots", ["--height", "2.2"])
    chain.games = {}                                         # a new run of the script
    chain.game_run(models, "boots", ["--height", "2.2"])
    assert len(calls) == 1                                    # nothing changed: the figure is kept
    paint.write_bytes(b"new paint, longer")                   # the stage before wrote its model again
    chain.games = {}
    chain.game_run(models, "boots", ["--height", "2.2"])
    assert len(calls) == 2


def test_a_game_figure_made_for_the_same_models_is_taken_under_another_name(tmp_path):
    # the hands stage's figure is the head stage's figure before: made once, also across runs of the script
    import hero_chain

    rig = tmp_path / "rig.glb"
    rig.write_bytes(b"rig")
    models = ch.Models(rig, {})
    calls = []

    def fake_call(command, log, what, gpu=False):
        calls.append(what)
        glb = Path(command[3])
        glb.write_bytes(b"figure")
        glb.with_suffix(".report.json").write_text("{}")
        return 0

    chain = hero_chain.Chain.__new__(hero_chain.Chain)
    chain.out, chain.games, chain.call = tmp_path, {}, fake_call
    first, _ = chain.game_run(models, "hands", ["--height", "2.2"])
    chain.games = {}                                          # a new run of the script
    again, _ = chain.game_run(models, "before_head", ["--height", "2.2"])
    assert len(calls) == 1 and again == first


def test_a_stage_whose_game_figure_lost_its_feet_is_dropped(tmp_path, monkeypatch):
    # skin_sorcerer_c's boots stood on a made-up sole and game_ready cut their feet off: colour and holes passed,
    # only the game figure showed it (2026-11-24)
    import hero_chain

    before, after = ch.Models(tmp_path / "a.glb", {}), ch.Models(tmp_path / "b.glb", {})
    chain = hero_chain.Chain.__new__(hero_chain.Chain)
    chain.game_run = lambda models, tag, game: (tmp_path / f"{tag}.glb", {"pose_check": {"hole_px_left": 0}})
    lost = fc.Result("feet forward", "fail", -0.009, "-", "the left and right foot lost its toe")
    monkeypatch.setattr(fc, "stage_against", lambda stage, old, new: [lost] if stage == "boots" else [])
    stage = ch.Stage("boots", "kept", "")
    checks = chain.game_checks("boots", before, after, stage, [])
    assert (False, "the game figure: the left and right foot lost its toe") in checks
    assert stage.measured["game figure: feet forward"] == -0.009
    hands = chain.game_checks("hands", before, after, ch.Stage("hands", "kept", ""), [])
    assert [ok for ok, _ in hands] == [True]                  # the hands: game-ready's own problems alone


def test_new_boots_stand_where_the_game_step_cuts_the_figures_feet(monkeypatch):
    # skin_sorcerer_c's game options cut a made-up platform sole at 5.8 % of its height (--feet-from its Finish run);
    # the new boots stood on the platform's floor and the cut took their feet (2026-11-24)
    import game_ready
    import hero_chain

    parts = {"parts": [{"side": "below", "low": 0.0, "high": 0.05, "reach": 0.0578}]}
    monkeypatch.setattr(game_ready, "outline_parts", lambda path: parts)
    assert hero_chain.game_cut_share(["--height", "2.2", "--feet", "cut", "--feet-from", "C:/finish/run"]) == 0.0578
    assert hero_chain.game_cut_share(["--height", "2.2", "--feet", "off", "--feet-from", "C:/finish/run"]) is None
    assert hero_chain.game_cut_share(["--height", "2.2"]) is None
    models = ch.Models(Path("C:/f/rig.glb"), {})
    boots = hero_chain.stage_command("boots", models, Path("C:/f/front.png"), Path("C:/o/3_boots"), {"floor_at": 0.0578})
    assert boots[boots.index("--floor-at") + 1] == "0.0578"


def test_a_stage_that_drops_a_made_up_sole_is_compared_in_one_frame():
    # skin_sorcerer_c's boots took away a made-up platform sole, its lowest point rose 5.7 % of its height, and the
    # two models were drawn each in its own frame: a few pixels apart, every red trim read as a changed patch (52)
    from test_colour_compare import cube, texture

    tex = texture()
    for k in range(6):
        tex[2::6, k * 64:(k + 1) * 64] = (220, 30, 40)         # thin red trims: a shift of a pixel or two shows
        tex[:, k * 64 + 2:(k + 1) * 64:6] = (220, 30, 40)
    p, uv, f = cube(n=8)
    old = cc.Model.single(p, uv, f, tex)
    raised = p.copy()
    raised[raised[:, 1] < -0.45, 1] = -0.43                   # the bottom 7 % of the height gone
    new = cc.Model.single(raised, uv, f, tex)

    def changed(points):
        return points[:, 1] < -0.3

    assert ch.colour_outside(old, new, changed) == []
    seen = {}
    real = cc.compare
    cc.compare = lambda *a, **k: seen.update(k) or []
    try:
        ch.colour_outside(old, new, changed)
    finally:
        cc.compare = real
    (centre, scale), other = seen["frame"]
    want = cc.shared_frame(old, new)[0]
    assert other is seen["frame"][0] and np.allclose(centre, want[0]) and scale == pytest.approx(want[1])
    assert cc.frames(old, new)[0] is not cc.frames(old, new)[1]        # alone, the two boxes count as two places


def test_after_new_boots_the_game_step_cuts_no_feet():
    # the boots stage replaced the feet that stood on a made-up sole; the game step's cut, measured from the new lowest
    # point, then took 5.8 % off the new boots: their feet (skin_sorcerer_c, 2026-11-24)
    import hero_chain

    models = ch.Models(Path("C:/f/rig.glb"), ch.game_models(GAME))
    assert ch.option_value(models.game_args(GAME), "--feet") == "cut"
    booted = hero_chain.stage_models("boots", models, {models.rig: Path("C:/o/rig_boots.glb")})
    args = booted.game_args(GAME)
    assert ch.option_value(args, "--feet") == "off" and ch.option_value(args, "--feet-from") is None
    assert ch.Models.from_dict(booted.as_dict()).new_feet is True
    hands = hero_chain.stage_models("hands", booted, {booted.rig: Path("C:/o/rig_boots_hands.glb")})
    assert hands.new_feet is True                              # a later stage keeps the new feet
    assert hero_chain.stage_models("hands", models, {}).new_feet is False


def test_after_new_hands_the_game_step_keeps_their_fingers():
    # skin_sorcerer_c's new hands came out of the even thinning to 14,000 triangles as thin flat claws, about 200
    # triangles a hand; --hand-protect 0.6 kept four fingers and a thumb (round 27); the chain passes it itself once its
    # hands stage is kept, and an explicit value of the caller's stands
    import hero_chain

    models = ch.Models(Path("C:/f/rig.glb"), ch.game_models(GAME))
    assert ch.option_value(models.game_args(GAME), "--hand-protect") is None
    handed = hero_chain.stage_models("hands", models, {models.rig: Path("C:/o/rig_hands.glb")})
    assert ch.option_value(handed.game_args(GAME), "--hand-protect") == str(ch.HAND_PROTECT)
    assert ch.option_value(handed.game_args([*GAME, "--hand-protect", "0.3"]), "--hand-protect") == "0.3"
    assert ch.Models.from_dict(handed.as_dict()).new_hands is True
    booted = hero_chain.stage_models("boots", handed, {handed.rig: Path("C:/o/rig_hands_boots.glb")})
    assert booted.new_hands is True and booted.new_feet is True       # a later stage keeps the new hands


def test_a_problem_game_ready_names_after_a_stage_and_not_before_drops_it():
    # skin_sorcerer_b's boots, built far finer than its body, took most of the 14,000 triangles: the body came out
    # crude, 1.813 m tall for 1.85, and game_ready said so in its report; every stage check passed (2026-11-24)
    before = {"problems": []}
    after = {"problems": ["height 1.813 m, asked for 1.850 m",
                          "52 patches differ in colour from paint_boots.glb (first: front: a darker patch)"]}
    ok, why = ch.problems_verdict(after, before)
    assert ok is False and "52 patches differ in colour" in why and "height 1.813 m" in why
    had = {"problems": ["2 materials: an engine draws each one separately"]}
    assert ch.problems_verdict({"problems": ["2 materials: an engine draws each one separately"]}, had)[0] is True
    assert ch.problems_verdict({"problems": []}, None)[0] is True


def test_a_shape_stage_is_counted_for_see_through_on_the_models_the_game_step_shapes_from():
    # the face stage's holes were in its own rigged and shape models (145 and 223 pixels from behind, round 27): both
    # are counted before the game step runs; the paint source is not a shape (round 28)
    import hero_chain

    models = ch.Models(Path("C:/f/rig.glb"), {"--mesh-from": Path("C:/f/fine.glb"), "--colour-from": Path("C:/f/paint.glb")})
    after = hero_chain.stage_models("head", models, {models.rig: Path("C:/o/rig_face.glb"),
                                                     Path("C:/f/fine.glb"): Path("C:/o/fine_face.glb"),
                                                     Path("C:/f/paint.glb"): Path("C:/o/paint_face.glb")})
    pairs = ch.shape_pairs(models, after)
    assert [label for label, _, _ in pairs] == ["rigged model", "shape model"]
    assert pairs[1][1:] == (Path("C:/f/fine.glb"), Path("C:/o/fine_face.glb"))
    assert ch.shape_pairs(models, models) == []


def test_after_new_hands_the_game_step_keeps_the_glow_off_them(capsys):
    # the Arkan-Duellant's new hands came out with gold fingers: the game step's --glow #ffc23a found the drawn hand's
    # warm highlights (D300); the chain passes --glow-not-on-hands once its hands stage is kept, unless the caller asks
    # for glowing hands with the chain's --glow-on-hands
    import hero_chain

    glowing = [*GAME, "--glow", "#ffc23a"]
    models = ch.Models(Path("C:/f/rig.glb"), ch.game_models(GAME))
    assert "--glow-not-on-hands" not in models.game_args(glowing)
    handed = hero_chain.stage_models("hands", models, {models.rig: Path("C:/o/rig_hands.glb")})
    assert handed.game_args(glowing).count("--glow-not-on-hands") == 1
    assert handed.game_args([*glowing, "--glow-not-on-hands"]).count("--glow-not-on-hands") == 1
    assert "--glow-not-on-hands" not in handed.game_args(GAME)                 # nothing glows: nothing to say
    lit = ch.Models(Path("C:/f/rig.glb"), ch.game_models(GAME), hand_glow=True)
    lit = hero_chain.stage_models("hands", lit, {lit.rig: Path("C:/o/rig_hands.glb")})
    assert "--glow-not-on-hands" not in lit.game_args(glowing)
    with pytest.raises(SystemExit):
        hero_chain.main(["--help"])
    assert "--glow-on-hands" in capsys.readouterr().out


def test_a_hands_stage_that_keeps_the_figures_own_hands_is_skipped(tmp_path):
    # no drawing kept the picture's glove: the hand stage keeps the figure's own hands and says why, like the boots gate
    import hero_chain

    (tmp_path / "hands.json").write_text(json.dumps({"skipped": "no drawing kept the picture's black glove",
                                                     "picture_hand": {"right": {"lab": [15.3, 0.5, -1.3], "skin": 0.0,
                                                                                "pixels": 1732}}}))
    from types import SimpleNamespace

    chain = hero_chain.Chain.__new__(hero_chain.Chain)
    chain.args = SimpleNamespace(picture=tmp_path / "front.png")
    chain.call = lambda command, log, what, gpu=False: hero_chain.SKIPPED
    models = ch.Models(tmp_path / "rig.glb", {})
    stage, new = chain.run_stage("hands", models, tmp_path, {}, [])
    assert (stage.status, new) == ("skipped", None) and "black glove" in stage.why
    assert stage.measured["right hand in the picture"] == "L*a*b* [15.3, 0.5, -1.3], 0% skin-toned"
