"""The Hero parts command's own logic: what it checks before starting, how it finds the figure's cameras, which
drawings it keeps.

The heavy steps (the picture editor, Pixal3D, its clean-up, the graft on a real figure) run real models and are
exercised by hand; these pin the cheap parts that decide whether a run starts and what it builds from.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import hero_parts


def test_hero_parts_stops_before_any_work_when_a_part_is_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(hero_parts, "needed_parts", lambda drawing=True: {"Pixal3D (3D models)": tmp_path / "gone"})
    monkeypatch.setattr(hero_parts, "picture_engine",
                        lambda: (None, "the picture editor (FLUX.2 klein, edit mode) is not set up"))
    code = hero_parts.main([str(tmp_path / "rigged.glb"), str(tmp_path / "front.png"), "--boots",
                            "--out", str(tmp_path / "run")])
    err = capsys.readouterr().err
    assert code == 2
    assert "Pixal3D (3D models)" in err and "picture editor" in err and "nothing was downloaded" in err
    assert not (tmp_path / "run").exists()


def test_a_part_given_as_a_build_needs_no_picture_editor(monkeypatch, tmp_path, capsys):
    asked = []
    monkeypatch.setattr(hero_parts, "needed_parts", lambda drawing=True: asked.append(drawing) or {
        "Real-ESRGAN": tmp_path / "gone"})
    monkeypatch.setattr(hero_parts, "picture_engine", lambda: pytest.fail("the picture editor was asked for"))
    code = hero_parts.main([str(tmp_path / "rigged.glb"), str(tmp_path / "front.png"), "--box", "1", "2", "3", "4",
                            "--build", str(tmp_path / "pistol.glb"), "--out", str(tmp_path / "run")])
    assert code == 2 and asked == [False] and "Real-ESRGAN" in capsys.readouterr().err


def test_mirror_needs_a_second_box(tmp_path):
    with pytest.raises(SystemExit):
        hero_parts.main([str(tmp_path / "m.glb"), str(tmp_path / "p.png"), "--box", "1", "2", "3", "4", "--mirror"])


def test_the_figures_cameras_are_found_through_its_licence_record(tmp_path):
    raw = tmp_path / "run" / "figure.glb"
    views = raw.with_suffix(".svviews")
    views.mkdir(parents=True)
    (views / "transforms.json").write_text("{}")
    record = {"operation": "hero_hands", "license": {"name": "MIT"}, "source_record": {
        "operation": "finish", "license": {"name": "MIT"},
        "source_record": {"backend": "pixal3d", "license": {"name": "MIT"}, "output": {"path": str(raw)}}},
        "component_records": [{"backend": "pixal3d", "output": {"path": str(tmp_path / "hand.glb")}}]}
    model = tmp_path / "rigged_hands.glb"
    model.write_bytes(b"glb")
    model.with_suffix(".provenance.json").write_text(json.dumps(record))
    assert hero_parts.run_views(model) == views                  # the figure's run, not a hand's build
    (views / "transforms.json").unlink()
    assert hero_parts.run_views(model) is None


def boot_mask(toe="right", cut=False, crumb=False):
    mask = np.zeros((100, 100), bool)
    mask[10:80, 40:60] = True                                   # the shaft
    if toe == "right":
        mask[80:95, 40:85] = True                               # the foot, toe to the right
    else:
        mask[80:95, 15:60] = True
    if cut:
        mask[0:10, 40:60] = True                                # runs off the top: cut off
    if crumb:
        mask[20:60, 75:95] = True                               # a second thing beside it
    return mask


def test_a_drawing_is_kept_when_it_shows_one_whole_part_with_its_toe_where_it_was_told():
    assert hero_parts.drawing_check(boot_mask(), toe="right")["ok"]
    assert hero_parts.drawing_check(boot_mask(), toe=None)["ok"]
    left = hero_parts.drawing_check(boot_mask("left"), toe="right")
    assert not left["ok"] and left["toe_at"] == "left"
    assert not hero_parts.drawing_check(boot_mask(cut=True), toe="right")["ok"]
    assert not hero_parts.drawing_check(boot_mask(crumb=True), toe="right")["ok"]


def test_the_finer_models_get_a_part_cleaned_up_to_a_size_a_game_step_can_take():
    # the raw build is about a million faces a part: two raw boots in the Zauberer's mesh source made the game-ready
    # step walk 2.4 million faces instead of 0.42 million; a part of the finer models is cleaned up to FINE_FACES
    assert hero_parts.FINE_FACES == 150000
    assert hero_parts.FINE_FACES > hero_parts.PART_FACES


def test_a_build_that_touches_itself_is_made_one_closed_skin():
    # the Zauberer's boot touched itself on 1.47 per cent of its edges, and the game-ready step's thinning tore those
    # sheets into shards at the ankle and the heel; the pistol (0.08 per cent) thinned cleanly and stays as built
    assert hero_parts.wants_solid("auto", 0.0147)
    assert not hero_parts.wants_solid("auto", 0.0008)
    assert hero_parts.wants_solid("on", 0.0)
    assert not hero_parts.wants_solid("off", 0.5)


class FakeRun:
    def __init__(self):
        self.commands, self.said = [], []

    def say(self, text):
        self.said.append(text)

    def run(self, command, **_):
        self.commands.append([str(c) for c in command])
        return True


def test_the_closed_skin_is_the_finish_retopology_at_the_parts_own_size(monkeypatch, tmp_path):
    monkeypatch.setattr(hero_parts, "find_blender", lambda: tmp_path / "blender.exe")
    clean = tmp_path / "boot_clean20k.glb"
    run = FakeRun()
    solid = hero_parts.solid_part(run, clean, 20000, tmp_path)
    assert solid == tmp_path / "boot_clean20k_solid.glb"
    command = run.commands[0]
    assert command[0] == str(tmp_path / "blender.exe")
    assert command[command.index("--python") + 1].endswith("blender_retopo_bake.py")
    start = command.index("--") + 1
    assert command[start:start + 6] == [str(clean), str(solid), "20000", str(hero_parts.PAINT_SIZE), "89.0",
                                        str(hero_parts.SOLID_VOXEL)]
    solid.write_bytes(b"glb")                                   # a rerun keeps it
    assert hero_parts.solid_part(run, clean, 20000, tmp_path) == solid and len(run.commands) == 1
    monkeypatch.setattr(hero_parts, "find_blender", lambda: None)
    with pytest.raises(SystemExit, match="Blender"):
        hero_parts.solid_part(FakeRun(), tmp_path / "other.glb", 20000, tmp_path)
