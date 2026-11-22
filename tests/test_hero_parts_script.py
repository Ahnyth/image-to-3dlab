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
