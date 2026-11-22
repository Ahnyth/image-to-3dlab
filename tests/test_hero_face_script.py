"""The Hero face command's own logic: what it checks before starting, and how it finds the part with the head.

The heavy steps (Real-ESRGAN, Pixal3D, its clean-up, Pixel Match) run real models and are exercised by hand; these pin
the cheap parts that decide whether a run starts at all and which part of the figure is cut.
"""
from __future__ import annotations

import numpy as np

from scripts import hero_face


def test_hero_face_stops_before_any_work_when_a_part_is_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(hero_face, "needed_parts", lambda: {"Pixal3D (3D models)": tmp_path / "gone"})
    code = hero_face.main([str(tmp_path / "rigged.glb"), str(tmp_path / "front.png"), "--out", str(tmp_path / "run")])
    err = capsys.readouterr().err
    assert code == 2
    assert "Pixal3D (3D models)" in err and "nothing was downloaded" in err
    assert not (tmp_path / "run").exists()                   # nothing started, nothing written


def skinned(points, bones, count=3):
    dense = np.zeros((len(points), count))
    dense[np.arange(len(points)), bones] = 1.0
    return {"positions": np.asarray(points, dtype=float), "faces": np.zeros((0, 3), np.int64), "dense": dense}


def test_head_of_finds_the_part_holding_the_head_by_the_rigs_top_bone():
    hands = skinned([[0.3, 0.0, 0.0], [0.31, 0.01, 0.0]] * 20, [2] * 40)
    body = skinned([[0.0, y, 0.0] for y in np.linspace(-1, 1, 60)], [0] * 40 + [1] * 20)
    index, head, picked = hero_face.head_of([hands, body], None)
    assert index == 1 and picked == [1]
    assert head.sum() == 20 and (body["positions"][head, 1] > 0.3).all()


def test_head_of_reads_a_separate_rig_for_an_unrigged_model():
    rig = skinned([[0.0, y, 0.0] for y in np.linspace(-1, 1, 60)], [0] * 40 + [1] * 20)
    model = {"positions": np.array([[0.0, y, 0.0] for y in np.linspace(-1, 1, 30)]),
             "faces": np.zeros((0, 3), np.int64)}
    index, head, _ = hero_face.head_of([model], [rig])
    assert index == 0 and head.sum() == 10
