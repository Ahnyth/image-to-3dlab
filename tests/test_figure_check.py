"""Tests for `image_to_3dlab/figure_check.py`: a game figure checked the way a second reviewer looks.

The faults it measures were each found by a person first (a hand with a digit too many, a neck
that reads as a pale ring, a loose flake under a hat's brim). These tests build small shapes
with exactly that fault, and the same shape without it.
"""

from __future__ import annotations

import io
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab import colour_compare as cc
from image_to_3dlab import figure_check as fc

REPO = Path(__file__).resolve().parents[1]

_BOX_FACES = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                       [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


def box(lo, hi):
    """A closed box from corner LO to corner HI, its faces wound outward."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    p = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    return p, _BOX_FACES.copy()


def join(*boxes):
    points, faces = [], []
    for p, f in boxes:
        faces.append(f + sum(len(q) for q in points))
        points.append(p)
    return np.concatenate(points), np.concatenate(faces)


def model_of(points, faces, colour=(200, 160, 140)) -> cc.Model:
    tex = np.full((4, 4, 3), colour, np.uint8)
    return cc.Model.single(points, np.full((len(points), 2), 0.5), faces, tex)


def hand_figure(digits: int, fused: bool = False) -> fc.Figure:
    """A hand hanging down from a wrist at the origin: a palm, a thumb that branches off near the
    wrist, and DIGITS fingers below the palm (one block when FUSED)."""
    parts = [box((-0.05, -0.08, -0.01), (0.05, 0.0, 0.01)),          # the palm
             box((0.055, -0.07, -0.008), (0.07, -0.03, 0.008))]      # the thumb, off the palm's side
    if fused:
        parts.append(box((-0.05, -0.18, -0.01), (0.05, -0.08, 0.01)))
    else:
        width = 0.1 / digits
        for k in range(digits):
            x = -0.05 + k * width
            parts.append(box((x + 0.15 * width, -0.18, -0.008), (x + 0.85 * width, -0.075, 0.008)))
    # a bridge from the thumb to the palm keeps it one piece with the hand
    parts.append(box((0.045, -0.06, -0.006), (0.06, -0.045, 0.006)))
    p, f = join(*parts)
    weights = np.ones((len(p), 1))
    return fc.Figure(model_of(p, f), {"LeftHand": np.zeros(3)}, ["LeftHand"], weights)


def test_a_cut_through_separate_bars_counts_each_bar():
    p, f = join(*[box((x, 0, 0), (x + 0.6, 1, 0.5)) for x in (0.0, 1.0, 2.0)])
    assert fc.section_pieces(p, f, (0, 0.5, 0), (0, 1, 0), 0.02) == 3


def test_bars_that_touch_are_one_piece():
    p, f = join(box((0, 0, 0), (1, 1, 0.5)), box((1, 0, 0), (2, 1, 0.5)))
    assert fc.section_pieces(p, f, (0, 0.5, 0), (0, 1, 0), 0.02) == 1


def test_a_five_digit_hand_passes_and_shows_four_fingers_in_a_cut():
    r = fc.fingers(hand_figure(4))
    assert r.status == "pass" and "left hand 4" in r.value


def test_a_hand_with_a_digit_too_many_fails():
    r = fc.fingers(hand_figure(5))
    assert r.status == "fail" and "left hand 5" in r.value


def test_a_mitten_is_pointed_at():
    r = fc.fingers(hand_figure(4, fused=True))
    assert r.status == "look"


def test_a_figure_without_a_skin_is_not_counted():
    p, f = box((0, 0, 0), (1, 1, 1))
    r = fc.fingers(fc.Figure(model_of(p, f), {}, [], None))
    assert r.status == "info"


def flake_figure(with_flake: bool) -> fc.Figure:
    parts = [box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))]
    if with_flake:
        parts.append(box((0.1, 0.1, 0.55), (0.14, 0.14, 0.58)))    # a chip floating in front
    p, f = join(*parts)
    return fc.Figure(model_of(p, f), {}, [], None)


def test_a_loose_chip_in_sight_is_pointed_at_and_a_clean_model_passes():
    for with_flake, status in ((True, "look"), (False, "pass")):
        figure = flake_figure(with_flake)
        views = {"front": fc.draw(figure, "front", (0, 0, 1), (0, 0, 0), 0.8, 320)}
        r = fc.flakes(figure, views)
        assert r.status == status
        assert len(r.spots) == (1 if with_flake else 0)


def test_pieces_are_welded_by_position_and_measured_by_area():
    p, f = join(box((0, 0, 0), (1, 1, 1)), box((3, 0, 0), (3.5, 0.5, 0.5)))
    piece, area = fc.pieces(model_of(p, f))
    assert len(area) == 2 and sorted(area) == pytest.approx([1.5, 6.0])
    assert len(piece) == len(f)


def neck_view(step: float) -> fc.View:
    """A front view of a neck of skin, with a level step of STEP in lightness half way down."""
    size = 80
    rgb = np.zeros((size, size, 3))
    lab_light, lab_dark = 70.0, 70.0 - step
    rows = np.arange(size)
    light = np.where(rows < size // 2, lab_light, lab_dark)
    grey = np.array([cc.lab(np.full(3, g / 100.0))[0] for g in range(101)])
    rgb[:] = (np.interp(light, grey, np.arange(101)) / 100.0)[:, None, None]
    covered = np.zeros((size, size), bool)
    covered[:, 20:60] = True
    point = np.zeros((size, size, 3))
    point[..., 1] = (1.0 - rows / size)[:, None]                        # y falls down the picture
    normal = np.zeros((size, size, 3))
    normal[..., 2] = 1.0
    return fc.View("front", np.array([0.0, 0.0, 1.0]), rgb, covered, np.where(covered, 0, -1), normal, point,
                   np.zeros((size, size), bool), 1.0 / size)


def test_a_hard_level_step_around_the_neck_fails_and_an_even_neck_passes():
    head = fc.Head(np.array([0.0, 0.5, 0.0]), 0.5, 0.0, 0.9, 1.2, 0.5)
    for step, status in ((25.0, "fail"), (0.0, "pass")):
        v = neck_view(step)
        skin = cc.lab(v.rgb[10, 30][None])[0]
        r = fc.neck_ring({"front": v, "from below": v}, skin, head)
        assert r.status == status, r.message


def test_holes_in_motion_come_from_game_readys_report(tmp_path):
    report = tmp_path / "fig.report.json"
    report.write_text(json.dumps({"pose_check": {"poses": 4, "hole_px": 20, "hole_px_left": 0}}))
    assert fc.pose_holes(report).status == "pass"
    report.write_text(json.dumps({"pose_check": {"poses": 4, "hole_px": 20, "hole_px_left": 3}}))
    assert fc.pose_holes(report).status == "fail"
    assert fc.pose_holes(tmp_path / "missing.json").status == "info"


def skinned_glb(points, faces, joints: dict, weight_to: int = 0) -> bytes:
    """A GLB with one skinned mesh: every vertex weighted fully to joint WEIGHT_TO."""
    binary, views, accessors = b"", [], []

    def add(data: bytes) -> int:
        nonlocal binary
        binary += b"\0" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary += data
        return len(views) - 1

    n = len(points)
    for data, kind, count, typ in (
            (points.astype(np.float32).tobytes(), 5126, n, "VEC3"),
            (np.full((n, 2), 0.5, np.float32).tobytes(), 5126, n, "VEC2"),
            (faces.astype(np.uint32).tobytes(), 5125, faces.size, "SCALAR"),
            (np.tile(np.array([weight_to, 0, 0, 0], np.uint8), (n, 1)).tobytes(), 5121, n, "VEC4"),
            (np.tile(np.array([255, 0, 0, 0], np.uint8), (n, 1)).tobytes(), 5121, n, "VEC4")):
        accessor = {"bufferView": add(data), "componentType": kind, "count": count, "type": typ}
        if typ == "VEC3":
            accessor["min"], accessor["max"] = points.min(0).tolist(), points.max(0).tolist()
        if kind == 5121 and accessors and accessors[-1]["componentType"] == 5121:
            accessor["normalized"] = True
        accessors.append(accessor)
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(np.full((4, 4, 3), (200, 160, 140), np.uint8)).save(buf, "PNG")
    image = add(buf.getvalue())
    names = list(joints)
    nodes = [{"mesh": 0, "skin": 0}] + [{"name": k, "translation": list(map(float, v))} for k, v in joints.items()]
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
           "nodes": nodes, "skins": [{"joints": list(range(1, len(names) + 1))}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1, "JOINTS_0": 3,
                                                      "WEIGHTS_0": 4}, "indices": 2, "material": 0}]}],
           "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
           "textures": [{"source": 0}], "images": [{"bufferView": image, "mimeType": "image/png"}],
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def test_a_skinned_glb_reads_its_joints_and_weights(tmp_path):
    p, f = box((-0.2, 0.0, -0.1), (0.2, 1.8, 0.1))
    path = tmp_path / "figure.glb"
    path.write_bytes(skinned_glb(p, f, {"Hips": (0, 1.0, 0), "Head": (0, 1.6, 0)}, weight_to=1))
    figure = fc.read_figure(path)
    assert figure.names == ["Hips", "Head"]
    assert figure.joints["Head"] == pytest.approx([0, 1.6, 0])
    assert figure.owned_by("Head") == pytest.approx(np.ones(len(p)))
    assert figure.height == pytest.approx(1.8)


def test_the_script_writes_a_report_and_compares_with_the_version_before(tmp_path):
    p, f = box((-0.2, 0.0, -0.1), (0.2, 1.8, 0.1))
    old, new = tmp_path / "old.glb", tmp_path / "new.glb"
    old.write_bytes(skinned_glb(p, f, {"Hips": (0, 1.0, 0), "Head": (0, 1.6, 0)}))
    new.write_bytes(skinned_glb(p, f, {"Hips": (0, 1.0, 0), "Head": (0, 1.6, 0)}))
    out = tmp_path / "check"
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "figure_check.py"), str(old), str(new),
                           "--series", "--out", str(out), "--size", "96"], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((out / "report.json").read_text())
    assert set(report) == {"old", "new"}
    checks = [r["check"] for r in report["new"]["results"]]
    assert "colour against the version before" in checks and "fingers" in checks
    assert (out / "index.html").read_text(encoding="utf-8").count("<img") >= 2


def test_a_colour_change_is_a_spot_to_look_at_unless_the_look_should_stay(tmp_path):
    p, f = box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    old = fc.Figure(model_of(p, f, (200, 160, 140)), {}, [], None)
    new = fc.Figure(model_of(p, f, (60, 40, 30)), {}, [], None)
    assert fc.colour_against(old, new).status == "look"
    assert fc.colour_against(old, new, same_look=True).status == "fail"
    assert fc.colour_against(old, old, same_look=True).status == "pass"


# --- a raw rig: hands found by the skeleton's shape, not by name ---------------------------------------------------


def rigged_glb(points, faces, parents: dict, heads: dict, owner: np.ndarray) -> bytes:
    """A GLB with one mesh skinned to a joint tree (PARENTS: name -> parent name, HEADS: name -> world position), each
    vertex weighted fully to the joint OWNER names (an index into HEADS' order), as a raw Rig output is: bones
    named bone_0 ... in no telling order."""
    names = list(heads)
    binary, views, accessors = b"", [], []

    def add(data: bytes) -> int:
        nonlocal binary
        binary += b"\0" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary += data
        return len(views) - 1

    n = len(points)
    joints = np.zeros((n, 4), np.uint8)
    joints[:, 0] = owner
    for data, kind, count, typ in (
            (points.astype(np.float32).tobytes(), 5126, n, "VEC3"),
            (np.full((n, 2), 0.5, np.float32).tobytes(), 5126, n, "VEC2"),
            (faces.astype(np.uint32).tobytes(), 5125, faces.size, "SCALAR"),
            (joints.tobytes(), 5121, n, "VEC4"),
            (np.tile(np.array([255, 0, 0, 0], np.uint8), (n, 1)).tobytes(), 5121, n, "VEC4")):
        accessor = {"bufferView": add(data), "componentType": kind, "count": count, "type": typ}
        if typ == "VEC3":
            accessor["min"], accessor["max"] = points.min(0).tolist(), points.max(0).tolist()
        if kind == 5121 and accessors and accessors[-1]["componentType"] == 5121:
            accessor["normalized"] = True
        accessors.append(accessor)
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(np.full((4, 4, 3), (200, 160, 140), np.uint8)).save(buf, "PNG")
    image = add(buf.getvalue())
    index = {name: i + 1 for i, name in enumerate(names)}
    nodes = [{"mesh": 0, "skin": 0}]
    for name in names:
        parent = parents[name]
        local = np.asarray(heads[name], float) - (np.asarray(heads[parent], float) if parent else 0.0)
        node = {"name": name, "translation": local.tolist()}
        kids = [index[c] for c, p in parents.items() if p == name]
        if kids:
            node["children"] = kids
        nodes.append(node)
    roots = [0] + [index[name] for name in names if parents[name] is None]
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": roots}],
           "nodes": nodes, "skins": [{"joints": [index[name] for name in names]}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1, "JOINTS_0": 3,
                                                      "WEIGHTS_0": 4}, "indices": 2, "material": 0}]}],
           "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
           "textures": [{"source": 0}], "images": [{"bufferView": image, "mimeType": "image/png"}],
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def test_a_raw_rigs_hands_are_found_by_the_skeletons_shape_and_counted(tmp_path):
    from test_hero_parts import stick_figure

    parents, heads, role = stick_figure()
    names = list(heads)
    hand = hand_figure(4)                     # five digits hanging from a wrist at the origin
    body_p, body_f = box((-0.15, 0.1, -0.1), (0.15, 1.5, 0.1))
    hand_p = hand.model.positions + heads[role["Rhand"]]
    points, faces = join((body_p, body_f), (hand_p, hand.model.faces))
    owner = np.r_[np.full(len(body_p), names.index(role["hips"])), np.full(len(hand_p), names.index(role["Rhand"]))]
    path = tmp_path / "rigged.glb"
    path.write_bytes(rigged_glb(points, faces, parents, heads, owner))
    figure = fc.read_figure(path)
    assert "RightHand" not in figure.joints and figure.parents[role["Rhand"]] == role["Rfa"]
    found = fc.hand(figure, "Right")
    assert found is not None and found[0].sum() == len(hand_p)
    assert np.allclose(found[1], heads[role["Rhand"]])
    result = fc.fingers(figure)
    assert result.status == "pass" and "right hand 4" in result.value


def test_a_long_thumb_reaching_among_the_fingers_is_no_sixth_finger():
    # Hero hands draw the hand flat with the thumb spread: a cut below the knuckles catches its tip beside the four
    # fingers, set apart from their row (the Sorcerer's grafted hands: five pieces in two cuts, five digits all told)
    parts = [box((-0.05, -0.08, -0.01), (0.05, 0.0, 0.01))]                     # the palm
    for k in range(4):
        x = -0.05 + k * 0.025
        parts.append(box((x + 0.004, -0.18, -0.008), (x + 0.021, -0.075, 0.008)))
    parts.append(box((0.078, -0.13, -0.008), (0.093, -0.03, 0.008)))           # the thumb, long, beside the hand
    parts.append(box((0.045, -0.06, -0.006), (0.082, -0.045, 0.006)))          # joined to the palm
    p, f = join(*parts)
    figure = fc.Figure(model_of(p, f), {"LeftHand": np.zeros(3)}, ["LeftHand"], np.ones((len(p), 1)))
    assert fc.fingers(figure).status == "pass"
    five = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]])
    assert fc.digits_in_cut(five) == 5                                        # evenly spaced: a digit too many
    assert fc.digits_in_cut(np.r_[five[:4], [[70.0, 5.0]]]) == 4               # one set apart: the thumb's tip


# --- feet: a boot set as a straight shaft has lost its foot (the Hero chain's boots, 2026-11-24) ------------------

def legs_figure(toe: float) -> fc.Figure:
    """Two shins 0.9 tall standing apart, each on a foot reaching TOE ahead of the shin's front (0: a stilt)."""
    parts = []
    for x in (-0.15, 0.05):
        parts.append(box((x, 0.05, -0.05), (x + 0.1, 0.9, 0.05)))              # the shin
        parts.append(box((x, 0.0, -0.06), (x + 0.1, 0.05, 0.05 + toe)))       # the foot
    parts.append(box((-0.2, 0.9, -0.1), (0.2, 1.8, 0.1)))                     # the body above
    p, f = join(*parts)
    return fc.Figure(model_of(p, f), {}, [], None)


def test_a_foot_forward_is_measured_from_the_ankle():
    good, stilt = fc.feet(legs_figure(0.10)), fc.feet(legs_figure(0.0))
    assert good.status == stilt.status == "info"
    assert good.value == pytest.approx(0.10 / 1.8, abs=0.006)
    assert abs(stilt.value) < 0.006


def test_feet_that_lost_their_toes_fail_against_the_version_before():
    before, after = legs_figure(0.10), legs_figure(0.0)
    assert fc.feet_against(before, after).status == "fail"
    assert fc.feet_against(before, legs_figure(0.09)).status == "pass"    # a little shorter is no lost foot
    assert fc.feet_against(after, after).status == "pass"                # a figure in a robe stays as it was


def test_a_stage_fails_only_on_what_got_clearly_worse():
    now = [fc.Result("jagged outline", "fail", 2.4, "-", "12 notches"), fc.Result("loose flakes", "look", 1, "-", ""),
           fc.Result("fingers", "fail", 3, "-", "")]
    was = {"jagged outline": fc.Result("jagged outline", "look", 0.4, "-", ""),
           "loose flakes": fc.Result("loose flakes", "look", 1, "-", "")}
    out = {r.name: r for r in fc.against(now, was)}
    assert out["jagged outline"].status == "fail" and "worse than the version before (0.4)" in out["jagged outline"].message
    assert out["loose flakes"].status == "pass"
    assert "fingers" not in out                                          # not a look the stage is judged by


def grid_box(n=8, half=(0.2, 0.9, 0.15), centre=(0.0, 0.9, 0.0)):
    """A closed box of n x n squares a side, faces wound outward, and each face's side as (axis, sign)."""
    points, faces, sides = [], [], []
    for axis in range(3):
        u_axis, v_axis = [a for a in range(3) if a != axis]
        for sign in (-1.0, 1.0):
            s = np.linspace(-1.0, 1.0, n + 1)
            uu, vv = np.meshgrid(s, s, indexing="ij")
            p = np.zeros((n + 1, n + 1, 3))
            p[..., axis], p[..., u_axis], p[..., v_axis] = sign, uu, vv
            base = sum(len(q) for q in points)
            idx = base + np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
            for i in range(n):
                for j in range(n):
                    a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                    for tri in ((a, b, c), (a, c, d)):
                        faces.append(tri)
                        sides.append((axis, sign))
            points.append(p.reshape(-1, 3))
    v = np.concatenate(points) * np.array(half) + np.array(centre)
    f = np.array(faces)
    n_ = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = np.array([[sign if k == axis else 0.0 for k in range(3)] for axis, sign in sides])
    flip = (n_ * out).sum(axis=1) < 0
    f[flip] = f[flip][:, ::-1]
    return v, f, np.array(sides)


def test_see_through_counts_what_a_game_shows_through_a_hole_from_each_side():
    # skin_sorcerer_c's face stage cut a hole through the back hair: from behind, with back faces culled, 223 pixels of
    # its shape model showed the inside of the head (round 27); the count takes seconds where the game step takes ten
    # minutes (round 28 of the AssetFurnace Support notes)
    v, f, sides = grid_box()
    whole = model_of(v, f)
    assert fc.see_through(whole) == {"front": 0, "back": 0, "left": 0, "right": 0}
    middle = v[f].mean(axis=1)
    hole = (sides[:, 0] == 2) & (sides[:, 1] < 0) & (np.abs(middle[:, 0]) < 0.08) & (np.abs(middle[:, 1] - 0.9) < 0.3)
    holed = model_of(v, f[~hole])
    counts = fc.see_through(holed)
    assert counts["back"] > 100                                  # through the hole onto the front wall's back
    assert counts["front"] == 0 and counts["left"] == 0 and counts["right"] == 0


def test_see_through_against_fails_a_stage_that_opens_a_hole():
    v, f, sides = grid_box()
    middle = v[f].mean(axis=1)
    hole = (sides[:, 0] == 2) & (sides[:, 1] < 0) & (np.abs(middle[:, 0]) < 0.08) & (np.abs(middle[:, 1] - 0.9) < 0.3)
    result = fc.see_through_against(model_of(v, f), model_of(v, f[~hole]))
    assert result.status == "fail" and "back" in result.message
    assert fc.see_through_against(model_of(v, f[~hole]), model_of(v, f[~hole])).status == "pass"   # there before


def test_the_head_is_drawn_twice_as_large_for_the_eye():
    # three light specks under skin_sorcerer_c's brim showed only on an enlarged picture (round 27); the report adds the
    # head's views at twice the size beside the measured ones (round 28 of the AssetFurnace Support notes)
    v, f, _ = grid_box()
    figure = fc.Figure(model_of(v, f), joints={}, names=[], weights=None)
    head = fc.find_head(figure)
    close = fc.close_head_views(figure, head, 64)
    assert close and all(view.rgb.shape[:2] == (128, 128) for view in close.values())
    assert set(close) == set(fc.head_views(figure, head, 64))


def test_a_hands_close_views_draw_the_hand_alone():
    # the second reviewer of round 28: "inside" looked through the body and showed the other hand's back, and "tips"
    # only the cape and the bracer; the palm was never shown. A hand's views draw its own triangles only
    hand_p, hand_f = box((0.5, 0.8, -0.05), (0.6, 0.9, 0.05))
    body_p, body_f = box((0.0, 0.6, -0.2), (0.3, 1.2, 0.2))          # between the hand and a camera at -x
    points, faces = join((hand_p, hand_f), (body_p, body_f))
    model = model_of(points, faces)
    model.textures = [np.full((4, 4, 3), (200, 160, 140), np.uint8), np.full((4, 4, 3), (20, 20, 200), np.uint8)]
    model.part = np.r_[np.zeros(len(hand_f), np.int64), np.ones(len(body_f), np.int64)]
    figure = fc.Figure(model, joints={}, names=[], weights=None)
    inside = np.array([-1.0, 0.0, 0.0])                                # the camera on the body's side of the hand
    whole = fc.draw(figure, "inside", inside, (0.55, 0.85, 0.0), 0.2, 64)
    alone = fc.draw(figure, "inside", inside, (0.55, 0.85, 0.0), 0.2, 64,
                    only=np.r_[np.ones(len(hand_f), bool), np.zeros(len(body_f), bool)])
    middle = (32, 32)
    assert whole.rgb[middle][2] > 0.5                                  # the body in front: blue
    assert alone.rgb[middle][0] > 0.6 and alone.rgb[middle][2] < 0.6   # the hand alone: its skin
