"""Tests for image_to_3dlab/fx_markers.py: mark where an effect was taken out of a picture, as empty nodes."""

from __future__ import annotations

import json
import struct

import numpy as np

from image_to_3dlab import fx_markers
from image_to_3dlab.glb_turn import build_glb, parse_glb


def box_glb(lo=(-0.5, 0.0, -0.3), hi=(0.5, 1.0, 0.3)) -> bytes:
    """A closed box as a GLB (glTF axes: +Y up, +Z front), its corners from lo to hi."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    pos = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float32)
    tri = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.uint16)
    pb, ib = pos.tobytes(), tri.tobytes() + b"\0\0"
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"name": "RETOPO", "mesh": 0}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
           "buffers": [{"byteLength": len(pb) + len(ib)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pb)},
                           {"buffer": 0, "byteOffset": len(pb), "byteLength": len(ib) - 2}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
                          "min": [x0, y0, z0], "max": [x1, y1, z1]},
                         {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"}]}
    return build_glb(doc, pb + ib)


def pictures(with_effect: bool, effect_over_body: bool = False) -> np.ndarray:
    """A 200 x 200 RGBA front view: the subject a 100 x 100 square; the effect a 20 px orange blob."""
    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    rgba[50:150, 50:150] = (60, 60, 60, 255)
    if with_effect:
        if effect_over_body:
            rgba[60:80, 60:80] = (255, 170, 40, 255)    # in front of the body, top left
        else:
            rgba[30:50, 140:160] = (255, 170, 40, 255)  # above the top right corner, touching it
    return rgba


def test_an_effect_outside_the_outline_is_marked_where_it_touches_the_subject():
    found = fx_markers.effect_regions(pictures(True), pictures(False))
    assert len(found) == 1
    r = found[0]
    # it touches the subject at its bottom edge, at the subject's top right corner
    assert 46 <= r["anchor"][1] <= 52 and 138 <= r["anchor"][0] <= 152


def test_an_effect_in_front_of_the_body_is_found_by_its_colour():
    found = fx_markers.effect_regions(pictures(True, effect_over_body=True), pictures(False))
    assert len(found) == 1 and 55 <= found[0]["centre"][0] <= 85


def test_no_effect_no_marker():
    assert fx_markers.effect_regions(pictures(False), pictures(False)) == []


def test_the_marker_lands_on_the_models_matching_corner_at_the_front():
    data = box_glb()
    out = fx_markers.add_markers(data, pictures(True), pictures(False), name="flame")
    doc, _ = parse_glb(out)
    marks = [n for n in doc["nodes"] if n.get("name", "").startswith("fx_flame")]
    assert len(marks) == 1
    x, y, z = marks[0]["translation"]
    assert abs(x - 0.5) < 0.08 and abs(y - 1.0) < 0.08   # the subject's top right in the picture
    assert z > 0.2                                        # on the front, the side the picture sees
    assert len(doc["scenes"][0]["nodes"]) == 2            # a root node of its own
    assert marks[0]["extras"]["radius"] > 0


def test_the_mesh_and_its_data_are_left_alone():
    data = box_glb()
    out = fx_markers.add_markers(data, pictures(True), pictures(False), name="flame")
    _, b0 = parse_glb(data)
    doc, b1 = parse_glb(out)
    assert b0 == b1 and doc["meshes"] == json.loads(json.dumps(parse_glb(data)[0]["meshes"]))
    assert struct.unpack("<I", out[8:12])[0] == len(out)   # the GLB header's length is right


def test_pictures_framed_differently_are_lined_up_first():
    # the effect-free picture drawn at another size and place (an image editor's own framing)
    orig = pictures(True)
    clean = np.zeros((300, 260, 4), dtype=np.uint8)
    clean[90:240, 40:190] = (60, 60, 60, 255)          # the square 1.5 x bigger, elsewhere
    s, _, _ = fx_markers.align(orig[..., 3] > 128, clean[..., 3] > 128)
    assert abs(s - 1 / 1.5) < 0.05                       # clean -> orig: shrink by 1.5
    found = fx_markers.effect_regions(orig, clean)
    assert len(found) == 1
    ax, ay = found[0]["anchor"]                          # in the clean picture's own pixels
    assert 175 <= ax <= 195 and 82 <= ay <= 98           # its top right corner
