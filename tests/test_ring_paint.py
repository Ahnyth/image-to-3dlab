"""Tests for repeating a band's seen arc round the band (a hat band's runes, a belt's studs).

Synthetic rings where the right answer is plain: a cylinder whose front carries stripes and whose back is plain grey,
seen by one camera in front.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import ring_paint as rp

IDENTITY = ((0, 1, 2), (1, 1, 1))


def _camera(size: int = 256, z: float = 3.0, fov: float = 0.5) -> pp.View:
    c2w = np.eye(4)
    c2w[2, 3] = z  # at +z, looking down -z towards the origin
    image = np.zeros((size, size, 4), dtype=np.uint8)
    image[..., 3] = 255
    return pp.View(image, c2w, fov, "front")


def _ring(segments: int = 180, radius: float = 0.5, height: float = 0.3, tilt_deg: float = 0.0):
    """A closed tube (outer wall only, open top and bottom), UVs: u round the ring from the back, v down it."""
    angle = np.linspace(-np.pi, np.pi, segments + 1)
    top = np.stack([radius * np.sin(angle), np.full_like(angle, height / 2), radius * np.cos(angle)], 1)
    bottom = top * [1, -1, 1]
    positions = np.concatenate([top, bottom])
    u = (angle + np.pi) / (2 * np.pi)
    uvs = np.concatenate([np.stack([u, np.zeros_like(u)], 1), np.stack([u, np.ones_like(u)], 1)])
    n = segments + 1
    faces = []
    for i in range(segments):
        faces += [[i, n + i, n + i + 1], [i, n + i + 1, i + 1]]
    tilt = np.radians(tilt_deg)
    turn = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    return positions @ turn.T, np.array(faces), uvs


def _striped(width: int = 360, height: int = 32, seen_deg: float = 50.0, period_deg: float = 12.0):
    """Front arc (|angle| < seen_deg) in black and white stripes, the rest plain grey; column = angle + 180."""
    texture = np.full((height, width, 3), 128, dtype=np.uint8)
    angle = (np.arange(width) + 0.5) / width * 360 - 180
    front = np.abs(angle) < seen_deg
    stripes = np.where(np.floor(angle / period_deg) % 2 == 0, 255, 0).astype(np.uint8)
    texture[:, front] = stripes[front][None, :, None]
    return texture


# --- the pieces ---------------------------------------------------------------------------------------------------


def test_a_circle_is_found_from_a_quarter_of_it():
    angle = np.linspace(-0.8, 0.8, 50)
    cx, cz, radius = rp.circle_fit(0.3 + 2.0 * np.sin(angle), -1.0 + 2.0 * np.cos(angle))
    assert (cx, cz, radius) == pytest.approx((0.3, -1.0, 2.0), abs=1e-6)


def test_the_ring_frame_finds_a_tilted_ring():
    positions, faces, _ = _ring(tilt_deg=10.0)
    normals = pp.vertex_normals(positions, faces) * (pp.outward_sign(positions, faces) or 1.0)
    frame = rp.ring_frame(positions, normals)
    assert frame.radius == pytest.approx(0.5, rel=0.03)
    assert (frame.cx, frame.cz) == pytest.approx((0.0, 0.0), abs=0.01)
    # tilted about x: the front (+z, angle 0) sits lower by r * sin(10 degrees)
    assert frame.tilt[1] == pytest.approx(-0.5 * np.sin(np.radians(10)), rel=0.1)
    assert frame.outward.all()


def test_the_seen_arc_is_cut_where_it_holds_the_least_detail():
    theta = np.linspace(-1.0, 1.0, 2001)
    luminance = np.where(np.arange(len(theta)) % 2, 0.0, 255.0)     # busy everywhere ...
    luminance[np.abs(theta + 0.83) < 0.04] = 100.0                  # ... but in two quiet gaps
    luminance[np.abs(theta - 0.9) < 0.04] = 100.0
    start, end = rp.cut_arc(theta, luminance, np.ones_like(theta, dtype=bool))
    assert start == pytest.approx(-0.83, abs=0.03) and end == pytest.approx(0.9, abs=0.03)


def test_the_arc_repeats_round_the_ring():
    theta = np.linspace(-np.pi, np.pi, 3601)[:-1]
    height = np.zeros_like(theta)
    colours = np.zeros((len(theta), 3))
    source = np.abs(theta) <= 0.5
    colours[source, 0] = theta[source]               # a ramp along the seen arc
    target = ~source
    out = rp.repeat_arc(theta, height, colours, source, target, (-0.5, 0.5), radius=1.0)
    expected = -0.5 + (theta[target] + 0.5) % 1.0
    assert out[:, 0] == pytest.approx(expected, abs=2e-3)


# --- a whole ring through a camera --------------------------------------------------------------------------------


def _painted_ring():
    positions, faces, uvs = _ring()
    texture = _striped()
    view = _camera()
    x, y, _ = pp.project(positions, view)
    box = (x.min() - 2, y.min() - 2, x.max() + 2, y.max() + 2)
    return positions, faces, uvs, texture, view, box


def _column_std(texture, angle_deg):
    column = int((angle_deg + 180) / 360 * texture.shape[1])
    return texture[4:-4, column - 8:column + 8].astype(float).std()


def test_the_back_takes_the_front_stripes():
    positions, faces, uvs, texture, view, box = _painted_ring()
    before = _column_std(texture, 180 - 10)
    out, report = rp.paint_ring(texture, positions, uvs, faces, view, box, frame=IDENTITY)
    assert before == 0.0                                 # plain grey behind
    assert _column_std(out, 180 - 10) > 80               # stripes now
    assert _column_std(out, 90) > 80 and _column_std(out, -90) > 80
    assert report["repainted"] > 0 and 60 < report["arc_degrees"] < 110


def test_the_seen_front_keeps_its_own_paint():
    positions, faces, uvs, texture, view, box = _painted_ring()
    out, _ = rp.paint_ring(texture, positions, uvs, faces, view, box, frame=IDENTITY)
    middle = slice(int(160 / 360 * texture.shape[1]), int(200 / 360 * texture.shape[1]))
    assert np.array_equal(out[:, middle], texture[:, middle])


def test_surfaces_outside_the_box_keep_their_paint():
    positions, faces, uvs, texture, view, box = _painted_ring()
    # a box over the ring's lower half only: the top row of texels keeps its paint everywhere
    lower = (box[0], (box[1] + box[3]) / 2, box[2], box[3])
    out, _ = rp.paint_ring(texture, positions, uvs, faces, view, lower, frame=IDENTITY)
    assert np.array_equal(out[0], texture[0])
    assert not np.array_equal(out[-1], texture[-1])


# --- the GLB --------------------------------------------------------------------------------------------------------


def test_a_glb_gets_its_ring_repainted_and_nothing_else():
    positions, faces, uvs, texture, view, box = _painted_ring()
    mesh = trimesh.Trimesh(positions, faces, process=False)
    # trimesh writes glTF's v flipped (its UVs follow OpenGL)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs * [1, -1] + [0, 1], image=Image.fromarray(texture))
    data = mesh.export(file_type="glb")
    out, reports = rp.paint_glb(data, view, [box], frame=IDENTITY)
    doc, _ = pp._split_glb(out)
    old_doc, _ = pp._split_glb(data)
    assert len(doc["meshes"][0]["primitives"]) == 1 and reports[0]["primitive"] == 0
    assert doc["accessors"] == old_doc["accessors"]          # the shape is untouched
    image = rp.read_primitive(out, 0)[3]
    assert _column_std(image, 180 - 10) > 80


def test_a_face_spread_over_the_whole_picture_is_left_out():
    # a cut's closing cap can carry corner UVs that span the whole atlas; it has no island of its own, and walking
    # every texel of its box for each such face asked for 24 GB on a real figure
    uvs = np.array([[0.1, 0.1], [0.12, 0.1], [0.1, 0.12], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    index, face, _ = rp.texels(uvs, faces, (512, 512))
    assert len(index) > 0 and set(face.tolist()) == {0}
