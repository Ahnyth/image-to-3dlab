"""Tests for `image_to_3dlab/colour_compare.py`: two models of the same figure compared by colour.

A check that counts holes in motion does not see a wrong colour; a game model's bake once put a
dark, faceted patch on a cape's back that only a look at the colour showed. These tests use a
cube whose six sides each read their own strip of a small texture.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab import colour_compare as cc

REPO = Path(__file__).resolve().parents[1]

# side: (normal axis, sign, the two in-plane axes)
SIDES = [(2, 1, 0, 1), (2, -1, 0, 1), (0, 1, 2, 1), (0, -1, 2, 1), (1, 1, 0, 2), (1, -1, 0, 2)]
COLOURS = [(200, 60, 50), (40, 90, 160), (60, 150, 70), (210, 190, 60), (150, 150, 150), (90, 60, 40)]


def cube(n: int = 1, jitter: float = 0.0, seed: int = 0):
    """A unit cube centred on the origin, each side an n-by-n grid of quads reading strip k of the texture."""
    rng = np.random.default_rng(seed)
    positions, uvs, faces = [], [], []
    for k, (axis, sign, a, b) in enumerate(SIDES):
        base = len(positions)
        for i in range(n + 1):
            for j in range(n + 1):
                p = np.zeros(3)
                p[axis] = 0.5 * sign
                p[a], p[b] = i / n - 0.5, j / n - 0.5
                if 0 < i < n and 0 < j < n:
                    p += rng.normal(0, jitter, 3) * (np.arange(3) != axis)
                positions.append(p)
                uvs.append(((k + 0.1 + 0.8 * i / n) / 6, 0.1 + 0.8 * j / n))
        for i in range(n):
            for j in range(n):
                q = base + i * (n + 1) + j
                quad = (q, q + n + 1, q + n + 2, q + 1)
                tri1, tri2 = (quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])
                for t in (tri1, tri2):   # wind each triangle so it faces out
                    v = np.array([positions[x] for x in t])
                    if np.cross(v[1] - v[0], v[2] - v[0])[axis] * sign < 0:
                        t = (t[0], t[2], t[1])
                    faces.append(t)
    return np.array(positions, float), np.array(uvs, float), np.array(faces, np.int64)


def texture(dark_side: int | None = None, speck: bool = False) -> np.ndarray:
    tex = np.zeros((64, 6 * 64, 3), np.uint8)
    for k, colour in enumerate(COLOURS):
        tex[:, k * 64:(k + 1) * 64] = colour
        tex[:, k * 64:(k + 1) * 64] += (np.arange(64)[:, None, None] // 8 * 3).astype(np.uint8)   # some shading
    if dark_side is not None:
        tex[16:40, dark_side * 64 + 16:dark_side * 64 + 40] //= 3   # a dark square in that side's middle
    if speck:
        tex[30:32, 1 * 64 + 30:1 * 64 + 32] = 0   # two by two texels
    return tex


def model(**kw) -> cc.Model:
    tex_kw = {k: kw.pop(k) for k in ("dark_side", "speck") if k in kw}
    p, uv, f = cube(**kw)
    return cc.Model.single(p, uv, f, texture(**tex_kw))


def test_the_same_model_twice_has_no_patches():
    assert cc.compare(model(), model()) == []


def test_a_dark_patch_is_found_on_the_side_it_is_on():
    patches = cc.compare(model(), model(dark_side=1))   # side 1 faces -z: the back
    assert patches, "the darker square was not found"
    views = {p.view for p in patches}
    assert "back" in views and "front" not in views
    back = next(p for p in patches if p.view == "back")
    assert back.darker and back.share > 0.05 and back.delta_e > 12
    assert "middle" in back.where


def test_a_finer_mesh_with_the_same_colour_has_no_patches():
    assert cc.compare(model(), model(n=6, jitter=0.01)) == []


def test_a_speck_below_the_smallest_patch_is_left_out():
    assert cc.compare(model(), model(speck=True)) == []


def test_the_views_go_round_the_figure_and_say_where_they_look_from():
    names = [name for name, _ in cc.VIEWS]
    assert names[:4] == ["front", "back", "left side", "right side"] and len(names) == 8
    front = dict(cc.VIEWS)["front"]
    assert front == pytest.approx((0.0, 0.0, 1.0))   # glTF's front is +Z


def _glb(parts, node_translation=None) -> bytes:
    """A minimal GLB: one mesh with a primitive and a PNG base colour per part, under one node."""
    import io

    from PIL import Image
    binary, views, accessors, images, materials, primitives = b"", [], [], [], [], []

    def add(data: bytes, target=None) -> int:
        nonlocal binary
        binary += b"\0" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary += data
        return len(views) - 1

    for part in parts:
        positions, uvs, faces, tex = part[:4]
        glow = part[4] if len(part) > 4 else None
        p = add(positions.astype(np.float32).tobytes())
        accessors.append({"bufferView": p, "componentType": 5126, "count": len(positions), "type": "VEC3",
                          "min": positions.min(0).tolist(), "max": positions.max(0).tolist()})
        u = add(uvs.astype(np.float32).tobytes())
        accessors.append({"bufferView": u, "componentType": 5126, "count": len(uvs), "type": "VEC2"})
        i = add(faces.astype(np.uint32).tobytes())
        accessors.append({"bufferView": i, "componentType": 5125, "count": faces.size, "type": "SCALAR"})
        buf = io.BytesIO()
        Image.fromarray(tex).save(buf, "PNG")
        images.append({"bufferView": add(buf.getvalue()), "mimeType": "image/png"})
        material = {"pbrMetallicRoughness": {"baseColorTexture": {"index": len(images) - 1}}}
        if glow is not None:
            buf = io.BytesIO()
            Image.fromarray(glow).save(buf, "PNG")
            images.append({"bufferView": add(buf.getvalue()), "mimeType": "image/png"})
            material["emissiveTexture"] = {"index": len(images) - 1}
            material["emissiveFactor"] = [1.0, 1.0, 1.0]
        materials.append(material)
        primitives.append({"attributes": {"POSITION": len(accessors) - 3, "TEXCOORD_0": len(accessors) - 2},
                           "indices": len(accessors) - 1, "material": len(materials) - 1})
    node = {"mesh": 0}
    if node_translation is not None:
        node["translation"] = list(node_translation)
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [node],
           "meshes": [{"primitives": primitives}], "accessors": accessors, "bufferViews": views,
           "buffers": [{"byteLength": len(binary)}], "images": images, "materials": materials,
           "textures": [{"source": k} for k in range(len(images))]}
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def test_a_glb_with_two_parts_and_a_moved_node_reads_as_one_model(tmp_path):
    p, uv, f = cube()
    half = len(f) // 2
    used = np.unique(f[:half])
    path = tmp_path / "two.glb"
    path.write_bytes(_glb([(p, uv, f[:half], texture()), (p, uv, f[half:], texture(dark_side=1))], (0, 2.0, 0)))
    m = cc.read_model(path)
    assert len(m.faces) == len(f) and len(m.textures) == 2
    assert m.positions[:, 1].min() == pytest.approx(1.5)   # the node's move applied
    assert set(np.unique(m.part)) == {0, 1} and len(used) > 0


def test_the_script_says_what_changed_and_exits_1(tmp_path):
    p, uv, f = cube()
    old, new = tmp_path / "old.glb", tmp_path / "new.glb"
    old.write_bytes(_glb([(p, uv, f, texture())]))
    new.write_bytes(_glb([(p, uv, f, texture(dark_side=1))]))
    picture = tmp_path / "marked.png"
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "colour_compare.py"), str(old), str(new),
                           "--picture", str(picture)], capture_output=True, text=True)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "back" in done.stdout and "darker" in done.stdout and picture.exists()
    same = subprocess.run([sys.executable, str(REPO / "scripts" / "colour_compare.py"), str(old), str(old)],
                          capture_output=True, text=True)
    assert same.returncode == 0, same.stdout + same.stderr
    assert "no patch" in same.stdout


def glow_texture(side: int) -> np.ndarray:
    """Black, except the square `texture(dark_side=side)` darkens: that part glows."""
    tex = np.zeros((64, 6 * 64, 3), np.uint8)
    tex[16:40, side * 64 + 16:side * 64 + 40] = (40, 230, 255)
    return tex


def test_a_patch_where_the_new_model_glows_on_purpose_is_left_out():
    """A game model may glow on purpose (game_ready's accents): against its source that is no fault."""
    p, uv, f = cube()
    glow = cc.Model.single(p, uv, f, glow_texture(1))
    assert cc.compare(model(), model(dark_side=1))
    assert cc.compare(model(), model(dark_side=1), ignore=glow) == []
    assert cc.compare(model(), model(dark_side=0), ignore=glow), "a patch elsewhere is still found"


def test_read_glow_reads_the_emissive_texture(tmp_path):
    p, uv, f = cube()
    lit, plain = tmp_path / "lit.glb", tmp_path / "plain.glb"
    lit.write_bytes(_glb([(p, uv, f, texture(), glow_texture(1))]))
    plain.write_bytes(_glb([(p, uv, f, texture())]))
    glow = cc.read_glow(lit)
    assert glow is not None and glow.textures[0].max() == 255
    assert cc.read_glow(plain) is None


def test_the_script_can_leave_out_where_the_new_model_glows(tmp_path):
    p, uv, f = cube()
    old, new = tmp_path / "old.glb", tmp_path / "new.glb"
    old.write_bytes(_glb([(p, uv, f, texture())]))
    new.write_bytes(_glb([(p, uv, f, texture(dark_side=1), glow_texture(1))]))
    script = str(REPO / "scripts" / "colour_compare.py")
    plain = subprocess.run([sys.executable, script, str(old), str(new)], capture_output=True, text=True)
    assert plain.returncode == 1, plain.stdout + plain.stderr
    lit = subprocess.run([sys.executable, script, str(old), str(new), "--ignore-glow"], capture_output=True, text=True)
    assert lit.returncode == 0, lit.stdout + lit.stderr
