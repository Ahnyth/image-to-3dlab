"""The game-ready atlas does not paint a texel twice.

Smart UV Project at game_ready's wide angle (72 degrees) folds an island onto itself where the
surface turns, so two faces share texels and the bake writes one over the other: a Wächter's
cape came out with a dark, faceted patch, its texels overwritten by hidden faces from inside
(black in the sky light, another colour in the albedo). The Finish fixed the same fold with
`blender_retopo_bake.unwrap_islands`; game_ready's new UVs use it too.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_retopo_uv_overlap import _full_version, _load, overlap_percent

from image_to_3dlab.blender import find_blender

REPO = Path(__file__).resolve().parents[1]

SCRIPT = '''
import sys, math, json, bpy
sys.path.insert(0, {scripts!r})
import blender_game_ready as bg
import game_ready as gr
bpy.ops.wm.read_factory_settings(use_empty=True)
# a ramp turning twice round: one connected surface whose faces all face up, so a projection at
# a wide angle keeps it one island stacked on itself
verts, faces = [], []
turns, steps, rings = 2, 64, 6
for i in range(steps * turns + 1):
    a = 2 * math.pi * i / steps
    for j in range(rings + 1):
        r = 0.3 + 0.7 * j / rings
        verts.append((r * math.cos(a), r * math.sin(a), 0.02 * a))
for i in range(steps * turns):
    for j in range(rings):
        k = i * (rings + 1) + j
        faces.append((k, k + 1, k + rings + 2, k + rings + 1))
data = bpy.data.meshes.new("ramp")
data.from_pydata(verts, [], faces)
data.uv_layers.new(name="UVMap")          # the source's own UVs, which new_uvs keeps as "src"
obj = bpy.data.objects.new("ramp", data)
bpy.context.scene.collection.objects.link(obj)
report = bg.new_uvs(obj, gr.settings([]))
mesh = obj.data
mesh.calc_loop_triangles()
uv = mesh.uv_layers["UVMap"].data
tris = [[list(uv[loop].uv) for loop in tri.loops] for tri in mesh.loop_triangles]
json.dump({{"tris": tris, "report": report}}, open({out!r}, "w"))
'''


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_the_game_ready_atlas_paints_no_texel_twice(tmp_path):
    version = _full_version(find_blender())
    if version is None or _load().unwrap_crashes(version):
        pytest.skip(f"Blender {version} crashes in the unwrap's overlap check (or its version is unknown)")
    script = tmp_path / "uvs.py"
    out = tmp_path / "uvs.json"
    script.write_text(SCRIPT.format(scripts=str(REPO / "scripts"), out=str(out)), encoding="utf-8")
    done = subprocess.run([str(find_blender()), "--background", "--factory-startup", "--python-exit-code", "1",
                           "--python", str(script)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    result = json.loads(out.read_text(encoding="utf-8"))
    assert overlap_percent(result["tris"]) < 1.0
    assert result["report"]["uv_overlapping_faces"] == 0
