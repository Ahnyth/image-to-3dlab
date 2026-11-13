"""Tests for `scripts/prop_ready.py`: the parts of the prop step that need no Blender.

Turning the front to +Z, the real size, the pivot on the floor, the collider box and the LOD
plan are plain functions, imported here exactly as `scripts/blender_prop_ready.py` imports
them.
"""

import math
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import prop_ready as pr
from compress_glb_textures import parse_glb


def _turn(direction, degrees):
    """A horizontal (x, z) direction turned about the up axis, as glTF and the Blender
    half turn it: counter-clockwise seen from above, +X towards -Z."""
    x, z = direction
    a = math.radians(degrees)
    return (round(x * math.cos(a) + z * math.sin(a), 9), round(-x * math.sin(a) + z * math.cos(a), 9))


# --- front ----------------------------------------------------------------------------------

@pytest.mark.parametrize("front, direction", [("+z", (0, 1)), ("-z", (0, -1)),
                                              ("+x", (1, 0)), ("-x", (-1, 0))])
def test_the_turn_brings_every_front_to_plus_z(front, direction):
    assert _turn(direction, pr.yaw_for_front(front)) == (0, 1)


def test_a_front_that_is_not_an_axis_is_refused():
    with pytest.raises(ValueError, match="front"):
        pr.yaw_for_front("forward")


def test_a_front_is_read_without_caring_about_case_or_a_missing_plus():
    assert pr.yaw_for_front("Z") == pr.yaw_for_front("+z")
    assert pr.yaw_for_front("-X") == pr.yaw_for_front("-x")


# --- real size ------------------------------------------------------------------------------

def test_no_size_keeps_the_scale():
    assert pr.scale_for((2.0, 1.0, 5.0)) == 1.0


def test_a_length_is_measured_along_the_front():
    # extent is (width x, height y, length z) once the front points to +Z
    assert pr.scale_for((0.5, 0.4, 1.0), length=12.0) == pytest.approx(12.0)
    assert pr.scale_for((0.5, 0.4, 1.0), height=3.2) == pytest.approx(8.0)
    assert pr.scale_for((0.5, 0.4, 1.0), width=2.5) == pytest.approx(5.0)


def test_only_one_size_can_set_the_scale():
    with pytest.raises(ValueError, match="one of"):
        pr.scale_for((1.0, 1.0, 1.0), length=12.0, height=3.0)


def test_a_flat_side_cannot_be_scaled_to_a_size():
    with pytest.raises(ValueError, match="flat"):
        pr.scale_for((1.0, 0.0, 1.0), height=2.0)


# --- pivot ----------------------------------------------------------------------------------

def test_the_pivot_goes_to_the_bottom_centre():
    low, high = (-1.0, 0.3, 2.0), (3.0, 2.3, 8.0)
    move = pr.floor_translation(low, high)
    assert move == pytest.approx((-1.0, -0.3, -5.0))
    placed_low = [a + b for a, b in zip(low, move)]
    placed_high = [a + b for a, b in zip(high, move)]
    assert placed_low[1] == pytest.approx(0.0)
    assert (placed_low[0] + placed_high[0]) / 2 == pytest.approx(0.0)
    assert (placed_low[2] + placed_high[2]) / 2 == pytest.approx(0.0)


# --- collider -------------------------------------------------------------------------------

def test_the_box_stands_on_the_floor_around_the_pivot():
    box = pr.collider_box((2.55, 3.27, 12.0))
    assert box == {"centre": [0.0, pytest.approx(1.635), 0.0], "size": [2.55, 3.27, 12.0]}


def test_the_box_glb_is_a_closed_box_of_twelve_triangles_named_as_asked():
    data = pr.box_glb({"centre": [0.0, 1.5, 0.0], "size": [2.0, 3.0, 10.0]}, "bus_COL")
    document, binary = parse_glb(data)
    assert document["nodes"][0]["name"] == "bus_COL"
    assert document["meshes"][0]["name"] == "bus_COL"
    primitive = document["meshes"][0]["primitives"][0]
    positions = document["accessors"][primitive["attributes"]["POSITION"]]
    indices = document["accessors"][primitive["indices"]]
    assert positions["count"] == 8 and indices["count"] == 36
    assert positions["min"] == [-1.0, 0.0, -5.0] and positions["max"] == [1.0, 3.0, 5.0]
    view = document["bufferViews"][indices["bufferView"]]
    triangles = struct.unpack_from("<36H", binary, view.get("byteOffset", 0))
    # closed: every edge is shared by exactly two triangles
    edges = {}
    for i in range(0, 36, 3):
        a, b, c = triangles[i:i + 3]
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            edges[key] = edges.get(key, 0) + 1
    assert set(edges.values()) == {2}


def test_the_box_faces_outwards():
    data = pr.box_glb({"centre": [0.0, 0.5, 0.0], "size": [1.0, 1.0, 1.0]}, "c")
    document, binary = parse_glb(data)
    primitive = document["meshes"][0]["primitives"][0]
    pos_view = document["bufferViews"][document["accessors"][primitive["attributes"]["POSITION"]]["bufferView"]]
    idx_view = document["bufferViews"][document["accessors"][primitive["indices"]]["bufferView"]]
    points = struct.unpack_from("<24f", binary, pos_view.get("byteOffset", 0))
    points = [points[i:i + 3] for i in range(0, 24, 3)]
    triangles = struct.unpack_from("<36H", binary, idx_view.get("byteOffset", 0))
    for i in range(0, 36, 3):
        a, b, c = (points[j] for j in triangles[i:i + 3])
        u = [b[k] - a[k] for k in range(3)]
        v = [c[k] - a[k] for k in range(3)]
        normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        middle = [(a[k] + b[k] + c[k]) / 3 - (0.0, 0.5, 0.0)[k] for k in range(3)]
        assert sum(n * m for n, m in zip(normal, middle)) > 0


# --- LODs -----------------------------------------------------------------------------------

def test_each_further_lod_gets_half_the_atlas_down_to_the_floor():
    assert pr.lod_plan([2500, 1000, 400], 4096) == [(1, 2500, 2048), (2, 1000, 1024), (3, 400, 512)]
    assert pr.lod_plan([2500, 1000], 512) == [(1, 2500, 256), (2, 1000, 256)]


def test_lod_names_follow_finish_props():
    assert pr.lod_name("bus", 0) == "bus_LOD0"
    assert pr.lod_name("bus", 2) == "bus_LOD2"


# --- problems -------------------------------------------------------------------------------

def _record(**changes):
    record = {"size_m": [2.55, 3.27, 12.0], "floor_y": 0.0,
              "lods": [{"triangles": 5000}, {"triangles": 2500}, {"triangles": 1000}],
              "collider": {"kind": "box"}}
    record.update(changes)
    return record


def test_a_good_prop_has_no_problems():
    assert pr.problems(_record()) == []


def test_a_lod_that_is_not_lighter_than_the_one_before_is_named():
    found = pr.problems(_record(lods=[{"triangles": 5000}, {"triangles": 5200}]))
    assert any("LOD1" in line for line in found)


def test_a_lod_file_heavier_than_the_one_before_is_named():
    # a LOD baked with uncompressed textures came out larger than LOD0 (the ambulance, 10.9 MB
    # against 6.6 MB) before the LODs were compressed too
    lods = [{"triangles": 5000, "bytes": 6_600_000}, {"triangles": 2500, "bytes": 10_900_000}]
    assert any("LOD1" in line and "MB" in line for line in pr.problems(_record(lods=lods)))


def test_a_hull_over_the_convex_collider_limit_is_named():
    found = pr.problems(_record(collider={"kind": "hull", "triangles": 300}))
    assert any("255" in line for line in found)


def test_a_prop_off_the_floor_is_named():
    assert any("floor" in line for line in pr.problems(_record(floor_y=0.05)))


def test_a_size_no_vehicle_or_prop_has_is_named():
    # Pixal3D builds in a unit cube: a prop left unscaled is a metre long, a scale typed
    # wrong can make it a hundred
    assert any("size" in line for line in pr.problems(_record(size_m=[25.0, 30.0, 120.0])))


# --- command line ---------------------------------------------------------------------------

def test_lods_are_read_as_face_counts_after_lod0():
    args = pr.build_parser().parse_args(["in.glb", "out", "--name", "bus", "--lods", "2500,1000"])
    assert pr.parse_lods(args.lods) == [2500, 1000]
    assert args.front == "+z" and args.collider == "box" and args.hull_faces == 255


def test_lods_must_get_lighter():
    with pytest.raises(SystemExit):
        pr.parse_lods("1000,2500")


def test_no_lods_is_allowed():
    assert pr.parse_lods("") == []


def test_the_blender_command_runs_headless_with_the_job():
    command = pr.blender_command(Path("blender"), Path("job.json"))
    assert command[:2] == ["blender", "--background"]
    assert "--factory-startup" in command and "--python-exit-code" in command
    assert command[-2:] == ["--", "job.json"]
    assert any(part.endswith("blender_prop_ready.py") for part in command)
