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


def test_a_minus_front_is_given_with_an_equals_sign_as_the_help_says():
    # argparse reads "--front -z" as two options; "--front=-z" is the way, and the help says so
    args = pr.build_parser().parse_args(["in.glb", "out", "--name", "amb", "--front=-z"])
    assert pr.yaw_for_front(args.front) == 180.0
    assert "--front=-z" in pr.build_parser().format_help() and "--front=-x" in pr.__doc__


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


# --- parts: models with moving parts (route (b) vehicles and buildings) ---------------------
# A model built by hand keeps its wheels, doors and booms as separate parts with their pivots
# where they turn, so the engine can spin a wheel or open a door. Remeshing it into one piece
# faceted the ambulance's wheels and made it a decoration.

def _glb_with_image(image: bytes, mime: str = "image/png", nodes=None) -> bytes:
    """A GLB with one material whose base colour is `image`, and the given nodes."""
    from compress_glb_textures import build_glb

    binary = image + b"\x00" * (-len(image) % 4)
    document = {
        "asset": {"version": "2.0"}, "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes or [])))}],
        "nodes": nodes or [],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}],
        "images": [{"bufferView": 0, "mimeType": mime}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(image)}],
    }
    return build_glb(document, binary)


def _png(colour) -> bytes:
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (4, 4), colour).save(out, "PNG")
    return out.getvalue()


def _base_colour(glb: bytes):
    import io

    from PIL import Image

    document, binary = parse_glb(glb)
    view = document["bufferViews"][document["images"][0]["bufferView"]]
    start = view.get("byteOffset", 0)
    return Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGB").getpixel((0, 0))


@pytest.mark.parametrize("name, axis", [("wheel_FL", [1.0, 0.0, 0.0]), ("Wheel_RR", [1.0, 0.0, 0.0]),
                                        ("door_1a", [0.0, 1.0, 0.0]), ("door", [0.0, 1.0, 0.0])])
def test_wheels_turn_about_the_axle_and_doors_about_an_upright_hinge(name, axis):
    # once the front is +Z a vehicle's axles run along X and a door's hinge stands along Y
    assert pr.part_axis(name) == axis


def test_a_part_with_no_known_way_of_moving_has_no_axis():
    assert pr.part_axis("boom") is None
    assert pr.part_axis("body") is None


def test_only_the_parts_that_move_are_listed_with_their_pivots():
    parts = [{"name": "body", "pivot": [0, 0, 0]}, {"name": "interior", "pivot": [0, 0.5, 0]},
             {"name": "wheel_FL", "pivot": [1.0, 0.36, 2.5]}, {"name": "door_cab_L", "pivot": [-1.0, 1.0, 1.9]},
             {"name": "boom", "pivot": [1.5, 1.0, 0.0]}]
    listed = pr.moving_parts(parts)
    assert [part["name"] for part in listed] == ["wheel_FL", "door_cab_L", "boom"]
    assert listed[0] == {"name": "wheel_FL", "pivot": [1.0, 0.36, 2.5], "motion": "turn", "axis": [1.0, 0.0, 0.0]}
    assert listed[2]["axis"] is None and "motion" not in listed[2]


def test_a_vector_turns_with_the_prop_and_grows_with_its_scale():
    # the same turn as the front's: counter-clockwise seen from above, +X towards -Z
    assert pr.turn_vector([1.0, 0.0, 0.0], 90.0, 1.0) == pytest.approx([0.0, 0.0, -1.0])
    assert pr.turn_vector([0.0, 0.5, -0.7], 180.0, 2.0) == pytest.approx([0.0, 1.0, 1.4])


def test_a_part_that_says_how_it_moves_keeps_that_turned_with_the_prop():
    # a metro car's plug door (the kit writes it into the model as glTF extras): out of the skin, then forward.
    # The kit's front is -Z, so prop_ready turns it 180 degrees; the leaf's way out and its slide turn along
    parts = [{"name": "door_1a", "pivot": [-1.33, 1.1, 6.68], "motion": "slide",
              "out": [0.10, 0.0, 0.0], "slide": [0.0, 0.0, -0.70]}]
    listed = pr.moving_parts(parts, yaw=180.0, scale=1.0)
    assert listed[0]["motion"] == "slide" and "axis" not in listed[0]
    assert listed[0]["out"] == pytest.approx([-0.10, 0.0, 0.0])
    assert listed[0]["slide"] == pytest.approx([0.0, 0.0, 0.70])


def test_a_model_that_says_a_part_turns_overrides_the_name():
    parts = [{"name": "hatch_roof", "pivot": [0.0, 3.0, 1.0], "motion": "turn", "axis": [1.0, 0.0, 0.0]}]
    listed = pr.moving_parts(parts, yaw=90.0, scale=1.0)
    assert listed[0]["motion"] == "turn" and listed[0]["axis"] == pytest.approx([0.0, 0.0, -1.0])


def test_each_part_gets_its_share_of_a_lod_and_the_shares_add_up():
    triangles = {"body": 4000, "interior": 600, "wheel_FL": 400, "wheel_FR": 400, "wheel_RL": 400, "wheel_RR": 400}
    targets = pr.part_targets(triangles, 3100)
    assert abs(sum(targets.values()) - 3100) <= len(triangles)
    assert all(targets[name] <= triangles[name] for name in triangles)
    assert targets["body"] > targets["interior"]


def test_a_wheel_keeps_enough_triangles_to_stay_round_even_in_a_small_lod():
    triangles = {"body": 4000, "wheel_FL": 400, "wheel_FR": 400}
    targets = pr.part_targets(triangles, 600)
    assert targets["wheel_FL"] >= pr.PART_FLOOR and targets["wheel_FR"] >= pr.PART_FLOOR
    assert sum(targets.values()) <= 600 + len(triangles)


def test_a_part_already_under_the_floor_keeps_every_triangle():
    targets = pr.part_targets({"body": 3000, "handle": 40}, 1000)
    assert targets["handle"] == 40


def test_a_lod_as_large_as_the_model_keeps_every_part_whole():
    triangles = {"body": 3000, "wheel_FL": 300}
    assert pr.part_targets(triangles, 5000) == triangles


# A metro car's parts as the kit builds it (LOD0 8,016 triangles): 16 door leaves of 44, eight
# 0.84 m wheels of 572 on an 18 m car, body and interior. With one floor of 96 for every part
# its 1,000-triangle LOD2 came out at 1,663 and left the body 96 triangles.
METRO = {"body": 2036, "interior": 700, **{f"wheel_{n}": 572 for n in range(8)},
         **{f"door_{n}": 44 for n in range(16)}}
METRO_SHARES = {"body": 1.0, "interior": 0.95, **{f"wheel_{n}": 0.84 / 18 for n in range(8)},
                **{f"door_{n}": 1.95 / 18 for n in range(16)}}


def test_a_part_keeps_the_full_floor_at_a_bus_wheels_size_and_less_when_smaller():
    # a 1 m wheel on a 12 m bus is the size the floor of 96 was set on
    assert pr.part_floor(1.0 / 12) == pr.PART_FLOOR
    assert pr.part_floor(1.0) == pr.PART_FLOOR
    assert pr.part_floor(0.84 / 18) == round(pr.PART_FLOOR * 0.84 / 18 * 12)
    assert pr.part_floor(0.001) == pr.LEAST_FLOOR


def test_small_parts_of_a_long_vehicle_leave_the_body_its_share_of_a_small_lod():
    targets = pr.part_targets(METRO, 1000, shares=METRO_SHARES)
    assert abs(sum(targets.values()) - 1000) <= len(METRO)
    assert targets["body"] > 1000 / 4
    assert all(targets[f"wheel_{n}"] >= pr.LEAST_FLOOR for n in range(8))
    assert all(targets[f"door_{n}"] >= pr.LEAST_FLOOR for n in range(16))


def test_the_floors_together_take_at_most_their_share_and_the_lod_keeps_its_count():
    # without sizes every part starts at the full floor; the floors shrink alike to fit
    floors = pr.part_floors(METRO, 1000)
    assert sum(floors.values()) <= 1000 * pr.FLOOR_BUDGET + len(METRO)
    assert abs(sum(pr.part_targets(METRO, 1000).values()) - 1000) <= len(METRO)


def test_floors_that_fit_are_not_shrunk():
    floors = pr.part_floors({"body": 4000, "wheel_FL": 400, "wheel_FR": 400}, 600)
    assert floors == {"body": pr.PART_FLOOR, "wheel_FL": pr.PART_FLOOR, "wheel_FR": pr.PART_FLOOR}


def test_a_small_part_is_thinned_more_than_the_body():
    targets = pr.part_targets(METRO, 2500, shares=METRO_SHARES)
    assert targets["wheel_0"] / METRO["wheel_0"] < targets["body"] / METRO["body"]


def test_the_floors_of_each_part_follow_its_size():
    floors = pr.part_floors(METRO, 2500, shares=METRO_SHARES)
    assert floors["door_0"] <= METRO["door_0"]          # never more than a part has
    assert pr.LEAST_FLOOR <= floors["wheel_0"] < floors["body"]


def test_the_size_of_each_part_is_read_from_the_glb():
    from compress_glb_textures import build_glb

    binary = b"\x00" * 64
    document = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "metro_LOD0", "children": [1, 2]}, {"name": "body", "mesh": 0},
                  {"name": "wheel_FL", "mesh": 1}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}, {"primitives": [{"attributes": {"POSITION": 1}}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [-1.4, 0.0, -9.0], "max": [1.4, 3.94, 9.0]},
                      {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [-0.07, -0.42, -0.42], "max": [0.07, 0.42, 0.42]}],
        "buffers": [{"byteLength": 64}], "bufferViews": [{"buffer": 0, "byteLength": 64}],
    }
    sizes = pr.part_sizes(build_glb(document, binary))
    assert sizes == pytest.approx({"body": 18.0, "wheel_FL": 0.84})


def test_the_triangles_of_each_part_are_read_from_the_glb():
    from compress_glb_textures import build_glb

    binary = b"\x00" * 64
    document = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "bus_LOD0", "children": [1, 2]}, {"name": "body", "mesh": 0},
                  {"name": "wheel_FL", "mesh": 1}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]},
                   {"primitives": [{"attributes": {"POSITION": 0}, "indices": 2}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                      {"bufferView": 0, "componentType": 5123, "count": 30, "type": "SCALAR"},
                      {"bufferView": 0, "componentType": 5123, "count": 12, "type": "SCALAR"}],
        "buffers": [{"byteLength": 64}], "bufferViews": [{"buffer": 0, "byteLength": 64}],
    }
    assert pr.part_triangles(build_glb(document, binary)) == {"body": 10, "wheel_FL": 4}


def test_a_lod_of_a_parts_model_renames_only_its_root_and_keeps_the_part_names():
    nodes = [{"name": "bus_LOD0", "children": [1, 2]}, {"name": "body", "mesh": 0}, {"name": "wheel_FL", "mesh": 1}]
    renamed = pr.name_parts_lod(_glb_with_image(_png((0, 0, 0)), nodes=nodes), "bus_LOD2")
    document, _binary = parse_glb(renamed)
    assert [node["name"] for node in document["nodes"]] == ["bus_LOD2", "body", "wheel_FL"]


def test_the_painted_atlas_goes_into_the_parts_model_and_nothing_else_changes():
    nodes = [{"name": "bus_parts", "children": [1]}, {"name": "wheel_FL"}]
    parts = _glb_with_image(_png((255, 255, 255)), nodes=nodes)
    painted = _glb_with_image(_png((200, 30, 30)))
    swapped = pr.with_painted_atlas(parts, painted)
    assert _base_colour(swapped) == (200, 30, 30)
    assert parse_glb(swapped)[0]["nodes"] == nodes


def test_a_jpeg_atlas_from_finish_is_carried_over_too():
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 200, 30)).save(out, "JPEG", quality=95)
    painted = _glb_with_image(out.getvalue(), mime="image/jpeg")
    swapped = pr.with_painted_atlas(_glb_with_image(_png((0, 0, 0)), nodes=[{"name": "body"}]), painted)
    red, green, blue = _base_colour(swapped)
    assert green > 180 and red < 60 and blue < 60


def test_parts_are_asked_for_on_the_command_line():
    args = pr.build_parser().parse_args(["bus_parts.glb", "out", "--name", "bus", "--parts",
                                         "--paint-from", "bus_painted.glb"])
    assert args.parts is True and args.paint_from == Path("bus_painted.glb")
    assert pr.build_parser().parse_args(["in.glb", "out", "--name", "bus"]).parts is False


def test_the_place_job_tells_blender_to_keep_the_parts():
    job = pr.place_job(Path("in.glb"), Path("out.glb"), 180.0, {"height": 3.27}, "bus_LOD0", None, parts=True)
    assert job["parts"] is True and job["node"] == "bus_LOD0" and job["yaw"] == 180.0
    assert pr.place_job(Path("in.glb"), Path("out.glb"), 0.0, {}, "bus_LOD0", None)["parts"] is False


def test_a_parts_model_whose_wheel_lost_its_roundness_is_named():
    lods = [{"triangles": 5000}, {"triangles": 2500}]
    record = _record(lods=lods, parts=[{"name": "wheel_FL", "pivot": [1, 0.4, 2], "axis": [1, 0, 0],
                                        "triangles": [400, 60]}])
    assert any("wheel_FL" in line for line in pr.problems(record))


def test_a_wheel_a_triangle_or_two_under_its_floor_is_not_named():
    # Blender's decimation lands within a few triangles of its target: the bus's twin wheel came
    # out at 95 of 96 in LOD2, which is just as round
    record = _record(parts=[{"name": "wheel_RR", "pivot": [-0.94, 0.5, -2.65], "axis": [1, 0, 0],
                             "triangles": [828, 315, 95]}])
    assert pr.problems(record) == []


def test_a_small_wheel_at_its_own_smaller_floor_is_not_named():
    # a metro car's 0.84 m wheel is thinned to its size's floor in LOD2: as round as that distance needs
    record = _record(parts=[{"name": "wheel_F1L", "pivot": [-0.8, 0.42, 6.0], "motion": "turn", "axis": [1, 0, 0],
                             "triangles": [572, 140, 30], "floors": [0, 54, 31]}])
    assert pr.problems(record) == []


def test_a_wheel_well_under_its_own_floor_is_named():
    record = _record(parts=[{"name": "wheel_F1L", "pivot": [-0.8, 0.42, 6.0], "motion": "turn", "axis": [1, 0, 0],
                             "triangles": [572, 140, 12], "floors": [0, 54, 31]}])
    assert any("wheel_F1L" in line for line in pr.problems(record))
