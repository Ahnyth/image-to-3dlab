"""Tests for `scripts/game_ready.py`: the parts of the game-ready step that need no Blender.

Colour, weight and bone-plan maths, the PNG writer and the Blender command line are plain
functions, imported here exactly as `scripts/blender_game_ready.py` imports them.
"""

import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import game_ready as gr
import kimodo_retarget as kr
import unity_humanoid_names as uhn
from test_kimodo_retarget import _humanoid

# --- settings -------------------------------------------------------------------------------

def test_a_colour_reads_as_srgb_floats():
    assert gr.parse_colour("#19e6ff") == pytest.approx((0x19 / 255, 0xE6 / 255, 1.0))
    assert gr.parse_colour("19E6FF") == pytest.approx((0x19 / 255, 0xE6 / 255, 1.0))


@pytest.mark.parametrize("bad", ["", "#12345", "#gggggg", "red", "#19e6ff00"])
def test_a_colour_that_is_not_six_hex_digits_is_refused(bad):
    with pytest.raises(ValueError):
        gr.parse_colour(bad)


def test_the_defaults_change_nothing_a_user_did_not_ask_for():
    s = gr.settings([])
    assert s["readability"] == 0.0 and s["glow"] == [] and s["fingers"] == "keep"
    assert s["style"] == "unity" and s["max_influences"] == 4 and s["atlas"] == 1024


def test_settings_read_the_command_line():
    s = gr.settings(["--height", "1.95", "--atlas", "512", "--readability", "1", "--glow", "#19e6ff",
                     "--glow", "ff3010", "--fingers", "merge", "--style", "mixamo"])
    assert s["height"] == 1.95 and s["atlas"] == 512 and s["readability"] == 1.0
    assert s["glow"] == [pytest.approx(gr.parse_colour("#19e6ff")), pytest.approx(gr.parse_colour("#ff3010"))]
    assert s["fingers"] == "merge" and s["style"] == "mixamo"


@pytest.mark.parametrize("args", [["--readability", "1.5"], ["--height", "0"], ["--atlas", "1000"],
                                  ["--max-influences", "0"]])
def test_settings_out_of_range_are_refused(args):
    with pytest.raises(SystemExit):
        gr.settings(args)


# --- colour -----------------------------------------------------------------------------------

def test_srgb_and_linear_are_inverse():
    x = np.linspace(0, 1, 101)
    assert gr.to_srgb(gr.to_linear(x)) == pytest.approx(x, abs=1e-6)


def test_curve_strength_zero_is_the_identity_and_one_is_the_tested_curve():
    assert gr.curve_params(0.0) == pytest.approx((0.0, 1.0, 1.0, 1.0))
    assert gr.curve_params(1.0) == pytest.approx((0.02, 1.06, 0.90, 1.45))


def test_value_curve_at_zero_changes_nothing():
    rng = np.random.default_rng(1)
    srgb = rng.uniform(0.1, 0.5, (16, 16, 3))
    inside = np.ones((16, 16), bool)
    assert gr.value_curve(srgb, inside, 0.0) == pytest.approx(srgb)


def test_value_curve_lifts_the_darkest_and_spreads_the_light_parts():
    # A dark figure: most texels near 0.2, a few plates at 0.35, the deepest shadow at 0.1.
    srgb = np.full((10, 10, 3), 0.2)
    srgb[0, :3] = 0.1
    srgb[9, :5] = 0.35
    out = gr.value_curve(srgb, np.ones((10, 10), bool), 1.0)
    assert out.min() > 0.1                        # the darkest is lifted
    assert out[9, 0, 0] - out[5, 5, 0] > 0.35 - 0.2  # plates stand further apart from the body
    assert out.max() <= 1.0


def test_an_accent_is_found_by_its_colour_and_nothing_else():
    cyan = gr.parse_colour("#29e6ff")
    texels = np.array([[(0.10, 0.60, 0.65),    # a cyan light strip, darker than the pick
                        (0.40, 0.40, 0.40),    # grey
                        (0.20, 0.22, 0.25),    # bluish grey suit
                        (0.20, 0.60, 0.20),    # green
                        (0.70, 0.20, 0.10)]])  # red
    m = gr.accent_mask(texels, cyan)
    assert m[0, 0] > 0.95
    assert m[0, 1:].max() < 0.05


def test_downsample_is_a_box_filter():
    a = np.arange(16, dtype=float).reshape(4, 4, 1)
    assert gr.downsample(a, 2)[..., 0] == pytest.approx(np.array([[2.5, 4.5], [10.5, 12.5]]))


def test_dilate_grows_a_texel_into_a_disc():
    a = np.zeros((9, 9))
    a[4, 4] = 1.0
    d = gr.dilate(a, 2)
    assert d[4, 6] == 1.0 and d[6, 4] == 1.0 and d[4, 7] == 0.0


def test_push_out_fills_the_empty_texels_next_to_an_island_only():
    rgb = np.zeros((1, 8, 3))
    rgb[0, :2] = (1.0, 0.5, 0.25)
    have = np.zeros((1, 8), bool)
    have[0, :2] = True
    out, filled = gr.push_out(rgb, have, 3)
    assert filled[0, :5].all() and not filled[0, 5:7].any()
    assert out[0, 4] == pytest.approx((1.0, 0.5, 0.25))


def _atlas(n=16):
    """A dark grey island with a cyan strip, the right half of the atlas empty."""
    alb = np.zeros((n, n, 3))
    cov = np.zeros((n, n))
    alb[:, : n // 2] = gr.to_linear(np.array((0.22, 0.22, 0.24)))
    alb[4:6, 2:6] = gr.to_linear(np.array((0.10, 0.60, 0.65)))
    cov[:, : n // 2] = 1.0
    lit = np.full((n, n), 0.5)
    return alb, lit, cov


def test_compose_without_options_keeps_the_painted_colour():
    alb, lit, cov = _atlas()
    s = gr.settings(["--no-shade"])
    out, emission, stats = gr.compose(alb, lit, cov, s)
    assert out[:, :8] == pytest.approx(gr.to_srgb(alb[:, :8]), abs=1e-6)
    assert emission is None and stats["accent_share"] == 0.0


def test_compose_pads_the_empty_atlas_with_the_island_colours():
    alb, lit, cov = _atlas()
    out, _, _ = gr.compose(alb, lit, cov, gr.settings(["--no-shade"]))
    assert out[10, 8] == pytest.approx(out[10, 7], abs=1e-6)  # pushed out next to the island
    assert np.isfinite(out).all() and out[:, 15].min() > 0.0  # far texels get the mean, never black


def test_compose_glow_lights_the_accent_and_nothing_else():
    alb, lit, cov = _atlas()
    s = gr.settings(["--no-shade", "--glow", "#29e6ff"])
    out, emission, stats = gr.compose(alb, lit, cov, s)
    assert emission is not None
    assert emission[5, 3].max() > 0.8
    assert emission[12, 3].max() == 0.0
    assert out[5, 3] == pytest.approx(gr.parse_colour("#29e6ff"), abs=0.02)
    assert 0.0 < stats["accent_share"] < 0.2


def test_compose_shade_darkens_what_the_sky_does_not_reach():
    alb, lit, cov = _atlas()
    lit[8:, :] = 0.05  # a part facing down
    out, _, _ = gr.compose(alb, lit, cov, gr.settings([]))
    assert out[12, 3].mean() < out[1, 3].mean()


def _read_png(path):
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    w = h = 0
    while pos < len(data):
        n, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    rows = []
    for y in range(h):
        line = bytearray(raw[y * (1 + 3 * w):(y + 1) * (1 + 3 * w)])
        assert line[0] == 1  # Sub filter
        px = line[1:]
        for i in range(3, len(px)):
            px[i] = (px[i] + px[i - 3]) & 0xFF
        rows.append(np.frombuffer(bytes(px), np.uint8).reshape(w, 3))
    return np.array(rows)


def test_png_writer_round_trips_with_row_zero_at_the_bottom(tmp_path):
    rgb = np.zeros((2, 3, 3))
    rgb[0] = (1.0, 0.0, 0.0)  # Blender's row 0 is the bottom row of the picture
    rgb[1, 2] = (0.0, 0.5, 1.0)
    gr.write_png(tmp_path / "t.png", rgb)
    img = _read_png(tmp_path / "t.png")
    assert img.shape == (2, 3, 3)
    assert tuple(img[1, 0]) == (255, 0, 0)
    assert tuple(img[0, 2]) == (0, 128, 255)


# --- bones and weights ----------------------------------------------------------------------

def _unity(parents, heads):
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    return uhn.engine_names(parents, mapping, "unity")


def test_merging_fingers_sends_every_finger_bone_to_its_hand():
    parents, heads, roles = _humanoid()
    plan = gr.bone_plan(parents, _unity(parents, heads), "merge")
    for lr in ("L", "R"):
        hand = roles[f"{lr}hand"]
        fingers = [roles[f"{lr}{f}{j}"] for f in ("thumb", "index", "middle", "ring", "pinky") for j in range(3)]
        assert all(plan[f] == hand for f in fingers)
    assert roles["Lhand"] not in plan and roles["head"] not in plan


def test_keeping_fingers_merges_nothing():
    parents, heads, _ = _humanoid()
    assert gr.bone_plan(parents, _unity(parents, heads), "keep") == {}


def test_a_skeleton_without_named_hands_merges_nothing():
    assert gr.bone_plan({"a": None, "b": "a"}, {}, "merge") == {}


def test_merge_columns_adds_weights_into_the_target_and_drops_the_bone():
    W = np.array([[0.5, 0.2, 0.2, 0.1], [0.0, 0.0, 1.0, 0.0]])
    out, cols = gr.merge_columns(W, ["hips", "hand", "f1", "f2"], {"f1": "hand", "f2": "hand"})
    assert cols == ["hips", "hand"]
    assert out == pytest.approx(np.array([[0.5, 0.5], [0.0, 1.0]]))


def test_merge_columns_makes_a_column_for_a_target_without_weights():
    out, cols = gr.merge_columns(np.array([[0.3, 0.7]]), ["hips", "f1"], {"f1": "hand"})
    assert cols == ["hips", "hand"] and out == pytest.approx(np.array([[0.3, 0.7]]))


def test_limit_keeps_the_strongest_four_and_renormalises():
    W = np.array([[0.30, 0.25, 0.20, 0.15, 0.06, 0.04], [0, 0, 0, 0, 0, 0], [0.995, 0.005, 0, 0, 0, 0]])
    out = gr.limit_influences(W, 4)
    assert (out[0] > 0).sum() == 4 and out[0].sum() == pytest.approx(1.0)
    assert out[0, :4] == pytest.approx(np.array([0.30, 0.25, 0.20, 0.15]) / 0.90)
    assert out[1].sum() == 0.0  # an unweighted vertex stays unweighted, reported, not invented
    assert out[2] == pytest.approx([1, 0, 0, 0, 0, 0])  # below 1 % is dropped


def test_smoothing_moves_weights_towards_the_neighbours_and_keeps_the_sum():
    W = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    out = gr.smooth_weights(W, edges, 1, 0.5)
    assert 0.0 < out[1, 1] < 0.5 and 0.0 < out[2, 0] < 0.5
    assert out.sum(1) == pytest.approx(np.ones(4))
    assert gr.smooth_weights(W, edges, 0, 0.5) == pytest.approx(W)


def test_shells_are_the_connected_vertex_sets():
    found = gr.shells_from_edges(6, np.array([[0, 1], [1, 2], [3, 4]]))
    assert sorted(sorted(s.tolist()) for s in found) == [[0, 1, 2], [3, 4], [5]]


def test_a_small_separate_piece_moves_as_one():
    # a pouch of three vertices next to a body of five (Fable's rule: a piece is never half the mesh)
    W = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [1, 0], [1, 0], [1, 0]])
    co = np.array([[0, 0, 0], [0, 0, 0.05], [0, 0.02, 0], [0, 0, 0], [0, 0, 2.0], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
                  float)
    shells = [np.array([0, 1, 2]), np.array([3, 4, 5, 6, 7])]
    out, n = gr.rigid_shells(W, shells, co, max_diag=0.26)
    assert n == 1
    assert out[0] == pytest.approx([0.5, 0.5]) and out[1] == pytest.approx([0.5, 0.5])
    assert out[3] == pytest.approx([1.0, 0.0])  # the big shell is left alone


def test_the_body_is_never_made_rigid_even_if_it_is_small():
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    out, n = gr.rigid_shells(W, [np.array([0, 1])], np.zeros((2, 3)), max_diag=1.0)
    assert n == 0 and out == pytest.approx(W)


# --- measuring ------------------------------------------------------------------------------

def test_uv_area_is_the_share_of_the_atlas_covered():
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    tris = np.array([[0, 1, 2], [1, 3, 2]])
    assert gr.uv_area(uv, tris) == pytest.approx(1.0)
    assert gr.uv_area(uv * 0.5, tris[:1]) == pytest.approx(0.125)


def test_engine_vertices_count_each_split_once():
    vert = np.array([0, 0, 1, 1])
    normal = np.zeros((4, 3))
    uv = np.array([[0, 0], [0.5, 0], [0.2, 0.2], [0.2, 0.2]])
    assert gr.engine_vertices(vert, normal, uv) == 3


def test_problems_name_what_a_game_engine_would_trip_over():
    good = {"materials": 1, "max_influences": 4, "unweighted": 0, "height": 1.8, "min_z": 0.0,
            "animation_error_m": 0.0}
    assert gr.problems(good, max_influences=4, height=1.8) == []
    bad = {"materials": 2, "max_influences": 5, "unweighted": 3, "height": 1.5, "min_z": 0.2,
           "animation_error_m": 0.01}
    found = " ".join(gr.problems(bad, max_influences=4, height=1.8))
    for word in ("materials", "influences", "unweighted", "height", "floor", "moves"):
        assert word in found


def test_blender_runs_headless_and_fails_loudly(tmp_path):
    cmd = gr.blender_command(Path("blender.exe"), tmp_path / "job.json")
    assert cmd[:5] == ["blender.exe", "--background", "--python-exit-code", "1", "--python"]
    assert cmd[5].endswith("blender_game_ready.py") and cmd[6:] == ["--", str(tmp_path / "job.json")]
