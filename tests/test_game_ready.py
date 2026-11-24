"""Tests for `scripts/game_ready.py`: the parts of the game-ready step that need no Blender.

Colour, weight and bone-plan maths, the PNG writer and the Blender command line are plain
functions, imported here exactly as `scripts/blender_game_ready.py` imports them.
"""

import itertools
import json
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
    assert s["glow_grow"] == 2  # the widening the hand-made finish used
    assert gr.settings(["--glow-width", "0"])["glow_grow"] == 0  # thin lines stay as thin as painted


@pytest.mark.parametrize("args", [["--readability", "1.5"], ["--height", "0"], ["--atlas", "1000"],
                                  ["--max-influences", "0"], ["--glow-width", "-1"], ["--glow-width", "9"]])
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
    s = gr.settings(["--no-shade", "--glow", "#29e6ff", "--glow-max-area", "1"])  # a toy atlas: the strip is past the cap
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

def _square(side_m: float):
    """A flat square of `side_m` metres, two triangles, mapped onto the whole UV square."""
    co = np.array([[0, 0, 0], [side_m, 0, 0], [0, side_m, 0], [side_m, side_m, 0]], float)
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    return co, uv, np.array([[0, 1, 2], [1, 3, 2]])


def test_atlas_density_is_texels_per_cm_of_surface():
    co, uv, tris = _square(1.0)                       # 10,000 cm2 on 1,000 x 1,000 texels
    m = gr.atlas_density(co, uv, tris, tris, 1000)
    assert m["texels_per_cm"] == pytest.approx(10.0)
    assert m["atlas_covered"] == pytest.approx(1.0, abs=0.01)
    assert m["texels_twice"] == pytest.approx(0.0, abs=0.001)


def test_a_folded_island_is_not_counted_as_more_atlas():
    """The summed UV area counted a fold twice (the 55 % and 68 % 'used' of the Wachter's atlas were inflated so):
    two triangles laid on the same texels cover half the atlas, and every texel of that half is painted twice."""
    co = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
    uv = np.array([[0, 0], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]], float)
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    m = gr.atlas_density(co, uv, tris, tris, 400)
    assert m["atlas_covered"] == pytest.approx(0.5, abs=0.01)
    assert m["texels_twice"] == pytest.approx(1.0, abs=0.01)


def test_atlas_density_reads_uvs_by_corner():
    """Blender keeps UVs per corner (loop), positions per vertex: each triangle names both."""
    co, uv, tris = _square(0.5)
    loops = np.array([[0, 1, 2], [3, 4, 5]])
    uv_loops = uv[tris.ravel()]
    m = gr.atlas_density(co, uv_loops, tris, loops, 1000)
    assert m["texels_per_cm"] == pytest.approx(20.0)


def test_atlas_density_of_nothing():
    m = gr.atlas_density(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3), int), np.zeros((0, 3), int), 512)
    assert m == {"texels_per_cm": 0.0, "body_texels_per_cm": 0.0, "atlas_covered": 0.0, "texels_twice": 0.0}


def test_the_body_density_is_the_bulk_of_the_surface_not_a_grown_face():
    """A face grown to a quarter of the atlas raises the figure's average density; the body's own (the density of
    most of the surface) is what a glow line's width in texels depends on (AssetFurnace Support D232)."""
    body, _, body_tris = _square(1.0)                 # 10,000 cm2 on a quarter of the atlas: 5 texels per cm
    face = body * 0.1 + [0, 0, 1]                     # 100 cm2 on another quarter: 50 texels per cm
    co = np.concatenate([body, face])
    uv = np.concatenate([np.array([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]]),
                         np.array([[0.5, 0.5], [1, 0.5], [0.5, 1], [1, 1]])])
    tris = np.concatenate([body_tris, body_tris + 4])
    m = gr.atlas_density(co, uv, tris, tris, 1000)
    assert m["texels_per_cm"] == pytest.approx((500000 / 10100) ** 0.5, rel=1e-3)
    assert m["body_texels_per_cm"] == pytest.approx(5.0, rel=1e-3)


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


def test_problems_name_colour_patches_against_the_source():
    """game_ready compares the finished figure with the model its colour came from (Taha's wish, #145)."""
    good = {"materials": 1, "max_influences": 4, "unweighted": 0, "height": 1.8, "min_z": 0.0,
            "animation_error_m": 0.0}
    clean = dict(good, colour_check={"against": "raw.glb", "patches": [], "picture": None})
    assert gr.problems(clean, max_influences=4, height=1.8) == []
    found = gr.problems(dict(good, colour_check={
        "against": "raw.glb", "picture": "game.colour.png",
        "patches": ["back: a darker patch, 0.6 % of the figure in that view, middle (dE 14, L* -10)",
                    "back left: a darker patch, 0.5 % of the figure in that view, middle (dE 14, L* -9)"]}),
        max_influences=4, height=1.8)
    assert len(found) == 1
    assert "2 patches" in found[0] and "raw.glb" in found[0] and "back:" in found[0] and "game.colour.png" in found[0]


def test_the_colour_check_finds_a_patch_and_leaves_out_what_glows(tmp_path):
    from test_colour_compare import _glb, cube, glow_texture, texture
    p, uv, f = cube()
    source, dark, lit = tmp_path / "source.glb", tmp_path / "dark.glb", tmp_path / "lit.glb"
    source.write_bytes(_glb([(p, uv, f, texture())]))
    dark.write_bytes(_glb([(p, uv, f, texture(dark_side=1))]))
    lit.write_bytes(_glb([(p, uv, f, texture(dark_side=1), glow_texture(1))]))
    picture = tmp_path / "dark.colour.png"
    found = gr.colour_check(source, dark, picture)
    assert found["against"] == "source.glb" and found["patches"] and found["picture"] == "dark.colour.png"
    assert picture.is_file()
    calm = gr.colour_check(source, lit, tmp_path / "lit.colour.png")
    assert calm["patches"] == [] and calm["picture"] is None
    assert not (tmp_path / "lit.colour.png").exists()


def test_the_curve_at_a_known_pivot_is_the_readability_curve():
    """game_ready keeps the readability curve's pivot in its report, so the colour check can put the same curve on
    the source and does not take a figure lightened on purpose for a fault (the Hoellenwaechter's face, #145)."""
    rng = np.random.default_rng(5)
    srgb = rng.uniform(0, 1, (12, 12, 3))
    grey = np.ones((12, 12), bool)
    for t in (0.0, 0.35, 1.0):
        pivot = gr.readability_pivot(srgb, grey, t)
        assert gr.curve_at(srgb, t, pivot) == pytest.approx(gr.value_curve(srgb, grey, t))
    assert gr.readability_pivot(srgb, grey, 0.0) is None


def test_the_colour_check_puts_the_readability_curve_on_the_source(tmp_path):
    from test_colour_compare import _glb, cube, texture
    p, uv, f = cube()
    lifted = np.round(gr.curve_at(texture() / 255.0, 1.0, 0.4) * 255).astype(np.uint8)
    source, game = tmp_path / "source.glb", tmp_path / "game.glb"
    source.write_bytes(_glb([(p, uv, f, texture())]))
    game.write_bytes(_glb([(p, uv, f, lifted)]))
    assert gr.colour_check(source, game, tmp_path / "plain.png")["patches"], "the curve changes the colour"
    assert gr.colour_check(source, game, tmp_path / "curved.png", curve=(1.0, 0.4))["patches"] == []


def test_settings_check_the_colour_unless_told_not_to():
    assert gr.settings([])["colour_check"] is True
    assert gr.settings(["--no-colour-check"])["colour_check"] is False


def test_blender_runs_headless_and_fails_loudly(tmp_path):
    cmd = gr.blender_command(Path("blender.exe"), tmp_path / "job.json")
    assert cmd[:5] == ["blender.exe", "--background", "--python-exit-code", "1", "--python"]
    assert cmd[5].endswith("blender_game_ready.py") and cmd[6:] == ["--", str(tmp_path / "job.json")]


# --- readability chosen from the figure, thin accents, atlas margin --------------------------

def _figure(median, top, n=4000, seed=3):
    """Grey texels around `median`, 2 % of them at the bright `top` (plates, lit skin)."""
    rng = np.random.default_rng(seed)
    v = np.clip(rng.normal(median, 0.04, n), 0, 1)
    v[: n // 50] = top
    return np.repeat(v[:, None, None], 3, axis=2), np.ones((n, 1), bool)


def test_settings_accept_auto_readability():
    assert gr.settings(["--readability", "auto"])["readability"] == "auto"


def test_auto_readability_brightens_a_dark_figure_fully():
    srgb, grey = _figure(0.22, 0.39)  # the Phantom: dark suit, darkish plates
    assert gr.auto_readability(srgb, grey) == 1.0


def test_auto_readability_holds_back_where_light_parts_would_wash_out():
    srgb, grey = _figure(0.24, 0.67)  # the Bruiser: dark clothes, bright lit skin
    t = gr.auto_readability(srgb, grey)
    assert 0.0 < t < 1.0
    out = gr.value_curve(srgb, grey, t)
    assert np.percentile(out.mean(-1)[grey], 99) <= gr.READABILITY_CEILING + 1e-6


def test_auto_readability_leaves_an_already_bright_figure_alone():
    srgb, grey = _figure(0.45, 0.85)
    assert gr.auto_readability(srgb, grey) == 0.0


def test_compose_reports_the_readability_it_chose():
    alb, lit, cov = _atlas()
    _, _, stats = gr.compose(alb, lit, cov, gr.settings(["--readability", "auto", "--no-shade"]))
    assert 0.0 <= stats["readability"] <= 1.0


def test_thin_glow_keeps_a_light_line_and_skips_a_plate_of_the_same_colour():
    n = 32
    red = gr.to_linear(np.array((0.85, 0.12, 0.10)))
    alb = np.zeros((n, n, 3))
    alb[:] = gr.to_linear(np.array((0.20, 0.20, 0.22)))
    alb[4, 2:28] = red          # a light line, one texel wide
    alb[14:26, 8:20] = red      # an armour plate, twelve texels wide
    cov = np.ones((n, n))
    lit = np.full((n, n), 0.5)
    s = gr.settings(["--no-shade", "--glow", "#d91f1a", "--glow-max-width", "4", "--glow-width", "0", "--glow-max-area", "1"])
    _, emission, _ = gr.compose(alb, lit, cov, s)
    assert emission[4, 15].max() > 0.5
    assert emission[20, 14].max() == 0.0
    s_all = gr.settings(["--no-shade", "--glow", "#d91f1a", "--glow-width", "0", "--glow-max-area", "1"])
    _, emission_all, _ = gr.compose(alb, lit, cov, s_all)
    assert emission_all[20, 14].max() > 0.5  # without the limit the plate glows too


def test_the_atlas_margin_is_two_texels_at_any_size():
    assert gr.settings([])["uv_margin_px"] == 2
    assert gr.uv_margin(gr.settings(["--atlas", "512"])) == pytest.approx(2 / 512)
    assert gr.uv_margin(gr.settings([])) == pytest.approx(2 / 1024)


# --- a missing core bone, glow kept off the head ---------------------------------------------

def test_problems_name_a_missing_core_bone():
    report = {"materials": 1, "max_influences": 4, "unweighted": 0, "height": 1.8, "min_z": 0.0,
              "missing_core_bones": ["Neck"]}
    found = " ".join(gr.problems(report, max_influences=4, height=1.8))
    assert "Neck" in found


def test_missing_core_bones_lists_what_a_humanoid_avatar_lacks():
    parents, heads, roles = _humanoid()
    assert gr.missing_core_bones(_unity(parents, heads)) == []
    names = _unity(parents, heads)
    names.pop(roles["neck"])
    assert gr.missing_core_bones(names) == ["Neck"]


def test_the_required_bones_are_the_ones_the_rig_step_names_required():
    from image_to_3dlab.bone_names import REQUIRED
    assert set(gr.REQUIRED_BONES) == set(REQUIRED["unity"])


def test_a_rig_without_a_required_bone_is_refused_by_name():
    # a broken auto-rig (no spine, no shins) bends wrongly in every move: stop before baking
    message = gr.rig_refusal(["Spine", "Chest", "LeftLowerLeg", "RightLowerLeg"])
    assert "Spine" in message and "LeftLowerLeg" in message and "RightLowerLeg" in message
    assert "Chest" not in message
    assert "--allow-missing-bones" in message


@pytest.mark.parametrize("missing", [[], ["Neck"], ["Chest", "UpperChest", "LeftToes", "LeftShoulder"]])
def test_a_rig_missing_only_optional_bones_goes_ahead(missing):
    assert gr.rig_refusal(missing) is None


def test_allow_missing_bones_is_an_option():
    import argparse
    parser = argparse.ArgumentParser()
    gr.add_options(parser)
    assert parser.parse_args([]).allow_missing_bones is False
    assert parser.parse_args(["--allow-missing-bones"]).allow_missing_bones is True


def test_problems_say_a_missing_required_bone_breaks_the_avatar():
    report = {"materials": 1, "max_influences": 4, "unweighted": 0, "height": 1.8, "min_z": 0.0,
              "missing_core_bones": ["Spine", "Neck"]}
    found = gr.problems(report, max_influences=4, height=1.8)
    spine = next(p for p in found if "Spine" in p)
    neck = next(p for p in found if "Neck" in p)
    assert "needs" in spine and "can do without" in neck


def test_head_bones_are_the_head_and_everything_below_it():
    parents, heads, roles = _humanoid()
    parents, heads = dict(parents), dict(heads)
    parents["helmet_lamp"] = roles["head"]
    heads["helmet_lamp"] = np.array((0.0, 0.1, 1.58))  # in front of the face, like a visor lamp
    found = gr.head_bones(parents, _unity(parents, heads))
    assert roles["head"] in found and "helmet_lamp" in found
    assert roles["neck"] not in found and roles["chest"] not in found


def test_settings_glow_on_the_head_by_default():
    assert gr.settings([])["glow_not_on_head"] is False
    assert gr.settings(["--glow-not-on-head"])["glow_not_on_head"] is True


def _stripes(dark_rgb, bright_rgb, bright_every: int, size: int = 64):
    """An atlas of thin stripes on a grey ground: every `bright_every`-th stripe bright, the others dark."""
    a = np.full((size, size, 3), 0.25)
    for i, y in enumerate(range(2, size - 2, 4)):
        a[y, :] = bright_rgb if i % bright_every == 0 else dark_rgb
    return a


def test_a_glow_hue_that_is_mostly_dark_paint_glows_only_where_bright():
    # the Purpur-Samurai: dark crimson plate edges and a few bright red lines in the glow's own hue (D207)
    red = [1.0, 0x2A / 255, 0x3A / 255]
    atlas = _stripes([0.45, 0.08, 0.12], [0.95, 0.2, 0.25], bright_every=5)
    inside = np.ones(atlas.shape[:2], bool)
    mask, split = gr.glow_only_bright(gr.accent_mask(atlas, red) * inside, atlas, inside)
    assert split["used"] and split["bright_share"] <= 0.25
    assert mask[2].min() > 0.9                    # a bright red line still glows
    assert mask[6].max() < 0.05                   # a dark crimson edge no longer does


def test_a_glow_hue_that_is_mostly_bright_lines_is_left_alone():
    cyan = [0x29 / 255, 0xE6 / 255, 1.0]
    atlas = _stripes([0.1, 0.55, 0.65], [0.15, 0.9, 1.0], bright_every=2)   # lines, some of them dimmer
    inside = np.ones(atlas.shape[:2], bool)
    before = gr.accent_mask(atlas, cyan) * inside
    mask, split = gr.glow_only_bright(before, atlas, inside)
    assert not split["used"] and np.array_equal(mask, before)


def test_the_brightness_split_finds_the_two_groups():
    found = gr.brightness_split(np.r_[np.full(800, 0.4), np.full(200, 0.9)])
    assert 0.4 <= found["at"] < 0.9 and found["bright_share"] == pytest.approx(0.2)
    assert found["dark"] == pytest.approx(0.4, abs=0.01) and found["bright"] == pytest.approx(0.9, abs=0.01)


def test_hand_bones_are_both_hands_and_everything_below_them():
    parents, heads, _roles = _humanoid()
    parents, heads = dict(parents), dict(heads)
    unity = _unity(parents, heads)
    left = next(b for b, n in unity.items() if n == "LeftHand")
    parents["left_index_1"] = left
    parents["left_index_2"] = "left_index_1"
    heads["left_index_1"], heads["left_index_2"] = heads[left] + 0.05, heads[left] + 0.08
    found = gr.hand_bones(parents, _unity(parents, heads))
    hands = {b for b, n in unity.items() if n in ("LeftHand", "RightHand")}
    assert hands <= set(found) and {"left_index_1", "left_index_2"} <= set(found)
    assert not {b for b, n in unity.items() if n in ("LeftLowerArm", "RightLowerArm", "Head")} & set(found)


def test_glow_stays_on_the_hands_by_default():
    assert gr.settings([])["glow_not_on_hands"] is False
    assert gr.settings(["--glow-not-on-hands"])["glow_not_on_hands"] is True


def test_the_glow_is_kept_off_the_bones_asked_for():
    parents, heads, _roles = _humanoid()
    unity = _unity(dict(parents), dict(heads))
    s = gr.settings(["--glow-not-on-head", "--glow-not-on-hands"])
    off = gr.glow_off_bones(dict(parents), unity, s)
    assert set(gr.head_bones(dict(parents), unity)) <= set(off) and set(gr.hand_bones(dict(parents), unity)) <= set(off)
    assert gr.glow_off_bones(dict(parents), unity, gr.settings([])) == []
    only_hands = gr.glow_off_bones(dict(parents), unity, gr.settings(["--glow-not-on-hands"]))
    assert set(only_hands) == set(gr.hand_bones(dict(parents), unity))


def test_compose_keeps_the_glow_off_an_excluded_area():
    alb, lit, cov = _atlas()
    s = gr.settings(["--no-shade", "--glow", "#29e6ff", "--glow-width", "0", "--glow-max-area", "1"])
    exclude = np.zeros(cov.shape)
    exclude[4:6, 2:4] = 1.0          # the left half of the cyan strip lies on the head
    _, emission, _ = gr.compose(alb, lit, cov, s, exclude=exclude)
    assert emission[5, 2].max() == 0.0
    assert emission[5, 5].max() > 0.5


# --- the face: thin with the head kept, a sharper head, colour from a finer paint ----------------

def test_the_head_switches_are_off_by_default_but_the_head_texel_is_auto():
    s = gr.settings([])
    assert s["decimate_to"] == 0 and s["head_protect"] == 0.0
    assert s["head_texel"] == "auto"     # D224: a figure's face gets what its colour source holds, never more
    assert gr.settings(["--head-texel", "1"])["head_texel"] == 1.0     # the old behaviour on request
    assert s["colour_from"] is None and s["mesh_from"] is None


def test_settings_read_a_finer_mesh_to_carry_the_rig(tmp_path):
    finer = tmp_path / "finish_40k.glb"
    s = gr.settings(["--mesh-from", str(finer), "--decimate-to", "8000", "--head-protect", "1"])
    assert s["mesh_from"] == str(finer)


def test_settings_read_the_head_switches(tmp_path):
    paint = tmp_path / "painted.glb"
    s = gr.settings(["--decimate-to", "8000", "--head-protect", "1", "--head-texel", "2",
                     "--colour-from", str(paint)])
    assert s["decimate_to"] == 8000 and s["head_protect"] == 1.0 and s["head_texel"] == 2.0
    assert s["colour_from"] == str(paint)


@pytest.mark.parametrize("args", [["--decimate-to", "500"], ["--decimate-to", "-1"], ["--head-protect", "1.5"],
                                  ["--head-texel", "0.5"], ["--head-texel", "5"]])
def test_head_switches_out_of_range_are_refused(args):
    with pytest.raises(SystemExit):
        gr.settings(args)


def test_the_decimate_ratio_never_adds_triangles():
    assert gr.decimate_ratio(40000, 8000) == pytest.approx(0.2)
    assert gr.decimate_ratio(8000, 8000) == 1.0
    assert gr.decimate_ratio(6000, 8000) == 1.0      # already lighter: keep it
    assert gr.decimate_ratio(40000, 0) == 1.0        # 0 = do not thin


def test_head_share_sums_the_head_bones_and_caps_at_one():
    W = np.array([[1.0, 0.0, 0.0],      # all on the chest
                  [0.0, 0.6, 0.4],      # neck and head share ...
                  [0.0, 0.0, 1.0],      # all on the head
                  [0.0, 0.7, 0.7]])     # unnormalised weights still count once
    share = gr.head_share(W, ["chest", "neck", "head"], ["head", "neck"])
    assert share == pytest.approx([0.0, 1.0, 1.0, 1.0])
    assert gr.head_share(W, ["chest", "neck", "head"], ["head"]) == pytest.approx([0.0, 0.4, 1.0, 0.7])
    assert gr.head_share(W, ["chest", "neck", "head"], []) == pytest.approx([0.0] * 4)


def test_scale_islands_grows_the_chosen_islands_around_their_own_centres():
    # two unit squares as four loops each; island 1 is on the head
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [3, 0], [3, 1], [2, 1]], float)
    island = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    out = gr.scale_islands(uv, island, {1}, 2.0)
    assert out[:4] == pytest.approx(uv[:4])                       # the body's island is left alone
    assert out[4:] == pytest.approx(np.array([[1.5, -0.5], [3.5, -0.5], [3.5, 1.5], [1.5, 1.5]]))
    assert gr.scale_islands(uv, island, {1}, 1.0) == pytest.approx(uv)


def test_head_islands_are_the_islands_mostly_on_the_head():
    island = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    share = np.array([0.0, 0.1, 0.0, 1.0, 1.0, 0.4, 0.6, 0.5])    # per loop, from its vertex
    assert gr.head_islands(island, share) == {1, 2}


# --- --head-texel auto: the face gets what its colour source holds, no more ---------------------------

def test_head_texel_reads_auto_and_the_game_texture():
    s = gr.settings(["--head-texel", "auto"])
    assert s["head_texel"] == "auto" and s["game_texture"] == 0     # 0: the size the atlas is baked at
    s = gr.settings(["--head-texel", "auto", "--game-texture", "1024"])
    assert s["game_texture"] == 1024
    assert gr.head_texel_on(s) and gr.head_texel_on(gr.settings(["--head-texel", "2"]))
    assert not gr.head_texel_on(gr.settings(["--head-texel", "1"]))
    assert gr.head_texel_on(gr.settings([]))     # auto is the default


@pytest.mark.parametrize("args", [["--head-texel", "big"], ["--game-texture", "1000"], ["--game-texture", "64"]])
def test_head_texel_auto_switches_out_of_range_are_refused(args):
    with pytest.raises(SystemExit):
        gr.settings(args)


def test_texel_density_is_texels_per_cm_over_the_picked_faces():
    # two 10 cm squares side by side; the first fills a quarter of a 1024 map (512 texels across 10 cm),
    # the second an eighth of the width (128 texels across 10 cm)
    co = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0],
                   [0.2, 0, 0], [0.3, 0, 0], [0.3, 0.1, 0], [0.2, 0.1, 0]], float)
    vert_tris = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    uv = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5],
                   [0.5, 0.5], [0.625, 0.5], [0.625, 0.625], [0.5, 0.625]], float)
    uv_tris = vert_tris.copy()
    assert gr.texel_density(co, vert_tris, uv, uv_tris, 1024, 1024, pick=[0, 1]) == pytest.approx(51.2)
    assert gr.texel_density(co, vert_tris, uv, uv_tris, 1024, 1024, pick=[2, 3]) == pytest.approx(12.8)
    both = gr.texel_density(co, vert_tris, uv, uv_tris, 1024, 1024)
    assert both == pytest.approx(np.sqrt((51.2 ** 2 + 12.8 ** 2) / 2))   # texels over area, then the root
    assert gr.texel_density(co, vert_tris, uv, uv_tris, 1024, 1024, pick=[]) == 0.0


def test_auto_head_factor_gives_the_face_what_the_source_holds():
    head_uv, body_uv = 0.1, 0.5          # UV area of head and body after a pack with the head at factor 1
    f = gr.auto_head_factor(30.0, 15.0, head_uv, body_uv)
    after = 15.0 * f * np.sqrt((head_uv + body_uv) / (f * f * head_uv + body_uv))   # the head after growing, repacked
    assert 1.0 < f < 4.0 and after == pytest.approx(30.0, rel=0.01)


def test_auto_head_factor_stays_within_one_and_four():
    assert gr.auto_head_factor(10.0, 15.0, 0.1, 0.5) == 1.0     # the atlas already holds more than the source
    assert gr.auto_head_factor(200.0, 15.0, 0.1, 0.5) == 4.0    # more than the head can ever take
    assert gr.auto_head_factor(60.0, 15.0, 0.3, 0.1) == 4.0     # out of reach: the head would need the whole map
    assert gr.auto_head_factor(30.0, 0.0, 0.0, 0.5) == 1.0      # no head
    assert gr.auto_head_factor(0.0, 15.0, 0.1, 0.5) == 1.0      # no source measured


# --- close-up versions: a normal map from the finer model, soft edges ---------------------------

def test_the_close_up_switches_are_off_by_default():
    s = gr.settings([])
    assert s["normal_from"] is None


def test_soft_edges_are_the_default_and_hard_seams_can_be_asked_for():
    # a seam above 25 degrees made every face triangle show on a thinned figure (D145): soft by default now
    assert gr.settings([])["sharp_seam_deg"] == gr.settings([])["sharp_deg"]
    assert gr.settings(["--soft-edges"])["sharp_seam_deg"] == gr.settings([])["sharp_deg"]   # old command lines
    assert gr.settings(["--hard-seams"])["sharp_seam_deg"] == 25.0


def test_settings_read_the_close_up_switches(tmp_path):
    raw = tmp_path / "raw.glb"
    s = gr.settings(["--normal-from", str(raw), "--soft-edges"])
    assert s["normal_from"] == str(raw)
    assert s["sharp_seam_deg"] == s["sharp_deg"]   # a UV seam alone no longer makes an edge hard


def test_normal_texels_are_flat_where_the_finer_model_was_missed_and_unit_length_elsewhere():
    baked = np.array([[[0.5, 0.5, 1.0], [0.9, 0.5, 0.6]],
                      [[0.25, 0.75, 0.8], [0.6, 0.55, 0.7]]])     # box-filtered: no longer unit length
    hit = np.array([[1.0, 0.0], [1.0, 0.5]])
    out = gr.normal_texels(baked, hit)
    assert out[0, 0] == pytest.approx([0.5, 0.5, 1.0])         # straight up stays straight up
    assert out[0, 1] == pytest.approx([0.5, 0.5, 1.0])         # a missed ray: the coarse surface as it is
    n = out * 2.0 - 1.0
    assert np.linalg.norm(n, axis=-1) == pytest.approx(np.ones((2, 2)), abs=1e-6)
    assert out[1, 0, 0] < 0.5 and out[1, 0, 1] > 0.5           # the direction survives the renormalising
    assert out[1, 1, 0] < gr.normal_texels(baked[1:, 1:], np.ones((1, 1)))[0, 0, 0]   # half a hit: half the tilt


def test_blend_transfer_takes_the_finer_paint_where_it_hit_and_keeps_the_own_elsewhere():
    own = np.full((2, 2, 3), 0.2)
    transfer = np.full((2, 2, 3), 0.8)
    hit = np.array([[1.0, 0.0], [0.5, 1.0]])
    out = gr.blend_transfer(own, transfer, hit)
    assert out[0, 0] == pytest.approx([0.8] * 3)       # the finer paint
    assert out[0, 1] == pytest.approx([0.2] * 3)       # a missed ray keeps the model's own paint
    assert out[1, 0] == pytest.approx([0.5] * 3)       # half covered texels blend
    assert gr.blend_transfer(own, transfer, None) == pytest.approx(own)


def test_the_colour_transfer_reach_can_be_set():
    s = gr.settings([])
    assert (s["transfer_cage_m"], s["transfer_reach_m"]) == (0.012, 0.03)
    s = gr.settings(["--transfer-cage", "0.03", "--transfer-reach", "0.06"])
    assert (s["transfer_cage_m"], s["transfer_reach_m"]) == (0.03, 0.06)


# --- the hidden inner wall (--drop-hidden) --------------------------------------------------------

def test_drop_hidden_is_an_option_and_needs_a_budget():
    assert gr.settings([])["drop_hidden"] is False
    s = gr.settings(["--drop-hidden", "--decimate-to", "14000"])
    assert s["drop_hidden"] is True and gr.drop_hidden_refusal(s) is None
    assert "--decimate-to" in gr.drop_hidden_refusal(gr.settings(["--drop-hidden"]))


def test_the_first_thinning_aims_above_the_budget():
    # The Wachter: thinned to 22,000, the drop left 15,275 (D190); the first try starts there.
    assert gr.first_target(14000) == 22000
    assert gr.first_target(8000) == round(8000 * 22000 / 14000)


def test_the_next_try_scales_by_what_the_drop_left():
    assert gr.next_first_target(22000, 15275, 14000) == round(22000 * 14000 / 15275)


def test_a_figure_without_an_inner_wall_ends_at_the_budget():
    assert gr.next_first_target(22000, 22000, 14000) == 14000


def test_the_next_try_never_aims_below_the_budget_or_without_limit():
    assert gr.next_first_target(22000, 30000, 14000) == 14000   # never thinner than the budget itself
    assert gr.next_first_target(22000, 100, 14000) == 4 * 14000
    assert gr.next_first_target(22000, 0, 14000) == 4 * 14000


def test_a_drop_is_settled_within_three_percent_of_the_budget():
    assert gr.drop_settled(14300, 14000) and gr.drop_settled(13600, 14000)
    assert not gr.drop_settled(14500, 14000) and not gr.drop_settled(13000, 14000)


def test_the_check_poses_name_the_rigs_own_bones():
    unity = {"thigh_l": "LeftUpperLeg", "shin_l": "LeftLowerLeg", "thigh_r": "RightUpperLeg", "spine_1": "Spine"}
    poses = gr.veto_poses(unity)
    assert [name for name, _ in poses] == list(gr.VETO_POSES)
    stride = dict(poses)["stride"]
    assert ("thigh_l", (1.0, 0.0, 0.0), -60.0) in stride
    assert all(bone in unity for _, moves in poses for bone, _, _ in moves)   # bones the rig lacks are skipped


def test_the_drop_settings_are_fixed_values():
    s = gr.settings(["--drop-hidden", "--decimate-to", "14000"])
    assert (s["hidden_rays"], s["hidden_cells"], s["hidden_close"], s["hidden_corner_reach"]) == (16, 400, 2, 0.66)
    assert (s["drop_tries"], s["drop_slack"]) == (3, 0.03)


# --- the pose check: holes the drop opened in motion, found and repaired -----------------------

def test_the_pose_check_moves_the_figure_by_its_own_bones_and_not_like_the_veto():
    poses = gr.check_poses({"mixamorig:LeftUpLeg": "LeftUpperLeg", "mixamorig:Spine": "Spine"})
    assert [name for name, _moves in poses] == list(gr.CHECK_POSES)
    run = dict(poses)["run"]
    assert ("mixamorig:Spine", (1.0, 0.0, 0.0), 15.0) in run
    assert {bone for bone, _axis, _deg in run} <= {"mixamorig:LeftUpLeg", "mixamorig:Spine"}
    # an independent look: other moves than the ones the drop already asked
    assert set(gr.CHECK_POSES).isdisjoint(gr.VETO_POSES)


def test_a_view_looks_at_the_figure_from_its_side_and_covers_it():
    origins, direction = gr.view_rays((0.0, 0.0, 1.0), 1.0, 0, 0, 8)
    assert direction == pytest.approx((0.0, 1.0, 0.0))   # from the front (-Y), looking toward +Y
    assert origins.shape == (64, 3)
    assert (origins[:, 1] < -1.0).all()                   # every ray starts outside the figure's sphere
    assert origins[:, 0].min() == pytest.approx(-0.875) and origins[:, 0].max() == pytest.approx(0.875)
    assert origins[0, 2] == pytest.approx(1.875) and origins[-1, 2] == pytest.approx(0.125)   # top row first
    _o, side = gr.view_rays((0.0, 0.0, 0.0), 1.0, 90, 0, 4)
    assert side == pytest.approx((-1.0, 0.0, 0.0))
    _o, below = gr.view_rays((0.0, 0.0, 0.0), 1.0, 0, -35, 4)
    assert below[2] > 0                                   # from below, looking up


def test_a_hole_is_a_dropped_face_seen_with_nothing_close_behind_it():
    doomed = np.array([False, True, True])
    before_face = np.array([-1, 0, 1, 1, 2])                # nothing; a kept face; three dropped faces
    before_depth = np.array([np.inf, 1.0, 1.0, 1.0, 1.0])
    after_depth = np.array([np.inf, 1.0, np.inf, 1.005, 1.5])   # empty; real surface 5 mm behind; far behind
    hole = gr.pose_holes(before_face, before_depth, after_depth, doomed, gap=0.01)
    assert hole.tolist() == [False, False, True, False, True]
    assert gr.faces_seen_in(before_face, hole).tolist() == [1, 2]


def test_problems_name_holes_the_pose_check_could_not_repair():
    good = {"materials": 1, "max_influences": 4, "unweighted": 0, "height": 1.8, "min_z": 0.0,
            "animation_error_m": 0.0}
    repaired = dict(good, pose_check={"hole_px": 120, "restored": 14, "hole_px_left": 0, "picture": "g.pose.png"})
    assert gr.problems(repaired, max_influences=4, height=1.8) == []
    left = dict(good, pose_check={"hole_px": 120, "restored": 14, "hole_px_left": 9, "picture": "g.pose.png",
                                  "worst": "fall, back"})
    found = gr.problems(left, max_influences=4, height=1.8)
    assert len(found) == 1 and "9 px" in found[0] and "fall, back" in found[0] and "g.pose.png" in found[0]


def test_settings_check_the_poses_unless_told_not_to():
    assert gr.settings([])["pose_check"] is True
    assert gr.settings(["--no-pose-check"])["pose_check"] is False


# --- the feet: a sole the generator made up below the pictures' feet ------------------------------

def _part(reach, side="below", low=0.0, high=0.05, view="front"):
    return {"view": view, "side": side, "reach": reach, "low": low, "high": high, "vertices": 150}


def test_a_second_shoe_glued_under_the_real_one_is_cut_off():
    # the Sturmschuetze of the AssetFurnace Support notes: 6.8 % of his height below the pictures' feet
    plan = gr.feet_plan([_part(0.0677), _part(0.0667), _part(0.0555, view="back"), _part(0.0429, side="right",
                                                                                          low=0.46, high=0.52)])
    assert plan["mode"] == "cut" and plan["at"] == pytest.approx(0.0677) and plan["parts"] == 3


def test_a_heel_a_little_too_high_is_pressed_flat_to_a_sole():
    plan = gr.feet_plan([_part(0.0346), _part(0.0299, view="back")], height=1.85)
    assert plan["mode"] == "squash" and plan["at"] == pytest.approx(0.0346)
    assert plan["sole"] == pytest.approx(gr.FIXED["feet_sole_m"] / 1.85, abs=1e-4)


def test_feet_inside_the_pictures_or_parts_off_the_floor_change_nothing():
    assert gr.feet_plan([]) is None
    assert gr.feet_plan([_part(0.015)]) is None                       # within the outline's slack
    assert gr.feet_plan([_part(0.08, side="left", low=0.4, high=0.5)]) is None   # a hand past the outline
    assert gr.feet_plan([_part(0.08, low=0.05)]) is None              # not on the floor
    assert gr.feet_plan([_part(0.08, high=0.3)]) is None              # a robe down to the floor, not a shoe


def test_the_feet_can_be_told_what_to_do_or_left_alone():
    assert gr.feet_plan([_part(0.065)], mode="squash")["mode"] == "squash"
    assert gr.feet_plan([_part(0.03)], mode="cut")["mode"] == "cut"
    assert gr.feet_plan([_part(0.065)], mode="off") is None
    assert gr.feet_plan([_part(0.01)], mode="squash", height=1.8) is None  # already thinner than a sole


def test_a_cut_moves_everything_above_down_and_lays_what_was_below_on_the_floor():
    plan = {"mode": "cut", "at": 0.06, "sole": 0.0}
    z = np.array([0.0, 0.03, 0.06, 0.5, 1.0])
    assert gr.feet_heights(z, plan) == pytest.approx([0.0, 0.0, 0.0, 0.44, 0.94])
    assert gr.feet_drop(plan) == pytest.approx(0.06)


def test_a_squash_presses_the_sole_and_keeps_the_shoe_joined():
    plan = {"mode": "squash", "at": 0.04, "sole": 0.01}
    z = np.array([0.0, 0.02, 0.04, 0.5, 1.0])
    assert gr.feet_heights(z, plan) == pytest.approx([0.0, 0.005, 0.01, 0.47, 0.97])
    assert gr.feet_drop(plan) == pytest.approx(0.03)


def test_a_model_drawn_for_the_colour_check_gets_the_same_feet_at_its_own_height():
    plan = {"mode": "cut", "at": 0.1, "sole": 0.0}
    positions = np.array([[0.0, -1.0, 0.0], [0.0, -0.9, 0.2], [0.3, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2 tall, y up
    out = gr.feet_positions(positions, plan, up=1)
    assert out[:, 1].min() == pytest.approx(-1.0) and out[:, 1].max() == pytest.approx(1.0)
    assert out[1, 1] == pytest.approx(-1.0)                              # the cut sole lies on the floor
    assert out[2, 1] == pytest.approx(-1.0 + 2.0 * 0.4 / 0.9)         # halfway up: 0.4 above the cut, scaled back
    assert np.array_equal(out[:, [0, 2]], positions[:, [0, 2]])


def test_the_inner_walls_cap_inside_each_sole_is_found():
    def square(cx, cy, r):
        return np.array([[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]])

    left, right = square(-0.15, 0.0, 0.06), square(0.15, 0.0, 0.06)
    inner_left, inner_right = square(-0.15, 0.0, 0.05), square(0.15, 0.01, 0.045)
    assert gr.nested_caps([left, right]) == [False, False]
    assert gr.nested_caps([inner_left, left, right, inner_right]) == [True, False, False, True]


def test_settings_take_the_feet_from_a_finish_run_and_auto_by_default():
    s = gr.settings([])
    assert s["feet_from"] is None and s["feet"] == "auto"
    s = gr.settings(["--feet-from", "output/finish/run", "--feet", "squash"])
    assert s["feet_from"] == str(Path("output/finish/run")) and s["feet"] == "squash"


def _box(x0, y0, z0, x1, y1, z1):
    """A closed box as (vertices, triangles), its faces turned outward."""
    v = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], dtype=float)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return v, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


def _join(*meshes):
    verts, faces, n = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(f + n)
        n += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def test_a_cut_through_a_boot_leaves_no_thin_plate():
    boot = _box(0.0, 0.0, 0.0, 0.1, 0.25, 0.15)      # a boot 10 x 25 cm, hollow above its sole
    found = gr.splinters(*boot, height=1.85)
    assert found["sole_m2"] == pytest.approx(0.025) and found["thin_m2"] == 0.0 and found["reach_m"] == 0.0


def test_a_plate_of_the_made_up_shoe_in_front_of_the_boot_is_found():
    boot = _box(0.0, 0.0, 0.0, 0.1, 0.25, 0.15)
    plate = _box(0.0, 0.26, 0.0, 0.1, 0.36, 0.004)    # 10 x 10 cm, 4 mm thick, out in front of the toe
    found = gr.splinters(*_join(boot, plate), height=1.85)
    assert found["thin_m2"] == pytest.approx(0.01) and found["thin_share"] == pytest.approx(0.01 / 0.035, abs=1e-3)
    assert found["reach_m"] > 0.1
    assert gr.splintered(found, 1.85)


def test_only_soles_near_the_feet_count_so_a_robe_on_the_floor_is_left_out():
    boot = _box(0.0, 0.0, 0.0, 0.1, 0.25, 0.15)
    hem = _box(0.6, 0.0, 0.0, 0.9, 0.05, 0.004)      # a thin robe hem lying on the floor, 60 cm away
    feet = [[0.05, 0.05, 0.08], [0.05, 0.2, 0.02]]
    assert gr.splinters(*_join(boot, hem), height=1.85)["thin_m2"] > 0
    assert gr.splinters(*_join(boot, hem), height=1.85, feet=feet)["thin_m2"] == 0.0


def test_a_small_thin_tip_of_a_pointed_shoe_is_no_reason_to_press_the_sole():
    # measured in Blender right after the cut on four figures whose platform shoes were cut (D207 of the AssetFurnace
    # Support notes): two keep a clean cut, two leave blades of the made-up shoe in front of the boot
    assert not gr.splintered({"thin_m2": 0.00739, "reach_m": 0.0323}, 1.85)  # Arkan-Duellant: stays cut
    assert not gr.splintered({"thin_m2": 0.00684, "reach_m": 0.0217}, 1.85)  # Sturmschuetze: stays cut
    assert gr.splintered({"thin_m2": 0.0185, "reach_m": 0.0451}, 1.85)       # Revolverheld: pressed flat
    assert gr.splintered({"thin_m2": 0.01518, "reach_m": 0.0621}, 2.2)       # Daemonenhexer: pressed flat


def test_with_plates_under_6_5_mm_only_the_daemonenhexers_broad_plates_are_cleared():
    # measured in Blender right after the cut with plates thinner than 0.35 % of the height (#183, D214 of the
    # AssetFurnace Support notes): the Daemonenhexer's broad sheet is cleared; the others' thin parts are a real toe's
    # sloping cap (Arkan-Duellant, Sturmschuetze: clean boots) or a short blade the Revolverheld keeps rather than lose
    # his toe, as the 1.5 % rule made him
    assert gr.splintered({"thin_m2": 0.01327, "reach_m": 0.0621}, 2.2)       # Daemonenhexer: cleared
    assert not gr.splintered({"thin_m2": 0.00692, "reach_m": 0.0171}, 1.85)  # Arkan-Duellant: stays cut
    assert not gr.splintered({"thin_m2": 0.00486, "reach_m": 0.0195}, 1.85)  # Sturmschuetze: stays cut
    assert not gr.splintered({"thin_m2": 0.00237, "reach_m": 0.0198}, 1.85)  # Revolverheld: keeps his toe


def test_a_cut_chosen_by_itself_clears_its_plates():
    plan = gr.feet_plan([_part(0.065)], height=1.85)
    assert plan["mode"] == "cut" and plan["auto"]
    assert not gr.feet_plan([_part(0.065)], mode="cut")["auto"]    # a cut asked for by hand is kept as cut


def _l_shoe(plate_to=0.35, thick=0.004, x1=0.1):
    """A boot 25 cm long and 15 cm high whose cut left a plate `thick` thick out in front of its toe, as one connected
    surface (an L profile in y and z, drawn along x); the faces on the floor face it, the rest any way."""
    profile = [(0.0, 0.0), (0.25, 0.0), (plate_to, 0.0), (plate_to, thick), (0.25, thick), (0.25, 0.15), (0.0, 0.15)]
    return _profile_shoe(profile, [(0, 1, 4), (0, 4, 5), (0, 5, 6), (1, 2, 3), (1, 3, 4)], x1)


def _profile_shoe(profile, side, x1=0.1):
    """A closed solid from a profile in y and z drawn along x (its two ends triangulated by `side`), the faces on the
    floor facing it."""
    n = len(profile)
    v = np.array([[x, y, z] for x in (0.0, x1) for y, z in profile], dtype=float)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, j, n + j), (i, n + j, n + i)]
    tris += side + [(n + a, n + c, n + b) for a, b, c in side]
    f = np.array(tris)
    t = v[f]
    nz = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2]
    floor = t[:, :, 2].max(axis=1) == 0.0
    f[floor & (nz > 0)] = f[floor & (nz > 0)][:, ::-1]
    return v, f


def test_the_plate_in_front_of_the_boot_is_what_goes():
    v, f = _l_shoe()
    drop = gr.plate_faces(v, f, height=1.85)
    centre = v[f].mean(axis=1)
    assert drop[centre[:, 1] > 0.25].all()           # the plate: its underside, top, front and sides
    assert not drop[centre[:, 1] < 0.25].any()       # the boot: its sole, walls and top stay
    assert gr.splinters(v, f, height=1.85)["thin_m2"] > 0
    assert gr.splinters(v, f[~drop], height=1.85)["thin_m2"] == 0.0


def test_a_plate_tip_that_rises_off_the_floor_goes_with_the_plate():
    # the Daemonenhexer (D210): the made-up shoe's toe tilts up, so its front is no sole on the floor, yet it hangs on the
    # plate as a spike once the plate is gone
    profile = [(0.0, 0.0), (0.25, 0.0), (0.35, 0.0), (0.42, 0.004), (0.35, 0.006), (0.25, 0.006), (0.25, 0.15), (0.0, 0.15)]
    side = [(0, 1, 5), (0, 5, 6), (0, 6, 7), (1, 2, 3), (1, 3, 4), (1, 4, 5)]
    v, f = _profile_shoe(profile, side)
    drop = gr.plate_faces(v, f, height=1.85)
    centre = v[f].mean(axis=1)
    assert drop[centre[:, 1] > 0.25].all()           # the plate and its tip
    assert not drop[centre[:, 1] < 0.25].any()


def test_a_clean_boot_loses_nothing():
    v, f = _box(0.0, 0.0, 0.0, 0.1, 0.25, 0.15)
    assert not gr.plate_faces(v, f, height=1.85).any()


def test_a_plate_far_from_the_feet_stays():
    v, f = _l_shoe()
    feet = [[0.05, 0.05, 0.08], [0.05, 0.2, 0.02]]    # ankle and toe bones
    assert gr.plate_faces(v, f, height=1.85, feet=feet).sum() == gr.plate_faces(v, f, height=1.85).sum()
    far = [[2.0, 2.0, 0.08]]
    assert not gr.plate_faces(v, f, height=1.85, feet=far).any()


def test_a_thick_toe_is_no_plate():
    v, f = _l_shoe(thick=0.06)                         # a step 6 cm high in front of the boot: a real toe cap
    assert not gr.plate_faces(v, f, height=1.85).any()


def _tapered_toe():
    """A boot cut through low: 15 cm high at the back, its toe 1.5 cm high from 15 to 25 cm, then thinning to its rim
    at 27 cm, as the Revolverheld's real toe after his cut (#183 of the AssetFurnace Support notes: 10-15 mm thick, thin
    only at its rim)."""
    profile = [(0.0, 0.0), (0.15, 0.0), (0.25, 0.0), (0.265, 0.0), (0.27, 0.0), (0.265, 0.00375), (0.25, 0.015),
               (0.15, 0.015), (0.15, 0.15), (0.0, 0.15)]
    side = [(0, 1, 8), (0, 8, 9), (1, 2, 6), (1, 6, 7), (2, 3, 5), (2, 5, 6), (3, 4, 5)]
    return _profile_shoe(profile, side)


def test_a_real_toe_that_thins_only_at_its_rim_is_no_plate():
    # the Revolverheld (#183): the old rule took everything under 2.8 cm for a plate and cut his real toe off
    v, f = _tapered_toe()
    found = gr.splinters(v, f, height=1.85)
    assert found["thin_m2"] <= 0.0006 and found["reach_m"] < 0.02
    assert not gr.splintered(found, 1.85)
    centre = v[f].mean(axis=1)
    assert not gr.plate_faces(v, f, height=1.85)[centre[:, 1] < 0.25].any()   # the toe's core stays in any case


def test_a_dense_normal_source_shaded_flat_is_smoothed_and_a_hard_prop_is_not():
    # Hi3D's 1M-triangle output came without vertex normals, so its facets went into the knight's normal map (D214)
    assert gr.flat_normal_source(0, 1_000_000)
    assert not gr.flat_normal_source(990_000, 1_000_000)          # already smooth (a model with normals)
    assert not gr.flat_normal_source(0, 3_000)                      # a hard-edged low model stays as it is


def _provenance(path: Path, record: dict) -> None:
    path.with_name(path.stem + ".provenance.json").write_text(json.dumps(record), encoding="utf-8")


def test_the_finish_run_is_found_in_the_rigs_provenance(tmp_path):
    run = tmp_path / "output" / "finish" / "hero__a1__pixal3d__20261114__finish__20261115"
    run.mkdir(parents=True)
    rigged = tmp_path / "output" / "animate" / "hero_8k__rig" / "hero_8k_rigged.glb"
    rigged.parent.mkdir(parents=True)
    rigged.write_bytes(b"glTF")
    _provenance(rigged, {"operation": "rig", "source_record": {
        "operation": "finish", "output": {"path": str(run / "hero_8k.glb")}, "source_record": {"operation": "pixal3d"}}})
    assert gr.finish_run_from_provenance(rigged) == run


def test_no_finish_run_without_a_provenance_or_when_the_run_is_gone(tmp_path):
    rigged = tmp_path / "hero_rigged.glb"
    rigged.write_bytes(b"glTF")
    assert gr.finish_run_from_provenance(rigged) is None
    _provenance(rigged, {"source_record": {"operation": "finish", "output": {"path": str(tmp_path / "gone" / "a.glb")}}})
    assert gr.finish_run_from_provenance(rigged) is None
    _provenance(rigged, {"source_record": {"operation": "pixal3d", "output": {"path": str(rigged)}}})
    assert gr.finish_run_from_provenance(rigged) is None


# --- the face zone: only the face grows, hair and hats keep the body's texels (TODO 148 of the AssetFurnace
# Support notes: the Leerenklinge's ponytail took 76 % of the map and her back plate's rune melted into a blob) ---

def _face_case():
    """Seven faces of a 1.8 m figure whose head joint stands at 1.5 m, x = 0, facing -Y; all but the last on the head."""
    centres = np.array([[0.0, -0.1, 1.55],     # 0 the face: between the temples, in the band, facing the front
                        [0.05, -0.1, 1.52],    # 1 a cheek, still between the temples
                        [0.0, -0.1, 1.70],     # 2 a hat's front, above the band
                        [0.1, 0.0, 1.55],      # 3 the side of the head (an ear), facing +X
                        [0.0, 0.1, 1.55],      # 4 the back of the head
                        [0.0, -0.1, 1.40],     # 5 a collar under the chin, below the band
                        [0.0, -0.1, 1.55]])    # 6 a face-like spot on the body (a mask on the chest)
    normals = np.array([[0, -1, 0], [0.3, -0.95, 0], [0, -0.9, 0.44], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0]],
                       float)
    share = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    return centres, normals, share


def test_the_face_zone_is_the_heads_front_from_the_chin_to_the_brows():
    centres, normals, share = _face_case()
    zone = gr.face_zone(centres, normals, share, head_joint=(0.0, 0.0, 1.5), height=1.8)
    assert zone.tolist() == [True, True, False, False, False, False, False]


def test_the_face_zone_follows_the_head_sideways_and_its_own_settings():
    centres, normals, share = _face_case()
    moved = centres + [0.4, 0.0, 0.0]                   # the figure stands 40 cm to the side
    assert gr.face_zone(moved, normals, share, (0.4, 0.0, 1.5), 1.8).tolist()[:2] == [True, True]
    s = dict(gr.FIXED, face_above=0.12)                 # a band reaching higher takes the hat's front too
    assert gr.face_zone(centres, normals, share, (0.0, 0.0, 1.5), 1.8, s)[2]


def test_zone_islands_split_the_face_off_its_island_before_it_grows():
    # one island of eight loops (two quads); the second quad is the face
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [1, 0], [2, 0], [2, 1], [1, 1]], float)
    island = np.zeros(8, int)
    in_zone = np.array([False] * 4 + [True] * 4)
    split, chosen = gr.zone_islands(island, in_zone)
    assert chosen == {0}
    assert set(split[:4]).isdisjoint(chosen)            # the hair's quad is no longer part of a chosen island
    out = gr.scale_islands(uv, split, chosen, 2.0)
    assert out[:4] == pytest.approx(uv[:4])             # hair keeps the body's texels
    assert out[4:] == pytest.approx(np.array([[0.5, -0.5], [2.5, -0.5], [2.5, 1.5], [0.5, 1.5]]))


def test_head_zone_is_the_face_by_default_and_the_whole_head_on_request():
    assert gr.settings([])["head_zone"] == "face"
    assert gr.settings(["--head-zone", "head"])["head_zone"] == "head"
    with pytest.raises(SystemExit):
        gr.settings(["--head-zone", "hat"])


# --- look: how much the material shines, and which parts are metal ----------------------------

def test_the_look_sets_how_much_a_figure_shines():
    assert gr.settings([])["look"] == "stylized"
    assert gr.settings([])["specular"] == pytest.approx(0.5)  # the glTF default: every surface mirrors 4 %
    for look in ("clean",):
        s = gr.settings(["--look", look])
        assert s["look"] == look
        # a fifth of the sheen: a bright room no longer lays a grey veil over dark cloth
        assert s["specular"] == pytest.approx(0.1)


# --- the glow gate counts widths in centimetres -----------------------------------------------

def _glow_lines(k: int):
    """Columns of one glow hue on grey, `k` texels per unit: thin lines (2 units) bright and dark, and medium dark
    lines (4 units). At k = 1 a fifth of the thin-enough hue is bright, so only the bright lines are paint to glow."""
    hue = np.array([0.16, 0.9, 1.0])
    cols = []
    for _ in range(6):
        cols += [None] * 4 + [0.98] * 2 + [None] * 4 + [0.45] * 2 + [None] * 4 + [0.45] * 2 + [None] * 4 + [0.45] * 4
    srgb = np.full((48, len(cols), 3), 0.3)
    for x, v in enumerate(cols):
        if v is not None:
            srgb[:, x] = hue * v
    srgb = srgb.repeat(k, axis=0).repeat(k, axis=1)
    return gr.to_linear(srgb), np.ones(srgb.shape[:2]), np.ones(srgb.shape[:2])


def test_the_glow_gate_does_not_flip_when_the_body_is_packed_denser():
    """A face grown into the atlas packs the body denser (the Sturmklinge's v18a: 7.0 texels per cm against 3.6),
    and widths counted in texels then left out lines that had counted, so her jacket's light blue glowed whole
    (AssetFurnace Support D231). Counted against the body's density, the same figure gives the same gate."""
    s = gr.settings(["--glow", "#29e6ff", "--glow-max-width", "4", "--no-shade", "--glow-max-area", "1"])
    splits = []
    for k in (1, 2):
        alb, lit, cov = _glow_lines(k)
        s["body_texels_per_cm"] = s["glow_ref_texels_per_cm"] * k
        _, _, stats = gr.compose(alb, lit, cov, s)
        splits.append(stats["glow_split"]["#29e6ff"])
    assert splits[0]["used"] and splits[1]["used"]
    assert splits[1]["bright_share"] == pytest.approx(splits[0]["bright_share"], abs=0.02)


def test_without_a_measured_body_the_gate_counts_texels_as_before():
    s = gr.settings(["--glow", "#29e6ff", "--glow-max-width", "4", "--no-shade", "--glow-max-area", "1"])
    s.pop("body_texels_per_cm", None)
    alb, lit, cov = _glow_lines(1)
    _, _, stats = gr.compose(alb, lit, cov, s)
    assert stats["glow_split"]["#29e6ff"]["used"]


# --- the colour transfer takes nothing from the back of a surface ---------------------------------

def _baked_normals(vectors):
    """Unit normals as a normal bake writes them in object space: 0..1 colours."""
    n = np.asarray(vectors, float)
    return n / np.linalg.norm(n, axis=-1, keepdims=True) * 0.5 + 0.5


def test_a_ray_that_met_a_surface_facing_away_is_told_apart():
    """A neck's ray, started outside a high collar, meets the collar's inside first: its normal points back at the
    neck (a wizard's neck took the collar's dark that way, AssetFurnace Support D261)."""
    own = _baked_normals([[0, 0, 1], [0, 0, 1], [1, 0, 0]])
    met = _baked_normals([[0, 0.3, 1], [0, 0, -1], [-1, 0.1, 0]])
    facing = gr.met_facing(met, own)
    assert facing[0] > 0.9 and facing[1] < -0.9 and facing[2] < -0.9


def test_each_texel_takes_the_longest_cage_whose_ray_met_a_surface_facing_it():
    long_colour, short_colour = np.full((4, 3), 0.1), np.full((4, 3), 0.8)
    tries = [(long_colour, np.array([1.0, 1.0, 1.0, 0.0]), np.array([1.0, -1.0, -1.0, 0.0])),
             (short_colour, np.array([1.0, 1.0, 0.0, 0.0]), np.array([1.0, 0.9, -1.0, 0.0])),
             (np.full((4, 3), 0.5), np.array([1.0, 1.0, 1.0, 0.0]), np.array([1.0, 1.0, -0.5, 0.0]))]
    colour, hit, step = gr.pick_transfer(tries)
    assert colour[0] == pytest.approx(0.1) and hit[0] == 1.0 and step[0] == 0       # the long cage was fine
    assert colour[1] == pytest.approx(0.8) and hit[1] == 1.0 and step[1] == 1       # the collar's inside, then the neck
    assert hit[2] == 0.0 and step[2] == -1                                           # nothing facing it: own paint
    assert hit[3] == 0.0 and step[3] == -1                                           # no ray met anything


def test_a_texel_whose_rays_all_met_surfaces_facing_away_keeps_the_models_own_paint():
    own = np.full((2, 3), 0.6)
    colour, hit, _ = gr.pick_transfer([(np.full((2, 3), 0.05), np.ones(2), np.array([-0.8, 0.7]))])
    out = gr.blend_transfer(own, colour, hit)
    assert out[0] == pytest.approx([0.6, 0.6, 0.6]) and out[1] == pytest.approx([0.05, 0.05, 0.05])


def test_a_texel_whose_long_ray_met_another_layer_first_takes_the_shorter_cage():
    """A collar flap stands 8-11 mm off the jaw, facing the same way, inside the full cage: the long ray paints the
    flap's dark onto the jaw, a shorter one meets the jaw. The model's own paint (skin there) tells which is right;
    a hair spike the thinning sank keeps the long ray's paint, since its own paint is hair too."""
    skin, flap, hair = [0.55, 0.42, 0.38], [0.03, 0.02, 0.06], [0.30, 0.32, 0.40]
    own = np.array([skin, hair, skin])
    long_colour = np.array([flap, hair, skin])
    short_colour = np.array([skin, skin, skin])
    tries = [(long_colour, np.ones(3), np.ones(3)), (short_colour, np.ones(3), np.ones(3))]
    colour, _, step = gr.pick_transfer(tries, own=own)
    assert step.tolist() == [1, 0, 0]
    assert colour[0] == pytest.approx(skin) and colour[1] == pytest.approx(hair)
    assert gr.pick_transfer(tries)[2].tolist() == [0, 0, 0]          # without the model's own paint, as before


def test_the_cage_steps_start_at_the_cage_and_shrink():
    steps = gr.FIXED["transfer_cage_steps"]
    assert steps[0] == 1.0 and all(a > b > 0 for a, b in itertools.pairwise(steps))


# --- a glow that lights more than a small share of the figure is narrowed (TODO 195 of the support notes) ----------

def _cape_atlas(n=64):
    """A dark grey figure with a large dark-red cape (a faint match for a red glow) and a thin eye slit of the
    glow's own red: the black knight of ArcherSoul's #636, whose cape lit 25 % of it."""
    alb = np.zeros((n, n, 3))
    alb[:] = gr.to_linear(np.array((0.18, 0.18, 0.20)))
    alb[8:56, 4:40] = gr.to_linear(np.array((0.30, 0.15, 0.15)))   # the cape: a faint match (about 0.47)
    alb[20, 44:60] = gr.to_linear(np.array((0.855, 0.161, 0.298)))  # the eye slit: #da294c
    return alb, np.full((n, n), 0.5), np.ones((n, n))


def test_the_report_counts_the_glow_as_it_shows_faint_parts_too():
    alb, lit, cov = _cape_atlas()
    s = gr.settings(["--no-shade", "--glow", "#da294c", "--glow-width", "0", "--glow-max-area", "1"])
    _, emission, stats = gr.compose(alb, lit, cov, s)
    assert emission[30, 20].max() > 0.1          # without the cap the faint cape glows, as it did on the knight
    assert stats["glow_share"] > 0.3             # and the report says so (accent_share counted only the slit)
    assert stats["accent_share"] < 0.02
    assert "glow_cap" not in stats


def test_a_glow_over_the_cap_keeps_only_what_stands_by_a_real_accent():
    alb, lit, cov = _cape_atlas()
    s = gr.settings(["--no-shade", "--glow", "#da294c", "--glow-width", "0"])
    _, emission, stats = gr.compose(alb, lit, cov, s)
    assert emission[30, 20].max() < 0.05         # the cape no longer glows
    assert emission[20, 52].max() > 0.5          # the eye slit still does
    assert stats["glow_share"] <= 0.08
    cap = stats["glow_cap"]
    assert cap["before"] > 0.3 and cap["after"] == stats["glow_share"] and cap["cap"] == 0.08
    assert cap["steps"] == {"#da294c": "only beside a solid accent"}


def test_a_glow_that_stays_over_the_cap_is_dropped_and_said():
    n = 64
    alb = np.zeros((n, n, 3))
    alb[:] = gr.to_linear(np.array((0.18, 0.18, 0.20)))
    alb[::2, :] = gr.to_linear(np.array((0.855, 0.161, 0.298)))   # half the figure in thin lines of the glow's red
    s = gr.settings(["--no-shade", "--glow", "#da294c", "--glow-width", "0"])
    _, emission, stats = gr.compose(alb, np.full((n, n), 0.5), np.ones((n, n)), s)
    assert stats["glow_cap"]["steps"] == {"#da294c": "dropped"}
    assert stats["glow_share"] == 0.0 and emission is None   # no emission map at all


def test_the_cap_is_a_setting_and_one_turns_it_off():
    assert gr.settings([])["glow_max_area"] == 0.08
    assert gr.settings(["--glow-max-area", "0.2"])["glow_max_area"] == 0.2
    assert gr.settings(["--glow-max-area", "1"])["glow_max_area"] == 1.0
    for bad in ("0", "1.5", "-0.1"):
        with pytest.raises(SystemExit):
            gr.settings(["--glow-max-area", bad])


def test_a_thin_accent_under_the_cap_is_left_exactly_as_before():
    alb, lit, cov = _atlas(64)   # the strip is 0.4 % of the figure, its glow about 3 %
    s = gr.settings(["--no-shade", "--glow", "#29e6ff"])
    _, emission, stats = gr.compose(alb, lit, cov, s)
    _, emission_off, _ = gr.compose(alb, lit, cov, gr.settings(["--no-shade", "--glow", "#29e6ff", "--glow-max-area", "1"]))
    assert "glow_cap" not in stats and np.array_equal(emission, emission_off)


def test_a_glow_only_on_the_head_lights_the_eyes_and_not_the_same_red_on_the_body():
    """The black knight's eye slit and its trims share one red: colour cannot tell them apart, the head can."""
    parents, heads, _roles = _humanoid()
    unity = _unity(dict(parents), dict(heads))
    s = gr.settings(["--glow-only-on-head", "--glow-not-on-hands"])
    assert set(gr.glow_off_bones(dict(parents), unity, s)) == set(gr.head_bones(dict(parents), unity))
    alb, lit, cov = _cape_atlas()
    alb[40, 44:60] = gr.to_linear(np.array((0.855, 0.161, 0.298)))   # a trim of the eye's red, on the body
    head = np.zeros(cov.shape)
    head[14:26, 40:64] = 1.0                                         # the head's skin, where the slit is
    s = gr.settings(["--no-shade", "--glow", "#da294c", "--glow-width", "0", "--glow-only-on-head"])
    _, emission, _ = gr.compose(alb, lit, cov, s, exclude=head)
    assert emission[20, 52].max() > 0.5 and emission[40, 52].max() == 0.0


def test_glow_only_on_the_head_and_not_on_the_head_cannot_both_be_asked():
    assert gr.settings([])["glow_only_on_head"] is False
    with pytest.raises(SystemExit):
        gr.settings(["--glow-only-on-head", "--glow-not-on-head"])


def test_the_log_says_when_the_glow_was_narrowed():
    assert gr.glow_note({"glow_share": 0.01}) is None
    note = gr.glow_note({"glow_cap": {"before": 0.2576, "after": 0.0602, "cap": 0.08,
                                      "steps": {"#da294c": "only beside a solid accent"}}})
    assert "25.8%" in note and "6.0%" in note and "#da294c: only beside a solid accent" in note
    assert "--glow-max-area 1" in note and "--glow-only-on-head" in note
