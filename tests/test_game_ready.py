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
    assert m == {"texels_per_cm": 0.0, "atlas_covered": 0.0, "texels_twice": 0.0}


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
    s = gr.settings(["--no-shade", "--glow", "#d91f1a", "--glow-max-width", "4", "--glow-width", "0"])
    _, emission, _ = gr.compose(alb, lit, cov, s)
    assert emission[4, 15].max() > 0.5
    assert emission[20, 14].max() == 0.0
    s_all = gr.settings(["--no-shade", "--glow", "#d91f1a", "--glow-width", "0"])
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


def test_compose_keeps_the_glow_off_an_excluded_area():
    alb, lit, cov = _atlas()
    s = gr.settings(["--no-shade", "--glow", "#29e6ff", "--glow-width", "0"])
    exclude = np.zeros(cov.shape)
    exclude[4:6, 2:4] = 1.0          # the left half of the cyan strip lies on the head
    _, emission, _ = gr.compose(alb, lit, cov, s, exclude=exclude)
    assert emission[5, 2].max() == 0.0
    assert emission[5, 5].max() > 0.5


# --- the face: thin with the head kept, a sharper head, colour from a finer paint ----------------

def test_the_head_switches_are_off_by_default():
    s = gr.settings([])
    assert s["decimate_to"] == 0 and s["head_protect"] == 0.0 and s["head_texel"] == 1.0
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
