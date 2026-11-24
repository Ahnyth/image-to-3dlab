"""Hero parts: a part built on its own set into a figure, on the body (a holstered pistol) or at a limb's end (a boot).

The shapes are small synthetic stand-ins for what the Revolverheld and the Zauberer showed (2026-11-22): a thigh with a
thin slab standing off it where the picture has a holstered pistol, a strap crossing it, a lower leg whose boot is a
smooth tube, a built boot drawn from its outer side, a hard part whose flat sides came out slightly wavy. The graft's
claims are tested as mesh properties: what the new part replaces is gone, what lies behind it stays, and no edge round
it is left open.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab import hero_hands as hh
from image_to_3dlab import hero_parts as hp
from image_to_3dlab import photo_paint as pp

# --- shapes -------------------------------------------------------------------------------------------------


def box(lo, hi, subdivide=0):
    """A closed box from corner LO to corner HI as a part (faces out), split SUBDIVIDE times."""
    mesh = trimesh.creation.box(extents=np.subtract(hi, lo))
    mesh.apply_translation((np.asarray(lo, float) + np.asarray(hi, float)) / 2)
    for _ in range(subdivide):
        mesh = mesh.subdivide()
    return {"positions": np.asarray(mesh.vertices, float), "faces": np.asarray(mesh.faces, np.int64)}


def merged(*parts):
    return hh.merge_parts([dict(p) for p in parts])


def open_edges(part):
    group = hh.weld_groups(part["positions"])
    f = group[np.asarray(part["faces"])]
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def gauge_view(size=1376):
    """Pixal3D's front camera as a run writes it once the model faces +Z (`.svviews/transforms.json`)."""
    c2w = np.array([[-1.0, 0, 0, 0], [0, 0, 1.0, 2.83564091], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
    image = np.zeros((size, size, 4), np.uint8)
    image[..., 3] = 255
    return pp.View(image, c2w, 0.3490658503988659, "front")


# --- the figure's front camera ------------------------------------------------------------------------------


def test_staging_takes_picture_pixels_to_where_the_run_framed_them():
    from scripts.pixal3d_generate import pad_view

    picture = np.zeros((500, 300, 4), np.uint8)
    picture[60:461, 40:241] = (90, 90, 90, 255)                # the subject, 201 x 401 pixels
    picture[300:304, 100:104] = (255, 0, 0, 255)               # a mark on it
    view = np.asarray(pad_view(Image.fromarray(picture), 768))
    stage = hp.staging(hp.subject_box(picture[..., 3]), hp.subject_box(view[..., 3]))
    red = (view[..., 0] > 200) & (view[..., 1] < 60) & (view[..., 3] > 200)
    ys, xs = np.nonzero(red)
    x0, y0, x1, y1 = hp.staged_box((100, 300, 104, 304), stage)
    assert abs((x0 + x1) / 2 - (xs.min() + xs.max() + 1) / 2) < 1.0
    assert abs((y0 + y1) / 2 - (ys.min() + ys.max() + 1) / 2) < 1.0


def test_a_point_unprojected_onto_its_depth_comes_back_where_it_was():
    view = gauge_view()
    rng = np.random.default_rng(3)
    points = np.c_[rng.uniform(-0.3, 0.3, (20, 2)), np.full(20, 0.07)]
    x, y, _ = hp.view_pixels(view, points)
    back = hp.unproject(view, x, y, 0.07)
    assert np.allclose(back, points, atol=1e-9)


def test_a_crop_of_the_view_sees_points_where_the_whole_view_does():
    view = gauge_view()
    points = np.array([[-0.08, -0.04, 0.09], [-0.05, 0.0, 0.08]])
    x, y, _ = hp.view_pixels(view, points)
    crop = hp.crop_view(view, (530, 660, 730, 860), np.zeros((1024, 1024, 4), np.uint8))
    cx, cy, _ = hp.view_pixels(crop, points)
    assert np.allclose(cx, (x - 530) * 1024 / 200) and np.allclose(cy, (y - 660) * 1024 / 200)


def test_a_figure_without_cameras_is_seen_as_the_box_frame_sees_it():
    from image_to_3dlab import hero

    picture = Image.new("RGBA", (300, 500), (0, 0, 0, 0))
    picture.paste((80, 80, 80, 255), (40, 60, 241, 461))
    points = np.random.default_rng(4).uniform([-0.2, -0.45, -0.1], [0.2, 0.45, 0.1], (50, 3))
    view = hp.far_view(points, picture)
    x, y, _ = hp.view_pixels(view, points)
    bx, by = hero.project(points, hero.frame(points, hero.picture_box(picture)))
    assert np.abs(x - (bx + 0.5)).max() < 0.05 and np.abs(y - (by + 0.5)).max() < 0.05


# --- a part on the body -------------------------------------------------------------------------------------


def test_a_part_fills_its_box_by_height_and_stands_at_the_given_front():
    part = box((-0.5, -1.0, -0.25), (0.5, 1.0, 0.25))
    tf = hp.place_in_box(part["positions"], (0.1, 0.3, -0.2, 0.2), 0.15)
    p = hh.moved(tf, part["positions"])
    assert np.ptp(p[:, 1]) == pytest.approx(0.4) and p[:, 2].max() == pytest.approx(0.15)
    assert p[:, 0].mean() == pytest.approx(0.2) and p[:, 1].mean() == pytest.approx(0.0)


def thigh_and_slab(strap=False):
    """A thigh (a thick box) with a thin slab standing 1 cm off its front, as Pixal3D built the Revolverheld's
    holster; STRAP adds a thin strap across the thigh's front, wider than the slab, between thigh and slab."""
    thigh = box((-0.05, -0.2, -0.04), (0.05, 0.1, 0.04), subdivide=2)
    slab = box((-0.012, -0.12, 0.05), (0.012, 0.0, 0.058), subdivide=2)
    parts = [thigh, slab]
    if strap:
        parts.append(box((-0.06, -0.07, 0.041), (0.06, -0.05, 0.046), subdivide=3))
    return merged(*parts)


def test_the_body_behind_a_thin_slab_is_the_thigh():
    figure = thigh_and_slab()
    body = hp.body_behind(figure["positions"], figure["faces"], (-0.02, 0.02, -0.11, -0.01), 0.002, thin=0.02)
    assert np.allclose(body, 0.04, atol=1e-6)                  # the slab (0.008 thick) is passed, the thigh is not


def test_a_part_settles_with_its_back_on_the_body():
    back = np.full((4, 4), 0.06)
    back[0, 0] = 0.07
    body = np.full((4, 4), 0.04)
    dz = hp.settle(back, body, touch=0.1, tuck=0.001)
    assert np.quantile((back + dz) - body, 0.1) == pytest.approx(-0.001)


def test_a_part_that_leans_away_from_the_body_is_turned_along_it():
    # a body whose front leans back 11 degrees toward the top (a thigh), a part built upright (its back flat): the
    # turn lays the part's back along the body, so it neither stands off at the top nor sinks in at the bottom
    pixel, frame = 0.002, (-0.03, 0.03, -0.1, 0.1)
    rows, cols = 100, 30
    y = frame[3] - (np.arange(rows) + 0.5) * pixel
    body = np.repeat((0.04 - 0.2 * y)[:, None], cols, axis=1)
    part = box((-0.02, -0.08, 0.06), (0.02, 0.08, 0.08), subdivide=3)
    turn = hp.fit_tilt(hp.depth_maps(part["positions"], part["faces"], frame, pixel)[1], body, frame, pixel)
    centre = part["positions"].mean(axis=0)
    turned = (part["positions"] - centre) @ turn.T + centre
    _, back = hp.depth_maps(turned, part["faces"], frame, pixel)
    both = np.isfinite(back)
    slope = np.polyfit(np.repeat(y[:, None], cols, axis=1)[both], (back - body)[both], 1)[0]
    assert abs(slope) < 0.01 and np.degrees(np.arccos(turn[1, 1])) == pytest.approx(11.3, abs=0.5)


def test_a_part_is_set_on_the_thigh_behind_the_slab_not_on_the_slab():
    # the picture's box says where and how big; the build leans 15 degrees (Pixal3D's depth guess) and is as deep as
    # half its height: it ends upright with its back on the thigh (z 0.04), in front of which the slab stands
    figure = thigh_and_slab()
    part = box((-0.5, -2.0, -0.5), (0.5, 2.0, 0.5), subdivide=2)
    a = np.radians(15.0)
    leaning = part["positions"] @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]).T
    tf, report = hp.place_on_body(leaning, part["faces"], (-0.016, 0.016, -0.125, 0.004), 0.058,
                                  figure["positions"], figure["faces"])
    p = hh.moved(tf, leaning)
    assert abs(report["lean_deg"]) == pytest.approx(15.0, abs=1.0)
    low, high = p[p[:, 1] < -0.09], p[p[:, 1] > -0.03]
    assert low[:, 2].min() == pytest.approx(high[:, 2].min(), abs=0.002)        # upright again
    assert p[:, 2].min() == pytest.approx(0.04, abs=0.003)                      # its back on the thigh


def placed_pistol():
    """The new part, settled on the thigh: a little wider and longer than the slab, from the thigh's front out."""
    return box((-0.016, -0.125, 0.039), (0.016, 0.004, 0.06), subdivide=2)


def test_the_slab_goes_the_thigh_stays_and_nothing_is_left_open():
    figure = thigh_and_slab()
    part = placed_pistol()
    field = hp.part_field(part, pixel=0.001, grow=0.002)
    (cut,), _, report = hp.graft_on_body([figure], field, behind=0.002)
    p = cut["positions"]
    middle = p[cut["faces"]].mean(axis=1)
    assert not ((middle[:, 2] > 0.045) & (np.abs(middle[:, 0]) < 0.012)).any()     # no slab left
    assert np.ptp(p[:, 2]) == pytest.approx(0.08)                                   # the thigh whole
    assert open_edges(cut) == 0 and report["faces_removed"] > 0


def test_a_strap_cut_by_the_part_is_closed_at_both_ends():
    figure = thigh_and_slab(strap=True)
    part = placed_pistol()
    field = hp.part_field(part, pixel=0.001, grow=0.002)
    (cut,), _, report = hp.graft_on_body([figure], field, behind=0.002)
    middle = cut["positions"][cut["faces"]].mean(axis=1)
    strap = (middle[:, 1] < -0.049) & (middle[:, 1] > -0.071) & (middle[:, 2] > 0.0405)
    assert strap.any() and not (strap & (np.abs(middle[:, 0]) < 0.015)).any()      # cut out under the part ...
    assert open_edges(cut) == 0 and report["outlines_closed"] >= 2                  # ... and closed at both ends


def test_no_crack_where_the_outline_crosses_from_what_goes_to_what_stays():
    # one solid block reaching from behind the part's back to in front of it: its front half inside the outline goes,
    # its back half stays, and along the outline both halves meet the uncut rest without a crack (each face the outline
    # crosses is split, whether it goes or stays)
    block = box((-0.01, -0.2, 0.0), (0.01, 0.1, 0.08), subdivide=4)     # narrower than the part, longer
    part = box((-0.016, -0.125, 0.04), (0.016, 0.004, 0.09), subdivide=2)
    field = hp.part_field(part, pixel=0.001, grow=0.002)
    (cut,), _, report = hp.graft_on_body([block], field, behind=0.002)
    assert report["faces_removed"] > 0 and open_edges(cut) == 0


def test_a_slab_longer_than_the_new_part_leaves_no_flap_below_it():
    # Pixal3D's slab reaches 2 cm further down than the new part: the piece the outline cuts off below goes too, the
    # thigh stays whole
    thigh = box((-0.05, -0.2, -0.04), (0.05, 0.1, 0.04), subdivide=2)
    slab = box((-0.012, -0.14, 0.05), (0.012, 0.0, 0.058), subdivide=3)
    figure = merged(thigh, slab)
    field = hp.part_field(placed_pistol(), pixel=0.001, grow=0.002)
    (cut,), _, report = hp.graft_on_body([figure], field, behind=0.002)
    middle = cut["positions"][cut["faces"]].mean(axis=1)
    assert not (middle[:, 2] > 0.045).any() and report["pieces_dropped"] >= 1
    assert open_edges(cut) == 0


def test_a_hard_part_moves_with_the_mean_of_what_it_replaced():
    rows = np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.9, 0.1]])
    assert np.allclose(hp.rigid_weights(rows), [0.0, 0.8, 0.2])


# --- a part at a limb's end ---------------------------------------------------------------------------------


def stick_figure():
    """A glTF (Y-up) humanoid facing +Z in SkinTokens' tree shape, names shuffled: (parents, heads, role -> name)."""
    parents, heads, role = {}, {}, {}
    names = iter(f"bone_{n}" for n in np.random.default_rng(2).permutation(80))

    def add(key, parent, pos):
        role[key] = next(names)
        parents[role[key]] = role.get(parent)
        heads[role[key]] = np.array(pos, float)

    add("hips", None, (0, 1.0, 0)); add("s1", "hips", (0, 1.1, 0)); add("s2", "s1", (0, 1.2, 0))
    add("chest", "s2", (0, 1.35, 0)); add("neck", "chest", (0, 1.5, 0)); add("head", "neck", (0, 1.6, 0))
    for side, sx in (("L", 1.0), ("R", -1.0)):        # facing +Z the figure's left is +X
        add(f"{side}sh", "chest", (0.05 * sx, 1.45, 0)); add(f"{side}arm", f"{side}sh", (0.15 * sx, 1.45, 0))
        add(f"{side}fa", f"{side}arm", (0.4 * sx, 1.2, 0)); add(f"{side}hand", f"{side}fa", (0.55 * sx, 0.95, 0))
        add(f"{side}leg", "hips", (0.1 * sx, 0.95, 0)); add(f"{side}shin", f"{side}leg", (0.1 * sx, 0.5, 0))
        add(f"{side}foot", f"{side}shin", (0.1 * sx, 0.08, 0))
        add(f"{side}toe", f"{side}foot", (0.1 * sx, 0.02, 0.12))
    return parents, heads, role


def test_leg_bones_finds_each_leg_by_the_tree_and_the_figures_own_side():
    parents, heads, role = stick_figure()
    legs, frame = hp.leg_bones(parents, heads)
    assert legs["right"]["shin"] == role["Rshin"] and legs["right"]["foot"] == role["Rfoot"]
    assert legs["left"]["thigh"] == role["Lleg"] and legs["left"]["below"] == [role["Ltoe"]]
    assert np.allclose(frame[:, 2], [0, 0, 1], atol=1e-6)


def test_a_boot_drawn_from_its_outer_side_stands_toe_forward_outer_side_out():
    right, left = hp.BOOT_TURN["right"], hp.BOOT_TURN["left"]
    assert np.allclose(right @ [1, 0, 0], [0, 0, 1]) and np.allclose(right @ [0, 0, 1], [-1, 0, 0])
    assert np.allclose(left @ [-1, 0, 0], [0, 0, 1]) and np.allclose(left @ [0, 0, 1], [1, 0, 0])
    assert np.linalg.det(right) == pytest.approx(1.0) and np.linalg.det(left) == pytest.approx(1.0)


def turned_y(points, degrees):
    a = np.radians(degrees)
    r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return np.asarray(points) @ r.T


def test_a_boot_takes_the_old_foots_length_heading_and_floor():
    # the old foot: 0.09 long, its toe turned 12 degrees out, standing on -0.44; the new boot: twice as big
    old = box((-0.02, -0.44, -0.03), (0.02, -0.40, 0.06), subdivide=1)["positions"]
    old = turned_y(old, -12.0) + [-0.07, 0.0, 0.0]
    new = box((-0.04, 0.0, -0.06), (0.04, 0.36, 0.12), subdivide=1)["positions"]
    new[new[:, 1] > 0.1, 2] = np.clip(new[new[:, 1] > 0.1, 2], -0.06, 0.02)   # a shaft above the foot
    tf = hp.place_boot(new, old, floor=-0.44)
    p = hh.moved(tf, new)
    assert p[:, 1].min() == pytest.approx(-0.44)
    heading = hp.foot_heading(p[p[:, 1] < -0.40])
    assert np.degrees(np.arctan2(heading[0], heading[1])) == pytest.approx(-12.0, abs=1.0)
    assert tf[0, 0] ** 2 + tf[0, 1] ** 2 + tf[0, 2] ** 2 == pytest.approx(0.5 ** 2, rel=0.03)
    assert np.allclose(p[p[:, 1] < -0.40][:, [0, 2]].mean(axis=0), old[:, [0, 2]].mean(axis=0), atol=0.004)


def test_a_boot_given_its_height_takes_it_whatever_its_foot():
    # the picture says how tall the boot is (its top in the front view); the drawing's short foot no longer decides
    old = box((-0.02, -0.44, -0.03), (0.02, -0.40, 0.06), subdivide=1)["positions"]
    new = box((-0.04, 0.0, -0.06), (0.04, 0.36, 0.12), subdivide=1)["positions"]
    tf = hp.place_boot(new, old, floor=-0.44, height=0.18)
    p = hh.moved(tf, new)
    assert np.ptp(p[:, 1]) == pytest.approx(0.18) and p[:, 1].min() == pytest.approx(-0.44)


def test_a_tilted_sole_is_laid_flat():
    sole = box((-0.03, 0.0, -0.05), (0.03, 0.1, 0.08), subdivide=2)["positions"]
    a = np.radians(6.0)
    tilted = sole @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]).T
    level = hp.level_sole(tilted)
    p = hh.moved(level, tilted)
    low = p[p[:, 1] < p[:, 1].min() + 0.003]
    assert np.ptp(low[:, 2]) > 0.1                                # the whole sole touches the floor again


def test_a_boots_rim_is_where_its_top_is_lowest():
    # a shaft cut slanting, its top 2 cm lower at the back: the leg must be cut under the back's rim, or its closed
    # end stands above the boot there as a ledge (the Zauberer's boots, 2026-11-22)
    tube = trimesh.creation.cylinder(radius=0.02, height=0.18, sections=32)
    p = np.asarray(tube.vertices)[:, [0, 2, 1]] + [0.0, 0.09, 0.0]           # standing on y = 0, 0.18 tall
    top = p[:, 1] > 0.1
    p[top, 1] = 0.18 + 0.5 * p[top, 2]                                    # front (+z) 1 cm higher, back 1 cm lower
    assert hp.boot_rim(p) == pytest.approx(0.17, abs=0.002)


def test_an_old_leg_wider_than_the_new_boot_is_tucked_inside_it_up_to_its_top():
    # the old boot's top stood wider than the new boot at the back and showed as a ledge over its rim: between the cut
    # and the new boot's top every old point outside the boot goes just inside it; above its top the trousers stay
    boot = trimesh.creation.cylinder(radius=0.02, height=0.18, sections=32)
    boot_points = np.asarray(boot.vertices)[:, [0, 2, 1]] + [0.0, 0.09, 0.0]
    rings = np.linspace(0.1, 0.3, 11)
    old = np.concatenate([np.c_[0.025 * np.cos(a), np.full(24, y), 0.025 * np.sin(a)]
                          for y in rings for a in [np.linspace(0, 2 * np.pi, 24, endpoint=False)]])
    moved = hp.tuck_inside(old, boot_points, np.asarray(boot.faces), 0.12, 0.18)
    radius = np.hypot(moved[:, 0], moved[:, 2])
    band = (old[:, 1] >= 0.12) & (old[:, 1] <= 0.18)
    assert radius[band].max() < 0.02 and np.allclose(moved[~band], old[~band])


def lower_leg(x=-0.07):
    """An old lower leg: a shin tube (bone 1) down to an ankle at -0.385, a foot (bone 2) to the floor at -0.44;
    a coat panel (bone 3) hangs beside it down to -0.25."""
    shin = box((x - 0.02, -0.39, -0.02), (x + 0.02, -0.20, 0.02), subdivide=2)
    foot = box((x - 0.022, -0.44, -0.03), (x + 0.022, -0.38, 0.07), subdivide=2)
    coat = box((x - 0.12, -0.25, 0.03), (x - 0.08, -0.05, 0.04), subdivide=1)
    figure = merged(shin, foot, coat)
    n = [len(shin["positions"]), len(foot["positions"]), len(coat["positions"])]
    bone = np.repeat([1, 2, 3], n)
    figure["dense"] = np.eye(4)[bone]
    return figure, bone


def test_the_old_boot_goes_below_the_cut_and_the_cut_is_closed_inside_the_new_one():
    figure, bone = lower_leg()
    plan = hp.leg_plan(figure["positions"], bone, shin=1, foot=[2], knee=np.array([-0.07, -0.2, 0.0]),
                       ankle=np.array([-0.07, -0.385, 0.0]))
    (cut,), report = hp.graft_boot([figure], [bone], plan, cut_y=-0.27)
    p = cut["positions"]
    middle = p[cut["faces"]].mean(axis=1)
    on_leg = np.abs(middle[:, 0] + 0.07) < 0.03
    assert middle[on_leg, 1].min() >= -0.2701                    # nothing of the old boot below the cut
    assert (middle[:, 1] < -0.24).any()                           # the coat beside it stays whole
    assert open_edges(cut) == 0 and report["outlines_closed"] == 1


def test_a_sole_below_the_pictures_feet_is_measured_in_its_pixels():
    view = gauge_view()
    image = np.zeros_like(view.image)
    image[200:1000, 500:900, 3] = 255                            # the picture's subject ends at row 1000
    view = pp.View(image, view.cam_to_world, view.fov_x)
    boot = box((-0.05, -0.30, 0.0), (0.05, -0.10, 0.1), subdivide=1)["positions"]
    _, y, _ = hp.view_pixels(view, boot)
    lowest = y[boot[:, 1] <= boot[:, 1].min() + 1e-9].max()
    assert hp.sole_below_feet(view, boot) == pytest.approx(lowest - 1000.0)


def test_a_made_up_sole_under_the_old_foot_lowers_no_new_boot():
    # Pixal3D made up a sole below skin_sorcerer_c's drawn feet; the new boot stood on it, 71 px too low, and game_ready
    # cut everything below the picture's feet off: the boot's foot, so a straight shaft was left (2026-11-24)
    view = gauge_view()
    image = np.zeros_like(view.image)
    image[200:1000, 500:900, 3] = 255                            # the picture's subject ends at row 1000
    view = pp.View(image, view.cam_to_world, view.fov_x)
    foot = box((-0.05, -0.30, 0.0), (0.05, -0.10, 0.1), subdivide=1)["positions"]
    below = hp.sole_below_feet(view, foot)
    assert below > 20                                             # the old foot's sole: made up below the feet
    floor = hp.picture_floor(view, foot, float(foot[:, 1].min()))
    raised = foot + [0.0, floor - foot[:, 1].min(), 0.0]
    assert abs(hp.sole_below_feet(view, raised)) < 1.0            # a boot on this floor stands on the drawn feet
    high = foot + [0.0, 0.5, 0.0]                                 # a sole above the picture's feet stays as it is
    assert hp.picture_floor(view, high, float(high[:, 1].min())) == pytest.approx(float(high[:, 1].min()))


def test_a_boot_stands_where_the_game_step_will_cut_a_made_up_sole():
    figure = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])                 # 2 tall, standing on y = -1
    assert hp.cut_floor(-1.0, figure, 0.058) == pytest.approx(-1.0 + 0.116)
    assert hp.cut_floor(-0.5, figure, 0.058) == -0.5                       # an old floor above the cut stays
    assert hp.cut_floor(-1.0, figure, None) == -1.0


def test_a_boot_takes_the_old_legs_weights_point_by_point():
    figure, bone = lower_leg()
    plan = hp.leg_plan(figure["positions"], bone, shin=1, foot=[2], knee=np.array([-0.07, -0.2, 0.0]),
                       ankle=np.array([-0.07, -0.385, 0.0]))
    weights = hp.boot_weights(np.array([[-0.07, -0.43, 0.06], [-0.07, -0.30, 0.0]]), figure, bone, plan)
    assert weights.argmax(axis=1).tolist() == [2, 1]              # the toe moves with the foot, the shaft with the shin


# --- hard edges ---------------------------------------------------------------------------------------------


def test_a_box_has_six_planes_and_a_ball_none():
    cube = box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), subdivide=3)
    label, planes = hp.face_planes(cube["positions"], cube["faces"])
    assert len(planes) == 6 and (label >= 0).all()
    ball = trimesh.creation.icosphere(subdivisions=4)
    label, planes = hp.face_planes(np.asarray(ball.vertices), np.asarray(ball.faces))
    assert len(planes) == 0 and (label < 0).all()


def test_a_wavy_flat_side_is_made_flat_and_its_edges_sharp():
    cube = box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), subdivide=3)
    rng = np.random.default_rng(5)
    noisy = dict(cube, positions=cube["positions"] + rng.normal(0.0, 0.002, cube["positions"].shape))
    hard, report = hp.harden(noisy)
    p, n = hard["positions"], hard["normals"]
    assert report["planes"] == 6
    assert np.abs(np.abs(noisy["positions"]).max(axis=1) - 0.5).max() > 4e-3
    assert np.abs(np.abs(p).max(axis=1) - 0.5).max() < 1.5e-3      # every point back on its flat side
    corner = np.linalg.norm(p - [0.5, 0.5, 0.5], axis=1) < 1e-3
    assert corner.sum() == 3                                         # the corner split three ways ...
    assert np.allclose(np.sort(np.abs(n[corner]).max(axis=1)), 1.0, atol=1e-6)   # ... each with its side's normal


def test_a_planes_edge_follows_its_neighbours_not_single_stray_faces():
    # a face whose neighbours lie on a plane joins it, a plane face with one neighbour on it leaves: the edge the hard
    # shading draws runs straight instead of round single triangles (a notched holster panel, 2026-11-22)
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 2], [2, 4], [4, 5]])
    label = np.array([0, 0, -1, 0, 0, 1])
    normal = np.tile([0.0, 0.0, 1.0], (6, 1))
    planes = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    out = hp.tidy_planes(label, pairs, normal, planes)
    assert out[2] == 0 and out[5] == -1                  # the hole filled, the lone face of plane 1 dropped


def test_a_round_part_stays_round_where_it_is_not_flat():
    tube = trimesh.creation.cylinder(radius=0.3, height=1.0, sections=64)
    hard, report = hp.harden({"positions": np.asarray(tube.vertices, float), "faces": np.asarray(tube.faces)})
    assert report["planes"] == 2                                     # the two caps
    side = np.abs(hard["positions"][:, 2]) < 0.45
    radial = hard["positions"][side][:, :2] / np.linalg.norm(hard["positions"][side][:, :2], axis=1, keepdims=True)
    assert np.allclose(hard["normals"][side][:, :2], radial, atol=0.05)    # the side's normals still smooth


# --- counting -----------------------------------------------------------------------------------------------


def test_crumbs_of_a_build_go_and_its_pieces_are_counted():
    main = box((0, 0, 0), (1, 1, 1), subdivide=5)                 # 12,288 faces
    crumb = box((2, 2, 2), (2.01, 2.01, 2.01))                     # 12: a speck floating beside it
    part, dropped = hp.drop_crumbs(merged(main, crumb))
    assert dropped == 1 and hp.pieces(part) == 1
    part, dropped = hp.drop_crumbs(merged(main, box((3, 0, 0), (4, 1, 1), subdivide=3)))
    assert dropped == 0 and hp.pieces(part) == 2                   # a second piece of its own is no crumb


def test_a_surface_that_touches_itself_is_counted():
    # two boxes sharing an edge: four faces meet there, as where a build's thin wall pinched onto itself (the
    # Zauberer's first boot build: 21,107 such edges, and the game-ready step could not thin it)
    single = box((0, 0, 0), (1, 1, 1), subdivide=1)
    faults = hp.edge_faults(single)
    assert faults["open"] == 0 and faults["shared_by_more"] == 0 and faults["share"] == 0.0
    touching = merged(single, box((1, 1, 0), (2, 2, 1), subdivide=1))
    faults = hp.edge_faults(touching)
    assert faults["shared_by_more"] > 0 and faults["share"] > 0.0


# --- the boot's top, found in the picture -------------------------------------------------------------------------


def legs_picture(top=(900, 900), boot=(25, 25, 28), trousers=(95, 60, 50), strap=True, background=False):
    """A front view (RGBA) of two legs from the hips (y 500) to the soles (y 1100), knees at 800 and ankles at 1040:
    trousers above each TOP, boots below it (a grey strap round the shaft at 960-975), each foot wider than its shaft.
    Returns (picture, {side: (knee (x, y), ankle (x, y))}) as `boot_top` takes them; the figure's right leg is on the
    picture's left."""
    image = np.zeros((1200, 600, 4), dtype=np.uint8)
    legs = {}
    for side, middle, line in (("right", 200, top[0]), ("left", 400, top[1])):
        image[500:1100, middle - 30:middle + 30] = (*trousers, 255)
        image[line:1100, middle - 32:middle + 32] = (*boot, 255)
        image[1040:1100, middle - 40:middle + 50] = (*boot, 255)
        if strap:
            image[960:975, middle - 32:middle + 32] = (130, 130, 135, 255)
        legs[side] = ((float(middle), 800.0), (float(middle), 1040.0))
    if background:
        image[..., 3] = 255
    return image, legs


def test_the_boots_top_is_where_its_colour_ends_up_the_shin():
    image, legs = legs_picture()
    found = hp.boot_top(image, legs)
    assert found["top"] == pytest.approx(900, abs=4), found
    assert set(found["sides"]) == {"right", "left"}


def test_shoes_bare_legs_one_colour_or_legs_that_disagree_are_no_boots():
    shoes = hp.boot_top(*legs_picture(top=(1030, 1030), strap=False))
    bare = hp.boot_top(*legs_picture(boot=(225, 180, 150), strap=False))
    plain = hp.boot_top(*legs_picture(boot=(95, 60, 50), strap=False))
    apart = hp.boot_top(*legs_picture(top=(860, 960)))
    for found in (shoes, bare, plain, apart):
        assert found["top"] is None and found["why"]
    assert "shoes" in shoes["why"] and "skin" in bare["why"] and "disagree" in apart["why"]


def test_a_boot_built_as_a_pair_one_behind_the_other_is_no_boot():
    one = box((0.0, 0.0, 0.0), (0.5, 0.8, 0.25), subdivide=1)["positions"]          # toe along x, deep along z
    pair = np.concatenate([one, one + [0.0, 0.0, 0.55]])                             # a second one behind it
    assert hp.single_boot(one)[0] and not hp.single_boot(pair)[0]
    assert hp.single_boot(pair)[1] == pytest.approx(1.6, abs=0.01)
