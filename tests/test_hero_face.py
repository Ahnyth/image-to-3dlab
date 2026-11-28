"""Hero face: finding the face between a brim and a collar, putting a face built alone where the picture has it, taking
its skin seen from the front, and setting it in under the figure's cut edge.

The shapes are small synthetic stand-ins for what the Zauberer showed (2026-11-22): a head under a wide brim, a
strand standing in front of the face, a collar hidden behind the jaw, a new face a little larger than the old one,
and a build that is a hollow shell with its inner wall joined to its skin.
"""
from __future__ import annotations

import numpy as np
import pytest

from image_to_3dlab import hero_face as hf
from image_to_3dlab import hero_hands as hh

# --- shapes -------------------------------------------------------------------------------------------------------


def sphere(radius, centre=(0.0, 0.0, 0.0), rings=48, segments=64):
    """A closed UV sphere, faces out (counter-clockwise seen from outside)."""
    vertices = [[0.0, radius, 0.0]]
    for i in range(1, rings):
        theta = np.pi * i / rings
        for j in range(segments):
            phi = 2 * np.pi * j / segments
            vertices.append([radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta),
                             radius * np.sin(theta) * np.cos(phi)])
    vertices.append([0.0, -radius, 0.0])
    faces = []
    for j in range(segments):
        faces.append([0, 1 + j, 1 + (j + 1) % segments])
    for i in range(rings - 2):
        a, b = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            j1 = (j + 1) % segments
            faces += [[a + j, b + j, b + j1], [a + j, b + j1, a + j1]]
    last, end = 1 + (rings - 2) * segments, len(vertices) - 1
    for j in range(segments):
        faces.append([last + j, end, last + (j + 1) % segments])
    return np.asarray(vertices) + np.asarray(centre), np.asarray(faces)


def box(lo, hi):
    """A closed box, faces out."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    return v, f


def joined(*meshes):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(v)
        faces.append(np.asarray(f) + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def disk_window(radius, centre=(0.0, 0.0), width=0.2, pixel=0.002):
    """A window as `face_window` returns it, but a plain disk seen from the front."""
    cx, cy = centre
    frame = (cx - 2 * radius, cx + 2 * radius, cy - 2 * radius, cy + 2 * radius)
    rows, cols = hf.canvas(frame, pixel)
    x = frame[0] + (np.arange(cols) + 0.5) * pixel
    y = frame[3] - (np.arange(rows) + 0.5) * pixel
    mask = (x[None, :] - cx) ** 2 + (y[:, None] - cy) ** 2 <= radius ** 2
    return {"box": frame, "pixel": pixel, "field": hf.signed_distance(mask, pixel), "mask": mask, "outline": mask,
            "depth": np.zeros(mask.shape), "width": width, "seed": (rows // 2, cols // 2), "floor": None}


# --- measuring the head ---------------------------------------------------------------------------------------------

def test_plate_floor_finds_a_brims_underside_and_nothing_without_one():
    rng = np.random.default_rng(0)
    head_y = rng.uniform(0.0, 0.20, 6000)
    head_x = rng.uniform(-0.04, 0.04, 6000)
    brim_y = rng.uniform(0.15, 0.16, 3000)
    brim_x = rng.uniform(-0.12, 0.12, 3000)
    floor = hf.plate_floor(np.r_[head_y, brim_y], np.r_[head_x, brim_x])
    assert floor == pytest.approx(0.15, abs=0.005)
    assert hf.plate_floor(head_y, head_x) is None
    assert hf.head_width(np.r_[head_y, brim_y], np.r_[head_x, brim_x], floor) == pytest.approx(0.08, abs=0.01)


def test_depth_map_sees_the_nearest_surface_and_who_made_it():
    back = box((-1, -1, -0.2), (1, 1, 0.0))
    front = box((-0.5, -0.5, 0.3), (0.5, 0.5, 0.5))
    v, f = joined(back, front)
    depth, owner = hf.depth_map(v, f, (-2, 2, -2, 2), 0.05)
    middle = depth.shape[0] // 2, depth.shape[1] // 2
    assert depth[middle] == pytest.approx(0.5)
    assert owner[middle] >= 12                                   # one of the front box's faces
    assert depth[middle[0], 6] == -np.inf and owner[middle[0], 6] == -1          # nothing at x = -1.7
    assert depth[middle[0], 24] == pytest.approx(0.0)            # x = -0.775: the back box's front
    # one big triangle and many small ones give the same picture
    many = hf.depth_map(*sphere(0.8, rings=60, segments=80), (-1, 1, -1, 1), 0.02)[0]
    few = hf.depth_map(*sphere(0.8, rings=6, segments=8), (-1, 1, -1, 1), 0.02)[0]
    assert np.isfinite(many).sum() == pytest.approx(np.pi * 40 ** 2, rel=0.03)
    assert np.isfinite(few).sum() > 0.6 * np.isfinite(many).sum()


def test_pixel_hits_reports_every_layer_not_only_the_nearest():
    back = box((-1, -1, -0.2), (1, 1, 0.0))
    front = box((-0.5, -0.5, 0.3), (0.5, 0.5, 0.5))
    v, f = joined(back, front)
    pixel_id, z, face = hf.pixel_hits(v, f, (-2, 2, -2, 2), 0.05)
    rows, cols = hf.canvas((-2, 2, -2, 2), 0.05)
    middle = (rows // 2) * cols + cols // 2
    assert sorted(np.round(z[pixel_id == middle], 6)) == [-0.2, 0.0, 0.3, 0.5]   # both boxes, front and back
    assert len(pixel_id) == len(z) == len(face) and hf.pixel_hits(v, f[:0], (-2, 2, -2, 2), 0.05)[0].size == 0


def test_face_outline_crosses_a_strand_but_stops_at_a_brim_and_a_chin():
    depth = np.zeros((60, 40))
    depth[:10] = 1.0                       # a brim, far in front, above
    depth[15:22, 10:14] = 0.1              # a strand in front of an eye: a small step
    for k, row in enumerate(range(48, 51)):
        depth[row:] = -0.1 * (k + 1)       # under the chin the surface falls away over three rows
    facing = np.ones(depth.shape)
    outline = hf.face_outline(depth, facing, (30, 20), jump=0.25)
    assert outline[18, 12]                 # the strand is part of the face
    assert not outline[5].any()            # the brim is not
    assert outline[46, 20] and not outline[52, 20]   # the chin's slope ends it
    facing[:, 35:] = 0.1                   # the side turning away ends it too
    assert not hf.face_outline(depth, facing, (30, 20), jump=0.25)[30, 36]


def test_face_shape_is_a_box_above_the_nose_and_narrows_to_the_chin():
    shape = hf.face_shape((100, 100), (50, 50), 10, 90, 30)
    assert shape[15, 28]                   # a high corner: brow and hair under a brim are about as wide as the eyes
    assert not hf.face_shape((100, 100), (50, 50), 10, 90, 30, upper_power=2.0)[15, 28]   # an ellipse is not
    assert shape[50, 21] and not shape[50, 19]
    assert shape[88, 50] and not shape[88, 38]   # the chin is narrow
    assert not shape[95, 50] and not shape[5, 50]


def test_column_reach_walks_up_and_down_from_the_seed():
    outline = np.zeros((20, 5), bool)
    outline[4:15, 2] = True
    outline[17:19, 2] = True
    assert hf.column_reach(outline, (8, 2)) == (4, 14)


def test_signed_distance_is_zero_on_the_edge_positive_inside():
    window = disk_window(0.1)
    values = hf.window_values(window, np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.15, 0.0, 0.0], [5, 5, 0]]))
    assert values[0] == pytest.approx(0.1, abs=0.004)
    assert values[1] == pytest.approx(0.0, abs=0.003)
    assert values[2] == pytest.approx(-0.05, abs=0.004)
    assert values[3] == -hf.FAR


def test_window_mask_shrinks_smooths_and_keeps_the_seeds_piece():
    outline = np.zeros((80, 80), bool)
    outline[10:70, 10:70] = True
    outline[30:34, 70:79] = True           # a thin spur the margin removes
    mask = hf.window_mask(outline, margin=5, smooth=2, seed=(40, 40))
    assert mask[40, 40] and not mask[12, 40] and mask[17, 40] and not mask[32, 74]


# --- the whole measure on a head under a brim -----------------------------------------------------------------------

def test_face_window_takes_the_face_under_the_brim_and_above_the_chin():
    head = sphere(0.05, rings=40, segments=60)
    brim = box((-0.15, 0.035, -0.15), (0.15, 0.040, 0.15))
    collar = box((-0.07, -0.08, -0.03), (0.07, -0.045, 0.03))
    v, f = joined(head, brim, collar)
    is_head = np.zeros(len(v), bool)
    is_head[:len(head[0]) + len(brim[0])] = True
    window = hf.face_window(v, f, is_head)
    assert window["floor"] == pytest.approx(0.035, abs=0.004)
    x0, x1, y0, y1 = hf.window_bounds(window)
    assert y1 < 0.035                      # under the brim
    assert y0 > -0.05                      # above the chin
    assert x0 < -0.015 and x1 > 0.015      # wide enough for both eyes
    centre = window["box"][0] + (window["seed"][1] + 0.5) * window["pixel"]
    assert centre == pytest.approx(0.0, abs=0.01)


def test_solid_cutout_puts_back_eyes_the_background_remover_took_but_not_the_background():
    picture = np.full((40, 40, 3), 200, np.uint8)
    picture[15:18, 12:16] = [40, 230, 255]                   # a glowing eye
    cut = np.zeros((40, 40, 4), np.uint8)
    cut[5:35, 5:35, :3], cut[5:35, 5:35, 3] = 120, 255       # the face
    cut[15:18, 12:16, 3] = 0                                 # ... with the eye taken for background
    cut[14, 12:16, 3] = 180                                  # ... and a half-transparent rim above it
    cut[5, 5:35, 3] = 140                                    # the outline's own soft edge
    solid, changed = hf.solid_cutout(picture, cut)
    assert changed == 12 + 4
    assert (solid[14:18, 12:16, 3] == 255).all() and (solid[16, 13, :3] == [40, 230, 255]).all()
    assert (solid[5, 5:35, 3] == 140).all()                  # the soft edge stays soft
    assert (solid[0:5, :, 3] == 0).all() and (solid[20, 20] == [120, 120, 120, 255]).all()


def test_face_window_finds_the_nose_not_a_collar_standing_in_front_below_it():
    head = sphere(0.05, rings=40, segments=60)
    collar = box((-0.03, -0.075, 0.0), (0.03, -0.04, 0.07))      # nearer to the camera than the nose, below the chin
    v, f = joined(head, collar)
    is_head = np.zeros(len(v), bool)
    is_head[:len(head[0])] = True
    window = hf.face_window(v, f, is_head)
    nose_y = window["box"][3] - (window["seed"][0] + 0.5) * window["pixel"]
    assert window["floor"] is None and abs(nose_y) < 0.01
    assert hf.window_bounds(window)[3] < hf.BROW_SHARE * window["width"] + 0.002          # no brim: the seam stays below the hairline's cap


def test_window_doubts_name_a_window_too_tall_and_a_nose_too_high():
    window = disk_window(0.03)
    assert hf.window_doubts(window) == []
    tall = dict(window, mask=window["mask"] & (np.abs(np.arange(window["mask"].shape[1]) - 30) < 8)[None, :])
    assert any("taller than wide" in d for d in hf.window_doubts(tall))
    high = dict(window, seed=(window["seed"][0] - 12, window["seed"][1]))
    assert any("nose was found" in d for d in hf.window_doubts(high))


def test_crop_box_is_square_padded_and_inside_the_picture():
    assert hf.crop_box(np.array([100, 140]), np.array([200, 230]), 768, 1376) == (80, 175, 160, 255)
    left, top, right, bottom = hf.crop_box(np.array([0, 60]), np.array([0, 40]), 768, 1376)
    assert (left, top) == (0, 0) and right - left == bottom - top == 120


# --- lining up ------------------------------------------------------------------------------------------------------

def turn_z(degrees):
    a = np.radians(degrees)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def test_similarity_2d_recovers_size_turn_and_shift_in_the_picture_plane():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(100, 3))
    target = 0.12 * points @ turn_z(4).T + [0.3, -0.2, 0.0]
    tf = hf.similarity_2d(points, target)
    assert np.allclose(hh.moved(tf, points)[:, :2], target[:, :2], atol=1e-9)
    assert tf[2, 2] == pytest.approx(0.12) and tf[2, 3] == 0.0


def test_box_to_box_maps_a_subject_between_two_framings():
    rgba = np.zeros((20, 30, 4), np.uint8)
    rgba[5:15, 10:20, 3] = 255
    source = hf.alpha_box(rgba)
    assert source == (10.0, 5.0, 20.0, 15.0)
    mapped = hf.box_to_box(np.array([[10.0, 5.0], [15.0, 10.0]]), source, (100.0, 50.0, 120.0, 70.0))
    assert np.allclose(mapped, [[100.0, 50.0], [110.0, 60.0]])


def test_depth_fit_moves_a_face_built_deep_and_turned_onto_the_figures_face():
    v, f = sphere(0.05)
    window = disk_window(0.03)
    window["depth"], _ = hf.depth_map(v, f, window["box"], window["pixel"])
    face_v, face_f = sphere(0.05, centre=(0.0, 0.0, -0.037))
    face_v[:, 2] -= 0.1 * face_v[:, 0]                       # and turned a little about Y
    piece, _ = hf.cut_face({"positions": face_v, "faces": face_f}, window)
    tf = hf.depth_fit(window, piece)
    assert tf[2, 3] == pytest.approx(0.037, abs=5e-4) and tf[2, 0] == pytest.approx(0.1, abs=0.01)
    assert np.allclose(tf[:2], np.eye(4)[:2])                # X and Y stay where the picture put them


# --- cutting and tucking --------------------------------------------------------------------------------------------

def test_front_piece_keeps_the_front_not_the_back_of_the_head_or_specks():
    v, f = sphere(0.05)
    part = {"positions": v, "faces": f}
    cut, _ = hh.cut_part_values(part, -hf.window_values(disk_window(0.03), v))
    keep = hf.front_piece(cut["positions"], cut["faces"])
    middles = cut["positions"][cut["faces"]].mean(axis=1)
    assert (middles[keep, 2] > 0).all() and (middles[~keep, 2] < 0).all()
    assert keep.sum() and (~keep).sum()


def hollow_head(radius=0.05, wall=0.002):
    """A head built as a hollow shell, as Pixal3D builds some: a skin and an inner wall WALL behind it, facing in,
    joined to it through a slit over the brow (the wall's faces there run into the skin's)."""
    outer = sphere(radius, rings=40, segments=56)
    inner = sphere(radius - wall, rings=40, segments=56)
    v, f = joined(outer, (inner[0], inner[1][:, ::-1]))
    n = len(outer[0])
    # the slit: a band of points on the brow where the inner wall's points are the skin's
    brow = (outer[0][:, 1] > 0.02) & (outer[0][:, 1] < 0.024) & (outer[0][:, 2] > 0.03)
    v[n:][brow] = outer[0][brow]
    return v, f


def test_face_relief_takes_the_skin_of_a_hollow_build_and_nothing_behind_it():
    v, f = hollow_head()
    window = disk_window(0.035, width=0.1)
    frame = (np.zeros(3), 0.001, np.array([500.0, 500.0]))              # 1 mm a pixel, picture centre at 500
    face = hf.face_relief({"positions": v, "faces": f}, window, (400, 400, 600, 600), 1024, frame)
    piece, _ = hf.cut_face(face, window)
    assert np.abs(np.linalg.norm(piece["positions"], axis=1) - 0.05).max() < 1e-3     # all of it on the skin
    normals = hf.face_normals(piece["positions"], piece["faces"])
    assert (normals[:, 2] > 0).mean() > 0.99                                          # facing the camera
    tri = piece["positions"][piece["faces"]]
    seen = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])).sum()
    assert seen == pytest.approx(np.pi * 0.035 ** 2, rel=0.03)                     # one layer: no wall behind it


def roof(half=0.04, step=0.001, height=0.05):
    """A build whose front is a sharp ridge down the middle, as the Zauberer's nose came out built alone: z = HEIGHT -
    |x|, two planes meeting at 90 degrees along x = 0."""
    xs = np.arange(-half, half + step / 2, step)
    x, y = np.meshgrid(xs, xs)
    v = np.stack([x.ravel(), y.ravel(), height - np.abs(x.ravel())], axis=1)
    n = len(xs)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    f = np.concatenate([np.stack([c, d, b], -1).reshape(-1, 3), np.stack([c, b, a], -1).reshape(-1, 3)])
    return v, f


def sharpest_turn(points, row_y, pixel):
    """The most the surface's slope turns (degrees) from one pixel to the next along the row at ROW_Y."""
    row = points[np.abs(points[:, 1] - row_y) < pixel / 2]
    row = row[np.argsort(row[:, 0])]
    angle = np.degrees(np.arctan2(np.diff(row[:, 2]), np.diff(row[:, 0])))
    return float(np.abs(np.diff(angle)).max())


def test_round_relief_rounds_a_crease_keeps_planes_and_does_not_reach_across_a_cliff():
    x = np.arange(80) - 39.5
    ridge = np.tile(-np.abs(x), (60, 1))                              # pixel units: 90 degrees at the middle
    out = hf.round_relief(ridge, np.ones(ridge.shape, bool), 3.0)
    angle = np.degrees(np.arctan(np.diff(out[30])))
    assert np.abs(np.diff(angle)).max() < 20.0                        # the crease is a curve now
    far = (np.abs(x) > 15) & (np.abs(x) < 25)                         # the planes away from it and from the edge
    assert np.allclose(out[20:40][:, far], ridge[20:40][:, far], atol=1e-6)
    step = np.zeros((60, 40))
    step[30:] = 10.0                                                  # a brim's edge: a cliff, not a crease
    out = hf.round_relief(step, np.ones(step.shape, bool), 3.0, jump=5.0)
    assert np.allclose(out, step, atol=1e-9)
    holes = np.where(np.arange(40)[None, :] < 5, -np.inf, step)
    out = hf.round_relief(holes, np.ones(step.shape, bool), 3.0, jump=5.0)
    assert np.isneginf(out[:, :5]).all() and np.allclose(out[:, 5:], step[:, 5:], atol=1e-9)


def test_face_relief_rounds_a_build_made_of_crisp_planes():
    v, f = roof()
    window = disk_window(0.03, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    frame = (np.zeros(3), 0.001, np.array([500.0, 500.0]))
    sharp = hf.face_relief({"positions": v, "faces": f}, window, (400, 400, 600, 600), 1024, frame, rounding=0.0)
    face = hf.face_relief({"positions": v, "faces": f}, window, (400, 400, 600, 600), 1024, frame)
    middle = face["positions"][np.argmin(np.abs(face["positions"][:, 1]))][1]
    assert sharpest_turn(sharp["positions"], middle, window["pixel"]) > 40.0        # the build as it is
    assert sharpest_turn(face["positions"], middle, window["pixel"]) < 20.0         # no hard line down the ridge
    flank = ((np.abs(np.abs(face["positions"][:, 0]) - 0.012) < 0.002)               # the planes keep their place
             & (np.abs(face["positions"][:, 1]) < 0.015))
    assert np.allclose(face["positions"][flank, 2], 0.05 - np.abs(face["positions"][flank, 0]), atol=1e-5)


def painted_plane(colour, half=0.05, step=0.002):
    """A flat figure facing the camera at z = 0, painted one COLOUR all over (its own material and UVs)."""
    import io

    from PIL import Image

    xs = np.arange(-half, half + step / 2, step)
    x, y = np.meshgrid(xs, xs)
    v = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    n = len(xs)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    f = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(buffer, format="PNG")
    return {"positions": v, "faces": f, "uvs": (v[:, :2] + half) / (2 * half),
            "material": hf.picture_material(buffer.getvalue())}


CROP = (400, 400, 600, 600)
FRAME = (np.zeros(3), 0.001, np.array([500.0, 500.0]))        # 1 mm a picture pixel, the picture's centre at 500


def test_seam_paint_reads_the_figures_paint_against_the_crops_just_outside_the_seam():
    window = disk_window(0.02, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    crop = np.full((1024, 1024, 3), 100.0)
    points, offsets = hf.seam_paint([painted_plane((130, 120, 110))], window, crop, CROP, FRAME)
    assert len(points) > 20
    distance = np.hypot(points[:, 0], points[:, 1])
    assert distance.min() >= 0.02 - window["pixel"] and distance.max() <= 0.02 + 5 * window["pixel"]
    assert np.allclose(offsets, [30, 20, 10])
    assert len(hf.seam_paint([{k: v for k, v in painted_plane((0, 0, 0)).items() if k != "material"}], window, crop,
                             CROP, FRAME)[0]) == 0                         # an unpainted figure: nothing to match


def test_seam_paint_reads_a_coarse_figure_at_every_pixel_of_the_band():
    # the Nachtschwinge's 8k rigged model had few points in the band just outside the seam, and its cheek, 5 L*
    # lighter than the crop there, still drew a line down the seam (round 28 of the AssetFurnace Support notes)
    window = disk_window(0.02, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    coarse = painted_plane((130, 120, 110), step=0.05)
    values = hf.window_values(window, coarse["positions"])
    assert not ((values <= 0) & (values > -hf.RING_PIXELS * window["pixel"])).any()   # no point of it in the band
    points, offsets = hf.seam_paint([coarse], window, np.full((1024, 1024, 3), 100.0), CROP, FRAME)
    turn = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    assert len(points) > 100 and np.histogram(turn, bins=12, range=(-180, 180))[0].min() > 0   # all round the seam
    assert np.allclose(offsets, [30, 20, 10])


def test_offset_field_meets_the_seam_all_round_and_runs_smoothly_inside():
    window = disk_window(0.02, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    turn = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    ring = 0.0203 * np.stack([np.cos(turn), np.sin(turn)], axis=1)
    even = hf.offset_field(window, ring, np.tile([30.0, 20.0, -10.0], (len(ring), 1)))
    inside = window["mask"]
    assert np.allclose(even[inside], [30.0, 20.0, -10.0], atol=0.5)      # darker all round: lighter all over
    left = np.where(ring[:, :1] < 0, 40.0, 0.0) * np.ones((1, 3))
    split = hf.offset_field(window, ring, left)[..., 0]
    middle = window["seed"]
    assert split[middle] == pytest.approx(20.0, abs=2.0)                  # halfway between the two sides
    core = window["field"] > 8 * window["pixel"]                          # away from where the two sides meet
    assert np.abs(np.diff(split, axis=1)[core[:, 1:] & core[:, :-1]]).max() < 2.0       # no step inside
    assert np.allclose(hf.offset_field(window, ring, np.full((len(ring), 3), 500.0))[inside], hf.PAINT_LIMIT)


def test_matched_picture_lifts_the_crop_to_the_figures_paint_and_keeps_its_detail():
    window = disk_window(0.02, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    crop = np.full((1024, 1024, 3), 100.0)
    crop[500:524, 500:524] = 20.0                                         # an eye's dark outline in the middle
    figure = painted_plane((130, 120, 110))
    points, offsets = hf.seam_paint([figure], window, crop, CROP, FRAME)
    out = hf.matched_picture(crop, window, hf.offset_field(window, points, offsets), CROP, FRAME).astype(float)
    assert np.allclose(out[440, 440], [130, 120, 110], atol=1.0)          # inside the face, away from the dot
    assert np.allclose(out[512, 512] - out[440, 440], [-80, -80, -80], atol=1.0)   # the dot as dark as before


def test_match_face_paint_repaints_the_face_to_meet_the_figure_and_leaves_an_unpainted_one():
    import io

    from PIL import Image

    window = disk_window(0.02, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    crop = np.full((1024, 1024, 3), 100.0)
    face = {"positions": np.zeros((3, 3)), "faces": np.array([[0, 1, 2]]), "material": "crop as it is"}
    matched, offset = hf.match_face_paint(face, [painted_plane((130, 120, 110))], window, crop, CROP, FRAME)
    assert offset == pytest.approx([30.0, 20.0, 10.0])
    doc, binary, _ = matched["material"]
    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    start = view.get("byteOffset", 0)
    picture = np.asarray(Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGB"))
    assert np.allclose(picture[512, 512], [130, 120, 110], atol=1)       # the face's middle, now the figure's tone
    bare = {k: v for k, v in painted_plane((0, 0, 0)).items() if k != "material"}
    assert hf.match_face_paint(face, [bare], window, crop, CROP, FRAME) == (face, None)
    webp = painted_plane((130, 120, 110))                     # a texture whose picture sits in an extension (WebP)
    doc = webp["material"][0]
    doc["textures"][0] = {"sampler": 0, "extensions": {"EXT_texture_webp": {"source": doc["textures"][0]["source"]}}}
    assert hf.match_face_paint(face, [webp], window, crop, CROP, FRAME)[1] == pytest.approx([30.0, 20.0, 10.0])


def test_relief_step_follows_the_models_own_faces_inside_the_window():
    window = disk_window(0.02, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    pixel = window["pixel"]
    assert hf.relief_step([painted_plane((0, 0, 0), step=0.0005)], window) == pixel           # a fine model
    assert hf.relief_step([painted_plane((0, 0, 0), step=0.003)], window) == pytest.approx(0.0015)
    assert hf.relief_step([painted_plane((0, 0, 0), step=0.01)], window) == pytest.approx(hf.COARSEST_STEP * pixel)
    far = painted_plane((0, 0, 0), step=0.003)
    far["positions"] = far["positions"] + [1.0, 0.0, 0.0]
    assert hf.relief_step([far], window) == pixel                                             # nothing in the window
    back = painted_plane((0, 0, 0), step=0.01)
    back["positions"] = back["positions"] - [0.0, 0.0, 0.05]                                  # the back of the head
    assert hf.relief_step([painted_plane((0, 0, 0), step=0.0005), back], window) == pixel    # does not count


def test_face_relief_laid_out_coarser_is_lighter_and_reaches_as_far_past_the_seam():
    v, f = roof()
    window = disk_window(0.03, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    pixel = window["pixel"]
    fine = hf.face_relief({"positions": v, "faces": f}, window, CROP, 1024, FRAME)
    coarse = hf.face_relief({"positions": v, "faces": f}, window, CROP, 1024, FRAME, step=4 * pixel)
    assert len(coarse["faces"]) < len(fine["faces"]) / 12
    past = -hf.window_values(window, coarse["positions"])
    assert past.max() >= (hf.SKIRT_PIXELS + 4) * pixel                  # the skirt under the cut edge stays whole
    new, _ = hh.cut_part_values(coarse, -(hf.window_values(window, coarse["positions"]) + hf.SKIRT_PIXELS * pixel))
    tri = new["positions"][new["faces"]]
    seen = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])).sum()
    assert seen == pytest.approx(np.pi * (0.03 + hf.SKIRT_PIXELS * pixel) ** 2, rel=0.03)   # no bite out of it


def test_relief_puts_two_triangles_on_every_square_of_kept_pixels():
    depth = np.zeros((4, 5))
    keep = np.ones((4, 5), bool)
    keep[0, 0] = False
    part = hf.relief(depth, (0.0, 5.0, 0.0, 4.0), 1.0, keep)
    assert len(part["positions"]) == 19 and len(part["faces"]) == 2 * (3 * 4 - 1)
    assert (hf.face_normals(part["positions"], part["faces"])[:, 2] > 0.99).all()
    assert np.allclose(part["positions"][0], [1.5, 3.5, 0.0])          # row 0, column 1: x right, y up


def test_picture_uvs_undo_crop_to_figure():
    frame = (np.array([0.0, 0.1, 0.0]), 0.0007, np.array([384.0, 688.0]))
    pixels = np.array([[0.0, 0.0], [512.0, 300.0], [1023.0, 1023.0]])
    points = hf.crop_to_figure(pixels, (300, 290, 474, 464), 1024, frame)
    uvs = hf.picture_uvs(np.c_[points, np.zeros(3)], (300, 290, 474, 464), 1024, frame)
    assert np.allclose(uvs, pixels / 1024)


def test_picture_material_travels_into_the_model_with_its_picture():
    import io

    from PIL import Image

    from image_to_3dlab.glb_turn import parse_glb

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (40, 230, 255)).save(buffer, format="PNG")
    v, f = box((0, 0, 0), (1, 1, 1))
    writer = hh.GlbWriter()
    writer.doc.update({"meshes": [{"primitives": [writer.primitive({"positions": v, "faces": f})]}],
                       "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0})
    part = {"positions": v, "faces": f, "uvs": np.zeros((8, 2)), "material": hf.picture_material(buffer.getvalue())}
    doc, binary = parse_glb(hh.regraft_glb(writer.build(), [hh.finish_part(part)]))
    material = doc["materials"][0]
    assert material["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0 and len(doc["images"]) == 1
    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    picture = Image.open(io.BytesIO(binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]))
    assert picture.convert("RGB").getpixel((3, 3)) == (40, 230, 255)


def test_pairs_apart_ignores_the_quarter_that_lies_furthest():
    target = np.zeros((10, 3))
    points = np.r_[np.full((6, 3), 0.001), np.full((2, 3), 1.0)]
    assert hf.pairs_apart(points, target) == pytest.approx(0.001 * np.sqrt(3))


def test_front_piece_keeps_an_inner_skin_apart_that_touches_the_outer_at_an_edge():
    outer = sphere(0.05, rings=24, segments=32)
    inner_v = outer[0] * 0.98
    inner_v[0] = outer[0][0]                      # the two skins share the top point ...
    inner_v[1:3] = outer[0][1:3]                  # ... and one edge beside it: four faces use that edge
    v, f = joined(outer, (inner_v, outer[1][:, ::-1]))
    window = disk_window(0.035)
    cut, _ = hh.cut_part_values({"positions": v, "faces": f}, -hf.window_values(window, v))
    keep = hf.front_piece(cut["positions"], cut["faces"])
    middle = cut["positions"][cut["faces"][keep]].mean(axis=1)
    assert (np.linalg.norm(middle, axis=1) > 0.0495).all()      # the outer skin only


def test_tuck_moves_the_skirt_in_full_the_middle_not_at_all_and_forward_at_most_the_cap():
    points = np.zeros((5, 3))
    points[:, 2] = 1.0
    values = np.array([1.0, -0.01, 0.05, -0.01, -0.01])
    target = np.array([0.5, 0.5, 0.5, 2.0, np.nan])
    out = hf.tuck(points, values, target, blend=0.1, cap=0.2, gap=0.01)
    assert out[0, 2] == 1.0                              # the middle of the face is its own
    assert out[1, 2] == pytest.approx(0.49)              # the skirt just behind the figure's surface
    assert 0.49 < out[2, 2] < 1.0                        # in between, eased
    assert out[3, 2] == pytest.approx(1.2)               # forward, at most the cap: a strand's cut end hides the rest
    assert out[4, 2] == 1.0                              # no figure there: stays


def test_reference_depth_takes_the_deeper_of_the_two_where_both_are():
    own = np.array([[0.5, -np.inf, 0.2, -np.inf]])
    new = np.array([[0.4, 0.3, -np.inf, -np.inf]])
    assert hf.reference_depth(own, new).tolist() == [[0.4, 0.3, 0.2, np.inf]]


def test_reference_depth_leaves_out_a_new_face_lying_further_than_dive_behind_the_figure():
    own = np.array([[0.5, 0.5, -np.inf]])
    new = np.array([[0.3, 0.1, 0.0]])                    # 0.2 and 0.4 behind the figure; alone where it has nothing
    assert hf.reference_depth(own, new, dive=0.25).tolist() == [[0.3, 0.5, 0.0]]


def figure_head():
    """The old head: a sphere, a strand standing in front of its face, and a collar piece hidden behind the jaw (in
    the window, deeper than the new face). Skin weights: the front on bone 1, the back on bone 0."""
    head = sphere(0.05)
    strand = box((-0.02, 0.005, 0.052), (-0.012, 0.02, 0.058))
    collar = box((-0.02, -0.03, 0.0), (0.02, -0.022, 0.02))
    v, f = joined(head, strand, collar)
    dense = np.zeros((len(v), 2))
    dense[np.arange(len(v)), (v[:, 2] > 0).astype(int)] = 1.0
    return {"positions": v, "faces": f, "dense": dense, "normals": v / np.linalg.norm(v, axis=1, keepdims=True),
            "uvs": np.zeros((len(v), 2))}, len(head[0]), len(strand[0])


def test_graft_face_takes_the_strand_keeps_the_hidden_collar_and_tucks_the_face_under_the_cut_edge():
    body, _, _ = figure_head()
    new_v, new_f = sphere(0.052, rings=40, segments=56)          # the new face stands 2 mm proud of the old
    face = {"positions": new_v, "faces": new_f, "uvs": np.full((len(new_v), 2), 0.5)}
    window = disk_window(0.035, width=0.1)
    parts, new, report = hf.graft_face([body], window, face)
    kept = parts[0]["positions"]
    assert not ((kept[:, 2] > 0.052) & (kept[:, 0] < -0.011)).any()     # the strand went
    assert ((kept[:, 1] < -0.021) & (kept[:, 2] > 0.019)).any()         # the collar behind the jaw stayed
    assert report["parts"][0]["old_in_front_max"] > 0.004               # the strand stood that far in front
    values = hf.window_values(window, new["positions"])
    radius = np.linalg.norm(new["positions"], axis=1)
    skirt = values < -0.5 * window["pixel"]
    assert skirt.any() and values.min() > -(hf.SKIRT_PIXELS + 1) * window["pixel"]   # a few pixels past the seam
    assert ((radius[skirt] < 0.05) & (radius[skirt] > 0.047)).all()                 # just behind the old surface
    middle = values > hf.BLEND_SHARE * window["width"]
    assert np.allclose(radius[middle], 0.052, atol=2e-4)                            # the middle is the new face's own
    assert report["seam_gap_max"] == pytest.approx(0.0385 - 0.0357 + hf.TUCK_SHARE * 0.1, abs=1e-3)
    assert np.allclose(new["dense"], [0.0, 1.0])                                    # the old face's weights
    assert report["off_cliff_share"] == 0.0                                         # a whole face, no pit


def test_graft_face_leaves_parts_away_from_the_face_alone():
    body, _, _ = figure_head()
    hand_v, hand_f = box((0.3, -0.3, 0.0), (0.4, -0.2, 0.1))
    hand = {"positions": hand_v, "faces": hand_f}
    face = dict(zip(("positions", "faces"), sphere(0.051)))
    parts, _, report = hf.graft_face([hand, body], disk_window(0.035, width=0.1), face)
    assert parts[0] is hand and report["parts"][0] == {"cut": False} and report["body_part"] == 1


def pitted_face(surface, floor, pit=(0.02, -0.01), pit_radius=0.006, half=0.045, pixel=0.001):
    """A new face laid out as a relief over the window, SURFACE(x, y) its depth, with a pit where the build showed no
    face: its relief fell to the build's own back there, to FLOOR, as skin_sorcerer_c's did at both lower cheeks
    (round 27 of the AssetFurnace Support notes)."""
    box = (-half, half, -half, half)
    rows, cols = hf.canvas(box, pixel)
    x, y = np.meshgrid(box[0] + (np.arange(cols) + 0.5) * pixel, box[3] - (np.arange(rows) + 0.5) * pixel)
    depth = surface(x, y)
    depth[(x - pit[0]) ** 2 + (y - pit[1]) ** 2 <= pit_radius ** 2] = floor
    return hf.relief(depth, box, pixel, np.ones(depth.shape, bool))


def diving_face(radius=0.0525, floor=-0.035):
    """A sphere's front 2.5 mm proud of the old face, its pit falling through the head to FLOOR."""
    return pitted_face(lambda x, y: np.sqrt(np.clip(radius ** 2 - x ** 2 - y ** 2, 0.0, None)), floor)


def holes_from_behind(parts, half=0.06, pixel=0.001):
    """Pixels where the parts, seen from straight behind as a game draws them (back faces culled), show something
    but no surface facing the camera: see-through holes."""
    v = np.concatenate([np.asarray(p["positions"], dtype=float) for p in parts])
    offsets = np.cumsum([0] + [len(p["positions"]) for p in parts])
    f = np.concatenate([np.asarray(p["faces"]) + o for p, o in zip(parts, offsets)])
    toward = hf.face_normals(v, f)[:, 2] < 0
    turned = v * np.array([-1.0, 1.0, -1.0])                 # turned round: the back camera's view as a front view
    box = (-half, half, -half, half)
    shown = np.isfinite(hf.depth_map(turned, f, box, pixel)[0])
    drawn = np.isfinite(hf.depth_map(turned, f[toward], box, pixel)[0])
    return int((shown & ~drawn).sum())


def test_graft_face_never_cuts_through_the_head_where_the_new_face_dives_into_it():
    v, f = hollow_head()
    body = {"positions": v, "faces": f}
    window = disk_window(0.035, width=0.1)
    assert holes_from_behind([body]) == 0
    parts, new, report = hf.graft_face([body], window, diving_face())
    kept = parts[0]["positions"][parts[0]["faces"]].mean(axis=1)
    assert (kept[:, 2] < 0).sum() == (v[f].mean(axis=1)[:, 2] < 0).sum()      # nothing of the head's back went
    assert holes_from_behind(parts + [new]) == 0                              # seen from behind, no hole
    beside_pit = np.hypot(kept[:, 0] - 0.02, kept[:, 1] + 0.01) < 0.014       # the pit, its cliff band: kept
    assert not ((kept[:, 2] > 0.03) & (np.hypot(kept[:, 0], kept[:, 1]) < 0.03) & ~beside_pit).any()  # the old face went
    assert 0.03 < report["off_cliff_share"] < 0.2                             # the pit and its rim, said


def filled_over(parts, new, pit, radius):
    """(old faces left over the pit, the new face's points over it, the front depth over it)."""
    kept = parts[0]["positions"][parts[0]["faces"]].mean(axis=1)
    old = (np.hypot(*(kept[:, :2] - pit).T) < radius - 0.002) & (kept[:, 2] > 0.03)
    over = np.hypot(*(new["positions"][:, :2] - pit).T) < radius - 0.002
    front = hf.depth_map(np.concatenate([parts[0]["positions"], new["positions"]]),
                         np.concatenate([parts[0]["faces"], new["faces"] + len(parts[0]["positions"])]),
                         (pit[0] - 0.003, pit[0] + 0.003, pit[1] - 0.003, pit[1] + 0.003), 0.001)[0]
    return int(old.sum()), new["positions"][over], front


def test_graft_face_fills_a_pit_with_the_figures_own_shape_where_the_new_face_is_no_face():
    # skin_sorcerer_c's lower cheeks (rounds 27 and 28 of the AssetFurnace Support notes): where the build showed no
    # face the new face falls into a pit; the old cheek was cut away all the same, and the pit's walls showed from the
    # front as mottled grey, white and maroon patches. Keeping the old cheek there left a crumpled piece of the old jaw
    # inside the new face (round 28, TODO 207). The new face takes the figure's own shape there instead
    v, f = hollow_head()
    body = {"positions": v, "faces": f}
    pit, radius = np.array([0.02, -0.01]), 0.006
    window = disk_window(0.035, width=0.1)
    window["depth"], _ = hf.depth_map(v, f, window["box"], window["pixel"])
    parts, new, report = hf.graft_face([body], window, diving_face())
    old, over, front = filled_over(parts, new, pit, radius)
    assert old == 0                                                           # the old cheek went ...
    skin = np.sqrt(0.05 ** 2 - (over[:, :2] ** 2).sum(axis=1))
    assert len(over) and np.abs(over[:, 2] - (skin - hf.TUCK_SHARE * 0.1)).max() < 0.0025   # ... the face took its shape
    assert np.isfinite(front).all() and front.min() > 0.04                   # from the front: skin, no pit
    assert holes_from_behind(parts + [new]) == 0
    assert 0.03 < report["off_cliff_share"] < 0.2                             # still said
    assert 0.0 < report["face_filled_share"] < 0.2                            # and how much was filled


def plate(depth, half=0.06, pixel=0.002, back=False):
    """A flat sheet at DEPTH over the square of HALF, facing the camera (or away from it with BACK)."""
    box = (-half, half, -half, half)
    rows, cols = hf.canvas(box, pixel)
    sheet = hf.relief(np.full((rows, cols), depth), box, pixel, np.ones((rows, cols), bool))
    if back:
        sheet["faces"] = sheet["faces"][:, ::-1]
    return sheet


def test_graft_face_never_cuts_through_a_shallow_head_where_the_new_face_falls_off_a_cliff():
    # the raw model's side of the head: its front 0.01, its back only 0.03 behind it; the new face stands 0.01 proud
    # and falls 0.03 into a pit, its floor less than DIVE_SHARE of the head's width behind the figure's front
    front, back = plate(0.01), plate(-0.02, back=True)
    v, f = joined((front["positions"], front["faces"]), (back["positions"], back["faces"]))
    body = {"positions": v, "faces": f}
    face = pitted_face(lambda x, y: np.full(x.shape, 0.02), floor=-0.01)
    assert 0.03 - 0.01 < hf.DIVE_SHARE * 0.1                     # the depth limit alone lets this pit count
    parts, new, _ = hf.graft_face([body], disk_window(0.035, width=0.1), face)
    kept = parts[0]["positions"][parts[0]["faces"]].mean(axis=1)
    assert (kept[:, 2] < 0).sum() == (v[f].mean(axis=1)[:, 2] < 0).sum()      # the back plate is whole
    assert holes_from_behind(parts + [new], half=0.05) == 0


def test_graft_face_fills_a_pit_well_inside_the_face():
    # skin_sorcerer_c's jaw patches (round 28 of the AssetFurnace Support notes, TODO 207): its two pits lay inside the
    # window without touching its edge; the old front was cut away there, the new face's pit lay 3 to 8 cm behind it,
    # and the front showed the old head's inside as grey, white and maroon patches
    v, f = hollow_head()
    body = {"positions": v, "faces": f}
    pit, radius = np.array([0.008, -0.01]), 0.006
    face = pitted_face(lambda x, y: np.sqrt(np.clip(0.0525 ** 2 - x ** 2 - y ** 2, 0.0, None)), -0.035, pit=pit)
    window = disk_window(0.035, width=0.1)
    window["depth"], _ = hf.depth_map(v, f, window["box"], window["pixel"])
    assert np.hypot(*pit) + radius + 2 * hf.CLIFF_PIXELS * window["pixel"] < 0.035 - 0.004   # an island, off the edge
    parts, new, report = hf.graft_face([body], window, face)
    old, over, front = filled_over(parts, new, pit, radius)
    skin = np.sqrt(0.05 ** 2 - (over[:, :2] ** 2).sum(axis=1))
    assert old == 0 and len(over) and np.abs(over[:, 2] - (skin - hf.TUCK_SHARE * 0.1)).max() < 0.0015
    assert np.isfinite(front).all() and front.min() > 0.04                   # from the front: skin, no pit
    assert holes_from_behind(parts + [new]) == 0
    assert 0.0 < report["face_filled_share"] < 0.2


def test_seam_surface_drops_a_strand_and_keeps_a_cheek():
    rows, cols = np.mgrid[0:60, 0:60]
    dome = 0.05 - ((cols - 30.0) ** 2 + (rows - 30.0) ** 2) * 1e-5      # a cheek
    own = dome.copy()
    own[:, 27:30] += 0.012                                               # a strand 3 pixels wide in front of it
    own[:5] = -np.inf                                                    # nothing there
    out = hf.seam_surface(own, across=6, least=0.001)
    assert np.array_equal(out[5:, :24], own[5:, :24])                   # the cheek as it was
    assert np.allclose(out[10:50, 27:30], dome[10:50, 27:30], atol=5e-4)  # under the strand: the cheek beside it
    assert np.isneginf(out[:5]).all() and np.isfinite(out[5:]).all()
    brow = np.broadcast_to(0.03 + 0.0016 * np.arange(60.0)[:, None], (60, 60)).copy()    # a brow sloping up the head
    slope = brow.copy()
    slope[:, 27:31] += 0.012                                             # with a strand 4 pixels wide down it
    assert np.allclose(hf.seam_surface(slope, across=5, least=0.001)[10:50, 27:31], brow[10:50, 27:31],
                       atol=1e-5)                                        # the brow at its own height, row by row
    assert np.array_equal(hf.seam_surface(own, across=1), own)          # too narrow to tell: as it was


def test_seam_target_carries_the_figure_in_along_its_slope_and_ignores_what_lies_inside():
    # read inside the seam, the target took what the new face replaces: on skin_sorcerer_c a hood's edge 2 cm in front
    # of the jaw pulled one point of the face up into a light spike (round 28 of the AssetFurnace Support notes)
    window = disk_window(0.03, width=0.1)
    rows, cols = window["field"].shape
    x = window["box"][0] + (np.arange(cols) + 0.5) * window["pixel"]
    plane = np.broadcast_to(0.02 + 0.3 * x[None, :], (rows, cols)).copy()   # a cheek sloping up to the right
    own = plane.copy()
    own[window["field"] > 0.005] = 0.06                                     # inside: a hood's edge far in front
    out = hf.seam_target(own, window["field"], reach=10, pixel=window["pixel"])
    near = (window["field"] > 0) & (window["field"] < 8 * window["pixel"])
    assert np.allclose(out[near], plane[near], atol=2e-3)                    # along its slope, not up to the hood
    assert np.array_equal(out[window["field"] <= 0], own[window["field"] <= 0])   # outside: as it is
    assert np.isneginf(out[window["field"] > 12 * window["pixel"]]).all()    # past the reach: nothing


def test_rise_target_keeps_the_figures_front_behind_the_face_and_limits_what_stands_in_front():
    # the Nachtschwinge v19h: under its chin the figure's front without strands (`seam_surface`) took the old chin for a
    # strand and dropped to the neck 7 cm behind it, the cap carried in from the seam read the neck too, and the new chin
    # was pulled back by up to 3.4 cm: its coarse triangles tilted away and drew a dark grey patch (round 28)
    own = np.array([0.0558, 0.06, 0.047, 0.073, 0.05, -np.inf])    # chin, cheek, strand, thin hood edge, brow, none
    surface = np.array([0.0268, 0.06, 0.040, 0.045, 0.05, -np.inf])  # without strands (the chin wrongly taken for one)
    carried = np.array([0.0306, 0.059, 0.041, 0.047, 0.05, -np.inf])  # carried in from the seam
    new = np.array([0.061, 0.045, 0.040, 0.054, 0.049, 0.03])        # the new face's own depth
    out = hf.rise_target(own, surface, carried, new, over=0.003)
    assert out[0] == 0.0558                       # the figure behind the face: its own front, never the neck behind it
    assert out[1] == pytest.approx(0.06)          # a cheek in front of the face: the face rises all the way to it
    assert out[2] == pytest.approx(0.040)         # a strand in front: no rise past the surface beside it
    assert out[3] == pytest.approx(0.045)         # a thin hood's edge in front: the face sinks under it, beside it
    assert out[4] == pytest.approx(0.05)          # the brow just in front: as it is
    assert np.isneginf(out[5])                    # no figure there: nothing


def strip(y, z, x0=0.0, x1=-0.03, n=31):
    """A strip of surface beside the seam at height Y and depth Z, x from X0 (the seam) outward to X1: (points, faces);
    its first two points (x = X0) make its cut end."""
    x = np.linspace(x0, x1, n)
    points = np.stack([np.r_[x, x], np.r_[np.full(n, y), np.full(n, y + 0.001)], np.full(2 * n, z)], axis=1)
    faces = np.concatenate([[[i, i + 1, n + i], [i + 1, n + 1 + i, n + i]] for i in range(n - 1)])
    return points, faces


def test_ease_onto_lays_a_cut_end_standing_over_the_face_down_and_fades_out_over_the_band():
    # a strip of the figure's surface beside the seam whose cut end stands 0.009 over the new face (the face at 0.0
    # under it, a gap of 0.001); a second strip whose cut end lies just over the face already; a lone triangle not
    # joined to any cut end
    over, over_f = strip(0.0, 0.01)
    flush, flush_f = strip(0.05, 0.001)
    lone = np.array([[-0.005, 0.01, 0.02], [-0.006, 0.01, 0.02], [-0.005, 0.011, 0.02]])
    points, faces = joined((over, over_f), (flush, flush_f), (lone, [[0, 1, 2]]))
    rim = np.array([[0, 31], [62, 93]])
    under = np.zeros(len(points))
    moved, by = hf.ease_onto(points, faces, rim, points[:, 0].copy(), under, band=0.02, gap=0.001)
    assert moved[0, 2] == pytest.approx(0.001) and moved[31, 2] == pytest.approx(0.001)   # the cut end: just over it
    assert moved[25, 2] == pytest.approx(0.01) and by[25] == 0.0  # past the band (x -0.025): as it was
    assert 0.001 < moved[10, 2] < 0.01                              # in between, eased
    assert np.all(np.diff(moved[:21, 2]) >= -1e-12)                 # rising smoothly outward: no notch
    assert np.allclose(moved[62:124], flush)                        # a cut end on the face already: stays
    assert np.allclose(moved[124:], lone)                           # not joined to a cut end: stays
    assert np.allclose(moved[:, :2], points[:, :2])                 # in depth only


def inside_seen(parts, direction, half=0.06, pixel=0.001):
    """(see-through pixels, pixels showing the hollow head's inside) of PARTS seen from DIRECTION as a game draws them
    (back faces culled): rays through a gap at the seam meet the inner wall (marked by u = 1 in its uvs) or nothing."""
    v = np.concatenate([np.asarray(p["positions"], dtype=float) for p in parts])
    offsets = np.cumsum([0] + [len(p["positions"]) for p in parts])
    f = np.concatenate([np.asarray(p["faces"]) + o for p, o in zip(parts, offsets)])
    u = np.concatenate([np.asarray(p["uvs"], dtype=float)[:, 0] if p.get("uvs") is not None and k == 0
                        else np.zeros(len(p["positions"])) for k, p in enumerate(parts)])
    d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    right = np.cross([0.0, 1.0, 0.0], d)
    right /= np.linalg.norm(right)
    turned = v @ np.stack([right, np.cross(d, right), d], axis=1)
    box = (-half, half, -half, half)
    toward = hf.face_normals(turned, f)[:, 2] > 0
    shown = np.isfinite(hf.depth_map(turned, f, box, pixel)[0])
    depth, owner = hf.depth_map(turned, f[toward], box, pixel)
    drawn = np.isfinite(depth)
    wall = np.zeros(shown.shape, bool)
    wall[drawn] = u[f[toward][owner[drawn]]].mean(axis=1) > 0.5
    return int((shown & ~drawn).sum()), int(wall.sum())


def deep_face(deeper=0.015, half=0.045, pixel=0.001):
    """A new face lying DEEPER behind the old head's front all over, as the Nachtschwinge's did beside its cheek: the
    figure's cut edge stood up to 2.5 cm in front of it at the seam (round 28 of the AssetFurnace Support notes)."""
    return pitted_face(lambda x, y: np.sqrt(np.clip(0.05 ** 2 - x ** 2 - y ** 2, 0.0, None)) - deeper, 0.0,
                       pit_radius=0.0, half=half, pixel=pixel)


def test_graft_face_meets_the_figure_where_the_face_lies_deeper_and_leaves_no_gap():
    # where the figure stood further in front of the new face at the seam than the face could rise (0.05 of the head's
    # width then), its cut edge hung over the face: a light line down the Nachtschwinge's cheek, and from the side and
    # below a gap into the head. The face rises to the figure's surface at the seam
    v, f = hollow_head()
    uvs = np.zeros((len(v), 2))
    uvs[len(v) // 2:, 0] = 1.0                                       # the inner wall, to tell it in a picture
    body = {"positions": v, "faces": f, "uvs": uvs}
    window = disk_window(0.035, width=0.1)
    parts, new, report = hf.graft_face([body], window, deep_face())
    gap = hf.TUCK_SHARE * 0.1
    cut = parts[0]
    edges = np.sort(np.concatenate([cut["faces"][:, [0, 1]], cut["faces"][:, [1, 2]], cut["faces"][:, [2, 0]]]), axis=1)
    found, counts = np.unique(edges, axis=0, return_counts=True)
    rim = np.unique(found[counts == 1])
    rim = rim[np.abs(hf.window_values(window, cut["positions"][rim])) < 2 * window["pixel"]]
    face_depth, _ = hf.depth_map(new["positions"], new["faces"], window["box"], window["pixel"] / 2)
    under = hf.sample(face_depth, window["box"], window["pixel"] / 2, cut["positions"][rim], outside=np.nan)
    over = cut["positions"][rim, 2] - under
    assert len(rim) > 50 and np.isfinite(over).mean() > 0.95                 # the face lies under the whole rim
    assert np.nanmax(over) < 2 * gap                                          # the cut edge: just in front of it
    assert report["seam_rise_max"] > 0.015 - 2 * gap                          # the face rose to the figure ...
    assert report["seam_eased_max"] < gap                                     # ... so nothing had to be eased
    away = hf.window_values(window, cut["positions"]) < -1.5 * window["pixel"]   # past the cut's own new points
    radius = np.linalg.norm(cut["positions"][away], axis=1)
    assert np.all((np.abs(radius - 0.05) < 1e-9) | (np.abs(radius - 0.048) < 1e-9))   # the figure: as it was
    for direction in ((1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (0.0, -1.0, 1.0), (0.0, 1.0, 1.0), (0.6, -0.6, 0.5),
                      (2.0, 0.0, 1.0), (0.0, -2.0, 1.0)):
        through, wall = inside_seen([cut, new], direction)
        # no gap into the head from any side (the unchanged code: 421 to 593 such pixels a view); what is left is a
        # sliver about one gap (1 mm) wide along the silhouette where the seam runs right on it, rays grazing the head
        assert wall <= 10 and through + wall <= 0.25 * 2 * np.pi * 0.035 / 0.001, (direction, through, wall)


def test_graft_face_does_not_rise_to_a_strand_and_lays_only_a_low_cut_end_down_onto_the_face():
    # at the Zauberer's brow a strand stood 0.012 in front of the face at the seam, and raising the face to it made a
    # ramp across the brow (round 22). The face goes on into the skin beside a strand; a strand standing far over the
    # face stays as it is (laid down 2.8 cm, the Nachtschwinge's hair locks scratched its cheeks dark, round 28), while
    # a cut end standing a little over the face eases down onto it
    head = sphere(0.05)
    strand = box((-0.003, 0.02, 0.03), (0.003, 0.065, 0.047))   # 6 mm wide, 1 cm in front at the seam at the brow
    low = box((0.017, 0.015, 0.03), (0.023, 0.065, 0.039))      # 3 mm over the face where it crosses the seam
    v, f = joined(head, strand, low)
    body = {"positions": v, "faces": f}
    window = disk_window(0.035, width=0.1)
    face = pitted_face(lambda x, y: np.sqrt(np.clip(0.0505 ** 2 - x ** 2 - y ** 2, 0.0, None)), 0.0, pit_radius=0.0)
    parts, new, report = hf.graft_face([body], window, face)
    gap = hf.TUCK_SHARE * 0.1
    skin = np.sqrt(0.05 ** 2 - 0.034 ** 2)                       # the skin at the seam under the strand
    face_depth, _ = hf.depth_map(new["positions"], new["faces"], window["box"], window["pixel"] / 2)
    at_seam = hf.sample(face_depth, window["box"], window["pixel"] / 2, np.array([[0.0, 0.034, 0.0]]))[0]
    assert at_seam < skin + 2 * gap                               # the face did not rise to the strand (0.047)
    p = parts[0]["positions"]
    seam = np.abs(hf.window_values(window, p)) < window["pixel"]
    tall = (np.abs(p[:, 0]) <= 0.0031) & seam & (p[:, 2] > 0.046)
    assert tall.sum() >= 2                                        # the tall strand's cut end: as it was
    end = (np.abs(p[:, 0] - 0.02) <= 0.0031) & seam & (p[:, 2] > 0.03)
    assert end.sum() >= 4                                         # the low strand's cut end ...
    under = hf.sample(face_depth, window["box"], window["pixel"] / 2, p[end], outside=np.nan)
    assert np.nanmax(p[end, 2] - under) < 3 * gap                 # ... lies on the face, no edge over it
    assert 0.0 < report["seam_eased_max"] < hf.STRAND_EASE_SHARE * 0.1


def test_reached_takes_the_face_from_the_nose_and_leaves_a_pit_behind_a_cliff():
    depth = np.full((40, 40), 0.05)
    depth[:, 20:] -= np.linspace(0.0, 0.01, 20)[None, :]        # a gentle slope: still the face
    depth[5:12, 5:12] = -0.03                                    # a pit 0.08 deep
    depth[30:, :5] = -np.inf                                      # nothing there
    out = hf.reached(depth, (20, 20), 0.025)
    assert out[20, 35] and out[20, 2] and out[35, 6]
    assert not out[8, 8] and not out[35, 2]
    assert not out[8, 13] and out[8, 16]                          # the cliff's own band stays out, not more
    assert hf.reached(depth, (8, 12), 0.025).sum() == np.isfinite(depth).sum()   # a seed on a cliff: all of it


def test_graft_face_says_so_when_the_new_face_misses_the_window():
    body, _, _ = figure_head()
    far = dict(zip(("positions", "faces"), sphere(0.05, centre=(1.0, 0.0, 0.0))))
    with pytest.raises(ValueError, match="does not reach"):
        hf.graft_face([body], disk_window(0.035, width=0.1), far)


# --- GLB in ---------------------------------------------------------------------------------------------------------

def test_read_parts_reads_every_part_of_one_mesh():
    from image_to_3dlab.glb_turn import parse_glb

    writer = hh.GlbWriter()
    a_v, a_f = box((0, 0, 0), (1, 1, 1))
    b_v, b_f = box((2, 0, 0), (3, 1, 1))
    primitives = [writer.primitive({"positions": a_v, "faces": a_f}),
                  writer.primitive({"positions": b_v, "faces": b_f})]
    writer.doc.update({"meshes": [{"primitives": primitives}], "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}],
                       "scene": 0})
    doc, _, parts = hf.read_parts(writer.build())
    assert len(parts) == 2 and np.allclose(parts[1]["positions"], b_v) and (parts[0]["faces"] == a_f).all()
    assert parse_glb(writer.build())[0]["meshes"][0]["primitives"] == doc["meshes"][0]["primitives"]
    writer.doc["nodes"][0]["translation"] = [0, 1, 0]
    with pytest.raises(ValueError, match="moved by its node"):
        hf.read_parts(writer.build())


def test_auto_builds_the_whole_head_under_a_brim_and_the_face_alone_without_one():
    assert hf.pick_region("auto", {"floor": 1.62})[0] == "head"
    assert hf.pick_region("auto", {"floor": None})[0] == "face"
    assert hf.pick_region("face", {"floor": 1.62})[0] == "face"
    assert hf.pick_region("head", {"floor": None})[0] == "head"


def strand_step(half=0.04, step=0.001, rise=0.01):
    """A build whose front is flat skin with a strand over its upper left: z = 0.05, RISE more above the line
    y = 0.5 x, the edge crossing the build at a slant as a strand of hair crosses a face."""
    xs = np.arange(-half, half + step / 2, step)
    x, y = np.meshgrid(xs, xs)
    v = np.stack([x.ravel(), y.ravel(), 0.05 + rise * (y.ravel() > 0.5 * x.ravel())], axis=1)
    n = len(xs)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    f = np.concatenate([np.stack([c, d, b], -1).reshape(-1, 3), np.stack([c, b, a], -1).reshape(-1, 3)])
    return v, f


def step_widths(part, rise=0.01, within=0.025):
    """How far across the strand's edge (the line y = 0.5 x) each triangle reaches that climbs it (spans most of
    RISE), WITHIN that radius: the width of the steps the edge is drawn in, one grid square on a grid."""
    tri = np.asarray(part["positions"])[np.asarray(part["faces"])]
    middle = tri.mean(axis=1)
    climbs = (np.ptp(tri[:, :, 2], axis=1) > 0.5 * rise) & (np.hypot(middle[:, 0], middle[:, 1]) < within)
    return np.ptp((tri[climbs, :, 1] - 0.5 * tri[climbs, :, 0]) / np.sqrt(1.25), axis=1)


def test_a_face_made_lighter_keeps_its_outline_and_the_strands_edge_fine_and_thins_the_flat_skin():
    # skin_sorcerer_c's rigged model: its new face had 4,705 triangles where the head it replaced had about 2,026, the
    # game step's head protect kept them and thinned the body about 30 % harder (round 27), so the face takes the rule
    # of the boots (hero_parts.matched_faces, round 28). Laid out on a coarser grid for it, every edge across the face
    # became a staircase of the grid's squares: square notches round the Nachtschwinge's eyes, brows and nose root,
    # gone in a rerun with the fine layout (round 30 of the AssetFurnace Support notes). Made lighter instead, the
    # flat skin gives up its triangles and the outline and the strand's edge keep the fine layout's points.
    v, f = strand_step(half=0.035, step=0.0002)
    window = disk_window(0.03, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    pixel = window["pixel"]
    fine = hf.face_relief({"positions": v, "faces": f}, window, CROP, 1024, FRAME, rounding=0.0)
    light = hf.face_relief({"positions": v, "faces": f}, window, CROP, 1024, FRAME, rounding=0.0, keep=0.1)
    coarse = hf.face_relief({"positions": v, "faces": f}, window, CROP, 1024, FRAME, rounding=0.0, step=3 * pixel)
    assert len(light["faces"]) <= 0.11 * len(fine["faces"])
    assert len(light["faces"]) < len(coarse["faces"])                          # lighter than the coarse grid, and
    assert np.median(step_widths(coarse)) > 2.5 * pixel                        # where the grid draws the edge in
    assert np.median(step_widths(light)) < 0.7 * np.median(step_widths(coarse))  # steps of its squares, the light
    assert np.median(step_widths(light)) < 2 * pixel                           # face keeps it nearly as fine
    tri = light["positions"][light["faces"]]
    middle = tri.mean(axis=1)
    flat = (np.hypot(middle[:, 0], middle[:, 1]) < 0.02) & (np.abs(middle[:, 1] - 0.5 * middle[:, 0]) > 0.006)
    area = 0.5 * np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2])       # seen from the front
    assert np.median(area[flat]) > 4 * 0.5 * pixel ** 2                        # the flat skin in large triangles
    for part in (light, fine):                                                  # the skirt under the cut edge whole
        new, _ = hh.cut_part_values(part, -(hf.window_values(window, part["positions"]) + hf.SKIRT_PIXELS * pixel))
        t = new["positions"][new["faces"]]
        seen = 0.5 * np.abs(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2]).sum()
        assert seen == pytest.approx(np.pi * (0.03 + hf.SKIRT_PIXELS * pixel) ** 2, rel=0.03)
    assert np.allclose(np.sort(light["positions"][:, 2])[[0, -1]], [0.05, 0.06], atol=1e-4)   # on the build's planes
    assert light["uvs"].shape == (len(light["positions"]), 2)
    assert hf.face_relief({"positions": v, "faces": f}, window, CROP, 1024, FRAME, keep=1.0)["faces"].shape == \
        hf.face_relief({"positions": v, "faces": f}, window, CROP, 1024, FRAME)["faces"].shape   # nothing to take


def rim_over(cut, new, window):
    """How far each point of the figure's cut edge (CUT's open edges at the seam) stands in front of the new face."""
    edges = np.sort(np.concatenate([cut["faces"][:, [0, 1]], cut["faces"][:, [1, 2]], cut["faces"][:, [2, 0]]]), axis=1)
    found, counts = np.unique(edges, axis=0, return_counts=True)
    rim = np.unique(found[counts == 1])
    rim = rim[np.abs(hf.window_values(window, cut["positions"][rim])) < 2 * window["pixel"]]
    face_depth, _ = hf.depth_map(new["positions"], new["faces"], window["box"], window["pixel"] / 2)
    under = hf.sample(face_depth, window["box"], window["pixel"] / 2, cut["positions"][rim], outside=np.nan)
    return cut["positions"][rim, 2] - under


def test_a_face_made_lighter_after_it_is_set_in_still_meets_the_figure_all_along_its_seam():
    # made lighter before it was cut and tucked (round 30), the Nachtschwinge's face crossed its seam in large flat
    # triangles: they could follow neither the seam's line nor the tuck up to the figure, the old cut edge hung over
    # the face along 17.5 % of the seam at the face stage (10.3 % with the fine face) and a wedge of the face's hair
    # paint stood out beside the eye (round 31 of the AssetFurnace Support notes). Set in fine and made lighter after,
    # where its final shape is flat, the face keeps the seam as close as the fine one
    v, f = hollow_head()
    uvs = np.zeros((len(v), 2))
    uvs[len(v) // 2:, 0] = 1.0
    body = {"positions": v, "faces": f, "uvs": uvs}
    window = disk_window(0.035, width=0.1)
    gap = hf.TUCK_SHARE * 0.1
    built = deep_face()
    built["uvs"] = hf.picture_uvs(built["positions"], CROP, 1024, FRAME)
    fine_parts, fine, _ = hf.graft_face([body], window, built)
    parts, light, report = hf.graft_face([body], window, built, keep=0.1)
    # lighter, and no lighter than its light allows (`lighter_without_facets`)
    assert len(light["faces"]) < 0.5 * len(fine["faces"]) and report["face_faces"] == len(light["faces"])
    assert report["face_facets"] <= hf.FACET_LIMIT
    over = rim_over(parts[0], light, window)
    assert np.isfinite(over).mean() > 0.95 and np.nanmax(over) < 2 * gap        # as the fine face: just in front
    assert np.nanmax(rim_over(fine_parts[0], fine, window)) < 2 * gap
    early_parts, early, _ = hf.graft_face([body], window, hf.lighter(built, 0.1))
    assert np.nanmax(rim_over(early_parts[0], early, window)) > 2 * gap         # made lighter first: it hangs over
    # its paint still where the picture shows it: the texture coordinates follow the points it kept
    assert np.allclose(light["uvs"], hf.picture_uvs(light["positions"], CROP, 1024, FRAME), atol=1e-9)
    normals = hf.face_normals(light["positions"], light["faces"])
    assert (normals[:, 2] > 0).mean() > 0.99                                     # facing the camera, none flipped


def test_the_faces_paint_takes_the_figures_skin_where_the_picture_shows_hair_just_inside_the_seam():
    # the picture's hair lock reached a few mm further than the figure's: just inside the seam at the Nachtschwinge's
    # right jaw corner and temple the new face, laid onto the old cheek's shape there, carried the lock's paint, black
    # notches from the front and a black wedge from the side (round 31 of the AssetFurnace Support notes)
    import io

    from PIL import Image

    window = disk_window(0.02, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    skin, hair = (200, 160, 140), (20, 40, 30)
    figure = two_tone_plane(skin, hair, 0.017)                # its own hair only at the right edge, past x = 0.017
    crop = np.empty((1024, 1024, 3))
    crop[:] = skin
    crop[:, 404:451] = hair                                   # the picture's locks: x -0.021 .. -0.012 ...
    crop[:, 573:620] = hair                                   # ... and 0.012 .. 0.021
    matched, _ = hf.match_face_paint({"positions": np.zeros((3, 3)), "faces": np.array([[0, 1, 2]])}, [figure],
                                     window, crop, CROP, FRAME)
    doc, binary, _ = matched["material"]
    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    start = view.get("byteOffset", 0)
    picture = np.asarray(Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGB"), float)
    assert np.allclose(picture[512, 412], skin, atol=12)      # 0.5 mm inside the seam on the figure's skin: skin
    assert np.allclose(picture[512, 405], skin, atol=12)      # and on the skirt under the cut edge, which a side shows
    assert picture[512, 448].max() < 60                       # 7.5 mm inside: the picture's own lock stays
    assert picture[512, 612].max() < 60                       # where the figure has hair there too, the lock stays
    assert np.allclose(picture[300, 512], skin, atol=2)       # the skin round it as it was


# --- the seam along the old cheek's edge (round 29) ----------------------------------------------------------------

def texture_material(texture):
    """A material painted by TEXTURE (an RGB array)."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(texture, np.uint8), "RGB").save(buffer, format="PNG")
    return hf.picture_material(buffer.getvalue())


def two_tone_plane(skin, hair, hair_from_x, half=0.05, step=0.002):
    """`painted_plane` painted SKIN, and HAIR where x is past HAIR_FROM_X."""
    plane = painted_plane(skin, half, step)
    texture = np.empty((64, 64, 3), np.uint8)
    texture[:] = skin
    texture[:, int(np.ceil((hair_from_x + half) / (2 * half) * 64)):] = hair
    return {**plane, "material": texture_material(texture)}


def test_seam_paint_leaves_out_pairs_that_are_not_one_surface():
    # with the seam along the old cheek's edge the band just outside it runs over hair locks: on the Nachtschwinge a
    # third of its pairs differed by more than 45 levels (hair against the crop's skin, up to 196), and clipped to 60
    # they lit the jaw corners up (round 29 of the AssetFurnace Support notes)
    window = disk_window(0.02, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    crop = np.full((1024, 1024, 3), 100.0)
    figure = two_tone_plane((130, 120, 110), (20, 40, 30), 0.012)
    points, offsets = hf.seam_paint([figure], window, crop, CROP, FRAME)
    assert len(points) > 100 and np.allclose(offsets, [30, 20, 10])       # the skin's side of the seam is read ...
    assert (points[:, 0] > 0.013).sum() == 0                                # ... the hair's side left out
    matched, median = hf.match_face_paint({"positions": np.zeros((3, 3)), "faces": np.array([[0, 1, 2]])}, [figure],
                                          window, crop, CROP, FRAME)
    assert median == pytest.approx([30.0, 20.0, 10.0])
    doc, binary, _ = matched["material"]
    import io

    from PIL import Image

    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    start = view.get("byteOffset", 0)
    picture = np.asarray(Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGB"))
    assert np.allclose(picture[500, 500], [130, 120, 110], atol=2)         # the face's middle: the skin's tone
    assert np.allclose(picture[500, 515], [130, 120, 110], atol=2)         # next to the hair too


def painted_head(hair_from_x=None):
    """A sphere head (0.05 across half) and its part, painted skin all over, or hair past HAIR_FROM_X (a lock hanging
    beside the cheek on the figure's left), UVs straight from the front."""
    v, f = sphere(0.05, rings=40, segments=60)
    uvs = np.stack([(v[:, 0] + 0.05) / 0.1, (0.05 - v[:, 1]) / 0.1], axis=1)
    texture = np.empty((64, 64, 3), np.uint8)
    texture[:] = (220, 180, 150)
    if hair_from_x is not None:
        texture[:, int(np.ceil((hair_from_x + 0.05) / 0.1 * 64)):] = (25, 45, 35)
    return v, f, {"positions": v, "faces": f, "uvs": uvs, "material": texture_material(texture)}


def test_face_window_runs_the_seam_out_to_the_old_cheeks_edge_over_skin_and_never_over_hair():
    # the Nachtschwinge's hard lines down both cheeks were the seam itself: it ran 0.06 of the head's width inside the
    # face's reach, and the strip of old cheek it left (one flat facet of the coarse head, its own paint) showed
    # between it and the side hair lock. Run out to the reach's edge over the figure's own skin, the lines went; run out
    # over the hair lock too, the cut took the lock (see-through from the side 12 -> 939 px) (round 29)
    v, f, part = painted_head(hair_from_x=0.04)
    head = np.ones(len(v), bool)
    old = hf.face_window(v, f, head)
    new = hf.face_window(v, f, head, part=part)
    pixel, box = new["pixel"], new["box"]
    x = box[0] + (np.arange(new["mask"].shape[1]) + 0.5) * pixel
    assert (new["mask"] | old["mask"] == new["mask"]).all()                  # it only grows
    grown = new["mask"] & ~old["mask"]
    rows = np.nonzero(old["mask"][:, :old["seed"][1]].any(axis=1))[0]
    left = lambda mask: x[np.argmax(mask, axis=1)]
    middle = rows[len(rows) // 2]
    assert left(new["mask"])[middle] < left(old["mask"])[middle] - 0.03 * new["width"]    # out on the skin's side
    assert grown.any()                                                      # never over the lock's paint, nor
    assert x[np.nonzero(grown)[1]].max() < 0.04 - hf.HAIR_MARGIN_SHARE * new["width"] + pixel   # near it
    bottom = lambda mask: np.nonzero(mask[:, new["seed"][1]])[0].max()
    assert abs(bottom(new["mask"]) - bottom(old["mask"])) <= 1              # the chin keeps its margin
    assert hf.window_bounds(new) == hf.window_bounds(old)                   # the crop and the fit stay where they were
    bare = {k: val for k, val in part.items() if k != "material"}
    assert (hf.face_window(v, f, head, part=bare)["mask"] == old["mask"]).all()   # unpainted: as it was


def test_the_face_rises_fully_to_the_hair_beside_it_and_not_to_a_strand_across_it():
    # skin_sorcerer_c's window reached over the hair hanging beside its face; cut away there, the hair gave way to the
    # new face's relief, which the strand rule held up to 2.7 cm under it, and from three quarters one looked over the
    # face into the hood's inside (433 and 356 see-through pixels). Risen to the hair's own depth there: 5 and 0 (round 29)
    own = np.full((20, 20), 0.02)
    new = np.zeros((20, 20))
    held = hf.rise_target(own, np.zeros((20, 20)), np.zeros((20, 20)), new, 0.003)
    assert np.allclose(held, 0.0)                                           # a strand across the face: no rise to it
    beside = np.zeros((20, 20), bool)
    beside[:, 15:] = True
    out = hf.rise_target(own, np.zeros((20, 20)), np.zeros((20, 20)), new, 0.003, beside=beside)
    assert np.allclose(out[:, 15:], 0.02) and np.allclose(out[:, :10], 0.0)   # beside the face: to the hair, in full
    v, f, part = painted_head(hair_from_x=0.02)                              # hair over the window's right side
    window = hf.face_window(v, f, np.ones(len(v), bool), part=part)
    x = window["box"][0] + (np.arange(window["mask"].shape[1]) + 0.5) * window["pixel"]
    reach = hf.BESIDE_REACH_SHARE * window["width"]
    _, cols = np.nonzero(window["beside"])
    assert len(cols) and x[cols].min() > 0.02 - reach - window["pixel"]       # the hair beside the face, nothing else
    assert window["beside"][window["mask"] & (x[None, :] > 0.021)].mean() > 0.9


def test_the_seam_keeps_off_the_heads_own_edge_where_the_figure_falls_away_behind_it():
    # a seam on the jaw's edge over the neck carried the neck's depth in: the Nachtschwinge's face lay up to 1.4 cm
    # under the jaw's cut edge, the jaw's facets were eased back onto it, and the face showed through the dent from the
    # side as a pale patch with a hard edge (round 31 of the AssetFurnace Support notes)
    window = disk_window(0.029, width=0.1, pixel=0.001)          # the seam one pixel inside the jaw's edge all round
    rows, cols = window["mask"].shape
    x = window["box"][0] + (np.arange(cols) + 0.5) * window["pixel"]
    y = window["box"][3] - (np.arange(rows) + 0.5) * window["pixel"]
    r = np.hypot(x[None, :], y[:, None])
    lower = np.broadcast_to(y[:, None] < -0.005, (rows, cols))
    depth = np.where(r <= 0.03, 0.02, 0.02 - 0.3 * (r - 0.03))   # above: the surface runs on, gently turning away
    depth[lower & (r > 0.03)] = 0.0                               # below: the neck 2 cm behind the jaw's edge
    reach = hf.SKIRT_PIXELS + hf.NEAR_PIXELS
    out = hf.keep_off_steps({**window, "depth": depth})
    assert out["step_cut_share"] > 0
    assert not (out["mask"] & lower & (0.03 - r < reach * window["pixel"])).any()   # kept off the jaw's edge
    upper = np.broadcast_to(y[:, None] > 0.01, (rows, cols))
    assert np.array_equal(out["mask"] & upper, window["mask"] & upper)              # no step above: as it was
    assert np.array_equal(out["field"], hf.signed_distance(out["mask"], window["pixel"]))
    for outside in (0.04, 0.016):        # a lock or a hood in front of the face's edge, or a small step: as it was
        same = hf.keep_off_steps({**window, "depth": np.where(r <= 0.03, 0.02, outside)})
        assert same["step_cut_share"] == 0.0 and np.array_equal(same["mask"], window["mask"])
    strand = np.where(r <= 0.03, 0.02, 0.02 - 0.3 * (r - 0.03))
    strand[lower & (r >= 0.025) & (r <= 0.0285)] = 0.04            # a strand 2 cm in front, just inside the seam
    same = hf.keep_off_steps({**window, "depth": strand})
    assert same["step_cut_share"] == 0.0 and np.array_equal(same["mask"], window["mask"])


def outline_points(part):
    """PART's points on its outline (edges of one triangle only) and those edges."""
    f = np.asarray(part["faces"])
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    found, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(found[counts == 1]), found[counts == 1]


def test_a_face_made_lighter_spends_its_triangles_inside_and_keeps_its_outline_within_the_tolerance():
    # the Nachtschwinge's face as set in: 392 of its points lay on its outline, the skirt hidden under the figure's cut
    # edge, and the decimation kept every one, so 523 of its 1,500 triangles touched that hidden edge and its cheeks and
    # nose came out in large flat facets (round 31 of the AssetFurnace Support notes). Thinned along its straight runs
    # first, the outline stays within the tolerance of its line and gives its triangles to the face.
    v, f = strand_step(half=0.035, step=0.0002)
    window = disk_window(0.03, width=0.1, pixel=0.1 / hf.PIXELS_PER_HEAD)
    pixel = window["pixel"]
    relief = hf.face_relief({"positions": v, "faces": f}, window, CROP, 1024, FRAME, rounding=0.0)
    fine, _ = hh.cut_part_values(relief, -(hf.window_values(window, relief["positions"]) + hf.SKIRT_PIXELS * pixel))
    plain = hf.lighter(fine, 0.1)
    thin = hf.lighter(fine, 0.1, border=pixel)
    assert len(thin["faces"]) == pytest.approx(len(plain["faces"]), rel=0.02)
    old, _ = outline_points(fine)
    kept, edges = outline_points(thin)
    assert len(kept) < 0.5 * len(outline_points(plain)[0])
    a, b = thin["positions"][edges[:, 0], :2], thin["positions"][edges[:, 1], :2]
    q = fine["positions"][old, :2][:, None, :]
    t = np.clip(((q - a) * (b - a)).sum(-1) / np.maximum(((b - a) ** 2).sum(-1), 1e-20), 0.0, 1.0)
    apart = np.linalg.norm(q - (a + t[..., None] * (b - a)), axis=-1).min(axis=1)
    assert apart.max() <= pixel * 1.01                                          # the outline where it was
    inside = ~np.isin(thin["faces"], kept).any(axis=1)
    assert inside.sum() > 1.2 * (~np.isin(plain["faces"], outline_points(plain)[0]).any(axis=1)).sum()
    assert thin["uvs"].shape == (len(thin["positions"]), 2)


def test_the_face_rising_in_full_to_hair_beside_it_tapers_where_that_hair_ends_at_the_seam():
    # the Nachtschwinge's left temple: the face rose in full to the side lock it stands in for (`beside`, round 29),
    # 8 mm at the seam, and just below the lock's tip, where the seam meets the cheek, not at all: a ledge of the face
    # along the lock's underside, painted with the picture's dark hair and in shadow, a black stroke from the eye's
    # outer corner seen from three quarters (round 31 of the AssetFurnace Support notes). Tapered at 45 degrees from
    # where the face rises less at the seam, no ledge; away from the lock's end and off the hair nothing changes.
    pixel = 0.001
    rows, cols = np.mgrid[0:60, 0:60]
    field = (29.5 - cols) * pixel                    # the seam between columns 29 and 30, the face to its left
    new = np.zeros((60, 60))
    target = np.where(rows < 30, 0.008, 0.0)         # the lock beside the face above row 30, the cheek below ...
    target[40:46] = 0.008                            # ... and a cheek standing in front further down: not hair
    beside = rows < 30
    out = hf.taper_rise(target, new, beside, field, pixel, 20)
    slope = hf.TAPER_SLOPE * pixel
    assert np.abs(np.diff(out[:36, 29])).max() <= slope * (1 + 1e-6)           # along the seam at most the slope
    assert np.abs(np.diff(out[:30, 29])).max() == pytest.approx(slope)
    calm = int(30 - np.ceil(0.008 / slope))
    assert np.array_equal(out[:calm], target[:calm])                           # away from the lock's end as it was
    assert np.array_equal(out[30:], target[30:])                               # off the hair as it was
    assert (out <= target).all()
    assert np.array_equal(hf.taper_rise(target, new, np.zeros_like(beside), field, pixel, 20), target)



def bumpy_relief(half=0.03, step=0.0005):
    """A face-like relief, two triangles on every square of its grid: a broad dome (cheeks, forehead) and a narrow
    ridge down its middle (a nose)."""
    xs = np.arange(-half, half + step / 2, step)
    n = len(xs)
    x, y = np.meshgrid(xs, xs)
    z = 0.015 * np.exp(-(x ** 2 + y ** 2) / (2 * 0.018 ** 2)) + 0.006 * np.exp(-x ** 2 / (2 * 0.003 ** 2)
                                                                            - y ** 2 / (2 * 0.012 ** 2))
    a = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).reshape(-1)
    faces = np.concatenate([np.c_[a, a + 1, a + n + 1], np.c_[a, a + n + 1, a + n]])
    return np.c_[x.reshape(-1), y.reshape(-1), z.reshape(-1)], faces


def test_the_facet_share_counts_where_a_face_made_lighter_catches_the_light_unlike_the_fine_face():
    # the Nachtschwinge's face at 1,500 triangles: its light strayed from the fine face's by more than 0.05 on 12 % of
    # it from the front (round 31 of the AssetFurnace Support notes), and a reviewer saw pale polygons on its cheeks
    # and facets round its nose; the measure is the same light, drawn smooth as the game step shades it
    v, f = bumpy_relief()
    fine = {"positions": v, "faces": f}
    pixel = 0.0003
    assert hf.facet_share(fine, fine, pixel) == 0.0
    coarse = hf.facet_share(fine, hf.lighter(fine, 0.02), pixel)
    finer = hf.facet_share(fine, hf.lighter(fine, 0.2), pixel)
    assert coarse > 0.05
    assert finer < 0.5 * coarse


def test_a_face_made_lighter_keeps_triangles_until_it_shows_no_facets_and_a_flat_one_gives_them_up():
    # Taha's decision (#285 of the AssetFurnace Support notes): neither a decimation weighed by the normal's turn nor
    # a relaxation laid back onto the fine face took the Nachtschwinge's facets away at 1,500 triangles (round 32);
    # about 2,500 did. The face keeps what its own shape needs, a flat one no more than asked.
    v, f = bumpy_relief()
    fine = {"positions": v, "faces": f}
    pixel = 0.0003
    asked = 0.02
    first = hf.facet_share(fine, hf.lighter(fine, asked), pixel)
    light, share = hf.lighter_without_facets(fine, asked, pixel, limit=0.5 * first)
    assert share <= 0.5 * first
    assert len(light["faces"]) > 1.2 * asked * len(f)
    flat = {"positions": v * np.array([1.0, 1.0, 0.0]), "faces": f}
    plain, none = hf.lighter_without_facets(flat, asked, pixel, limit=0.5 * first)
    assert none <= 0.5 * first
    assert len(plain["faces"]) <= 1.05 * asked * len(f)

