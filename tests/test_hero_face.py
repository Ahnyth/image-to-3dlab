"""Hero face: finding the face between a brim and a collar, lining a face built alone up on it, cutting both along one
seam and stitching them.

The shapes are small synthetic stand-ins for what the Zauberer showed (2026-11-22): a head under a wide brim, a
strand standing in front of the face, a collar hidden behind the jaw, a new face a little larger than the old one.
The graft's claim is tested as a mesh property, as for the hands: after the stitch every edge is shared by exactly two
faces, so there is no gap and no overlap.
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


def closed_edges(vertices, faces):
    """(edges used by other than two faces, directed edges used twice) of a mesh welded by position."""
    group = hh.weld_groups(vertices)
    f = group[np.asarray(faces)]
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    directed = directed[directed[:, 0] != directed[:, 1]]
    _, counts = np.unique(np.sort(directed, 1), axis=0, return_counts=True)
    _, same = np.unique(directed, axis=0, return_counts=True)
    return int((counts != 2).sum()), int((same > 1).sum())


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


# --- cutting and stitching ------------------------------------------------------------------------------------------

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


def test_face_relief_takes_the_skin_of_a_hollow_build_and_gives_one_clean_rim():
    v, f = hollow_head()
    window = disk_window(0.035, width=0.1)
    frame = (np.zeros(3), 0.001, np.array([500.0, 500.0]))              # 1 mm a pixel, picture centre at 500
    face = hf.face_relief({"positions": v, "faces": f}, window, (400, 400, 600, 600), 1024, frame)
    piece, rim = hf.cut_face(face, window)
    ids, loop = hf.front_rim(piece["positions"], rim, np.zeros(3))
    assert len(ids) > 50
    assert hh.loop_area(loop, hf.FRONT) == pytest.approx(np.pi * 0.035 ** 2, rel=0.05)
    assert np.abs(np.linalg.norm(piece["positions"], axis=1) - 0.05).max() < 1e-3     # all of it on the skin
    normals = hf.face_normals(piece["positions"], piece["faces"])
    assert (normals[:, 2] > 0).mean() > 0.99                                          # facing the camera


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
    cut, rim = hh.cut_part_values({"positions": v, "faces": f}, -hf.window_values(window, v))
    keep = hf.front_piece(cut["positions"], cut["faces"])
    middle = cut["positions"][cut["faces"][keep]].mean(axis=1)
    assert (np.linalg.norm(middle, axis=1) > 0.0495).all()      # the outer skin only
    piece, piece_rim = hh.keep_faces(cut, keep, rim)
    _, loop = hf.front_rim(piece["positions"], piece_rim, np.zeros(3))
    assert hh.loop_area(loop, hf.FRONT) == pytest.approx(np.pi * 0.035 ** 2, rel=0.05)


def test_front_rim_takes_the_outer_skin_not_the_inner_one_behind_it():
    outer, inner = sphere(0.05), sphere(0.049)
    inner = (inner[0], inner[1][:, ::-1])          # a second skin a millimetre inside, facing in
    v, f = joined(outer, inner)
    window = disk_window(0.03)
    cut, rim = hh.cut_part_values({"positions": v, "faces": f}, hf.window_values(window, v),
                                  v[f].mean(axis=1)[:, 2] > 0)
    ids, loop = hf.front_rim(cut["positions"], rim, np.zeros(3))
    assert len(ids) > 20
    assert np.linalg.norm(loop, axis=1).mean() == pytest.approx(0.05, abs=2e-4)


def test_ease_depth_moves_the_rim_in_full_and_the_middle_not_at_all():
    points = np.array([[0.0, 0.0, 1.0], [0.9, 0.0, 1.0], [0.5, 0.0, 1.0]])
    values = np.array([1.0, 0.0, 0.05])
    rim_xy = np.array([[1.0, 0.0], [-1.0, 0.0]])
    out = hf.ease_depth(points, values, rim_xy, np.array([-0.2, 0.4]), blend=0.1)
    assert out[0, 2] == pytest.approx(1.0) and out[1, 2] == pytest.approx(0.8)
    assert 0.8 < out[2, 2] < 1.0


def test_loop_depth_reads_the_outline_between_its_corners():
    square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    depth = hf.loop_depth(square, np.array([[0.25, -0.1], [1.2, 0.5], [0.5, 0.9], [0.0, 0.5]]))
    assert np.allclose(depth, [0.25, 1.0, 0.5, 0.0])


def test_reference_depth_takes_the_deeper_of_the_two_where_both_are():
    own = np.array([[0.5, -np.inf, 0.2, -np.inf]])
    new = np.array([[0.4, 0.3, -np.inf, -np.inf]])
    assert hf.reference_depth(own, new).tolist() == [[0.4, 0.3, 0.2, np.inf]]


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


def test_graft_face_leaves_one_closed_surface_without_the_strand_and_with_the_hidden_collar():
    body, _, _ = figure_head()
    new_v, new_f = sphere(0.052, rings=40, segments=56)          # the new face stands 2 mm proud of the old,
    new_v = new_v @ turn_z(1.7).T                                # turned so no rim point of it lands on one of the old
    face = {"positions": new_v, "faces": new_f, "uvs": np.full((len(new_v), 2), 0.5)}
    window = disk_window(0.035, width=0.1)
    parts, stitched, report = hf.graft_face([body], window, face)
    v = np.concatenate([parts[0]["positions"], stitched["positions"]])
    f = np.concatenate([parts[0]["faces"], stitched["faces"] + len(parts[0]["positions"])])
    assert closed_edges(v, f) == (0, 0)
    assert not report["rims_turn_alike"]
    kept = parts[0]["positions"]
    assert not ((kept[:, 2] > 0.052) & (kept[:, 0] < -0.011)).any()     # the strand went
    assert ((kept[:, 1] < -0.021) & (kept[:, 2] > 0.019)).any()         # the collar behind the jaw stayed
    assert report["rim_gap_max"] == pytest.approx(0.0385 - 0.0357, abs=1e-3)   # the two spheres' depths at the seam
    assert report["parts"][0]["old_in_front_max"] > 0.004               # the strand stood that far in front
    # the new face meets the old rim: its rim was eased to the figure's depth
    rim = stitched["positions"][-3 * (report["stitch_faces"] // 3):]
    assert np.abs(np.linalg.norm(rim, axis=1) - 0.05).max() < 1e-3
    # the face took the front's weights, and the band the body's or the face's
    assert np.allclose(stitched["dense"], [0.0, 1.0])


def test_graft_face_leaves_parts_away_from_the_face_alone():
    body, _, _ = figure_head()
    hand_v, hand_f = box((0.3, -0.3, 0.0), (0.4, -0.2, 0.1))
    hand = {"positions": hand_v, "faces": hand_f}
    face = dict(zip(("positions", "faces"), sphere(0.051)))
    parts, _, report = hf.graft_face([hand, body], disk_window(0.035, width=0.1), face)
    assert parts[0] is hand and report["parts"][0] == {"cut": False} and report["body_part"] == 1


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
