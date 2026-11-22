"""Hero head under the brim: the head unrolled round its axis, the collar told from the neck and from a hollow, the
brim's tilted underside, the region cut out with hat and collar whole, and the new head set in closed all round.

The shapes are small synthetic stand-ins for what the Zauberer showed (2026-11-22): a skull under a tilted brim, a neck
behind the skull's middle, a high collar round it with a hollow inside its wall, and a build that is a little larger.
"""
from __future__ import annotations

import numpy as np
import pytest

from image_to_3dlab import hero_face as hf
from image_to_3dlab import hero_hands as hh
from image_to_3dlab import hero_head as hd

UP = np.array([0.0, 1.0, 0.0])


# --- shapes -------------------------------------------------------------------------------------------------------

def tube(radius, y0, y1, centre=(0.0, 0.0), segments=64, rings=24, inward=False, closed=True, flare=None):
    """A vertical tube round CENTRE (x, z) from Y0 to Y1, capped (CLOSED), faces out (or IN). FLARE (y, radius): below y
    the tube widens to that radius at Y0, as a neck fused into a collar."""
    cx, cz = centre
    ys = np.linspace(y0, y1, rings + 1)
    turn = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    r = np.full(len(ys), float(radius))
    if flare is not None:
        at, wide = flare
        low = ys < at
        r[low] = radius + (wide - radius) * (at - ys[low]) / max(at - y0, 1e-12)
    v = np.stack([cx + r[:, None] * np.sin(turn)[None, :], np.repeat(ys[:, None], segments, axis=1),
                  cz + r[:, None] * np.cos(turn)[None, :]], axis=-1).reshape(-1, 3)
    ids = np.arange(len(v)).reshape(len(ys), segments)
    a, b = ids[:-1], np.roll(ids[:-1], -1, axis=1)
    c, d = np.roll(ids[1:], -1, axis=1), ids[1:]
    f = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    if closed:
        bottom, top = len(v), len(v) + 1
        v = np.concatenate([v, [[cx, y0, cz], [cx, y1, cz]]])
        f = np.concatenate([f, np.stack([np.full(segments, bottom), np.roll(ids[0], -1), ids[0]], 1),
                            np.stack([np.full(segments, top), ids[-1], np.roll(ids[-1], -1)], 1)])
    if inward:
        f = f[:, ::-1]
    return v, f


def sphere(radius, centre=(0.0, 0.0, 0.0), rings=32, segments=48):
    """A closed UV sphere, faces out."""
    vertices = [[0.0, radius, 0.0]]
    for i in range(1, rings):
        theta = np.pi * i / rings
        for j in range(segments):
            phi = 2 * np.pi * j / segments
            vertices.append([radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta),
                             radius * np.sin(theta) * np.cos(phi)])
    vertices.append([0.0, -radius, 0.0])
    faces = [[0, 1 + j, 1 + (j + 1) % segments] for j in range(segments)]
    for i in range(rings - 2):
        a, b = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            j1 = (j + 1) % segments
            faces += [[a + j, b + j, b + j1], [a + j, b + j1, a + j1]]
    last, end = 1 + (rings - 2) * segments, len(vertices) - 1
    faces += [[last + j, end, last + (j + 1) % segments] for j in range(segments)]
    return np.asarray(vertices) + np.asarray(centre), np.asarray(faces)


def plate(inner, outer, y, tilt=(0.0, 0.0), centre=(0.0, 0.0), thickness=0.004, segments=64):
    """A closed flat ring (a brim) round CENTRE from INNER to OUTER, its underside at Y tilted by TILT (dy per x, per
    z), THICKNESS thick, faces out."""
    cx, cz = centre
    turn = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = []
    for radius in (inner, outer):
        x, z = cx + radius * np.sin(turn), cz + radius * np.cos(turn)
        ring.append((x, z))
    v = []
    for lift in (0.0, thickness):
        for x, z in ring:
            v.append(np.stack([x, y + tilt[0] * (x - cx) + tilt[1] * (z - cz) + lift, z], axis=1))
    v = np.concatenate(v)
    n = segments
    ids = np.arange(4 * n).reshape(4, n)        # bottom inner, bottom outer, top inner, top outer
    faces = []
    for a, b, flip in ((0, 1, False), (2, 3, True), (1, 3, False), (0, 2, True)):
        p, q = ids[a], ids[b]
        quad = np.concatenate([np.stack([p, np.roll(p, -1), np.roll(q, -1)], 1), np.stack([p, np.roll(q, -1), q], 1)])
        faces.append(quad[:, ::-1] if flip else quad)
    return v, np.concatenate(faces)


def joined(*meshes):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(np.asarray(v, dtype=float))
        faces.append(np.asarray(f) + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def open_count(v, f):
    return hd.open_edges([{"positions": v, "faces": f}])


# --- the axis -----------------------------------------------------------------------------------------------------

def test_skin_loop_passes_over_a_hollow_builds_inner_wall_and_a_collar_round_it():
    skin = tube(0.03, -0.1, 0.1)
    wall = tube(0.028, -0.1, 0.1, inward=True)
    collar = tube(0.05, -0.1, 0.1)
    v, f = joined(skin, wall, collar)
    points, faces, _ = hh.weld(v, f)
    loop = hd.skin_loop(points, faces, 0.0, (0.0, 0.0))
    assert np.allclose(np.hypot(loop[:, 0], loop[:, 2]), 0.03, atol=1e-3)
    assert hd.skin_loop(points, faces, 0.0, (0.2, 0.2)) is None


def test_neck_and_axis_follows_the_neck_down_to_the_collar_and_leans_through_it():
    skull = sphere(0.05, centre=(0.0, 0.1, 0.012))
    neck = tube(0.02, 0.0, 0.07)
    body = tube(0.045, -0.05, 0.0005)                            # the neck runs into a collar at y = 0
    v, f = joined(skull, neck, body)
    found = hd.neck_and_axis(v, f, (0.0, 0.012), 0.1, 0.05, 0.1, 0.001)
    assert 0.0 < found["neck"] < 0.003                           # the lowest section still a neck
    _, z = hd.centre(found["axis"], np.array([found["neck"], 0.1]))
    assert z[0] == pytest.approx(0.0, abs=2e-3) and z[1] == pytest.approx(0.012, abs=2e-3)   # through neck and skull


# --- the head unrolled --------------------------------------------------------------------------------------------

def frame_round(radius=0.04, y0=-0.05, y1=0.05, pixel=0.002):
    return hd.cylinder(hd.line_axis((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), y0, y1, pixel, radius)


def test_polar_and_unpolar_undo_each_other_round_a_leaning_axis():
    frame = hd.cylinder(hd.line_axis((0.01, 0.0, 0.02), (0.03, 1.0, 0.1)), -0.1, 0.1, 0.002, 0.05)
    points = np.array([[0.05, 0.03, 0.02], [-0.02, -0.05, 0.0], [0.0, 0.08, 0.2]])
    angle, radius = hd.polar(frame, points)
    assert np.allclose(hd.unpolar(frame, angle, radius, points[:, 1]), points)


def test_cylinder_hits_see_every_layer_with_the_way_it_faces():
    v, f = joined(tube(0.03, -0.1, 0.1, closed=False), tube(0.02, -0.1, 0.1, inward=True, closed=False))
    frame = frame_round()
    hits = hd.cylinder_hits(v, f, frame)
    middle = (frame["rows"] // 2) * frame["cols"] + frame["cols"] // 3
    here = hits["pixel"] == middle
    order = np.argsort(hits["radius"][here])
    assert np.allclose(hits["radius"][here][order], [0.02, 0.03], atol=5e-4)
    assert hits["facing"][here][order][0] < -0.9 and hits["facing"][here][order][1] > 0.9
    covered = np.unique(hits["pixel"])
    assert len(covered) == frame["rows"] * frame["cols"]        # every ray meets the tube: no gaps at the back seam


def test_brim_plane_reads_a_tilted_brim_and_finds_none_without_one():
    head = sphere(0.04)
    brim = plate(0.045, 0.12, 0.03, tilt=(0.1, 0.05))
    v, f = joined(head, brim)
    plane = hd.brim_plane(v, f, (0.0, 0.0), 0.0, 0.08, 0.001)
    assert plane["a"] == pytest.approx(0.03, abs=1e-3)
    assert plane["b"] == pytest.approx(0.1, abs=0.01) and plane["c"] == pytest.approx(0.05, abs=0.01)
    assert hd.brim_plane(*head, (0.0, 0.0), 0.0, 0.08, 0.001) is None


def test_collar_mask_takes_a_collar_standing_round_the_head_but_not_the_throat_between_its_fronts():
    rows, cols = 60, 80
    outer = np.full((rows, cols), 0.04)                 # the head
    outer[30:, :] = 0.05                                # a collar from row 30 down ...
    front = slice(30, 50)                               # ... open at the front: there the throat ...
    outer[40:, front] = 0.02
    outer[30:40, front] = 0.045                         # ... runs up into the chin, which stands out over it
    mask = hd.collar_mask(outer, 55, jump=0.004, stand=0.004, grow=0)
    assert mask[45, 5] and mask[31, 70] and not mask[29, 5]
    assert not mask[45, 40]                             # the throat
    grown = hd.collar_mask(outer, 55, jump=0.004, stand=0.004, grow=2)
    assert grown[28, 5] and not grown[26, 5]            # grown over the collar's own rim


def hits_of(layers_by_pixel, frame):
    """A hits dict from {(row, col): [(radius, facing), ...]}."""
    pixel, radius, facing = [], [], []
    for (row, col), crossings in layers_by_pixel.items():
        for r, fa in crossings:
            pixel.append(row * frame["cols"] + col)
            radius.append(r)
            facing.append(fa)
    n = len(pixel)
    return {"pixel": np.array(pixel), "radius": np.array(radius), "facing": np.array(facing), "up": np.zeros(n),
            "face": np.zeros(n, np.int64)}


def test_outermost_inside_tells_a_collars_hollow_from_the_air_between_collar_and_neck():
    frame = frame_round(y0=-0.02, y1=0.02, pixel=0.004)            # 10 rows
    cells = {}
    for row in range(frame["rows"]):
        for col in range(frame["cols"]):
            neck = [(0.02, 1.0), (0.018, -1.0)]
            if row >= 3:                                          # the collar, a hollow shell, up to row 3
                collar = [(0.05, 1.0), (0.048, -1.0), (0.032, 1.0), (0.03, -1.0)] if row > 3 else [(0.05, 1.0),
                                                                                                    (0.03, -1.0)]
                cells[(row, col)] = collar + neck
            else:
                cells[(row, col)] = neck
    inside = hd.outermost_inside(np.zeros((0, 3)), np.zeros((0, 3), np.int64), hits_of(cells, frame), frame, 0.06, 0.001)
    assert inside[6, 5] == pytest.approx(0.03, abs=0.0015)     # the collar's inside, not its hollow's far wall
    assert inside[1, 5] == -np.inf                             # above the collar the head reaches the axis


# --- the cut --------------------------------------------------------------------------------------------------------

def test_cycles_follow_a_cut_whatever_way_its_edges_point():
    edges = np.array([[0, 1], [1, 2], [3, 2], [3, 0], [5, 6], [6, 7], [7, 5], [8, 9]])   # one edge turned the wrong way
    loops = hd.cycles(edges)
    assert len(loops) == 2                                     # the open run 8-9 is left out
    square = next(loop for loop in loops if len(loop) == 4)
    assert list(np.roll(square, -int(np.argmin(square)))) == [0, 1, 2, 3]   # walked the way most edges point


def test_winding_counts_how_often_an_outline_goes_round_a_point():
    turn = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.stack([np.sin(turn), np.zeros(40), np.cos(turn)], axis=1)
    assert abs(hd.winding(ring, np.zeros(3))) == 1
    assert hd.winding(ring, np.array([5.0, 0.0, 0.0])) == 0


def test_cap_loops_closes_a_cut_and_leaves_no_edge_open():
    v, f = tube(0.03, -0.1, 0.1, closed=False)
    part = {"positions": v, "faces": f, "uvs": np.zeros((len(v), 2))}
    cut, rim = hh.cut_part_values(part, v[:, 1] - 0.05)         # keep below y = 0.05
    placed, first = hd.rim_ids(cut["positions"], rim)
    loops = [first[loop] for loop in hd.cycles(placed)]
    capped, closed = hd.cap_loops(cut, loops)
    assert closed == 1 and len(capped["uvs"]) == len(capped["positions"])
    assert hd.open_edges([capped]) == 64                        # only the tube's own open bottom is left
    normals = hf.face_normals(capped["positions"], capped["faces"][-64:])
    assert (normals[:, 1] > 0.99).all()                         # the cap faces up, out of the cut


# --- the whole head ---------------------------------------------------------------------------------------------------

def collar_ring(inner, outer, y0, top_back, top_front, centre=(0.0, 0.0), segments=64, rows=12):
    """A closed solid ring round CENTRE (x, z) from INNER to OUTER, from Y0 up to a top edge TOP_BACK high at the back
    and TOP_FRONT at the front (a high collar, low in front, where the face is), faces out."""
    cx, cz = centre
    turn = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    top = top_front + (top_back - top_front) * (1 - np.cos(turn)) / 2
    s = np.linspace(0, 1, rows + 1)[:, None]
    heights = y0 + s * (top[None, :] - y0)
    walls = [np.stack([cx + r * np.sin(turn)[None, :].repeat(rows + 1, 0), heights,
                       cz + r * np.cos(turn)[None, :].repeat(rows + 1, 0)], axis=-1).reshape(-1, 3)
             for r in (outer, inner)]
    v = np.concatenate(walls)
    ids = np.arange(2 * (rows + 1) * segments).reshape(2, rows + 1, segments)

    def band(p, q, flip):
        quads = np.concatenate([np.stack([p, np.roll(p, -1, axis=-1), np.roll(q, -1, axis=-1)], -1).reshape(-1, 3),
                                np.stack([p, np.roll(q, -1, axis=-1), q], -1).reshape(-1, 3)])
        return quads[:, ::-1] if flip else quads

    faces = [band(ids[0, :-1], ids[0, 1:], False), band(ids[1, :-1], ids[1, 1:], True),
             band(ids[1, -1], ids[0, -1], True), band(ids[1, 0], ids[0, 0], False)]
    return v, np.concatenate(faces)


def figure():
    """A head under a tilted brim with a high collar: a skull, a neck behind its middle that runs into the body (as wide
    as the collar's inside) at y = -0.045, a collar (a closed ring) round the neck from y = -0.09 up to y = -0.015 at
    the back and y = -0.04 in front, under the chin, and a brim whose underside is at about y = 0.03."""
    skull = sphere(0.032, centre=(0.0, 0.005, 0.006))
    middle = (0.0, -0.005)
    neck = joined(tube(0.018, -0.0455, -0.01, centre=middle), tube(0.034, -0.09, -0.045, centre=middle))
    collar = collar_ring(0.034, 0.04, -0.09, -0.015, -0.04, middle)
    brim = plate(0.036, 0.1, 0.03, tilt=(0.05, 0.02))
    pieces = (("skull", skull), ("neck", neck), ("collar", collar), ("brim", brim))
    starts = np.cumsum([0] + [len(v) for _, (v, _) in pieces])
    return joined(*(m for _, m in pieces)), {name: np.arange(a, b) for (name, _), a, b in zip(pieces, starts[:-1],
                                                                                             starts[1:])}


def test_head_zone_cuts_under_the_brim_inside_the_collar_and_above_where_the_neck_fuses():
    (v, f), ids = figure()
    is_head = v[:, 1] > -0.06
    window = hf.face_window(v, f, is_head)
    zone = hd.head_zone(v, f, window)
    assert zone["plane"]["a"] == pytest.approx(0.03, abs=2e-3)
    assert -0.045 < zone["bottom"] < -0.03                      # above the fusion, under the chin
    values = hd.zone_values(zone, v)
    skull_top = np.argmax(v[:, 1] * (np.hypot(v[:, 0], v[:, 2]) < 0.01))
    assert values[skull_top] < 0                                # the skull's top is under the hat: it stays
    side = ids["skull"][np.abs(v[ids["skull"], 1] - 0.005) < 0.002]
    assert (values[side] > 0).all()                             # the face's sides go
    assert (values[ids["collar"]] < 0).all() and (values[ids["brim"]] < 0).all()   # collar and brim stay whole


def test_graft_head_sets_a_new_head_in_closed_all_round():
    (v, f), _ = figure()
    is_head = v[:, 1] > -0.06
    window = hf.face_window(v, f, is_head)
    zone = hd.head_zone(v, f, window)
    new_v, new_f = sphere(0.033, centre=(0.0, 0.005, 0.006))    # the build: a little larger
    neck_v, neck_f = tube(0.019, -0.09, -0.01, centre=(0.0, -0.005))
    built = hd.build_radius(zone, *joined((new_v, new_f), (neck_v, neck_f)))
    radius = hd.tube_radius(zone, built)
    body = {"positions": v, "faces": f, "uvs": np.zeros((len(v), 2)),
            "dense": np.tile([1.0, 0.0], (len(v), 1))}
    parts, head, report = hd.graft_head([body], zone, radius, 2 * zone["pixel"])
    assert report["stitched"] and report["open_edges_near"] == 0
    assert hd.open_edges(parts + [head]) == hd.open_edges([body])      # nothing opened anywhere
    top = head["positions"][:, 1].max()
    assert top > zone["plane"]["a"]                              # the new hair reaches up into the hat
    kept = parts[0]["positions"]
    assert ((kept[:, 1] > 0.03) & (kept[:, 1] < 0.04) & (np.hypot(kept[:, 0], kept[:, 2]) > 0.09)).any()   # brim
    assert np.allclose(head["dense"], [1.0, 0.0])               # the old head's skin weights


def test_tube_paint_takes_the_picture_where_the_head_faces_the_camera_and_the_builds_paint_behind():
    frame = frame_round()
    radius = np.full((frame["rows"], frame["cols"]), 0.04)
    zone = {"frame": frame, "bottom": -0.04, "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)},
            "width": 0.08}
    points, normals = hd.tube_points(zone, radius, (16, 64))
    crop = np.full((64, 64, 3), [200.0, 100.0, 100.0])          # the picture: the face's own colour
    frame_full = (np.zeros(3), 0.1 / 64, np.array([32.0, 32.0]))
    build_v = points.reshape(-1, 3)
    front = build_v[:, 2] > 0
    paint = np.zeros((4, 4, 3))
    paint[:, :2], paint[:, 2:] = [180.0, 80.0, 80.0], [50.0, 50.0, 200.0]   # the build's face a little darker; its back
    uvs = np.where(front[:, None], [0.1, 0.5], [0.9, 0.5])
    build = {"positions": build_v, "uvs": uvs, "paint": paint}
    picture = hd.tube_paint(points, normals, crop, np.ones((64, 64)), (0, 0, 64, 64), frame_full, build, None, 0.08)
    assert np.allclose(picture[8, 32], [200, 100, 100], atol=2)     # the face: the picture's own pixels
    assert np.allclose(picture[8, 0], [70, 70, 220], atol=3)        # the back: the build's paint, moved as its face was
