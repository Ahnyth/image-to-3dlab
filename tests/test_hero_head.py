"""Hero head under the brim: the head unrolled round its axis, the collar told from the neck and from a hollow, the
brim's tilted underside, the region cut out with hat and collar whole, and the new head set in closed all round.

The shapes are small synthetic stand-ins for what the Zauberer showed (2026-11-22): a skull under a tilted brim, a neck
behind the skull's middle, a high collar round it with a hollow inside its wall, and a build that is a little larger.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

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
    brim = ids["brim"][np.hypot(v[ids["brim"], 0], v[ids["brim"], 2]) > 0.036 + 0.03 * zone["width"]]
    assert (values[ids["collar"]] < 0).all() and (values[brim] < 0).all()   # collar and brim stay whole (but the
    # brim's innermost ring at the hat's opening, which goes with the head so the cut runs smooth)


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
    assert report["neck"] == "into the collar" and report["open_edges_near"] == 0   # a collar round the neck: the new
    # neck reaches down into it (the stitched neck: below, without the collar)
    assert hd.open_edges(parts + [head]) == hd.open_edges([body])      # nothing opened anywhere
    top = head["positions"][:, 1].max()
    assert top > zone["plane"]["a"]                              # the new hair reaches up into the hat
    kept = parts[0]["positions"]
    assert ((kept[:, 1] > 0.03) & (kept[:, 1] < 0.04) & (np.hypot(kept[:, 0], kept[:, 2]) > 0.09)).any()   # brim
    assert np.allclose(head["dense"], [1.0, 0.0])               # the old head's skin weights
    (v, f), ids = figure()
    keep = ~np.isin(f, ids["collar"]).any(axis=1)               # without the collar the new head is stitched on
    bare = hd.head_zone(v, f[keep], hf.face_window(v, f[keep], v[:, 1] > -0.06))
    radius = hd.tube_radius(bare, hd.build_radius(bare, *joined((new_v, new_f), (neck_v, neck_f))))
    body = {"positions": v, "faces": f[keep], "uvs": np.zeros((len(v), 2)), "dense": np.tile([1.0, 0.0], (len(v), 1))}
    parts, head, report = hd.graft_head([body], bare, radius, 2 * bare["pixel"])
    assert report["stitched"] and report["neck"] == "stitched" and report["open_edges_near"] == 0
    assert hd.open_edges(parts + [head]) == hd.open_edges([body])


def test_build_radius_drops_a_thin_plate_the_rays_only_graze():
    (v, f), _ = figure()
    is_head = v[:, 1] > -0.06
    window = hf.face_window(v, f, is_head)
    zone = hd.head_zone(v, f, window)
    heights = hd.row_heights(zone["frame"])
    row = int(np.argmin(np.abs(heights)))
    # the build: a head, and a piece of its brim lower than the hat's, so thin that one row of rays crosses it
    plate_v, plate_f = plate(0.02, 0.05, heights[row] - 0.3 * zone["pixel"], thickness=0.6 * zone["pixel"])
    built = hd.build_radius(zone, *joined(sphere(0.033, centre=(0.0, 0.005, 0.006)), (plate_v, plate_f)))
    r = built["radius"]
    assert np.abs(r[row] - 0.5 * (r[row - 1] + r[row + 1])).max() < 0.005   # no plate standing out of the head
    assert not built["known"][row].any()        # nor a row the head's highest would be carried up from into the hat


def test_build_radius_leaves_out_a_tilted_plate_further_out_than_the_old_head_reached():
    (v, f), _ = figure()
    is_head = v[:, 1] > -0.06
    window = hf.face_window(v, f, is_head)
    zone = hd.head_zone(v, f, window)
    # a piece of the build's brim hanging lower on one side, tilted, out past the region's reach
    plate_v, plate_f = plate(0.02, 0.09, 0.0, tilt=(0.3, 0.0), thickness=0.6 * zone["pixel"])
    built = hd.build_radius(zone, *joined(sphere(0.033, centre=(0.0, 0.005, 0.006)), (plate_v, plate_f)))
    middle = np.abs(hd.row_heights(zone["frame"])) < 0.02
    reach = zone["cover"].max() + hd.COVER_MARGIN_SHARE * zone["width"]
    assert built["radius"][middle].max() <= reach                # the head, not the plate's streak out round it


def test_tube_radius_keeps_the_new_head_under_the_brim_about_as_wide_as_the_old():
    (v, f), _ = figure()
    is_head = v[:, 1] > -0.06
    window = hf.face_window(v, f, is_head)
    zone = hd.head_zone(v, f, window)
    frame = zone["frame"]
    heights = hd.row_heights(frame)[:, None]
    radius = np.full((frame["rows"], frame["cols"]), 0.03)
    wide = (heights > zone["band"].max()) & (np.arange(frame["cols"]) < frame["cols"] // 8)[None, :]
    radius[wide] = 0.07                                         # a piece of the build's brim hanging low at one side
    out = hd.tube_radius(zone, {"radius": radius, "known": np.ones(radius.shape, bool)})
    least = zone["cover"] + 1.5 * hd.COVER_MARGIN_SHARE * zone["width"]
    top = np.argmin(np.abs(heights[:, 0] - (zone["plane"]["a"] - zone["cut"])))   # at the cut, under the brim
    most = np.maximum(least, 0.03) + hd.COVER_CAP_SHARE * zone["width"]   # than the old head, or its own lower down
    assert (out[top] <= most + 1e-9).all()      # no wider than that there
    low = heights[:, 0] < zone["band"].min() - 4 * zone["pixel"]
    assert np.allclose(out[low & (heights[:, 0] > zone["bottom"] + 0.02)], 0.03)   # lower down the build's own


def test_graft_head_leaves_no_bit_of_the_old_head_that_dipped_just_under_the_stitch():
    (v, f), _ = figure()
    is_head = v[:, 1] > -0.06
    window = hf.face_window(v, f, is_head)
    zone = hd.head_zone(v, f, window)
    # the old chin's tip, in front of the neck, reaching half a pixel under the stitch (the Zauberer's: 0.2 mm)
    tip = (0.0, zone["bottom"] + 0.006 - 0.5 * zone["pixel"], 0.02)
    v2, f2 = joined((v, f), sphere(0.006, centre=tip))
    new_v, new_f = sphere(0.033, centre=(0.0, 0.005, 0.006))
    neck_v, neck_f = tube(0.019, -0.09, -0.01, centre=(0.0, -0.005))
    built = hd.build_radius(zone, *joined((new_v, new_f), (neck_v, neck_f)))
    radius = hd.tube_radius(zone, built)
    body = {"positions": v2, "faces": f2, "uvs": np.zeros((len(v2), 2))}
    parts, _, report = hd.graft_head([body], zone, radius, 2 * zone["pixel"])
    kept = parts[0]["positions"]
    under_tip = ((np.hypot(kept[:, 0] - tip[0], kept[:, 2] - tip[2]) < 0.006)
                 & (kept[:, 1] > zone["bottom"] - 2 * zone["pixel"]) & (kept[:, 1] < zone["bottom"] + 1e-9))
    assert not under_tip.any()                                  # the tip went with the head, it stays no patch
    assert report["slivers_dropped"] == 1 and report["open_edges_near"] == 0


def test_figure_paint_takes_the_figures_own_paint_where_the_picture_does_not_see():
    frame = frame_round()
    radius = np.full((frame["rows"], frame["cols"]), 0.04)
    zone = {"frame": frame, "axis": frame["axis"], "bottom": -0.04, "cut": 0.0, "width": 0.08, "pixel": 0.002,
            "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)}, "radius": radius,
            "limit": np.full((frame["rows"], frame["cols"]), 0.06), "reach": 0.06, "chin": -0.03,
            "nose": np.array([0.0, -0.005, 0.04])}
    points, normals = hd.tube_points(zone, radius, (40, 64))
    v, f = tube(0.039, -0.045, 0.035)                           # the figure's old head, just inside the new one
    skin = (v[:, 2] > 0) | ((v[:, 0] > 0.03) & (np.abs(v[:, 1]) < 0.012))   # its face, and an ear where the new
    part = {"positions": v, "faces": f, "uvs": np.where(skin[:, None], [0.25, 0.5], [0.75, 0.5]),   # head has hair
            "material": painted([200, 100, 100], right=[215, 215, 225])}    # its skin as the picture's, its hair pale
    colour = np.full((40, 64, 3), [200.0, 100.0, 100.0])        # the picture
    weight = np.clip((normals[..., 2] - 0.35) / 0.45, 0, 1)    # it sees the front
    picture = np.where((weight > 0.5)[..., None], colour, [50.0, 50.0, 200.0])    # elsewhere the build's own: blue
    out, offset = hd.figure_paint(picture, points, part, zone, crop=(colour, weight))
    middle = (points[..., 1] > -0.02) & (points[..., 1] < 0.02)
    assert np.allclose(out[(normals[..., 2] < -0.5) & middle], [215, 215, 225], atol=2)   # the figure's own hair
    assert np.allclose(out[(weight > 0.99) & middle], [200, 100, 100], atol=2)           # the picture's face
    assert np.allclose(offset, 0, atol=1)
    ear = (normals[..., 0] > 0.95) & (np.abs(points[..., 1]) < 0.008)
    assert np.allclose(out[ear], [215, 215, 225], atol=3)       # the old ear's skin is not painted on the new hair


def test_neck_fill_fills_the_pocket_under_a_jaw_with_a_slope_down_to_the_stitch():
    frame = frame_round()
    heights = hd.row_heights(frame)[:, None]
    turn = hd.column_angles(frame)[None, :]
    jaw = np.where(np.cos(turn) > 0.5, -0.034, -0.015)          # the chin's underside low, the jaw's at the back high
    radius = np.where(heights < jaw, 0.02, 0.04) * np.ones_like(turn)
    side = (np.sin(turn[0]) > 0.7) & (np.cos(turn[0]) > 0.5)    # a collar's edge under one side of the chin
    limit = np.where(side[None, :], 0.026, 0.06) * np.ones_like(heights)
    zone = {"frame": frame, "bottom": -0.04, "width": 0.08, "limit": limit}
    filled = hd.neck_fill(zone, radius)
    front = np.argmin(np.abs(turn[0]))
    row = np.argmin(np.abs(heights[:, 0] - (-0.035)))           # halfway between the stitch and the chin's edge
    assert 0.028 < filled[row, front] < 0.032                   # on the slope from the stitch's 0.02 to the chin's 0.04
    assert filled[row + 1, front] < filled[row, front]          # lower down, nearer the stitch's
    assert (filled[row, side] <= 0.026 + 1e-12).all() and (filled[row, side] > 0.022).all()   # filled, not into
    # the collar (its edge columns a little less, rounded into the neck beside them)
    back = np.argmax(np.abs(turn[0]))
    assert np.allclose(filled[:, back], radius[:, back])        # a jaw higher than the neck's reach: left as it is
    chin = np.cos(turn[0]) > 0.6
    assert np.allclose(filled[heights[:, 0] > -0.03][:, chin], radius[heights[:, 0] > -0.03][:, chin], atol=1e-4)   # the head
    # above the chin stays (beside it the slope runs round into the higher jaw)
    assert np.allclose(filled[heights[:, 0] < -0.04], radius[heights[:, 0] < -0.04])    # nothing under the stitch


def test_neck_fill_rounds_a_narrow_chin_into_the_jaw_beside_it():
    frame = frame_round()
    heights = hd.row_heights(frame)[:, None]
    turn = hd.column_angles(frame)[None, :]
    jaw = np.where(np.cos(turn) > 0.95, -0.034, -0.025)         # a narrow pointed chin, the jaw beside it higher
    radius = np.where(heights < jaw, 0.02, 0.04) * np.ones_like(turn)
    zone = {"frame": frame, "bottom": -0.04, "width": 0.08, "limit": np.full(radius.shape, 0.06)}
    filled = hd.neck_fill(zone, radius)
    row = np.argmin(np.abs(heights[:, 0] - (-0.036)))
    steps = np.abs(np.diff(np.r_[filled[row], filled[row, :1]]))
    assert steps.max() < 0.002                                  # its underside runs round into the jaw, no wall


def test_neck_fill_fills_the_groove_under_a_high_jaw_beside_the_chin():
    frame = frame_round()
    heights = hd.row_heights(frame)[:, None]
    turn = hd.column_angles(frame)[None, :]
    side = (np.abs(np.cos(turn)) < 0.8) & (np.cos(turn) > 0)           # the jaw beside the chin, its edge 0.15 of the
    jaw = np.where(side, -0.04 + 0.15 * 0.08, -0.034)                  # head's width over the stitch (the Zauberer's)
    radius = np.where(heights < jaw, 0.02, 0.04) * np.ones_like(turn)
    groove = side & (heights > jaw - 0.004) & (heights < jaw)           # under its edge the build's skin dips in
    radius = np.where(groove, 0.016, radius)
    zone = {"frame": frame, "bottom": -0.04, "width": 0.08, "pixel": 0.002, "limit": np.full(radius.shape, 0.06)}
    filled = hd.neck_fill(zone, radius)
    under = side & (heights > -0.04) & (heights < jaw - 0.002)
    assert filled[under].min() >= 0.02 - 1e-9                   # no groove under the jaw (a dark mark in the game)


def test_neck_fill_rounds_the_chins_lower_edge():
    frame = frame_round(pixel=0.0005)
    heights = hd.row_heights(frame)[:, None]
    radius = np.where(heights < -0.0375, 0.02, 0.04) * np.ones((1, frame["cols"]))   # a chin 2.5 mm over the stitch
    zone = {"frame": frame, "bottom": -0.04, "width": 0.08, "pixel": 0.0005, "limit": np.full(radius.shape, 0.06)}
    filled = hd.neck_fill(zone, radius)
    front = np.argmin(np.abs(hd.column_angles(frame)))
    corner = (heights[:, 0] > -0.0375) & (heights[:, 0] < -0.0365)
    assert filled[corner, front].max() < 0.0395                  # its front turns in toward the underside, no edge
    above = heights[:, 0] > -0.03
    assert np.allclose(filled[above, front], 0.04)               # the chin above it as it was


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


def painted(colour, right=None):
    """A material of one COLOUR, or COLOUR on its left half (u < 0.5) and RIGHT on its right."""
    import io

    from PIL import Image

    image = Image.new("RGB", (4, 4), tuple(int(c) for c in colour))
    if right is not None:
        image.paste(tuple(int(c) for c in right), (2, 0, 4, 4))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hf.picture_material(buffer.getvalue())


def test_figure_paint_paints_the_new_neck_under_the_chin_from_the_figures_own_paint():
    frame = frame_round()
    heights = hd.row_heights(frame)[:, None]
    radius = np.where(heights < -0.034, 0.02, 0.04) * np.ones((1, frame["cols"]))     # a neck under a chin
    zone = {"frame": frame, "axis": frame["axis"], "bottom": -0.04, "cut": 0.0, "width": 0.08, "pixel": 0.002,
            "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)}, "radius": radius,
            "limit": np.full((frame["rows"], frame["cols"]), 0.06)}
    points, _ = hd.tube_points(zone, radius, (40, 64))
    picture = np.full((40, 64, 3), [200.0, 170.0, 160.0])                       # the face
    lips = (points[..., 1] > -0.031) & (points[..., 1] < -0.027)     # over the chin's lower edge, which goes with
    # the neck (its underside)
    picture[lips] = [220.0, 120.0, 120.0]
    picture[points[..., 1] < -0.034] = [100.0, 75.0, 70.0]      # under the chin the throat in its shadow, a shirt
    turn = np.linspace(-np.pi, np.pi, 90, endpoint=False)
    rings = [(0.021, y) for y in np.arange(-0.0395, -0.034, 0.001)] + [(0.041, y) for y in np.arange(-0.033, 0.02, 0.002)]
    old = np.concatenate([hd.unpolar(frame, turn, np.full(90, r), np.full(90, y)) for r, y in rings])
    uvs = np.where((np.hypot(old[:, 0], old[:, 2]) < 0.03)[:, None], [0.75, 0.5], [0.25, 0.5])
    part = {"positions": old, "faces": np.zeros((0, 3), np.int64), "uvs": uvs,    # the figure's own paint: its face
            "material": painted([190, 160, 160], right=[120, 90, 90])}          # a little darker, its neck shaded
    out, offset = hd.figure_paint(picture, points, part, zone)
    assert np.allclose(offset, [-10, -10, 0], atol=1)           # the tone read on the face, not on the dark throat
    assert 118 <= out[-1, :, 0].min() and out[-1, :, 0].max() <= 140    # at the stitch: the figure's own neck ...
    under_chin = (points[..., 1] < -0.034) & (points[..., 1] > -0.0365)
    assert out[under_chin][:, 0].min() > 160                    # ... eased to the chin's own skin under its edge
    assert np.allclose(out[lips], [210, 110, 120], atol=2)      # the chin above it keeps the picture


def test_figure_paint_takes_no_collar_dark_onto_the_neck_beside_the_jaw():
    frame = frame_round()
    heights = hd.row_heights(frame)[:, None]
    radius = np.where(heights < -0.034, 0.02, 0.04) * np.ones((1, frame["cols"]))
    zone = {"frame": frame, "axis": frame["axis"], "bottom": -0.04, "cut": 0.0, "width": 0.08, "pixel": 0.002,
            "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)}, "radius": radius, "chin": -0.034,
            "nose": np.array([0.0, -0.005, 0.04]), "limit": np.full((frame["rows"], frame["cols"]), 0.06),
            "reach": 0.06}
    points, normals = hd.tube_points(zone, radius, (40, 64))
    skin = [200.0, 160.0, 150.0]
    old, faces = joined(tube(0.021, -0.0399, -0.0335, segments=96, rings=8), tube(0.039, -0.0335, 0.02, segments=96))
    aside = (np.abs(np.sin(np.arctan2(old[:, 0], old[:, 2]))) > 0.6) & (old[:, 1] < -0.034)
    part = {"positions": old, "faces": faces, "uvs": np.where(aside[:, None], [0.75, 0.5], [0.25, 0.5]),
            "material": painted(skin, right=[25, 25, 45])}      # beside the jaw the figure's neck in the collar's dark
    colour = np.full((40, 64, 3), skin)
    weight = np.clip((normals[..., 2] - 0.35) / 0.45, 0, 1)
    out, _ = hd.figure_paint(colour, points, part, zone, crop=(colour, weight, colour))
    neck = (points[..., 1] < -0.035) & (np.abs(points[..., 0]) > 0.6 * np.hypot(points[..., 0], points[..., 2]))
    assert neck.any() and out[neck].sum(axis=-1).min() > 0.7 * sum(skin)   # its skin, a little shaded, not black


def test_figure_paint_takes_no_collar_dark_into_a_deep_necks_band():
    # a neck reaching down into a collar eases its band from the paint just under the stitch; beside the jaw that is
    # the collar's dark, and on the Zauberer's v19l it stood under his jaw as a black ragged blotch (2026-11-24)
    frame = frame_round()
    heights = hd.row_heights(frame)[:, None]
    radius = np.where(heights < -0.024, 0.02, 0.04) * np.ones((1, frame["cols"]))
    zone = {"frame": frame, "axis": frame["axis"], "bottom": -0.03, "low": -0.045, "cut": 0.0, "width": 0.08,
            "pixel": 0.002, "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)}, "radius": radius,
            "chin": -0.024, "nose": np.array([0.0, -0.005, 0.04]), "reach": 0.06,
            "limit": np.full((frame["rows"], frame["cols"]), 0.06)}
    points, normals = hd.tube_points(zone, radius, (40, 64))
    skin = [200.0, 160.0, 150.0]
    old, faces = joined(tube(0.021, -0.0445, -0.0235, segments=96, rings=10), tube(0.039, -0.0235, 0.02, segments=96))
    aside = (np.abs(np.sin(np.arctan2(old[:, 0], old[:, 2]))) > 0.6) & (old[:, 1] < -0.024)
    part = {"positions": old, "faces": faces, "uvs": np.where(aside[:, None], [0.75, 0.5], [0.25, 0.5]),
            "material": painted(skin, right=[25, 25, 45])}      # beside the jaw the figure's neck in the collar's dark
    colour = np.full((40, 64, 3), skin)
    aside = np.abs(points[..., 0]) > 0.3 * np.hypot(points[..., 0], points[..., 2])
    colour[aside & (points[..., 1] < zone["bottom"])] = [30.0, 28.0, 40.0]   # and the picture shows the collar there
    weight = np.clip((normals[..., 2] - 0.35) / 0.45, 0, 1)
    out, _ = hd.figure_paint(colour, points, part, zone, crop=(colour, weight, colour))
    base = hd.stitch_rows(points[..., 1] >= zone["bottom"] - 0.5 * zone["pixel"])
    assert (base < points.shape[0] - 1).all()                   # the new neck goes on under the stitch
    band = hd.neck_band(points, zone)
    beside = band & aside
    assert beside.any() and out[beside].sum(axis=-1).min() > 0.7 * sum(skin)   # skin, a little shaded, not black
    under = aside & (points[..., 1] < zone["bottom"] - zone["pixel"])
    assert under.any() and np.median(out[under].sum(axis=-1)) < 0.5 * sum(skin)   # under the stitch the shirt stays
    cols = np.arange(points.shape[1])
    step = np.abs(np.diff(out[:, cols[aside.any(axis=0)]].sum(axis=-1), axis=0))[band[1:, aside.any(axis=0)]]
    assert step.max() < 0.25 * sum(skin)                        # and the band eases smoothly, no ragged jump in it


# --- the hairline under the brim (2026-11-23) -------------------------------------------------------------------------

def kinds_zone():
    frame = frame_round()
    zone = {"frame": frame, "bottom": -0.04, "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)},
            "width": 0.08, "chin": -0.03, "nose": np.array([0.0, -0.005, 0.04])}
    points, normals = hd.tube_points(zone, np.full((frame["rows"], frame["cols"]), 0.04), (40, 64))
    weight = np.clip((normals[..., 2] - 0.35) / 0.45, 0, 1)            # the picture sees the front
    turn = np.abs(np.arctan2(points[..., 0], points[..., 2]))
    return zone, points, weight, turn


SKIN, HAIR = [200.0, 150.0, 140.0], [210.0, 215.0, 230.0]


def test_same_kind_keeps_the_figures_hair_where_the_build_has_no_paint():
    zone, points, weight, _ = kinds_zone()
    face = (weight > 0.5) & (points[..., 1] < 0.0)                     # the face under the brow
    own = np.where(face[..., None], SKIN, HAIR)                         # the figure's own paint: hair over the face
    build = np.where(face[..., None], [230.0, 170.0, 150.0], [90.0, 120.0, 220.0])   # the build's: its skin, blue hair
    top = points[..., 1] > 0.015
    build[top] = np.nan                       # over its own brim (and where the new head stands out of it) no paint
    out = hd.same_kind(own, build, weight, points, zone)
    assert np.allclose(out[top], HAIR)        # the figure's hair under the brim stays hair, not its skin colour


def test_same_kind_keeps_the_figures_hair_at_the_back_where_the_build_made_up_skin():
    zone, points, weight, turn = kinds_zone()
    face = (weight > 0.5) & (points[..., 1] < 0.0)
    own = np.where(face[..., None], SKIN, HAIR)
    build = np.where(face[..., None], [230.0, 170.0, 150.0], [90.0, 120.0, 220.0])
    bald = (turn > 0.75 * np.pi) & (points[..., 1] < -0.01)            # built from the front, its back came out bald
    build[bald] = [230.0, 170.0, 150.0]
    out = hd.same_kind(own, build, weight, points, zone)
    assert np.allclose(out[bald], HAIR)       # the figure's own hair there stays, no tan patch at the back
    cheek = face & (turn > 0.2 * np.pi) & (turn < 0.3 * np.pi) & (np.abs(points[..., 1] + 0.015) < 0.006)
    own2 = np.where(cheek[..., None], HAIR, own)                        # a strand of the old head over the new cheek
    assert np.allclose(hd.same_kind(own2, build, weight, points, zone)[cheek], SKIN, atol=40)   # still turned skin


def test_same_kind_keeps_hair_under_the_brim_where_the_build_has_no_paint_over_its_forehead():
    zone, points, weight, turn = kinds_zone()
    forehead = (weight > 0.5) & (points[..., 1] < 0.015)               # the build shows its face up to its own brim
    build = np.where(forehead[..., None], [230.0, 170.0, 150.0], [90.0, 120.0, 220.0])
    build[points[..., 1] > 0.015] = np.nan                              # and nothing over it
    own = np.where(((weight > 0.5) & (points[..., 1] < 0.0))[..., None], SKIN, HAIR)
    top_front = (turn < 0.2 * np.pi) & (points[..., 1] > 0.022)
    out = hd.same_kind(own, build, weight, points, zone)
    assert np.allclose(out[top_front], HAIR)   # the figure's hair under the brim, not the build's forehead carried up


def test_same_kind_takes_no_hat_colour_onto_the_hair_under_the_brim():
    zone, points, weight, _ = kinds_zone()
    face = (weight > 0.5) & (points[..., 1] < 0.0)
    hat = np.array([42.0, 39.0, 75.0])                                  # the brim's underside, navy
    own = np.where(face[..., None], SKIN, HAIR)
    strip = points[..., 1] > 0.02                                       # right under the brim the figure's paint is
    own[strip] = [70.0, 45.0, 90.0]                                     # the hat's, plum (Pixel Match bled it there)
    build = np.where(face[..., None], [230.0, 170.0, 150.0], [90.0, 120.0, 220.0])
    out = hd.carry_up(hd.same_kind(own, build, weight, points, zone, hat=hat))
    assert np.allclose(out[strip], HAIR, atol=1)                        # the figure's hair carried up, no dark strip
    plain = hd.same_kind(own, build, weight, points, zone)
    assert np.allclose(out[~strip], plain[~strip])                      # the rest as without the hat's colour


def test_figure_paint_carries_the_figures_hair_up_into_the_hat_over_the_old_head():
    frame = frame_round()
    radius = np.full((frame["rows"], frame["cols"]), 0.04)
    zone = {"frame": frame, "axis": frame["axis"], "bottom": -0.04, "cut": 0.0, "width": 0.08, "pixel": 0.002,
            "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)}, "radius": radius,
            "limit": np.full((frame["rows"], frame["cols"]), 0.06), "reach": 0.06, "chin": -0.03,
            "nose": np.array([0.0, -0.005, 0.04])}
    points, normals = hd.tube_points(zone, radius, (40, 64))
    v, f = tube(0.039, -0.045, 0.012)          # the figure's old head: it ends under the new head's top (in the hat)
    skin = v[:, 2] > 0
    part = {"positions": v, "faces": f, "uvs": np.where(skin[:, None], [0.25, 0.5], [0.75, 0.5]),
            "material": painted([200, 100, 100], right=[215, 215, 225])}
    colour = np.full((40, 64, 3), [200.0, 100.0, 100.0])
    weight = np.clip((normals[..., 2] - 0.35) / 0.45, 0, 1)
    picture = np.where((weight > 0.5)[..., None], colour, [50.0, 50.0, 200.0])    # elsewhere the build's own: blue
    out, _ = hd.figure_paint(picture, points, part, zone, crop=(colour, weight))
    high = (points[..., 1] > 0.02) & (normals[..., 2] < -0.5)
    assert high.any() and np.allclose(out[high], [215, 215, 225], atol=3)   # the figure's hair, not the build's blue


def test_tube_radius_softens_the_builds_step_into_its_own_brim():
    (v, f), _ = figure()
    is_head = v[:, 1] > -0.06
    window = hf.face_window(v, f, is_head)
    zone = hd.head_zone(v, f, window)
    frame = zone["frame"]
    heights = hd.row_heights(frame)[:, None]
    radius = np.full((frame["rows"], frame["cols"]), 0.03)
    cut = zone["plane"]["a"] - zone["cut"]
    step = (heights > cut - 0.35 * (cut - zone["band"].min())) & (np.arange(frame["cols"]) < frame["cols"] // 2)
    radius[step] = 0.042                         # on one side the build runs out into its own brim, a step over the brow
    out = hd.tube_radius(zone, {"radius": radius, "known": np.ones(radius.shape, bool)})
    near = np.abs(heights[:, 0] - (cut - 0.1 * (cut - zone["band"].min()))) < 2 * zone["pixel"]
    steps = np.abs(np.diff(np.c_[out[near], out[near][:, :1]], axis=1))
    assert steps.max() < 0.0015                  # no notch: neighbouring columns run smoothly into each other


def test_head_zone_covers_only_the_old_head_that_stays_over_the_cut():
    (v, f), _ = figure()
    is_head = v[:, 1] > -0.06
    window = hf.face_window(v, f, is_head)
    plain = hd.head_zone(v, f, window)
    cut = plain["plane"]["a"] - plain["cut"]
    skirt = tube(0.048, cut - 0.014, cut - 0.006, closed=False)        # old hair hanging under the cut: it goes
    v2, f2 = joined((v, f), skirt)
    zone = hd.head_zone(v2, f2, hf.face_window(v2, f2, v2[:, 1] > -0.06))
    assert (zone["cover"] - plain["cover"]).max() < 0.001    # the new head is not pushed out to cover what goes


# --- the slim head under the hat (2026-11-23) --------------------------------------------------------------------------

def lathe(profile, centre=(0.0, 0.0), segments=64):
    """A closed surface turned round a vertical axis through CENTRE (x, z) from PROFILE, (radius, height) pairs from the
    bottom pole up the outside to the top pole, faces out."""
    cx, cz = centre
    turn = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = profile[1:-1]
    v = [[cx, profile[0][1], cz]]
    for r, y in rings:
        v += [[cx + r * np.sin(a), y, cz + r * np.cos(a)] for a in turn]
    v.append([cx, profile[-1][1], cz])
    ids = 1 + np.arange(len(rings) * segments).reshape(len(rings), segments)
    a, b = ids[:-1], np.roll(ids[:-1], -1, axis=1)
    c, d = np.roll(ids[1:], -1, axis=1), ids[1:]
    f = [np.stack([a, c, b], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3),
         np.stack([np.zeros(segments, int), np.roll(ids[0], -1), ids[0]], 1),
         np.stack([np.full(segments, len(v) - 1), ids[-1], np.roll(ids[-1], -1)], 1)]
    v, f = np.asarray(v, dtype=float), np.concatenate(f)
    if np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() < 0:
        f = f[:, ::-1]
    return v, f


HAIR_LINE = [(0.032 * np.sin(p), 0.005 - 0.032 * np.cos(p)) for p in np.linspace(0, np.pi / 2, 9)[1:]] + [
    (0.032, 0.016), (0.034, 0.021), (0.037, 0.025), (0.040, 0.0275)]           # the head, its hair flaring under the brim
BRIM_LINE = [(0.041, 0.028), (0.05, 0.0275), (0.06, 0.026), (0.075, 0.023), (0.09, 0.019), (0.1, 0.017)]   # droops
HAT_LINE = [(0.1, 0.021), (0.09, 0.023), (0.075, 0.027), (0.06, 0.030), (0.045, 0.032), (0.042, 0.034), (0.03, 0.06),
       (0.01, 0.085)]


def hat_figure():
    """A head and its hat as one closed surface, as Pixal3D builds them: the hair flares out under the brim into its
    underside, the brim droops toward its rim, so near the head its underside lies lower than where the hair meets it
    (the Zauberer's back, 2026-11-23); a neck and a high collar as `figure`'s."""
    head = lathe([(0.0, -0.027)] + HAIR_LINE + BRIM_LINE + HAT_LINE + [(0.0, 0.09)], centre=(0.0, 0.006))
    middle = (0.0, -0.005)
    neck = joined(tube(0.018, -0.0455, -0.01, centre=middle), tube(0.034, -0.09, -0.045, centre=middle))
    collar = collar_ring(0.034, 0.04, -0.09, -0.015, -0.04, middle)
    v, f = joined(head, neck, collar)
    return v, f, len(head[0])


def hat_zone():
    v, f, _ = hat_figure()
    return hd.head_zone(v, f, hf.face_window(v, f, v[:, 1] > -0.06))


def ring_points(radius, height, centre=(0.0, 0.006), count=32):
    turn = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.stack([centre[0] + radius * np.sin(turn), np.full(count, height), centre[1] + radius * np.cos(turn)], 1)


def test_head_zone_reaches_up_to_where_the_hair_meets_the_brim_and_leaves_the_brim():
    zone = hat_zone()
    assert zone["plane"]["a"] - zone["cut"] < 0.0255          # the old cut lay under the brim's drooping underside
    flare = ring_points(0.039, 0.0268)                         # the hair flaring out just under where it meets the brim
    assert (hd.zone_values(zone, flare) > 0).all()             # it goes with the old head: no shelf of it stays
    for radius, height in BRIM_LINE[1:3]:                     # the brim's underside, lower than that further out
        assert (hd.zone_values(zone, ring_points(radius, height)) < 0).all()   # stays whole


def test_tube_radius_keeps_the_new_head_at_its_own_width_up_into_the_hat():
    zone = hat_zone()
    frame = zone["frame"]
    heights = hd.row_heights(frame)
    radius = np.full((frame["rows"], frame["cols"]), 0.03)     # the build: a head 0.03 round, narrower than the flare
    out = hd.tube_radius(zone, {"radius": radius, "known": np.ones(radius.shape, bool)})
    high = heights > zone["plane"]["a"] - zone["cut"]          # over the old cut, up into the hat
    assert np.abs(out[high] - 0.03).max() < 0.002              # its own width, not widened to the old hair's 0.04


def test_opening_cap_spans_a_saddle_shaped_notched_outline_smoothly_without_folds():
    turn = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    radius = 0.05 * (1 + 0.15 * np.cos(2 * turn))              # an oval opening, 2 cm higher at the sides (a saddle)
    notch = np.arange(30, 34)                                  # and a notch whose inner end turns back, as seen from
    radius[notch] = [0.04, 0.03, 0.03, 0.04]                   # the middle (a fan from there folds over it)
    turn = turn.copy()
    turn[31] = turn[29] - 0.01
    loop = np.stack([radius * np.sin(turn), 0.03 - 0.01 * np.cos(2 * turn), radius * np.cos(turn)], axis=1)
    part = {"positions": loop, "faces": np.zeros((0, 3), np.int64), "uvs": np.zeros((120, 2))}
    cap = hd.opening_cap(part, [np.arange(120)], material="hat")
    tri = cap["positions"][cap["faces"]]
    area = 0.5 * ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 2] - tri[:, 0, 2])
                  - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 2] - tri[:, 0, 2]))
    folded = np.abs(area[np.sign(area) != np.sign(area.sum())]).sum() / np.abs(area).sum()
    assert folded < 0.005                                       # seen from below, no triangle turned over
    longest = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max()
    assert longest < 0.04                                       # no sliver across the opening, 0.1 wide
    assert hd.open_edges([cap]) == 120                          # closed but for its rim, where the cut meets it


def test_opening_cap_keeps_the_inner_walls_cap_over_the_outer_one():
    turn = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    outer = np.stack([0.06 * np.sin(turn), -0.002 - 0.012 * np.cos(2 * turn), 0.06 * np.cos(turn)], axis=1)
    inner = np.stack([0.05 * np.sin(turn), -0.0012 - 0.012 * np.cos(2 * turn), 0.05 * np.cos(turn)], axis=1)[::-1]
    part = {"positions": np.concatenate([outer, inner]), "faces": np.zeros((0, 3), np.int64),
            "uvs": np.zeros((192, 2))}       # the cut's outline on the brim (a saddle) and on the hollow's inner wall,
    # 0.8 mm over it all round (the Zauberer's at his back)
    cap = hd.opening_cap(part, [np.arange(96), 96 + np.arange(96)], material="hat")
    normals = hf.face_normals(cap["positions"], cap["faces"])
    seen = np.stack([cap["positions"][:, 0], -cap["positions"][:, 2], -cap["positions"][:, 1]], axis=1)
    _, owner = hf.depth_map(seen, cap["faces"], (-0.055, 0.055, -0.055, 0.055), 0.001)
    assert (normals[owner[owner >= 0], 1] < 0).all()    # seen from below only the outer cap, facing down: the inner
    # one (facing up, into the hollow) never shows through it, a see-through hole in the gallery


def test_graft_head_closes_the_hats_opening_in_the_hats_colour():
    v, f, head_vertices = hat_figure()
    zone = hd.head_zone(v, f, hf.face_window(v, f, v[:, 1] > -0.06))
    rings = 1 + np.arange(head_vertices - 2) // 64             # each lathe vertex's profile point
    profile = [(0.0, -0.027)] + HAIR_LINE + BRIM_LINE + HAT_LINE
    brim = np.zeros(len(v), bool)
    brim[1:head_vertices - 1] = (rings >= len(HAIR_LINE) + 1) & (rings <= len(HAIR_LINE) + len(BRIM_LINE) + 1)
    assert profile[len(HAIR_LINE) + 1] == BRIM_LINE[0]
    body = {"positions": v, "faces": f, "uvs": np.where(brim[:, None], [0.25, 0.5], [0.75, 0.5]),
            "material": painted([30, 30, 70], right=[210, 215, 230]), "dense": np.tile([1.0, 0.0], (len(v), 1))}
    built = hd.build_radius(zone, *joined(sphere(0.03, centre=(0.0, 0.005, 0.006)),
                                          tube(0.019, -0.09, -0.01, centre=(0.0, -0.005))))
    radius = hd.tube_radius(zone, built)
    parts, _, report = hd.graft_head([body], zone, radius, 2 * zone["pixel"])
    assert report["open_edges_near"] == 0 and report["neck"] == "into the collar"
    cap = parts[report["opening_cap"]]
    normals = hf.face_normals(cap["positions"], cap["faces"])
    middle = cap["positions"][cap["faces"]].mean(axis=1)
    _, distance = hd.polar(zone["frame"], middle)
    outside = distance > 0.032                                 # the cap's ring round the slim new head, seen from below
    assert outside.any() and (normals[outside, 1] < -0.5).all()
    assert np.allclose(hd.hat_colour(body, zone), [30, 30, 70], atol=2)   # the brim's underside, not the hair's silver


def test_figure_paint_takes_no_picture_over_the_old_heads_cut():
    frame = frame_round()
    radius = np.full((frame["rows"], frame["cols"]), 0.04)
    zone = {"frame": frame, "axis": frame["axis"], "bottom": -0.04, "cut": 0.004, "width": 0.08, "pixel": 0.002,
            "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)}, "radius": radius,
            "limit": np.full((frame["rows"], frame["cols"]), 0.06), "reach": 0.06, "chin": -0.03,
            "nose": np.array([0.0, -0.005, 0.04])}
    points, normals = hd.tube_points(zone, radius, (40, 64))
    v, f = tube(0.039, -0.045, 0.026)          # the old head ends at the cut, under the brim
    part = {"positions": v, "faces": f, "uvs": np.tile([0.75, 0.5], (len(v), 1)),
            "material": painted([200, 100, 100], right=[215, 215, 225])}
    colour = np.full((40, 64, 3), [40.0, 200.0, 200.0])      # over the cut the picture shows the hat: its rune band
    weight = np.clip((normals[..., 2] - 0.35) / 0.45, 0, 1)
    out, _ = hd.figure_paint(colour, points, part, zone, crop=(colour, weight))
    top = (weight > 0.9) & (points[..., 1] > 0.028)
    assert top.any() and np.allclose(out[top], [215, 215, 225], atol=3)   # the figure's hair, not the hat


# --- the neck down into the collar, the ears, a strand, the nape (2026-11-23, round 24) ------------------------------

def deep_zone():
    (v, f), ids = figure()
    window = hf.face_window(v, f, v[:, 1] > -0.06)
    return v, f, ids, hd.head_zone(v, f, window)


def test_head_zone_reaches_the_new_neck_down_into_a_collar_round_it():
    v, f, ids, zone = deep_zone()
    assert -0.045 < zone["bottom"] < -0.03                      # the cut stays where the old neck is free
    assert zone["low"] == pytest.approx(zone["bottom"] - hd.NECK_LOW_SHARE * zone["width"])
    assert zone["frame"]["y_lo"] < zone["low"]                  # the head's picture reaches down there
    keep = ~np.isin(f, ids["collar"]).any(axis=1)               # the same figure without its collar
    zone2 = hd.head_zone(v, f[keep], hf.face_window(v, f[keep], v[:, 1] > -0.06))
    assert zone2["low"] == zone2["bottom"]                      # a bare neck: stitched on as before


def test_head_zone_keeps_the_region_over_the_stitch_when_the_neck_reaches_down_into_a_collar(monkeypatch):
    # a frame reaching down into the collar can find the air between collar and head open to the outside lower down,
    # and the collar's inside measured further out over the stitch: the Zauberer's grew 10 to 15 mm round his back,
    # the cut took the collar's inside away there, and the fans closing it showed as pink-white stripes beside his jaw
    (v, f), _ = figure()
    window = hf.face_window(v, f, v[:, 1] > -0.06)
    real = hd.collar_limit

    def deep_sees_further(vertices, faces, hits, frame, bottom, width, reach):
        limit, collar, outer = real(vertices, faces, hits, frame, bottom, width, reach)
        if frame["y_lo"] < bottom - (hd.BLEND_ROWS + 1) * frame["pixel"]:   # the frame down into the collar
            limit = np.where(collar, np.minimum(limit + 0.1 * width, reach), limit)
        return limit, collar, outer

    monkeypatch.setattr(hd, "collar_limit", deep_sees_further)
    zone = hd.head_zone(v, f, window)
    assert zone["low"] < zone["bottom"] and zone["frame"]["y_lo"] < zone["low"]
    monkeypatch.setattr(hd, "NECK_COLLAR_LEAST", 2.0)           # the same figure measured down to the stitch only
    shallow = hd.head_zone(v, f, window)
    assert shallow["low"] == shallow["bottom"]
    rows = shallow["frame"]["rows"]
    over = hd.row_heights(shallow["frame"]) > shallow["bottom"]
    assert np.allclose(zone["limit"][:rows][over], shallow["limit"][over])
    assert ((hd.zone_values(zone, v) > 0) == (hd.zone_values(shallow, v) > 0)).all()   # the same cut


def test_graft_head_reaches_the_new_neck_down_into_the_collar_closed_round_the_old_one():
    v, f, _, zone = deep_zone()
    built = hd.build_radius(zone, *joined(sphere(0.033, centre=(0.0, 0.005, 0.006)),
                                          tube(0.017, -0.09, -0.01, centre=(0.0, -0.005))))   # its neck a bit narrower
    radius = hd.tube_radius(zone, built)
    body = {"positions": v, "faces": f, "uvs": np.zeros((len(v), 2)), "dense": np.tile([1.0, 0.0], (len(v), 1))}
    parts, head, report = hd.graft_head([body], zone, radius, 2 * zone["pixel"])
    assert report["neck"] == "into the collar" and not report["stitched"]
    assert hd.open_edges([head]) == 0                           # the new head closed, its neck's end too
    assert hd.open_edges(parts) == hd.open_edges([body])        # the old neck's cut closed
    y = head["positions"][:, 1]
    assert y.min() < zone["bottom"] - 0.5 * hd.NECK_LOW_SHARE * zone["width"]   # down into the collar
    _, r = hd.polar(zone["frame"], head["positions"])
    free = (y < zone["bottom"] - 0.5 * zone["pixel"]) & (y > zone["bottom"] - 2.5 * zone["pixel"])
    assert free.any() and r[free].min() >= 0.018 + 0.5 * hd.NECK_COVER_SHARE * zone["width"]   # round the old neck
    # (0.018 round) where it stands free under the stitch, so it never shows through the new one
    assert np.allclose(head["dense"], [1.0, 0.0])


def test_neck_band_stays_over_the_stitch_when_the_neck_reaches_down_into_the_collar():
    frame = frame_round()
    radius = np.full((frame["rows"], frame["cols"]), 0.02)
    zone = {"frame": frame, "axis": frame["axis"], "bottom": -0.03, "low": -0.045, "width": 0.08, "pixel": 0.002,
            "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)}, "cut": 0.0}
    points, _ = hd.tube_points(zone, radius, (40, 64))
    assert points[..., 1].min() < -0.04                         # the head's picture reaches down into the collar
    band = hd.neck_band(points, zone)
    assert band.any() and not band[points[..., 1] < zone["bottom"] - 0.001].any()   # the neck under the chin only


def test_figure_paint_starts_a_deep_necks_band_from_the_new_neck_under_the_stitch():
    frame = frame_round()
    heights = hd.row_heights(frame)[:, None]
    radius = np.where(heights < -0.024, 0.02, 0.04) * np.ones((1, frame["cols"]))     # a neck under a chin
    zone = {"frame": frame, "axis": frame["axis"], "bottom": -0.03, "low": -0.045, "cut": 0.0, "width": 0.08,
            "pixel": 0.002, "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)}, "radius": radius,
            "limit": np.full((frame["rows"], frame["cols"]), 0.06)}
    points, _ = hd.tube_points(zone, radius, (40, 64))
    picture = np.full((40, 64, 3), [200.0, 170.0, 160.0])                       # the face
    picture[points[..., 1] < zone["bottom"]] = [235.0, 215.0, 205.0]   # down in the collar: the picture's pale throat
    turn = np.linspace(-np.pi, np.pi, 90, endpoint=False)
    rings = [(0.021, y) for y in np.arange(-0.0445, -0.024, 0.001)] + [(0.041, y) for y in np.arange(-0.023, 0.02, 0.002)]
    old = np.concatenate([hd.unpolar(frame, turn, np.full(90, r), np.full(90, y)) for r, y in rings])
    uvs = np.where((np.hypot(old[:, 0], old[:, 2]) < 0.03)[:, None], [0.75, 0.5], [0.25, 0.5])
    part = {"positions": old, "faces": np.zeros((0, 3), np.int64), "uvs": uvs,    # the figure's own neck lies in
            "material": painted([190, 160, 160], right=[110, 80, 80])}          # its collar's shadow
    out, _ = hd.figure_paint(picture, points, part, zone)
    base = hd.stitch_rows(points[..., 1] >= zone["bottom"] - 0.5 * zone["pixel"])
    cols = np.arange(points.shape[1])
    assert (base < points.shape[0] - 1).all()                   # the new neck goes on under the stitch
    step = np.abs(out[base, cols] - out[base + 1, cols]).max()
    assert step < 12, step                                      # no hard edge where the band meets the new neck
    band = hd.neck_band(points, zone)
    assert out[band][:, 0].min() > 150                          # and no old, shaded neck in the band


def test_tube_radius_lays_a_strand_standing_off_the_face_onto_it():
    _, _, _, zone = deep_zone()
    frame, w = zone["frame"], zone["width"]
    heights = hd.row_heights(frame)[:, None]
    turn = hd.column_angles(frame)[None, :]
    middle = (heights > zone["bottom"] + 0.024) & (heights < zone["band"].min() - 0.004)
    radius = np.full((frame["rows"], frame["cols"]), 0.035)
    inner = radius.copy()
    strand = middle & (np.abs(turn - 0.3) < 2.2 * frame["turn"])      # 4 columns wide, 8 mm off the forehead
    lock = middle & (np.abs(turn + 1.6) < 30 * frame["turn"])         # a lock of hair 60 columns wide, off it too
    fin = middle & (np.abs(turn - 1.6) < 2.2 * frame["turn"])         # an ear seen edge-on: no gap behind it
    radius[strand | lock | fin] = 0.043
    inner[fin] = 0.043
    row = int(np.argmax(middle[:, 0])) + 10
    blade = np.zeros(radius.shape, bool)                              # a strand's tip seen edge-on: a blade one row
    blade[row, (np.abs(turn[0] + 0.6) < 15 * frame["turn"])] = True   # thin standing 8 mm out of the head
    radius[blade], inner[blade] = 0.043, 0.043
    out = hd.tube_radius(zone, {"radius": radius, "known": np.ones(radius.shape, bool), "inner": inner})
    assert out[blade].max() < 0.035 + hd.STRAND_GAP_SHARE * w                # it goes
    assert out[strand].max() < 0.035 + 2 * hd.STRAND_RELIEF_SHARE * w     # laid onto the face
    inside = ndimage.binary_erosion(middle[:, 0], iterations=20)[:, None]
    core = inside & (np.abs(turn + 1.6) < 10 * frame["turn"])
    assert out[core].min() > 0.042                                        # the lock keeps its volume (an ear's
    # stand-out at most, `round_flaps`: its rim rounded)
    cut = zone["plane"]["a"] - zone["cut"]
    brim = (heights > cut - 0.5 * hd.FLAP_BRIM_SHARE * w) & (heights < cut) & (np.abs(turn + 1.6) < 3 * frame["turn"])
    radius[brim] = 0.043                                  # a strand right under the brim at the side, where the hair
    pressed = hd.press_strands(zone, radius, inner, np.ones(radius.shape, bool))   # runs up into the hat's opening:
    assert np.allclose(pressed[brim], 0.043)                                    # left as it is
    assert pressed[middle & (np.abs(turn - 1.6) < 0.5 * frame["turn"])].max() > 0.042   # no air behind it: not laid
    # down as a strand (`round_flaps` rounds what stands out of the head's side on its own)


def test_neck_fill_eases_a_neck_into_a_narrower_nape_over_it():
    frame = frame_round()
    heights = hd.row_heights(frame)[:, None]
    radius = np.where(heights < -0.032, 0.02, 0.016) * np.ones((1, frame["cols"]))   # the nape narrower than its base
    zone = {"frame": frame, "bottom": -0.04, "width": 0.08, "limit": np.full((frame["rows"], frame["cols"]), 0.06)}
    filled = hd.neck_fill(zone, radius)
    back = int(np.argmax(np.abs(hd.column_angles(frame))))
    rows = (heights[:, 0] > -0.04) & (heights[:, 0] < -0.04 + hd.NECK_BAND_SHARE * 0.08 + 0.006)
    assert np.abs(np.diff(filled[rows, back])).max() < 0.002   # no shelf where the neck's fill ends under the hair


def striped(*colours):
    """A material of vertical stripes, one per colour; stripe k sits at u = (k + 0.5) / len(colours)."""
    import io

    from PIL import Image

    image = Image.new("RGB", (4 * len(colours), 4))
    for k, colour in enumerate(colours):
        image.paste(tuple(int(c) for c in colour), (4 * k, 0, 4 * k + 4, 4))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hf.picture_material(buffer.getvalue())


def face_window_box(half_x=0.02, half_y=0.03, pixel=0.001):
    """A face window (as hero_face.face_window gives one) round the front: |x| < HALF_X, |y| < HALF_Y."""
    box = (-0.06, 0.06, -0.06, 0.06)
    xs = box[0] + (np.arange(120) + 0.5) * pixel
    ys = box[3] - (np.arange(120) + 0.5) * pixel
    x, y = np.meshgrid(xs, ys)
    return {"field": np.minimum(half_x - np.abs(x), half_y - np.abs(y)), "box": box, "pixel": pixel}


def side_scene():
    """A round head (0.04) with an ear standing 6 mm out of its left side, over a figure whose old head is pale in
    front, pink at its sides and silver behind; a build painting tan skin, blue hair behind and the ear's red bowl; a
    picture that sees the face, and the brim's dark shadow on the ear's front. Returns (zone, points, normals, part,
    build, weight, colour, angle, ear, face colour)."""
    frame = frame_round()
    radius = np.full((frame["rows"], frame["cols"]), 0.04)
    side = np.abs(hd.column_angles(frame)[None, :] - np.pi / 2) < 0.15
    radius[side & (np.abs(hd.row_heights(frame))[:, None] < 0.006)] += 0.006
    zone = {"frame": frame, "axis": frame["axis"], "bottom": -0.04, "cut": 0.0, "width": 0.08, "pixel": 0.002,
            "plane": {"a": 0.03, "b": 0.0, "c": 0.0, "axis": (0.0, 0.0)}, "radius": radius,
            "limit": np.full((frame["rows"], frame["cols"]), 0.06), "reach": 0.06, "chin": -0.03,
            "nose": np.array([0.0, -0.005, 0.04]), "window": face_window_box()}
    points, normals = hd.tube_points(zone, radius, (40, 64))
    face_c, side_c, hair_c = [200.0, 185.0, 185.0], [215.0, 150.0, 150.0], [210.0, 215.0, 230.0]
    v, f = tube(0.039, -0.045, 0.035)                             # the figure's old head: its face pale, its sides
    turn = np.abs(np.arctan2(v[:, 0], v[:, 2]))                   # pink (a generator's guess), its hair silver
    u = np.where(turn < np.pi / 6, 1 / 6, np.where(turn > 2 * np.pi / 3, 5 / 6, 0.5))
    part = {"positions": v, "faces": f, "uvs": np.stack([u, np.full(len(v), 0.5)], 1),
            "material": striped(face_c, side_c, hair_c)}
    angle = np.arctan2(points[..., 0], points[..., 2])
    build = np.full(points.shape, [220.0, 170.0, 130.0])          # the build: tan skin ...
    build[(np.abs(angle) > 0.8 * np.pi) & (points[..., 1] > -0.005)] = [90.0, 120.0, 220.0]   # ... blue hair behind
    ear = (np.abs(angle - np.pi / 2) < 0.15) & (np.abs(points[..., 1]) < 0.006)
    build[ear] = [140.0, 70.0, 60.0]                              # ... and an ear with a red bowl at its side
    weight = np.clip((normals[..., 2] - 0.35) / 0.45, 0, 1)
    weight[ear] = 1.0                                             # the ear's front faces the camera ...
    colour = np.full(points.shape, face_c)
    colour[ear] = [60.0, 50.0, 110.0]                             # ... but the picture shows the brim's shadow there
    return zone, points, normals, part, build, weight, colour, angle, ear, face_c


def test_figure_paint_keeps_the_picture_on_the_heads_sides_off_the_ears():
    # a lock of hair laid onto the temple (`press_strands`) lies where the build painted skin: the picture's hair stays
    # on it, and nothing changes at the face window's edge; gated on all the skin off the face, the Zauberer's lock
    # turned skin and the gate's edge stood as a hard line down both temples (2026-11-23)
    zone, points, _, part, build, weight, colour, angle, ear, face_c = side_scene()
    lock = (np.abs(angle + 0.35 * np.pi) < 0.12) & (np.abs(points[..., 1]) < 0.01)   # on the right temple, off the face
    assert lock.any() and not (hf.window_values(zone["window"], points[lock]) > 0).any()
    lock_c = [205.0, 210.0, 228.0]
    colour[lock], weight[lock] = lock_c, 0.9
    out, _ = hd.figure_paint(colour, points, part, zone, crop=(colour, weight, build))
    assert np.allclose(out[lock], lock_c, atol=6)
    ear_c = out[ear].mean(axis=0)
    assert ear_c.sum() < 0.85 * sum(face_c) and np.allclose(ear_c / ear_c.sum(), np.array(face_c) / sum(face_c),
                                                            atol=0.02)     # the ear still the build's, in skin's hue


def test_figure_paint_paints_unseen_skin_from_the_builds_paint_in_the_faces_tone_never_the_pictures_edge():
    zone, points, _, part, build, weight, colour, angle, ear, face_c = side_scene()
    out, _ = hd.figure_paint(colour, points, part, zone, crop=(colour, weight, build))
    middle = np.abs(points[..., 1]) < 0.02
    assert np.allclose(out[(weight > 0.99) & middle & ~ear], face_c, atol=2)     # the picture's face
    ear_c = out[ear].mean(axis=0)
    assert 0.5 < ear_c.sum() / sum(face_c) < 0.85              # the build's ear: its bowl darker than the cheek ...
    assert np.allclose(ear_c / ear_c.sum(), np.array(face_c) / sum(face_c), atol=0.02)   # ... in the face's hue, not
    # the picture's dark violet there
    cheek = (np.abs(np.abs(angle) - 0.45 * np.pi) < 0.1) & middle & (weight < 0.01) & ~ear
    assert cheek.any() and np.allclose(out[cheek], face_c, atol=6)     # the unseen skin in the face's tone, not pink


def test_tube_radius_rounds_an_ears_rim_and_keeps_it_from_standing_out_as_a_plate():
    _, _, _, zone = deep_zone()
    frame, w = zone["frame"], zone["width"]
    heights = hd.row_heights(frame)[:, None]
    turn = hd.column_angles(frame)[None, :]
    middle = (heights > zone["bottom"] + 0.024) & (heights < zone["band"].min() - 0.004)
    radius = np.full((frame["rows"], frame["cols"]), 0.035)
    rows = np.flatnonzero(middle[:, 0])
    ear_rows = (np.arange(frame["rows"]) >= rows[len(rows) // 2] - 12) & (np.arange(frame["rows"]) < rows[len(rows) // 2] + 12)
    ear = ear_rows[:, None] & (np.abs(turn - np.pi / 2) < 16 * frame["turn"])   # an ear at the side: a plate 12 mm out,
    radius[ear] = 0.047                                                       # its rim a wall all round
    nose = ear_rows[:, None] & (np.abs(turn) < 16 * frame["turn"])           # the same at the front: a nose
    radius[nose] = 0.047
    out = hd.tube_radius(zone, {"radius": radius, "known": np.ones(radius.shape, bool)})
    side = ear_rows & True
    col = int(np.argmin(np.abs(turn[0] - np.pi / 2 - 16 * frame["turn"])))   # across the ear's back rim
    rim = np.abs(np.diff(out[side][:, col - 4:col + 5], axis=1)).max()
    assert rim < 0.006                                     # its rim a rounded slope, not a 12 mm wall
    assert out[ear].max() <= 0.035 + hd.FLAP_STAND_SHARE * w + 1e-9   # no further out than an ear stands
    assert out[ear].max() > 0.035 + 0.5 * hd.FLAP_STAND_SHARE * w    # still an ear
    assert out[nose].max() > 0.046                         # the face's own features stay as they are
    cut = zone["plane"]["a"] - zone["cut"]
    high = (heights > cut - 0.5 * hd.FLAP_BRIM_SHARE * w) & (heights < cut) & (np.abs(turn - np.pi / 2) < 16 * frame["turn"])
    flat = np.full(radius.shape, 0.035)
    flat[high] = 0.047                                     # hair right under the brim at the side, where it runs up
    assert np.allclose(hd.round_flaps(zone, flat)[high], 0.047)   # into the hat's opening: left as it is


def test_rounding_an_ear_or_laying_a_strand_down_never_narrows_the_head_under_the_brim(monkeypatch):
    zone = hat_zone()
    frame, w = zone["frame"], zone["width"]
    heights = hd.row_heights(frame)[:, None]
    turn = hd.column_angles(frame)[None, :]
    radius = np.full((frame["rows"], frame["cols"]), 0.03)
    inner = radius.copy()
    side = np.abs(turn - np.pi / 2) < 14 * frame["turn"]
    band = (heights > zone["band"][None, :]) & (heights < zone["plane"]["a"] - zone["cut"])
    radius[side & band] = 0.034                       # hair standing off the temple under the brim, a gap behind it
    plain = hd.tube_radius(zone, {"radius": radius, "known": np.ones(radius.shape, bool)})
    full = hd.tube_radius(zone, {"radius": radius, "known": np.ones(radius.shape, bool), "inner": inner})
    cut = zone["plane"]["a"] - zone["cut"]
    near = (heights[:, 0] > cut - 0.05 * w) & (heights[:, 0] < cut)
    assert np.allclose(full[near], plain[near])       # the head's width right under the brim as without them: taken
    # in there, the hat's opening film showed its up-facing parts round it from below (see-through slits)
    monkeypatch.setattr(hd, "round_flaps", lambda zone, r: r)
    unrounded = hd.tube_radius(zone, {"radius": radius, "known": np.ones(radius.shape, bool), "inner": inner})
    assert np.allclose(full[near], unrounded[near])   # nor does rounding the ears ...
    calls, crown, flaps = [], hd.crown_width, hd.round_flaps
    monkeypatch.setattr(hd, "crown_width", lambda *a: calls.append("crown") or crown(*a))
    monkeypatch.setattr(hd, "round_flaps", lambda *a: calls.append("flaps") or flaps(*a))
    hd.tube_radius(zone, {"radius": radius, "known": np.ones(radius.shape, bool), "inner": inner})
    assert calls.index("crown") < calls.index("flaps")   # ... which comes after the head's own width under the brim is
    # read (`crown_width`): rounded before, the Zauberer's ears narrowed it (533 see-through pixels against 20)


def freed_zone(outer=0.05, new=0.03):
    """A head's zone round an upright axis at x = z = 0 from y 0.1 to 0.3: the old head's outermost surface OUTER from
    the axis, the new head's NEW; the columns facing the back (|angle| > 2.5) held a collar (outer -inf)."""
    frame = hd.cylinder(hd.line_axis((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), 0.1, 0.3, 0.002, 0.05)
    old = np.full((frame["rows"], frame["cols"]), outer)
    old[:, np.abs(hd.column_angles(frame)) > 2.5] = -np.inf
    zone = {"frame": frame, "outer": old, "low": 0.1, "bottom": 0.12, "width": 0.1, "pixel": 0.002}
    return zone, np.full((frame["rows"], frame["cols"]), new)


def test_a_piece_the_old_head_hid_and_the_new_one_leaves_free_is_found():
    # the Zauberer's jaw (round 27): 13 triangles of his collar's inner edge, 3.4-4.0 cm from the neck's axis, sat
    # inside his old, wider jaw; beside the slimmer new head they stood free in front of it as a dark wedge
    zone, radius = freed_zone()
    points = np.array([[0.0, 0.2, 0.04],     # between the new head and where the old one was: freed
                       [0.0, 0.2, 0.02],     # inside the new head: hidden still
                       [0.0, 0.2, 0.06],     # outside where the old head was: it showed before too
                       [0.0, 0.35, 0.04],    # above the head's region
                       [0.0, 0.2, -0.04]])   # behind, where a collar stood outermost: not the old head's
    assert hd.freed_points(points, zone, radius).tolist() == [True, False, False, False, False]


def test_freed_pieces_are_tucked_inside_the_new_head_and_the_rest_stays():
    zone, radius = freed_zone()
    part = {"positions": np.array([[0.0, 0.2, 0.04], [0.004, 0.2, 0.04], [0.0, 0.204, 0.04], [0.0, 0.2, 0.06]]),
            "faces": np.array([[0, 1, 2], [0, 2, 3]])}
    out, moved = hd.tuck_freed(part, zone, radius, keep=np.array([False, False, False, False]))
    assert moved == 3
    _, r = hd.polar(zone["frame"], out["positions"])
    assert np.all(r[:3] < 0.03) and r[3] == pytest.approx(0.06)               # tucked inside; the outside one stays
    assert np.allclose(out["positions"][:3, 1], part["positions"][:3, 1])     # at its own height
    again, moved = hd.tuck_freed(part, zone, radius, keep=np.array([True, False, False, False]))
    assert moved == 2 and np.allclose(again["positions"][0], part["positions"][0])   # a rim point stays where it is
