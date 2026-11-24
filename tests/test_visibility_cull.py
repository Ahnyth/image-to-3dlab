"""Tests for visibility-based face culling.

These import the shipping helpers directly. The Blender rendering half is deliberately
thin and untested here; everything that can be got wrong in arithmetic lives in the pure
module and is checked below.
"""

import numpy as np
import pytest

from scripts.visibility_cull import (
    MAX_FACES,
    color_to_index,
    cull_report,
    dense_surface,
    face_samples,
    fibonacci_directions,
    first_air_sealed,
    hidden_from_rays,
    hollows_only,
    in_sealed_air,
    index_to_color,
    keep_seen_rims,
    poking_out,
    sealed_air,
    shading_roofs,
    visible_faces,
)


def test_directions_are_unit_length():
    dirs = fibonacci_directions(200)
    assert dirs.shape == (200, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)


def test_directions_cover_the_sphere_evenly():
    """A lat/long grid clumps at the poles; the point of Fibonacci is that it does not."""
    dirs = fibonacci_directions(1000)
    # Mean of a uniform spherical distribution is the origin.
    np.testing.assert_allclose(dirs.mean(axis=0), 0.0, atol=0.05)
    # Every octant should be populated for a genuinely even spread.
    signs = {tuple(np.sign(d).astype(int)) for d in dirs if np.all(d != 0)}
    assert len({s for s in signs if 0 not in s}) == 8


def test_single_direction_is_allowed():
    assert fibonacci_directions(1).shape == (1, 3)


def test_zero_directions_is_rejected():
    with pytest.raises(ValueError):
        fibonacci_directions(0)


def test_index_colour_round_trip():
    idx = np.array([0, 1, 255, 256, 65535, 65536, 1_000_000])
    np.testing.assert_array_equal(color_to_index(index_to_color(idx)), idx)


def test_background_black_decodes_to_minus_one():
    """Reserving 0 is what stops an unrendered pixel from being read as face 0."""
    assert color_to_index(np.array([0, 0, 0])) == -1


def test_face_zero_is_not_background():
    assert not np.array_equal(index_to_color(0), np.array([0, 0, 0]))
    assert color_to_index(index_to_color(0)) == 0


def test_index_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        index_to_color(MAX_FACES)
    with pytest.raises(ValueError):
        index_to_color(-1)


def test_colour_to_index_rejects_bad_shape():
    with pytest.raises(ValueError):
        color_to_index(np.zeros((4, 4, 4), dtype=np.uint8))


def test_visible_faces_marks_only_what_was_seen():
    buffer = np.zeros((2, 2, 3), dtype=np.uint8)
    buffer[0, 0] = index_to_color(3)
    buffer[1, 1] = index_to_color(7)
    seen = visible_faces([buffer], face_count=10)
    assert seen.sum() == 2
    assert seen[3] and seen[7]


def test_visible_faces_unions_across_buffers():
    """A face seen from any one direction must survive, even if hidden in all others."""
    a = np.zeros((1, 1, 3), dtype=np.uint8)
    a[0, 0] = index_to_color(1)
    b = np.zeros((1, 1, 3), dtype=np.uint8)
    b[0, 0] = index_to_color(4)
    seen = visible_faces([a, b], face_count=6)
    assert seen[1] and seen[4]
    assert seen.sum() == 2


def test_all_background_culls_everything():
    seen = visible_faces([np.zeros((3, 3, 3), dtype=np.uint8)], face_count=5)
    assert not seen.any()


def test_indices_beyond_the_mesh_are_ignored():
    """A stray colour from compression or dithering must not index out of bounds."""
    buffer = np.zeros((1, 2, 3), dtype=np.uint8)
    buffer[0, 0] = index_to_color(2)
    buffer[0, 1] = index_to_color(9999)
    seen = visible_faces([buffer], face_count=5)
    assert seen[2] and seen.sum() == 1


def test_no_buffers_culls_everything():
    assert not visible_faces([], face_count=4).any()


def test_report_counts_correctly():
    seen = np.array([True, False, True, True])
    assert "3 of 4" in cull_report(seen) and "75.0%" in cull_report(seen)


# --- the inner wall: faces from which no ray reaches the outside -----------------------


def _shells(*radii):
    """A cast against concentric spheres: True when the ray meets one of them."""

    def cast(origin, direction, _face):
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        b = o @ d
        for r in radii:
            disc = b * b - (o @ o - r * r)
            if disc >= 0 and (-b + np.sqrt(disc)) > 1e-9:
                return True
        return False

    return cast


def _on_sphere(radius, count=40):
    return fibonacci_directions(count) * radius


def test_a_wall_inside_another_never_sees_out_and_the_outer_one_does():
    """Pixal3D's double wall: the inner one is hidden from every direction, the outer is not."""
    dirs = fibonacci_directions(16)
    cast = _shells(1.0, 0.97)
    centres = np.vstack([_on_sphere(1.0), _on_sphere(0.97)])
    hidden = hidden_from_rays(centres, dirs, cast, eps=1e-6)
    assert not hidden[:40].any(), "the outer wall is what the camera sees"
    assert hidden[40:].all(), "the wall inside it is never seen from anywhere"


def test_a_face_is_kept_as_soon_as_one_ray_escapes():
    """A seen face costs one ray when its first direction gets out, so the test stays fast."""
    calls = []

    def cast(origin, direction, face):
        calls.append(face)
        return False

    hidden = hidden_from_rays(np.zeros((3, 3)), fibonacci_directions(16), cast, eps=1e-6)
    assert not hidden.any()
    assert calls == [0, 1, 2]


def test_faces_start_at_different_directions():
    """Each face starts at its own direction, so a whole region never tries the same blocked
    ray first."""
    first = {}

    def cast(origin, direction, face):
        first.setdefault(face, tuple(np.round(direction, 6)))
        return False

    hidden_from_rays(np.zeros((4, 3)), fibonacci_directions(4), cast, eps=1e-6)
    assert len(set(first.values())) == 4


def test_the_ray_starts_just_off_the_face_along_its_direction():
    seen = []

    def cast(origin, direction, face):
        seen.append(np.asarray(origin) - np.asarray([1.0, 2.0, 3.0]))
        return True

    dirs = fibonacci_directions(2)
    hidden_from_rays(np.array([[1.0, 2.0, 3.0]]), dirs, cast, eps=0.01)
    assert np.allclose(seen[0], dirs[0] * 0.01)


def test_no_faces_gives_an_empty_answer():
    assert hidden_from_rays(np.zeros((0, 3)), fibonacci_directions(4), _shells(1.0), 1e-6).shape == (0,)



def _sphere(radius, count=4000, centre=(0.0, 0.0, 0.0)):
    return fibonacci_directions(count) * radius + np.asarray(centre)


def _box(lo, hi, step):
    """Dense points on the six sides of an axis-aligned box."""
    axes = [np.arange(lo[i], hi[i] + step / 2, step) for i in range(3)]
    pts = []
    for i in range(3):
        j, k = [a for a in range(3) if a != i]
        gj, gk = np.meshgrid(axes[j], axes[k], indexing="ij")
        for side in (lo[i], hi[i]):
            p = np.zeros((gj.size, 3))
            p[:, i], p[:, j], p[:, k] = side, gj.ravel(), gk.ravel()
            pts.append(p)
    return np.concatenate(pts)


def test_the_hollow_inside_a_double_wall_is_sealed_air():
    """Pixal3D's balloon: the air inside the inner wall is cut off from the outside."""
    points = np.concatenate([_sphere(1.0, 20000), _sphere(0.85, 15000)])
    grid = sealed_air(points, voxel=0.05)
    assert in_sealed_air(grid, np.array([[0.0, 0.0, 0.0], [0.0, 0.7, 0.0]])).all()
    assert not in_sealed_air(grid, np.array([[0.0, 1.2, 0.0], [1.3, 0.0, 0.0]])).any()


def test_the_air_under_a_cape_is_open_to_the_outside():
    """A sheet hanging in front of a body leaves a gap the outside reaches from below:
    the sheet's inner side faces open air, so it must stay, however few rays get out."""
    body = _box((-0.3, -0.2, 0.0), (0.3, 0.2, 1.6), 0.01)
    cape = _box((-0.35, 0.3, 0.4), (0.35, 0.33, 1.7), 0.01)    # 10 cm in front of the body
    roof = _box((-0.35, 0.18, 1.6), (0.35, 0.33, 1.7), 0.01)   # closed at the shoulders
    grid = sealed_air(np.concatenate([body, cape, roof]), voxel=0.02)
    gap = np.array([[0.0, 0.25, 1.2], [0.0, 0.25, 0.6]])
    assert not in_sealed_air(grid, gap).any()
    assert in_sealed_air(grid, np.array([[0.0, 0.0, 0.8]])).all()   # the body's own inside


def test_closing_seals_a_slit_but_not_a_wide_opening():
    """A cup whose rim leaves a slit one voxel wide is sealed once the slit is closed; one
    with a wide mouth stays open, so a coat's open hem is never mistaken for a hollow."""
    shell = _box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), 0.01)
    slit = shell[~((np.abs(shell[:, 0]) < 0.015) & (shell[:, 2] > 0.49))]      # 3 cm slit on top
    mouth = shell[~((np.abs(shell[:, 0]) < 0.3) & (np.abs(shell[:, 1]) < 0.3) & (shell[:, 2] > 0.49))]
    centre = np.array([[0.0, 0.0, 0.0]])
    assert not in_sealed_air(sealed_air(slit, voxel=0.02), centre).any()
    assert in_sealed_air(sealed_air(slit, voxel=0.02, close=1), centre).all()
    assert not in_sealed_air(sealed_air(mouth, voxel=0.02, close=1), centre).any()


def test_sealed_air_is_found_next_to_the_wall_after_closing():
    """Closing thickens the walls on the grid; the sealed hollow is grown back, so a point
    a voxel and a half inside the inner wall still reads as sealed."""
    points = np.concatenate([_sphere(1.0, 20000), _sphere(0.85, 15000)])
    grid = sealed_air(points, voxel=0.05, close=1)
    assert in_sealed_air(grid, np.array([[0.0, 0.85 - 0.075, 0.0]])).all()


def test_points_off_the_grid_are_not_sealed():
    grid = sealed_air(_sphere(1.0, 20000), voxel=0.05)
    assert not in_sealed_air(grid, np.array([[5.0, 5.0, 5.0], [-9.0, 0.0, 0.0]])).any()


def test_sealed_air_wants_points_and_a_positive_voxel():
    with pytest.raises(ValueError):
        sealed_air(np.zeros((0, 3)), voxel=0.05)
    with pytest.raises(ValueError):
        sealed_air(_sphere(1.0, 100), voxel=0.0)


def test_a_flat_pocket_behind_a_cape_is_not_a_hollow_but_the_body_is():
    """Both are sealed when the cape lies close; only the body's hollow is a balloon's inside.
    The pocket is flat like a sheet (wide, a few cells deep), so its walls must stay: a
    run or a fall lifts the cape and shows them."""
    body = _box((-0.3, -0.2, 0.0), (0.3, 0.2, 1.6), 0.01)
    pocket = _box((-0.3, 0.2, 0.3), (0.3, 0.26, 1.5), 0.01)   # a closed 6 cm gap on the back
    grid = hollows_only(sealed_air(np.concatenate([body, pocket]), voxel=0.02))
    assert in_sealed_air(grid, np.array([[0.0, 0.0, 0.8]])).all()
    assert not in_sealed_air(grid, np.array([[0.0, 0.23, 0.9]])).any()


def test_a_limb_is_a_hollow_too():
    """An arm's hollow is a tube, not a sheet: its inner wall goes like the body's."""
    arm = _box((-0.06, -0.06, 0.0), (0.06, 0.06, 0.6), 0.005)
    grid = hollows_only(sealed_air(arm, voxel=0.01))
    assert in_sealed_air(grid, np.array([[0.0, 0.0, 0.3], [0.03, 0.0, 0.1]])).all()


def test_the_wall_band_of_a_hollow_is_kept_with_it():
    """The test points sit a cell and a half off the wall, where the coarse search cannot
    see; the hollow is grown back to the wall."""
    points = np.concatenate([_sphere(1.0, 20000), _sphere(0.85, 15000)])
    grid = hollows_only(sealed_air(points, voxel=0.05, close=1))
    assert in_sealed_air(grid, np.array([[0.0, 0.85 - 0.075, 0.0]])).all()


def test_the_first_free_cell_answers_not_the_wall_beside_the_face():
    """A wall drawn three cells thick: a step of a cell and a half from its inner face stays in
    the wall, the walk finds the hollow beyond it; outward it finds the open air."""
    radii = np.linspace(0.85, 0.95, 6)                      # a thick band of points: a thick wall
    band = np.concatenate([_sphere(r, 15000) for r in radii] + [_sphere(1.0, 20000)])
    grid = sealed_air(band, voxel=0.05)
    face = np.array([[0.0, 0.85, 0.0]])
    assert first_air_sealed(grid, face, np.array([[0.0, -1.0, 0.0]])).all()
    assert not first_air_sealed(grid, np.array([[0.0, 1.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])).any()


# --- dense surface samples (for sealed_air on a coarse game mesh) -----------------------------------

def test_a_dense_surface_samples_a_triangle_no_further_apart_than_the_step():
    tris = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]]])   # longest edge 1
    points = dense_surface(tris, 0.25)
    assert len(points) == 15                       # k = 4 steps along the longest edge: (k + 1)(k + 2) / 2
    assert {(0.0, 0.0), (1.0, 0.0), (0.5, 0.8)} <= {tuple(np.round(p[:2], 9)) for p in points}
    gaps = np.linalg.norm(points[:, None] - points[None], axis=2) + np.eye(len(points)) * 9
    assert gaps.min(axis=1).max() <= 0.25 + 1e-9  # every point has a neighbour within the step


def test_a_small_triangle_keeps_its_corners_only():
    tris = np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]])
    assert len(dense_surface(tris, 0.25)) == 3


def test_dense_surface_of_nothing_is_empty():
    assert dense_surface(np.zeros((0, 3, 3)), 0.25).shape == (0, 3)


def test_face_samples_are_the_centre_and_three_points_toward_the_corners():
    tris = np.array([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])
    samples = face_samples(tris, 0.5)
    assert samples.shape == (1, 4, 3)
    assert samples[0, 0] == pytest.approx([1.0, 1.0, 0.0])          # the centre first
    assert samples[0, 1] == pytest.approx([0.5, 0.5, 0.0])          # halfway to the first corner
    assert samples[0, 2] == pytest.approx([2.0, 0.5, 0.0])
    assert samples[0, 3] == pytest.approx([0.5, 2.0, 0.0])


def test_face_samples_of_nothing_is_empty():
    assert face_samples(np.zeros((0, 3, 3))).shape == (0, 4, 3)


# --- inner wall poking out past the outer surface (#141) ---------------------------------------

def _quad(a, b, c, d):
    """Two triangles a-b-c and a-c-d: wound counter-clockwise, so they face (b - a) x (c - a)."""
    return [[a, b, c], [a, c, d]]


def _box_tris(lo, hi):
    """A closed box with outward-facing triangles."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    p = {(i, j, k): (x1 if i else x0, y1 if j else y0, z1 if k else z0) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    faces = [((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
             ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
             ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))]
    tris = []
    for q in faces:
        tris += _quad(*[p[c] for c in q])
    return tris


def _caster(tris):
    """Plain ray casts against triangles (Moller-Trumbore), for `poking_out`: whether a ray meets anything,
    and the first hit's distance and the hit face's normal (by its winding)."""
    tris = np.asarray(tris, dtype=float)
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    def first(origin, direction, face, reach=np.inf):
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        pv = np.cross(d, e2)
        det = (e1 * pv).sum(1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - tris[:, 0]
        u = (tv * pv).sum(1) * inv
        qv = np.cross(tv, e1)
        v = (qv * d).sum(1) * inv
        t = (qv * e2).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9) & (t <= reach)
        hit[face] = False
        if not hit.any():
            return None
        k = int(np.flatnonzero(hit)[np.argmin(t[hit])])
        return float(t[k]), normals[k]

    def cast(origin, direction, face):
        return first(origin, direction, face) is not None

    return tris, normals, cast, first


def _poking(tris, near=0.05):
    tris, normals, cast, first = _caster(tris)
    return poking_out(face_samples(tris), normals, fibonacci_directions(64), cast, first, near, eps=1e-6)


def test_an_inner_wall_piece_lying_over_the_outer_surface_and_facing_it_is_found():
    """After a thinning a piece of the inner wall pokes past the outer surface: its back faces the sky, its front
    faces the surface a few millimetres under it. A game culls it, but a sky-light bake lets it shade that surface."""
    body = _box_tris((0, 0, 0), (1, 1, 1))
    piece = _quad((1.01, 0.4, 0.4), (1.01, 0.4, 0.6), (1.01, 0.6, 0.6), (1.01, 0.6, 0.4))  # faces -x, toward the box
    poke = _poking(body + piece)
    assert poke[len(body):].all()
    assert not poke[:len(body)].any(), "the outer surface itself is kept"


def test_a_sheet_facing_away_from_the_surface_is_kept():
    """A sheet over the surface that faces outward is what the camera sees (a decal, a strap)."""
    body = _box_tris((0, 0, 0), (1, 1, 1))
    sheet = _quad((1.01, 0.4, 0.4), (1.01, 0.6, 0.4), (1.01, 0.6, 0.6), (1.01, 0.4, 0.6))  # faces +x
    assert not _poking(body + sheet).any()


def test_an_outer_face_wound_the_wrong_way_is_kept():
    """A flipped outer face also shows its back to the sky, but its front looks into the body and meets the back of
    the other side: dropping it would open a hole for a game that draws both sides."""
    slab = _box_tris((0, 0, 0), (1, 1, 0.03))
    top = [i for i, t in enumerate(slab) if all(abs(p[2] - 0.03) < 1e-9 for p in t)]
    for i in top:
        slab[i] = slab[i][::-1]
    assert not _poking(slab).any()


def test_the_underside_of_a_cape_over_the_back_is_kept():
    """A cape's underside faces the body close by, but the cape's own top is behind it: it never sees the sky."""
    body = _box_tris((0, 0, 0), (1, 1, 1))
    cape = _box_tris((1.01, 0.1, 0.1), (1.03, 0.9, 0.9))
    poke = _poking(body + cape)
    assert not poke.any()


def test_a_flap_whose_tip_a_camera_sees_past_the_edge_is_kept():
    """A piece over a cape's back whose tip hangs past the cape's edge: from in front, past the edge, a camera sees
    that tip, so the piece stays though its middle lies over the surface (dropping such faces opened holes)."""
    body = _box_tris((0, 0, 0), (1, 1, 1))
    flap = _quad((1.01, 0.6, 0.4), (1.01, 0.6, 0.6), (1.01, 1.1, 0.6), (1.01, 1.1, 0.4))  # faces -x, past y = 1
    tris, normals, cast, first = _caster(body + flap)
    samples = face_samples(tris)
    assert (samples[len(body):, 0, 1] < 1.0).all(), "both triangles' middles lie over the box"
    assert not poking_out(samples, normals, fibonacci_directions(64), cast, first, 0.05, eps=1e-6)[len(body):].any()


def test_a_piece_seen_at_an_angle_past_a_strap_is_kept():
    """A piece over a narrow strap: straight down it meets the strap, but at an angle it is seen past the strap's
    edge, so a camera draws it and it stays."""
    strap = _box_tris((0, 0.48, 0), (1, 0.52, 1))
    piece = _quad((1.01, 0.47, 0.4), (1.01, 0.47, 0.6), (1.01, 0.53, 0.6), (1.01, 0.53, 0.4))  # faces -x
    assert not _poking(strap + piece)[len(strap):].any()


def test_a_piece_seen_only_at_a_grazing_angle_is_kept():
    """Hair strands lie over one another, each facing the next: a strand at the edge of the hair is seen from behind
    only at a grazing angle, and dropping it opened a hole the arm showed through (the Daemonenrufer's v15j, #141).
    Any front direction that reaches the outside keeps a face, however flat."""
    strap = _box_tris((0, 0.45, 0), (1, 0.55, 1))
    piece = _quad((1.01, 0.47, 0.4), (1.01, 0.47, 0.6), (1.01, 0.53, 0.6), (1.01, 0.53, 0.4))  # faces -x
    assert not _poking(strap + piece)[len(strap):].any()


def test_a_piece_further_off_than_near_is_kept():
    body = _box_tris((0, 0, 0), (1, 1, 1))
    piece = _quad((1.2, 0.4, 0.4), (1.2, 0.4, 0.6), (1.2, 0.6, 0.6), (1.2, 0.6, 0.4))
    assert not _poking(body + piece, near=0.05).any()


def test_poking_out_of_nothing_is_empty():
    assert poking_out(np.zeros((0, 4, 3)), np.zeros((0, 3)), fibonacci_directions(8), None, None, 0.1,
                      eps=1e-6).shape == (0,)


# --- roofs: what may not shade the surface in the sky-light bake (TODO 103) -----------------------

def _roofs(tris, near=0.05):
    tris, normals, cast, first = _caster(tris)
    return shading_roofs(face_samples(tris), normals, fibonacci_directions(64), cast, first, near, eps=1e-6)


def test_a_piece_over_the_surface_is_a_roof_even_where_a_camera_sees_its_front():
    """The Frostwaechter's grey fur (TODO 103): pieces the strict rule keeps, because a camera sees their front at
    some angle, still shade the fur under them in the sky-light bake. As roofs they stay in the mesh but cast no shade."""
    body = _box_tris((0, 0, 0), (1, 1, 1))
    piece = _quad((1.01, 0.4, 0.4), (1.01, 0.4, 0.6), (1.01, 0.6, 0.6), (1.01, 0.6, 0.4))
    roofs = _roofs(body + piece)
    assert roofs[len(body):].all() and not roofs[:len(body)].any()
    flap = _quad((1.01, 0.6, 0.4), (1.01, 0.6, 0.6), (1.01, 1.1, 0.6), (1.01, 1.1, 0.4))   # its tip is seen past the edge
    assert _roofs(body + flap)[len(body):].all()
    strap = _box_tris((0, 0.48, 0), (1, 0.52, 1))
    over = _quad((1.01, 0.47, 0.4), (1.01, 0.47, 0.6), (1.01, 0.53, 0.6), (1.01, 0.53, 0.4))
    assert _roofs(strap + over)[len(strap):].all()


def test_what_a_roof_is_not():
    body = _box_tris((0, 0, 0), (1, 1, 1))
    sheet = _quad((1.01, 0.4, 0.4), (1.01, 0.6, 0.4), (1.01, 0.6, 0.6), (1.01, 0.4, 0.6))   # faces outward
    assert not _roofs(body + sheet).any()
    cape = _box_tris((1.01, 0.1, 0.1), (1.03, 0.9, 0.9))                                  # its underside: no sky behind
    assert not _roofs(body + cape).any()
    far = _quad((1.2, 0.4, 0.4), (1.2, 0.4, 0.6), (1.2, 0.6, 0.6), (1.2, 0.6, 0.4))
    assert not _roofs(body + far, near=0.05).any()
    slab = _box_tris((0, 0, 0), (1, 1, 0.03))
    for i, t in enumerate(slab):
        if all(abs(p[2] - 0.03) < 1e-9 for p in t):
            slab[i] = t[::-1]
    assert not _roofs(slab).any()
    assert shading_roofs(np.zeros((0, 4, 3)), np.zeros((0, 3)), fibonacci_directions(8), None, None, 0.1,
                         eps=1e-6).shape == (0,)


# --- a drop never opens an edge a camera sees (2026-11-23) ------------------------------------------------------------

def _strip(count, z=0.0):
    """COUNT unit quads in a row along x at height Z: (triangles (2 COUNT, 3, 3), the pairs of triangles sharing an
    edge, those edges' middles). Quad k is triangles 2k and 2k + 1."""
    tris, pairs, middles = [], [], []
    for k in range(count):
        a, b, c, d = (k, 0, z), (k + 1, 0, z), (k + 1, 1, z), (k, 1, z)
        tris += [[a, b, c], [a, c, d]]
        pairs.append((2 * k, 2 * k + 1))
        middles.append(((k + k + 1) / 2, 0.5, z))
        if k:
            pairs.append((2 * k - 2, 2 * k + 1))          # the edge between quad k - 1 and quad k
            middles.append((k, 0.5, z))
    return np.asarray(tris, float), np.asarray(pairs), np.asarray(middles, float)


def _escapes(tris, directions):
    """`seen` for `keep_seen_rims`: from each point, does a ray along one of DIRECTIONS meet none of the kept TRIS?"""
    tris = np.asarray(tris, float)
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]

    def meets(o, d, keep):
        pv = np.cross(d, e2[keep])
        det = (e1[keep] * pv).sum(1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - tris[keep, 0]
        u = (tv * pv).sum(1) * inv
        qv = np.cross(tv, e1[keep])
        v = (qv * d).sum(1) * inv
        t = (qv * e2[keep]).sum(1) * inv
        return bool((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)).any())

    def seen(points, doomed):
        keep = ~np.asarray(doomed, bool)
        return np.array([any(not meets(p + d * 1e-6, d, keep) for d in directions) for p in np.asarray(points)])

    return seen


def test_a_drop_whose_rim_a_camera_sees_is_put_back():
    """A strip lying in the open: dropping its middle would open two edges anyone sees, so the middle stays."""
    tris, pairs, middles = _strip(3)
    doomed = np.zeros(len(tris), bool)
    doomed[2:4] = True
    kept, back = keep_seen_rims(pairs, middles, doomed, _escapes(tris, fibonacci_directions(32)))
    assert not kept.any() and back == 2


def test_a_drop_whose_rim_lies_inside_the_figure_stays():
    """The same strip inside a closed box: no ray from its rims gets out, so the middle still goes."""
    tris, pairs, middles = _strip(3, z=0.5)
    box = np.asarray(_box_tris((-1.0, -1.0, -1.0), (4.0, 2.0, 2.0)), float)
    doomed = np.zeros(len(tris) + len(box), bool)
    doomed[2:4] = True
    kept, back = keep_seen_rims(pairs, middles, doomed, _escapes(np.concatenate([tris, box]),
                                                                fibonacci_directions(32)))
    assert kept[2:4].all() and back == 0


def test_rims_opened_by_what_was_put_back_are_asked_again():
    """Five quads in the open with the middle three doomed: each round puts back the triangles on the rims the last one
    opened, until the strip is whole; a single round puts back only the two next to kept quads."""
    tris, pairs, middles = _strip(5)
    doomed = np.zeros(len(tris), bool)
    doomed[2:8] = True
    seen = _escapes(tris, fibonacci_directions(32))
    kept, back = keep_seen_rims(pairs, middles, doomed, seen)
    assert not kept.any() and back == 6
    once, back = keep_seen_rims(pairs, middles, doomed, seen, rounds=1)
    assert back == 2 and np.flatnonzero(once).tolist() == [2, 4, 5, 7]


def test_keep_seen_rims_asks_nothing_when_nothing_is_doomed():
    calls = []
    tris, pairs, middles = _strip(2)
    kept, back = keep_seen_rims(pairs, middles, np.zeros(len(tris), bool), lambda p, d: calls.append(1) or [])
    assert not kept.any() and back == 0 and not calls
