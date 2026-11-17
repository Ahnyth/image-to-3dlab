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
    sealed_air,
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
