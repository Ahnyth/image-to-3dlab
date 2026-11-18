#!/usr/bin/env python3
"""Pure helpers for visibility-based face culling: keep only what is seen from outside.

A generated mesh arrives shattered -- the thorn-knot Snag has 664 components and 46,253
boundary edges, most of it loose flaps and fragments produced by sign flips where the
decoder could not resolve two nearby surfaces. Every repair that reasons about *distance*
fails on it: merge-by-distance has nothing to weld to, and voxel remeshing either fuses
neighbouring coils (coarse) or faithfully rebuilds the shards as slabs (fine). Both were
measured and both failed.

Visibility asks a different question. Stand outside, look from every direction, and keep
only the triangles you actually saw. That is:

- **topology-preserving** -- it only deletes faces, never moves a vertex, so creases stay
  exactly as sharp as they were;
- **scale-free** -- no threshold, no voxel size, nothing to tune wrong;
- **incapable of fusing anything** -- it never asks whether two surfaces are close, so
  adjacent coils cannot blend into each other. That is the failure mode of every
  alternative, and this method is immune to it by construction.

**Honest limit:** an outward-facing flap floating just above the true surface *is* seen,
so it survives. Expect this to remove interior junk, enclosed fragments and back-facing
debris -- a large fraction, not all of it. It is the first step of a cleanup, not the
whole cleanup.

The rendering half lives in ``blender_visibility_cull.py``; everything here is pure so it
can be tested without Blender.
"""

from __future__ import annotations

import numpy as np

# 8 bits per channel over three channels addresses 16.7M faces, far beyond any mesh we
# generate. Index 0 is reserved so that "nothing here" and "face 0" cannot be confused.
_CHANNEL = 256
MAX_FACES = _CHANNEL**3 - 1


def fibonacci_directions(count: int) -> np.ndarray:
    """`count` unit vectors spread evenly over the sphere.

    A latitude/longitude grid clumps at the poles and would over-sample whatever happens
    to point up. The Fibonacci spiral gives near-uniform coverage, so every part of the
    surface gets a comparable chance of being seen.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    i = np.arange(count, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / count)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )


def index_to_color(index: np.ndarray | int) -> np.ndarray:
    """Encode 0-based face indices as RGB bytes, offset by one so 0 means background."""
    index = np.asarray(index)
    if np.any(index < 0) or np.any(index >= MAX_FACES):
        raise ValueError(f"face index out of encodable range 0..{MAX_FACES - 1}")
    shifted = index.astype(np.int64) + 1
    return np.stack(
        [
            shifted % _CHANNEL,
            (shifted // _CHANNEL) % _CHANNEL,
            (shifted // (_CHANNEL * _CHANNEL)) % _CHANNEL,
        ],
        axis=-1,
    ).astype(np.uint8)


def color_to_index(rgb: np.ndarray) -> np.ndarray:
    """Decode RGB bytes back to face indices. Background (black) decodes to -1."""
    rgb = np.asarray(rgb).astype(np.int64)
    if rgb.shape[-1] != 3:
        raise ValueError("expected a trailing RGB axis of size 3")
    packed = rgb[..., 0] + rgb[..., 1] * _CHANNEL + rgb[..., 2] * _CHANNEL * _CHANNEL
    return packed - 1


def visible_faces(buffers: list[np.ndarray], face_count: int) -> np.ndarray:
    """Boolean mask of faces observed in at least one face-ID buffer.

    Each buffer is an HxWx3 uint8 image whose pixels encode face indices.
    """
    if face_count < 0:
        raise ValueError("face_count must be non-negative")
    seen = np.zeros(face_count, dtype=bool)
    for buffer in buffers:
        idx = color_to_index(buffer).ravel()
        idx = idx[(idx >= 0) & (idx < face_count)]
        seen[idx] = True
    return seen


def cull_report(seen: np.ndarray) -> str:
    total = int(seen.size)
    kept = int(seen.sum())
    pct = 100.0 * kept / total if total else 0.0
    return f"{kept:,} of {total:,} faces visible ({pct:.1f}%); culling {total - kept:,}"


def hidden_from_rays(centres, directions, cast, eps: float) -> np.ndarray:
    """Boolean mask of the faces from whose centre no ray reaches the outside.

    The render above needs every face to win a pixel, which a dense mesh cannot: a raw
    Pixal3D figure has about 900,000 faces, most smaller than a pixel at 512. This asks each
    face instead. From its centre, rays go out along `directions` (a Fibonacci sphere, no
    hemisphere: the winding of a generated mesh cannot be trusted), and the face counts as
    seen as soon as one of them meets nothing. A face that every ray brings back to the
    mesh is enclosed: the inner wall of a generated double-walled surface, which no camera
    and no game ever shows.

    `cast(origin, direction, face)` answers whether the ray meets the mesh; it is Blender's
    BVH in ``blender_retopo_bake.py`` and a plain function in the tests. Each face starts
    at a different direction, so a crease does not try the same blocked ray everywhere,
    and a seen face costs one ray when its first one gets out. Rays start `eps` off the
    face along their own direction, so they do not stop on the face they leave.
    """
    count = len(centres)
    hidden = np.zeros(count, dtype=bool)
    total = len(directions)
    for face in range(count):
        centre = centres[face]
        for k in range(total):
            direction = directions[(face + k) % total]
            if not cast(centre + direction * eps, direction, face):
                break
        else:
            hidden[face] = True
    return hidden


def _grow(mask: np.ndarray) -> np.ndarray:
    """`mask` grown by one voxel to its six face neighbours (no wrap at the grid's edge)."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def sealed_air(points, voxel: float, close: int = 0) -> dict:
    """The air around a surface that the outside cannot reach, on a grid of `voxel`.

    Rays alone cannot tell the inner wall of a double-walled figure from the inside of a
    cape or a coat's lining: in the rest pose both are enclosed, and no ray from them gets
    out. A pose tells them apart (the cape lifts, the lining shows), and so does the air:
    the hollow inside Pixal3D's balloon is sealed, while the gap behind a cape opens to the
    outside at its hem. So the surface is put on a grid (`points` sampled on it closer than a
    voxel, so the walls have no gaps the flood could slip through), the outside air is
    flooded in from the grid's border through face neighbours, and what it never reaches
    is sealed. `close` thickens the walls by that many voxels first, so a slit up to twice
    as wide is shut (a generated wall can leak at a seam); the sealed air is then grown
    back by the same amount into the free air, so a point just inside the wall still reads
    as sealed. Returns the grid for `in_sealed_air`.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(pts):
        raise ValueError("sealed_air needs points on the surface")
    if not voxel > 0:
        raise ValueError(f"voxel must be positive, got {voxel}")
    margin = 2 + close
    origin = pts.min(axis=0) - margin * voxel
    shape = np.floor((pts.max(axis=0) - origin) / voxel).astype(int) + margin + 1
    wall = np.zeros(shape, dtype=bool)
    cells = np.floor((pts - origin) / voxel).astype(int)
    wall[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    thick = wall
    for _ in range(close):
        thick = _grow(thick)
    air = ~thick
    outside = np.zeros(shape, dtype=bool)
    for axis in range(3):
        index = [slice(None)] * 3
        for end in (0, -1):
            index[axis] = end
            outside[tuple(index)] = True
    outside &= air
    count = int(outside.sum())
    for _ in range(int(shape.sum()) * 4):         # far more steps than any path needs
        outside = _grow(outside) & air
        now = int(outside.sum())
        if now == count:
            break
        count = now
    sealed = air & ~outside
    free = ~wall
    for _ in range(close):
        sealed = _grow(sealed) & free & ~outside
    return {"origin": origin, "voxel": float(voxel), "sealed": sealed, "outside": outside, "wall": wall}


def _shrink(mask: np.ndarray) -> np.ndarray:
    """`mask` less every cell with a face neighbour outside it (the grid's border counts as outside)."""
    out = mask.copy()
    out[1:] &= mask[:-1]
    out[:-1] &= mask[1:]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    out[:, :, 1:] &= mask[:, :, :-1]
    out[:, :, :-1] &= mask[:, :, 1:]
    out[0], out[-1], out[:, 0], out[:, -1], out[:, :, 0], out[:, :, -1] = False, False, False, False, False, False
    return out


def _parts(mask: np.ndarray) -> np.ndarray:
    """A label per cell of `mask` for its connected part (face neighbours), 0 outside it.

    Each cell starts with its own number and takes the smallest of its neighbours' until nothing
    changes; following each label to the label of the cell it names (pointer jumping) keeps the
    number of rounds near the logarithm of a part's length instead of the length itself.
    """
    flat_index = np.arange(1, mask.size + 1, dtype=np.int64).reshape(mask.shape)
    label = np.where(mask, flat_index, 0)
    top = np.int64(mask.size + 1)
    for _ in range(int(sum(mask.shape)) * 2):
        cur = np.where(mask, label, top)
        low = cur.copy()
        low[1:] = np.minimum(low[1:], cur[:-1])
        low[:-1] = np.minimum(low[:-1], cur[1:])
        low[:, 1:] = np.minimum(low[:, 1:], cur[:, :-1])
        low[:, :-1] = np.minimum(low[:, :-1], cur[:, 1:])
        low[:, :, 1:] = np.minimum(low[:, :, 1:], cur[:, :, :-1])
        low[:, :, :-1] = np.minimum(low[:, :, :-1], cur[:, :, 1:])
        low = np.where(mask, low, 0)
        flat = low.ravel()
        jumped = np.where(mask, flat[np.maximum(low, 1) - 1].reshape(mask.shape), 0)
        jumped = np.minimum(low, np.where(mask, jumped, top))
        if np.array_equal(jumped, label):
            break
        label = jumped
    return label


# A sealed part is a hollow (the inside of Pixal3D's balloon: the body, a limb, the head) when it
# is at least this deep, in cells of the half-resolution grid (about 4 grid cells)...
HOLLOW_DEPTH = 2
# ...and compact: its volume over its depth cubed stays below this. A flat pocket (the gap behind a
# cape lying close to the back, under a coat or a plate) is wide and shallow; on two knights the
# cape's pocket measured 213, every body part 30 or less.
HOLLOW_SHEET = 100.0


def hollows_only(grid: dict) -> dict:
    """`grid` with its sealed air narrowed to the hollows: compact parts deep enough to be a body's inside.

    Closing the walls also seals a pocket where a cape or a coat lies close to the body, and the
    walls of that pocket are the cape's lining, which motion shows. So the sealed air is split into
    its connected parts (on a half-resolution grid, a cell sealed only where all eight below it are,
    so parts never merge), each part's depth is found by shrinking it until it vanishes, and only
    the deep, compact parts stay; they are grown back to the walls on the full grid.
    """
    sealed = grid["sealed"]
    even = [n + n % 2 for n in sealed.shape]
    padded = np.zeros(even, dtype=bool)
    padded[: sealed.shape[0], : sealed.shape[1], : sealed.shape[2]] = sealed
    half = np.ones([n // 2 for n in even], dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                half &= padded[dx::2, dy::2, dz::2]
    keep_half = np.zeros_like(half)
    if half.any():
        label = _parts(half)
        depth = np.zeros(half.shape, dtype=np.int32)
        inner = half
        while inner.any():
            depth += inner
            inner = _shrink(inner)
        names, which = np.unique(label[half], return_inverse=True)
        volume = np.bincount(which).astype(float)
        deepest = np.zeros(len(names), dtype=np.int32)
        np.maximum.at(deepest, which, depth[half])
        good = (deepest >= HOLLOW_DEPTH) & (volume <= HOLLOW_SHEET * deepest.astype(float) ** 3)
        keep_half[half] = good[which]
    keep = np.repeat(np.repeat(np.repeat(keep_half, 2, 0), 2, 1), 2, 2)[
        : sealed.shape[0], : sealed.shape[1], : sealed.shape[2]] & sealed
    for _ in range(3):
        keep = _grow(keep) & sealed
    return dict(grid, sealed=keep)


def first_air_sealed(grid: dict, points, directions, reach: float = 4.0) -> np.ndarray:
    """For each point, whether the first free cell along its direction lies in sealed air.

    A face sits in a wall drawn a cell or two thick on the grid, so a fixed step beside it may
    land in the wall itself and ask the wall instead of the air. This walks from each point in
    half-cell steps, up to `reach` cells, and answers for the first cell that is not wall; a
    point that meets none answers False.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    dirs = np.asarray(directions, dtype=float).reshape(-1, 3)
    wall, sealed = grid["wall"], grid["sealed"]
    shape = np.array(sealed.shape)
    answer = np.zeros(len(pts), dtype=bool)
    open_ = np.ones(len(pts), dtype=bool)
    for step in range(1, int(reach * 2) + 1):
        at = pts + dirs * (0.5 * step * grid["voxel"])
        cells = np.floor((at - grid["origin"]) / grid["voxel"]).astype(int)
        inside = np.all((cells >= 0) & (cells < shape), axis=1)
        safe = np.clip(cells, 0, shape - 1)
        is_wall = wall[safe[:, 0], safe[:, 1], safe[:, 2]] & inside
        here = open_ & ~is_wall
        answer[here] = (sealed[safe[:, 0], safe[:, 1], safe[:, 2]] & inside)[here]
        open_ &= is_wall
        if not open_.any():
            break
    return answer


def in_sealed_air(grid: dict, points) -> np.ndarray:
    """For each point, whether it lies in the sealed air `sealed_air` found."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    sealed = grid["sealed"]
    cells = np.floor((pts - grid["origin"]) / grid["voxel"]).astype(int)
    inside = np.all((cells >= 0) & (cells < np.array(sealed.shape)), axis=1)
    answer = np.zeros(len(pts), dtype=bool)
    hit = cells[inside]
    answer[inside] = sealed[hit[:, 0], hit[:, 1], hit[:, 2]]
    return answer


def dense_surface(tris, step: float) -> np.ndarray:
    """Points on every triangle (an (F, 3, 3) array of corners) no further apart than `step`.

    `sealed_air` reads the surface from points; a coarse game mesh's corners alone leave holes
    in its air grid, so every hollow leaks and none counts as sealed (on a 22,000-triangle
    figure with a 4.6 mm grid the corners found 0-528 hidden faces next to sealed air; points
    half a cell apart found the body's).
    """
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    if not len(tris):
        return np.zeros((0, 3))
    edge = np.max([np.linalg.norm(tris[:, i] - tris[:, (i + 1) % 3], axis=1) for i in range(3)], axis=0)
    steps = np.maximum(np.ceil(edge / step - 1e-9).astype(int), 1)
    out = []
    for k in np.unique(steps):
        sel = tris[steps == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = i + j <= k
        a, b = (i[keep] / k)[None, :, None], (j[keep] / k)[None, :, None]
        points = sel[:, None, 0] * (1 - a - b) + sel[:, None, 1] * a + sel[:, None, 2] * b
        out.append(points.reshape(-1, 3))
    return np.concatenate(out)


def face_samples(tris, reach: float = 0.66) -> np.ndarray:
    """Four points per triangle (an (F, 3, 3) array of corners): its centre, then `reach` of the way from the
    centre to each corner. (F, 4, 3)

    A face is hidden only if no ray gets out from any of them. The centre alone misjudges a face that is only
    partly covered: after a thinning the inner wall pokes through the outer one here and there, the poking
    sliver is what the eye sees, and its face's centre still lies behind the outer wall (a 20,000-triangle
    figure lost such a sliver on a shin, seen in every pose; asking four points kept it).
    """
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    centre = tris.mean(axis=1)
    toward = centre[:, None] + reach * (tris - centre[:, None])
    return np.concatenate([centre[:, None], toward], axis=1)


def shading_roofs(samples, normals, directions, cast, first_hit, near: float, eps: float,
                  slant: float = 0.08) -> np.ndarray:
    """Boolean mask of the faces that may not shade the surface in a sky-light bake: the first two questions of
    `poking_out` without the third. Such a face looks with its front at the front of another face within `near`, and
    its back sees the sky, so a game that culls back faces shows the surface under it from outside, but the bake
    lets it roof that surface over: grey triangles on the Frostwaechter's white fur where a piece kept for being seen
    at a grazing angle lay over it (TODO 103 of the AssetFurnace Support notes). A roof stays in the mesh; only the
    sky-light bake looks through it."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 2:
        samples = samples[:, None]
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    directions = np.asarray(directions, dtype=float).reshape(-1, 3)
    out = np.zeros(len(normals), dtype=bool)
    for face in range(len(normals)):
        n = normals[face]
        hit = first_hit(samples[face, 0] + n * eps, n, face, near)
        if hit is None or float(np.dot(hit[1], n)) >= 0.0:
            continue
        back = directions[directions @ n < -slant]
        out[face] = any(not cast(p + d * eps, d, face) for p in samples[face] for d in back)
    return out


def poking_out(samples, normals, directions, cast, first_hit, near: float, eps: float,
               slant: float = 0.08) -> np.ndarray:
    """Boolean mask of the faces that are inner wall poking out past the outer surface.

    After a thinning, a few pieces of the inner wall of a generated double-walled surface end up a millimetre or
    two outside the outer one. Their back faces the sky and their front faces the surface under them, so a game
    that culls back faces never draws them, but a sky-light bake sees them as a roof and shades that surface: small
    dark triangles at a cape's hem and on the legs (game_ready, #141). Such a face is found by two questions:

    - its front (by its winding, `normals`) looks at the *front* of another face within `near` along its normal
      (`first_hit(origin, direction, face, reach)` gives the first hit's distance and that face's normal, or None);
    - from at least one of its `samples` ((F, K, 3), `face_samples`) a ray into its back half of `directions`
      reaches the outside (`cast(origin, direction, face)` answers whether a ray meets the mesh);
    - from none of its samples does a ray into its front half reach the outside: no camera ever draws it.

    The first keeps what the camera sees (a face looking outward meets nothing) and a flipped outer face (it looks
    into the body and meets the back of the other side); the second keeps a cape's underside (the cape's own top
    is behind it) and the inner wall still inside the body; the third keeps everything a camera can draw, however
    flat it sees it: hair strands lie over one another, each facing the next, and a strand at the hair's edge seen
    at a grazing angle passed a looser test and left a hole when dropped. Directions within `slant` of the face's
    plane belong to neither half.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 2:                    # one point per face
        samples = samples[:, None]
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    directions = np.asarray(directions, dtype=float).reshape(-1, 3)
    count = len(normals)
    out = np.zeros(count, dtype=bool)
    for face in range(count):
        n = normals[face]
        centre = samples[face, 0]
        hit = first_hit(centre + n * eps, n, face, near)
        if hit is None or float(np.dot(hit[1], n)) >= 0.0:
            continue
        ahead = directions[directions @ n > slant]
        if any(not cast(p + d * eps, d, face) for p in samples[face] for d in ahead):
            continue
        back = directions[directions @ n < -slant]
        if not len(back):
            continue
        out[face] = any(not cast(p + back[(face + k) % len(back)] * eps, back[(face + k) % len(back)], face)
                        for p in samples[face] for k in range(len(back)))
    return out
