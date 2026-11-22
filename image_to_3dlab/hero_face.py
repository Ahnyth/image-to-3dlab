"""Hero face: a figure's face built again on its own and set into the figure, as pure functions.

A face is a few dozen pixels of a full-body picture, and Pixal3D gives it few voxels. On the Zauberer a hair strand
that is thin in the picture came out fat and a few centimetres in front of his right eye, and the paint step then
painted the eye's glow onto the strand (2026-11-22). Built alone from an upscaled crop, the same face fills Pixal3D's
whole box and keeps both eyes free. The Hero route (`hero.py`) swaps the whole head at the neck, and a big hat or a
high collar does not survive that: loose slabs at the hat's sides, the face seated deeper under the brim, and at a
tight collar the overlapping skins of issue #229.

So only the face is swapped: what the front view shows of it, between the things that stand in front of it (a brim
above, a collar below) and the sides where the head turns away. That outline is found on the figure's front depth
map as the area round the nose the surface reaches without a jump in depth (`face_window`); the seam runs a little
inside it, through skin and hair, where the old and the new face nearly agree. Hat, collar and the rest of the head
stay the figure's own. The figure is cut along that outline (`hero_hands.clip_values`). The new face is put where
the picture has it through its build's own camera, taken as the build's skin seen from the front (`relief`) and
painted by the crop itself; it reaches a little past the seam and is tucked just behind the figure's surface there
(`tuck`), so the figure's cut edge lies over it. A rigged figure's new face takes the old face's skin weights.
`scripts/hero_face.py` runs the steps.

Models face +Z (glTF's front) with Y up, as Pixal3D builds them; "depth" is Z, larger is nearer the camera.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from image_to_3dlab import hero_hands as hh

PLATE_RATIO = 1.5      # a slice of the head this much wider than its median slice is a brim, not head
SLICES = 80            # slices over the head region's height when looking for a brim
MIN_SLICE_POINTS = 20  # fewer points than this in a slice and its width is noise
PIXELS_PER_HEAD = 240  # the front depth map's resolution, in pixels across the head's width
SEED_BAND = 0.2        # the nose is looked for within this share of the head's width either side of its middle ...
SEED_SPAN = (0.3, 0.8)  # ... and this far down from the head's top (or a brim's underside) to its bottom
JUMP_SHARE = 0.25      # a step in depth of this share of the head's width between neighbouring pixels is an edge where
                       # the face goes behind something: the Zauberer's brim drops 0.065 onto his forehead and his jaw
                       # 0.04 onto the collar behind it (head width 0.075); the strand in front of his eye stands 0.01
CLIFF_PIXELS = 3       # ... and so is that step spread over up to this many pixels (`cliffs`)
FACING = 0.2           # the face faces the camera at least this much (the cosine): its sides and the chin's underside,
                       # which turn away, end it
SIDE_SHARE = 0.45      # the face reaches at most this share of the head's width either side of the nose
CHIN_SHARE = 0.6       # and at most this share of it below the nose (the Zauberer's chin: 0.42) ...
BROW_SHARE = 0.55      # ... and above it (his brim: 0.43); without a brim the seam then runs through the hairline
MARGIN_SHARE = 0.06    # the seam runs this share of the head's width inside the outline
SMOOTH_SHARE = 0.04    # the seam's line is smoothed over this share of the head's width
BEHIND_SHARE = 0.15    # what of the figure lies less than this share of the head's width behind the new face goes: the
                       # old eye under a strand, an inner skin; a collar hidden behind the jaw lies deeper and stays
NEAR_PIXELS = 3        # the new face's depth is carried this many pixels past its outline, for the faces the seam cuts
BLEND_SHARE = 0.12     # the new face eases into the figure's depth over this share of the head's width ...
EASE_SHARE = 0.05      # ... toward the front by at most this share of it (`tuck`)
SKIRT_PIXELS = 3       # the new face reaches this many pixels past the seam, under the figure's cut edge ...
TUCK_SHARE = 0.01      # ... and lies this share of the head's width behind the figure's surface there
ROUND_SHARE = 0.0125   # the new face's relief is rounded over this share of the head's width (`round_relief`): built
                       # alone, the Zauberer's face came out in crisp planes, its nose a ridge 60 degrees sharp within
                       # two pixels, and the game figure's decimation kept the planes: a hard line down the nose, its
                       # far flank and the tip's underside dark facets (round 22, F7). At 3 pixels of 240 a head the
                       # sharpest one per cent of its creases turn 11 degrees a pixel, the old face's 26
CROP_PAD = 2.0         # the face's crop is this many times the window's size: hair, brim and collar round the face, so
                       # Pixal3D builds a head and not a mask (the Zauberer's 174 px crop round an 85 px face)
SOFT_EDGE_PX = 2       # the face crop's cut-out keeps its soft alpha only this close to its outline (`solid_cutout`)
OPAQUE = 250           # alpha this high counts as solid (the background remover leaves most of a subject at 254)
ICP_KEEP = 0.75        # fits and checks use the closest three quarters of the point pairs: the old strand is no match
SMALL_SPAN = 4         # triangles up to this many pixels across are drawn into the depth map all at once
FRONT = np.array([0.0, 0.0, 1.0])


# --- The head's measures ----------------------------------------------------------------------------------------

def slice_widths(heights: np.ndarray, xs: np.ndarray, slices: int = SLICES) -> tuple[np.ndarray, np.ndarray]:
    """(edges, width of each slice): the head region cut into SLICES by height, each slice's width across X (its
    2nd to 98th percentile, so a stray point does not count); NaN where a slice holds too few points."""
    heights, xs = np.asarray(heights, dtype=np.float64), np.asarray(xs, dtype=np.float64)
    edges = np.linspace(heights.min(), heights.max(), slices + 1)
    widths = np.full(slices, np.nan)
    which = np.clip(np.searchsorted(edges, heights, side="right") - 1, 0, slices - 1)
    for i in range(slices):
        inside = xs[which == i]
        if len(inside) >= MIN_SLICE_POINTS:
            widths[i] = np.percentile(inside, 98) - np.percentile(inside, 2)
    return edges, widths


def plate_floor(heights: np.ndarray, xs: np.ndarray, slices: int = SLICES, ratio: float = PLATE_RATIO) -> float | None:
    """The underside of a brim: the bottom of the run of slices round the widest one that are all at least RATIO
    times wider than the head's median slice. None when no slice is that wide (no hat, or a narrow one)."""
    edges, widths = slice_widths(heights, xs, slices)
    valid = np.flatnonzero(~np.isnan(widths))
    if not len(valid):
        return None
    median = float(np.median(widths[valid]))
    widest = int(valid[np.argmax(widths[valid])])
    if widths[widest] < ratio * median:
        return None
    low = widest
    while low > 0 and not np.isnan(widths[low - 1]) and widths[low - 1] >= ratio * median:
        low -= 1
    return float(edges[low])


def head_width(heights: np.ndarray, xs: np.ndarray, floor: float | None = None, slices: int = SLICES) -> float:
    """The head's width: its median slice below FLOOR (a brim's underside), or of all of it without one."""
    heights, xs = np.asarray(heights, dtype=np.float64), np.asarray(xs, dtype=np.float64)
    below = heights < floor if floor is not None else np.ones(len(heights), bool)
    if below.sum() < 2 * MIN_SLICE_POINTS:
        below = np.ones(len(heights), bool)
    _, widths = slice_widths(heights[below], xs[below], slices)
    if np.all(np.isnan(widths)):
        return float(np.ptp(xs[below]))
    return float(np.nanmedian(widths))


# --- The front depth map ------------------------------------------------------------------------------------------

def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normal of every triangle (counter-clockwise faces out, as glTF has them)."""
    v = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)


def canvas(box: tuple[float, float, float, float], pixel: float) -> tuple[int, int]:
    """(rows, columns) of a picture covering BOX (x0, x1, y0, y1) with square pixels PIXEL wide."""
    x0, x1, y0, y1 = box
    return max(1, int(np.ceil((y1 - y0) / pixel))), max(1, int(np.ceil((x1 - x0) / pixel)))


def to_pixels(points: np.ndarray, box, pixel: float) -> tuple[np.ndarray, np.ndarray]:
    """(column, row) of points in a picture of BOX, in pixel units with pixel centres at .5; row 0 is the top."""
    p = np.asarray(points, dtype=np.float64)
    return (p[:, 0] - box[0]) / pixel, (box[3] - p[:, 1]) / pixel


def depth_map(vertices: np.ndarray, faces: np.ndarray, box, pixel: float) -> tuple[np.ndarray, np.ndarray]:
    """(depth, owner): the model seen straight from the front over BOX, as the nearest surface's Z in every pixel
    (-inf where nothing is) and the triangle that is nearest there (-1 where nothing is). Every pixel whose centre
    lies in a triangle counts; small triangles are drawn together, larger ones one by one."""
    rows, cols = canvas(box, pixel)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    px, py = to_pixels(v, box, pixel)
    tx, ty, tz = px[f], py[f], v[f][:, :, 2]
    lo_x, hi_x = np.ceil(tx.min(1) - 0.5), np.floor(tx.max(1) - 0.5)
    lo_y, hi_y = np.ceil(ty.min(1) - 0.5), np.floor(ty.max(1) - 0.5)
    area = (tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0])
    on = ((hi_x >= 0) & (hi_y >= 0) & (lo_x <= cols - 1) & (lo_y <= rows - 1) & (hi_x >= lo_x) & (hi_y >= lo_y)
          & (np.abs(area) > 1e-12))
    small = on & (hi_x - lo_x <= SMALL_SPAN) & (hi_y - lo_y <= SMALL_SPAN)
    found_pixel, found_z, found_face = [], [], []

    def test(index, cx, cy):
        """Pixel centres (cx, cy) against triangles INDEX (one centre per triangle): barycentric, inside or not."""
        x, y = tx[index], ty[index]
        a = area[index]
        w0 = ((x[:, 1] - cx) * (y[:, 2] - cy) - (x[:, 2] - cx) * (y[:, 1] - cy)) / a
        w1 = ((x[:, 2] - cx) * (y[:, 0] - cy) - (x[:, 0] - cx) * (y[:, 2] - cy)) / a
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        z = w0 * tz[index, 0] + w1 * tz[index, 1] + w2 * tz[index, 2]
        return inside, z

    index = np.flatnonzero(small)
    for dx in range(SMALL_SPAN + 1):
        for dy in range(SMALL_SPAN + 1):
            c, r = lo_x[index] + dx, lo_y[index] + dy
            keep = (c <= hi_x[index]) & (r <= hi_y[index]) & (c >= 0) & (r >= 0) & (c < cols) & (r < rows)
            inside, z = test(index[keep], c[keep] + 0.5, r[keep] + 0.5)
            found_pixel.append((r[keep][inside] * cols + c[keep][inside]).astype(np.int64))
            found_z.append(z[inside])
            found_face.append(index[keep][inside])
    for i in np.flatnonzero(on & ~small):
        c, r = np.meshgrid(np.arange(max(lo_x[i], 0), min(hi_x[i], cols - 1) + 1),
                           np.arange(max(lo_y[i], 0), min(hi_y[i], rows - 1) + 1))
        c, r = c.reshape(-1), r.reshape(-1)
        inside, z = test(np.full(len(c), i), c + 0.5, r + 0.5)
        found_pixel.append((r[inside] * cols + c[inside]).astype(np.int64))
        found_z.append(z[inside])
        found_face.append(np.full(int(inside.sum()), i))
    depth = np.full(rows * cols, -np.inf)
    owner = np.full(rows * cols, -1, dtype=np.int64)
    if found_pixel:
        pixel_id, z, face = (np.concatenate(found_pixel), np.concatenate(found_z), np.concatenate(found_face))
        order = np.lexsort((-z, pixel_id))                 # by pixel, nearest first
        pixel_id, z, face = pixel_id[order], z[order], face[order]
        first = np.r_[True, pixel_id[1:] != pixel_id[:-1]]
        depth[pixel_id[first]] = z[first]
        owner[pixel_id[first]] = face[first]
    return depth.reshape(rows, cols), owner.reshape(rows, cols)


def sample(grid: np.ndarray, box, pixel: float, points: np.ndarray, outside: float = -np.inf,
           nearest: bool = False) -> np.ndarray:
    """GRID (a picture of BOX) read at the points' (x, y), between pixel centres in a straight line (or at the
    nearest pixel); OUTSIDE where a point lies off the picture."""
    cols, rows = to_pixels(points, box, pixel)
    h, w = grid.shape
    off = (cols < 0) | (rows < 0) | (cols > w) | (rows > h)
    if nearest:
        c = np.clip(np.floor(cols).astype(np.int64), 0, w - 1)
        r = np.clip(np.floor(rows).astype(np.int64), 0, h - 1)
        values = grid[r, c].astype(np.float64)
    else:
        values = ndimage.map_coordinates(grid.astype(np.float64), [rows - 0.5, cols - 0.5], order=1, mode="nearest")
    values[off] = outside
    return values


# --- The face's window ----------------------------------------------------------------------------------------------

def face_seed(depth: np.ndarray, facing: np.ndarray, box, pixel: float, centre_x: float, band: float,
              low: float, high: float) -> tuple[int, int]:
    """(row, column) of the nose: the nearest pixel that faces the camera, within BAND either side of CENTRE_X and
    between heights LOW and HIGH (the middle of the head below its top or brim: a brim's edge, a chin over a high
    collar or a hair spike stand nearer than a nose, and on four figures of round 22 one of them was taken)."""
    rows, cols = depth.shape
    x = box[0] + (np.arange(cols) + 0.5) * pixel
    y = box[3] - (np.arange(rows) + 0.5) * pixel
    allowed = (np.abs(x - centre_x) <= band)[None, :] & ((y >= low) & (y < high))[:, None] & (facing >= FACING)
    candidates = np.where(allowed & np.isfinite(depth), depth, -np.inf)
    if not np.isfinite(candidates).any():
        raise ValueError("no surface faces the camera in the middle of the head: is the model facing +Z?")
    return tuple(int(i) for i in np.unravel_index(np.argmax(candidates), depth.shape))


def cliffs(depth: np.ndarray, jump: float, span: int = CLIFF_PIXELS) -> np.ndarray:
    """Pixels on a cliff: where the depth changes by JUMP or more within SPAN pixels along a row or a column. A coarse
    build turns the edge under a chin into a slope two or three pixels wide (the Zauberer's: 0.022 over three), which
    a step between neighbours alone walked over; a strand's edge (0.01 in one pixel) stays no cliff."""
    out = np.zeros(depth.shape, bool)
    with np.errstate(invalid="ignore"):
        for k in range(1, span + 1):
            for axis in (0, 1):
                a = np.take(depth, range(k, depth.shape[axis]), axis=axis)
                b = np.take(depth, range(depth.shape[axis] - k), axis=axis)
                steep = np.abs(a - b) >= jump
                for j in range(k + 1):
                    index = [slice(None), slice(None)]
                    index[axis] = slice(j, j + depth.shape[axis] - k)
                    out[tuple(index)] |= steep
    return out


def face_outline(depth: np.ndarray, facing: np.ndarray, seed: tuple[int, int], jump: float,
                 least: float = FACING) -> np.ndarray:
    """The pixels the face's surface reaches from SEED without a step in depth of JUMP or more, on surface that faces
    the camera at least LEAST (the cosine), with the holes inside filled (a nostril, the gap under a strand)."""
    rows, cols = depth.shape
    ok = np.isfinite(depth) & (facing >= least) & ~cliffs(depth, jump)
    if not ok[seed]:
        raise ValueError("the face's middle faces away from the camera")
    ids = np.arange(rows * cols).reshape(rows, cols)
    with np.errstate(invalid="ignore"):
        across = ok[:, :-1] & ok[:, 1:] & (np.abs(depth[:, :-1] - depth[:, 1:]) < jump)
        down = ok[:-1, :] & ok[1:, :] & (np.abs(depth[:-1, :] - depth[1:, :]) < jump)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    a = np.concatenate([ids[:, :-1][across], ids[:-1, :][down]])
    b = np.concatenate([ids[:, 1:][across], ids[1:, :][down]])
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(rows * cols, rows * cols))
    _, label = connected_components(graph, directed=False)
    label = label.reshape(rows, cols)
    return ndimage.binary_fill_holes(ok & (label == label[seed]))


def column_reach(outline: np.ndarray, seed: tuple[int, int]) -> tuple[int, int]:
    """(top row, bottom row) the outline reaches up and down from SEED without a break, in the seed's column: under
    a brim the top, at the chin the bottom. Down the face's middle no collar is in the way (it opens there)."""
    column = outline[:, seed[1]]
    top = bottom = seed[0]
    while top > 0 and column[top - 1]:
        top -= 1
    while bottom < len(column) - 1 and column[bottom + 1]:
        bottom += 1
    return top, bottom


def face_shape(shape: tuple[int, int], seed: tuple[int, int], top: int, bottom: int, half_width: float,
               upper_power: float = 4.0, lower_power: float = 2.0) -> np.ndarray:
    """The most a face can take in a picture of SHAPE: HALF_WIDTH pixels either side of the seed's column, from row
    TOP to row BOTTOM. Above the seed it is a rounded box (POWER 4: the brow and the hair under a brim are about as
    wide as the eyes); below, an ellipse narrowing to the chin, as a jaw does. The reach on the depth map alone
    went on down a high collar where a coarse build fused it to the jaw (the Zauberer, 2026-11-22)."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    dx = np.abs(cols - seed[1]) / max(half_width, 1.0)
    up = np.abs(rows - seed[0]) / max(seed[0] - top + 0.5, 1.0)
    down = np.abs(rows - seed[0]) / max(bottom - seed[0] + 0.5, 1.0)
    above = rows <= seed[0]
    return np.where(above, dx ** upper_power + up ** upper_power, dx ** lower_power + down ** lower_power) <= 1.0


def window_mask(outline: np.ndarray, margin: float, smooth: float, seed: tuple[int, int]) -> np.ndarray:
    """OUTLINE shrunk by MARGIN pixels and its edge smoothed over SMOOTH pixels: the area the seam closes round. The
    piece holding SEED is kept (or the largest, if the shrinking took the seed's)."""
    inner = ndimage.distance_transform_edt(outline) > margin
    if smooth > 0:
        inner = ndimage.gaussian_filter(inner.astype(np.float64), smooth) > 0.5
    label, count = ndimage.label(inner)
    if not count:
        raise ValueError("the face's outline is narrower than the seam's margin")
    keep = label[seed] if label[seed] else 1 + int(np.argmax(ndimage.sum(inner, label, range(1, count + 1))))
    return ndimage.binary_fill_holes(label == keep)


def signed_distance(mask: np.ndarray, pixel: float) -> np.ndarray:
    """Distance to MASK's edge in model units, positive inside: zero halfway between an inside and an outside pixel."""
    inside = ndimage.distance_transform_edt(mask) - 0.5
    outside = ndimage.distance_transform_edt(~mask) - 0.5
    return np.where(mask, inside, -outside) * pixel


def face_window(vertices: np.ndarray, faces: np.ndarray, head: np.ndarray) -> dict:
    """Where the face lies and where its seam runs, measured on a figure (its mesh and which vertices are its head,
    from the rig's head bones): {"box", "pixel", "field" (signed distance to the seam, positive inside), "depth" and
    "outline" (the figure's front depth and the face's reach on it, for a picture), "seed" (the nose), "width" (the
    head's), "floor" (the brim's underside or None), "axis_z" (the head's middle depth)}."""
    v = np.asarray(vertices, dtype=np.float64)
    hv = v[np.asarray(head, dtype=bool)]
    if len(hv) < 2 * MIN_SLICE_POINTS:
        raise ValueError("the head region holds too few points to measure")
    floor = plate_floor(hv[:, 1], hv[:, 0])
    width = head_width(hv[:, 1], hv[:, 0], floor)
    below = hv[:, 1] < floor if floor is not None else np.ones(len(hv), bool)
    centre_x = float(np.median(hv[below, 0]))
    bottom = float(hv[:, 1].min())
    top = float(hv[:, 1].max()) if floor is None else min(float(hv[:, 1].max()), floor + width)
    box = (centre_x - 1.5 * width, centre_x + 1.5 * width, bottom - 0.5 * width, top + 0.25 * width)
    pixel = width / PIXELS_PER_HEAD
    f = np.asarray(faces, dtype=np.int64)
    middle = v[f].mean(axis=1)
    near = ((middle[:, 0] > box[0] - width) & (middle[:, 0] < box[1] + width)
            & (middle[:, 1] > box[2] - width) & (middle[:, 1] < box[3] + width))
    depth, owner = depth_map(v, f[near], box, pixel)
    facing = np.where(owner >= 0, face_normals(v, f[near])[np.maximum(owner, 0), 2], -1.0)
    upper = floor if floor is not None else float(hv[:, 1].max())
    seed = face_seed(depth, facing, box, pixel, centre_x, SEED_BAND * width,
                     upper - SEED_SPAN[1] * (upper - bottom), upper - SEED_SPAN[0] * (upper - bottom))
    reach = face_outline(depth, facing, seed, JUMP_SHARE * width)
    top, chin = column_reach(reach, seed)
    top = max(top, seed[0] - int(round(BROW_SHARE * width / pixel)))
    chin = min(chin, seed[0] + int(round(CHIN_SHARE * width / pixel)))
    outline = reach & face_shape(depth.shape, seed, top, chin, SIDE_SHARE * width / pixel)
    mask = window_mask(outline, MARGIN_SHARE * width / pixel, SMOOTH_SHARE * width / pixel, seed)
    return {"box": box, "pixel": pixel, "field": signed_distance(mask, pixel), "depth": depth, "outline": outline,
            "mask": mask, "seed": seed, "width": width, "floor": floor, "rows": (top, chin),
            "axis_z": float(np.median(hv[below, 2])), "centre_x": centre_x}


FAR = 1e6                  # how far outside the window a point off its picture counts: finite, so a cut between it
                           # and a point inside still falls on the edge (an infinity made the edge's mix NaN)


TALL_LIMIT = 1.6         # a face window this many times taller than wide ran past the chin (gunner_h, round 22)
NOSE_SPAN = (0.3, 0.75)  # the nose sits this far down its window; the Zauberer's at 0.5, skin_sorcerer_a's at 0.1


def window_doubts(window: dict) -> list[str]:
    """Plain-words reasons to look at window.png before trusting the result: the window's shape or the nose's place in
    it is unlike a face's. Measured on six figures of round 22: the Zauberer (built and checked) and four more looked
    right; one window ran past the chin into a collar's opening, one sat on the brow and hair."""
    x0, x1, y0, y1 = window_bounds(window)
    doubts = []
    if (y1 - y0) > TALL_LIMIT * (x1 - x0):
        doubts.append(f"the face's window is {(y1 - y0) / (x1 - x0):.1f} times taller than wide: it may run past the "
                      "chin")
    nose = window["box"][3] - (window["seed"][0] + 0.5) * window["pixel"]
    share = (y1 - nose) / max(y1 - y0, 1e-12)
    if not NOSE_SPAN[0] <= share <= NOSE_SPAN[1]:
        doubts.append(f"the nose was found {share:.0%} of the way down the face's window (a face's is about half): "
                      "it may be a hair spike, a chin or a collar")
    return doubts


def window_values(window: dict, points: np.ndarray) -> np.ndarray:
    """The signed distance of POINTS (seen from the front) to the seam: positive inside the face's window, FAR
    outside it where a point lies off the window's picture."""
    return sample(window["field"], window["box"], window["pixel"], points, outside=-FAR)


def window_bounds(window: dict) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of the window, in model units."""
    rows, cols = np.nonzero(window["mask"])
    box, pixel = window["box"], window["pixel"]
    return (box[0] + cols.min() * pixel, box[0] + (cols.max() + 1) * pixel,
            box[3] - (rows.max() + 1) * pixel, box[3] - rows.min() * pixel)


def solid_cutout(picture: np.ndarray, cutout: np.ndarray, edge: int = SOFT_EDGE_PX) -> tuple[np.ndarray, int]:
    """(the cut-out RGBA made solid inside its outline, from PICTURE's RGB; how many pixels changed).

    The background remover took the Zauberer's glowing eyes for background (2026-11-22, his face crop at 156 px), and
    filling those holes alone still left a one-pixel ring of half-transparent alpha round each eye: Pixal3D carved a
    slit through the face along it, the clean-up widened it to a hole, and Pixel Match painted the glow onto the back
    of the head seen through it. A face crop holds no real hole, so everything inside the outline but its own soft
    EDGE (pixels wide) is made opaque."""
    out = np.array(cutout, dtype=np.uint8, copy=True)
    alpha = out[..., 3]
    filled = ndimage.binary_fill_holes(alpha > 127)
    core = ndimage.binary_erosion(filled, iterations=edge) if edge else filled
    fix = (filled & (alpha <= 127)) | (core & (alpha < OPAQUE))
    out[fix, :3] = np.asarray(picture)[fix][:, :3]
    out[fix, 3] = 255
    return out, int(fix.sum())


def crop_box(px: np.ndarray, py: np.ndarray, width: int, height: int, pad: float = CROP_PAD):
    """A square crop (x0, y0, x1, y1) of the picture round the face window's corners (PX, PY in picture pixels),
    PAD times its size and kept in the picture."""
    x0, x1, y0, y1 = min(px), max(px), min(py), max(py)
    side = min(max(x1 - x0, y1 - y0) * pad, width, height)
    left = int(round(np.clip((x0 + x1) / 2 - side / 2, 0, width - side)))
    top = int(round(np.clip((y0 + y1) / 2 - side / 2, 0, height - side)))
    side = int(round(side))
    return left, top, left + side, top + side


# --- Lining the new face up -----------------------------------------------------------------------------------------

def alpha_box(rgba: np.ndarray) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of a cut-out's subject (alpha above half), in pixels, its far edges exclusive."""
    ys, xs = np.nonzero(np.asarray(rgba)[..., 3] > 128)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_to_box(points: np.ndarray, source: tuple, target: tuple) -> np.ndarray:
    """Pixel POINTS (x, y) in a picture whose subject fills box SOURCE, in one where it fills box TARGET: the build's
    input picture is the crop's cut-out cropped to its subject, scaled and centred on a padded square."""
    p = np.asarray(points, dtype=np.float64)
    k = (target[2] - target[0]) / (source[2] - source[0])
    return np.stack([target[0] + (p[:, 0] - source[0]) * k, target[1] + (p[:, 1] - source[1]) * k], axis=1)


def crop_to_figure(pixels: np.ndarray, crop: tuple[int, int, int, int], crop_size: int, frame) -> np.ndarray:
    """Pixels (x, y) of the upscaled crop (CROP_SIZE wide, cut from the picture at CROP) as the figure's model (x, y),
    by the figure's picture frame (`hero.frame`; its own camera is all but straight-on at its face)."""
    centre, per_px, box_centre = frame
    k = (crop[2] - crop[0]) / crop_size
    u = crop[0] + np.asarray(pixels, dtype=np.float64)[:, 0] * k
    v = crop[1] + np.asarray(pixels, dtype=np.float64)[:, 1] * k
    return np.stack([centre[0] + (u - box_centre[0]) * per_px, centre[1] - (v - box_centre[1]) * per_px], axis=1)


def pairs_apart(points: np.ndarray, target: np.ndarray, keep: float = ICP_KEEP) -> float:
    """How far POINTS lie from TARGET: the mean distance to the nearest target point of the KEEP share that lie
    closest (an old strand is no match for anything)."""
    distance, _ = cKDTree(np.asarray(target, dtype=np.float64)).query(np.asarray(points, dtype=np.float64),
                                                                       workers=-1)
    return float(distance[distance <= np.quantile(distance, keep)].mean())


def similarity_2d(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The 4x4 transform whose turn about Z, shift in X and Y and one size best move SOURCE's (x, y) onto TARGET's,
    row for row (Umeyama in the picture plane); Z is scaled by the same size and not shifted."""
    s, t = np.asarray(source, dtype=np.float64)[:, :2], np.asarray(target, dtype=np.float64)[:, :2]
    ms, mt = s.mean(axis=0), t.mean(axis=0)
    a, b = s - ms, t - mt
    u, sing, vt = np.linalg.svd(b.T @ a / len(s))
    flip = np.diag([1.0, np.sign(np.linalg.det(u @ vt)) or 1.0])
    turn = u @ flip @ vt
    k = float(np.trace(np.diag(sing) @ flip) / max((a ** 2).sum() / len(s), 1e-30))
    tf = np.eye(4)
    tf[:2, :2] = k * turn
    tf[2, 2] = k
    tf[:2, 3] = mt - k * turn @ ms
    return tf


def depth_fit(window: dict, part: dict, keep: float = ICP_KEEP, rounds: int = 3) -> np.ndarray:
    """The 4x4 transform that moves PART (a face put in by the picture) in depth onto the figure's face: Z plus a
    plane in X and Y, fitted over the window to the figure's front depth less the part's, each round on the KEEP share
    of pixels that fit best (the old strand in front of an eye is no match). The crop's frame centres a build's depth
    on the figure's middle, and a face built alone is mostly the head behind it: the Zauberer's sat 0.09 too deep and
    a little turned, its jaw 0.009 proud on one side. A plane, not a turn: it keeps X and Y where the picture put
    them."""
    box, pixel = window["box"], window["pixel"]
    own, _ = depth_map(part["positions"], part["faces"], box, pixel)
    both = window["mask"] & np.isfinite(own) & np.isfinite(window["depth"])
    if not both.any():
        raise ValueError("the new face does not reach into the window: check its placement")
    rows, cols = np.nonzero(both)
    x = box[0] + (cols + 0.5) * pixel
    y = box[3] - (rows + 0.5) * pixel
    gap = window["depth"][both] - own[both]
    design = np.stack([np.ones(len(x)), x, y], axis=1)
    use = np.ones(len(x), bool)
    for _ in range(rounds):
        coef = np.linalg.lstsq(design[use], gap[use], rcond=None)[0]
        miss = np.abs(design @ coef - gap)
        use = miss <= np.quantile(miss, keep)
    tf = np.eye(4)
    tf[2, 0], tf[2, 1], tf[2, 3] = coef[1], coef[2], coef[0]
    return tf


# --- Cutting and tucking --------------------------------------------------------------------------------------------

def sheets(faces: np.ndarray, group: np.ndarray) -> np.ndarray:
    """A piece id per face, faces joined only across an edge exactly two of them share. Pixal3D's clean-up joins the
    outer skin to the inner one at a few edges four faces share (55 on the Zauberer's face); joined at those, the
    face's rim ran round the outer skin and back round the inner one, a loop with no area."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(group)[np.asarray(faces, dtype=np.int64)]
    m = len(f)
    if not m:
        return np.zeros(0, dtype=np.int64)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(m), 3)
    _, edge_id, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    edge_id = edge_id.reshape(-1)
    shared = counts[edge_id] == 2
    order = np.argsort(edge_id[shared], kind="stable")
    pairs = owner[shared][order].reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
    return connected_components(graph, directed=False)[1]


def front_piece(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Which faces belong to the piece seen most from the front: the most area facing the camera. After the window's
    cut, a face built alone falls apart into its front, the back of its head, an inner skin and specks of brim or
    collar the cut caught; only the front belongs in the figure's face. Pieces join across edges two faces share
    (`sheets`), so an inner skin touching the outer one at an edge stays apart."""
    v = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    if not len(v):
        return np.zeros(0, bool)
    label = sheets(faces, hh.weld_groups(vertices))
    toward = ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
              - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))   # twice the area seen from the front
    score = np.bincount(label, weights=np.clip(toward, 0.0, None))
    return label == int(np.argmax(score))


def tuck(points: np.ndarray, values: np.ndarray, target: np.ndarray, blend: float, cap: float,
         gap: float) -> np.ndarray:
    """POINTS of the new face moved in depth toward TARGET less GAP (the figure's own surface where each point is,
    from its front depth map): in full where VALUES (the distance inside the seam) is at most 0, the skirt under the
    figure's cut edge, fading out smoothly over BLEND inside. Toward the back without limit: a new face standing proud
    would show its edge. Toward the front at most CAP: at the Zauberer's brow the figure's own strand stood 0.012 in
    front at the seam, and raising the face to it made a ramp across the brow; its cut end hides the edge instead.
    Where TARGET is NaN (no figure there) a point stays."""
    p = np.array(points, dtype=np.float64)
    w = np.clip(1.0 - np.asarray(values, dtype=np.float64) / max(blend, 1e-12), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    change = np.asarray(target, dtype=np.float64) - gap - p[:, 2]
    change = np.where(np.isfinite(change), np.minimum(change, cap), 0.0)
    p[:, 2] += w * change
    return p


def reference_depth(own: np.ndarray, new: np.ndarray) -> np.ndarray:
    """What a figure's face is measured against: the nearer-to-the-back of its own front depth and the new face's,
    where both are (+inf where neither is, so nothing there counts as in front)."""
    out = np.where(np.isfinite(own), own, np.inf)
    both = np.isfinite(own) & np.isfinite(new)
    out[both] = np.minimum(own[both], new[both])
    only_new = np.isfinite(new) & ~np.isfinite(own)
    out[only_new] = new[only_new]
    return out


def spread(depth: np.ndarray, reach: int) -> np.ndarray:
    """DEPTH with its empty pixels within REACH of a full one given that one's value: the new face's depth carried a
    little past its outline, for the faces the seam cuts there."""
    empty = ~np.isfinite(depth)
    if not empty.any() or empty.all():
        return depth
    distance, (r, c) = ndimage.distance_transform_edt(empty, return_indices=True)
    out = depth.copy()
    fill = empty & (distance <= reach)
    out[fill] = depth[r[fill], c[fill]]
    return out


def relief(depth: np.ndarray, box, pixel: float, keep: np.ndarray) -> dict:
    """A surface through every KEEP pixel of DEPTH (a front depth map of BOX): one point per pixel at its depth, two
    triangles per square of four kept pixels, facing the camera.

    A face built alone is a hollow shell on the Zauberer: an inner wall about 0.002 behind its skin, joined to it all
    over the face, so a cut along the seam ran round the skin and back round the wall, a rim with no area. The nearest
    surface seen from the front is the skin and nothing else, and a face is seen from the front; what it gives up is
    the underside of the nose and of the strands, which a game camera above the figure never sees."""
    ok = np.asarray(keep, dtype=bool) & np.isfinite(depth)
    index = np.full(depth.shape, -1, dtype=np.int64)
    index[ok] = np.arange(int(ok.sum()))
    r, c = np.nonzero(ok)
    positions = np.stack([box[0] + (c + 0.5) * pixel, box[3] - (r + 0.5) * pixel, depth[r, c]], axis=1)
    tl, tr, bl, br = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    full = (tl >= 0) & (tr >= 0) & (bl >= 0) & (br >= 0)
    faces = np.concatenate([np.stack([bl[full], br[full], tr[full]], axis=1),
                            np.stack([bl[full], tr[full], tl[full]], axis=1)])
    return {"positions": positions, "faces": faces}


def round_relief(depth: np.ndarray, keep: np.ndarray, sigma: float, jump: float | None = None) -> np.ndarray:
    """DEPTH (a front depth map) rounded over SIGMA pixels within KEEP: every kept pixel becomes the Gaussian-weighted
    mean of the kept pixels round it, so a crease turns into a curve while a plane stays a plane. A cliff (a step of
    JUMP or more within CLIFF_PIXELS, `cliffs`) splits the map into pieces that are rounded each on its own, and its
    own pixels stay, so a brim's edge does not drag the brow up to it. Empty pixels stay empty."""
    depth = np.asarray(depth, dtype=np.float64)
    ok = np.asarray(keep, dtype=bool) & np.isfinite(depth)
    if jump is not None:
        ok &= ~cliffs(np.where(np.isfinite(depth), depth, np.nan), jump)
    out = depth.copy()
    if sigma <= 0 or not ok.any():
        return out
    label, count = ndimage.label(ok)
    for piece in range(1, count + 1):
        mine = label == piece
        if mine.sum() < MIN_SLICE_POINTS:
            continue
        num = ndimage.gaussian_filter(np.where(mine, depth, 0.0), sigma, mode="constant")
        den = ndimage.gaussian_filter(mine.astype(np.float64), sigma, mode="constant")
        out[mine] = num[mine] / den[mine]
    return out


def picture_uvs(points: np.ndarray, crop: tuple[int, int, int, int], crop_size: int, frame) -> np.ndarray:
    """glTF texture coordinates (u right, v down) in the upscaled crop of the figure's model points (x, y): the inverse
    of `crop_to_figure`, so the crop paints the face where the picture shows it."""
    centre, per_px, box_centre = frame
    p = np.asarray(points, dtype=np.float64)
    k = (crop[2] - crop[0]) / crop_size
    u = (box_centre[0] + (p[:, 0] - centre[0]) / per_px - crop[0]) / k
    v = (box_centre[1] - (p[:, 1] - centre[1]) / per_px - crop[1]) / k
    return np.stack([u / crop_size, v / crop_size], axis=1)


def picture_material(png: bytes, name: str = "HeroFace", roughness: float = 0.8) -> tuple[dict, bytes, int]:
    """A material painted by one picture (PNG bytes), as (document, binary, index) of a small GLB, the form a part's
    "material" takes (`hero_hands.regraft_glb` copies it into the model)."""
    from image_to_3dlab.glb_turn import parse_glb

    writer = hh.GlbWriter()
    writer.doc["images"] = [{"bufferView": writer.view(png), "mimeType": "image/png"}]
    writer.doc["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
    writer.doc["textures"] = [{"source": 0, "sampler": 0}]
    writer.doc["materials"] = [{"name": name, "pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": roughness}}]
    doc, binary = parse_glb(writer.build())
    return doc, binary, 0


def face_relief(face: dict, window: dict, crop: tuple[int, int, int, int], crop_size: int, frame,
                material: tuple[dict, bytes, int] | None = None, step: float | None = None,
                rounding: float | None = None) -> dict:
    """The placed new FACE (a build in the figure's frame) as a relief over the window and a little beyond it
    (`relief`), every STEP (default the window's pixel), rounded over ROUNDING (default ROUND_SHARE of the head's
    width, `round_relief`) and painted by the crop picture (`picture_uvs`)."""
    box, pixel = window["box"], window["pixel"]
    step = pixel if step is None else step
    rounding = ROUND_SHARE * window["width"] if rounding is None else rounding
    p = np.asarray(face["positions"], dtype=np.float64)
    f = np.asarray(face["faces"], dtype=np.int64)
    middle = p[f].mean(axis=1)
    near = ((middle[:, 0] > box[0]) & (middle[:, 0] < box[1]) & (middle[:, 1] > box[2]) & (middle[:, 1] < box[3]))
    depth, _ = depth_map(p, f[near], box, step)
    rows, cols = depth.shape
    grid = np.stack(np.meshgrid(box[0] + (np.arange(cols) + 0.5) * step, box[3] - (np.arange(rows) + 0.5) * step),
                    axis=-1).reshape(-1, 2)
    inside = window_values(window, np.c_[grid, np.zeros(len(grid))]).reshape(rows, cols) > -(NEAR_PIXELS + 1) * pixel
    depth = round_relief(depth, inside, rounding / step, JUMP_SHARE * window["width"])
    out = relief(depth, box, step, inside)
    out["uvs"] = picture_uvs(out["positions"], crop, crop_size, frame)
    if material is not None:
        out["material"] = material
    return out


def place_face(face: dict, tf: np.ndarray) -> dict:
    """FACE (a part) moved by the 4x4 TF; its normals turned along."""
    out = {**face, "positions": hh.moved(tf, face["positions"])}
    if face.get("normals") is not None:
        turn = tf[:3, :3] / np.cbrt(np.linalg.det(tf[:3, :3]))
        out["normals"] = np.asarray(face["normals"], dtype=np.float64) @ turn.T
    return out


def cut_face(face: dict, window: dict) -> tuple[dict, np.ndarray]:
    """(the placed new FACE cut along the window's seam, its front piece only; that piece's rim edges)."""
    values = window_values(window, face["positions"])
    inside, rim = hh.cut_part_values(face, -values)
    keep = front_piece(inside["positions"], inside["faces"])
    return hh.keep_faces(inside, keep, rim)


def graft_face(parts: list[dict], window: dict, face: dict, behind: float | None = None,
               blend: float | None = None, ease: float | None = None) -> tuple[list[dict], dict, dict]:
    """PARTS (one model of the figure, part by part, in the figure's frame) with their face cut out along the
    window's seam and the placed new FACE set in. Returns (the model's parts cut, the new face part, report).

    What goes of the figure: inside the window, whatever is not more than BEHIND (default BEHIND_SHARE of the head's
    width) behind the new face or the figure's own front, so a strand in front of an eye, the eye under it and an
    inner skin go, while a collar hidden behind the jaw stays. The new face reaches SKIRT_PIXELS past the seam and is
    tucked just behind the figure's surface there (`tuck`, at most EASE toward the front, default EASE_SHARE of the
    head's width), so the figure's cut edge lies over it. Not stitched: Pixal3D builds skin and inner wall as one
    surface joined all over (the Zauberer's figure and face both), so neither side's cut gives one clean rim, and a
    stitch between two such rims tore across the face. With skin weights (dense) the new face takes the weights of
    the nearest old face point."""
    width = window["width"]
    behind = BEHIND_SHARE * width if behind is None else behind
    blend = BLEND_SHARE * width if blend is None else blend
    cap = EASE_SHARE * width if ease is None else ease
    box, pixel = window["box"], window["pixel"]
    skirt = SKIRT_PIXELS * pixel
    new, _ = hh.cut_part_values(face, -(window_values(window, face["positions"]) + skirt))
    if not len(new["faces"]):
        raise ValueError("the new face does not reach into the window: check its placement")
    new, _ = hh.keep_faces(new, front_piece(new["positions"], new["faces"]))
    new_depth = spread(depth_map(new["positions"], new["faces"], box, pixel)[0], NEAR_PIXELS)
    out_parts, old_points, old_dense, report = [], [], [], {"parts": []}
    body_index, target, most = None, None, 0
    for index, part in enumerate(parts):
        p, f = np.asarray(part["positions"], dtype=np.float64), np.asarray(part["faces"], dtype=np.int64)
        values = window_values(window, p)
        middle = p[f].mean(axis=1)
        reach = (values[f] > -skirt - NEAR_PIXELS * pixel).any(axis=1)
        if not reach.any():
            out_parts.append(part)
            report["parts"].append({"cut": False})
            continue
        own, _ = depth_map(p, f[reach], box, pixel)
        ref = sample(reference_depth(own, new_depth), box, pixel, middle, outside=np.inf, nearest=True)
        cuttable = reach & (middle[:, 2] > ref - behind)
        gone = cuttable & (values[f] > 0).all(axis=1)
        if part.get("dense") is not None and gone.any():
            used = np.unique(f[gone])
            old_points.append(p[used])
            old_dense.append(np.asarray(part["dense"])[used])
        in_front = middle[gone, 2] - sample(new_depth, box, pixel, middle[gone], outside=-np.inf, nearest=True)
        cut, _ = hh.cut_part_values(part, values, cuttable)
        report["parts"].append({"cut": True, "faces_removed": int(gone.sum()),
                                "old_in_front_max": float(np.max(in_front[np.isfinite(in_front)], initial=0.0))})
        if gone.sum() > most:
            body_index, most = len(out_parts), int(gone.sum())
            target = spread(own, NEAR_PIXELS)
        out_parts.append(cut)
    if target is None:
        raise ValueError("nothing of the model lies in the window: check the window")
    values = window_values(window, new["positions"])
    at = sample(np.where(np.isfinite(target), target, np.nan), box, pixel, new["positions"], outside=np.nan)
    seam = np.abs(values) <= pixel
    gaps = (at - TUCK_SHARE * width - new["positions"][:, 2])[seam]
    gaps = gaps[np.isfinite(gaps)]
    new = {**new, "positions": tuck(new["positions"], values, at, blend, cap, TUCK_SHARE * width)}
    if old_points:
        new["dense"] = hh.borrowed_weights(new["positions"], np.concatenate(old_points), np.concatenate(old_dense))
        if parts[body_index].get("dense") is None:
            new.pop("dense")
    report.update({"body_part": body_index, "face_faces": int(len(new["faces"])),
                   "seam_gap_median": float(np.median(np.abs(gaps))) if len(gaps) else 0.0,
                   "seam_gap_max": float(np.abs(gaps).max()) if len(gaps) else 0.0,
                   "seam_raised_short": float(np.mean(gaps > cap)) if len(gaps) else 0.0})
    return out_parts, new, report


# --- GLB in ---------------------------------------------------------------------------------------------------------

def read_parts(data: bytes) -> tuple[dict, bytes, list[dict]]:
    """(document, binary, parts) of a GLB holding one mesh, every primitive as a part (`hero_hands.read_part`): a
    figure that already had its hands grafted has two, the body and the hands."""
    from image_to_3dlab.glb_turn import parse_glb

    doc, binary = parse_glb(data)
    nodes = [n for n in doc.get("nodes", []) if "mesh" in n]
    if len({n["mesh"] for n in nodes}) != 1 or len(nodes) != 1:
        raise ValueError("expected one mesh in the model")
    if any(key in nodes[0] for key in ("matrix", "translation", "rotation", "scale")):
        raise ValueError("the model's mesh is moved by its node; expected a model in the figure's own frame")
    mesh = doc["meshes"][nodes[0]["mesh"]]
    return doc, binary, [hh.primitive_part(doc, binary, primitive) for primitive in mesh["primitives"]]


def window_picture(window: dict):
    """The figure's front depth with the face's reach (blue) and the seam (red) drawn in: what a human checks."""
    from PIL import Image

    depth = window["depth"]
    seen = np.isfinite(depth)
    grey = np.zeros(depth.shape)
    if seen.any():
        lo, hi = depth[seen].min(), depth[seen].max()
        grey[seen] = 40 + 200 * (depth[seen] - lo) / max(hi - lo, 1e-12)
    rgb = np.stack([grey] * 3, axis=-1)
    reach = window["outline"] & ~window["mask"]
    rgb[reach] = rgb[reach] * 0.6 + np.array([0, 60, 160]) * 0.4
    edge = window["mask"] & ~ndimage.binary_erosion(window["mask"])
    rgb[ndimage.binary_dilation(edge)] = [230, 40, 40]
    r, c = window["seed"]
    rgb[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3] = [255, 220, 0]
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
