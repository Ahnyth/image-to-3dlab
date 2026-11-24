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
the picture has it through its build's own camera, taken as the build's skin seen from the front (`relief`), its
creases rounded a little (`round_relief`), and painted by the crop moved to the figure's own paint at the seam
(`match_face_paint`, read at every pixel just outside the seam); it rises to meet the figure's surface at the seam
(its front without the strands standing off it, `seam_surface`), reaches a little past the seam and is tucked just
behind the figure's surface there (`tuck`), so the figure's cut edge lies over it; a strand's cut end still standing
over the face eases down onto it (`ease_onto`), and where the new face is no face (a pit its build fell into) it takes
the figure's own shape (`fill_pits`). A rigged figure's new face takes the old face's skin weights.
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
DIVE_SHARE = 0.25      # ... but the new face counts there only where it lies less than this share of it behind the
                       # figure's front (`reference_depth`): skin_sorcerer_c's lay within 0.22 of it over 95 % of the
                       # face, and 0.4 to 0.9 behind it in two pits where the build showed no face (round 27)
NEAR_PIXELS = 3        # the new face's depth is carried this many pixels past its outline, for the faces the seam cuts
BLEND_SHARE = 0.12     # the new face eases into the figure's depth over this share of the head's width ...
EASE_SHARE = 0.2       # ... toward the front by at most this share of it (`tuck`): the Nachtschwinge's cheek stood up
                       # to 0.17 of its head's width in front of its new face at the seam (round 28); 0.05 left the
                       # figure's cut edge hanging over the face, a light line down the cheek
OVER_SHARE = 0.03      # inside the seam the face eases toward the figure's own front at most this share of the head's
                       # width above that front carried in from the seam (`seam_target`): a hood's edge (round 28)
STRAND_SHARE = 0.08    # a ridge of the figure's front narrower than this share of the head's width stands off the face
                       # (a strand): the face goes on into the surface beside it, not up to it (`seam_surface`)
SKIRT_PIXELS = 3       # the new face reaches this many pixels past the seam, under the figure's cut edge ...
TUCK_SHARE = 0.005     # ... and lies this share of the head's width behind the figure's surface there
SEAM_BAND_SHARE = 0.12  # where the figure still stands in front of the new face at the seam (a strand's cut end, or a
                        # rise past EASE_SHARE), its own surface eases back onto the face over this share of the head's
                        # width outside the seam (`ease_onto`)
EASED_LIMIT_SHARE = 0.15   # the figure's surface eased back further than this share of the head's width is said
STRAND_EASE_SHARE = 0.05   # a cut end standing further over the face than this share of the head's width is a strand
                           # and stays as it is (`ease_onto`): laid down, the Nachtschwinge's hair locks scratched its
                           # cheeks dark (round 28)
ROUND_SHARE = 0.0125   # the new face's relief is rounded over this share of the head's width (`round_relief`): built
                       # alone, the Zauberer's face came out in crisp planes, its nose a ridge 60 degrees sharp within
                       # two pixels, and the game figure's decimation kept the planes: a hard line down the nose, its
                       # far flank and the tip's underside dark facets (round 22, F7). At 3 pixels of 240 a head the
                       # sharpest one per cent of its creases turn 11 degrees a pixel, the old face's 26
COARSEST_STEP = 4      # a model's new face is laid out at most this many of the window's pixels apart (`relief_step`)
RING_PIXELS = 4        # the figure's paint is read in a band this many pixels wide just outside the seam ...
PAINT_LIMIT = 60.0     # ... and the crop's moved toward it by at most this many levels (0..255) a channel
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
    (-inf where nothing is) and the triangle that is nearest there (-1 where nothing is)."""
    rows, cols = canvas(box, pixel)
    pixel_id, z, face = pixel_hits(vertices, faces, box, pixel)
    depth = np.full(rows * cols, -np.inf)
    owner = np.full(rows * cols, -1, dtype=np.int64)
    if len(pixel_id):
        order = np.lexsort((-z, pixel_id))                 # by pixel, nearest first
        pixel_id, z, face = pixel_id[order], z[order], face[order]
        first = np.r_[True, pixel_id[1:] != pixel_id[:-1]]
        depth[pixel_id[first]] = z[first]
        owner[pixel_id[first]] = face[first]
    return depth.reshape(rows, cols), owner.reshape(rows, cols)


def pixel_hits(vertices: np.ndarray, faces: np.ndarray, box, pixel: float) -> tuple[np.ndarray, np.ndarray,
                                                                                     np.ndarray]:
    """(pixel id, Z, triangle) of every crossing of a straight-on view over BOX with the model: one entry per pixel
    and triangle whose centre lies in it, every layer, nearest or not (row-major pixel ids). Small triangles are drawn
    together, larger ones one by one."""
    rows, cols = canvas(box, pixel)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if not len(f):
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
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
    return np.concatenate(found_pixel), np.concatenate(found_z), np.concatenate(found_face)


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


def reached(depth: np.ndarray, seed: tuple[int, int], jump: float) -> np.ndarray:
    """The pixels of DEPTH (a front depth map) reached from SEED without crossing a cliff (`cliffs`, a step of JUMP or
    more within CLIFF_PIXELS); every pixel DEPTH holds when SEED lies on a cliff or on nothing. A new face's relief
    falls off a cliff where its build showed no face (to the build's own back of the head), and what lies beyond is
    no face."""
    ok = np.isfinite(depth) & ~cliffs(np.where(np.isfinite(depth), depth, np.nan), jump)
    label, _ = ndimage.label(ok)
    if not label[seed]:
        return np.isfinite(depth)
    return label == label[seed]


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


def pick_region(asked: str, window: dict) -> tuple[str, str]:
    """(what is built again, why): ASKED "face" or "head" as it is; "auto" the whole head under a hat's brim when the
    face's WINDOW found one above the face (its underside, `floor`), else the face alone."""
    if asked != "auto":
        return asked, f"asked for the {asked}"
    if window.get("floor") is not None:
        return "head", "a hat's brim stands over the face: the whole head under it is built again"
    return "face", "no hat's brim over the face: the face alone is built again"


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
    from its front depth map without its strands, `seam_surface`): in full where VALUES (the distance inside the seam)
    is at most 0, the skirt under the figure's cut edge, fading out smoothly over BLEND inside. Toward the back
    without limit: a new face standing proud would show its edge. Toward the front at most CAP; what still stands in
    front then is eased back onto the face (`ease_onto`). At the Zauberer's brow the figure's own strand stood 0.012
    in front at the seam, and raising the face to it made a ramp across the brow: the strand is no target. Where
    TARGET is NaN (no figure there) a point stays."""
    p = np.array(points, dtype=np.float64)
    w = np.clip(1.0 - np.asarray(values, dtype=np.float64) / max(blend, 1e-12), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    change = np.asarray(target, dtype=np.float64) - gap - p[:, 2]
    change = np.where(np.isfinite(change), np.minimum(change, cap), 0.0)
    p[:, 2] += w * change
    return p


def face_continued(positions: np.ndarray, faces: np.ndarray, window: dict, reach: int) -> np.ndarray:
    """The new face's depth seen from the front where it shows (inside the seam, the window's field above 0), carried
    REACH pixels past the seam, each pixel there taking its nearest face pixel's depth: where the face would go on
    under the figure's cut edge (`skirt_under`). -inf further out."""
    depth, _ = depth_map(positions, faces, window["box"], window["pixel"])
    return spread(np.where(window["field"] > 0, depth, -np.inf), reach)


def skirt_under(points: np.ndarray, values: np.ndarray, continued: np.ndarray, own: np.ndarray,
                gap: float) -> np.ndarray:
    """POINTS of the new face with their skirt (VALUES at most 0: past the seam, under the figure's cut edge) laid
    flat in depth at CONTINUED (`face_continued` at each point), never nearer than GAP behind OWN (the figure's
    surface there). Where the figure has none (NaN) a point stays: a skirt laid flat into the air would show. Tucked
    as its own relief ran, the skirt went down a pit's wall at skin_sorcerer_c's jaw, deep under the figure's edge, and
    one saw past the edge into the head from the side and below (round 28 of the AssetFurnace Support notes)."""
    p = np.array(points, dtype=np.float64)
    own = np.asarray(own, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        z = np.where(np.isfinite(own), np.fmin(np.asarray(continued, dtype=np.float64), own - gap), np.nan)
    flat = (np.asarray(values, dtype=np.float64) <= 0) & np.isfinite(z)
    p[flat, 2] = z[flat]
    return p


def seam_target(surface: np.ndarray, field: np.ndarray, reach: int, pixel: float) -> np.ndarray:
    """What the new face eases into near its seam (`tuck`): SURFACE (a front depth map) where FIELD (the signed distance
    to the seam) is at most 0, and up to REACH pixels inside the seam the surface carried in along its own slope: each
    pixel there takes its nearest outside pixel's depth plus that pixel's slope (read two pixels further out, at most
    PIXEL of depth a pixel, 45 degrees) times the way. Read inside the seam instead, the target took what the new face replaces:
    on skin_sorcerer_c a hood's edge 2 cm in front of the jaw pulled one point of the face up into a light spike (round
    28 of the AssetFurnace Support notes). -inf where nothing is known."""
    surface = np.asarray(surface, dtype=np.float64)
    outside = (np.asarray(field) <= 0) & np.isfinite(surface)
    out = np.where(outside, surface, -np.inf)
    if not outside.any():
        return out
    distance, (qr, qc) = ndimage.distance_transform_edt(~outside, return_indices=True)
    rows, cols = np.mgrid[0:surface.shape[0], 0:surface.shape[1]]
    inside = ~outside & (distance <= reach) & (distance > 0)
    if not inside.any():
        return out
    r, c, d = rows[inside], cols[inside], distance[inside]
    q_r, q_c = qr[inside], qc[inside]
    step_r = np.round(2 * (q_r - r) / d).astype(np.int64)
    step_c = np.round(2 * (q_c - c) / d).astype(np.int64)
    r2 = np.clip(q_r + step_r, 0, surface.shape[0] - 1)
    c2 = np.clip(q_c + step_c, 0, surface.shape[1] - 1)
    near, far = surface[q_r, q_c], surface[r2, c2]
    known = outside[r2, c2] & np.isfinite(far)
    apart = np.maximum(np.hypot(r2 - q_r, c2 - q_c), 1.0)
    slope = np.where(known, (near - far) / apart, 0.0)
    out[r, c] = near + np.clip(slope, -pixel, pixel) * d
    return out


def rise_target(own: np.ndarray, surface: np.ndarray, carried: np.ndarray, new: np.ndarray,
                over: float) -> np.ndarray:
    """What the new face eases into near its seam (`tuck`), pixel by pixel. Where the figure's own front OWN lies
    behind the face (NEW, the face's own depth): that front, so the face sinks just behind it. Where it stands in
    front: no higher than the front without its strands (SURFACE, `seam_surface`) nor than OVER above that front
    carried in from the seam (CARRIED, `seam_target`), so the face rises to a cheek but not to a strand, and sinks
    under a thin hood's edge to the surface beside it. Used where the figure lies behind the face too, the strand-free
    front pulled the Nachtschwinge's chin back: it took the old chin for a strand and dropped to the neck 7 cm behind
    it, and the new chin's coarse triangles, pulled back by up to 3.4 cm, tilted away and drew a dark grey patch
    (round 28 of the AssetFurnace Support notes). -inf where OWN has nothing."""
    own = np.asarray(own, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    out = own.copy()
    with np.errstate(invalid="ignore"):
        up = np.isfinite(own) & np.isfinite(new) & (own > new)
        limit = np.fmin(np.asarray(surface, dtype=np.float64), np.asarray(carried, dtype=np.float64) + over)
        out[up] = np.fmin(own[up], limit[up])
    return out


def seam_surface(own: np.ndarray, across: float, least: float = 0.0) -> np.ndarray:
    """OWN (the figure's front depth map) with what stands off it narrower than ACROSS pixels, and more than LEAST in
    front of the surface round it, replaced by that surface: such a strand is found where the depth stands above its
    grey opening (its erosion, then its dilation, over a disc ACROSS wide; an opening keeps any plane or dome as it
    is), and filled from the pixels beside it, each weighed by a Gaussian ACROSS / 2 wide, so a strand in front of the
    brow gives way to the brow beside it at its own height. A cheek or a mass of hair, broad, stays exactly as it is.
    The figure's outline is carried outward first, so the empty background does not eat into it; -inf where OWN has
    nothing. The surface a new face goes on into at its seam (`tuck`): the face rises to a cheek standing in front of
    it, never up to a strand."""
    own = np.asarray(own, dtype=np.float64)
    seen = np.isfinite(own)
    if across < 2 or not seen.any():
        return own.copy()
    _, (r, c) = ndimage.distance_transform_edt(~seen, return_indices=True)
    filled = own[r, c]
    radius = across / 2.0
    k = int(np.ceil(radius))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    opened = ndimage.grey_opening(filled, footprint=(xx * xx + yy * yy) <= radius * radius)
    strand = seen & (filled - opened > least)
    if not strand.any():
        return own.copy()
    keep = seen & ~strand
    weight = ndimage.gaussian_filter(keep.astype(np.float64), radius)
    beside = ndimage.gaussian_filter(np.where(keep, filled, 0.0), radius) / np.maximum(weight, 1e-12)
    out = own.copy()
    out[strand] = np.where(weight[strand] > 1e-3, beside[strand], opened[strand])
    return out


def ease_onto(points: np.ndarray, faces: np.ndarray, rim: np.ndarray, values: np.ndarray, under: np.ndarray,
              band: float, gap: float, most: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(POINTS moved, how far each moved back): the figure's own surface beside the seam eased back in depth onto the
    new face where its cut edge still stands in front of it. Each point of the cut's RIM (edges, in the part's
    numbering) stands over the face by its depth less UNDER (the new face's depth there; NaN where the face is not
    under it) less GAP; a point outside the seam (VALUES at most 0) less than BAND from it, joined to the rim through
    such points (welded by position, so a UV seam does not stop it, while a separate strand or a hidden collar stays),
    moves back by what its nearest rim point (in 3D) stands over, times a weight easing from 1 at the seam to 0 at
    BAND outside it, smoothly: the surface keeps its own shape and is laid down at its cut end. In depth only, so seen
    from the front nothing folds.

    A strand's cut end, or a cheek standing further in front than the face may rise (EASE_SHARE), hung over the face
    as an edge: down the Nachtschwinge's cheek a light line, up to 2.5 cm in front, and from the side and below one saw
    past it into the head (round 28 of the AssetFurnace Support notes). Pushing only the points over the face behind it
    left saw-tooth notches (its v19f), dropping the triangles over it small notches and the line seen from the other
    side (v19g); and measuring each point against the face carried past the seam pulled the jaw down where the face
    turned away under the chin: the surface has to ease from its rim, not each point on its own. A cut end (one piece
    of the rim) standing more than MOST over the face anywhere is a strand and is left as it is: laid down 2.8 cm onto
    the Nachtschwinge's cheeks, its hair locks drew dark scratches on the skin."""
    p = np.array(points, dtype=np.float64)
    moved = np.zeros(len(p))
    rim = np.asarray(rim, dtype=np.int64).reshape(-1, 2)
    if not len(rim) or band <= 0:
        return p, moved
    values = np.asarray(values, dtype=np.float64)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    group = hh.weld_groups(p)              # copies of a point (split along UV seams) move as one
    rim_group = np.zeros(group.max() + 1, bool)
    rim_group[group[rim.reshape(-1)]] = True
    on_rim = rim_group[group]
    near_group = rim_group.copy()
    near_group[group[(values <= 0) & (values > -band)]] = True
    near = near_group[group]
    f = group[np.asarray(faces, dtype=np.int64)]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = e[near_group[e[:, 0]] & near_group[e[:, 1]]]
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(near_group), len(near_group)))
    _, label = connected_components(graph, directed=False)
    joined = near & np.isin(label[group], np.unique(label[rim_group.nonzero()[0]]))
    ends = np.unique(rim.reshape(-1))
    with np.errstate(invalid="ignore"):
        over = p[ends, 2] - (np.asarray(under, dtype=np.float64)[ends] + gap)
    over = np.where(np.isfinite(over), np.maximum(over, 0.0), 0.0)
    if most is not None:      # a cut end standing further over the face anywhere is a strand: left as it is
        loops = coo_matrix((np.ones(len(rim)), (group[rim[:, 0]], group[rim[:, 1]])),
                           shape=(len(near_group), len(near_group)))
        _, loop = connected_components(loops, directed=False)
        worst = np.zeros(loop.max() + 1)
        np.maximum.at(worst, loop[group[ends]], over)
        over[worst[loop[group[ends]]] > most] = 0.0
    if not over.any() or not joined.any():
        return p, moved
    which = np.flatnonzero(joined)
    _, nearest = cKDTree(p[ends]).query(p[which], workers=-1)
    w = np.where(on_rim[which], 1.0, np.clip(1.0 + values[which] / band, 0.0, 1.0))
    moved[which] = w * w * (3.0 - 2.0 * w) * over[nearest]
    p[:, 2] -= moved
    return p, moved


def reference_depth(own: np.ndarray, new: np.ndarray, dive: float = np.inf) -> np.ndarray:
    """What a figure's face is measured against: the nearer-to-the-back of its own front depth and the new face's,
    where both are (+inf where neither is, so nothing there counts as in front). The new face counts only where it
    lies less than DIVE behind the figure's front: where the build shows no face, its relief falls to the build's own
    back of the head, and measured against that the cut went through the figure's head and out of its back hair, a
    see-through hole a game shows from behind (skin_sorcerer_c, its new face up to 0.9 head widths behind its lower
    cheeks; round 27 of the AssetFurnace Support notes)."""
    out = np.where(np.isfinite(own), own, np.inf)
    both = np.isfinite(own) & np.isfinite(new)
    with np.errstate(invalid="ignore"):
        both &= own - new < dive
    out[both] = np.minimum(own[both], new[both])
    only_new = np.isfinite(new) & ~np.isfinite(own)
    out[only_new] = new[only_new]
    return out


def fill_pits(part: dict, new_depth: np.ndarray, is_face: np.ndarray, surface: np.ndarray, window: dict,
              limit: float, blend: float, gap: float) -> tuple[dict, float, np.ndarray]:
    """(PART, the new face, with its pits filled; the share of the window's mask they take; where they are, a mask of
    NEW_DEPTH's pixels). A pit is where the new
    face is there (NEW_DEPTH, its front depth map) and is no face: off a cliff from the nose (IS_FACE, `reached`), or
    LIMIT or more under SURFACE (the figure's own front without its strands, `seam_surface`). There the face takes that
    surface's depth less GAP, but never higher than the face round the pit's rim, and within BLEND round a pit it eases
    toward it (`tuck`, rising freely): a pit becomes the figure's own shape, painted by the picture like the rest of the
    face.

    Pixal3D shows no face where the picture's face is dark or thin: built alone, skin_sorcerer_c's face fell 3 to 8 cm
    into the head beside its jaw. The figure's old skin was cut away there and the pit showed the old head's inside as
    grey, white and maroon patches (rounds 27 and 28 of the AssetFurnace Support notes); keeping the old skin there
    instead left a crumpled piece of the old jaw inside the new face, its shape and paint unlike the face round it
    (round 28, TODO 207). Filled, a pit lies just behind the figure's surface, so the cut measuring against it there
    (`reference_depth`) never takes the back of the head with it."""
    surface = np.asarray(surface, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        deep = np.isfinite(surface) & np.isfinite(new_depth) & (surface - new_depth >= limit)
    pit = np.isfinite(new_depth) & np.isfinite(surface) & (~np.asarray(is_face, bool) | deep)
    if not pit.any():
        return part, 0.0, pit
    box, pixel = window["box"], window["pixel"]
    p = np.asarray(part["positions"], dtype=np.float64)
    f = np.asarray(part["faces"], dtype=np.int64)
    # a point within one of the face's own steps of a pit is filled in full: laid out coarser (`matched_step`), a
    # point next to the pit kept its depth and its triangles fell into the pit as a wall (round 28)
    step = float(np.median(np.linalg.norm(p[f[:, 0], :2] - p[f[:, 1], :2], axis=1))) if len(f) else 0.0
    away = sample(signed_distance(~pit, pixel), box, pixel, p, outside=FAR) - step    # at most 0 in a pit
    # filled up to its rim at most: where the figure's surface stands higher (a hood's edge in front of the jaw),
    # following it raised a spike of the face, painted as skin (round 28)
    rim = spread(np.where(pit, -np.inf, new_depth), sum(pit.shape))
    level = np.where(np.isfinite(rim), np.fmin(surface, rim), surface)
    target = sample(np.where(np.isfinite(level), level, np.nan), box, pixel, p, outside=np.nan)
    share = float((pit & window["mask"]).sum() / max(int(window["mask"].sum()), 1))
    return {**part, "positions": tuck(p, away, target, blend, np.inf, gap)}, share, pit


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


def part_paint(part: dict) -> np.ndarray | None:
    """The base colour picture of PART's material as an RGB array (0..255), or None when it has none (or no UVs)."""
    import io

    from PIL import Image

    if part.get("material") is None or part.get("uvs") is None:
        return None
    doc, binary, index = part["material"]
    texture = doc["materials"][index].get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if texture is None:
        return None
    entry = doc["textures"][texture["index"]]
    sources = ([entry["source"]] if "source" in entry else []) + [
        extension["source"] for extension in entry.get("extensions", {}).values() if "source" in extension]
    images = [doc["images"][s] for s in sources if "bufferView" in doc["images"][s]]   # a WebP one sits in an extension
    if not images:
        return None
    image = images[0]
    view = doc["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    with Image.open(io.BytesIO(binary[start:start + view["byteLength"]])) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.float64)


def paint_at(paint: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """PAINT (an RGB array) at glTF texture coordinates (u right, v down), the nearest texel."""
    h, w = paint.shape[:2]
    uv = np.asarray(uvs, dtype=np.float64)
    x = np.clip(np.floor(uv[:, 0] * w).astype(np.int64), 0, w - 1)
    y = np.clip(np.floor(uv[:, 1] * h).astype(np.int64), 0, h - 1)
    return paint[y, x]


def front_uvs(positions: np.ndarray, faces: np.ndarray, uvs: np.ndarray, owner: np.ndarray, rows: np.ndarray,
              cols: np.ndarray, box, pixel: float) -> tuple[np.ndarray, np.ndarray]:
    """((x, y), texture coordinates) of the pixel centres ROWS, COLS of a front view over BOX, each on its OWNER
    triangle (`depth_map`): its corners' UVs mixed by where the centre lies in it seen from the front."""
    x = box[0] + (np.asarray(cols) + 0.5) * pixel
    y = box[3] - (np.asarray(rows) + 0.5) * pixel
    f = np.asarray(faces, dtype=np.int64)[owner]
    t = np.asarray(positions, dtype=np.float64)[f][:, :, :2]
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    flat = np.abs(den) < 1e-18
    den = np.where(flat, 1.0, den)
    w0 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / den
    w1 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / den
    w = np.where(flat[:, None], [1.0, 0.0, 0.0], np.clip(np.stack([w0, w1, 1.0 - w0 - w1], axis=1), 0.0, 1.0))
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    uv = np.einsum("nk,nkd->nd", w, np.asarray(uvs, dtype=np.float64)[f])
    return np.stack([x, y], axis=1), uv


def seam_paint(parts: list[dict], window: dict, crop_rgb: np.ndarray, crop: tuple[int, int, int, int], frame,
               ring: int = RING_PIXELS) -> tuple[np.ndarray, np.ndarray]:
    """(points, offsets): the figure's own surface in a band RING pixels wide just outside the seam, and there the
    figure's paint less the crop's (RGB). The crop is the picture as it is, and a figure's paint is often lighter or
    darker (the Zauberer's was 12 to 36 levels lighter all round the seam), so the new face's outline showed. Read at
    every pixel of the band the figure shows there (`front_uvs`) as well as at its points: on the Nachtschwinge's 8k
    rigged model a few points lay in the band, and its cheek, lighter than the crop by 5 L*, still drew a line down the
    seam (round 28 of the AssetFurnace Support notes)."""
    pixel, box = window["pixel"], window["box"]
    points, offsets = [], []
    ring_px = (window["field"] <= 0) & (window["field"] > -ring * pixel)
    for part in parts:
        paint = part_paint(part)
        if paint is None:
            continue
        p = np.asarray(part["positions"], dtype=np.float64)
        values = window_values(window, p)
        seen = sample(window["depth"], box, pixel, p, outside=-np.inf, nearest=True)
        band = (values <= 0) & (values > -ring * pixel) & (p[:, 2] > seen - ring * pixel)
        if band.any():
            own = paint_at(paint, np.asarray(part["uvs"])[band])
            new = paint_at(crop_rgb, picture_uvs(p[band], crop, crop_rgb.shape[1], frame))
            points.append(p[band, :2])
            offsets.append(own - new)
        depth, owner = depth_map(p, part["faces"], box, pixel)
        shown = ring_px & (owner >= 0) & (depth > window["depth"] - ring * pixel)
        if shown.any():
            rows, cols = np.nonzero(shown)
            xy, uv = front_uvs(p, part["faces"], part["uvs"], owner[rows, cols], rows, cols, box, pixel)
            new = paint_at(crop_rgb, picture_uvs(np.c_[xy, np.zeros(len(xy))], crop, crop_rgb.shape[1], frame))
            points.append(xy)
            offsets.append(paint_at(paint, uv) - new)
    if not points:
        return np.zeros((0, 2)), np.zeros((0, 3))
    return np.concatenate(points), np.concatenate(offsets)


def offset_field(window: dict, points: np.ndarray, offsets: np.ndarray, limit: float = PAINT_LIMIT) -> np.ndarray:
    """An RGB offset for every pixel of the window's picture: the seam's offsets (`seam_paint`, each clipped to
    LIMIT) on the band just outside the seam, and inside it the smoothest field that meets them there (Laplace's
    equation, as in seamless cloning), so a crop darker all round turns lighter all over and the face's own detail
    stays. Pixels further out take the nearest value. Zero where no seam point lies near."""
    field = window["field"]
    rows, cols = field.shape
    pixel = window["pixel"]
    out = np.zeros((rows, cols, 3))
    if not len(points):
        return out
    r, c = np.nonzero((field <= 0) & (field > -RING_PIXELS * pixel))
    x = window["box"][0] + (c + 0.5) * pixel
    y = window["box"][3] - (r + 0.5) * pixel
    distance, nearest = cKDTree(np.asarray(points)[:, :2]).query(np.stack([x, y], axis=1), workers=-1)
    near = distance <= RING_PIXELS * pixel
    fixed = np.zeros((rows, cols), bool)
    fixed[r[near], c[near]] = True
    out[r[near], c[near]] = np.clip(np.asarray(offsets, dtype=np.float64)[nearest[near]], -limit, limit)
    inside = field > 0
    if fixed.any() and inside.any():
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve

        index = np.full((rows, cols), -1, dtype=np.int64)
        index[inside] = np.arange(int(inside.sum()))
        ir, ic = np.nonzero(inside)
        diagonal = np.zeros(len(ir))
        rhs = np.zeros((len(ir), 3))
        a, b = [], []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = ir + dr, ic + dc
            on = (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
            nr, nc = np.where(on, nr, 0), np.where(on, nc, 0)
            free = on & inside[nr, nc]
            held = on & fixed[nr, nc] & ~inside[nr, nc]
            diagonal += free | held
            a.append(np.flatnonzero(free))
            b.append(index[nr[free], nc[free]])
            rhs[held] += out[nr[held], nc[held]]
        lone = diagonal == 0
        diagonal[lone] = 1.0
        a, b = np.concatenate(a), np.concatenate(b)
        n = len(ir)
        matrix = coo_matrix((np.r_[diagonal, -np.ones(len(a))], (np.r_[np.arange(n), a], np.r_[np.arange(n), b])),
                            shape=(n, n)).tocsr()
        solved = np.stack([spsolve(matrix, rhs[:, k]) for k in range(3)], axis=1)
        out[ir, ic] = solved
        fixed |= inside
    if fixed.any():
        _, (nr, nc) = ndimage.distance_transform_edt(~fixed, return_indices=True)
        out = out[nr, nc]
    return out


def matched_picture(crop_rgb: np.ndarray, window: dict, offsets: np.ndarray, crop: tuple[int, int, int, int],
                    frame) -> np.ndarray:
    """CROP_RGB (the upscaled crop, as the face's paint) with OFFSETS (`offset_field`) added where each of its pixels
    lands on the figure (`crop_to_figure`): the crop's paint meets the figure's at the seam."""
    h, w = crop_rgb.shape[:2]
    v, u = np.mgrid[0:h, 0:w]
    xy = crop_to_figure(np.stack([u.ravel() + 0.5, v.ravel() + 0.5], axis=1), crop, w, frame)
    points = np.c_[xy, np.zeros(len(xy))]
    shift = np.stack([sample(offsets[..., k], window["box"], window["pixel"], points, outside=0.0)
                      for k in range(3)], axis=1)
    out = np.asarray(crop_rgb, dtype=np.float64).reshape(-1, 3) + shift
    return np.clip(np.round(out), 0, 255).astype(np.uint8).reshape(h, w, 3)


def match_face_paint(face: dict, parts: list[dict], window: dict, crop_rgb: np.ndarray,
                     crop: tuple[int, int, int, int], frame) -> tuple[dict, list[float] | None]:
    """(FACE painted by the crop moved to the figure's own paint at the seam, the median offset the seam asked for in
    levels per channel): `seam_paint` on PARTS (the figure before the graft), `offset_field`, `matched_picture`. FACE
    as it is and None when the figure carries no paint there."""
    import io

    from PIL import Image

    points, offsets = seam_paint(parts, window, crop_rgb, crop, frame)
    if not len(points):
        return face, None
    picture = matched_picture(crop_rgb, window, offset_field(window, points, offsets), crop, frame)
    buffer = io.BytesIO()
    Image.fromarray(picture, "RGB").save(buffer, format="PNG")
    median = np.median(np.clip(offsets, -PAINT_LIMIT, PAINT_LIMIT), axis=0)
    return {**face, "material": picture_material(buffer.getvalue())}, [float(v) for v in median]


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
    reach = (NEAR_PIXELS + 1) * pixel + 2 * (step - pixel)     # a coarser layout reaches a little further, so its
    inside = window_values(window, np.c_[grid, np.zeros(len(grid))]).reshape(rows, cols) > -reach   # skirt is whole
    depth = round_relief(depth, inside, rounding / step, JUMP_SHARE * window["width"])
    out = relief(depth, box, step, inside)
    out["uvs"] = picture_uvs(out["positions"], crop, crop_size, frame)
    if material is not None:
        out["material"] = material
    return out


def relief_step(parts: list[dict], window: dict, most: int = COARSEST_STEP) -> float:
    """How finely a model's new face is laid out: half the median edge of the model's own face inside the window (its
    skin seen from the front, not the back of the head behind it), at least the window's pixel and at most MOST
    pixels. A relief at the window's pixel gave the Zauberer's 40k rigged model a face of 47,000 triangles where its
    old one had 650; a finer model (the paint source) keeps a finer one."""
    pixel = window["pixel"]
    lengths = []
    for part in parts:
        p, f = np.asarray(part["positions"], dtype=np.float64), np.asarray(part["faces"], dtype=np.int64)
        if not len(f):
            continue
        middle = p[f].mean(axis=1)
        seen = sample(window["depth"], window["box"], pixel, middle, outside=np.inf, nearest=True)
        inside = (window_values(window, p)[f] > 0).all(axis=1) & (middle[:, 2] > seen - NEAR_PIXELS * pixel)
        if inside.any():
            t = p[f[inside]]
            lengths.append(np.linalg.norm(t - np.roll(t, 1, axis=1), axis=2).ravel())
    if not lengths:
        return pixel
    return float(np.clip(np.median(np.concatenate(lengths)) / 2, pixel, most * pixel))


MATCHED_COARSEST = 8     # a face laid out coarser to match its head (`matched_step`) stays at most this many pixels apart


def matched_step(step: float, faces: int, target: int | None, pixel: float, most: int = MATCHED_COARSEST) -> float:
    """The relief step that gives a face of FACES triangles (laid out at STEP) about TARGET: a relief's triangles go
    with one over the step squared. Never finer than STEP, never coarser than MOST pixels; STEP as it was when TARGET
    is None (the model's density unknown) or the face is not finer than it. The target is the rule every Hero part
    takes (`hero_parts.matched_faces`: no finer than 1.5 times the model round it): at `relief_step`'s cap of
    COARSEST_STEP pixels skin_sorcerer_c's face had 4,705 triangles where the head it replaced had about 2,026, and the
    game step, keeping them, thinned the body about 30 % harder (round 27 of the AssetFurnace Support notes)."""
    if target is None or faces <= target:
        return step
    return float(min(step * np.sqrt(faces / target), most * pixel))


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
               blend: float | None = None, ease: float | None = None,
               band: float | None = None) -> tuple[list[dict], dict, dict]:
    """PARTS (one model of the figure, part by part, in the figure's frame) with their face cut out along the
    window's seam and the placed new FACE set in. Returns (the model's parts cut, the new face part, report).

    What goes of the figure: inside the window, whatever is not more than BEHIND (default BEHIND_SHARE of the head's
    width) behind the new face or the figure's own front, so a strand in front of an eye, the eye under it and an
    inner skin go, while a collar hidden behind the jaw stays. The new face counts there only where it is a face,
    reached from the nose without a cliff (`reached`) and less than DIVE_SHARE of the head's width behind the figure's
    front (`reference_depth`), so a new face that falls into the head where its build showed no face never takes the
    back of the head with it; the report's off_cliff_share says how much of the face that is. The new face reaches
    SKIRT_PIXELS past the seam and is tucked just behind the figure's surface there, its front without the strands
    standing off it (`seam_surface`; `tuck`, at most EASE toward the front, default EASE_SHARE of the head's width),
    so the figure's cut edge lies over it. Where that edge still stands in front of the face (a strand's cut end, a
    rise past the cap), the figure's own surface eases back onto the face over BAND (default SEAM_BAND_SHARE of the
    head's width, `ease_onto`): no edge hanging over the face, no gap to see into the head through; the report says
    how far the face rose (seam_rise_max) and the figure was eased (seam_eased_max). Not stitched: Pixal3D builds skin
    and inner wall as one surface joined all over (the Zauberer's figure and face both), so neither side's cut gives
    one clean rim, and a stitch between two such rims tore across the face. With skin weights (dense) the new face
    takes the weights of the nearest old face point."""
    width = window["width"]
    behind = BEHIND_SHARE * width if behind is None else behind
    blend = BLEND_SHARE * width if blend is None else blend
    cap = EASE_SHARE * width if ease is None else ease
    band = SEAM_BAND_SHARE * width if band is None else band
    box, pixel = window["box"], window["pixel"]
    skirt = SKIRT_PIXELS * pixel
    def set_in(window):
        new, _ = hh.cut_part_values(face, -(window_values(window, face["positions"]) + skirt))
        if not len(new["faces"]):
            raise ValueError("the new face does not reach into the window: check its placement")
        new, _ = hh.keep_faces(new, front_piece(new["positions"], new["faces"]))
        new_depth = spread(depth_map(new["positions"], new["faces"], box, pixel)[0], NEAR_PIXELS)
        # the cut measures against the new face only where it is a face: reached from the nose without a cliff
        return new, new_depth, reached(new_depth, window["seed"], JUMP_SHARE * width)

    new, new_depth, is_face = set_in(window)
    shown = np.isfinite(new_depth) & (window["field"] > 0)
    out_parts, old_points, old_dense, report = [], [], [], {"parts": []}
    report["off_cliff_share"] = float((shown & ~is_face).sum() / max(int(shown.sum()), 1))
    # where the new face is no face, or lies deeper under the figure's surface than it may rise at a seam, it takes
    # that surface's shape, still painted by the picture
    new, report["face_filled_share"], pit = fill_pits(
        new, new_depth, is_face, seam_surface(window["depth"], STRAND_SHARE * width / pixel, TUCK_SHARE * width),
        window, min(DIVE_SHARE * width, cap), blend, TUCK_SHARE * width)
    if pit.any():      # filled, a pit is face: the cut measures against it there, so nothing of the old pokes through
        new_depth = spread(depth_map(new["positions"], new["faces"], box, pixel)[0], NEAR_PIXELS)
        is_face = is_face | pit
    counted = np.where(is_face, new_depth, -np.inf)
    body_index, target, most, rims = None, None, 0, []
    for index, part in enumerate(parts):
        p, f = np.asarray(part["positions"], dtype=np.float64), np.asarray(part["faces"], dtype=np.int64)
        values = window_values(window, p)
        middle = p[f].mean(axis=1)
        reach = (values[f] > -skirt - NEAR_PIXELS * pixel).any(axis=1)
        if not reach.any():
            out_parts.append(part)
            rims.append(None)
            report["parts"].append({"cut": False})
            continue
        own, _ = depth_map(p, f[reach], box, pixel)
        ref = sample(reference_depth(own, counted, DIVE_SHARE * width), box, pixel, middle, outside=np.inf,
                     nearest=True)
        cuttable = reach & (middle[:, 2] > ref - behind)
        gone = cuttable & (values[f] > 0).all(axis=1)
        if part.get("dense") is not None and gone.any():
            used = np.unique(f[gone])
            old_points.append(p[used])
            old_dense.append(np.asarray(part["dense"])[used])
        in_front = middle[gone, 2] - sample(counted, box, pixel, middle[gone], outside=-np.inf, nearest=True)
        cut, rim = hh.cut_part_values(part, values, cuttable)
        report["parts"].append({"cut": True, "faces_removed": int(gone.sum()),
                                "old_in_front_max": float(np.max(in_front[np.isfinite(in_front)], initial=0.0))})
        if gone.sum() > most:
            body_index, most = len(out_parts), int(gone.sum())
            # the surface the face goes on into: the figure's own front, a rise toward it no higher than that front
            # without its strands, nor than a little over it carried in from the seam along its slope
            surface = spread(seam_surface(own, STRAND_SHARE * width / pixel, TUCK_SHARE * width), NEAR_PIXELS)
            carried = seam_target(surface, window["field"], int(np.ceil(blend / pixel)) + 1, pixel)
            target = rise_target(spread(own, NEAR_PIXELS), surface, carried, new_depth, OVER_SHARE * width)
        out_parts.append(cut)
        rims.append(rim)
    if target is None:
        raise ValueError("nothing of the model lies in the window: check the window")
    values = window_values(window, new["positions"])
    at = sample(np.where(np.isfinite(target), target, np.nan), box, pixel, new["positions"], outside=np.nan)
    seam = np.abs(values) <= pixel
    gap = TUCK_SHARE * width
    gaps = (at - gap - new["positions"][:, 2])[seam]
    gaps = gaps[np.isfinite(gaps)]
    before = new["positions"][:, 2]
    new = {**new, "positions": tuck(new["positions"], values, at, blend, cap, gap)}
    rise = (new["positions"][:, 2] - before)[seam]
    continued = face_continued(new["positions"], new["faces"], window, SKIRT_PIXELS + NEAR_PIXELS)
    continued = sample(np.where(np.isfinite(continued), continued, np.nan), box, pixel, new["positions"],
                       outside=np.nan)
    new = {**new, "positions": skirt_under(new["positions"], values, continued, at, gap)}
    # the seam closed: where the figure's cut edge still stands in front of the face (a strand's cut end, a rise past
    # the cap), its own surface eases back onto the face
    under, _ = depth_map(new["positions"], new["faces"], box, pixel)
    under = np.where(np.isfinite(under), under, np.nan)
    edge_over, rim_moved = [], []
    for index, rim in enumerate(rims):
        if rim is None or not len(rim):
            continue
        part = out_parts[index]
        beneath = sample(under, box, pixel, part["positions"], outside=np.nan)
        eased, by = ease_onto(part["positions"], part["faces"], rim, window_values(window, part["positions"]),
                              beneath, band, gap, STRAND_EASE_SHARE * width)
        report["parts"][index]["eased_max"] = float(by.max(initial=0.0))
        ends = np.unique(rim)
        with np.errstate(invalid="ignore"):
            edge_over.append((eased[ends, 2] - beneath[ends])[np.isfinite(beneath[ends])])
        rim_moved.append(by[ends])
        if by.any():
            out_parts[index] = {**part, "positions": eased}
            if part.get("normals") is not None:
                from image_to_3dlab.pixal3d_native import smooth_normals

                normals = np.array(part["normals"], dtype=np.float64)
                normals[by > 0] = smooth_normals(eased, part["faces"])[by > 0]
                out_parts[index]["normals"] = normals
    edge_over = np.concatenate(edge_over) if edge_over else np.zeros(0)
    rim_moved = np.concatenate(rim_moved) if rim_moved else np.zeros(0)
    report.update({"seam_rise_max": float(rise.max(initial=0.0)),
                   "seam_eased_max": float(rim_moved.max(initial=0.0)),
                   "seam_eased_share": float(np.mean(rim_moved > gap)) if len(rim_moved) else 0.0,
                   "seam_edge_over_max": float(edge_over.max(initial=0.0))})
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
