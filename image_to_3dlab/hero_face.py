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
stay the figure's own. Both models are cut along that outline (`hero_hands.clip_values`); the new face is eased in
depth to meet the figure's rim and stitched to it (`hero_hands.join_rims`), so there is no overlap and no gap; a
rigged figure's new face takes the old face's skin weights. `scripts/hero_face.py` runs the steps.

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
SEED_BAND = 0.2        # the nose is looked for within this share of the head's width either side of its middle
JUMP_SHARE = 0.25      # a step in depth of this share of the head's width between neighbouring pixels is an edge where
                       # the face goes behind something: the Zauberer's brim drops 0.065 onto his forehead and his jaw
                       # 0.04 onto the collar behind it (head width 0.075); the strand in front of his eye stands 0.01
CLIFF_PIXELS = 3       # ... and so is that step spread over up to this many pixels (`cliffs`)
FACING = 0.2           # the face faces the camera at least this much (the cosine): its sides and the chin's underside,
                       # which turn away, end it
SIDE_SHARE = 0.45      # the face reaches at most this share of the head's width either side of the nose
CHIN_SHARE = 0.6       # and at most this share of it below the nose (the Zauberer's chin: 0.42)
MARGIN_SHARE = 0.06    # the seam runs this share of the head's width inside the outline
SMOOTH_SHARE = 0.04    # the seam's line is smoothed over this share of the head's width
BEHIND_SHARE = 0.15    # what of the figure lies less than this share of the head's width behind the new face goes: the
                       # old eye under a strand, an inner skin; a collar hidden behind the jaw lies deeper and stays
NEAR_PIXELS = 3        # the new face's depth is carried this many pixels past its outline, for the faces the seam cuts
BLEND_SHARE = 0.12     # the new face eases into the figure's depth over this share of the head's width
CROP_PAD = 2.0         # the face's crop is this many times the window's size: hair, brim and collar round the face, so
                       # Pixal3D builds a head and not a mask (the Zauberer's 174 px crop round an 85 px face)
ICP_KEEP = 0.75        # the line-up uses the closest three quarters of the point pairs: the old strand is no match
ICP_STEPS = 60
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

def face_seed(depth: np.ndarray, box, pixel: float, centre_x: float, band: float,
              floor: float | None, bottom: float) -> tuple[int, int]:
    """(row, column) of the nose: the nearest pixel within BAND either side of CENTRE_X, between BOTTOM and FLOOR
    (a brim's underside, whose front edge stands nearer than any nose)."""
    rows, cols = depth.shape
    x = box[0] + (np.arange(cols) + 0.5) * pixel
    y = box[3] - (np.arange(rows) + 0.5) * pixel
    allowed = (np.abs(x - centre_x) <= band)[None, :] & (y >= bottom)[:, None]
    if floor is not None:
        allowed &= (y < floor)[:, None]
    candidates = np.where(allowed & np.isfinite(depth), depth, -np.inf)
    if not np.isfinite(candidates).any():
        raise ValueError("no surface in front of the head's middle: is the model facing +Z?")
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
    seed = face_seed(depth, box, pixel, centre_x, SEED_BAND * width, floor, bottom)
    reach = face_outline(depth, facing, seed, JUMP_SHARE * width)
    top, chin = column_reach(reach, seed)
    chin = min(chin, seed[0] + int(round(CHIN_SHARE * width / pixel)))
    outline = reach & face_shape(depth.shape, seed, top, chin, SIDE_SHARE * width / pixel)
    mask = window_mask(outline, MARGIN_SHARE * width / pixel, SMOOTH_SHARE * width / pixel, seed)
    return {"box": box, "pixel": pixel, "field": signed_distance(mask, pixel), "depth": depth, "outline": outline,
            "mask": mask, "seed": seed, "width": width, "floor": floor, "rows": (top, chin),
            "axis_z": float(np.median(hv[below, 2])), "centre_x": centre_x}


FAR = 1e6                  # how far outside the window a point off its picture counts: finite, so a cut between it
                           # and a point inside still falls on the edge (an infinity made the edge's mix NaN)


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


def solid_cutout(picture: np.ndarray, cutout: np.ndarray) -> tuple[np.ndarray, int]:
    """(the cut-out RGBA with every hole inside its outline made solid again from PICTURE's RGB; how many pixels).

    The background remover took the Zauberer's glowing eyes for background (2026-11-22, his face crop at 156 px): a
    build from that has holes, or dark paint, where the eyes should glow. A face crop holds no real hole."""
    out = np.array(cutout, dtype=np.uint8, copy=True)
    solid = out[..., 3] > 127
    holes = ndimage.binary_fill_holes(solid) & ~solid
    out[holes, :3] = np.asarray(picture)[holes][:, :3]
    out[holes, 3] = 255
    return out, int(holes.sum())


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

def turn_about(axis_angle: np.ndarray) -> np.ndarray:
    """The 3x3 turn by |AXIS_ANGLE| radians about its direction (Rodrigues)."""
    w = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(w))
    if angle < 1e-15:
        return np.eye(3)
    k = w / angle
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def plane_step(points: np.ndarray, targets: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """The small 4x4 turn and shift that best slides POINTS onto the planes through TARGETS with NORMALS (one
    linearised point-to-plane step, turned about the points' middle)."""
    middle = points.mean(axis=0)
    p = points - middle
    a = np.concatenate([np.cross(p, normals), normals], axis=1)
    b = -((points - targets) * normals).sum(axis=1)
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    turn = turn_about(x[:3])
    tf = np.eye(4)
    tf[:3, :3] = turn
    tf[:3, 3] = middle + x[3:] - turn @ middle
    return tf


def trimmed_icp(source: np.ndarray, target: np.ndarray, normals: np.ndarray, keep: float = ICP_KEEP,
                steps: int = ICP_STEPS) -> tuple[np.ndarray, float]:
    """(4x4 transform, mean distance of the pairs kept): SOURCE turned and shifted onto the surface through TARGET
    (whose NORMALS are given), point to plane, each step fitted on the KEEP share of nearest pairs that lie closest,
    so a part that differs (the old strand in front of an eye) does not pull the rest. Point to point slid along a
    smooth face onto its sample grid instead and stopped millimetres off; its size stays the crop's."""
    s = np.asarray(source, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)
    tree = cKDTree(t)
    tf = np.eye(4)
    for _ in range(steps):
        moved = hh.moved(tf, s)
        distance, nearest = tree.query(moved, workers=-1)
        use = distance <= np.quantile(distance, keep)
        step = plane_step(moved[use], t[nearest[use]], n[nearest[use]])
        tf = step @ tf
        if np.linalg.norm(step[:3, 3]) < 1e-9 * max(np.ptp(t, axis=0).max(), 1e-12) and np.allclose(
                step[:3, :3], np.eye(3), atol=1e-10):
            break
    distance, _ = tree.query(hh.moved(tf, s), workers=-1)
    return tf, float(distance[distance <= np.quantile(distance, keep)].mean())


# --- Cutting and stitching ------------------------------------------------------------------------------------------

def front_piece(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Which faces belong to the piece seen most from the front: the most area facing the camera. After the window's
    cut, a face built alone falls apart into its front, the back of its head, an inner skin and specks of brim or
    collar the cut caught; only the front belongs in the figure's face."""
    v = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    if not len(v):
        return np.zeros(0, bool)
    label = hh.components(faces, hh.weld_groups(vertices))
    toward = ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
              - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))   # twice the area seen from the front
    score = np.bincount(label, weights=np.clip(toward, 0.0, None))
    return label == int(np.argmax(score))


def front_rim(vertices: np.ndarray, rim: np.ndarray, centre: np.ndarray, share: float = 0.8):
    """(vertex ids in walking order, points) of the rim round the face: of the closed outlines a cut left that hold
    CENTRE seen from the front and enclose at least SHARE of the largest such area, the nearest to the camera. An
    inner skin a millimetre behind the face is cut along the same line and encloses the same area; it lies deeper."""
    if not len(rim):
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    group = hh.weld_groups(np.asarray(vertices))
    first = np.zeros(group.max() + 1, dtype=np.int64)
    first[group[::-1]] = np.arange(len(group))[::-1]
    edges = group[np.asarray(rim)]
    directed = {(int(a), int(b)) for a, b in edges}
    loops = [loop for loop in hh.chain_loops(edges)
             if len(loop) >= 3 and (int(loop[-1]), int(loop[0])) in directed]          # closed ones only
    points = [np.asarray(vertices)[first[loop]] for loop in loops]
    holding = [(i, hh.loop_area(p, FRONT)) for i, p in enumerate(points) if hh.contains(p, FRONT, centre)]
    if not holding:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    largest = max(area for _, area in holding)
    best = max((i for i, area in holding if area >= share * largest), key=lambda i: points[i][:, 2].mean())
    return first[loops[best]], points[best]


def ease_depth(points: np.ndarray, values: np.ndarray, rim_xy: np.ndarray, rim_dz: np.ndarray,
               blend: float) -> np.ndarray:
    """POINTS moved in depth by the change RIM_DZ of the rim point nearest each one (seen from the front), in full at
    the seam (VALUES, the distance inside the window, 0) and fading out smoothly BLEND inside. The new face then
    meets the figure's rim where they differ: hair under a brim, a cheek the line-up left a little proud."""
    p = np.array(points, dtype=np.float64)
    if not len(rim_xy) or not len(p):
        return p
    _, nearest = cKDTree(np.asarray(rim_xy, dtype=np.float64)).query(p[:, :2], workers=-1)
    w = np.clip(1.0 - np.asarray(values, dtype=np.float64) / max(blend, 1e-12), 0.0, 1.0)
    p[:, 2] += np.asarray(rim_dz, dtype=np.float64)[nearest] * (w * w * (3.0 - 2.0 * w))
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
               blend: float | None = None) -> tuple[list[dict], dict, dict]:
    """PARTS (one model of the figure, part by part, in the figure's frame) with their face cut out along the
    window's seam and the placed new FACE set in and stitched to the rim. Returns (the model's parts cut, the new face
    part with its stitch, report).

    What goes of the figure: inside the window, whatever is not more than BEHIND (default BEHIND_SHARE of the head's
    width) behind the new face or the figure's own front, so a strand in front of an eye, the eye under it and an
    inner skin go, while a collar hidden behind the jaw stays. With skin weights (dense) the new face takes the
    weights of the nearest old face point."""
    width = window["width"]
    behind = BEHIND_SHARE * width if behind is None else behind
    blend = BLEND_SHARE * width if blend is None else blend
    box, pixel = window["box"], window["pixel"]
    new, new_rim = cut_face(face, window)
    if not len(new["faces"]):
        raise ValueError("the new face does not reach into the window: check its placement")
    new_depth, _ = depth_map(new["positions"], new["faces"], box, pixel)
    new_depth = spread(new_depth, NEAR_PIXELS)
    centre = np.array([box[0] + (window["seed"][1] + 0.5) * pixel, box[3] - (window["seed"][0] + 0.5) * pixel, 0.0])
    out_parts, old_points, old_dense, report = [], [], [], {"parts": []}
    body_index, body_rim, body = None, None, None
    for index, part in enumerate(parts):
        p, f = np.asarray(part["positions"], dtype=np.float64), np.asarray(part["faces"], dtype=np.int64)
        values = window_values(window, p)
        middle = p[f].mean(axis=1)
        reach = (values[f] > -NEAR_PIXELS * pixel).any(axis=1)
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
        cut, rim = hh.cut_part_values(part, values, cuttable)
        ids, loop = front_rim(cut["positions"], rim, centre)
        report["parts"].append({"cut": True, "faces_removed": int(gone.sum()), "rim_points": int(len(ids)),
                                "rim_pieces": len(hh.rim_pieces(cut["positions"], rim)),
                                "old_in_front_max": float(np.max(in_front[np.isfinite(in_front)], initial=0.0))})
        if len(ids) and (body_rim is None or len(ids) > len(body_rim[0])):
            body_index, body_rim, body = len(out_parts), (ids, loop), cut
        out_parts.append(cut)
    if body is None:
        raise ValueError("no part of the model has a closed rim round the face: check the window")
    face_ids, face_loop = front_rim(new["positions"], new_rim, centre)
    if not len(face_ids):
        raise ValueError("the new face's cut left no closed rim round the face")
    body_ids, body_loop = body_rim
    _, nearest = cKDTree(body_loop[:, :2]).query(face_loop[:, :2], workers=-1)
    rim_dz = body_loop[nearest, 2] - face_loop[:, 2]
    values = window_values(window, new["positions"])
    new = {**new, "positions": ease_depth(new["positions"], np.maximum(values, 0.0), face_loop[:, :2], rim_dz, blend)}
    if old_points:
        new["dense"] = hh.borrowed_weights(new["positions"], np.concatenate(old_points), np.concatenate(old_dense))
        if body.get("dense") is None:
            new.pop("dense")
    stitched, band, alike = hh.join_rims(body, body_ids, new, face_ids, FRONT)
    report.update({"body_part": body_index, "rim_points": [int(len(body_ids)), int(len(face_ids))],
                   "stitch_faces": band, "rims_turn_alike": alike,
                   "ease_max": float(np.abs(rim_dz).max()), "ease_median": float(np.median(np.abs(rim_dz))),
                   "face_faces": int(len(new["faces"]))})
    return out_parts, stitched, report


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
