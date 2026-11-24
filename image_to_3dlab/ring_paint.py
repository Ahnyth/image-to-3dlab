"""Repeat a band's seen arc round the band, where no picture sees it.

A hat band, a belt or a bracelet is a ring. The front picture sees its pattern (a hat band's runes, a belt's studs)
only on the front arc; Pixel Match (`photo_paint`) trusts a picture only where the surface faces it, so the ring's
sides and back keep the generator's own paint, which redraws the pattern as a blur. The Zauberer's hat band showed
sharp runes in front and blue smears on its sides (2026-11-23): the front camera meets the band's sides at grazing
angles, where the picture's pixels are not used at all.

A ring's pattern goes on round it, so the seen arc is repeated round the rest: the ring found from its own surface
(its middle, its radius and its tilt), the arc the picture trusts cut where it holds the least detail (between two
runes, not through one), and every texel of the ring the picture does not trust takes the colour of the seen arc at
the same height, one arc-length further back. Only the marks travel: each keeps its difference from the band's plain
colour where it came from and takes the plain colour where it lands, a slow trend round the ring, so the picture's
painted light (one end of the arc darker than the other) leaves no seam at the joins. What another picture sees (a
drawn back view) keeps its paint. Only the ring's outer wall inside the given box changes.

Pure numpy + scipy; the GLB helpers change only the one base colour image, every other byte stays.
"""

from __future__ import annotations

import io
import itertools
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from image_to_3dlab import photo_paint as pp

TRUSTED = 0.9          # a texel the picture trusts this much is seen (photo_paint's own weight, 0..1)
RADIUS_BAND = 0.25     # the ring's wall lies within this share of its radius from the fitted circle
OUTWARD = 0.3          # ... and faces away from its middle at least this much (not the inner wall)
SIDE = 0.5             # ... and is a side, not a top or a bottom (|normal . up| below this)
ARC_BINS = 60          # the seen arc is cut at the quietest of these bins ...
ARC_ENDS = 0.25        # ... within this share of it at each end
MIN_SEEN = 50          # fewer seen texels than this: nothing to repeat
GUTTER_PX = 4          # repainted texels grow this far into the UV gutter (as photo_paint's)
# A face whose UV box spans more than this share of the picture has no island of its own: a cut's closing cap given
# corner UVs (the Zauberer's colour model after the Hero head had 601, each over up to 25 million texels). Skipped.
MAX_FACE_SHARE = 1.0 / 256
OTHER_KEEP = 0.6       # a band texel another picture trusts this much keeps that picture's paint (less: a blur)
BASE_BINS = 180        # slices of angle the band's plain colour is measured in (2 degrees) ...
BASE_HALF = np.radians(20.0)   # ... each from the points this far on either side ...
BASE_SHARE = 0.3       # ... in slices where at least this share of the band's points keep their paint
TILT_ROUNDS = 4        # rounds of fitting the band's tilt to the band found round it
BAND_MARGIN = 0.05     # the band's height, as the picture shows it in front, widened by this share at each edge
TILT_REACH = 1.0       # faces this many box heights above and below the box are looked at for the band's tilted rest
MAX_FACE_FLOOR = 256 * 256   # ... but a box this small is never too big (a small picture's faces span much of it)

# Finding a band in the picture (`find_bands`). Measured on the 27 front views of ArcherSoul's figures (2026-11-24): only
# the Zauberer's rune band passes all of these; a face's eyes and brows, a belt's buckle and loops, armour plates and a
# coat's trim each fail at least one.
BAND_WORK_HEIGHT = 1000      # the subject is looked at this many pixels tall
BAND_MEDIAN = 15             # a pixel's plain colour: the median of this many pixels square round it ...
BAND_MARK_DE = 18.0          # ... and a mark stands out of it by this much (CIE76 delta E)
BAND_MARK_HEIGHT = 0.035     # a mark is at most this share of the subject's height tall
BAND_MARKS = 5               # a band shows at least this many marks in a row ...
BAND_GAP_CV = 0.35           # ... evenly spaced round the ring (their gaps' spread against their mean) ...
BAND_SPAN = 0.7              # ... from end to end of the band (this share of its visible arc) ...
BAND_MASS_RATIO = 2.1        # ... none much bigger than the rest (a buckle, a clasp), the biggest against the median
BAND_MASS_CV = 0.45          # ... and their sizes alike
BAND_MARK_SPREAD = 12.0      # the marks share one colour (runes, studs; not eyes and brows)
BAND_PLAIN_SPREAD = 12.0     # the band between them is one plain colour ...
BAND_CONTRAST = 12.0         # ... set off from what lies above or below it (a zone of its own, not skin round eyes)
BAND_CONTAINED = 0.5         # the marks stop at its edges: just above and below them at most this share of their density
                             # (hair strands go on: 1.15 on the Runeblade's fringe; the Zauberer's runes 0.09)
BAND_EDGE_DE = 10.0          # the band's box grows up and down over rows this close to its plain colour ...
BAND_GROW = 0.3              # ... and then this share of its height more each way: a brim in front of a hat band hides
                             # its lower edge, which the sides show (the Zauberer's box found at its visible rows left a
                             # blue smear under the runes at both sides; the box read by hand reached into the brim)
BAND_SIDE = 0.06             # and this share of its width out past each end: `paint_ring` takes only the faces whose
                             # middle the picture shows inside the box, and the band's faces at its very sides lie on
                             # its outline (the found box left the sides' smear; the hand's box was 13 px wider there)


@dataclass(frozen=True)
class RingFrame:
    cx: float              # the ring's middle, x and z (glTF: y is up)
    cz: float
    radius: float
    tilt: tuple            # y = tilt[0] + tilt[1] cos(theta) + tilt[2] sin(theta) along the ring's middle
    outward: np.ndarray    # which of the given points lie on the ring's outer wall
    flip: float            # -1 when the given normals pointed inward (an open mesh's winding)


def circle_fit(x: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """(cx, cz, radius) of the circle through points (x, z), least squares (Kasa); an arc of a quarter is enough."""
    a = np.stack([x, z, np.ones_like(x)], axis=1)
    c = np.linalg.lstsq(a, x * x + z * z, rcond=None)[0]
    cx, cz = c[0] / 2.0, c[1] / 2.0
    return float(cx), float(cz), float(np.sqrt(max(c[2] + cx * cx + cz * cz, 0.0)))


def angles(points: np.ndarray, cx: float, cz: float, zero: float = 0.0) -> np.ndarray:
    """Angle round the ring's middle, radians in [-pi, pi), 0 at angle ZERO (0: toward +z, glTF's front)."""
    return (np.arctan2(points[:, 0] - cx, points[:, 2] - cz) - zero + np.pi) % (2 * np.pi) - np.pi


def ring_frame(points: np.ndarray, normals: np.ndarray, rounds: int = 3, fit: np.ndarray | None = None) -> RingFrame:
    """The ring that POINTS (with their NORMALS) mostly lie on: its middle and radius from its side walls, refitted on
    its outer wall alone, and its tilt as the best sine of height round it. Only the FIT points (all by default) find
    the ring; `outward` is told for every point."""
    fit = np.ones(len(points), dtype=bool) if fit is None else fit
    side = np.abs(normals[:, 1]) < SIDE
    if (side & fit).sum() < 3:
        raise ValueError("no side walls to find a ring by")
    cx, cz, radius = circle_fit(points[side & fit, 0], points[side & fit, 2])
    radial = np.stack([points[:, 0] - cx, np.zeros(len(points)), points[:, 2] - cz], axis=1)
    length = np.maximum(np.linalg.norm(radial, axis=1), 1e-12)
    walls = side & fit
    flip = -1.0 if np.median((normals[walls] * radial[walls]).sum(axis=1) / length[walls]) < 0 else 1.0
    outward = side
    for _ in range(rounds):
        radial = np.stack([points[:, 0] - cx, np.zeros(len(points)), points[:, 2] - cz], axis=1)
        length = np.maximum(np.linalg.norm(radial, axis=1), 1e-12)
        outward = (side & (flip * (normals * radial).sum(axis=1) / length > OUTWARD)
                   & (np.abs(length - radius) < RADIUS_BAND * radius))
        if (outward & fit).sum() < 3:
            raise ValueError("no outer wall round the fitted ring")
        cx, cz, radius = circle_fit(points[outward & fit, 0], points[outward & fit, 2])
    tilt = fit_tilt(points, cx, cz, outward & fit)
    return RingFrame(cx, cz, radius, tilt, outward, flip)


def fit_tilt(points: np.ndarray, cx: float, cz: float, mask: np.ndarray) -> tuple:
    """(y0, a, b): the ring's middle height y0 + a cos(theta) + b sin(theta) round it, least squares on MASK."""
    theta = angles(points[mask], cx, cz)
    design = np.stack([np.ones(len(theta)), np.cos(theta), np.sin(theta)], axis=1)
    return tuple(float(t) for t in np.linalg.lstsq(design, points[mask, 1], rcond=None)[0])


def ring_coords(points: np.ndarray, frame: RingFrame, zero: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(angle round the ring from ZERO, height above the ring's tilted middle) for each point."""
    theta = angles(points, frame.cx, frame.cz)
    middle = frame.tilt[0] + frame.tilt[1] * np.cos(theta) + frame.tilt[2] * np.sin(theta)
    return (theta - zero + np.pi) % (2 * np.pi) - np.pi, points[:, 1] - middle


def cut_arc(theta: np.ndarray, luminance: np.ndarray, seen: np.ndarray, bins: int = ARC_BINS,
            ends: float = ARC_ENDS) -> tuple[float, float]:
    """(start, end) of the seen arc to repeat: the seen angles' range, each end moved to the quietest bin (least
    spread of brightness) within ENDS of it, so the repeat joins between two runes rather than through one."""
    lo, hi = np.percentile(theta[seen], [1, 99])
    edges = np.linspace(lo, hi, bins + 1)
    which = np.clip(np.digitize(theta[seen], edges) - 1, 0, bins - 1)
    lum = luminance[seen]
    spread = np.full(bins, np.inf)
    for k in range(bins):
        here = lum[which == k]
        if len(here) >= 3:
            spread[k] = here.std()
    reach = max(1, int(round(ends * bins)))
    first = int(np.argmin(spread[:reach]))
    last = bins - reach + int(np.argmin(spread[-reach:]))
    middle = (edges[:-1] + edges[1:]) / 2
    return float(middle[first]), float(middle[last])


def repeat_arc(theta: np.ndarray, height: np.ndarray, colours: np.ndarray, source: np.ndarray, target: np.ndarray,
               arc: tuple[float, float], radius: float, base: np.ndarray | None = None) -> np.ndarray:
    """Colours for the TARGET points: the SOURCE point nearest to where each lands when the ring is cut into lengths
    of the arc and laid back onto it (angle as arc length, so heights and lengths weigh alike). With BASE (the band's
    plain colour per point, `band_base`) only the marks travel: each keeps its difference from the plain colour where
    it came from and takes the plain colour where it lands."""
    start, end = arc
    width = end - start
    if width <= 0:
        raise ValueError("the seen arc is empty")
    tree = cKDTree(np.stack([theta[source] * radius, height[source]], axis=1))
    folded = start + (theta[target] - start) % width
    _, nearest = tree.query(np.stack([folded * radius, height[target]], axis=1), workers=-1)
    if base is None:
        return colours[source][nearest]
    return colours[source][nearest] - base[source][nearest] + base[target]


def band_base(theta: np.ndarray, colours: np.ndarray, known: np.ndarray, bins: int = BASE_BINS,
              half: float = BASE_HALF, least: int = 5, among: np.ndarray | None = None,
              share: float = BASE_SHARE) -> np.ndarray:
    """The band's plain colour at each point, as a slow trend round the ring: at each slice of angle a straight line
    fitted to the colours of the KNOWN points (those keeping their paint) within HALF radians on either side, read at
    the slice; across slices no point knows, a straight blend between the known ones on either side, round the ring.
    A picture's painted light leaves one end of the seen arc darker than the other; repeated as it is, every join of
    the repeat showed that step as a seam. Marks (runes, studs) are narrower than the window and stay out of it. A
    slice counts only where at least SHARE of its points AMONG the band are known: a few texels at the band's edge
    that another picture half trusts darkened a whole side as an anchor."""
    which = np.clip(((theta + np.pi) / (2 * np.pi) * bins).astype(int), 0, bins - 1)
    middle = (np.arange(bins) + 0.5) / bins * 2 * np.pi - np.pi
    # each slice's median first (a mark covers less than half a slice's texels, so it drops out), then the trend
    medians, have = np.zeros((bins, colours.shape[1])), np.zeros(bins, dtype=bool)
    for k in range(bins):
        here = known & (which == k)
        total = (among & (which == k)).sum() if among is not None else here.sum()
        if here.sum() >= least and here.sum() >= share * total:
            medians[k], have[k] = np.median(colours[here], axis=0), True
    anchors, trend = [], []
    for k in np.flatnonzero(have):
        d = (middle[have] - middle[k] + np.pi) % (2 * np.pi) - np.pi
        near = np.abs(d) <= half
        design = np.stack([np.ones(near.sum()), d[near]], axis=1)
        anchors.append(middle[k])
        trend.append(np.linalg.lstsq(design, medians[have][near], rcond=None)[0][0])
    if not anchors:
        return np.repeat(np.median(colours, axis=0)[None], len(colours), axis=0)
    anchors, trend = np.asarray(anchors), np.asarray(trend)
    per_bin = np.stack([np.interp(middle, anchors, trend[:, c], period=2 * np.pi) for c in range(colours.shape[1])],
                       axis=1)
    return per_bin[which]


# --- texels ---------------------------------------------------------------------------------------------------------


def texels(uvs: np.ndarray, faces: np.ndarray, shape: tuple[int, int], chunk: int = 20000
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(flat texel index, face, barycentric weights) of every texel centre inside FACES' UV triangles (glTF UVs, (0, 0)
    the image's top-left), one entry per texel. Only the faces' own texels are held, never the whole image."""
    height, width = shape
    out_t, out_f, out_b = [], [], []
    for start in range(0, len(faces), chunk):
        xy = np.stack([uvs[:, 0] * width, uvs[:, 1] * height], axis=1)[faces[start:start + chunk]]
        lo = np.clip(np.floor(xy.min(axis=1) - 0.5), 0, [width - 1, height - 1]).astype(np.int64)
        hi = np.clip(np.ceil(xy.max(axis=1) - 0.5), 0, [width - 1, height - 1]).astype(np.int64)
        extent = hi - lo + 1
        counts = extent[:, 0] * extent[:, 1]
        counts[counts > max(MAX_FACE_SHARE * height * width, MAX_FACE_FLOOR)] = 0
        tri = np.repeat(np.arange(len(xy)), counts)
        offset = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        px, py = lo[tri, 0] + offset % extent[tri, 0], lo[tri, 1] + offset // extent[tri, 0]
        cx, cy = px + 0.5, py + 0.5
        a, b, c = xy[tri, 0], xy[tri, 1], xy[tri, 2]
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        keep = np.abs(den) > 1e-12
        den = np.where(keep, den, 1.0)
        w0 = ((b[:, 1] - c[:, 1]) * (cx - c[:, 0]) + (c[:, 0] - b[:, 0]) * (cy - c[:, 1])) / den
        w1 = ((c[:, 1] - a[:, 1]) * (cx - c[:, 0]) + (a[:, 0] - c[:, 0]) * (cy - c[:, 1])) / den
        w2 = 1.0 - w0 - w1
        inside = keep & (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        out_t.append(py[inside] * width + px[inside])
        out_f.append(tri[inside] + start)
        out_b.append(np.stack([w0[inside], w1[inside], w2[inside]], axis=1))
    index, first = np.unique(np.concatenate(out_t), return_index=True)
    return index, np.concatenate(out_f)[first], np.concatenate(out_b)[first]


def covered(uvs: np.ndarray, faces: np.ndarray, shape: tuple[int, int], chunk: int = 20000) -> np.ndarray:
    """Flat mask of the texels any face covers (the rest is UV gutter)."""
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    for start in range(0, len(faces), chunk):
        mask[texels(uvs, faces[start:start + chunk], shape)[0]] = True
    return mask


def grow_into_gutter(flat: np.ndarray, changed: np.ndarray, cover: np.ndarray, shape: tuple[int, int],
                     steps: int = GUTTER_PX) -> int:
    """Copy the CHANGED texels' colours (flat indices into FLAT, H*W x C) outward into uncovered texels, a ring at a
    time, so filtering at island borders does not pull the old paint back in. Returns how many texels it filled."""
    height, width = shape
    filled = np.zeros(height * width, dtype=bool)
    filled[changed] = True
    front = np.asarray(changed)
    total = 0
    for _ in range(steps):
        rows, cols = np.divmod(front, width)
        grown = []
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            r, c = rows + dr, cols + dc
            ok = (r >= 0) & (r < height) & (c >= 0) & (c < width)
            to, frm = r[ok] * width + c[ok], front[ok]
            free = ~cover[to] & ~filled[to]
            to, frm = to[free], frm[free]
            to, first = np.unique(to, return_index=True)
            flat[to] = flat[frm[first]]
            filled[to] = True
            grown.append(to)
        front = np.concatenate(grown)
        total += len(front)
        if not len(front):
            break
    return total


# --- one ring -------------------------------------------------------------------------------------------------------


def from_view_space(points: np.ndarray, frame=pp.PIXAL3D_FRAME, mesh_scale: float = 1.0) -> np.ndarray:
    """The inverse of `photo_paint.to_view_space`."""
    perm, signs = frame
    p = np.asarray(points, dtype=np.float64) * mesh_scale
    out = np.zeros_like(p)
    for i in range(3):
        out[:, perm[i]] = signs[i] * p[:, i]
    return out


def paint_ring(texture: np.ndarray, positions: np.ndarray, uvs: np.ndarray, faces: np.ndarray, view: pp.View,
               box, mesh_scale: float = 1.0, frame=pp.PIXAL3D_FRAME, settings: pp.Settings | None = None,
               trusted: float = TRUSTED, gutter: int = GUTTER_PX, others: list[pp.View] = ()) -> tuple[np.ndarray, dict]:
    """TEXTURE with the ring inside BOX (x0, y0, x1, y1 in VIEW's pixels: the band as the picture shows it) repainted
    round from its seen arc. Returns the new texture and a report (the ring found, the arc, texels repainted).

    The box finds the ring and its height on the band; the ring is then repainted over that height all round, also
    where a tilted band leaves the box's rows behind (a hat tilted back carries its band higher at the back, and the
    rows alone left its edges in the old paint, ragged)."""
    settings = settings or pp.DEFAULT_SETTINGS
    shape = texture.shape[:2]
    view_pos = pp.to_view_space(positions, frame, mesh_scale)
    x, y, _ = pp.project(view_pos, view)
    fx, fy = x[faces].mean(axis=1), y[faces].mean(axis=1)
    across = (fx >= box[0]) & (fx <= box[2])
    in_box = across & (fy >= box[1]) & (fy <= box[3])
    reach = TILT_REACH * (box[3] - box[1])
    region = np.flatnonzero(across & (fy >= box[1] - reach) & (fy <= box[3] + reach))
    report = {"box": [float(v) for v in box], "faces_in_box": int(in_box.sum()), "repainted": 0}
    if not in_box.any():
        return texture.copy(), report
    index, face, bary = texels(uvs, faces[region], shape)
    seed = in_box[region][face]
    corners = faces[region][face]
    points = (bary[:, :, None] * positions[corners]).sum(axis=1)
    outward_sign = pp.outward_sign(view_pos, faces)
    normals = (bary[:, :, None] * pp.vertex_normals(positions, faces)[corners]).sum(axis=1) * (outward_sign or 1.0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    trust, _ = pp.view_weights(pp.to_view_space(points, frame, mesh_scale), pp.to_view_space(normals, frame),
                               view, view_pos, faces, settings, outward_sign)
    try:
        ring = ring_frame(points, normals, fit=seed)
    except ValueError as error:
        report["skipped"] = str(error)
        return texture.copy(), report
    camera = from_view_space(view.position[None], frame, mesh_scale)
    zero = float(angles(camera, ring.cx, ring.cz)[0])
    theta, height = ring_coords(points, ring, zero)
    flat = texture.reshape(-1, texture.shape[2]).astype(np.float64)
    colours = flat[index, :3]
    seen = seed & ring.outward & (trust >= trusted)
    report.update(radius=ring.radius, middle=[ring.cx, ring.cz], tilt=list(ring.tilt), seen=int(seen.sum()))
    if seen.sum() < MIN_SEEN:
        report["skipped"] = "the picture trusts too little of the ring to repeat"
        return texture.copy(), report
    arc = cut_arc(theta, colours @ np.array([0.3, 0.59, 0.11]), seen)
    in_arc = (theta >= arc[0]) & (theta <= arc[1])
    # the band's height: as the box marks it in front (its edges count, though the picture's fades trust them less)
    low, high = np.percentile(height[seed & ring.outward & in_arc], [1, 99])
    low, high = low - BAND_MARGIN * (high - low), high + BAND_MARGIN * (high - low)
    band = ring.outward & (height >= low) & (height <= high)
    for _ in range(TILT_ROUNDS):     # the box saw the band's front: its tilt comes from the whole band, found round
        ring = replace(ring, tilt=fit_tilt(points, ring.cx, ring.cz, band))
        theta, height = ring_coords(points, ring, zero)
        band = ring.outward & (height >= low) & (height <= high)
    report["tilt"] = list(ring.tilt)
    report["band"] = [float(low), float(high)]       # its height round the tilted middle, for `in_ring`
    in_arc = (theta >= arc[0]) & (theta <= arc[1])
    source = seen & in_arc
    target = band & ~(in_arc & (trust >= trusted))
    view_points, view_normals = pp.to_view_space(points, frame, mesh_scale), pp.to_view_space(normals, frame)
    for other in others:              # what another picture paints (a drawn back) keeps its paint
        weight, _ = pp.view_weights(view_points, view_normals, other, view_pos, faces, settings, outward_sign)
        target &= weight < OTHER_KEEP
    base = band_base(theta, colours, band & ~target, among=band)
    flat = flat.copy()
    flat[index[target], :3] = repeat_arc(theta, height, colours, source, target, arc, ring.radius, base)
    grown = grow_into_gutter(flat, index[target], covered(uvs, faces, shape), shape, gutter)
    report.update(arc_degrees=float(np.degrees(arc[1] - arc[0])), arc=[float(np.degrees(a)) for a in arc],
                  repainted=int(target.sum()), gutter=grown)
    return np.clip(np.rint(flat), 0, 255).astype(texture.dtype).reshape(texture.shape), report


# --- finding the band in the picture ----------------------------------------------------------------------------------


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0..255 to CIE L*a*b* (D65), as `colour_compare.lab`."""
    from image_to_3dlab.colour_compare import lab

    return lab(np.asarray(rgb, dtype=np.float64) / 255.0)


def _median_colour(lab: np.ndarray, where: np.ndarray) -> np.ndarray | None:
    return np.median(lab[where], axis=0) if where.sum() >= 6 else None


def _band_strip(marks: np.ndarray, mask: np.ndarray, lab: np.ndarray, rows: tuple[int, int], run: tuple[int, int],
                height: float) -> dict | None:
    """The measures of one candidate strip: ROWS (y0, y1) of the marks' row across the RUN (x0, x1) of the subject.
    The run is unrolled round its middle (a ring seen from the front: x = middle + radius sin(angle)), so marks evenly
    spaced round a band are evenly spaced here too."""
    y0, y1 = rows
    xa, xb = run
    width = xb - xa + 1
    centre, radius = (xa + xb) / 2.0, width / 2.0
    reach = np.arcsin(0.95)
    count = max(16, int(2 * reach * radius))
    x = np.clip(np.round(centre + radius * np.sin(np.linspace(-reach, reach, count))).astype(int), 0, mask.shape[1] - 1)
    column = marks[y0:y1][:, x].mean(axis=0)
    smooth = ndimage.gaussian_filter1d(column, sigma=max(0.8, 0.15 * height))
    if smooth.max() < 0.15:
        return None
    peaks, _ = find_peaks(smooth, height=0.35 * smooth.max(), distance=max(2, int(0.35 * height)),
                          prominence=0.2 * smooth.max())
    if len(peaks) < BAND_MARKS:
        return None
    gaps = np.diff(peaks)
    ends = np.concatenate([[max(0, peaks[0] - gaps[0] // 2)], (peaks[:-1] + peaks[1:]) // 2,
                           [min(count, peaks[-1] + gaps[-1] // 2 + 1)]])
    mass = np.array([column[a:b].sum() for a, b in itertools.pairwise(ends)])
    colours = []
    for p in peaks:
        near = np.unique(x[max(0, p - 2):min(count, p + 3)])
        mine = marks[y0:y1][:, near]
        if mine.any():
            colours.append(np.median(lab[y0:y1][:, near][mine], axis=0))
    colours = np.asarray(colours)
    inner = np.zeros(mask.shape[1], dtype=bool)
    inner[int(xa + 0.15 * width):int(xb - 0.15 * width) + 1] = True
    plain_here = mask[y0:y1] & ~marks[y0:y1] & inner[None, :]
    plain = _median_colour(lab[y0:y1], plain_here)
    if plain is None or not len(colours):
        return None
    step = max(2, int(0.6 * (y1 - y0)))
    contrast = 0.0
    for a, b in ((y0 - step - 1, y0 - 1), (y1 + 1, y1 + step + 1)):
        a, b = max(0, a), min(mask.shape[0], b)
        if b > a:
            other = _median_colour(lab[a:b], mask[a:b] & inner[None, :])
            if other is not None:
                contrast = max(contrast, float(np.linalg.norm(other - plain)))
    tall = y1 - y0
    inside = marks[y0:y1][:, inner].mean()
    beyond = [marks[a:b][:, inner].mean() for a, b in ((max(0, y0 - tall - 2), y0 - 2), (y1 + 2, y1 + tall + 2))
              if b - a >= 2 and b <= mask.shape[0]]
    return {"marks": int(len(peaks)), "gap_cv": float(gaps.std() / gaps.mean()),
            "contained": float(max(beyond, default=0.0) / max(inside, 1e-9)),
            "span": float((peaks[-1] - peaks[0]) / count),
            "mass_ratio": float(mass.max() / max(np.median(mass), 1e-9)), "mass_cv": float(mass.std() / mass.mean()),
            "mark_spread": float(np.median(np.linalg.norm(colours - np.median(colours, axis=0), axis=1))),
            "plain_spread": float(np.median(np.linalg.norm(lab[y0:y1][plain_here] - plain, axis=1))),
            "contrast": contrast, "plain": plain}


def _is_band(strip: dict) -> bool:
    return (strip["marks"] >= BAND_MARKS and strip["gap_cv"] <= BAND_GAP_CV and strip["span"] >= BAND_SPAN
            and strip["mass_ratio"] <= BAND_MASS_RATIO and strip["mass_cv"] <= BAND_MASS_CV
            and strip["mark_spread"] <= BAND_MARK_SPREAD and strip["plain_spread"] <= BAND_PLAIN_SPREAD
            and strip["contrast"] >= BAND_CONTRAST and strip["contained"] <= BAND_CONTAINED)


def _grow_rows(lab: np.ndarray, mask: np.ndarray, marks: np.ndarray, rows: tuple[int, int], run: tuple[int, int],
               plain: np.ndarray) -> tuple[int, int]:
    """The strip's rows grown up and down over rows whose plain colour stays the band's (its top and bottom edges)."""
    xa, xb = run
    inner = np.zeros(mask.shape[1], dtype=bool)
    inner[int(xa + 0.15 * (xb - xa)):int(xb - 0.15 * (xb - xa)) + 1] = True
    y0, y1 = rows
    most = max(2, int(1.5 * (y1 - y0)))

    def close(y):
        here = mask[y] & ~marks[y] & inner
        colour = _median_colour(lab[y], here)
        return colour is not None and float(np.linalg.norm(colour - plain)) <= BAND_EDGE_DE

    top = y0
    while top > max(0, y0 - most) and close(top - 1):
        top -= 1
    bottom = y1
    while bottom < min(mask.shape[0], y1 + most) and close(bottom):
        bottom += 1
    return top, bottom


def in_ring(points: np.ndarray, report: dict, grow: float = 0.25, wall: float = 0.35) -> np.ndarray:
    """Which POINTS lie on the band a `paint_ring` REPORT repainted: within its height round the tilted middle (grown
    by GROW of that height at each edge) and within WALL of its radius from its axis. False for all without a band."""
    p = np.asarray(points, dtype=np.float64)
    if "band" not in report or not report.get("repainted"):
        return np.zeros(len(p), dtype=bool)
    cx, cz = report["middle"]
    low, high = report["band"]
    margin = grow * (high - low)
    theta = angles(p, cx, cz)
    middle = report["tilt"][0] + report["tilt"][1] * np.cos(theta) + report["tilt"][2] * np.sin(theta)
    height = p[:, 1] - middle
    radial = np.hypot(p[:, 0] - cx, p[:, 2] - cz)
    return ((height >= low - margin) & (height <= high + margin)
            & (np.abs(radial - report["radius"]) <= wall * report["radius"]))


def find_bands(image: np.ndarray, alpha: np.ndarray | None = None) -> list[dict]:
    """The bands in a front picture (RGBA, the subject cut out) whose pattern repeats along them: a hat band's runes, a
    belt's studs, a sleeve's hem. Each: {"box": (x0, y0, x1, y1) in the picture's pixels, the band from its left end to
    its right and its top edge to its bottom edge, ready for `paint_glb`; "marks", "gap_cv", ...: what was measured}.
    An empty list when nothing in the picture is such a band.

    A band is a level row of marks (each standing out of its neighbourhood's plain colour) that are evenly spaced round
    the ring, alike in size and colour, over most of the band's visible width, on one plain colour that differs from
    what lies above or below it. A single buckle, eyes and brows on a face, plates of armour or a coat's trim are not:
    their marks differ in size or colour, or sit on the colour round them."""
    picture = np.asarray(image)
    alpha = picture[..., 3] if alpha is None else np.asarray(alpha)
    ys, xs = np.nonzero(alpha > 127)
    if not len(ys):
        return []
    top, bottom, left, right = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    scale = BAND_WORK_HEIGHT / (bottom - top)
    crop = Image.fromarray(np.dstack([picture[..., :3], alpha]).astype(np.uint8)).crop((left, top, right, bottom))
    small = np.asarray(crop.resize((max(1, round(crop.width * scale)), BAND_WORK_HEIGHT), Image.LANCZOS))
    mask = small[..., 3] > 127
    lab = _lab(small[..., :3])
    plain = np.stack([ndimage.median_filter(lab[..., c], size=BAND_MEDIAN) for c in range(3)], axis=-1)
    marks = ndimage.binary_erosion(mask, iterations=3) & (np.linalg.norm(lab - plain, axis=-1) > BAND_MARK_DE)
    labels, _ = ndimage.label(marks)
    blobs = []
    for sl in ndimage.find_objects(labels):
        tall, wide = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 3 <= tall <= BAND_MARK_HEIGHT * BAND_WORK_HEIGHT and wide <= 3 * tall:
            blobs.append(((sl[0].start + sl[0].stop) / 2.0, tall))
    blobs.sort()
    middles = np.array([b[0] for b in blobs])
    found, tried = [], set()
    for middle, tall in blobs:
        row = np.abs(middles - middle) <= max(0.5 * tall, 2)
        if row.sum() < BAND_MARKS:
            continue
        height = float(np.median([blobs[i][1] for i in np.flatnonzero(row)]))
        rows = (int(middle - 0.6 * height), int(np.ceil(middle + 0.6 * height)) + 1)
        runs, count = ndimage.label(mask[int(round(middle))])
        for k in range(1, count + 1):
            cols = np.flatnonzero(runs == k)
            run = (int(cols[0]), int(cols[-1]))
            if run[1] - run[0] + 1 < 6 * height or (rows, run) in tried:
                continue
            tried.add((rows, run))
            strip = _band_strip(marks, mask, lab, rows, run, height)
            if strip is None or not _is_band(strip):
                continue
            grown = _grow_rows(lab, mask, marks, rows, run, strip["plain"])

            def overlap(f, run=run, grown=grown):
                across = min(f["run"][1], run[1]) - max(f["run"][0], run[0])
                return (grown[0] <= f["rows"][1] and f["rows"][0] <= grown[1]
                        and across >= 0.7 * min(f["run"][1] - f["run"][0], run[1] - run[0]))

            same = next((f for f in found if overlap(f)), None)
            if same is not None:            # the same band found from another of its marks (a row lower, say)
                same["rows"] = (min(same["rows"][0], grown[0]), max(same["rows"][1], grown[1]))
                same["run"] = (min(same["run"][0], run[0]), max(same["run"][1], run[1]))
                continue
            found.append({"rows": grown, "run": run, **{k: v for k, v in strip.items() if k != "plain"}})
    bands = []
    for f in found:
        (y0, y1), (xa, xb) = f.pop("rows"), f.pop("run")
        grow, side = BAND_GROW * (y1 - y0), BAND_SIDE * (xb + 1 - xa)
        y0, y1 = max(0.0, y0 - grow), min(float(mask.shape[0]), y1 + grow)
        box = (max(0.0, left + (xa - side) / scale), top + y0 / scale,
               min(float(picture.shape[1]), left + (xb + 1 + side) / scale), top + y1 / scale)
        bands.append({"box": tuple(round(float(v), 1) for v in box),
                      **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()}})
    return bands


# --- GLB ------------------------------------------------------------------------------------------------------------


def _image_of(doc: dict, material: int) -> int | None:
    texture = doc["materials"][material].get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if texture is None:
        return None
    entry = doc["textures"][texture["index"]]
    if "source" in entry:
        return entry["source"]
    return entry.get("extensions", {}).get("EXT_texture_webp", {}).get("source")


def _primitives(doc: dict) -> list[dict]:
    nodes = [n for n in doc.get("nodes", []) if "mesh" in n]
    if any(key in n for n in nodes for key in ("matrix", "translation", "rotation", "scale")):
        raise ValueError("node transforms are not supported; expected a model in its own frame")
    if len({n["mesh"] for n in nodes}) != 1:
        raise ValueError("expected one mesh in the model")
    return doc["meshes"][nodes[0]["mesh"]]["primitives"]


def read_primitive(data: bytes, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """(positions, glTF UVs, triangles, base colour image, its image index) of the mesh's primitive INDEX."""
    doc, binary = pp._split_glb(data)
    primitive = _primitives(doc)[index]
    image = _image_of(doc, primitive["material"]) if "material" in primitive else None
    if image is None or "TEXCOORD_0" not in primitive["attributes"]:
        raise ValueError(f"primitive {index} has no base colour picture to paint")
    attributes = primitive["attributes"]
    positions = pp._accessor(doc, binary, attributes["POSITION"]).astype(np.float64)
    uvs = pp._accessor(doc, binary, attributes["TEXCOORD_0"]).astype(np.float64)
    faces = pp._accessor(doc, binary, primitive["indices"]).reshape(-1, 3).astype(np.int64)
    view = doc["bufferViews"][doc["images"][image]["bufferView"]]
    start = view.get("byteOffset", 0)
    picture = Image.open(io.BytesIO(binary[start:start + view["byteLength"]]))
    texture = np.asarray(picture.convert("RGBA" if "A" in picture.getbands() else "RGB"))
    return positions, uvs, faces, texture, image


def replace_image(data: bytes, image: int, png: bytes) -> bytes:
    """The GLB with image IMAGE swapped for PNG, all else unchanged (`photo_paint.replace_base_colour` for any one
    image of a model with several materials)."""
    doc, binary = pp._split_glb(data)
    target = doc["images"][image]["bufferView"]
    packed = b""
    for number, view in enumerate(doc["bufferViews"]):
        start = view.get("byteOffset", 0)
        chunk = png if number == target else binary[start:start + view["byteLength"]]
        packed += b"\0" * (-len(packed) % 4)
        view["byteOffset"] = len(packed)
        view["byteLength"] = len(chunk)
        packed += chunk
    doc["buffers"][0]["byteLength"] = len(packed)
    doc["images"][image]["mimeType"] = "image/png"
    pp._drop_webp_extension(doc, image)
    return pp._join_glb(doc, packed)


def pick_primitive(data: bytes, view: pp.View, box, mesh_scale: float = 1.0, frame=pp.PIXAL3D_FRAME) -> int | None:
    """The painted primitive with the most triangles inside BOX (the ring's own surface, not a part set in near it)."""
    doc, binary = pp._split_glb(data)
    best, most = None, 0
    for number, primitive in enumerate(_primitives(doc)):
        if "material" not in primitive or _image_of(doc, primitive["material"]) is None:
            continue
        positions = pp._accessor(doc, binary, primitive["attributes"]["POSITION"]).astype(np.float64)
        faces = pp._accessor(doc, binary, primitive["indices"]).reshape(-1, 3).astype(np.int64)
        x, y, _ = pp.project(pp.to_view_space(positions, frame, mesh_scale), view)
        fx, fy = x[faces].mean(axis=1), y[faces].mean(axis=1)
        count = int(((fx >= box[0]) & (fx <= box[2]) & (fy >= box[1]) & (fy <= box[3])).sum())
        if count > most:
            best, most = number, count
    return best


def paint_glb(data: bytes, view: pp.View, boxes, mesh_scale: float = 1.0, frame=pp.PIXAL3D_FRAME,
              settings: pp.Settings | None = None, trusted: float = TRUSTED,
              others: list[pp.View] = ()) -> tuple[bytes, list[dict]]:
    """The GLB with each box's ring repainted round (`paint_ring`) on the primitive it belongs to; what one of the
    OTHERS (more pictures of the model, a drawn back) sees keeps its paint."""
    reports = []
    for box in boxes:
        number = pick_primitive(data, view, box, mesh_scale, frame)
        if number is None:
            reports.append({"box": [float(v) for v in box], "skipped": "no painted surface inside the box",
                            "repainted": 0})
            continue
        positions, uvs, faces, texture, image = read_primitive(data, number)
        painted, report = paint_ring(texture, positions, uvs, faces, view, box, mesh_scale, frame, settings, trusted,
                                     others=others)
        report["primitive"] = number
        reports.append(report)
        if report["repainted"]:
            data = replace_image(data, image, pp.encode_png(painted))
    return data, reports
