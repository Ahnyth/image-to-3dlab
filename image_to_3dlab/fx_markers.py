"""Mark where an effect (a flame, a glow, sparks) was taken out of a picture, as empty nodes in the model.

A flame built as geometry looks like a sheet of paper from the side; a game's particle effect moves and glows.
So the model is built from a picture with the effect taken out (the studio's own picture editor does that), and
this module compares that picture with the original to find where the effect sat. Each effect becomes an empty
node in the GLB (`fx_<name>_<n>`, with its radius in `extras`), where an engine attaches its own effect.

The two pictures may be framed differently (an editor draws at its own size and place); they are lined up first
by the outline they share. The front picture is the model's front view: glTF's +Z faces the camera, +X is to
the picture's right, +Y up, and the subject's outline in the effect-free picture spans the model's width and
height (the generator was fed that picture).
"""

from __future__ import annotations

import io

import numpy as np
from scipy import ndimage

from image_to_3dlab.glb_turn import build_glb, parse_glb

ALPHA_CUT = 128
COLOUR_JUMP = 150      # a pixel whose R+G+B changed by more than this between the pictures was part of the effect
MIN_AREA = 0.002       # an effect covers at least this share of the subject (smaller differences are drawing noise)
WORK = 160             # the side of the small masks the alignment searches on
MIN_VIVID = 0.3        # an effect (a flame, a glow) is bright and coloured: its mean value x saturation at least
                       # this; a redrawn shadow, crack or foot is dark or grey (the Flammenwolf's redrawn feet fell below)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _warp(mask: np.ndarray, s: float, tx: float, ty: float, shape: tuple[int, int]) -> np.ndarray:
    """`mask` (in the clean picture's pixels) moved into a picture of `shape` by p -> s * p + t (nearest)."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    sx = np.round((xs - tx) / s).astype(int)
    sy = np.round((ys - ty) / s).astype(int)
    ok = (sx >= 0) & (sy >= 0) & (sx < mask.shape[1]) & (sy < mask.shape[0])
    out = np.zeros(shape if mask.ndim == 2 else (h, w, mask.shape[2]), dtype=mask.dtype)
    out[ok] = mask[sy[ok], sx[ok]]
    return out


def align(orig: np.ndarray, clean: np.ndarray) -> tuple[float, float, float]:
    """Scale and shift (s, tx, ty) that carry the clean picture's pixels onto the original's: p -> s * p + t.

    Starts from the outlines' heights and bottoms (an effect seldom changes where the feet and the top are) and
    keeps the best overlap of the two outlines over a small search of scale and shift.
    """
    ox0, oy0, ox1, oy1 = _bbox(orig)
    cx0, cy0, cx1, cy1 = _bbox(clean)
    s0 = (oy1 - oy0 + 1) / (cy1 - cy0 + 1)
    # work on small masks: the original's subject region, padded
    k = WORK / max(ox1 - ox0 + 1, oy1 - oy0 + 1)
    pad = int(0.15 * max(ox1 - ox0, oy1 - oy0))
    X0, Y0 = max(0, ox0 - pad), max(0, oy0 - pad)
    small_o = orig[Y0:oy1 + pad + 1, X0:ox1 + pad + 1]
    so = ndimage.zoom(small_o.astype(float), k, order=1) > 0.5
    best = (-1.0, s0, 0.0, 0.0)
    for s in s0 * np.linspace(0.75, 1.1, 22):   # an effect above the subject makes the outline taller
        # line up the bottoms and the clean outline's centre with the original's, then search around that
        bx = (ox0 + ox1) / 2 - s * (cx0 + cx1) / 2
        by = oy1 - s * cy1
        for dx in np.linspace(-0.06, 0.06, 13) * (ox1 - ox0):
            for dy in np.linspace(-0.03, 0.03, 5) * (oy1 - oy0):
                tx, ty = bx + dx, by + dy
                # the clean mask in the small frame: p_small = k * (s * p + t - (X0, Y0))
                warped = _warp(clean, s * k, k * (tx - X0), k * (ty - Y0), so.shape)
                inter = np.logical_and(so, warped).sum()
                union = np.logical_or(so, warped).sum()
                iou = inter / union if union else 0.0
                if iou > best[0]:
                    best = (iou, s, tx, ty)
    return float(best[1]), float(best[2]), float(best[3])


def effect_regions(orig_rgba: np.ndarray, clean_rgba: np.ndarray) -> list[dict]:
    """Where the effect was, in the clean picture's pixels: per region its anchor (where it touches the subject),
    its centre and its radius. Both pictures are RGBA cut-outs (alpha above half = the subject)."""
    clean_a = clean_rgba[..., 3] > ALPHA_CUT
    orig_a = orig_rgba[..., 3] > ALPHA_CUT
    if not clean_a.any() or not orig_a.any():
        return []
    if orig_rgba.shape == clean_rgba.shape and np.array_equal(_bbox(orig_a)[3:], _bbox(clean_a)[3:]) and \
            (orig_a & clean_a).sum() >= 0.9 * clean_a.sum():
        s, tx, ty = 1.0, 0.0, 0.0       # already lined up
    else:
        s, tx, ty = align(orig_a, clean_a)
    # the original seen in the clean picture's frame: q (clean pixel) -> s * q + t (original pixel)
    h, w = clean_a.shape
    ys, xs = np.mgrid[0:h, 0:w]
    ox = np.round(s * xs + tx).astype(int)
    oy = np.round(s * ys + ty).astype(int)
    ok = (ox >= 0) & (oy >= 0) & (ox < orig_rgba.shape[1]) & (oy < orig_rgba.shape[0])
    o = np.zeros_like(clean_rgba)
    o[ok] = orig_rgba[oy[ok], ox[ok]]
    o_a = o[..., 3] > ALPHA_CUT
    outside = o_a & ~clean_a
    jump = np.abs(o[..., :3].astype(int) - clean_rgba[..., :3].astype(int)).sum(axis=2)
    changed = o_a & clean_a & (jump > COLOUR_JUMP)
    mask = ndimage.binary_opening(outside | changed, iterations=2)
    labels, n = ndimage.label(mask)
    min_area = max(12, MIN_AREA * clean_a.sum())
    dist = ndimage.distance_transform_edt(~clean_a)   # 0 on the subject, growing away from it
    found = []
    for i in range(1, n + 1):
        yy, xx = np.nonzero(labels == i)
        if len(yy) < min_area:
            continue
        rgb = o[yy, xx, :3].astype(float) / 255.0
        value, low = rgb.max(axis=1), rgb.min(axis=1)
        vivid = float(np.mean(value * np.where(value > 0, (value - low) / np.maximum(value, 1e-6), 0.0)))
        if vivid < MIN_VIVID:
            continue
        cx, cy = float(xx.mean()), float(yy.mean())
        d = dist[yy, xx]
        near = d <= d.min() + 1
        pick = np.argmin(np.where(near, (xx - cx) ** 2 + (yy - cy) ** 2, np.inf))
        found.append({"anchor": (int(xx[pick]), int(yy[pick])), "centre": (cx, cy),
                      "radius": float(np.sqrt(len(yy) / np.pi)), "pixels": int(len(yy))})
    return sorted(found, key=lambda r: -r["pixels"])


def pair_by_height(front: list[dict], side: list[dict], tolerance: float) -> list[int | None]:
    """For each front effect, the side effect nearest in height (model units, "y"), or None when none is within
    `tolerance`. Several front effects may share one side effect (a mane seen as three tongues from the front)."""
    out: list[int | None] = []
    for f in front:
        best = min(range(len(side)), key=lambda i: abs(side[i]["y"] - f["y"]), default=None)
        out.append(best if best is not None and abs(side[best]["y"] - f["y"]) <= tolerance else None)
    return out


def add_markers(glb: bytes, orig_rgba: np.ndarray, clean_rgba: np.ndarray, name: str = "flame",
                side: tuple[np.ndarray, np.ndarray] | None = None) -> bytes:
    """The GLB with one empty root node per effect region at its anchor: on the model's front surface, or, with
    `side` (the right side view with and without the effect, the subject facing right), at the depth where the
    side view shows the effect touching the subject."""
    import trimesh

    regions = effect_regions(orig_rgba, clean_rgba)
    doc, binary = parse_glb(glb)
    if not regions:
        return glb
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    verts = np.concatenate([g.vertices for g in scene.dump()])
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    x0, y0, x1, y1 = _bbox(clean_rgba[..., 3] > ALPHA_CUT)
    per_px = (hi[0] - lo[0]) / (x1 - x0 + 1)
    near_r = 0.03 * float(max(hi - lo))
    root = doc["scenes"][doc.get("scene", 0)]["nodes"]
    fronts = []
    for r in regions:
        ax, ay = r["anchor"]
        fronts.append({"x": lo[0] + (ax - x0 + 0.5) / (x1 - x0 + 1) * (hi[0] - lo[0]),
                       "y": hi[1] - (ay - y0 + 0.5) / (y1 - y0 + 1) * (hi[1] - lo[1])})
    sides: list[dict] = []
    if side is not None:
        s_orig, s_clean = side
        sx0, sy0, sx1, sy1 = _bbox(s_clean[..., 3] > ALPHA_CUT)
        for r in effect_regions(s_orig, s_clean):
            ax, ay = r["anchor"]
            sides.append({"z": lo[2] + (ax - sx0 + 0.5) / (sx1 - sx0 + 1) * (hi[2] - lo[2]),
                          "y": hi[1] - (ay - sy0 + 0.5) / (sy1 - sy0 + 1) * (hi[1] - lo[1])})
    pairs = pair_by_height(fronts, sides, tolerance=0.15 * float(hi[1] - lo[1])) if sides else [None] * len(fronts)
    for n, (r, f, k) in enumerate(zip(regions, fronts, pairs), 1):
        x, y = float(f["x"]), float(f["y"])
        if k is not None:
            z, depth = float(sides[k]["z"]), "side"
        else:
            d = np.hypot(verts[:, 0] - x, verts[:, 1] - y)
            close = d <= near_r
            z = float(verts[close, 2].max()) if close.any() else float(verts[np.argmin(d), 2])
            depth = "front"
        doc["nodes"].append({"name": f"fx_{name}_{n}", "translation": [x, y, z],
                             "extras": {"fx": name, "radius": round(float(r["radius"] * per_px), 4),
                                        "depth_from": depth, "made_by": "AssetFurnace fx_markers"}})
        root.append(len(doc["nodes"]) - 1)
    return build_glb(doc, binary)
