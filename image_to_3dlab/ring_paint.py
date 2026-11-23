"""Repeat a band's seen arc round the band, where no picture sees it.

A hat band, a belt or a bracelet is a ring. The front picture sees its pattern (a hat band's runes, a belt's studs)
only on the front arc; Pixel Match (`photo_paint`) trusts a picture only where the surface faces it, so the ring's
sides and back keep the generator's own paint, which redraws the pattern as a blur. The Zauberer's hat band showed
sharp runes in front and blue smears on its sides (2026-11-23): the front camera meets the band's sides at grazing
angles, where the picture's pixels are not used at all.

A ring's pattern goes on round it, so the seen arc is repeated round the rest: the ring found from its own surface
(its middle, its radius and its tilt), the arc the picture trusts cut where it holds the least detail (between two
runes, not through one), and every texel of the ring the picture does not trust takes the colour of the seen arc at
the same height, one arc-length further back. Only the ring's outer wall inside the given box changes.

Pure numpy + scipy; the GLB helpers change only the one base colour image, every other byte stays.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
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
MAX_FACE_FLOOR = 256 * 256   # ... but a box this small is never too big (a small picture's faces span much of it)


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


def ring_frame(points: np.ndarray, normals: np.ndarray, rounds: int = 3) -> RingFrame:
    """The ring that POINTS (with their NORMALS) mostly lie on: its middle and radius from its side walls, refitted on
    its outer wall alone, and its tilt as the best sine of height round it."""
    side = np.abs(normals[:, 1]) < SIDE
    if side.sum() < 3:
        raise ValueError("no side walls to find a ring by")
    cx, cz, radius = circle_fit(points[side, 0], points[side, 2])
    radial = np.stack([points[:, 0] - cx, np.zeros(len(points)), points[:, 2] - cz], axis=1)
    length = np.maximum(np.linalg.norm(radial, axis=1), 1e-12)
    flip = -1.0 if np.median((normals[side] * radial[side]).sum(axis=1) / length[side]) < 0 else 1.0
    outward = side
    for _ in range(rounds):
        radial = np.stack([points[:, 0] - cx, np.zeros(len(points)), points[:, 2] - cz], axis=1)
        length = np.maximum(np.linalg.norm(radial, axis=1), 1e-12)
        outward = (side & (flip * (normals * radial).sum(axis=1) / length > OUTWARD)
                   & (np.abs(length - radius) < RADIUS_BAND * radius))
        if outward.sum() < 3:
            raise ValueError("no outer wall round the fitted ring")
        cx, cz, radius = circle_fit(points[outward, 0], points[outward, 2])
    theta = angles(points, cx, cz)
    design = np.stack([np.ones(len(points)), np.cos(theta), np.sin(theta)], axis=1)
    tilt = np.linalg.lstsq(design[outward], points[outward, 1], rcond=None)[0]
    return RingFrame(cx, cz, radius, tuple(float(t) for t in tilt), outward, flip)


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
               arc: tuple[float, float], radius: float) -> np.ndarray:
    """Colours for the TARGET points: the SOURCE point nearest to where each lands when the ring is cut into lengths
    of the arc and laid back onto it (angle as arc length, so heights and lengths weigh alike)."""
    start, end = arc
    width = end - start
    if width <= 0:
        raise ValueError("the seen arc is empty")
    tree = cKDTree(np.stack([theta[source] * radius, height[source]], axis=1))
    folded = start + (theta[target] - start) % width
    _, nearest = tree.query(np.stack([folded * radius, height[target]], axis=1), workers=-1)
    return colours[source][nearest]


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
               trusted: float = TRUSTED, gutter: int = GUTTER_PX) -> tuple[np.ndarray, dict]:
    """TEXTURE with the ring inside BOX (x0, y0, x1, y1 in VIEW's pixels: the band as the picture shows it) repainted
    round from its seen arc. Returns the new texture and a report (the ring found, the arc, texels repainted)."""
    settings = settings or pp.DEFAULT_SETTINGS
    shape = texture.shape[:2]
    view_pos = pp.to_view_space(positions, frame, mesh_scale)
    x, y, _ = pp.project(view_pos, view)
    fx, fy = x[faces].mean(axis=1), y[faces].mean(axis=1)
    region = np.flatnonzero((fx >= box[0]) & (fx <= box[2]) & (fy >= box[1]) & (fy <= box[3]))
    report = {"box": [float(v) for v in box], "faces_in_box": int(len(region)), "repainted": 0}
    if not len(region):
        return texture.copy(), report
    index, face, bary = texels(uvs, faces[region], shape)
    corners = faces[region][face]
    points = (bary[:, :, None] * positions[corners]).sum(axis=1)
    outward_sign = pp.outward_sign(view_pos, faces)
    normals = (bary[:, :, None] * pp.vertex_normals(positions, faces)[corners]).sum(axis=1) * (outward_sign or 1.0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    trust, _ = pp.view_weights(pp.to_view_space(points, frame, mesh_scale), pp.to_view_space(normals, frame),
                               view, view_pos, faces, settings, outward_sign)
    try:
        ring = ring_frame(points, normals)
    except ValueError as error:
        report["skipped"] = str(error)
        return texture.copy(), report
    camera = from_view_space(view.position[None], frame, mesh_scale)
    zero = float(angles(camera, ring.cx, ring.cz)[0])
    theta, height = ring_coords(points, ring, zero)
    flat = texture.reshape(-1, texture.shape[2]).astype(np.float64)
    colours = flat[index, :3]
    seen = ring.outward & (trust >= trusted)
    report.update(radius=ring.radius, middle=[ring.cx, ring.cz], tilt=list(ring.tilt), seen=int(seen.sum()))
    if seen.sum() < MIN_SEEN:
        report["skipped"] = "the picture trusts too little of the ring to repeat"
        return texture.copy(), report
    arc = cut_arc(theta, colours @ np.array([0.3, 0.59, 0.11]), seen)
    in_arc = (theta >= arc[0]) & (theta <= arc[1])
    source = seen & in_arc
    target = ring.outward & ~(in_arc & (trust >= trusted))
    flat = flat.copy()
    flat[index[target], :3] = repeat_arc(theta, height, colours, source, target, arc, ring.radius)
    grown = grow_into_gutter(flat, index[target], covered(uvs, faces, shape), shape, gutter)
    report.update(arc_degrees=float(np.degrees(arc[1] - arc[0])), arc=[float(np.degrees(a)) for a in arc],
                  repainted=int(target.sum()), gutter=grown)
    return np.clip(np.rint(flat), 0, 255).astype(texture.dtype).reshape(texture.shape), report


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
              settings: pp.Settings | None = None, trusted: float = TRUSTED) -> tuple[bytes, list[dict]]:
    """The GLB with each box's ring repainted round (`paint_ring`) on the primitive it belongs to."""
    reports = []
    for box in boxes:
        number = pick_primitive(data, view, box, mesh_scale, frame)
        if number is None:
            reports.append({"box": [float(v) for v in box], "skipped": "no painted surface inside the box",
                            "repainted": 0})
            continue
        positions, uvs, faces, texture, image = read_primitive(data, number)
        painted, report = paint_ring(texture, positions, uvs, faces, view, box, mesh_scale, frame, settings, trusted)
        report["primitive"] = number
        reports.append(report)
        if report["repainted"]:
            data = replace_image(data, image, pp.encode_png(painted))
    return data, reports
