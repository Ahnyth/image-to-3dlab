"""Two models of the same figure compared by colour: the patches that look different.

A game model is thinned, unwrapped and baked anew from its source, and a check that counts
holes in motion does not see a wrong colour: a bake once put a dark, faceted patch on a cape's
back that only a look at the colour showed. This draws both models from eight sides around the
figure in their base colour as a game shows it (the nearest face, back faces culled, no light),
and reports every patch whose colour differs by more than `delta` (CIE76 delta E, about 2 is
just visible, 10 is plain) and covers at least `min_share` of the figure in that view. Colour
edges may move a few pixels (a thinner mesh, other UVs), so each pixel is matched against the
nearest few pixels of the other model, and the outline's rim is left out.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .photo_paint import _accessor, _split_glb, rasterize

_D = 0.5 ** 0.5
# (name, direction from the figure toward the camera); glTF's front is +Z, its up +Y, so the
# figure's own left side faces +X
VIEWS = (
    ("front", (0.0, 0.0, 1.0)), ("back", (0.0, 0.0, -1.0)),
    ("left side", (1.0, 0.0, 0.0)), ("right side", (-1.0, 0.0, 0.0)),
    ("front left", (_D, 0.0, _D)), ("front right", (-_D, 0.0, _D)),
    ("back left", (_D, 0.0, -_D)), ("back right", (-_D, 0.0, -_D)),
)


@dataclass
class Model:
    """Triangles with glTF UVs, each reading the base colour of its part."""
    positions: np.ndarray        # (V, 3)
    uvs: np.ndarray              # (V, 2), glTF's: v runs down the picture
    faces: np.ndarray            # (F, 3)
    part: np.ndarray             # (F,) index into `textures`
    textures: list               # (H, W, 3) uint8 per part

    @classmethod
    def single(cls, positions, uvs, faces, texture) -> Model:
        faces = np.asarray(faces, np.int64)
        return cls(np.asarray(positions, float), np.asarray(uvs, float), faces,
                   np.zeros(len(faces), np.int64), [np.asarray(texture, np.uint8)[..., :3]])


@dataclass
class Patch:
    view: str
    share: float                 # of the figure's pixels in that view
    delta_e: float               # median colour difference inside the patch
    lightness: float             # new minus old, CIE L* (0-100)
    where: str                   # e.g. "upper middle"
    box: tuple                   # (x0, y0, x1, y1) in the view's pixels

    @property
    def darker(self) -> bool:
        return self.lightness < 0

    def describe(self) -> str:
        tone = "darker" if self.darker else "lighter"
        if abs(self.lightness) < self.delta_e / 3:
            tone = "another colour"
        return (f"{self.view}: a {tone} patch, {self.share * 100:.1f} % of the figure in that view, "
                f"{self.where} (dE {self.delta_e:.0f}, L* {self.lightness:+.0f})")


# --- reading ---------------------------------------------------------------------------------

def _local(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], float).reshape(4, 4).T
    m = np.eye(4)
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, :3] *= np.asarray(node.get("scale", (1.0, 1.0, 1.0)), float)
    m[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return m


def _image(doc: dict, binary: bytes, index: int) -> np.ndarray:
    view = doc["bufferViews"][doc["images"][index]["bufferView"]]
    start = view.get("byteOffset", 0)
    picture = Image.open(io.BytesIO(binary[start:start + view["byteLength"]]))
    return np.asarray(picture.convert("RGB"))


def _base_colour(doc: dict, binary: bytes, material: int | None, cache: dict) -> np.ndarray:
    pbr = doc["materials"][material].get("pbrMetallicRoughness", {}) if material is not None else {}
    factor = np.asarray(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))[:3], float)
    ref = pbr.get("baseColorTexture")
    if ref is None:
        return np.round(np.clip(factor, 0, 1) * 255).astype(np.uint8).reshape(1, 1, 3)
    texture = doc["textures"][ref["index"]]
    source = texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source", texture.get("source"))
    if source not in cache:
        cache[source] = _image(doc, binary, source)
    picture = cache[source]
    if np.allclose(factor, 1.0):
        return picture
    return np.round(picture * np.clip(factor, 0, 1)).astype(np.uint8)


def read_model(path: Path) -> Model:
    """Every triangle primitive of a GLB in the scene's coordinates, each with its base colour.

    Node transforms are applied, except on a skinned mesh, whose node glTF says to ignore.
    """
    doc, binary = _split_glb(Path(path).read_bytes())
    nodes = doc.get("nodes", [])
    roots = doc["scenes"][doc.get("scene", 0)]["nodes"] if doc.get("scenes") else range(len(nodes))
    world: dict[int, np.ndarray] = {}
    stack = [(n, np.eye(4)) for n in roots]
    while stack:
        n, parent = stack.pop()
        world[n] = parent @ _local(nodes[n])
        stack.extend((c, world[n]) for c in nodes[n].get("children", []))
    positions, uvs, faces, part, textures, cache, materials = [], [], [], [], [], {}, {}
    count = 0
    for n, matrix in world.items():
        node = nodes[n]
        if "mesh" not in node:
            continue
        m = np.eye(4) if "skin" in node else matrix
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim["attributes"]
            p = _accessor(doc, binary, attrs["POSITION"]).astype(float)
            p = p @ m[:3, :3].T + m[:3, 3]
            uv = (_accessor(doc, binary, attrs["TEXCOORD_0"]).astype(float) if "TEXCOORD_0" in attrs
                  else np.zeros((len(p), 2)))
            f = (_accessor(doc, binary, prim["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in prim
                 else np.arange(len(p)).reshape(-1, 3))
            key = prim.get("material")
            if key not in materials:
                materials[key] = len(textures)
                textures.append(_base_colour(doc, binary, key, cache))
            positions.append(p)
            uvs.append(uv)
            faces.append(f + count)
            part.append(np.full(len(f), materials[key]))
            count += len(p)
    if not faces:
        raise ValueError(f"{path}: no triangles")
    return Model(np.concatenate(positions), np.concatenate(uvs), np.concatenate(faces),
                 np.concatenate(part), textures)


# --- drawing ---------------------------------------------------------------------------------

def frames(old: Model, new: Model) -> tuple[tuple, tuple]:
    """The (centre, scale) each model is drawn with: one shared frame when both stand in the same
    place at about the same size (two versions of one game model), otherwise each its own."""
    def box(m):
        return m.positions.min(0), m.positions.max(0)

    (a0, a1), (b0, b1) = box(old), box(new)
    size = max(float((a1 - a0).max()), float((b1 - b0).max()), 1e-9)
    if np.abs(a0 - b0).max() < 0.05 * size and np.abs(a1 - b1).max() < 0.05 * size:
        lo, hi = np.minimum(a0, b0), np.maximum(a1, b1)
        shared = ((lo + hi) / 2, 1.0 / float((hi - lo).max()))
        return shared, shared
    return (((a0 + a1) / 2, 1.0 / float((a1 - a0).max())), ((b0 + b1) / 2, 1.0 / float((b1 - b0).max())))


def render(model: Model, direction, size: int, frame: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The model's base colour seen orthographically from `direction` (figure toward camera), Y up:
    (size, size, 3) floats 0..1 and the mask of covered pixels."""
    centre, scale = frame or ((model.positions.min(0) + model.positions.max(0)) / 2,
                              1.0 / float(np.ptp(model.positions, axis=0).max()))
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(-d, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, -d)
    p = (model.positions - centre) * scale * 0.94
    x = (p @ right + 0.5) * size
    y = (0.5 - p @ up) * size
    depth = -(p @ d)
    tri = model.positions[model.faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    keep = np.flatnonzero(normal @ d > 0)          # back faces culled, as a game draws them
    rgb = np.zeros((size, size, 3))
    if not len(keep):
        return rgb, np.zeros((size, size), bool)
    f = model.faces[keep]
    face_of, bary = rasterize(np.stack([x, y], 1)[f], depth[f], (size, size))
    covered = face_of >= 0
    hit = keep[face_of[covered]]
    uv = (bary[covered][:, :, None] * model.uvs[model.faces[hit]]).sum(1)
    for k, picture in enumerate(model.textures):
        mine = model.part[hit] == k
        if not mine.any():
            continue
        h, w = picture.shape[:2]
        u = np.mod(uv[mine, 0], 1.0)
        v = np.mod(uv[mine, 1], 1.0)
        rows = np.minimum((v * h).astype(int), h - 1)
        cols = np.minimum((u * w).astype(int), w - 1)
        idx = np.flatnonzero(covered)[mine]
        rgb.reshape(-1, 3)[idx] = picture[rows, cols] / 255.0
    return rgb, covered


# --- comparing -------------------------------------------------------------------------------

def lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0..1 to CIE L*a*b* (D65)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _nearest_difference(a: np.ndarray, b: np.ndarray, b_ok: np.ndarray, reach: int) -> np.ndarray:
    """Per pixel of `a`: the smallest colour difference to a covered pixel of `b` within `reach`."""
    best = np.full(a.shape[:2], np.inf)
    padded = np.pad(b, ((reach, reach), (reach, reach), (0, 0)))
    ok = np.pad(b_ok, reach)
    h, w = a.shape[:2]
    for dy in range(2 * reach + 1):
        for dx in range(2 * reach + 1):
            cand = padded[dy:dy + h, dx:dx + w]
            d = np.linalg.norm(a - cand, axis=-1)
            d[~ok[dy:dy + h, dx:dx + w]] = np.inf
            np.minimum(best, d, out=best)
    return best


def _where(cy: float, cx: float, rows: np.ndarray, cols: np.ndarray) -> str:
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    t = (cy - top) / max(bottom - top, 1)
    s = (cx - left) / max(right - left, 1)
    height = "upper" if t < 1 / 3 else "lower" if t > 2 / 3 else "middle"
    across = "left" if s < 1 / 3 else "right" if s > 2 / 3 else "middle"
    return height if height == across else f"{height} {across}" if across != "middle" else f"{height} middle"


def compare_view(old: Model, new: Model, name: str, direction, size: int = 384, delta: float = 12.0,
                 min_share: float = 0.003, reach: int = 2, rim: int = 2, frame=None) -> tuple[list, dict]:
    """The patches of one view, and the two pictures and the changed mask for a marked picture."""
    fo, fn = frame or frames(old, new)
    a, ca = render(old, direction, size, fo)
    b, cb = render(new, direction, size, fn)
    la, lb = lab(a), lab(b)
    inside = ndimage.binary_erosion(ca, iterations=rim) & ndimage.binary_erosion(cb, iterations=rim)
    diff = np.maximum(_nearest_difference(la, lb, cb, reach), _nearest_difference(lb, la, ca, reach))
    changed = inside & (diff > delta)
    changed = ndimage.binary_opening(changed, structure=np.ones((3, 3), bool))
    labels, count = ndimage.label(changed)
    figure = max(int(ca.sum()), 1)
    rows, cols = np.nonzero(ca)
    patches = []
    for k in range(1, count + 1):
        mask = labels == k
        area = int(mask.sum())
        if area < max(min_share * figure, 16):
            continue
        ys, xs = np.nonzero(mask)
        patches.append(Patch(view=name, share=area / figure, delta_e=float(np.median(diff[mask])),
                             lightness=float(lb[..., 0][mask].mean() - la[..., 0][mask].mean()),
                             where=_where(ys.mean(), xs.mean(), rows, cols),
                             box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    patches.sort(key=lambda p: -p.share)
    return patches, {"old": a, "new": b, "old_covered": ca, "new_covered": cb, "changed": changed}


def compare(old: Model, new: Model, size: int = 384, delta: float = 12.0, min_share: float = 0.003,
            views=VIEWS) -> list[Patch]:
    """Every patch, over all `views`, where `new` differs in colour from `old`."""
    frame = frames(old, new)
    found = []
    for name, direction in views:
        found += compare_view(old, new, name, direction, size, delta, min_share, frame=frame)[0]
    return found


def marked_picture(old: Model, new: Model, path: Path, size: int = 384, delta: float = 12.0,
                   min_share: float = 0.003, views=VIEWS) -> list[Patch]:
    """`compare`, and a picture of every view with a patch: old | new | new with the patches in red."""
    frame = frames(old, new)
    rows, found = [], []
    for name, direction in views:
        patches, pics = compare_view(old, new, name, direction, size, delta, min_share, frame=frame)
        found += patches
        if not patches:
            continue
        tiles = []
        for key, cov in (("old", "old_covered"), ("new", "new_covered")):
            img = np.full((size, size, 3), 30, np.uint8)
            img[pics[cov]] = np.round(pics[key][pics[cov]] * 255).astype(np.uint8)
            tiles.append(img)
        marked = tiles[1].copy()
        marked[pics["changed"]] = (marked[pics["changed"]] * 0.3 + np.array([230, 30, 30]) * 0.7).astype(np.uint8)
        tiles.append(marked)
        row = Image.new("RGB", (3 * size, size + 18), (12, 12, 14))
        draw = ImageDraw.Draw(row)
        for i, tile in enumerate(tiles):
            row.paste(Image.fromarray(tile), (i * size, 18))
        for p in patches:
            x0, y0, x1, y1 = p.box
            draw.rectangle((2 * size + x0 - 3, 18 + y0 - 3, 2 * size + x1 + 3, 18 + y1 + 3), outline=(255, 220, 0))
        draw.text((4, 3), f"{name}: old | new | changed in red", fill=(255, 255, 120))
        rows.append(row)
    if rows:
        sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        sheet.save(path)
    return found
