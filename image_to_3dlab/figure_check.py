"""A finished game figure checked the way a second reviewer looks at it, by numbers.

A figure can pass every step of the pipeline and still show a fault a person sees at once up
close: a hand with six fingers, a second eye painted on a temple, a dark stroke along the jaw,
a neck that reads as a pale ring, hair at the nape cut into saw teeth, a stray bright flake.
This draws the figure from fixed views around its head, neck and hands (found from its
skeleton), in its own base colour, and measures each of those faults:

* fingers: cross-sections across each hand, from the wrist to the tips, counted in pieces;
* dark marks: thin dark lines or dots on skin that does not face the front (a jaw stroke from
  below, a lash line or eye on a temple), the face's own eyes, brows and mouth left out;
* neck ring: a hard, level colour step running around the neck's skin;
* jagged outline: small saw teeth along the head's outline seen from the sides and the back;
* flakes: tiny separate bits of the model, and small bright specks on the head;
* colour (with a version before): the patches that changed colour (`colour_compare`).

Some of these a number can judge (fingers, the neck ring, colour, holes in motion); others it can
only point at, because a drawn neck line or spiky hair looks like a fault to a measure: those
come back as "look", with the spot circled. Given the version before, a "look" that got worse
than it was there is a failure. It only reports. `check()` returns one result per check, each
with its number, its limit and the views and boxes where it found something; `write_report()`
turns that into an HTML page with every view and the spots circled.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import colour_compare as cc
from .photo_paint import _accessor, _split_glb, rasterize

FRONT = np.array([0.0, 0.0, 1.0])            # glTF: the figure faces +Z, Y is up
UP = np.array([0.0, 1.0, 0.0])


# --- reading ---------------------------------------------------------------------------------

@dataclass
class Figure:
    """A game figure: its coloured triangles, its joints and each vertex's skin weights."""
    model: cc.Model
    joints: dict                             # Unity bone name -> (3,) bind-pose position
    names: list                              # the skin's joint names, in weight order
    weights: np.ndarray | None               # (V, J) dense skin weights, rows sum to 1
    glow: cc.Model | None = None             # the emissive colour, where the figure glows
    parents: dict | None = None              # joint name -> its parent joint's name (None at the root)

    @property
    def height(self) -> float:
        return float(np.ptp(self.model.positions[:, 1]))

    def owned_by(self, *bones: str) -> np.ndarray:
        """Per vertex: the share of its weight on BONES (zeros without a skin)."""
        if self.weights is None:
            return np.zeros(len(self.model.positions))
        cols = [self.names.index(b) for b in bones if b in self.names]
        return self.weights[:, cols].sum(axis=1) if cols else np.zeros(len(self.model.positions))


def read_figure(path: Path) -> Figure:
    """A GLB's triangles (as `colour_compare.read_model`), its joints and its skin weights."""
    model = cc.read_model(path)
    doc, binary = _split_glb(Path(path).read_bytes())
    nodes = doc.get("nodes", [])
    roots = doc["scenes"][doc.get("scene", 0)]["nodes"] if doc.get("scenes") else range(len(nodes))
    world: dict[int, np.ndarray] = {}
    stack = [(n, np.eye(4)) for n in roots]
    while stack:
        n, parent = stack.pop()
        world[n] = parent @ cc._local(nodes[n])
        stack.extend((c, world[n]) for c in nodes[n].get("children", []))
    skins = doc.get("skins") or []
    names = [nodes[j].get("name", f"joint{j}") for j in skins[0]["joints"]] if skins else []
    joints = {nodes[j].get("name", f"joint{j}"): world.get(j, np.eye(4))[:3, 3].copy()
              for j in (skins[0]["joints"] if skins else [])}
    in_skin = set(skins[0]["joints"]) if skins else set()
    parent_of = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    parents = {}
    for j in in_skin:
        p = parent_of.get(j)
        while p is not None and p not in in_skin:      # the nearest joint above it (a scene node may sit between)
            p = parent_of.get(p)
        parents[nodes[j].get("name", f"joint{j}")] = None if p is None else nodes[p].get("name", f"joint{p}")
    weights = []
    for n in world:
        node = nodes[n]
        if "mesh" not in node:
            continue
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim["attributes"]
            count = doc["accessors"][attrs["POSITION"]]["count"]
            if not names or "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
                weights.append(None if names else np.zeros((count, 0)))
                continue
            index = _accessor(doc, binary, attrs["JOINTS_0"]).astype(np.int64)
            w = _accessor(doc, binary, attrs["WEIGHTS_0"]).astype(np.float64)
            kind = doc["accessors"][attrs["WEIGHTS_0"]]["componentType"]
            if kind != 5126:
                w /= {5121: 255.0, 5123: 65535.0}[kind]
            dense = np.zeros((count, len(names)))
            np.add.at(dense, (np.repeat(np.arange(count), 4), index.reshape(-1)), w.reshape(-1))
            weights.append(dense)
    dense = None
    if names and weights and all(w is not None for w in weights):
        dense = np.concatenate(weights)
        dense /= np.maximum(dense.sum(axis=1, keepdims=True), 1e-9)
    return Figure(model, joints, names, dense, cc.read_glow(path), parents or None)


# --- drawing ---------------------------------------------------------------------------------

@dataclass
class View:
    """One orthographic picture of the figure: colour, light, and what each pixel shows."""
    name: str
    direction: np.ndarray                    # from the figure toward the camera
    rgb: np.ndarray                          # (S, S, 3) base colour 0..1
    covered: np.ndarray                      # (S, S) bool
    face: np.ndarray                         # (S, S) triangle index, -1 where none
    normal: np.ndarray                       # (S, S, 3) the triangle's outward normal
    point: np.ndarray                        # (S, S, 3) the surface point
    glow: np.ndarray                         # (S, S) bool: the figure glows there
    scale: float                             # model units per pixel

    @property
    def lit(self) -> np.ndarray:
        """The colour under a light at the camera, so the shape reads as well as the paint."""
        shade = 0.45 + 0.55 * np.clip(self.normal @ self.direction, 0, 1)
        picture = self.rgb * shade[..., None]
        picture[~self.covered] = (0.11, 0.11, 0.13)
        return picture


def _axes(direction):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    up = UP if abs(d @ UP) < 0.95 else -FRONT
    right = np.cross(-d, up)
    right /= np.linalg.norm(right)
    return d, right, np.cross(right, -d)


def draw(figure: Figure, name: str, direction, centre, half: float, size: int = 320) -> View:
    """FIGURE seen from DIRECTION, the square of HALF model units around CENTRE filling SIZE pixels.

    Back faces are culled and the nearest face wins, as a game draws it."""
    model = figure.model
    d, right, up = _axes(direction)
    p = model.positions - np.asarray(centre, float)
    x = (p @ right / (2 * half) + 0.5) * size
    y = (0.5 - p @ up / (2 * half)) * size
    depth = -(p @ d)
    tri = model.positions[model.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    keep = np.flatnonzero(normals @ d > 0)
    face = np.full((size, size), -1, np.int64)
    rgb = np.zeros((size, size, 3))
    normal = np.zeros((size, size, 3))
    point = np.zeros((size, size, 3))
    glow = np.zeros((size, size), bool)
    if len(keep):
        f = model.faces[keep]
        face_of, bary = rasterize(np.stack([x, y], 1)[f], depth[f], (size, size))
        covered = face_of >= 0
        hit = keep[face_of[covered]]
        face[covered] = hit
        normal[covered] = normals[hit]
        point[covered] = (bary[covered][:, :, None] * model.positions[model.faces[hit]]).sum(1)
        rgb[covered] = _sample(model, hit, bary[covered])
        if figure.glow is not None:
            g = _sample(figure.glow, hit, bary[covered])
            glow[covered] = g.max(axis=1) > 0.08
    return View(name, d, rgb, face >= 0, face, normal, point, glow, 2 * half / size)


def _sample(model: cc.Model, hit: np.ndarray, bary: np.ndarray) -> np.ndarray:
    uv = (bary[:, :, None] * model.uvs[model.faces[hit]]).sum(1)
    out = np.zeros((len(hit), 3))
    for k, picture in enumerate(model.textures):
        mine = model.part[hit] == k
        if not mine.any():
            continue
        h, w = picture.shape[:2]
        rows = np.minimum((np.mod(uv[mine, 1], 1.0) * h).astype(int), h - 1)
        cols = np.minimum((np.mod(uv[mine, 0], 1.0) * w).astype(int), w - 1)
        out[mine] = picture[rows, cols] / 255.0
    return out


# --- where the parts are ---------------------------------------------------------------------

@dataclass
class Head:
    """The head and neck found from the skeleton (or from the figure's top without one)."""
    centre: np.ndarray                       # where the close views look
    half: float                              # half the close views' width
    neck_low: float                          # the neck's lowest height (y)
    chin: float                              # the head joint's height: the neck's top
    top: float                               # the head's highest point (a hat's top)
    size: float                              # the head's own height without a hat, roughly


def find_head(figure: Figure) -> Head:
    p = figure.model.positions
    top_all = float(p[:, 1].max())
    j = figure.joints
    if "Head" in j:
        head = j["Head"]
        neck = j.get("Neck", head - np.array([0.0, 0.04 * figure.height, 0.0]))
        owned = figure.owned_by("Head") > 0.5
        mine = p[owned] if owned.sum() > 50 else p[p[:, 1] > head[1]]
        top = float(mine[:, 1].max())
        # a bare head is about 2.2 neck lengths tall; a hat makes `top` higher, so take the smaller
        neck_len = max(float(head[1] - neck[1]), 0.02 * figure.height)
        size = min(top - float(head[1]), 2.6 * neck_len, 0.16 * figure.height)
        return _framed(np.asarray(head, float), float(neck[1]), float(head[1]), top, size)
    size = 0.12 * figure.height
    mid = p[p[:, 1] > top_all - 2 * size].mean(axis=0)
    return _framed(np.array([mid[0], top_all - size, mid[2]]), top_all - 1.4 * size, top_all - size, top_all, size)


def _framed(joint, neck_low, chin, top, size) -> Head:
    """The close views take in the neck from its foot to a little above the eyes and brow."""
    lower, upper = neck_low - 0.1 * size, chin + 0.75 * size
    centre = np.array([joint[0], (lower + upper) / 2, joint[2]])
    return Head(centre, (upper - lower) / 2, neck_low, chin, top, size)


_S = 0.5 ** 0.5
HEAD_VIEWS = (
    ("front", (0.0, 0.0, 1.0)),
    ("left side", (1.0, 0.0, 0.0)), ("right side", (-1.0, 0.0, 0.0)),
    ("back", (0.0, 0.0, -1.0)),
    ("from below", (0.0, -0.75, 0.66)),
    ("below left", (0.6, -0.6, 0.53)), ("below right", (-0.6, -0.6, 0.53)),
    ("back left", (_S, 0.0, -_S)), ("back right", (-_S, 0.0, -_S)),
)


def head_views(figure: Figure, head: Head, size: int = 320) -> dict:
    return {name: draw(figure, name, d, head.centre, head.half, size) for name, d in HEAD_VIEWS}


# --- results ---------------------------------------------------------------------------------

@dataclass
class Result:
    name: str                                # what was checked, in plain words
    status: str                              # "pass", "fail", "look" (a hint to look at) or "info"
    value: float | int | str
    limit: str
    message: str
    spots: list = field(default_factory=list)   # (view name, (x0, y0, x1, y1))
    picture: str = ""                           # a marked picture of its own (the colour check)

    def as_dict(self) -> dict:
        return {"check": self.name, "status": self.status, "value": self.value, "limit": self.limit,
                "message": self.message, "spots": [[v, list(b)] for v, b in self.spots]}


def _boxes(mask: np.ndarray, min_area: int):
    labels, _ = ndimage.label(mask)
    out = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        area = int((labels[sl] == k).sum())
        if area >= min_area:
            out.append(((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop), area))
    return out


# --- skin --------------------------------------------------------------------------------------

def skin_colour(front: View, head: Head) -> np.ndarray | None:
    """The face's skin in CIE Lab: the most common colour on the front-facing middle of the face."""
    p, n = front.point, front.normal
    # the lower face (cheeks, nose, mouth, chin) is skin on nearly every figure; brows, eyes and
    # fringes sit higher. Of its colours, take the most common one.
    band = front.covered & (n @ FRONT > 0.6) & ~front.glow
    band &= (p[..., 1] > head.chin + 0.05 * head.size) & (p[..., 1] < head.chin + 0.42 * head.size)
    band &= np.abs(p[..., 0] - head.centre[0]) < 0.22 * head.size
    if band.sum() < 30:
        return None
    lab = cc.lab(front.rgb[band])
    centre = np.median(lab, axis=0)
    for _ in range(3):      # settle on the main cluster: mouth lines and shadows drop out
        near = np.linalg.norm(lab - centre, axis=1) < 10
        if near.sum() < 10:
            break
        centre = np.median(lab[near], axis=0)
    return centre


SKIN_REACH = 14.0          # delta E from the face's skin that still counts as skin


def skin_mask(view: View, skin: np.ndarray, reach: float = SKIN_REACH) -> np.ndarray:
    lab = cc.lab(view.rgb)
    return view.covered & ~view.glow & (np.linalg.norm(lab - skin, axis=-1) < reach)


# --- the checks --------------------------------------------------------------------------------

DARK_DEPTH = 16.0          # L* a mark is darker than the skin around it
DARK_LIMIT = 0.012         # share of the skin in a view that may be dark marks


def face_front(figure: Figure, head: Head, bins: int = 48):
    """The head's frontmost depth (z) by height, from the vertices: a function of y."""
    p = figure.model.positions
    lo, hi = head.neck_low, head.chin + head.size
    sel = (p[:, 1] >= lo) & (p[:, 1] <= hi) & (np.abs(p[:, 0] - head.centre[0]) < 0.5 * head.size)
    if not sel.any():           # no corner in the head's band (a coarse block): its front is the model's
        return lambda y: np.full(np.shape(y), float(p[:, 2].max()))
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(p[sel, 1], edges) - 1, 0, bins - 1)
    front = np.full(bins, -np.inf)
    np.maximum.at(front, idx, p[sel, 2])
    for i in range(bins):
        if not np.isfinite(front[i]):
            front[i] = front[i - 1] if i and np.isfinite(front[i - 1]) else p[sel, 2].max()
    front = ndimage.maximum_filter1d(front, 3)

    def at(y):
        return front[np.clip(((np.asarray(y) - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)]
    return at


BEHIND = 0.10              # a mark this far behind the face's front (share of the head) is no feature of the face


def dark_marks(views: dict, skin: np.ndarray, head: Head, front=None) -> Result:
    """Thin dark lines and dots on skin away from the face's front (jaw strokes, temple eyes).

    A black top-hat on lightness finds what is darker than its surroundings and narrower than a
    small disc; such a pixel counts when skin surrounds it, its surface does not face the front
    and it lies behind the face's front, so the face's own eyes, brows and mouth are left out."""
    worst, spots, total = 0.0, [], 0
    for name in ("from below", "below left", "below right"):
        v = views[name]
        lab = cc.lab(v.rgb)
        light = np.where(v.covered, lab[..., 0], 0.0)
        r = max(2, int(round(0.035 * head.size / v.scale)))
        disc = _disc(r)
        closed = ndimage.grey_closing(light, footprint=disc)
        on_skin = skin_mask(v, skin)
        big = _disc(2 * r).astype(float)
        around = ndimage.convolve(on_skin.astype(float), big / big.sum(), mode="constant")
        facing_front = (v.normal @ FRONT) > 0.45
        # the jaw and the throat: below the cheekbones, inside the ears
        in_head = (v.point[..., 1] > head.neck_low) & (v.point[..., 1] < head.chin + 0.35 * head.size)
        in_head &= np.abs(v.point[..., 0] - head.centre[0]) < 0.33 * head.size
        mark = v.covered & (around > 0.45) & ~facing_front & in_head & ~v.glow
        if front is not None:
            mark &= (front(v.point[..., 1]) - v.point[..., 2]) > BEHIND * head.size
        mark &= (closed - light) > DARK_DEPTH
        mark &= ndimage.binary_erosion(v.covered, iterations=2)
        mark = ndimage.binary_opening(mark, structure=np.ones((2, 2), bool))
        base = max(int(on_skin.sum()), 1)
        found = _boxes(mark, 6)
        share = sum(a for _, a in found) / base
        total += sum(a for _, a in found)
        worst = max(worst, share)
        spots += [(name, b) for b, _ in found]
    status = "look" if worst > DARK_LIMIT else "pass"
    return Result("dark strokes under the jaw", status, round(worst * 100, 2), f"a hint above {DARK_LIMIT * 100:g} % "
                  "of the skin in a view", f"{len(spots)} dark mark(s), {total} pixels, worst view {worst * 100:.2f} % "
                  "of its skin (drawn neck lines count too: look at the circles)", spots if status == "look" else [])


def _disc(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


RING_STEP = 9.0            # L* across a seam that reads as an edge
RING_LIMIT = 0.80          # share of the neck's width the step runs across


def neck_ring(views: dict, skin: np.ndarray, head: Head) -> Result:
    """A hard, level colour step around the neck: two paints meeting in a line.

    Row by row through the neck in the front and from-below views: where the skin a few pixels
    above and below differs by more than RING_STEP in lightness, in the same direction, across
    most of the neck's width, the neck reads as a ring."""
    worst, spots = 0.0, []
    for name in ("front", "from below"):
        v = views[name]
        lab = cc.lab(v.rgb)
        sk = skin_mask(v, skin, reach=30.0)
        y = v.point[..., 1]
        neck = sk & (y > head.neck_low) & (y < head.chin + 0.25 * head.size)
        L = lab[..., 0]
        k = max(2, int(round(0.02 * head.size / v.scale)))
        above, below = np.roll(L, k, axis=0), np.roll(L, -k, axis=0)
        ok = np.roll(neck, k, axis=0) & np.roll(neck, -k, axis=0) & neck
        step = np.where(ok, below - above, 0.0)
        for row in range(k, L.shape[0] - k):
            width = int(neck[row].sum())
            if width < 8:
                continue
            for sign in (1, -1):
                hard = int(((sign * step[row]) > RING_STEP).sum())
                share = hard / width
                worst = max(worst, share)
                if share > RING_LIMIT:
                    cols = np.flatnonzero(neck[row])
                    spots.append((name, (int(cols.min()), row - k, int(cols.max()) + 1, row + k)))
    spots = _merge(spots)
    status = "fail" if worst > RING_LIMIT else "pass"
    return Result("neck ring", status, round(worst * 100, 1), f"a step of {RING_STEP:g} L* across at most "
                  f"{RING_LIMIT * 100:g} % of the neck", f"the hardest level step runs across {worst * 100:.0f} % of the "
                  "neck's width", spots if status == "fail" else [])


def _merge(spots):
    out = []
    for name, box in spots:
        for i, (n2, b2) in enumerate(out):
            if n2 == name and not (box[0] > b2[2] or box[2] < b2[0] or box[1] > b2[3] + 2 or box[3] < b2[1] - 2):
                out[i] = (name, (min(box[0], b2[0]), min(box[1], b2[1]), max(box[2], b2[2]), max(box[3], b2[3])))
                break
        else:
            out.append((name, box))
    return out


TOOTH_LIMIT = 0.006        # share of the head's outline area that may be small notches


def jagged_outline(views: dict, head: Head) -> Result:
    """Saw teeth along the head's outline seen from the sides and the back.

    The outline is closed with a small disc (about 3 % of the head): what that fills are notches
    narrower than the disc, the gaps between teeth. Big bays (under a hat's brim, between the
    chin and the collar) are wider than the disc and stay out."""
    worst, spots = 0.0, []
    for name in ("left side", "right side", "back", "back left", "back right"):
        v = views[name]
        y = v.point[..., 1]
        head_px = v.covered & (y > head.chin - 0.1 * head.size)
        r = max(2, int(round(0.03 * head.size / v.scale)))
        disc = _disc(r)
        filled = ndimage.binary_closing(v.covered, structure=disc, iterations=1)
        notches = filled & ~v.covered
        # only beside the head's own pixels, not along the collar or the hat
        notches &= ndimage.binary_dilation(head_px, structure=disc)
        notches = ndimage.binary_opening(notches, structure=np.ones((2, 2), bool))
        found = [(b, a) for b, a in _boxes(notches, 4)]
        base = max(int(head_px.sum()), 1)
        share = sum(a for _, a in found) / base
        worst = max(worst, share)
        spots += [(name, b) for b, _ in found]
    status = "look" if worst > TOOTH_LIMIT else "pass"
    return Result("jagged outline", status, round(worst * 100, 2), f"a hint above {TOOTH_LIMIT * 100:g} % of the "
                  "head in a view", f"{len(spots)} notch(es); worst view {worst * 100:.2f} % of the head (spiky or "
                  "long hair counts too: look at the circles)", spots if status == "look" else [])


FLAKE_SHARE = 0.005        # a piece under this share of the largest piece's area is a bit, not a part
FLAKE_PIXELS = 6           # a bit seen with this many pixels in a close view is a flake


def flakes(figure: Figure, views: dict) -> Result:
    """Small loose bits of the model that can be seen: a flake of old hair under a brim, a chip
    left by a cut. A part of its own (a hat, a grafted face, an eye) is bigger and stays out;
    crumbs inside the body or too small to cover a few pixels up close stay out too."""
    piece, area = pieces(figure.model)
    small = np.flatnonzero(area < FLAKE_SHARE * area.max())
    seen: dict = {}
    for name, v in views.items():
        hit = piece[v.face[v.covered]]
        ys, xs = np.nonzero(v.covered)
        for k in np.intersect1d(np.unique(hit), small):
            mine = hit == k
            n = int(mine.sum())
            if FLAKE_PIXELS <= n <= 0.01 * len(hit) and n > seen.get(int(k), (0,))[0]:
                box = (int(xs[mine].min()), int(ys[mine].min()), int(xs[mine].max()) + 1, int(ys[mine].max()) + 1)
                seen[int(k)] = (n, name, box)
    crumbs = int(len(small) - len(seen))
    spots = [(name, box) for _, name, box in seen.values()]
    status = "look" if seen else "pass"
    return Result("loose flakes", status, len(seen), f"no loose bit under {FLAKE_SHARE * 100:g} % of the largest "
                  f"piece covering {FLAKE_PIXELS}+ pixels up close",
                  f"{len(seen)} loose bit(s) that show up close; {crumbs} more too small or hidden to see", spots)


def pieces(model: cc.Model) -> tuple:
    """Each triangle's piece (vertices welded by position) and each piece's surface area."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    _, inverse = np.unique(np.round(model.positions, 6), axis=0, return_inverse=True)
    f = inverse.reshape(-1)[model.faces]
    n = int(f.max()) + 1
    graph = coo_matrix((np.ones(2 * len(f)), (np.r_[f[:, 0], f[:, 1]], np.r_[f[:, 1], f[:, 2]])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    tri = model.positions[model.faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    _, piece = np.unique(label[f[:, 0]], return_inverse=True)
    return piece, np.bincount(piece, weights=area)


# --- hands -------------------------------------------------------------------------------------

def _welded(model: cc.Model):
    points, inverse = np.unique(np.round(model.positions, 6), axis=0, return_inverse=True)
    return points, inverse.reshape(-1)[model.faces]


def section_pieces(points: np.ndarray, faces: np.ndarray, origin, axis, pixel: float) -> int:
    """How many separate solid pieces the plane through ORIGIN square to AXIS cuts out of the
    triangles FACES: the cut's outlines drawn on a grid of PIXEL, filled, and counted."""
    return len(section_parts(points, faces, origin, axis, pixel))


def section_parts(points: np.ndarray, faces: np.ndarray, origin, axis, pixel: float) -> np.ndarray:
    """The middles (in the cut's plane, in PIXEL units) of the separate solid pieces `section_pieces` counts."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    d = (points - np.asarray(origin, float)) @ axis
    fd = d[faces]
    crossing = faces[(fd.min(axis=1) < 0) & (fd.max(axis=1) > 0)]
    if not len(crossing):
        return np.zeros((0, 2))
    starts = np.full((len(crossing), 3), np.nan)
    stops = np.full((len(crossing), 3), np.nan)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[crossing[:, a]], d[crossing[:, b]]
        cut = np.flatnonzero((da < 0) != (db < 0))
        t = da[cut] / (da[cut] - db[cut])
        pa, pb = points[crossing[cut, a]], points[crossing[cut, b]]
        pts = pa + t[:, None] * (pb - pa)
        empty = np.isnan(starts[cut, 0])
        starts[cut[empty]] = pts[empty]
        stops[cut[~empty]] = pts[~empty]
    ok = ~np.isnan(starts[:, 0]) & ~np.isnan(stops[:, 0])
    starts, stops = starts[ok], stops[ok]
    side = np.cross(axis, [0.0, 1.0, 0.0] if abs(axis[1]) < 0.9 else [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    other = np.cross(axis, side)
    longest = float(np.linalg.norm(stops - starts, axis=1).max()) if len(starts) else 0.0
    steps = np.linspace(0.0, 1.0, int(min(64, np.ceil(2 * longest / pixel) + 2)))[:, None, None]
    samples = (starts[None] + steps * (stops - starts)[None]).reshape(-1, 3)
    uv = np.stack([samples @ side, samples @ other], axis=1) / pixel
    uv -= uv.min(axis=0) - 2
    grid = np.zeros(tuple((uv.max(axis=0) + 3).astype(int)[::-1]), bool)
    grid[uv[:, 1].astype(int), uv[:, 0].astype(int)] = True
    grid = ndimage.binary_fill_holes(ndimage.binary_dilation(grid))
    labels, _ = ndimage.label(grid)
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero(sizes >= 3) + 1
    if not len(keep):
        return np.zeros((0, 2))
    return np.asarray(ndimage.center_of_mass(grid, labels, keep), dtype=np.float64).reshape(-1, 2)


THUMB_APART = 1.5          # a piece this many times farther from its nearest neighbour than the rest are is a thumb


def digits_in_cut(middles: np.ndarray, fingers: int | None = None) -> int:
    """The fingers a cut across a hand shows: its pieces, less a thumb that reaches down among the fingers. Built alone
    and drawn flat, a hand's thumb is long and spread (Hero hands), so a cut below the knuckles can catch its tip beside
    the four fingers: five pieces, the thumb's set apart from the row (its nearest neighbour THUMB_APART times farther
    than the fingers' are). Five evenly spaced pieces are a digit too many and stay five."""
    fingers = FINGERS if fingers is None else fingers
    n = len(middles)
    if n <= fingers:
        return n
    gaps = np.linalg.norm(middles[:, None] - middles[None], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    nearest = gaps.min(axis=1)
    apart = int(np.argmax(nearest))
    rest = np.delete(nearest, apart)
    return n - 1 if nearest[apart] >= THUMB_APART * float(np.median(rest)) else n


HAND_PARTS = ("Hand", "Thumb", "Index", "Middle", "Ring", "Little")
FINGERS = 4                # a cut across the fingers of a five-digit hand: the thumb branched off before


def hand_bones(figure: Figure, side: str) -> tuple[str, list] | None:
    """(the hand's bone, it and every bone of it) for SIDE ("Left" or "Right", the figure's own): by Unity's names on
    a game figure, else by the skeleton's shape (a raw Rig output names its bones bone_0 ... in no telling order; the
    arms are found as `hero_hands.arm_bones` finds them, the hand being the forearm's end and all below it)."""
    if f"{side}Hand" in figure.joints:
        return f"{side}Hand", [n for n in figure.names if n.startswith(side) and any(k in n for k in HAND_PARTS)]
    if not figure.parents:
        return None
    from image_to_3dlab import hero_hands as hh

    try:
        arms, _ = hh.arm_bones(figure.parents, figure.joints)
    except (ValueError, StopIteration, KeyError):
        return None
    arm = arms.get(side.lower())
    return None if arm is None else (arm["hand"], [arm["hand"], *arm["below"]])


def hand(figure: Figure, side: str):
    """(the hand's vertex mask, the wrist, the axis to the fingertips, the hand's length), or None."""
    found = hand_bones(figure, side) if figure.weights is not None else None
    if found is None:
        return None
    bone, bones = found
    owned = figure.owned_by(*bones) > 0.5
    if owned.sum() < 30:
        return None
    wrist = figure.joints[bone]
    p = figure.model.positions[owned]
    reach = np.linalg.norm(p - wrist, axis=1)
    tips = p[reach >= np.percentile(reach, 90)]
    axis = tips.mean(axis=0) - wrist
    axis /= np.linalg.norm(axis)
    return owned, wrist, axis, float(np.percentile((p - wrist) @ axis, 99.5))


def fingers(figure: Figure) -> Result:
    """Cross-sections across each hand from the knuckles to the tips, counted in pieces.

    A cut across the fingers of a hand built well shows four pieces (the thumb branches off
    nearer the wrist); five is a hand with a digit too many; three or fewer are fingers grown
    together into a mitten."""
    points, wfaces = _welded(figure.model)
    per_side, worst = {}, "pass"
    for side in ("Left", "Right"):
        found = hand(figure, side)
        if found is None:
            continue
        owned, wrist, axis, length = found
        faces = wfaces[owned[figure.model.faces].any(axis=1)]
        counts = [digits_in_cut(section_parts(points, faces, wrist + t * axis, axis, length / 200))
                  for t in np.linspace(0.20, 0.95, 16) * length]
        most = max(counts)
        per_side[side.lower()] = most
        if most > FINGERS:
            worst = "fail"
        elif most < FINGERS and worst == "pass":
            worst = "look"
    if not per_side:
        return Result("fingers", "info", "-", f"{FINGERS} per hand", "no hands found in the skeleton with skin "
                      "weights: not counted")
    text = ", ".join(f"{s} hand {n}" for s, n in per_side.items())
    return Result("fingers", worst, text, f"{FINGERS} fingers per cut (more fails, fewer is a mitten)",
                  f"most separate fingers in a cut across the hand: {text} (the thumb branches off nearer the "
                  "wrist, or stands apart from the fingers where its tip reaches among them, so a five-digit hand "
                  "shows 4 here)")


# --- whole figure, hands, holes, colour ------------------------------------------------------

WHOLE_VIEWS = (("front", (0.0, 0.0, 1.0)), ("left side", (1.0, 0.0, 0.0)), ("back", (0.0, 0.0, -1.0)),
               ("right side", (-1.0, 0.0, 0.0)), ("game camera", (0.0, 0.75, 0.66)))


def whole_views(figure: Figure, size: int = 320) -> dict:
    p = figure.model.positions
    centre = (p.min(axis=0) + p.max(axis=0)) / 2
    half = 0.52 * float(np.ptp(p, axis=0).max())
    return {name: draw(figure, name, d, centre, half, size) for name, d in WHOLE_VIEWS}


def hand_views(figure: Figure, size: int = 240) -> dict:
    out = {}
    for side in ("Left", "Right"):
        found = hand(figure, side)
        if found is None:
            continue
        _, wrist, axis, length = found
        centre = wrist + 0.5 * length * axis
        outward = np.array([1.0 if side == "Left" else -1.0, 0.0, 0.0])
        for label, d in (("outside", outward), ("inside", -outward), ("front", FRONT), ("tips", -axis)):
            name = f"{side.lower()} hand {label}"
            out[name] = draw(figure, name, d, centre, 0.75 * length, size)
    return out


def open_edges(figure: Figure) -> Result:
    _, f = _welded(figure.model)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    e = e[e[:, 0] != e[:, 1]]
    _, counts = np.unique(e, axis=0, return_counts=True)
    n = int((counts == 1).sum())
    return Result("open edges", "info", n, "-", f"{n} edge(s) with a face on one side only (a seam left open, or a "
                  "piece's rim); the holes-in-motion check says whether any shows")


def pose_holes(report) -> Result:
    """Holes in motion, from game_ready's own report beside the model (its pose check)."""
    if report is None or not Path(report).exists():
        return Result("holes in motion", "info", "-", "0", "no game_ready report beside the model: not checked")
    data = json.loads(Path(report).read_text(encoding="utf-8"))
    pose = data.get("pose_check") or {}
    if "hole_px_left" not in pose:
        return Result("holes in motion", "info", "-", "0", "the game_ready report has no pose check")
    left = int(pose["hole_px_left"])
    return Result("holes in motion", "fail" if left else "pass", left, "0 see-through pixels",
                  f"{left} see-through pixel(s) left in {pose.get('poses', '?')} poses "
                  f"({pose.get('hole_px', 0)} found, the rest repaired)")


def colour_against(before: Figure, figure: Figure, picture=None, same_look: bool = False) -> Result:
    """Where the figure's colour changed against the version before (colour_compare): a spot to look
    at, since a new part changes colour on purpose; a failure when SAME_LOOK says the new version
    should only have lost faults (a rebuild of an approved figure)."""
    if picture is not None:
        patches = cc.marked_picture(before.model, figure.model, Path(picture), ignore=figure.glow)
    else:
        patches = cc.compare(before.model, figure.model, ignore=figure.glow)
    status = ("fail" if same_look else "look") if patches else "pass"
    text = "; ".join(p.describe() for p in patches[:4]) or "no patch changed colour"
    if len(patches) > 4:
        text += f" (and {len(patches) - 4} more)"
    result = Result("colour against the version before", status, len(patches), "no patch", text)
    if patches and picture is not None:
        result.picture = str(picture)
    return result


# --- all of it ---------------------------------------------------------------------------------

LOOKS = ("dark strokes under the jaw", "jagged outline", "loose flakes")
WORSE = 1.5                # a hint this much bigger than in the version before fails ...
MARGIN = {"dark strokes under the jaw": 0.5, "jagged outline": 0.2, "loose flakes": 1}   # ... and by this much


@dataclass
class Checked:
    path: Path
    results: list
    views: dict                              # group name -> {view name: View}

    @property
    def failed(self) -> bool:
        return any(r.status == "fail" for r in self.results)


def check(path, before=None, report=None, size: int = 320, colour_picture=None, same_look: bool = False) -> Checked:
    """Every check on the game figure at PATH. BEFORE is the version it replaces (optional);
    REPORT is game_ready's report (looked for beside the model when not given); SAME_LOOK says the
    new version should look like BEFORE except for faults it lost, so a colour change fails."""
    path = Path(path)
    if report is None:
        guess = path.with_name(path.stem + ".report.json")
        report = guess if guess.exists() else None
    figure = read_figure(path)
    head = find_head(figure)
    views = head_views(figure, head, size)
    results = [fingers(figure)]
    results += _head_checks(figure, head, views)
    results += [open_edges(figure), pose_holes(report)]
    if before is not None:
        old = read_figure(Path(before))
        old_head = find_head(old)
        was_all = {r.name: r for r in _head_checks(old, old_head, head_views(old, old_head, size))}
        for r in results:
            was = was_all.get(r.name)
            if r.name not in LOOKS or was is None or not isinstance(r.value, (int, float)) \
                    or not isinstance(was.value, (int, float)):
                continue
            if r.value > was.value * WORSE and r.value > was.value + MARGIN[r.name]:
                r.status = "fail"
                r.message += f"; worse than the version before ({was.value})"
            else:
                r.message += f"; the version before had {was.value}"
        results.append(colour_against(old, figure, colour_picture, same_look))
    return Checked(path, results, {"head": views, "whole": whole_views(figure, size),
                                   "hands": hand_views(figure)})


def _head_checks(figure: Figure, head: Head, views: dict) -> list:
    skin = skin_colour(views["front"], head)
    out = []
    if skin is None:
        out.append(Result("skin", "info", "-", "-", "no skin found on the lower face (a mask or a helmet): the jaw "
                          "and neck checks are skipped"))
    else:
        out += [neck_ring(views, skin, head), dark_marks(views, skin, head, face_front(figure, head))]
    out += [jagged_outline(views, head), flakes(figure, views)]
    return out


# --- the report --------------------------------------------------------------------------------

STATUS_COLOURS = {"pass": "#3fa34d", "fail": "#e04848", "look": "#e0951a", "info": "#8090a0"}
STATUS_WORDS = {"pass": "passed", "fail": "FAILED", "look": "look here", "info": "info"}


def sheet(views: dict, spots: list, columns: int = 5) -> Image.Image:
    """The views side by side, lit from the camera, each spot circled in red."""
    tiles = []
    for name, v in views.items():
        img = Image.fromarray((np.clip(v.lit, 0, 1) * 255).astype(np.uint8))
        pen = ImageDraw.Draw(img)
        for view, (x0, y0, x1, y1) in spots:
            if view == name:
                pen.ellipse((x0 - 7, y0 - 7, x1 + 7, y1 + 7), outline=(255, 40, 40), width=2)
        pen.text((5, 4), name, fill=(255, 240, 120))
        tiles.append(img)
    if not tiles:
        return Image.new("RGB", (8, 8))
    w, h = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns
    out = Image.new("RGB", (min(columns, len(tiles)) * w, rows * h), (18, 18, 20))
    for i, tile in enumerate(tiles):
        out.paste(tile, ((i % columns) * w, (i // columns) * h))
    return out


def write_report(checked: list, out, title: str = "Figure check", preface: str = "", extra: dict | None = None) -> Path:
    """An HTML page (and report.json) for one or more checked figures, side by side. PREFACE (HTML) goes above the
    figures' table (the Hero chain's stages); EXTRA joins report.json under "extra"."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = [c.path.stem for c in checked]
    rows = []
    for c in checked:
        rows += [r.name for r in c.results if r.name not in rows]
    table = ["<table><tr><th>check</th>" + "".join(f"<th>{html.escape(n)}</th>" for n in names) + "</tr>"]
    for row in rows:
        cells = []
        for c in checked:
            r = next((x for x in c.results if x.name == row), None)
            cells.append("<td></td>" if r is None else
                         f"<td title='{html.escape(r.message)}'><b style='color:{STATUS_COLOURS[r.status]}'>"
                         f"{STATUS_WORDS[r.status]}</b><br><small>{html.escape(str(r.value))}</small></td>")
        table.append(f"<tr><th>{html.escape(row)}</th>{''.join(cells)}</tr>")
    table.append("</table>")
    parts = []
    for c in checked:
        verdict = "fail" if c.failed else "pass"
        parts.append(f"<h2>{html.escape(c.path.stem)}: <span style='color:{STATUS_COLOURS[verdict]}'>"
                     f"{STATUS_WORDS[verdict]}</span></h2><ul>")
        for r in c.results:
            parts.append(f"<li><b style='color:{STATUS_COLOURS[r.status]}'>{STATUS_WORDS[r.status]}</b> "
                         f"{html.escape(r.name)}: {html.escape(r.message)} <small>(limit: {html.escape(r.limit)})"
                         "</small></li>")
        parts.append("</ul>")
        spots = [s for r in c.results for s in r.spots]
        for group, views in c.views.items():
            if not views:
                continue
            file = f"{c.path.stem}_{group}.jpg"
            sheet(views, spots if group == "head" else []).save(out / file, quality=90)
            parts.append(f"<p>{group}</p><img src='{file}'>")
        for r in c.results:
            if r.picture:
                parts.append(f"<p>{html.escape(r.name)} (old | new | changed in red)</p>"
                             f"<img src='{html.escape(Path(r.picture).name)}'>")
    page = ("<!doctype html><meta charset='utf-8'><title>" + html.escape(title) + "</title><style>"
            "body{background:#141416;color:#e8e8ea;font:14px system-ui;margin:16px}table{border-collapse:collapse}"
            "td,th{border:1px solid #333;padding:4px 8px;text-align:left;vertical-align:top}img{max-width:100%;"
            "display:block;margin:4px 0 14px}small{color:#aab}</style><h1>" + html.escape(title) + "</h1>"
            "<p>passed and FAILED are verdicts; <b style='color:#e0951a'>look here</b> points at a spot a measure "
            "cannot judge alone (circled in red); info is for the record. Hover a cell for its words.</p>"
            + preface + "".join(table) + "".join(parts))
    (out / "index.html").write_text(page, encoding="utf-8")
    report = {c.path.stem: {"failed": c.failed, "results": [r.as_dict() for r in c.results]} for c in checked}
    if extra is not None:
        report["extra"] = extra
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out / "index.html"
