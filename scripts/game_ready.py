#!/usr/bin/env python3
"""Make a rigged character game-ready: one clean mesh, one texture, real size, engine bone names.

The auto-rig's GLB is a good start, but a game engine wants more: one material with one texture
instead of three, the real height with the feet on the floor, bone names its humanoid avatar
understands, and at most four bones per vertex. This does it in one go and keeps every move in
the file:

1. Clean the mesh: weld split vertices, delete tiny floating bits, turn inside-out parts the
   right way, mark hard edges by angle.
2. One texture atlas (`--atlas`, 1024 by default): new UVs, the colour baked from the old
   textures. The normal map's detail goes into the colour as soft light from above
   (`--no-shade` leaves it out), so the model needs no normal map.
3. Optional, for figures that are small on screen: `--readability` (0 to 1, or `auto`) lifts the
   darkest tones and spreads the light ones; `auto` takes as much as the figure allows before
   lit skin or a light plate washes out. `--glow COLOUR` (repeatable) finds that colour's
   accents, widens them by `--glow-width` texels and makes them glow; `--glow-max-width` keeps
   it to thin lines, so a plate of the same colour stays as painted.
4. Real height (`--height`, metres) with the feet on the floor, centred under the hips.
5. Engine bone names (`--style unity|mixamo|keep`, the names of `unity_humanoid_names.py`);
   `--fingers merge` folds the finger bones into the hands, for figures too small for fingers.
6. Skin weights: at most `--max-influences` bones per vertex, smoothed a little; small separate
   pieces (a pouch, a lamp) move as one.

The numbers (triangles, atlas use, bones, weights, height) go to `<output>.report.json`, and the
licence record travels on as `<output>.provenance.json`.

    python scripts/game_ready.py output/animate/<run>/<name>_idle.glb game.glb --height 1.8
    python scripts/game_ready.py rigged.glb game.glb --height 1.8 --readability 1 --glow "#29e6ff" --fingers merge
    python scripts/game_ready.py rigged.glb game.glb --readability auto --glow "#d91f1a" --glow-max-width 4

Needs Blender 4.2 or newer; `blender_game_ready.py` does the Blender part. Everything here
except `main` is plain maths that the Blender part imports, so it is tested without Blender.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
STYLES = ("unity", "mixamo", "keep")
CORE_BONES = ("Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
              "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
              "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
              "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
              "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes")
HEX = re.compile(r"#?([0-9a-fA-F]{6})")

# Values from the hand-made finish this generalises (an 8k figure seen at 100-150 px), each one
# a setting, none tied to a figure.
FIXED = {
    "bake_scale": 2,          # bake at twice the atlas size, then box-filter down
    "uv_angle": 72.0,         # smart-project angle limit, degrees
    "uv_margin_px": 2,        # island spacing in texels: enough for close-ups; from afar mip-maps blend anyway
    "sharp_deg": 62.0,        # edges steeper than this are always hard ...
    "sharp_seam_deg": 25.0,   # ... and on a UV seam already above this (the vertex is split there anyway)
    "tiny_area_cm2": 1.5,     # floating bits smaller than this are deleted ...
    "tiny_faces": 3,          # ... and so are bits of this many faces or fewer
    "shade_lo": 0.78,         # sky light baked into the colour: shadowed parts x0.78 ...
    "shade_hi": 1.12,         # ... parts open to the sky x1.12
    "fill_px": 8,             # colours pushed this far beyond the islands (for mip-maps)
    "smooth_factor": 0.5,     # weight smoothing step
    "rigid_share": 0.145,     # a separate piece smaller than this share of the height moves as one
}


# --- settings -------------------------------------------------------------------------------

def parse_colour(text: str) -> tuple[float, float, float]:
    """'#29e6ff' or '29e6ff' -> sRGB floats 0..1."""
    match = HEX.fullmatch(text.strip())
    if not match:
        raise ValueError(f"not a colour like #29e6ff: {text!r}")
    value = match.group(1)
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _fraction(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def _positive(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be above 0")
    return value


def _atlas(text: str) -> int:
    value = int(text)
    if value < 128 or value > 4096 or value & (value - 1):
        raise argparse.ArgumentTypeError("a power of two from 128 to 4096")
    return value


def _readability(text: str) -> float | str:
    return "auto" if text.strip().lower() == "auto" else _fraction(text)


def _width(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 8:
        raise argparse.ArgumentTypeError("from 0 to 8 texels")
    return value


def _influences(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 8:
        raise argparse.ArgumentTypeError("from 1 to 8")
    return value


def add_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--height", type=_positive, default=1.8, help="height in metres (default 1.8)")
    parser.add_argument("--atlas", type=_atlas, default=1024, help="texture size (default 1024)")
    parser.add_argument("--readability", type=_readability, default=0.0,
                        help="0 to 1: lift the darkest tones, spread the light ones (default 0, off); auto: as much "
                             "as the figure takes before its light parts wash out")
    parser.add_argument("--glow", type=parse_colour, action="append", default=[], metavar="COLOUR",
                        help="make this colour's accents glow, e.g. '#29e6ff' (repeatable)")
    parser.add_argument("--glow-strength", type=_positive, default=3.0, help="emission strength (default 3)")
    parser.add_argument("--glow-width", type=_width, default=2,
                        help="texels each accent is widened by at 1024, so thin lines still read when small "
                             "(default 2; 0 keeps them as painted)")
    parser.add_argument("--glow-max-width", type=_width, default=0,
                        help="texels at 1024: only accents up to this wide glow, so a light line glows and an "
                             "armour plate of the same colour does not (default 0, no limit)")
    parser.add_argument("--glow-not-on-head", action="store_true",
                        help="no glow on the head and what hangs from it (paint on a face stays paint; "
                             "leave it off for a glowing visor)")
    parser.add_argument("--no-shade", action="store_true",
                        help="leave the soft sky light out of the colour (it keeps the normal map's detail)")
    parser.add_argument("--fingers", choices=("keep", "merge"), default="keep",
                        help="merge: fold finger bones into the hands (default keep)")
    parser.add_argument("--style", choices=STYLES, default="unity", help="bone names (default unity)")
    parser.add_argument("--max-influences", type=_influences, default=4, help="bones per vertex (default 4)")
    parser.add_argument("--smooth", type=int, default=2, help="weight smoothing passes (default 2)")
    parser.add_argument("--roughness", type=_fraction, default=0.8, help="material roughness (default 0.8)")


def _to_settings(ns: argparse.Namespace) -> dict:
    s = dict(FIXED)
    s.update(height=ns.height, atlas=ns.atlas, readability=ns.readability, glow=list(ns.glow),
             glow_strength=ns.glow_strength, glow_grow=ns.glow_width, glow_max_width=ns.glow_max_width,
             glow_not_on_head=ns.glow_not_on_head,
             shade=not ns.no_shade, fingers=ns.fingers, style=ns.style, max_influences=ns.max_influences,
             smooth_iters=max(0, ns.smooth), roughness=ns.roughness)
    if not s["shade"]:
        s["shade_lo"] = s["shade_hi"] = 1.0
    return s


def uv_margin(s: dict) -> float:
    """The island spacing in UV units: `uv_margin_px` texels at the chosen atlas size."""
    return s["uv_margin_px"] / s["atlas"]


def settings(argv: list[str]) -> dict:
    """The options alone (no file names), as the dict `blender_game_ready.py` reads."""
    parser = argparse.ArgumentParser(prog="game_ready.py")
    add_options(parser)
    return _to_settings(parser.parse_args(argv))


# --- colour -----------------------------------------------------------------------------------

def to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def smoothstep(e0: float, e1: float, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def curve_params(t: float) -> tuple[float, float, float, float]:
    """(lift, pivot gain, contrast below the median, contrast above) at strength `t`; 0 is no change."""
    return 0.02 * t, 1.0 + 0.06 * t, 1.0 - 0.10 * t, 1.0 + 0.45 * t


def value_curve(srgb, grey, t: float):
    """Lift the darkest values and push the lighter parts apart, around the median of `grey`."""
    lift, gain, lo, hi = curve_params(t)
    v = lift + (1.0 - lift) * np.asarray(srgb, float)
    if t == 0 or not np.any(grey):
        return np.clip(v, 0.0, 1.0)
    pivot = float(np.median(v.mean(-1)[grey]))
    d = v - pivot
    return np.clip(pivot * gain + np.where(d < 0, d * lo, d * hi), 0.0, 1.0)


READABILITY_CEILING = 200 / 255  # "auto" never lifts the brightest 1 % of the figure above this


def auto_readability(srgb, grey, ceiling: float = READABILITY_CEILING, step: float = 0.05) -> float:
    """The strongest readability that keeps the brightest 1 % of `grey` under `ceiling`.

    A dark figure takes all of it; lit skin or a light chest plate would wash out towards white,
    so a figure with bright parts gets less, and one already brighter than the ceiling none.
    """
    values = np.asarray(srgb, float)[np.asarray(grey)]
    if len(values) == 0:
        return 0.0
    if len(values) > 200_000:  # a sample tells the percentile just as well
        values = values[:: len(values) // 200_000]
    pixels = values[:, None, :]
    every = np.ones(pixels.shape[:2], bool)
    limit = max(ceiling, float(np.percentile(values.mean(-1), 99)))
    for t in np.round(np.arange(1.0, -step / 2, -step), 4):
        if np.percentile(value_curve(pixels, every, float(t)).mean(-1), 99) <= limit + 1e-9:
            return float(t)
    return 0.0


def accent_mask(srgb, colour, lo: float = 0.05, hi: float = 0.20, max_angle: float = 35.0):
    """0..1 per texel: how clearly it carries `colour`'s hue (its chroma, along the same direction)."""
    g = np.asarray(colour, float)
    gc = g - g.mean()
    norm = float(np.linalg.norm(gc))
    if norm < 1e-6:
        raise ValueError("a grey has no hue to look for")
    u = gc / norm
    c = np.asarray(srgb, float) - np.asarray(srgb, float).mean(-1, keepdims=True)
    score = (c * u).sum(-1)
    cos = score / np.maximum(np.linalg.norm(c, axis=-1), 1e-9)
    edge = math.cos(math.radians(max_angle))
    return smoothstep(lo, hi, score) * smoothstep(math.cos(math.radians(max_angle + 10)), edge, cos)


# --- images -----------------------------------------------------------------------------------

def downsample(a, k: int):
    """Box filter by `k` in both directions; `a` is (H, W, C)."""
    h, w, c = a.shape
    return a.reshape(h // k, k, w // k, k, c).mean(axis=(1, 3))


def _shift(a, dy: int, dx: int):
    """`a` moved by (dy, dx) with zeros coming in; nothing wraps round the edges."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def box_blur(a, r: int):
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax, dtype=np.float64)
        n = a.shape[ax]
        a = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax) - np.take(c, np.arange(0, n), axis=ax)) / (2 * r + 1)
    return a


def dilate(a, r: int):
    out = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx or dy) and dx * dx + dy * dy <= r * r + 1:
                out = np.maximum(out, _shift(a, dy, dx))
    return out


def push_out(rgb, have, iters: int):
    """Grow the colours into empty texels (mean of the filled neighbours), `iters` texels far."""
    rgb = np.asarray(rgb, float) * np.asarray(have)[..., None]
    have = np.asarray(have).astype(float)
    for _ in range(iters):
        acc = np.zeros_like(rgb)
        n = np.zeros_like(have)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    acc += _shift(rgb, dy, dx)
                    n += _shift(have, dy, dx)
        new = (have == 0) & (n > 0)
        rgb[new] = acc[new] / n[new][:, None]
        have[new] = 1.0
    return rgb, have > 0


def thin_only(mask, max_width: float):
    """`mask` without its parts wider than `max_width` texels (0 keeps everything).

    A part that survives shrinking by half the width is wide: a plate, not a line. It is dropped
    with a texel to spare, so no glowing rim is left round it.
    """
    if max_width <= 0:
        return mask
    r = max(1, int(round(max_width / 2)))
    core = (mask > 0.5).astype(float)
    shrunk = 1.0 - dilate(1.0 - core, r)
    return mask * (1.0 - dilate(shrunk, r + 1))


def compose(alb_lin, lit, cov, s: dict, exclude=None):
    """The atlas colours from the bakes: (albedo sRGB, emission sRGB or None, numbers).

    `alb_lin` is the painted colour (linear), `lit` the sky light on the normal-mapped surface,
    `cov` the share of each texel inside an island, all at atlas size. `exclude` (optional, 0..1)
    marks where nothing may glow, e.g. the head baked from its skin weights.
    """
    inside = cov > 0
    a_s = to_srgb(alb_lin)
    scale = s["atlas"] / 1024  # widths are given in texels at 1024
    keep = 1.0 if exclude is None else 1.0 - smoothstep(0.2, 0.5, exclude)
    masks = [thin_only(accent_mask(a_s, c) * inside, s["glow_max_width"] * scale) * keep for c in s["glow"]]
    found = np.max(masks, axis=0) if masks else np.zeros(inside.shape)
    grey = inside & (found < 0.1)
    lum = a_s.mean(-1)
    stats = {"island_share": float(inside.mean()),
             "lightness_before": _percentiles(lum[grey]),
             "accent_share": float((found[inside] > 0.5).mean()) if inside.any() else 0.0}
    t = auto_readability(a_s, grey) if s["readability"] == "auto" else float(s["readability"])
    stats["readability"] = t
    v = value_curve(a_s, grey, t)
    if s["shade_lo"] != 1.0 or s["shade_hi"] != 1.0:
        ref = float(np.percentile(lit[inside], 97)) if inside.any() else 1.0
        filled, _ = push_out(lit[..., None], inside, 2)
        shade = s["shade_lo"] + (s["shade_hi"] - s["shade_lo"]) * np.clip(box_blur(filled[..., 0], 1) / max(ref, 1e-6), 0, 1)
        v = np.clip(v * shade[..., None], 0.0, 1.0)
    emission = None
    if masks:
        emission = np.zeros_like(v)
        grow = int(round(s["glow_grow"] * scale))
        for colour, m in zip(s["glow"], masks):
            col = np.asarray(colour, float)
            grown = np.clip(np.maximum(box_blur(dilate(m, grow), 1), m), 0.0, 1.0) * keep
            halo = box_blur(dilate(m, grow + 2), 2) * 0.20 * keep  # widening never crosses onto the head
            v = v * (1 - grown[..., None]) + col * grown[..., None]
            v = np.clip(v + col * (halo * (1 - grown))[..., None] * 0.35, 0.0, 1.0)
            emission = np.maximum(emission, col * np.clip(grown + halo * (1 - grown), 0.0, 1.0)[..., None])
        stats["accent_share_widened"] = float((emission.max(-1)[inside] > 0.5).mean())
    stats["lightness_after"] = _percentiles(v.mean(-1)[grey])
    mean = v[inside].mean(0) if inside.any() else np.full(3, 0.5)
    v, have = push_out(v, inside, s["fill_px"])
    v[~have] = mean
    if emission is not None:
        emission, have = push_out(emission, inside, s["fill_px"])
        emission[~have] = 0.0
    return v, emission, stats


def _percentiles(values) -> list[float]:
    """sRGB 0..255 at the 1/5/25/50/75/95/99th percentiles: how dark and how spread a figure is."""
    if len(values) == 0:
        return []
    return [round(float(p), 1) for p in np.percentile(np.asarray(values) * 255, [1, 5, 25, 50, 75, 95, 99])]


def write_png(path, rgb) -> None:
    """8-bit RGB PNG from float sRGB (H, W, 3) whose row 0 is the bottom, as Blender stores images."""
    a = (np.clip(np.asarray(rgb)[::-1], 0, 1) * 255.0 + 0.5).astype(np.uint8)
    h, w, _ = a.shape
    f = a.astype(np.int16)
    f[:, 1:, :] -= a[:, :-1, :].astype(np.int16)  # PNG filter 1 (Sub)
    f = (f & 0xFF).astype(np.uint8)
    raw = b"".join(b"\x01" + f[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# --- bones and weights ----------------------------------------------------------------------

def bone_plan(parents: dict[str, str | None], unity_names: dict[str, str], fingers: str) -> dict[str, str]:
    """{bone: bone it merges into}. With fingers merged, everything below each hand joins the hand."""
    if fingers == "keep":
        return {}
    children: dict[str, list[str]] = defaultdict(list)
    for bone, parent in parents.items():
        if parent is not None:
            children[parent].append(bone)
    plan = {}
    for hand in (b for b, n in unity_names.items() if n in ("LeftHand", "RightHand")):
        stack = list(children[hand])
        while stack:
            bone = stack.pop()
            plan[bone] = hand
            stack.extend(children[bone])
    return plan


def missing_core_bones(unity_names: dict[str, str]) -> list[str]:
    """Unity Humanoid core slots no bone was found for, in skeleton order."""
    found = set(unity_names.values())
    return [b for b in CORE_BONES if b not in found]


def head_bones(parents: dict[str, str | None], unity_names: dict[str, str]) -> list[str]:
    """The head bone and every bone below it (a hood, a helmet lamp, hair)."""
    head = next((b for b, n in unity_names.items() if n == "Head"), None)
    if head is None:
        return []
    out, stack = [], [head]
    while stack:
        bone = stack.pop()
        out.append(bone)
        stack.extend(b for b, p in parents.items() if p == bone)
    return out


def merge_columns(W, columns: list[str], plan: dict[str, str]):
    """Weights of merged bones added into the bone they merge into; (weights, column names)."""
    keep = [c for c in columns if c not in plan]
    for target in dict.fromkeys(plan.values()):
        if target not in keep:
            keep.append(target)
    index = {c: i for i, c in enumerate(keep)}
    out = np.zeros((W.shape[0], len(keep)))
    for j, c in enumerate(columns):
        out[:, index[plan.get(c, c)]] += W[:, j]
    return out, keep


def limit_influences(W, k: int, floor: float = 0.01):
    """At most `k` bones per vertex, none below `floor`, renormalised; an unweighted vertex stays so."""
    W = np.array(W, float)
    if W.shape[1] > k:
        drop = np.argsort(W, axis=1)[:, :-k]
        np.put_along_axis(W, drop, 0.0, axis=1)
    W[W < floor] = 0.0
    total = W.sum(1, keepdims=True)
    return np.where(total > 0, W / np.maximum(total, 1e-12), 0.0)


def smooth_weights(W, edges, iters: int, lam: float):
    """Each pass moves every vertex's weights `lam` of the way to its neighbours' mean."""
    W = np.array(W, float)
    if iters <= 0 or len(edges) == 0:
        return W
    edges = np.asarray(edges)
    deg = np.bincount(edges.ravel(), minlength=len(W)).astype(float)
    for _ in range(iters):
        acc = np.zeros_like(W)
        np.add.at(acc, edges[:, 0], W[edges[:, 1]])
        np.add.at(acc, edges[:, 1], W[edges[:, 0]])
        avg = np.where(deg[:, None] > 0, acc / np.maximum(deg, 1)[:, None], W)
        W = (1 - lam) * W + lam * avg
    return W


def shells_from_edges(n: int, edges) -> list[np.ndarray]:
    """The connected vertex sets of a mesh with `n` vertices."""
    root = np.arange(n)

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for a, b in np.asarray(edges).reshape(-1, 2):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            root[ra] = rb
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [np.array(g) for g in groups.values()]


def rigid_shells(W, shells, co, max_diag: float, max_share: float = 0.5):
    """Separate pieces smaller than `max_diag` (and under half the mesh) take one weight row."""
    out = np.array(W, float)
    count = 0
    for s in shells:
        if len(s) >= max_share * len(W):
            continue
        if float(np.linalg.norm(co[s].max(0) - co[s].min(0))) < max_diag:
            out[s] = out[s].mean(0)
            count += 1
    return out, count


# --- measuring ------------------------------------------------------------------------------

def uv_area(uv, tris) -> float:
    """Share of the unit UV square the triangles cover (overlaps count twice)."""
    a, b, c = (np.asarray(uv)[np.asarray(tris)[:, i]] for i in range(3))
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return float(np.abs(cross).sum() / 2)


def engine_vertices(vertex, normal, uv) -> int:
    """Vertices a game engine uploads: one per distinct position, normal and UV."""
    keys = np.column_stack([np.asarray(vertex, float), np.round(normal, 3), np.round(uv, 5)])
    return len(np.unique(keys, axis=0))


def problems(report: dict, max_influences: int, height: float) -> list[str]:
    """What a game engine would still trip over, in plain words; empty when there is nothing."""
    out = []
    if report["materials"] != 1:
        out.append(f"{report['materials']} materials: an engine draws each one separately")
    if report["max_influences"] > max_influences:
        out.append(f"up to {report['max_influences']} bone influences per vertex (limit {max_influences})")
    if report["unweighted"]:
        out.append(f"{report['unweighted']} unweighted vertices stay behind when the figure moves")
    if abs(report["height"] - height) > 0.01:
        out.append(f"height {report['height']:.3f} m, asked for {height:.3f} m")
    if abs(report["min_z"]) > 0.005:
        out.append(f"lowest point {report['min_z']:.3f} m from the floor")
    for bone in report.get("missing_core_bones", []):
        out.append(f"no {bone} bone: Unity's Humanoid can do without one, but some importers expect it")
    error = report.get("animation_error_m")
    if error is not None and error > 0.002:
        out.append(f"moves changed by up to {error:.4f} m in the rescale")
    return out


def blender_command(blender: Path, job: Path) -> list[str]:
    """Headless Blender running the Blender part on a job file; it fails loudly, never silently."""
    return [str(blender), "--background", "--python-exit-code", "1",
            "--python", str(SCRIPTS / "blender_game_ready.py"), "--", str(job)]


# --- the command ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:  # pragma: no cover - runs Blender; its parts are tested
    import subprocess
    import tempfile

    sys.path.insert(0, str(SCRIPTS.parent))
    from kimodo_retarget import map_skintokens_to_soma
    from rig_check import Y_UP, skeleton
    from unity_humanoid_names import engine_names, rename

    from image_to_3dlab.blender import find_blender, missing_help
    from image_to_3dlab.provenance import set_glb_generator, write_derived_sidecar

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="rigged (or animated) GLB from the auto-rig")
    parser.add_argument("output", type=Path, help="game-ready GLB to write")
    parser.add_argument("--blender", type=Path, help="Blender to use (default: I2L_BLENDER, then the usual places)")
    add_options(parser)
    args = parser.parse_args(argv)
    s = _to_settings(args)
    blender = args.blender or find_blender()
    if not blender:
        print(missing_help())
        return 1
    if args.output.resolve() == args.source.resolve():
        parser.error("write to a new file; the source stays as it is")

    parents, heads = skeleton(args.source)
    mapping, _ = map_skintokens_to_soma(parents, heads, up=Y_UP)
    unity = engine_names(parents, mapping, "unity")
    if "Hips" not in unity.values():
        print("no humanoid hips found: this needs a rigged humanoid from the auto-rig")
        return 1
    plan = bone_plan(parents, unity, s["fingers"])
    names = {} if s["style"] == "keep" else engine_names(parents, mapping, s["style"])
    names = {b: n for b, n in names.items() if b not in plan}
    hips = next(b for b, n in unity.items() if n == "Hips")
    heads_off = head_bones(parents, unity) if s["glow_not_on_head"] else []

    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / "job.json"
        raw = Path(tmp) / "raw.glb"
        report_path = Path(tmp) / "report.json"
        job.write_text(json.dumps({"source": str(args.source.resolve()), "output": str(raw), "report": str(report_path),
                                   "work": tmp, "name": args.output.stem, "hips": hips, "merge": plan, "head_bones": heads_off,
                                   "settings": s}, indent=1))
        run = subprocess.run(blender_command(Path(blender), job), capture_output=True, text=True, check=False)
        if run.returncode != 0 or not raw.is_file():
            print(run.stdout[-4000:])
            print(run.stderr[-4000:])
            print("Blender stopped; nothing written")
            return 1
        report = json.loads(report_path.read_text())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rename(raw.read_bytes(), names))
    set_glb_generator(args.output)
    write_derived_sidecar(args.source, args.output, operation="game_ready")
    report["bones_named"] = len(names)
    report["bones_merged"] = len(plan)
    report["missing_core_bones"] = missing_core_bones(unity)
    report["style"] = s["style"]
    report["problems"] = problems(report, s["max_influences"], s["height"])
    report["settings"] = {k: v for k, v in s.items() if k != "glow"} | {
        "glow": ["#" + "".join(f"{round(c * 255):02x}" for c in g) for g in s["glow"]]}
    out_report = args.output.with_suffix(".report.json")
    out_report.write_text(json.dumps(report, indent=1))
    print(f"{report['triangles']} triangles, {report['materials']} material, atlas {s['atlas']} "
          f"({report['atlas_used'] * 100:.0f} % used, {report['uv_islands']} islands), "
          f"{report['bones']} bones ({len(names)} named, {len(plan)} merged), "
          f"height {report['height']:.2f} m, {len(report['clips'])} move(s)")
    for line in report["problems"] or ["ready: nothing an engine would trip over"]:
        print("  " + line)
    print(f"wrote {args.output} and {out_report.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
