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
   (`--no-shade` leaves it out), so the model needs no normal map. A close-up version (a menu, a
   hero stage) can have one anyway: `--normal-from` bakes it from the finer raw model, and
   `--soft-edges` keeps the triangles from showing.
3. Optional, for figures that are small on screen: `--readability` (0 to 1, or `auto`) lifts the
   darkest tones and spreads the light ones; `auto` takes as much as the figure allows before
   lit skin or a light plate washes out. `--glow COLOUR` (repeatable) finds that colour's
   accents, widens them by `--glow-width` texels and makes them glow; `--glow-max-width` keeps
   it to thin lines, so a plate of the same colour stays as painted.
4. Real height (`--height`, metres) with the feet on the floor, centred under the hips. With
   `--feet-from` (the Finish run the figure was made in) a sole the generator made up below the
   pictures' feet goes first: the outline check (`scripts/outline_check.py`) finds it, a sole
   reaching 4 % of the height or more below them is cut off (a second shoe glued under the real
   one), a lower one is pressed flat to 2.5 cm (a heel too high), and the figure is stood on its
   new soles at its height (`--feet cut|squash|off` decides by hand).
5. Engine bone names (`--style unity|mixamo|keep`, the names of `unity_humanoid_names.py`);
   `--fingers merge` folds the finger bones into the hands, for figures too small for fingers.
6. Skin weights: at most `--max-influences` bones per vertex, smoothed a little; small separate
   pieces (a pouch, a lamp) move as one.

Last, the finished figure is compared by colour with the model its colour came from (`--colour-from`,
else the source), from eight sides, leaving out where it glows on purpose: a patch that differs is a
bake fault, named in the report and marked in `<output>.colour.png` (`--no-colour-check` skips it).

The numbers (triangles, texels per centimetre and the share of the atlas painted twice, bones,
weights, height, the colour check) go to `<output>.report.json`, and the licence record travels on
as `<output>.provenance.json`.

    python scripts/game_ready.py output/animate/<run>/<name>_idle.glb game.glb --height 1.8
    python scripts/game_ready.py rigged.glb game.glb --height 1.8 --readability 1 --glow "#29e6ff" --fingers merge
    python scripts/game_ready.py rigged.glb game.glb --readability auto --glow "#d91f1a" --glow-max-width 4
    python scripts/game_ready.py rigged.glb game.glb --height 1.85 --feet-from output/finish/<run>

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
FEET_BONES = ("LeftFoot", "RightFoot", "LeftToes", "RightToes")  # where --feet-from looks for plates a cut left
CORE_BONES = ("Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
              "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
              "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
              "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
              "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes")
# The slots Unity's Humanoid avatar cannot map without (the Rig step's `bone_names.REQUIRED`); a rig missing one
# bends wrongly in every move, so the step stops instead of baking it. Neck, chests, shoulders, toes are optional.
REQUIRED_BONES = ("Hips", "Spine", "Head",
                  "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm", "RightLowerArm", "RightHand",
                  "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg", "RightLowerLeg", "RightFoot")
HEX =re.compile(r"#?([0-9a-fA-F]{6})")

# Values from the hand-made finish this generalises (an 8k figure seen at 100-150 px), each one
# a setting, none tied to a figure.
FIXED = {
    "bake_scale": 2,          # bake at twice the atlas size, then box-filter down
    "uv_angle": 72.0,         # smart-project angle limit, degrees
    "uv_margin_px": 2,        # island spacing in texels: enough for close-ups; from afar mip-maps blend anyway
    "sharp_deg": 62.0,        # edges steeper than this are always hard ...
    "sharp_seam_deg": 25.0,   # ... and on a UV seam above this with --hard-seams (soft by default: D145, a seam above
                              # 25 degrees made every triangle of a thinned face show)
    "tiny_area_cm2": 1.5,     # floating bits smaller than this are deleted ...
    "tiny_faces": 3,          # ... and so are bits of this many faces or fewer
    "shade_lo": 0.78,         # sky light baked into the colour: shadowed parts x0.78 ...
    "shade_hi": 1.12,         # ... parts open to the sky x1.12
    "fill_px": 8,             # colours pushed this far beyond the islands (for mip-maps)
    "smooth_factor": 0.5,     # weight smoothing step
    "rigid_share": 0.145,     # a separate piece smaller than this share of the height moves as one
    "transfer_cage_m": 0.012,  # --colour-from: rays start this far outside the surface (metres, after the rescale) ...
    "transfer_reach_m": 0.03,  # ... and look this far for the finer model; a miss keeps the model's own paint
    "hidden_rays": 16,        # --drop-hidden: a triangle no ray of these reaches the outside from is hidden ...
    "hidden_cells": 400,      # ... and goes if the air beside it is a sealed hollow, on a grid this many cells across
    "hidden_close": 2,        # ... closed this many cells, so a slit in the wall does not open the hollow
    "hidden_corner_reach": 0.66,  # a face is asked at its centre and this far toward each corner
    "poke_rays": 128,         # --drop-hidden also drops inner wall poking past the outer surface: seen from behind ...
    "poke_near": 0.02,        # ... (rays of these), its front facing another face's front within this share of the size
    "roof_near": 0.05,        # the sky-light bake looks through a piece over the surface this close that faces it
    "drop_tries": 3,          # thin-and-drop tries per figure, each aiming the thinning by what the last drop left ...
    "drop_slack": 0.03,       # ... until the figure lands within this share of --decimate-to
    "pose_check_res": 192,    # --drop-hidden's pose check: rays per side of each view ...
    "pose_check_gap": 0.01,   # ... and a dropped face counts as a hole when nothing lies within this share of the size behind it
    "feet_cut_at": 0.04,      # --feet-from: a sole reaching this share of the height below the pictures' feet is cut off ...
    "feet_squash_at": 0.02,   # ... one reaching this far is pressed flat ...
    "feet_sole_m": 0.025,     # ... to a sole this thick (metres); hand fixes of five figures set these (D205 of the
                              # AssetFurnace Support notes: cut at 6-7 %, squashed at 3.5 %)
    "feet_floor": 0.01,       # an outline part counts as a sole when it starts this close to the floor ...
    "feet_top": 0.15,         # ... and stays below this share of the height (higher, it is a robe or a cape)
    "feet_splinter_reach": 0.02,   # --feet auto presses flat instead of cutting when the cut leaves thin plates reaching
    "feet_splinter_area": 0.002,   # this share of the height past the boot, over this share of the height squared;
                                   # four cut figures set these (two clean cuts reached 1.2 and 1.7 %, two with blades
                                   # 2.4 and 2.8 %; D207 of the AssetFurnace Support notes)
    "glow_paint_bright_share": 0.25,  # a glow hue whose bright group is at most this share of its texels ...
    "glow_paint_gap": 0.30,           # ... and this much brighter than the rest is mostly paint: only the bright
                                      # group glows (the Purpur-Samurai's crimson plates: 20 %, 0.42; 25 figures
                                      # measured, four such, D207)
}

# --drop-hidden thins to this many times the budget first: the Wachter, thinned to 22,000, kept 15,275 after the
# drop (D190 of the AssetFurnace Support notes); each figure's next try is aimed by what its own drop left.
FIRST_GROWTH = 22000 / 14000
# A triangle the A-pose hides can show in motion (an inner thigh in a stride, a cape's lining in a crouch), so it
# goes only if it stays hidden in each of these poses too: (Unity bone, world axis, degrees).
VETO_POSES = {
    "stride": [("Spine", "X", 10), ("LeftUpperLeg", "X", -60), ("LeftLowerLeg", "X", 60), ("RightUpperLeg", "X", 45),
               ("RightLowerLeg", "X", 30)],
    "crouch": [("Spine", "X", 30), ("LeftUpperLeg", "X", -100), ("RightUpperLeg", "X", -100),
               ("LeftLowerLeg", "X", 125), ("RightLowerLeg", "X", 125)],
    "straddle": [("LeftUpperLeg", "Y", -40), ("RightUpperLeg", "Y", 40), ("LeftUpperArm", "X", -40),
                 ("RightUpperArm", "X", 40)],
    "arms_up": [("LeftUpperArm", "Y", -130), ("RightUpperArm", "Y", 130), ("Spine", "X", -15)],
    "twist": [("UpperChest", "Z", -40), ("Spine", "Z", -15), ("LeftUpperLeg", "X", 35), ("RightUpperLeg", "X", -40),
              ("RightLowerLeg", "X", 50)],
}
# After the drop the figure is checked in other moves than the veto's, as a game shows it (run, dodge, fall, reach),
# from twelve sides: a dropped face seen there is a hole, and it is put back (#153 of the AssetFurnace Support notes;
# its separate pose check found the holes of a looser drop rule that the colour check could not see).
CHECK_POSES = {
    "run": [("Spine", "X", 15), ("LeftUpperLeg", "X", -50), ("LeftLowerLeg", "X", 70), ("RightUpperLeg", "X", 35),
            ("RightLowerLeg", "X", 40), ("LeftUpperArm", "X", 40), ("LeftLowerArm", "X", -60),
            ("RightUpperArm", "X", -50), ("RightLowerArm", "X", -70)],
    "dodge": [("Spine", "X", 35), ("LeftUpperLeg", "X", -85), ("LeftLowerLeg", "X", 110), ("RightUpperLeg", "X", -70),
              ("RightLowerLeg", "X", 120), ("LeftUpperArm", "X", -70), ("RightUpperArm", "X", -70),
              ("LeftLowerArm", "X", -50), ("RightLowerArm", "X", -50)],
    "fall": [("Spine", "X", -30), ("LeftUpperArm", "Y", -110), ("RightUpperArm", "Y", 110), ("LeftUpperLeg", "X", -35),
             ("RightUpperLeg", "Y", 25), ("Head", "X", -25)],
    "reach": [("UpperChest", "Z", 35), ("LeftUpperArm", "X", -95), ("RightUpperArm", "X", -95),
              ("LeftUpperLeg", "Y", -25), ("RightUpperLeg", "Y", 25), ("LeftLowerLeg", "X", 20)],
}
# (azimuth, elevation) in degrees: eight round at 10 degrees up, four from 35 degrees below
CHECK_VIEWS = [(az, 10) for az in range(0, 360, 45)] + [(az, -35) for az in (0, 90, 180, 270)]
AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


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


def _triangles(text: str) -> int:
    value = int(text)
    if value != 0 and not 1000 <= value <= 200000:
        raise argparse.ArgumentTypeError("0 (keep) or from 1000 to 200000")
    return value


def _texel(text: str) -> float:
    value = float(text)
    if not 1.0 <= value <= 4.0:
        raise argparse.ArgumentTypeError("from 1 to 4")
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
    parser.add_argument("--glow-not-on-hands", action="store_true",
                        help="no glow on the hands and fingers (glowing hands read as too big at a game's camera; "
                             "leave it off for a glowing gauntlet or a spell in the hand)")
    parser.add_argument("--no-shade", action="store_true",
                        help="leave the soft sky light out of the colour (it keeps the normal map's detail)")
    parser.add_argument("--fingers", choices=("keep", "merge"), default="keep",
                        help="merge: fold finger bones into the hands (default keep)")
    parser.add_argument("--style", choices=STYLES, default="unity", help="bone names (default unity)")
    parser.add_argument("--max-influences", type=_influences, default=4, help="bones per vertex (default 4)")
    parser.add_argument("--smooth", type=int, default=2, help="weight smoothing passes (default 2)")
    parser.add_argument("--roughness", type=_fraction, default=0.8, help="material roughness (default 0.8)")
    parser.add_argument("--decimate-to", type=_triangles, default=0, metavar="N",
                        help="thin the mesh to about N triangles first (default 0, keep); rig a finer Finish "
                             "(say 40k) and thin it here, so --head-protect can keep the face")
    parser.add_argument("--head-protect", type=_fraction, default=0.0,
                        help="0 to 1: how strongly the head resists --decimate-to (1 = the body gives up the "
                             "triangles, the face keeps its nose, chin and eyes; default 0)")
    parser.add_argument("--head-texel", type=_texel, default=1.0,
                        help="1 to 4: the head's share of the atlas, times this (2 = a face twice as sharp, "
                             "the body a little softer; default 1)")
    parser.add_argument("--colour-from", type=Path, default=None, metavar="GLB",
                        help="bake the colour from this model of the same figure where it lies on the surface, "
                             "e.g. the raw model painted by Pixel Match, whose face keeps the picture's eyes; "
                             "same coordinates as the source (Finish and Rig keep them)")
    parser.add_argument("--mesh-from", type=Path, default=None, metavar="GLB",
                        help="use this finer Finish of the same figure as the mesh, its skin weights carried over "
                             "from the source's rig (so a figure rigged at 8k can keep a 40k face through "
                             "--decimate-to and --head-protect without rigging again)")
    parser.add_argument("--normal-from", type=Path, default=None, metavar="GLB",
                        help="bake a normal map from this finer model of the same figure (e.g. the raw model) onto "
                             "the atlas, so the engine lights the fine shape (nose, lips, folds) on the coarse mesh; "
                             "for close-up versions (a menu, a hero stage), costs a second texture")
    parser.add_argument("--soft-edges", action="store_true",
                        help="the default now (only edges steeper than 62 degrees are hard); kept for old command lines")
    parser.add_argument("--hard-seams", action="store_true",
                        help="the old shading: a UV seam above 25 degrees is hard too (crisper on blocky armour, but "
                             "every triangle of a thinned face shows up close)")
    parser.add_argument("--transfer-cage", type=float, default=FIXED["transfer_cage_m"], metavar="M",
                        help="--colour-from / --normal-from: rays start this far outside the surface, in metres "
                             f"(default {FIXED['transfer_cage_m']}); raise it when thinning sank a part (spiky hair) "
                             "deeper than that into the finer model and the rays pick up what lies beneath")
    parser.add_argument("--allow-missing-bones", action="store_true",
                        help="bake a rig that lacks a bone Unity's Humanoid needs (spine, shins, ...) instead of "
                             "stopping; the report still names it")
    parser.add_argument("--transfer-reach", type=float, default=FIXED["transfer_reach_m"], metavar="M",
                        help=f"... and look this far for the finer model (default {FIXED['transfer_reach_m']})")
    parser.add_argument("--drop-hidden", action="store_true",
                        help="drop the triangles nobody can see (the hidden inner wall a generated figure carries "
                             "inside its body), after the thinning and only those that stay hidden in a stride, a "
                             "crouch, a straddle, arms up and a twist too; the thinning aims higher first, so the "
                             "figure still lands on --decimate-to with more of it seen (needs --decimate-to)")
    parser.add_argument("--no-pose-check", action="store_true",
                        help="with --drop-hidden: skip the pose check (the figure before and after the drop posed "
                             "in a run, a dodge, a fall and a reach and seen from twelve sides; a dropped face seen "
                             "there is a hole and is put back; what is left is named in the report and marked in "
                             "<output>.pose.png)")
    parser.add_argument("--feet-from", type=Path, default=None, metavar="RUN",
                        help="the Finish run the figure was made in (or a model with its pictures, or the JSON of "
                             "scripts/outline_check.py): a sole the generator made up below the pictures' feet is "
                             "cut off or pressed flat")
    parser.add_argument("--feet", choices=("auto", "cut", "squash", "off"), default="auto",
                        help="with --feet-from: cut a sole 4 %% of the height or more below the pictures' feet, "
                             "press a lower one flat (auto, the default), always cut, always press, or leave them")
    parser.add_argument("--no-colour-check", action="store_true",
                        help="skip the closing colour check (the finished figure drawn from eight sides against the "
                             "model its colour came from, --colour-from or the source; patches that differ are "
                             "named in the report and marked in <output>.colour.png)")


def _to_settings(ns: argparse.Namespace) -> dict:
    s = dict(FIXED)
    s.update(height=ns.height, atlas=ns.atlas, readability=ns.readability, glow=list(ns.glow),
             glow_strength=ns.glow_strength, glow_grow=ns.glow_width, glow_max_width=ns.glow_max_width,
             glow_not_on_head=ns.glow_not_on_head, glow_not_on_hands=ns.glow_not_on_hands,
             shade=not ns.no_shade, fingers=ns.fingers, style=ns.style, max_influences=ns.max_influences,
             smooth_iters=max(0, ns.smooth), roughness=ns.roughness,
             decimate_to=ns.decimate_to, head_protect=ns.head_protect, head_texel=ns.head_texel,
             colour_from=str(ns.colour_from) if ns.colour_from else None,
             mesh_from=str(ns.mesh_from) if ns.mesh_from else None,
             normal_from=str(ns.normal_from) if ns.normal_from else None,
             transfer_cage_m=ns.transfer_cage, transfer_reach_m=ns.transfer_reach, drop_hidden=ns.drop_hidden,
             colour_check=not ns.no_colour_check, pose_check=not ns.no_pose_check,
             feet_from=str(ns.feet_from) if ns.feet_from else None, feet=ns.feet)
    if not ns.hard_seams:
        s["sharp_seam_deg"] = s["sharp_deg"]
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


def readability_pivot(srgb, grey, t: float) -> float | None:
    """The value the readability curve turns around: the median of `grey` after the lift (None at strength 0)."""
    lift = curve_params(t)[0]
    if t == 0 or not np.any(grey):
        return None
    v = lift + (1.0 - lift) * np.asarray(srgb, float)
    return float(np.median(v.mean(-1)[grey]))


def curve_at(srgb, t: float, pivot: float | None):
    """The readability curve of strength `t` turning around `pivot` (`readability_pivot`)."""
    lift, gain, lo, hi = curve_params(t)
    v = lift + (1.0 - lift) * np.asarray(srgb, float)
    if t == 0 or pivot is None:
        return np.clip(v, 0.0, 1.0)
    d = v - pivot
    return np.clip(pivot * gain + np.where(d < 0, d * lo, d * hi), 0.0, 1.0)


def value_curve(srgb, grey, t: float):
    """Lift the darkest values and push the lighter parts apart, around the median of `grey`."""
    return curve_at(srgb, t, readability_pivot(srgb, grey, t))


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


def brightness_split(values, bins: int = 64) -> dict:
    """Otsu's split of `values` (0..1) into a dark and a bright group: the threshold, how well it separates (the
    share of the variance between the groups), both groups' means and the bright group's share."""
    values = np.asarray(values, dtype=np.float64)
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    centres = (edges[:-1] + edges[1:]) / 2
    p = hist / max(hist.sum(), 1)
    w0, m0 = np.cumsum(p), np.cumsum(p * centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.nan_to_num((m0[-1] * w0 - m0) ** 2 / (w0 * (1.0 - w0)))
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))  # an empty gap gives a plateau: take its middle
    k = int(best[len(best) // 2])
    t = float(edges[k + 1])  # the dark group is bins 0..k
    total = float((p * (centres - m0[-1]) ** 2).sum())
    dark, bright = values[values < t], values[values >= t]
    return {"at": round(t, 3), "separation": round(float(between[k]) / max(total, 1e-12), 3),
            "dark": round(float(dark.mean()), 3) if len(dark) else 0.0,
            "bright": round(float(bright.mean()), 3) if len(bright) else 0.0,
            "bright_share": round(len(bright) / max(len(values), 1), 3)}


def glow_only_bright(mask, srgb, inside, s: dict | None = None):
    """`mask` (one glow colour's `accent_mask`) kept only where the texel is bright, when that hue is mostly paint:
    most texels of the hue clearly darker than its brightest group. A figure painted in the glow's own hue (dark
    crimson plates with bright red lines, D207 of the AssetFurnace Support notes) otherwise glowed on every thin
    painted edge and was painted over in the glow colour, a blotchy look. A hue that is mostly bright lines is left as
    it is. Returns (mask, the split's numbers or None)."""
    s = s or FIXED
    v = np.asarray(srgb, dtype=np.float64).max(-1)
    sel = np.asarray(inside, bool) & (np.asarray(mask) > 0.5)
    if sel.sum() < 200:
        return mask, None
    split = brightness_split(v[sel])
    split["used"] = bool(split["bright_share"] <= s["glow_paint_bright_share"]
                         and split["bright"] - split["dark"] >= s["glow_paint_gap"])
    if not split["used"]:
        return mask, split
    return mask * smoothstep(split["at"] - 0.05, split["at"] + 0.05, v), split


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
    masks, splits = [], {}
    for c in s["glow"]:
        # split only what would glow: the thin lines left after the wide parts (a face, a plate) and the head are out
        m, split = glow_only_bright(thin_only(accent_mask(a_s, c) * inside, s["glow_max_width"] * scale) * keep,
                                    a_s, inside, s)
        if split is not None:
            splits["#" + "".join(f"{round(x * 255):02x}" for x in c)] = split
        masks.append(m)
    found = np.max(masks, axis=0) if masks else np.zeros(inside.shape)
    grey = inside & (found < 0.1)
    lum = a_s.mean(-1)
    stats = {"island_share": float(inside.mean()),
             "lightness_before": _percentiles(lum[grey]),
             "accent_share": float((found[inside] > 0.5).mean()) if inside.any() else 0.0,
             "glow_split": splits}
    t = auto_readability(a_s, grey) if s["readability"] == "auto" else float(s["readability"])
    stats["readability"] = t
    stats["readability_pivot"] = readability_pivot(a_s, grey, t)
    v = curve_at(a_s, t, stats["readability_pivot"])
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

# --- the feet ---------------------------------------------------------------------------------

def feet_plan(parts: list[dict], mode: str = "auto", height: float = 1.8, s: dict | None = None) -> dict | None:
    """What to do with a sole the generator made up below the pictures' feet, from the outline check's parts
    (`image_to_3dlab.outline_check`, as plain dicts): the parts below the outline that start on the floor and stay at
    foot level are that sole, and the farthest one says how high it reaches (a share of the height). A second shoe
    glued under the real one reaches far (6-7 % on three figures) and is cut off; a heel a little too high (3.5 %) is
    pressed flat, which keeps the shoe's length. None when there is nothing to do."""
    s = s or FIXED
    soles = [p for p in parts if p["side"] == "below" and p["low"] <= s["feet_floor"] and p["high"] <= s["feet_top"]]
    if mode == "off" or not soles:
        return None
    reach = max(float(p["reach"]) for p in soles)
    auto = mode == "auto"
    if mode == "auto":
        mode = "cut" if reach >= s["feet_cut_at"] else "squash" if reach >= s["feet_squash_at"] else None
    sole = s["feet_sole_m"] / height if mode == "squash" else 0.0
    if mode is None or (mode == "squash" and reach <= sole):
        return None
    return {"mode": mode, "at": round(reach, 4), "sole": round(sole, 4), "parts": len(soles), "auto": auto}


def squash_instead(plan: dict, height: float = 1.8, s: dict | None = None) -> dict:
    """The same sole pressed flat instead of cut off: what `--feet auto` does when the cut left thin plates of the
    made-up shoe lying in front of the boot (`splintered`)."""
    s = s or FIXED
    return dict(plan, mode="squash", sole=round(s["feet_sole_m"] / height, 4), auto=False)


def splintered(found: dict, height: float = 1.8, s: dict | None = None) -> bool:
    """Whether a cut left plates of the made-up shoe lying out in front (`splinters`' result): enough thin sole that
    reaches far enough past the boot. A boot's own toe can leave a small thin tip (a pointed shoe); the Revolverheld's
    cut left blades several centimetres long (D206)."""
    s = s or FIXED
    return (found.get("reach_m", 0.0) >= s["feet_splinter_reach"] * height
            and found.get("thin_m2", 0.0) >= s["feet_splinter_area"] * height * height)


def feet_heights(z, plan: dict):
    """Heights (shares of the figure's height from its lowest point) after the feet are fixed, from the new lowest
    point: a cut lays what was below it on the floor and moves the rest down by the cut; a squash presses what was
    below into the sole and moves the rest down by the difference, so the shoe stays joined."""
    z = np.asarray(z, dtype=np.float64)
    at = plan["at"]
    if plan["mode"] == "cut":
        return np.maximum(z - at, 0.0)
    return np.where(z <= at, z * (plan["sole"] / at), z - (at - plan["sole"]))


def feet_drop(plan: dict) -> float:
    """How far everything above the fixed feet moves down, as a share of the height."""
    return plan["at"] - (plan["sole"] if plan["mode"] == "squash" else 0.0)


def feet_positions(positions, plan: dict, up: int = 1):
    """A model's points with the same feet (`feet_heights`), scaled back to its own height on its own floor; the
    colour check draws the source this way so it stands like the fixed figure."""
    out = np.array(positions, dtype=np.float64)
    bottom = float(out[:, up].min())
    size = max(float(out[:, up].max()) - bottom, 1e-12)
    out[:, up] = bottom + feet_heights((out[:, up] - bottom) / size, plan) * size / (1.0 - feet_drop(plan))
    return out


def splinters(vertices, faces, floor: float | None = None, height: float | None = None,
              thin: float = 0.015, feet=None, radius: float = 0.12) -> dict:
    """How much of a figure's soles is a thin plate: the soles are the faces facing the floor at the floor (`floor`,
    else the lowest point), with `feet` (the foot and toe bones' points, x and y) only those within `radius` (a share
    of the height) of one, so a robe or a cape lying on the floor is no sole; from each, straight up, the first
    surface it meets. A cut through a boot closes a hollow shell, so above its sole lies the boot's inside, decimetres
    high; a made-up shoe longer than the boot leaves the front of its toe as a flat plate a few millimetres thick
    lying out in front (the Revolverheld, D206). Returns the soles' area and the area under which another surface lies
    closer than `thin` (a share of the height), in square metres, that share, and how far the plates reach past the
    thick part of the soles (metres)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    z = v[:, 2]
    floor = float(z.min()) if floor is None else float(floor)
    height = float(z.max() - floor) if height is None else float(height)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    nz = cross[:, 2] / np.maximum(2.0 * area, 1e-18)
    sole = (nz < -0.9) & (tri[:, :, 2].max(axis=1) <= floor + 0.002 * height) & (area > 0)
    if feet is not None and len(feet):
        feet = np.asarray(feet, dtype=np.float64)[:, :2]
        c2 = tri[:, :, :2].mean(axis=1)
        near_foot = np.zeros(len(tri), dtype=bool)
        for q in feet:
            near_foot |= ((c2 - q) ** 2).sum(axis=1) <= (radius * height) ** 2
        sole &= near_foot
    if not sole.any():
        return {"sole_m2": 0.0, "thin_m2": 0.0, "thin_share": 0.0, "reach_m": 0.0}
    reach = floor + thin * height
    over = tri[~sole & (tri[:, :, 2].min(axis=1) < reach) & (area > 0)]
    points = tri[sole].mean(axis=1)
    hit = np.zeros(len(points), dtype=bool)
    if len(over):
        a, b, c = over[:, 0], over[:, 1], over[:, 2]
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        ok = np.abs(d) > 1e-18
        a, b, c, d = a[ok], b[ok], c[ok], d[ok]
        for start in range(0, len(points), 256):
            p = points[start:start + 256, None, :]
            w0 = ((b[:, 1] - c[:, 1]) * (p[..., 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (p[..., 1] - c[:, 1])) / d
            w1 = ((c[:, 1] - a[:, 1]) * (p[..., 0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (p[..., 1] - c[:, 1])) / d
            w2 = 1.0 - w0 - w1
            zz = w0 * a[:, 2] + w1 * b[:, 2] + w2 * c[:, 2]
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            hit[start:start + 256] = (inside & (zz > p[..., 2] + 1e-4 * height) & (zz < reach)).any(axis=1)
    sole_area = area[sole]
    thin_area = float(sole_area[hit].sum())
    # how far the plates reach past the thick part of the soles (the boot itself), area-weighted 90th percentile
    reach_m = 0.0
    if hit.any() and (~hit).any():
        thick = points[~hit][:, :2]
        dist = np.array([np.sqrt(((thick - q) ** 2).sum(axis=1)).min() for q in points[hit][:, :2]])
        order = np.argsort(dist)
        cum = np.cumsum(sole_area[hit][order])
        reach_m = float(dist[order][np.searchsorted(cum, 0.9 * cum[-1])])
    elif hit.any():
        reach_m = float(np.ptp(points[hit][:, :2], axis=0).max())
    return {"sole_m2": round(float(sole_area.sum()), 5), "thin_m2": round(thin_area, 5),
            "thin_share": round(thin_area / max(float(sole_area.sum()), 1e-12), 4), "reach_m": round(reach_m, 4)}


def _polygon_area(poly) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inside(points, poly):
    """Which of `points` (n, 2) lie inside the polygon (m, 2), by the even-odd rule."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (crosses & (x < at)).sum(axis=1) % 2 == 1


def nested_caps(polygons: list) -> list[bool]:
    """For the caps that close a cut (each an (n, 2) outline seen from below), which lie inside a larger one: the
    generator builds a second wall a few millimetres inside the outer one, and its cap would be a second sole inside
    the first; only the outer cap stays."""
    polygons = [np.asarray(p, dtype=np.float64) for p in polygons]
    order = sorted(range(len(polygons)), key=lambda i: -_polygon_area(polygons[i]))
    inner, kept = [False] * len(polygons), []
    for i in order:
        if any(_inside(polygons[i], polygons[k]).mean() >= 0.5 for k in kept):
            inner[i] = True
        else:
            kept.append(i)
    return inner


def finish_run_from_provenance(model: Path) -> Path | None:
    """The Finish run a rigged model was made from, read from its provenance sidecar (`<model>.provenance.json`,
    written by AssetFurnace beside every result): the first record in its chain of source records whose operation
    is "finish", the folder its output lies in. None without a sidecar, without a Finish in the chain, or when that
    folder is no longer there, so `--feet-from` need not be named for a figure rigged in the studio."""
    sidecar = Path(model).with_name(Path(model).stem + ".provenance.json")
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for _ in range(32):
        if not isinstance(record, dict):
            return None
        if record.get("operation") == "finish":
            out = (record.get("output") or {}).get("path")
            run = Path(out).parent if out else None
            return run if run is not None and run.is_dir() else None
        record = record.get("source_record")
    return None


def outline_parts(path: Path) -> dict:
    """The outline check's result for `--feet-from`: its JSON as written by `scripts/outline_check.py --json`, or
    the check run on a Finish run folder or a model with its pictures."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(SCRIPTS.parent))
    from image_to_3dlab import outline_check as oc

    model, views = oc.find_inputs(path)
    return oc.check(model, views, frame=oc.PIXAL3D_FRAME)


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


def rig_refusal(missing: list[str]) -> str | None:
    """Why this rig is not baked, or None. Only a missing required bone stops it; optional ones are a note."""
    required = [b for b in missing if b in REQUIRED_BONES]
    if not required:
        return None
    return (f"the rig has no {', '.join(required)} bone: Unity's Humanoid cannot map it and the figure bends "
            "wrongly in every move. Rig it again (another try usually finds them), or pass --allow-missing-bones "
            "to bake it anyway")


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


def hand_bones(parents: dict[str, str | None], unity_names: dict[str, str]) -> list[str]:
    """Both hand bones and every bone below them (the fingers, a held piece)."""
    out, stack = [], [b for b, n in unity_names.items() if n in ("LeftHand", "RightHand")]
    while stack:
        bone = stack.pop()
        out.append(bone)
        stack.extend(b for b, p in parents.items() if p == bone)
    return out


def glow_off_bones(parents: dict[str, str | None], unity_names: dict[str, str], s: dict) -> list[str]:
    """The bones whose skin keeps no glow: the head's with `--glow-not-on-head`, the hands' with
    `--glow-not-on-hands` (a figure whose glowing hands read as too big and odd, D980 of ArcherSoul's notes)."""
    out = head_bones(parents, unity_names) if s.get("glow_not_on_head") else []
    if s.get("glow_not_on_hands"):
        out += [b for b in hand_bones(parents, unity_names) if b not in out]
    return out


# --- the face ---------------------------------------------------------------------------------
# A whole figure thinned to 8k triangles gives each part a share by its size, and a face is small:
# its nose, chin and eye sockets melt into one rounded shape (Finish's 8k is where faces go flat).
# Rigging a finer Finish and thinning it here, with the head's vertices protected, keeps the face.

def decimate_ratio(before: int, target: int) -> float:
    """The collapse ratio that thins `before` triangles to `target`; never above 1 (nothing is added)."""
    if target <= 0 or before <= target:
        return 1.0
    return target / before


def drop_hidden_refusal(s: dict) -> str | None:
    """Why --drop-hidden cannot run with these settings, or None."""
    if s.get("drop_hidden") and not s.get("decimate_to"):
        return "--drop-hidden needs --decimate-to: it aims the thinning so the figure lands on that budget"
    return None


def first_target(target: int) -> int:
    """What --drop-hidden thins to before its first drop: above the budget, by what a figure's drop usually takes."""
    return round(target * FIRST_GROWTH)


def next_first_target(first: int, left: int, target: int) -> int:
    """The next try's thinning, aimed by what the last drop left: scaled so the drop lands on `target`, never
    below the budget itself (a figure with no inner wall) and at most four times it."""
    if left <= 0:
        return 4 * target
    return int(min(max(round(first * target / left), target), 4 * target))


def drop_settled(left: int, target: int, slack: float = FIXED["drop_slack"]) -> bool:
    """Whether a drop left the figure close enough to its budget."""
    return abs(left - target) <= slack * target


def veto_poses(unity_names: dict[str, str]) -> list[tuple[str, list[tuple[str, tuple[float, float, float], float]]]]:
    """VETO_POSES with the rig's own bone names (`unity_names`: rig bone -> Unity name); bones it lacks are left out."""
    bone_of = {unity: bone for bone, unity in unity_names.items()}
    return [(name, [(bone_of[unity], AXES[axis], float(degrees)) for unity, axis, degrees in moves if unity in bone_of])
            for name, moves in VETO_POSES.items()]


def check_poses(unity_names: dict[str, str]) -> list[tuple[str, list[tuple[str, tuple[float, float, float], float]]]]:
    """CHECK_POSES with the rig's own bone names, like `veto_poses`."""
    bone_of = {unity: bone for bone, unity in unity_names.items()}
    return [(name, [(bone_of[unity], AXES[axis], float(degrees)) for unity, axis, degrees in moves if unity in bone_of])
            for name, moves in CHECK_POSES.items()]


def view_rays(center, radius: float, azimuth: float, elevation: float, res: int):
    """An orthographic view of a sphere (`center`, `radius`) from `azimuth` (0: in front, at -Y; 90: at +X) and
    `elevation` degrees: (`res` x `res` ray origins outside the sphere, row by row from the top, and their direction)."""
    az, el = math.radians(azimuth), math.radians(elevation)
    eye = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    direction = -eye
    right = np.cross(direction, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    steps = (np.arange(res) - (res - 1) / 2) / (res / 2) * radius
    across, down = np.meshgrid(steps, steps[::-1])
    origins = (np.asarray(center, dtype=float) + eye * 2 * radius + across.reshape(-1, 1) * right
               + down.reshape(-1, 1) * up)
    return origins, direction


def pose_holes(before_face, before_depth, after_depth, doomed, gap: float):
    """Per ray: whether it shows a hole, i.e. it first meets a dropped face (`before_face`, -1 for none) and, with the
    dropped faces gone, nothing within `gap` behind it (`after_depth`, inf for nothing)."""
    before_face = np.asarray(before_face)
    seen = before_face >= 0
    dropped = np.zeros(len(before_face), dtype=bool)
    dropped[seen] = np.asarray(doomed)[before_face[seen]]
    return dropped & (np.asarray(after_depth) > np.asarray(before_depth) + gap)


def faces_seen_in(before_face, hole):
    """The dropped faces seen through the holes: those the repair puts back."""
    return np.unique(np.asarray(before_face)[np.asarray(hole)])


def head_share(W, columns: list[str], head: list[str]):
    """Per vertex, how much of it moves with the head (its weights on the head bones, at most 1)."""
    pick = [j for j, c in enumerate(columns) if c in set(head)]
    if not pick:
        return np.zeros(len(W))
    return np.minimum(np.asarray(W, float)[:, pick].sum(1), 1.0)


def head_islands(island, share, at_least: float = 0.5) -> set[int]:
    """The UV islands whose loops lie mostly on the head (mean head share at or above `at_least`)."""
    island = np.asarray(island)
    share = np.asarray(share, float)
    total = np.bincount(island, weights=share)
    count = np.bincount(island)
    return {int(i) for i in np.nonzero(count)[0] if total[i] / count[i] >= at_least}


def scale_islands(uv, island, chosen: set[int], factor: float):
    """Loop UVs with the `chosen` islands grown by `factor` around their own centres, before packing:
    the packer then gives them `factor` times the texels per metre of the rest."""
    out = np.array(uv, float)
    if factor == 1.0 or not chosen:
        return out
    island = np.asarray(island)
    for i in chosen:
        sel = island == i
        centre = out[sel].mean(0)
        out[sel] = centre + (out[sel] - centre) * factor
    return out


def blend_transfer(own, transfer, hit):
    """The colour from a finer model where its rays hit (`hit` 0..1 per texel), the model's own elsewhere."""
    if transfer is None or hit is None:
        return np.asarray(own, float)
    h = np.clip(np.asarray(hit, float), 0.0, 1.0)[..., None]
    return np.asarray(own, float) * (1.0 - h) + np.asarray(transfer, float) * h


def normal_texels(baked, hit):
    """Tangent-space normal texels (0..1, blue up) from a box-filtered bake: the finer model's normal where
    its rays hit (`hit` 0..1 per texel), the coarse surface as it is (straight up) where they missed, blended
    in between, and of unit length again after the filtering."""
    b = np.asarray(baked, float)[..., :3] * 2.0 - 1.0
    h = np.clip(np.asarray(hit, float), 0.0, 1.0)[..., None]
    n = b * h + np.array([0.0, 0.0, 1.0]) * (1.0 - h)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    return n * 0.5 + 0.5


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

def atlas_density(co, uv, vert_tris, uv_tris, atlas: int, grid_max: int = 2048) -> dict:
    """How well the atlas is spent, measured honestly.

    `texels_per_cm`: the atlas's texels per centimetre of surface, over the whole figure (positions `co` in metres,
    triangles `vert_tris`; UVs `uv` per corner, triangles `uv_tris`). `atlas_covered`: the share of the atlas's
    texels some triangle paints, and `texels_twice`: the share of those that two or more triangles paint, both on a
    grid of the atlas's size (at most `grid_max`). A summed UV area would count a folded island twice: the Wachter's
    atlas looked 55 % and then 68 % used while 1.2 % of its texels were painted twice (#146).
    """
    co, uv = np.asarray(co, float), np.asarray(uv, float)
    vert_tris, uv_tris = np.asarray(vert_tris, int).reshape(-1, 3), np.asarray(uv_tris, int).reshape(-1, 3)
    if not len(vert_tris):
        return {"texels_per_cm": 0.0, "atlas_covered": 0.0, "texels_twice": 0.0}
    p = co[vert_tris]
    surface_cm2 = float(np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum() / 2) * 1e4
    t = uv[uv_tris]
    cross = (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
    uv_texels = float(np.abs(cross).sum() / 2) * atlas * atlas
    n = min(int(atlas), grid_max)
    count = np.zeros(n * n, np.int32)
    px = t * n
    for (ax, ay), (bx, by), (cx, cy) in px:
        area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(area) < 1e-12:
            continue
        x0, x1 = max(0, int(np.floor(min(ax, bx, cx)))), min(n, int(np.ceil(max(ax, bx, cx))))
        y0, y1 = max(0, int(np.floor(min(ay, by, cy)))), min(n, int(np.ceil(max(ay, by, cy))))
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        w0 = ((bx - gx) * (cy - gy) - (cx - gx) * (by - gy)) / area
        w1 = ((cx - gx) * (ay - gy) - (ax - gx) * (cy - gy)) / area
        inside = (w0 >= 0) & (w1 >= 0) & (1 - w0 - w1 >= 0)
        count[(gy[inside].astype(int) * n + gx[inside].astype(int))] += 1
    covered = count > 0
    return {"texels_per_cm": round((uv_texels / surface_cm2) ** 0.5, 3) if surface_cm2 > 0 else 0.0,
            "atlas_covered": round(float(covered.mean()), 4),
            "texels_twice": round(float((count[covered] > 1).mean()), 4) if covered.any() else 0.0}


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
        if bone in REQUIRED_BONES:
            out.append(f"no {bone} bone: Unity's Humanoid needs one, so the avatar will not map")
        else:
            out.append(f"no {bone} bone: Unity's Humanoid can do without one, but some importers expect it")
    error = report.get("animation_error_m")
    if error is not None and error > 0.002:
        out.append(f"moves changed by up to {error:.4f} m in the rescale")
    pose = report.get("pose_check") or {}
    if pose.get("hole_px_left"):
        out.append(f"{pose['hole_px_left']} px of holes in motion left after putting {pose.get('restored', 0)} faces "
                   f"back (worst: {pose.get('worst')}); marked in {pose.get('picture')}")
    check = report.get("colour_check") or {}
    if check.get("patches"):
        n = len(check["patches"])
        out.append(f"{n} patch{'es' if n != 1 else ''} differ in colour from {check['against']} "
                   f"(first: {check['patches'][0]}); marked in {check['picture']}")
    return out


def colour_check(source: Path, output: Path, picture: Path, curve: tuple | None = None,
                 feet: dict | None = None) -> dict:
    """The finished figure against the model its colour came from, drawn from eight sides in their base colour
    (`image_to_3dlab.colour_compare`), leaving out where the figure glows on purpose (its accents). A bake fault
    shows here and nowhere else: a fold in the atlas once put a dark, faceted patch on a cape's back that the
    numbers and a check for holes in motion both missed. `curve` (strength, pivot) puts the readability curve the
    figure got onto the source first, so a figure lightened on purpose is no fault (without it, 5 of 21 figures
    were flagged for exactly that). `feet` (the plan of `--feet-from`) gives the source the same feet first, so it
    stands like the figure. The marked picture is written only when a patch differs.
    """
    from image_to_3dlab import colour_compare as cc

    old, new = cc.read_model(source), cc.read_model(output)
    if curve and curve[0] and curve[1] is not None:
        old = cc.Model(old.positions, old.uvs, old.faces, old.part,
                       [np.round(curve_at(t / 255.0, curve[0], curve[1]) * 255).astype(np.uint8) for t in old.textures])
    if feet:
        old = cc.Model(feet_positions(old.positions, feet, up=1), old.uvs, old.faces, old.part, old.textures)
    patches = cc.marked_picture(old, new, picture, ignore=cc.read_glow(output))
    return {"against": Path(source).name, "patches": [p.describe() for p in patches],
            "picture": Path(picture).name if patches else None}


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
    refusal = rig_refusal(missing_core_bones(unity))
    if refusal and not args.allow_missing_bones:
        print(f"{refusal}; nothing written")
        return 1
    plan = bone_plan(parents, unity, s["fingers"])
    names = {} if s["style"] == "keep" else engine_names(parents, mapping, s["style"])
    names = {b: n for b, n in names.items() if b not in plan}
    hips = next(b for b, n in unity.items() if n == "Hips")
    heads_off = glow_off_bones(parents, unity, s)  # the bake keeps the glow off these bones' skin
    if drop_hidden_refusal(s):
        parser.error(drop_hidden_refusal(s))
    if s["colour_from"] and not Path(s["colour_from"]).is_file():
        parser.error(f"--colour-from: not found: {s['colour_from']}")
    if s["mesh_from"] and not Path(s["mesh_from"]).is_file():
        parser.error(f"--mesh-from: not found: {s['mesh_from']}")
    if (s["head_protect"] > 0 or s["head_texel"] > 1) and "Head" not in unity.values():
        print("note: no head bone found, so --head-protect and --head-texel change nothing")
    feet = None
    if not s["feet_from"] and s["feet"] != "off":
        found = finish_run_from_provenance(args.source)
        if found:
            s["feet_from"] = str(found)
            print(f"feet: the Finish run from the rig's provenance: {found.name}")
    if s["feet_from"] and s["feet"] != "off":
        try:
            outline = outline_parts(Path(s["feet_from"]))
        except FileNotFoundError as error:
            parser.error(f"--feet-from: {error}")
        feet = feet_plan(outline["parts"], s["feet"], s["height"], s)
        print("feet: " + (f"{feet['mode']} at {feet['at']:.1%} of the height ({feet['parts']} part(s) below the "
                          "pictures' feet)" if feet else "nothing below the pictures' feet to fix"))

    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / "job.json"
        raw = Path(tmp) / "raw.glb"
        report_path = Path(tmp) / "report.json"
        job.write_text(json.dumps({"source": str(args.source.resolve()), "output": str(raw), "report": str(report_path),
                                   "work": tmp, "name": args.output.stem, "hips": hips, "merge": plan, "head_bones": heads_off,
                                   "all_head_bones": head_bones(parents, unity), "veto_poses": veto_poses(unity),
                                   "check_poses": check_poses(unity) if s["pose_check"] else [],
                                   "pose_picture": str(args.output.with_suffix(".pose.png").resolve()),
                                   "feet": feet, "feet_bones": [b for b, n in unity.items() if n in FEET_BONES],
                                   "settings": s}, indent=1))
        run = subprocess.run(blender_command(Path(blender), job), capture_output=True, text=True, check=False)
        if run.returncode != 0 or not raw.is_file():
            print(run.stdout[-4000:])
            print(run.stderr[-4000:])
            print("Blender stopped; nothing written")
            return 1
        report = json.loads(report_path.read_text())
        feet = (report.get("feet") or {}).get("plan", feet)  # a cut that left thin plates was pressed flat instead
        if (report.get("feet") or {}).get("switched"):
            print("feet: " + report["feet"]["switched"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rename(raw.read_bytes(), names))
    set_glb_generator(args.output)
    write_derived_sidecar(args.source, args.output, operation="game_ready")
    report["bones_named"] = len(names)
    report["bones_merged"] = len(plan)
    report["missing_core_bones"] = missing_core_bones(unity)
    report["style"] = s["style"]
    if s["colour_check"]:
        made = report.get("colour") or {}
        report["colour_check"] = colour_check(Path(s["colour_from"] or args.source), args.output,
                                              args.output.with_suffix(".colour.png"),
                                              (made.get("readability", 0), made.get("readability_pivot")), feet)
    report["problems"] = problems(report, s["max_influences"], s["height"])
    report["settings"] = {k: v for k, v in s.items() if k != "glow"} | {
        "glow": ["#" + "".join(f"{round(c * 255):02x}" for c in g) for g in s["glow"]]}
    out_report = args.output.with_suffix(".report.json")
    out_report.write_text(json.dumps(report, indent=1))
    print(f"{report['triangles']} triangles, {report['materials']} material, atlas {s['atlas']} "
          f"({report['texels_per_cm']:.1f} texels per cm, {report['atlas_covered'] * 100:.0f} % covered, "
          f"{report['texels_twice'] * 100:.1f} % painted twice, {report['uv_islands']} islands), "
          f"{report['bones']} bones ({len(names)} named, {len(plan)} merged), "
          f"height {report['height']:.2f} m, {len(report['clips'])} move(s)")
    for line in report["problems"] or ["ready: nothing an engine would trip over"]:
        print("  " + line)
    print(f"wrote {args.output} and {out_report.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
