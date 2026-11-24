"""The Hero chain's own logic: which models a stage changes, where it changed them, and whether to keep the change.

`scripts/hero_chain.py` runs Hero hands, Hero head, the boots, the band's runes and the game-ready step on a rigged
figure in order. Each stage changes every model of the figure in the same frame (the rigged one and the finer ones the
game-ready step bakes paint, bumps and shape from), checks itself, and is kept only when its check passes; otherwise
the chain goes on with the models from before it. Everything here is plain maths and bookkeeping, tested without
running a stage:

* the finer models are read from the game-ready options (`--colour-from`, `--normal-from`, `--mesh-from`) and swapped
  for a stage's results there;
* where a stage changed the figure's shape (`changed_region`: the points of either model that the other lacks, grown
  by a reach), or, for the band, its ring (`ring_paint.in_ring`);
* the colour outside that region against the stage before (`colour_compare` with `leave_out`): a patch there is a
  fault (paint bled, a part moved), and the stage is dropped;
* the fingers (`figure_check.fingers`), the holes in motion against the stage before (game-ready's pose check);
* the report: one row per stage (kept, dropped and why, skipped and why, what it measured) above the finished
  figure's own check (`figure_check.write_report`).
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GAME_MODELS = ("--colour-from", "--normal-from", "--mesh-from")
STAGES = ("hands", "head", "boots", "band")
GEOMETRY = ("hands", "head", "boots")      # the stages that change the figure's shape: checked in motion too
MOVED_SHARE = 0.0005     # a point farther than this share of the figure's height from all of the other model's moved
REACH_SHARE = 0.03       # a colour change this close to where the shape changed (share of the height) is the stage's own
STATUS_WORDS = {"kept": "kept", "dropped": "dropped", "skipped": "skipped", "off": "not asked for",
                "failed": "failed"}
STATUS_COLOURS = {"kept": "#3fa34d", "dropped": "#e04848", "skipped": "#8090a0", "off": "#8090a0",
                  "failed": "#e04848"}


# --- the game-ready options --------------------------------------------------------------------------------------


def option_value(args: list[str], flag: str) -> str | None:
    """The value of FLAG in a command line (`--flag value` or `--flag=value`; the last one counts), or None."""
    value = None
    for i, word in enumerate(args):
        if word == flag and i + 1 < len(args):
            value = args[i + 1]
        elif word.startswith(flag + "="):
            value = word[len(flag) + 1:]
    return value


def with_option(args: list[str], flag: str, value: str | None) -> list[str]:
    """ARGS with FLAG set to VALUE (every earlier FLAG taken out; VALUE None takes it out only)."""
    out, skip = [], False
    for i, word in enumerate(args):
        if skip:
            skip = False
            continue
        if word == flag and i + 1 < len(args):
            skip = True
            continue
        if word.startswith(flag + "="):
            continue
        out.append(word)
    return out if value is None else [*out, flag, str(value)]


def game_models(args: list[str]) -> dict[str, Path]:
    """The finer models a game-ready command line bakes from, by option: {"--colour-from": path, ...}."""
    return {flag: Path(value) for flag in GAME_MODELS if (value := option_value(args, flag))}


@dataclass
class Models:
    """A figure's models in one frame: the rigged one and the finer ones the game-ready step takes."""
    rig: Path
    finer: dict = field(default_factory=dict)        # game-ready option -> path
    new_feet: bool = False                           # a boots stage replaced the feet: the game step cuts none

    def all(self) -> list[Path]:
        """The rigged model first, then each finer one once."""
        seen, out = set(), []
        for path in [self.rig, *self.finer.values()]:
            key = str(Path(path).resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(Path(path))
        return out

    def also(self) -> list[Path]:
        return self.all()[1:]

    def colour(self) -> Path:
        """The model the game figure's colour comes from: --colour-from, else the rigged model."""
        return Path(self.finer.get("--colour-from", self.rig))

    def swapped(self, results: dict, new_feet: bool = False) -> Models:
        """These models with each one a stage changed (RESULTS: old path -> new path, by resolved path) swapped;
        NEW_FEET when the stage gave the figure new feet (kept from then on)."""
        index = {str(Path(old).resolve()).lower(): Path(new) for old, new in results.items()}

        def swap(path):
            return index.get(str(Path(path).resolve()).lower(), Path(path))

        return Models(swap(self.rig), {flag: swap(path) for flag, path in self.finer.items()},
                      self.new_feet or new_feet)

    def game_args(self, args: list[str]) -> list[str]:
        """A game-ready command line's options with these models as its finer ones. With new feet the game step cuts
        none: the sole it would cut (planned from the figure's Finish run) went with the old feet, and measured from
        the new lowest point its cut took the new boots' feet (skin_sorcerer_c, 2026-11-24)."""
        out = list(args)
        for flag, path in self.finer.items():
            out = with_option(out, flag, str(path))
        if self.new_feet:
            out = with_option(with_option(out, "--feet-from", None), "--feet", "off")
        return out

    def as_dict(self) -> dict:
        return {"rig": str(self.rig), **{flag.lstrip("-"): str(path) for flag, path in self.finer.items()},
                **({"new_feet": True} if self.new_feet else {})}

    @classmethod
    def from_dict(cls, data: dict) -> Models:
        """`as_dict` read back."""
        return cls(Path(data["rig"]), {f"--{key}": Path(value) for key, value in data.items()
                                       if key not in ("rig", "new_feet")}, bool(data.get("new_feet")))


# --- where a stage changed the figure ------------------------------------------------------------------------------


def changed_points(old: np.ndarray, new: np.ndarray, tolerance: float) -> np.ndarray:
    """The points of either cloud farther than TOLERANCE from every point of the other: what a stage took away and
    what it put in."""
    from scipy.spatial import cKDTree

    old, new = np.asarray(old, dtype=np.float64), np.asarray(new, dtype=np.float64)
    if not len(old) or not len(new):
        return np.concatenate([old, new])
    gone = cKDTree(new).query(old, distance_upper_bound=tolerance, workers=-1)[0] > tolerance
    came = cKDTree(old).query(new, distance_upper_bound=tolerance, workers=-1)[0] > tolerance
    return np.concatenate([old[gone], new[came]])


def near(points: np.ndarray, reach: float):
    """points (N, 3) -> bool (N,): within REACH of any of POINTS (none: never)."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return lambda p: np.zeros(len(p), dtype=bool)
    tree = cKDTree(points)

    def inside(p):
        p = np.asarray(p, dtype=np.float64)
        if not len(p):
            return np.zeros(0, dtype=bool)
        return np.isfinite(tree.query(p, distance_upper_bound=reach, workers=-1)[0])

    return inside


def changed_region(old: np.ndarray, new: np.ndarray, moved: float = MOVED_SHARE, reach: float = REACH_SHARE):
    """(inside, how many points changed): `inside(points)` says which lie where a stage changed the shape of a model
    whose vertices were OLD and are NEW (grown by REACH of the figure's height, for the paint eased in round a seam)."""
    old, new = np.asarray(old, dtype=np.float64), np.asarray(new, dtype=np.float64)
    height = float(np.ptp(np.concatenate([old[:, 1], new[:, 1]])))
    changed = changed_points(old, new, moved * height)
    return near(changed, reach * height), int(len(changed))


def colour_outside(old, new, inside, picture: Path | None = None, size: int = 384, glow=None) -> list:
    """The patches where NEW differs in colour from OLD (paths, or `colour_compare.Model`s read already) outside what
    INSIDE marks (and where NEW glows on purpose: GLOW, read from NEW's file when it is a path), with a marked picture
    at PICTURE when one differs (`colour_compare`)."""
    from image_to_3dlab import colour_compare as cc

    if isinstance(new, (str, Path)):
        glow = cc.read_glow(Path(new))
    a = cc.read_model(Path(old)) if isinstance(old, (str, Path)) else old
    b = cc.read_model(Path(new)) if isinstance(new, (str, Path)) else new
    frame = cc.shared_frame(a, b)          # a stage changes the figure in place, never moves it
    if picture is not None:
        return cc.marked_picture(a, b, Path(picture), size=size, ignore=glow, leave_out=inside, frame=frame)
    return cc.compare(a, b, size=size, ignore=glow, leave_out=inside, frame=frame)


def model_points(path: Path) -> np.ndarray:
    """A GLB's vertex positions in its own frame (every mesh, node transforms applied)."""
    from image_to_3dlab import colour_compare as cc

    return cc.read_model(Path(path)).positions


# --- verdicts -----------------------------------------------------------------------------------------------------


def fingers_verdict(result) -> tuple[bool, str]:
    """A `figure_check.fingers` result as a stage's check: both hands counted, each with as many pieces per cut as a
    five-digit hand shows."""
    if result.status == "info":
        return False, f"the hands could not be counted ({result.message})"
    if result.status != "pass":
        return False, f"the fingers do not count right: {result.value} (a five-digit hand shows 4 per cut)"
    if result.value.count("hand") < 2:
        return False, f"only {result.value} was counted"
    return True, f"{result.value} per cut across the fingers"


def pose_verdict(after: dict | None, before: dict | None) -> tuple[bool | None, str]:
    """Holes in motion (game-ready's pose check, `hole_px_left`) after a stage against before it: (passed, why); passed
    is None when there was no pose check (the game-ready options have no --drop-hidden)."""
    left = (after or {}).get("pose_check", {}).get("hole_px_left") if after else None
    if left is None:
        return None, "no pose check (the game-ready options do not drop hidden faces)"
    if left == 0:
        return True, "no see-through hole in motion"
    was = (before or {}).get("pose_check", {}).get("hole_px_left") if before else None
    if was is not None and left <= was:
        return True, f"{left} see-through pixel(s) in motion, as many as before it ({was})"
    return False, (f"{left} see-through pixel(s) in motion"
                   + (f", {was} before it" if was is not None else "") + ": the stage opened a hole")


def colour_verdict(patches: list, what: str) -> tuple[bool, str]:
    """The colour outside a stage's own region as its check: no patch, or the first ones named."""
    if not patches:
        return True, f"no colour changed outside {what}"
    words = "; ".join(p.describe() for p in patches[:3]) + (f" (and {len(patches) - 3} more)" if len(patches) > 3 else "")
    return False, f"{len(patches)} patch(es) changed colour outside {what}: {words}"


# --- a rerun -----------------------------------------------------------------------------------------------------


def rerun(before: dict | None, now: dict) -> str:
    """How a stage meets the run before it, from the two fingerprints ({"origin": the chain's own inputs, "files": the
    stage's input models, "options"}, each file as [size, time]): "same" (nothing changed: keep its result),
    "rebuild" (the chain's inputs are as they were and only the stages before it changed the models: its tool runs
    again in its folder and keeps its own builds, which depend on the picture and its own part of the figure, not on
    what the stages before it changed elsewhere), or "fresh" (the chain's own inputs or the stage's options changed,
    or there was no run before: start over)."""
    if not before:
        return "fresh"
    if before == now:
        return "same"
    if before.get("origin") != now.get("origin") or before.get("options") != now.get("options"):
        return "fresh"
    return "rebuild"


# --- the report ---------------------------------------------------------------------------------------------------


@dataclass
class Stage:
    """What one stage did."""
    name: str
    status: str = "off"                      # kept | dropped | skipped | off | failed
    why: str = ""
    measured: dict = field(default_factory=dict)
    seconds: float = 0.0
    models: dict = field(default_factory=dict)       # the models after it, when kept
    pictures: list = field(default_factory=list)     # (caption, path) shown in the report

    def as_dict(self) -> dict:
        return {"stage": self.name, "status": self.status, "why": self.why, "measured": self.measured,
                "seconds": round(self.seconds, 1), "models": self.models,
                "pictures": [[c, str(p)] for c, p in self.pictures]}


def _value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)) and len(value) <= 6 and all(isinstance(v, (int, float)) for v in value):
        return ", ".join(_value(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k} {_value(v)}" for k, v in value.items())
    return str(value)


def preface(stages: list[Stage], out: Path) -> str:
    """The stages as an HTML table (what each did, measured, and why it was kept, dropped or skipped) and the pictures
    each left (paths relative to OUT where they lie inside it)."""
    rows = [("<h2>Stages</h2><table><tr><th>stage</th><th>result</th><th>why</th><th>measured</th>"
             "<th>seconds</th></tr>")]
    for s in stages:
        measured = "<br>".join(f"{html.escape(k)}: {html.escape(_value(v))}" for k, v in s.measured.items())
        rows.append(f"<tr><th>{html.escape(s.name)}</th><td><b style='color:{STATUS_COLOURS[s.status]}'>"
                    f"{STATUS_WORDS[s.status]}</b></td><td>{html.escape(s.why)}</td><td><small>{measured}</small></td>"
                    f"<td>{s.seconds:.0f}</td></tr>")
    rows.append("</table>")
    for s in stages:
        for caption, path in s.pictures:
            path = Path(path)
            try:
                link = path.resolve().relative_to(Path(out).resolve()).as_posix()
            except ValueError:
                link = path.resolve().as_uri()
            rows.append(f"<p>{html.escape(s.name)}: {html.escape(caption)}</p><img src='{html.escape(link)}'>")
    return "".join(rows)


def write_summary(path: Path, stages: list[Stage], final: dict) -> None:
    """chain.json: every stage and the finished figure."""
    Path(path).write_text(json.dumps({"stages": [s.as_dict() for s in stages], "final": final}, indent=1,
                                     default=str), encoding="utf-8")
