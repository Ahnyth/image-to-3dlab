#!/usr/bin/env python3
"""Turn a rigged figure and its front picture into a finished game figure, every Hero stage checking itself.

    .venv/bin/python scripts/hero_chain.py rigged.glb front.png [--out output/hero_chain/NAME] [--name NAME]
        [--skip hands head boots band] [--box X0 Y0 X1 Y1 ...] [--boot-top Y] [--views DIR] [--before game.glb]
        [--gpu-lock DIR ...] -- <game_ready.py options: --height 1.85 --colour-from paint.glb --mesh-from fine.glb ...>

The Hero tools each fix one thing a full-body build gets wrong, and each needed numbers given by hand. Here they run
in order with none (the finer models the game-ready step bakes paint, bumps and shape from are read from its options
after `--`, and every stage changes them too):

  1. hands  Hero hands (`hero_hands.py`): the hands built alone from the picture, set in at the wrists the rig shows.
            Kept when the finger count (`figure_check`, the hands found by the skeleton's shape) shows a five-digit
            hand on both sides; the tool runs that same count on its own graft and builds again while it fails
            (Pixal3D's build differs from run to run). A drawing whose hand misses the picture's colour (a glove drawn bare) is not built
            from; when every one misses, the tool paints its build from the picture's hand (or, with its --colour-miss
            keep, the figure keeps its own hands and the stage is skipped). Once kept, the game step's glow stays off the new hands
            (`--glow-not-on-hands`; `--glow-on-hands` lets it on).
  2. head   Hero head (`hero_face.py --region auto`): the whole head under a hat's brim, or the face alone without a
            brim, built alone and set in; brim and collar are measured by the tool itself.
  3. boots  Hero parts (`hero_parts.py --boots --top auto`): where the boots end is read in the picture (the boot's
            colour ending up both shins); no boots in the picture, no stage (`--boot-top` gives the row by hand), and
            none when the figure's own boots follow the picture's outline (`hero_parts.own_boots`): a new boot is built
            from a drawing that may make up a boot the picture does not show.
  4. band   Ring paint (`ring_paint.py`): a band whose pattern repeats along it (a hat band's runes) is found in the
            picture the figure was painted from and its runes are carried round it; no such band, no stage (`--box`
            gives it by hand).
  5. game   game_ready.py with the options given, on the last models kept, then figure_check.py on the result.
            The figure's own foot option stays: Hero parts stands new boots on the picture's feet, not on a sole the
            generator made up below them (the Sorcerer's lay 71 px lower: boots stood on it lost their feet to the foot
            cut, which takes off everything below the pictures' feet), so the cut leaves the new boots whole.

Every stage checks itself and is dropped when its check fails, the chain going on with the models from before it:
the colour of the model the game figure takes its paint from must stay as it was outside where the stage changed the
figure (`colour_compare`, a patch elsewhere is paint that bled or a part that moved), and a stage that changes the
shape (hands, head, boots) must open no hole in motion that was not there before (game_ready's pose check, run on the
stage's models), and its game figure must be no worse than the one before it in the stage's own region (the head's
jagged outline, flakes and jaw strokes; the feet reaching ahead of the ankles: `figure_check.stage_against`). A stage
whose own run fails is dropped the same way.

Writes, in --out: a folder per stage (its tool's run folder, its log, stage.json), game/ (the game-ready runs the
checks made), <name>.glb (the finished figure, with its game-ready report) and report/ (index.html and report.json: what
each stage did, measured and why it was kept, dropped or skipped, above the finished figure's own check). A rerun
picks up where it stopped: a stage whose inputs did not change keeps its result, and one whose models only changed
because a stage before it did runs its tool again on them in its folder, keeping the tool's own builds (they depend on
the picture and the stage's own part of the figure). GPU steps take each --gpu-lock folder
first (mkdir is atomic; another run on the same card waits for it), so two runs share one card in turn.

The bookkeeping lives in `image_to_3dlab/hero_chain.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "viewer"), str(REPO / "scripts")]

from image_to_3dlab import hero_chain as hc

PYTHON = Path(sys.executable)
SCRIPTS = REPO / "scripts"
TOOLS = {"hands": "hero_hands.py", "head": "hero_face.py", "boots": "hero_parts.py", "band": "ring_paint.py"}
SUFFIX = {"hands": "hands", "boots": "boots", "band": "band"}
SKIPPED = 3                 # the exit code of hero_parts --top auto (no boots) and ring_paint without a band
GPU_STAGES = ("hands", "head", "boots")
STAGE_HOURS = 4.0           # a stage that runs longer is stopped (a build hung)
LOCK_MINUTES = 90.0         # how long a GPU step waits for the card


def split_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """(the chain's own arguments, game_ready's options after `--`)."""
    if "--" in argv:
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1:]
    return list(argv), []


@contextmanager
def gpu_locks(paths: list[Path], what: str, wait: float = LOCK_MINUTES * 60, step: float = 5.0):
    """Take each lock folder in PATHS in turn (mkdir is atomic: whoever made it holds it), waiting at most WAIT seconds
    for each, and give back only the ones taken here. A `who` file names the process and the step."""
    held = []
    try:
        for path in paths:
            path = Path(path)
            deadline = time.time() + wait
            while True:
                try:
                    path.mkdir()
                    break
                except FileExistsError:
                    if time.time() >= deadline:
                        raise SystemExit(f"[chain] {what}: the GPU lock {path} stayed taken for {wait / 60:.0f} "
                                         "minutes") from None
                    time.sleep(step)
            held.append(path)
            (path / "who").write_text(f"{os.getpid()} hero_chain: {what}", encoding="utf-8")
        yield
    finally:
        for path in reversed(held):
            shutil.rmtree(path, ignore_errors=True)


def game_cut_share(game: list[str]) -> float | None:
    """Where the game step will cut the figure's feet (a share of its height above its lowest point), as game_ready
    plans it from the figure's Finish run (`--feet-from`, `--feet`); None when it cuts nothing there. New boots stand
    there: stood lower, on the made-up sole's floor, the cut took their feet (skin_sorcerer_c, 2026-11-24)."""
    mode, run = hc.option_value(game, "--feet") or "auto", hc.option_value(game, "--feet-from")
    if mode == "off" or not run:
        return None
    import game_ready

    plan = game_ready.feet_plan(game_ready.outline_parts(Path(run))["parts"], mode,
                                float(hc.option_value(game, "--height") or 1.8), game_ready.FIXED)
    return float(plan["at"]) if plan else None


def stage_models(stage: str, models: hc.Models, results: dict) -> hc.Models:
    """MODELS after STAGE wrote RESULTS: the boots stage gives the figure new feet (`Models.new_feet`), the hands stage
    new hands (`Models.new_hands`)."""
    return models.swapped(results, new_feet=stage == "boots", new_hands=stage == "hands")


def stage_command(stage: str, models: hc.Models, picture: Path, folder: Path, given: dict) -> list[str]:
    """The command line of STAGE's tool on MODELS, writing into FOLDER. GIVEN: what a person gave by hand ("top": the
    boots' row in the picture; "boxes": the band's boxes; "views": the cameras' folder)."""
    tool = [str(PYTHON), str(SCRIPTS / TOOLS[stage])]
    also = [str(p) for p in models.also()]
    if stage == "hands":
        return [*tool, str(models.rig), str(picture), "--out", str(folder), *(["--also", *also] if also else [])]
    if stage == "head":
        return [*tool, str(models.rig), str(picture), "--out", str(folder), "--region", "auto",
                *(["--also", *also] if also else [])]
    if stage == "boots":
        command = [*tool, str(models.rig), str(picture), "--out", str(folder), "--boots", "--name", "boots",
                   "--top", str(given.get("top", "auto"))]
        if given.get("floor_at"):
            command += ["--floor-at", f"{given['floor_at']:g}"]
        if given.get("views"):
            command += ["--views", str(given["views"])]
        return command + (["--also", *also] if also else [])
    if stage == "band":
        colour = models.colour()
        command = [*tool, str(colour), str(given["views"]), str(folder / f"{colour.stem}_band.glb"),
                   "--report", str(folder / "band.json")]
        for box in given.get("boxes") or []:
            command += ["--box", *[f"{v:g}" if isinstance(v, float) else str(v) for v in box]]
        return command
    raise ValueError(f"no stage {stage}")


def stage_results(stage: str, models: hc.Models, folder: Path) -> dict:
    """{model: the stage's version of it} as its tool names them (the head's from its face.json, face or head)."""
    if stage in ("hands", "boots"):
        return {m: folder / f"{m.stem}_{SUFFIX[stage]}.glb" for m in models.all()}
    if stage == "band":
        return {models.colour(): folder / f"{models.colour().stem}_band.glb"}
    face = folder / "face.json"
    if not face.is_file():
        return {}
    done = json.loads(face.read_text(encoding="utf-8"))["models"]
    return {m: Path(done[m.name]["result"]) for m in models.all() if m.name in done}


def _files(paths: list[Path]) -> dict:
    """Each file's size and time (None when missing)."""
    files = {}
    for path in paths:
        path = Path(path)
        stat = path.stat() if path.exists() else None
        files[str(path)] = [stat.st_size, stat.st_mtime_ns] if stat else None
    return files


def _fingerprint(origin: dict, paths: list[Path], extra: dict) -> dict:
    """What a stage's result depends on: the chain's own inputs (ORIGIN), the stage's input files and its options."""
    return {"origin": origin, "files": _files(paths), "options": extra}


def _last_line(log: Path) -> str:
    try:
        lines = [line.strip() for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1][:300] if lines else ""


class Chain:
    def __init__(self, args, game: list[str], run):
        self.args, self.game, self.run = args, list(game), run
        self.out = run.out
        self.locks = [Path(p) for p in args.gpu_lock]
        self.games: dict = {}               # models' key -> (game glb, its report)
        self.origin: dict = {}              # the chain's own inputs, as `_files` gives them

    # --- running tools -------------------------------------------------------------------------------------------

    def call(self, command: list[str], log: Path, what: str, gpu: bool) -> int:
        """Run COMMAND, its output into LOG, under the GPU locks when GPU; its exit code (None: stopped, too long)."""
        self.run.say(f"{what}: $ " + " ".join(command))
        env = dict(os.environ, PYTHONIOENCODING="utf-8")

        def go():
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(" ".join(command) + "\n")
                handle.flush()
                try:
                    return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(REPO),
                                          timeout=self.args.stage_hours * 3600, check=False).returncode
                except subprocess.TimeoutExpired:
                    return None

        if gpu and self.locks:
            with gpu_locks(self.locks, what, self.args.lock_minutes * 60):
                return go()
        return go()

    def game_run(self, models: hc.Models, tag: str, game: list[str]) -> tuple[Path | None, dict | None]:
        """game_ready.py on MODELS with GAME's options into game/<tag>.glb, once per set of models and options. The models
        count with their size and time: a rerun of a stage writes new models under the same names, and keyed on the
        paths alone the game step kept skin_sorcerer_c's figure with its old boots (2026-11-24)."""
        key = json.dumps([_files(list(models.all())), *game])
        if key in self.games:
            return self.games[key]
        folder = self.out / "game"
        folder.mkdir(exist_ok=True)
        for stamp in folder.glob("*.key.json"):   # made before under another name: the hands' figure is the head's before
            made = stamp.with_name(stamp.name[:-len(".key.json")] + ".glb")
            if (made.exists() and made.with_suffix(".report.json").exists()
                    and stamp.read_text(encoding="utf-8") == key):
                self.games[key] = (made, json.loads(made.with_suffix(".report.json").read_text(encoding="utf-8")))
                return self.games[key]
        glb = folder / f"{tag}.glb"
        report_path = glb.with_suffix(".report.json")
        stamp = glb.with_suffix(".key.json")
        if not (glb.exists() and report_path.exists() and stamp.exists()
                and stamp.read_text(encoding="utf-8") == key):
            command = [str(PYTHON), str(SCRIPTS / "game_ready.py"), str(models.rig), str(glb), *models.game_args(game)]
            code = self.call(command, glb.with_suffix(".log"), f"game-ready {tag}", gpu=True)
            if code != 0 or not glb.exists():
                self.run.flag(f"game-ready on {tag} failed ({code}): {_last_line(glb.with_suffix('.log'))}")
                self.games[key] = (None, None)
                return self.games[key]
            stamp.write_text(key, encoding="utf-8")
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        self.games[key] = (glb, report)
        return self.games[key]

    # --- the stages ----------------------------------------------------------------------------------------------

    def views(self, models: hc.Models) -> Path | None:
        """The cameras the colour was painted through: --views, else the Pixal3D run's found through a licence record."""
        if self.args.views:
            return self.args.views
        import hero_parts

        for model in (models.colour(), models.rig, *models.also()):
            found = hero_parts.run_views(Path(model))
            if found is not None:
                return found
        return None

    def stage(self, index: int, name: str, models: hc.Models, game: list[str]) -> tuple[hc.Stage, hc.Models | None]:
        """Run stage NAME on MODELS (or keep its result from a run before with the same inputs): what it did, and the
        models after it when it is kept."""
        folder = self.out / f"{index}_{name}"
        given = {}
        if name == "boots":
            if self.args.boot_top is not None:
                given["top"] = self.args.boot_top
            try:
                cut = game_cut_share(game)
            except (OSError, ValueError, KeyError, LookupError) as error:
                cut = None
                self.run.say(f"boots: the game step's foot cut could not be read ({error}): the boots stand on the "
                             "figure's floor")
            if cut:
                given["floor_at"] = cut
            if self.args.views:
                given["views"] = self.args.views
        if name == "band":
            views = self.views(models)
            if views is None:
                return hc.Stage(name, "skipped", "no cameras of the figure's Pixal3D run were found to look for a band "
                                                 "in (give --views)"), None
            given["views"], given["boxes"] = views, self.args.box or []
        inputs = [*models.all(), self.args.picture] + ([Path(given["views"]) / "transforms.json"]
                                                       if given.get("views") else [])
        print_given = {k: (str(v) if isinstance(v, Path) else v) for k, v in given.items()}
        fingerprint = _fingerprint(self.origin, inputs, {"stage": name, "given": print_given})
        record = folder / "stage.json"
        if record.is_file():
            kept = json.loads(record.read_text(encoding="utf-8"))
            how = hc.rerun(kept.get("fingerprint"), fingerprint)
            result = kept.get("result", {})
            if how == "same" and result.get("status") in ("kept", "dropped", "skipped") and all(
                    Path(p).exists() for p in result.get("outputs", [])):
                self.run.say(f"{name}: kept from the run before ({result['status']})")
                done = hc.Stage(name, result["status"], result["why"], result.get("measured", {}),
                                result.get("seconds", 0.0), result.get("models", {}),
                                [tuple(p) for p in result.get("pictures", [])])
                return done, (hc.Models.from_dict(done.models) if done.status == "kept" else None)
            if how == "rebuild":               # the stages before it changed the models: its builds still hold
                for old in result.get("outputs", []):
                    if Path(old).parent.resolve() == folder.resolve():
                        Path(old).unlink(missing_ok=True)
                self.run.say(f"{name}: the stages before it changed the models; its tool runs again on them, keeping "
                             "its own builds")
            elif how == "fresh":
                aside = folder.with_name(folder.name + "_before")
                shutil.rmtree(aside, ignore_errors=True)
                folder.rename(aside)
                self.run.say(f"{name}: its inputs changed since the run before; that run moved to {aside.name}")
        folder.mkdir(parents=True, exist_ok=True)
        started = time.time()
        result, new = self.run_stage(name, models, folder, given, game)
        result.seconds = time.time() - started
        outputs = [str(p) for p in (new.all() if new is not None else [])]
        record.write_text(json.dumps({"fingerprint": fingerprint,
                                      "result": {**result.as_dict(), "outputs": outputs}}, indent=1, default=str),
                          encoding="utf-8")
        return result, new

    def run_stage(self, name: str, models: hc.Models, folder: Path, given: dict,
                  game: list[str]) -> tuple[hc.Stage, hc.Models | None]:
        command = stage_command(name, models, self.args.picture, folder, given)
        log = folder / "stage.log"
        code = self.call(command, log, name, gpu=name in GPU_STAGES)
        if code == SKIPPED and name in ("boots", "band", "hands"):
            why = self.skip_reason(name, folder)
            return hc.Stage(name, "skipped", why, self.measure(name, folder)), None
        if code != 0:
            return hc.Stage(name, "failed", f"its tool stopped ({code}): {_last_line(log)}"), None
        results = stage_results(name, models, folder)
        missing = [str(p) for p in results.values() if not Path(p).exists()]
        if not results or missing:
            return hc.Stage(name, "failed", f"its tool wrote no result for {missing or 'any model'}"), None
        new = stage_models(name, models, results)
        stage = hc.Stage(name, "kept", "", self.measure(name, folder), models=new.as_dict())
        checks = self.check(name, models, new, folder, stage, game)
        failed = [why for ok, why in checks if ok is False]
        stage.why = "; ".join(why for _, why in checks)
        if failed:
            stage.status, stage.why = "dropped", "; ".join(failed)
            stage.models = {}
            return stage, None
        return stage, new

    def skip_reason(self, name: str, folder: Path) -> str:
        if name == "boots":
            data = folder / "boots.json"
            found = json.loads(data.read_text(encoding="utf-8")) if data.is_file() else {}
            if found.get("own_boots"):              # the figure's own boots follow the picture: they stay
                return found.get("skipped") or "the figure's own boots follow the picture"
            return f"no boots in the picture: {found.get('skipped') or 'see stage.log'}"
        if name == "hands":                         # no drawing kept the picture's hand: the figure keeps its own
            data = folder / "hands.json"
            found = json.loads(data.read_text(encoding="utf-8")) if data.is_file() else {}
            return found.get("skipped") or "the figure keeps its own hands: see stage.log"
        return "no band in the picture whose pattern repeats along it"

    def measure(self, name: str, folder: Path) -> dict:
        """What the stage's tool measured, from its own JSON."""
        try:
            if name == "hands":
                data = json.loads((folder / "hands.json").read_text(encoding="utf-8"))
                out = {f"{side} hand": {"digits": s["count"]["digits"], "thumbs": s["count"]["thumbs"],
                                        "scale": s.get("scale"), "wrist eased": s.get("wrist_ratio")}
                       for side, s in data.get("sides", {}).items()}      # none when the figure kept its own hands
                for side, colour in (data.get("picture_hand") or {}).items():   # what the drawings were held to
                    if colour:
                        out[f"{side} hand in the picture"] = f"L*a*b* {colour['lab']}, {colour['skin']:.0%} skin-toned"
                return out | {"to look at": len(data.get("unsure", []))}
            if name == "head":
                data = json.loads((folder / "face.json").read_text(encoding="utf-8"))
                region = json.loads((folder / "region.json").read_text(encoding="utf-8"))
                first = next(iter(data["models"].values()))
                out = {"region": region["region"], "why": region["why"], "crop px": data["crop"][2] - data["crop"][0],
                       "apart from the old face": round(data["placement"]["apart_from_old_face"], 5)}
                if "head" in data:
                    out["collar round the neck"] = round(data["head"]["collar_share"], 2)
                    out["build shows its head in"] = round(data["head"]["build_shows"], 2)
                out["open edges round it"] = first.get("open_edges_near")
                return out | {"to look at": len(data.get("unsure", []))}
            if name == "boots":
                data = json.loads((folder / "boots.json").read_text(encoding="utf-8"))
                out = {}
                top = data.get("boot_top") or {}
                if top.get("top") is not None:
                    out["boots end at view row"] = round(top["top"], 1)
                for side, s in (top.get("sides") or {}).items():
                    if s.get("contrast") is not None:
                        out[f"{side} colour step"] = s["contrast"]
                for part, p in (data.get("parts") or {}).items():
                    out[part] = {"pieces": p.get("pieces"), "triangles": p.get("triangles"),
                                 "sole below the picture's feet px": p.get("sole_below_picture_feet_px"),
                                 "open edges after": p.get("open_edges_near_after")}
                return out | ({"to look at": len(data.get("unsure", []))} if "unsure" in data else {})
            if name == "band":
                data = json.loads((folder / "band.json").read_text(encoding="utf-8"))
                found = data.get("found", []) if isinstance(data, dict) else []
                rings = data.get("rings", data) if isinstance(data, dict) else data
                out = {f"band {i + 1}": {"box": b["box"], "marks": b["marks"], "spacing spread": b["gap_cv"]}
                       for i, b in enumerate(found)}
                for i, r in enumerate(rings):
                    out[f"ring {i + 1}"] = {"texels repainted": r.get("repainted"), "seen arc deg":
                                            round(r.get("arc_degrees", 0.0), 1), "skipped": r.get("skipped")}
                return out
        except (OSError, KeyError, ValueError, StopIteration, TypeError) as error:
            return {"not read": str(error)}
        return {}

    def check(self, name: str, before: hc.Models, after: hc.Models, folder: Path, stage: hc.Stage,
              game: list[str]) -> list[tuple]:
        """[(passed or None, why), ...]: the stage's own checks, cheapest first; the pose check only when the others
        passed."""
        from image_to_3dlab import figure_check as fc
        from image_to_3dlab import ring_paint as rp

        checks = []
        if name == "hands":
            stage.measured["fingers per cut before"] = fc.fingers(fc.read_figure(before.rig)).value
            result = fc.fingers(fc.read_figure(after.rig))
            ok, why = hc.fingers_verdict(result)
            stage.measured["fingers per cut"] = result.value
            checks.append((ok, why))
            if not ok:
                return checks
        from image_to_3dlab import colour_compare as cc

        if name in hc.GEOMETRY:      # seconds, where a hole otherwise shows only in the game figure ten minutes later
            for label, old_path, new_path in hc.shape_pairs(before, after):
                result = fc.see_through_against(cc.read_model(old_path), cc.read_model(new_path))
                stage.measured[f"see-through, {label}"] = result.value
                checks.append((result.status != "fail", f"the {label}: {result.message}"))
                if result.status == "fail":
                    return checks
        what = {"hands": "the hands", "head": "the head", "boots": "the boots", "band": "the band"}[name]
        old, new = cc.read_model(before.colour()), cc.read_model(after.colour())    # each read once: 6144 maps
        if name == "band":
            data = json.loads((folder / "band.json").read_text(encoding="utf-8"))
            rings = data.get("rings", data) if isinstance(data, dict) else data

            def inside(points, rings=rings):
                hit = rp.in_ring(points, {})
                for ring in rings:
                    hit = hit | rp.in_ring(points, ring)
                return hit
        else:
            inside, changed = hc.changed_region(old.positions, new.positions)
            stage.measured["points of the colour model changed"] = changed
        picture = folder / "colour_outside.png"
        patches = hc.colour_outside(old, new, inside, picture, glow=cc.read_glow(after.colour()))
        del old, new
        ok, why = hc.colour_verdict(patches, what)
        stage.measured["patches outside it"] = len(patches)
        if patches and picture.exists():
            stage.pictures.append(("colour outside the stage's own region (old | new | changed in red)", picture))
        checks.append((ok, why))
        if not ok or name not in hc.GEOMETRY or name not in self.args.pose_check:
            return checks
        glb, report = self.game_run(after, f"{name}", game)
        if glb is None:
            checks.append((False, "the game-ready step failed on its models"))
            return checks
        ok, why = hc.pose_verdict(report, None)
        if ok is False:                       # holes: were they there before the stage?
            _, before_report = self.game_run(before, f"before_{name}", game)
            ok, why = hc.pose_verdict(report, before_report)
        stage.measured["holes in motion"] = (report or {}).get("pose_check", {}).get("hole_px_left")
        checks.append((ok, why))
        if ok is not False:
            checks += self.game_checks(name, before, after, stage, game)
        return checks

    def game_checks(self, name: str, before: hc.Models, after: hc.Models, stage: hc.Stage,
                    game: list[str]) -> list[tuple]:
        """The stage judged where its fault shows: on the game figure, against the game figure before the stage, in
        the stage's own region (`figure_check.stage_against`). The stage checks above compare the paint source, so
        they passed skin_sorcerer_c's head stage, whose game figure had dark shards in the back hair, and its boots,
        whose feet game_ready cut off (2026-11-24)."""
        from image_to_3dlab import figure_check as fc

        glb, report = self.game_run(after, name, game)
        old, old_report = self.game_run(before, f"before_{name}", game)
        if glb is None or old is None:
            return [(None, "the game figure before or after the stage was not made: not compared")]
        out = [hc.problems_verdict(report, old_report)]
        if name not in fc.STAGE_LOOKS:
            return out
        for r in fc.stage_against(name, old, glb):
            stage.measured[f"game figure: {r.name}"] = r.value
            out.append((r.status != "fail", f"the game figure: {r.message}" if r.status == "fail"
                        else f"the game figure's {r.name} no worse than before"))
        return out


def main(argv: list[str] | None = None) -> int:
    own, game = split_args(sys.argv[1:] if argv is None else list(argv))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="the figure, rigged (AssetFurnace's Rig output)")
    parser.add_argument("picture", type=Path, help="the front picture it was built from")
    parser.add_argument("--out", type=Path, default=None, help="run folder (default output/hero_chain/<model>)")
    parser.add_argument("--name", default=None, help="the finished figure's file name (default: the model's)")
    parser.add_argument("--skip", nargs="*", choices=hc.STAGES, default=[], help="stages to leave out")
    parser.add_argument("--box", type=float, nargs=4, action="append", default=None, metavar=("X0", "Y0", "X1", "Y1"),
                        help="the band's box in the front view's pixels, instead of finding it (repeatable)")
    parser.add_argument("--boot-top", type=float, default=None, metavar="Y",
                        help="where the boots end in the front picture, instead of finding it")
    parser.add_argument("--views", type=Path, default=None,
                        help="the cameras of the Pixal3D run (its .svviews folder); found through the licence record "
                             "when left out")
    parser.add_argument("--before", type=Path, default=None,
                        help="the figure's game model before (its version in the game): the finished one is checked "
                             "against it")
    parser.add_argument("--glow-on-hands", action="store_true",
                        help="let the game step's glow reach new hands too (a glowing gauntlet or a spell in the hand); "
                             "by default the chain keeps it off them once its hands stage is kept")
    parser.add_argument("--pose-check", nargs="*", choices=hc.GEOMETRY, default=list(hc.GEOMETRY),
                        help="the stages checked in motion (a game-ready run each; default all that change the shape)")
    parser.add_argument("--gpu-lock", type=Path, action="append", default=[],
                        help="a lock folder every GPU step takes first (repeatable), so runs sharing a card take turns")
    parser.add_argument("--lock-minutes", type=float, default=LOCK_MINUTES, help="how long a GPU step waits for a lock")
    parser.add_argument("--stage-hours", type=float, default=STAGE_HOURS, help="a stage running longer is stopped")
    args = parser.parse_args(own)
    if not game:
        parser.error("give game_ready.py's options after --, as for the figure's game-ready step (at least --height)")
    args.model, args.picture = args.model.resolve(), args.picture.resolve()

    from hero_route import Run, stage

    from image_to_3dlab import figure_check as fc

    out = (args.out or REPO / "output" / "hero_chain" / args.model.stem[:60]).resolve()
    run = Run(out)
    name = args.name or args.model.stem
    models = hc.Models(args.model, {flag: path.resolve() for flag, path in hc.game_models(game).items()},
                       hand_glow=args.glow_on_hands)
    game = models.game_args(game)
    run.say(f"hero chain: {args.model} + {args.picture} -> {out}; finer models {models.as_dict()}")
    chain = Chain(args, game, run)
    chain.origin = _files([*models.all(), args.picture])
    stages = []
    for index, stage_name in enumerate(hc.STAGES, start=1):
        if stage_name in args.skip:
            stages.append(hc.Stage(stage_name, "off", "left out with --skip"))
            continue
        stage(run, stage_name, f"Stage {index}: {stage_name}")
        result, after = chain.stage(index, stage_name, models, game)
        stages.append(result)
        run.say(f"{stage_name}: {hc.STATUS_WORDS[result.status]}: {result.why}")
        if after is not None:                  # a stage kept from a run before reads its models back without the
            models = replace(after, hand_glow=args.glow_on_hands)   # caller's choice of this run

    stage(run, "game", "The game-ready step")
    glb, report = chain.game_run(models, "final", game)
    final = {"models": models.as_dict(), "game_options": game}
    if glb is None:
        hc.write_summary(out / "chain.json", stages, {**final, "failed": "the game-ready step failed"})
        run.say("game-ready failed on the last models; see game/final.log")
        return 1
    result_path = out / f"{name}.glb"
    for suffix in (".glb", ".report.json", ".provenance.json", ".pose.png", ".colour.png"):
        source = glb.with_suffix(suffix) if suffix != ".glb" else glb
        if source.exists():
            shutil.copy2(source, result_path.with_suffix(suffix) if suffix != ".glb" else result_path)
    final.update(glb=str(result_path), triangles=(report or {}).get("triangles"),
                 holes_in_motion=(report or {}).get("pose_check", {}).get("hole_px_left"))

    stage(run, "check", "Checking the finished figure")
    report_dir = out / "report"
    picture = report_dir / f"{name}_colour.png" if args.before else None
    report_dir.mkdir(exist_ok=True)
    checked = fc.check(result_path, before=args.before, colour_picture=picture)
    final["check"] = {"failed": checked.failed, "results": [r.as_dict() for r in checked.results]}
    hc.write_summary(out / "chain.json", stages, final)
    page = fc.write_report([checked], report_dir, f"Hero chain: {name}", preface=hc.preface(stages, report_dir),
                           extra={"stages": [s.as_dict() for s in stages], "final": final})
    run.say(f"done: {result_path} ({final['triangles']} triangles); report {page}")
    for s in stages:
        run.say(f"  {s.name}: {hc.STATUS_WORDS[s.status]}: {s.why}")
    for r in checked.results:
        run.say(f"  check {fc.STATUS_WORDS[r.status]} {r.name}: {r.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
