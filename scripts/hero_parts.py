#!/usr/bin/env python3
"""Set a small part of a figure in again, built on its own at full detail: a holstered pistol, a pouch, or both boots.

    .venv/bin/python scripts/hero_parts.py rigged.glb front.png --box X0 Y0 X1 Y1 [--box X0 Y0 X1 Y1 --mirror]
        [--name pistol] [--look "black pistol in a black holster with teal runes"] [--hard] [--also finer.glb ...]
    .venv/bin/python scripts/hero_parts.py rigged.glb front.png --boots [--top Y|auto] [--left mirror|own] [--look "..."] ...

A part that is a few dozen pixels of a full-body picture gets few of Pixal3D's voxels: the Revolverheld's holstered
pistol came out as a flat slab with the pistol's outline, the Zauberer's boots as smooth tubes with their straps only
painted on. Built on its own from a clean drawing, the part fills Pixal3D's whole box and keeps its shape. The Hero
hands do this for hands; this does it for any part on the body (--box) and for boots (--boots):

  1. measure  the figure's front camera (the one its Pixal3D run used, found through its licence record, or --views);
              with --box the part's box in the front picture (pixels, as you see it in any image viewer), with --boots
              the legs from the rig by the tree's shape (no names, no hand-drawn box).
  2. crop     the part cut from the figure's front view and upscaled 4x with Real-ESRGAN.
  3. draw     the picture editor (FLUX.2 klein, edit mode) redraws the crop as the part alone and complete: a part on
              the body as the picture shows it, a boot from its outer side with the toe to the right; --tries drawings,
              the first --keep that are one whole piece clear of the edges (a boot with its toe where it was told) kept.
              --drawing gives one instead.
  4. build    Pixal3D on each kept drawing, --seeds each; the first build that is one piece is the part (--build gives
              one instead).
  5. clean    Pixal3D's own clean-up of that build at --part-faces; a build that touches itself in many places (sheets
              meeting at one edge, which the game-ready step's thinning tears into shards) is then made one closed
              skin with its paint baked back on (--solid, needs Blender); with --hard its nearly flat pieces made
              flat and their edges sharp (a pistol, a buckle; never a face or cloth).
  6. place    a part on the body fills its box by height, its back settled on the body behind the slab Pixal3D made of
              it; a second --box with --mirror takes the first part mirrored (a pair of holsters). A boot stands on the
              figure's floor with its sole flat, turned as the old foot was, as tall as the picture shows it (--top:
              where its shaft ends in the picture; --top auto finds it there, where the boot's colour ends up the shin
              of both legs, and with no boots in the picture stops with exit code 3) or else as long as the old foot
              (--left mirror: the right boot mirrored; own: drawn and built from its own crop).
  7. graft    inside a part's outline the figure loses what lies in front of the part's back (the slab, a strap across
              it) and keeps what lies behind (the thigh); a leg is cut straight across just under the new boot's top and
              the old boot below goes. Every outline the cut leaves is closed, so no edge stays open. A part on the body
              moves rigidly with the bones the slab followed; a boot takes the old leg's weights point by point, so it
              bends at the ankle. Every --also model (the finer ones a game-ready step bakes paint and bumps from, in
              the same frame) gets the same parts, built finer (--fine-faces).
  8. paint    each part painted by the figure's own picture where that sees it (through its camera, past the coat
              in front of it) and by its own build elsewhere, moved to the picture's tones.
  9. check    each part's pieces counted, the open edges round it, a boot's sole against the picture's feet.

Writes <out>/<model>_<name>.glb (and <out>/<also>_<name>.glb), with <name>.json (every number) and a licence record.
Every step skips when its output exists, so a rerun picks up after a fix. Where the code is unsure the log says
UNSURE: that is where a human should look.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "viewer"), str(REPO / "scripts")]

from hero_route import (
    Run,
    stage,
)

from image_to_3dlab.blender import find_blender, missing_help

PYTHON = Path(sys.executable)
GENERATE = REPO / "scripts" / "pixal3d_generate.py"
UPSCALE = REPO / "scripts" / "upscale_esrgan.py"
TRIES, KEEP, SEEDS = 6, 2, (42, 7)
PART_FACES = 20000            # Pixal3D's clean-up of a part built alone, for the rigged model ...
FINE_FACES = 150000           # ... and for the finer --also models: the raw build is about a million faces a part, and two
                              # raw boots in the Zauberer's mesh source made game_ready walk 2.4 million faces, not 0.42
DRAW_SIZE = 1024
CROP_PAD = 1.25               # a part's crop is this many times its own size
PAINT_SIZE = 2048             # a part's paint is laid out at most this many texels square
SOLE_LIMIT_PX = 3.0           # a boot's sole this many of the picture's pixels below its feet is a made-up sole
FAULT_SHARE = 0.005           # a build may touch itself on at most this share of its edges (the pistol: 0.08 per cent;
                              # the Zauberer's first boot: 1.4, and game_ready kept 81,000 of 14,000 triangles)
SOLID_VOXEL = 0.003           # a build that touches itself more is made one closed skin, voxels this share of its size
                              # (the Zauberer's boot at 20,000 faces: 1,715 such edges before, none after, its straps
                              # and buckles kept)
LICENCES = [{"component": "Real-ESRGAN x4plus", "license": "BSD-3-Clause", "purpose": "part crop upscale"}]

PART_PROMPT = (
    "The same {name} as in the reference image, a stylized 3D game asset like the reference, smooth simple surfaces, "
    "not a photo. Shown on its own and complete, nothing of it hidden by anything in front of it. Standing upright "
    "exactly as in the reference and seen from the same side as in the reference. {look} Nothing else in the "
    "picture. Flat even lighting, plain light grey background.")
BOOT_PROMPT = (
    "The same {side} boot as in the reference image, a stylized 3D game asset like the reference, smooth simple "
    "surfaces, not a photo. Shown on its own and complete, from the top of its shaft down to its sole, nothing of "
    "the trousers or the coat. Seen exactly from its outer side in strict profile, standing upright on a flat "
    "ground, the toe pointing to the {toe} of the picture. {look} Nothing else in the picture. Flat even lighting, "
    "plain light grey background.")
DEFAULT_LOOK = "The same shapes, details and colours as in the reference."


NO_BOOTS = 3                  # the exit code of --boots --top auto when the picture shows no boots


def _top(value: str):
    """--top: a row of the front picture, or auto."""
    return value if value == "auto" else float(value)


def fresh(run: Run, name: str) -> None:
    """Start a step that always runs (it is quick): its notes from an earlier run go, so the closing list says only
    what this run found."""
    run.current = name
    if run.notes.pop(name, None) is not None:
        run.unsure_file.write_text(json.dumps(run.notes, indent=2), encoding="utf-8")


def needed_parts(drawing: bool = True) -> dict[str, Path]:
    """Everything the steps run, by what to install to get it."""
    import pixal3d_generate

    from image_to_3dlab import pixal3d_native, upscale

    parts = {"Pixal3D's own clean-up (Setup: Rebuild Pixal3D)": pixal3d_native.POST_REPLAY,
             "Hero face upscaler (Real-ESRGAN, 67 MB)": upscale.WEIGHTS_FILE}
    if drawing:
        parts["Pixal3D (3D models)"] = pixal3d_generate.CLI
    return parts


def picture_engine():
    """The picture editor in edit mode, or a plain-words reason it cannot run (nothing is downloaded)."""
    import image_api

    try:
        engine = image_api.pick_engine("flux", edit=True)
        image_api.check_ready(engine, edit=True)
    except Exception as error:
        return None, (f"the picture editor (FLUX.2 klein, edit mode) is not set up: {error}. Set it up on the "
                      "Setup page (nothing was downloaded).")
    return engine, None


def run_views(model: Path) -> Path | None:
    """The cameras of the Pixal3D run a model was made from: its licence record's chain of sources, down to the
    pixal3d record, whose model has its views beside it (`<run>.svviews`). None when the chain has none."""
    from image_to_3dlab.provenance import read_model_record

    record = read_model_record(model)
    for _ in range(64):
        if not isinstance(record, dict):
            return None
        if record.get("backend") == "pixal3d" or record.get("operation") == "pixal3d":
            out = (record.get("output") or {}).get("path")
            views = Path(out).with_suffix(".svviews") if out else None
            return views if views is not None and (views / "transforms.json").is_file() else None
        record = record.get("source_record")
    return None


def mask_of(picture: Path):
    """The subject's mask: its own transparency, else the background remover's cut-out."""
    import numpy as np
    from PIL import Image

    from image_to_3dlab import matte

    with Image.open(picture) as opened:
        cut = opened.convert("RGBA") if matte.is_matted(opened) else matte.cut_out(opened.convert("RGB"))[0]
    return np.asarray(cut)[..., 3] > 127


def drawing_check(mask, toe: str | None = None) -> dict:
    """Whether a drawing shows one whole part: one piece (95 per cent of its outline in the largest), clear of the
    picture's edges (not cut off), at least a fifth of its height; with TOE ('left' or 'right') a boot whose foot
    reaches out that way from its shaft (where the picture editor was told to point it)."""
    import numpy as np
    from scipy import ndimage

    mask = np.asarray(mask, bool)
    label, count = ndimage.label(mask)
    if not count:
        return {"ok": False, "why": "nothing drawn"}
    sizes = np.bincount(label.ravel())[1:]
    main = label == 1 + int(np.argmax(sizes))
    ys, xs = np.nonzero(main)
    out = {"one_piece": round(float(sizes.max() / sizes.sum()), 3),
           "clear": bool(xs.min() > 0 and ys.min() > 0 and xs.max() < mask.shape[1] - 1 and ys.max() < mask.shape[0] - 1),
           "height": round(float(np.ptp(ys) / mask.shape[0]), 3)}
    ok = out["one_piece"] >= 0.95 and out["clear"] and out["height"] >= 0.2
    if toe is not None:
        low = main & (np.arange(mask.shape[0])[:, None] > ys.max() - 0.15 * np.ptp(ys))
        high = main & (np.arange(mask.shape[0])[:, None] < ys.min() + 0.5 * np.ptp(ys))
        reach = np.nonzero(low)[1].mean() - np.nonzero(high)[1].mean()
        out["toe_at"] = "right" if reach > 0 else "left"
        ok = ok and out["toe_at"] == toe
    out["ok"] = bool(ok)
    return out


def crop_part(run: Run, view, box, name: str, folder: Path):
    """The part's square crop of the figure's front view (BOX in its pixels), upscaled: (the crop's box, its image
    upscaled as RGBA, the path of that image)."""
    import numpy as np
    from PIL import Image

    from image_to_3dlab import hero_hands as hh

    height, width = view.image.shape[:2]
    x0, y0, x1, y1 = box
    square = hh.crop_box(np.array([x0, x1]), np.array([y0, y1]), width, height, CROP_PAD)
    crop, up = folder / f"{name}_crop.png", folder / f"{name}_up.png"
    if run.step(f"crop {name}", up):
        Image.fromarray(np.asarray(view.image)).crop(square).save(crop)
        run.run([PYTHON, UPSCALE, crop, up, "--size", str(DRAW_SIZE)])
    with Image.open(up) as opened:
        return square, np.asarray(opened.convert("RGBA")), up


def draw_parts(run: Run, name: str, up: Path, folder: Path, args, engine, prompt: str,
               toe: str | None) -> list[Path]:
    """Up to --keep drawings of the part alone (`drawing_check`)."""
    import image_api
    from PIL import Image

    from image_to_3dlab import back_view
    from image_to_3dlab.provenance import write_provenance

    reference = folder / f"{name}_reference.png"
    if not reference.exists():
        with Image.open(up) as opened:
            back_view.reference_on_grey(opened.convert("RGBA")).save(reference)
    weights, vision = image_api.resolve_weights(engine), image_api.resolve_vision(engine)
    kept = []
    for seed in range(1, args.tries + 1):
        drawing = folder / f"{name}_draw_s{seed}.png"
        checked = drawing.with_suffix(".check.json")
        if not drawing.exists():
            settings = image_api.clean_settings({"seed": seed, "width": DRAW_SIZE, "height": DRAW_SIZE}, engine=engine)
            started = time.time()
            command = image_api.build_command(prompt, settings, drawing, weights, reference=reference,
                                              vision=vision, engine=engine)
            run.say(f"draw: {name}, seed {seed}")
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                    cwd=str(REPO), check=False)
            if result.returncode or not drawing.exists():
                run.flag(f"the picture editor failed on seed {seed}: {(result.stdout + result.stderr)[-300:]}")
                continue
            record = image_api.provenance(prompt, settings, time.time() - started, drawing, engine)
            record["reference"] = {"picture": str(up), "mode": f"edit ({name} redrawn alone)"}
            write_provenance(drawing.with_suffix(".provenance.json"), record)
        if not checked.exists():
            checked.write_text(json.dumps(drawing_check(mask_of(drawing), toe)), encoding="utf-8")
        check = json.loads(checked.read_text(encoding="utf-8"))
        run.say(f"draw: {drawing.name} {check}" + (" - kept" if check["ok"] else ""))
        if check["ok"]:
            kept.append(drawing)
            if len(kept) >= args.keep:
                break
    return kept


def build_part(run: Run, drawings: list[Path], folder: Path, seeds) -> Path | None:
    """The first Pixal3D build of a kept drawing that is one piece once its crumbs are gone and whose surface touches
    itself in few places (`hero_parts.edge_faults`, at most FAULT_SHARE of its edges); else the one-piece build that
    touches itself least, with a note."""
    from image_to_3dlab import hero_hands as hh
    from image_to_3dlab import hero_parts as hp

    best = None
    for drawing in drawings:
        for seed in seeds:
            raw = folder / f"{drawing.stem}_b{seed}.glb"
            if not raw.exists():
                run.say(f"build: Pixal3D on {drawing.name}, seed {seed}")
                if not run.run([PYTHON, "-u", GENERATE, drawing, raw, "--seed", str(seed)], required=False):
                    run.flag(f"Pixal3D failed on {drawing.name} seed {seed} (see run.log)")
                    continue
            part, crumbs = hp.drop_crumbs(hh.read_part(raw.read_bytes())[2])
            count, faults = hp.pieces(part), hp.edge_faults(part)
            ok = count == 1 and faults["share"] <= FAULT_SHARE
            run.say(f"build: {raw.name} {count} piece(s), {crumbs} crumb(s), {faults['shared_by_more']} edges where "
                    f"it touches itself ({faults['share']:.2%})" + (" - kept" if ok else ""))
            if ok:
                return raw
            if count == 1 and (best is None or faults["share"] < best[1]):
                best = (raw, faults["share"])
    if best is not None:
        run.flag(f"every build touches itself in many places; {best[0].name} least ({best[1]:.2%} of its edges): it "
                 "is made one closed skin, unless --solid off")
        return best[0]
    return None


def clean_part(run: Run, raw: Path, faces: int, folder: Path) -> Path:
    """Pixal3D's own clean-up of the build: a light mesh with one painted atlas, in the build's own frame, written
    into the run's FOLDER (a build given with --build may lie in someone else's)."""
    from image_to_3dlab import pixal3d_native as native

    clean = folder / f"{raw.stem}_clean{faces // 1000}k.glb"
    if not clean.exists():
        dump = raw.with_suffix(native.DUMP_SUFFIX)
        if not dump.is_file():
            raise SystemExit(f"[hero] {raw.name} has no clean-up dump ({dump.name}): build it again with Pixal3D")
        replay = folder / f"{raw.stem}.replay.glb"
        run.run(native.replay_command(dump, replay, faces, 2048))
        converted = native.convert_replay_glb(replay.read_bytes())
        problem = native.frame_problem(native.glb_bounds(raw.read_bytes()), native.glb_bounds(converted))
        if problem:
            raise SystemExit(f"[hero] the part's clean-up does not sit on its build: {problem}")
        clean.write_bytes(converted)
        replay.unlink(missing_ok=True)
    return clean


def wants_solid(mode: str, touching: float) -> bool:
    """Whether a build is made one closed skin: --solid on always, off never, auto when it touches itself on more
    than FAULT_SHARE of its edges."""
    return mode == "on" or (mode == "auto" and touching > FAULT_SHARE)


def solid_part(run: Run, clean: Path, faces: int, folder: Path) -> Path:
    """The cleaned part made one closed skin: Blender's voxel remesh (voxels SOLID_VOXEL of its size), thinned back to
    FACES, its paint baked back on (`blender_retopo_bake.py`, the Finish's own retopology), written into FOLDER.

    Why: where a build touches itself, Pixal3D left sheets meeting at one edge (the Zauberer's boot: 1,715 such edges
    at 20,000 faces), and the game-ready step's thinning tore them into shards at the ankle and the heel. The skin has
    none; the small specks the remesh leaves go with the part's crumbs."""
    from retopo_repaint import retopo_command

    solid = folder / f"{clean.stem}_solid.glb"
    if not solid.exists():
        blender = find_blender()
        if blender is None:
            raise SystemExit(f"[hero] {missing_help()}")
        run.say(f"clean: {clean.name} made one closed skin ({faces} faces)")
        run.run(retopo_command(clean, solid, faces, PAINT_SIZE, 89.0, SOLID_VOXEL, 0.25, 0.65, 1.45, blender))
    return solid


def built_part(path: Path, hard: bool, run: Run, cache: Path) -> tuple[dict, dict | None]:
    """A built part read as a part dict, its crumbs gone, made hard with HARD (cached beside the run)."""
    from image_to_3dlab import hero_hands as hh
    from image_to_3dlab import hero_parts as hp

    source = cache if hard and cache.exists() else path
    part = hp.drop_crumbs(hh.read_part(source.read_bytes())[2])[0]
    if not hard or source == cache:
        return part, None
    part, report = hp.harden(part)
    run.say(f"hard edges: {path.name}: {report}")
    writer_part = hh.finish_part(part)
    cache.write_bytes(_part_glb(writer_part))
    return part, report


def _part_glb(part: dict) -> bytes:
    """One part as a GLB of its own (its material copied)."""
    from image_to_3dlab import hero_hands as hh

    writer = hh.GlbWriter()
    part = dict(part)
    if part.get("material") is not None:
        part["material"] = writer.copy_material(*part["material"])
    writer.doc["meshes"] = [{"primitives": [writer.primitive(part)]}]
    writer.doc["nodes"] = [{"mesh": 0}]
    writer.doc["scenes"] = [{"nodes": [0]}]
    writer.doc["scene"] = 0
    return writer.build()


def _picture_record(raw: Path) -> dict | None:
    """The licence record of the drawing a build was made from, as a component record (`hero_hands._picture_record`)."""
    import hero_hands

    record = hero_hands._picture_record(raw)
    if record is not None:
        record["operation"] = "part drawing"
        record["components"][0]["purpose"] = "the part redrawn alone"
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="the figure, rigged (or with --rig, any model in its frame)")
    parser.add_argument("picture", type=Path, help="the front picture it was built from")
    parser.add_argument("--out", type=Path, default=None, help="run folder (default output/hero_parts/<model>)")
    parser.add_argument("--rig", type=Path, default=None, help="the figure's rig, when MODEL is not rigged")
    parser.add_argument("--also", type=Path, nargs="*", default=[],
                        help="more models of the figure in the same frame (the finer ones a game-ready step bakes "
                             "paint and bumps from): each gets the same parts")
    parser.add_argument("--views", type=Path, default=None,
                        help="the cameras of the Pixal3D run the figure was made from (its .svviews folder); found "
                             "through the model's licence record when left out")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--box", type=float, nargs=4, action="append", metavar=("X0", "Y0", "X1", "Y1"),
                      help="a part on the body inside this box of the front picture, in its pixels (a holstered "
                           "pistol from its grip to its muzzle); give it again for another")
    what.add_argument("--boots", action="store_true", help="both boots, set in at the shins")
    parser.add_argument("--mirror", action="store_true",
                        help="with two --box: the second takes the first part mirrored (a pair of holsters)")
    parser.add_argument("--top", type=_top, default=None, metavar="Y",
                        help="with --boots: where the boots' shafts end in the front picture, its pixels from the top, "
                             "or auto: found there, where the boot's colour ends up the shin (exit code 3 when the "
                             "picture shows no boots); the drawing's own proportions decide without it, from the old "
                             "foot's length")
    parser.add_argument("--left", choices=("mirror", "own"), default="mirror",
                        help="with --boots: mirror: the left boot is the right one mirrored; own: drawn and built "
                             "from its own crop")
    parser.add_argument("--name", default=None, help="what the part is, for the drawing and the file names "
                                                     "(default: part, or boots)")
    parser.add_argument("--look", default=None, help="what the part looks like, in words, for the drawing")
    parser.add_argument("--drawing", type=Path, default=None, help="a drawing of the part alone to build from")
    parser.add_argument("--build", type=Path, default=None, help="a build of the part alone to set in")
    parser.add_argument("--tries", type=int, default=TRIES, help="drawings per part")
    parser.add_argument("--keep", type=int, default=KEEP, help="good drawings to build from")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="Pixal3D seeds per drawing")
    parser.add_argument("--part-faces", type=int, default=PART_FACES, help="the build's clean-up size")
    parser.add_argument("--fine-faces", type=int, default=FINE_FACES,
                        help="the build's clean-up size for the --also models (0: the raw build itself)")
    parser.add_argument("--hard", action="store_true",
                        help="a hard part (a pistol, a buckle): its nearly flat pieces made flat and their edges sharp")
    parser.add_argument("--solid", choices=("auto", "on", "off"), default="auto",
                        help="make the cleaned part one closed skin (needs Blender): auto when the build touches "
                             "itself in many places, which the game-ready step's thinning tears into shards")
    args = parser.parse_args(argv)
    boxes = [tuple(b) for b in (args.box or [])]
    if args.mirror and len(boxes) < 2:
        parser.error("--mirror needs a second --box")
    if args.boots and args.left == "own" and (args.drawing or args.build):
        parser.error("--left own draws and builds each boot: leave out --drawing and --build")
    name = args.name or ("boots" if args.boots else "part")

    drawing_needed = args.build is None and args.drawing is None
    missing = [what for what, path in needed_parts(args.build is None).items() if not Path(path).exists()]
    engine, problem = picture_engine() if drawing_needed else (None, None)
    reasons = ([f"not installed: {', '.join(missing)}. Install them from the Setup page (nothing was downloaded)"]
               if missing else []) + ([problem] if problem else [])
    if reasons:
        print("[hero] cannot start: " + "; ".join(reasons), file=sys.stderr)
        return 2

    import numpy as np
    from PIL import Image
    from rig_check import skeleton

    from image_to_3dlab import hero, matte, photo_paint
    from image_to_3dlab import hero_face as hf
    from image_to_3dlab import hero_hands as hh
    from image_to_3dlab import hero_head as hd
    from image_to_3dlab import hero_parts as hp
    from image_to_3dlab.provenance import read_model_record, write_derived_sidecar

    out = (args.out or REPO / "output" / "hero_parts" / args.model.stem[:60]).resolve()
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    run = Run(out)
    run.say(f"hero parts: {args.model} + {args.picture} -> {out} ({name})")

    # 1. measure
    stage(run, "measure", "Finding the part")
    fresh(run, "measure")
    rig = args.rig or args.model
    rig_doc, _, rig_parts = hf.read_parts(rig.read_bytes())
    skinned = [p for p in rig_parts if p.get("dense") is not None]
    if not skinned:
        raise SystemExit(f"[hero] {rig.name} has no skin: give the figure's rig with --rig")
    rig_points = np.concatenate([p["positions"] for p in skinned])
    rig_bone = np.concatenate([p["dense"].argmax(axis=1) for p in skinned])

    def bones(part):
        if part.get("dense") is not None:
            return part["dense"].argmax(axis=1)
        return hero.regions(part["positions"], rig_points, rig_bone)

    data = args.model.read_bytes()
    _, _, figure = hf.read_parts(data)
    figure_bones = [bones(p) for p in figure]
    every = np.concatenate([p["positions"] for p in figure])
    every_faces = np.concatenate([np.asarray(p["faces"]) + o for p, o in
                                  zip(figure, np.cumsum([0] + [len(p["positions"]) for p in figure]))])
    cutout = out / "picture_cutout.png"
    if run.step("cut-out", cutout):
        matte.cut_out(Image.open(args.picture).convert("RGB"))[0].save(cutout)
    picture = hero.load_picture(cutout)
    views_dir = args.views or run_views(args.model)
    if views_dir is not None:
        views, mesh_scale = photo_paint.load_views(views_dir)
        view = views[0]
        stage_map = hp.staging(hp.subject_box(np.asarray(picture.convert("RGBA"))[..., 3]),
                               hp.subject_box(view.image[..., 3]))
        run.say(f"measure: the figure's own camera from {views_dir}; picture to view x{stage_map[0]:.4f}")
    else:
        view, mesh_scale, stage_map = hp.far_view(every, picture), 1.0, (1.0, np.zeros(2))
        run.flag("no cameras of the figure's Pixal3D run were found (--views): the picture is laid on the model "
                 "straight on, and its paint may land a few pixels off")

    targets = []        # {"name", "source", "turn" (3x3), "mirror", "box" (view pixels) or "leg"}
    if boxes:
        for index, box in enumerate(boxes):
            staged = hp.staged_box(box, stage_map)
            source = 0 if args.mirror else index
            targets.append({"name": f"{name}{index + 1}" if len(boxes) > 1 else name, "source": source,
                            "mirror": bool(args.mirror and index > 0), "box": staged, "picture_box": list(box)})
            run.say(f"measure: {targets[-1]['name']}: picture box {box} -> view {np.round(staged, 1).tolist()}")
    else:
        parents, heads = skeleton(rig)
        joint_index = {rig_doc["nodes"][j].get("name") or f"joint{j}": i
                       for i, j in enumerate(rig_doc["skins"][0]["joints"])}
        legs, _ = hp.leg_bones(parents, heads)
        if set(legs) != set(hh.SIDES):
            raise SystemExit(f"[hero] the rig shows {sorted(legs) or 'no'} leg(s) with a foot; both are needed")
        all_bones = np.concatenate(figure_bones)
        plans = {side: hp.leg_plan(every, all_bones, joint_index[legs[side]["shin"]],
                                   [joint_index[b] for b in [legs[side]["foot"], *legs[side]["below"]]],
                                   heads[legs[side]["shin"]], heads[legs[side]["foot"]]) for side in hh.SIDES}
        top_view = None                  # where the boots end, in the view's pixels
        if args.top == "auto":
            pixels = {}
            for side, plan in plans.items():
                x, y, _ = hp.view_pixels(view, np.array([plan["knee"], plan["ankle"]]), mesh_scale)
                pixels[side] = ((float(x[0]), float(y[0])), (float(x[1]), float(y[1])))
            boot_top = hp.boot_top(view.image, pixels)
            if boot_top["top"] is None:
                run.say(f"measure: no boots in the picture: {boot_top['why']}")
                (out / f"{name}.json").write_text(json.dumps({"model": str(args.model), "picture": str(args.picture),
                                                              "name": name, "skipped": boot_top["why"],
                                                              "boot_top": boot_top}, indent=1), encoding="utf-8")
                return NO_BOOTS
            top_view = boot_top["top"]
            run.say(f"measure: the boots end at row {top_view:.1f} of the figure's front view ("
                    + ", ".join(f"{side} {s['top']:.0f}, colour step {s['contrast']:.1f}"
                                for side, s in boot_top["sides"].items()) + ")")
        elif args.top is not None:
            top_view = args.top * stage_map[0] + stage_map[1][1]
        for side in hh.SIDES:
            plan = plans[side]
            near = (np.isin(all_bones, plan["bones"]) & (every[:, 1] < plan["knee"][1])
                    & (hp._across(every, plan["knee"], plan["ankle"]) < plan["reach"]))
            x, y, _ = hp.view_pixels(view, every[near], mesh_scale)
            source = "right" if args.left == "mirror" else side
            targets.append({"name": f"{side}_boot", "source": source, "mirror": side != source, "leg": plan,
                            "box": (float(x.min()), float(y.min()), float(x.max()), float(y.max())), "side": side})
            if top_view is not None:     # the picture's top of the shaft, carried to the shin's front
                box = targets[-1]["box"]
                top = hp.unproject(view, [(box[0] + box[2]) / 2], [top_view], plan["knee"][2] + plan["radius"],
                                   mesh_scale)[0, 1]
                targets[-1]["boot_height"] = float(top - plan["floor"])
            run.say(f"measure: {side} leg: floor {plan['floor']:.4f}, knee {np.round(plan['knee'], 4).tolist()}, "
                    f"ankle {np.round(plan['ankle'], 4).tolist()}, reach {plan['reach']:.4f}"
                    + (f", boot {targets[-1]['boot_height']:.4f} tall" if top_view is not None else ""))

    # 2.-5. crop, draw, build, clean
    sources = sorted({t["source"] for t in targets}, key=str)
    crops, built = {}, {}
    records = []
    for source in sources:
        label = f"{name}{source + 1}" if isinstance(source, int) and len(boxes) > 1 else (
            f"{source}_boot" if args.boots else name)
        target = next(t for t in targets if t["source"] == source and not t["mirror"])
        stage(run, "crop", f"Cropping and upscaling {label}")
        square, image, up = crop_part(run, view, target["box"], label, folder)
        crops[source] = (square, image)
        for t in targets:
            if t["source"] == source and not t["mirror"]:
                t["crop"] = (square, image)
        raw = args.build
        if raw is None:
            drawings = [args.drawing] if args.drawing else None
            if drawings is None:
                stage(run, "draw", f"Drawing {label} alone")
                toe = target.get("side") and ("right" if target["side"] == "right" else "left")
                look = (args.look or DEFAULT_LOOK).strip()
                prompt = (BOOT_PROMPT.format(side=target["side"], toe=toe, look=look) if args.boots
                          else PART_PROMPT.format(name=name, look=look))
                drawings = draw_parts(run, label, up, folder, args, engine, prompt, toe)
                if not drawings:
                    raise SystemExit(f"[hero] none of {args.tries} drawings of {label} shows it whole: try more "
                                     "(--tries), describe it (--look), or give one (--drawing)")
            stage(run, "build", f"Building {label} on its own (the long step)")
            raw = build_part(run, drawings, folder, args.seeds)
            if raw is None:
                raise SystemExit(f"[hero] no build of {label} is one piece: try more seeds (--seeds) or drawings")
        stage(run, "clean", f"Cleaning up {label}")
        clean = clean_part(run, raw, args.part_faces, folder)
        touching = hp.edge_faults(hh.read_part(raw.read_bytes())[2])["share"]
        solid = wants_solid(args.solid, touching)
        if solid:
            clean = solid_part(run, clean, args.part_faces, folder)
        if args.also and (solid or touching > FAULT_SHARE):
            # a build that touches itself kept it in the bigger clean-up too (the Zauberer's boot: 5,839 such edges at
            # 150,000 faces, 1,715 at 20,000), and the game-ready step thinned the finer model it went into to 81,000
            # triangles and dropped half the figure; the finer models take the part's own clean-up (or skin) instead
            fine = clean
            if not solid:
                run.flag(f"{raw.name} touches itself on {touching:.2%} of its edges: the finer models get its "
                         f"{args.part_faces}-face clean-up too")
        else:
            fine = clean_part(run, raw, args.fine_faces, folder) if args.also and args.fine_faces else raw
        clean_part_dict, hard_clean = built_part(clean, args.hard, run, folder / f"{clean.stem}_hard.glb")
        raw_part_dict, hard_raw = built_part(fine, args.hard, run, folder / f"{fine.stem}_hard.glb")
        built[source] = {"clean": clean_part_dict, "raw": raw_part_dict, "clean_path": clean, "raw_path": raw,
                         "hard": {"clean": hard_clean, "raw": hard_raw}, "solid": solid,
                         "build_touches_itself": round(touching, 5)}
        records += [r for r in (read_model_record(raw), _picture_record(raw)) if r]

    # 6. place
    stage(run, "place", "Placing the parts")
    fresh(run, "place")
    for t in targets:
        src = built[t["source"]]
        turn = np.eye(3) if not args.boots else hp.BOOT_TURN["right" if t["source"] == "right" else "left"]
        t["turn"] = turn
        points = src["clean"]["positions"] @ turn.T
        if t["mirror"]:
            points = points * [-1.0, 1.0, 1.0]
        if args.boots:
            plan = t["leg"]
            tf = hp.place_boot(points, plan["foot"], plan["floor"], height=t.get("boot_height"))
            moved = hh.moved(tf, points)
            level = hp.level_sole(moved)
            tf = level @ tf
            moved = hh.moved(tf, points)
            tf[1, 3] += plan["floor"] - moved[:, 1].min()
            moved = hh.moved(tf, points)
            top, height = float(moved[:, 1].max()), float(np.ptp(moved[:, 1]))
            rim = hp.boot_rim(moved)                 # the leg is cut under the rim's lowest point, inside the boot
            t["cut_y"] = rim - hp.BOOT_OVERLAP_SHARE * height
            t["report"] = {"scale": round(float(np.cbrt(np.linalg.det(tf[:3, :3]))), 5), "top": round(top, 5),
                           "rim_lowest": round(rim, 5), "cut_y": round(t["cut_y"], 5),
                           "knee_y": round(float(plan["knee"][1]), 5),
                           "level_deg": round(float(np.degrees(np.arccos(np.clip(level[1, 1], -1, 1)))), 2)}
            if t["cut_y"] > plan["knee"][1]:
                run.flag(f"the new {t['side']} boot reaches above the knee ({top:.4f} > {plan['knee'][1]:.4f}): "
                         "the leg is cut above the knee; check the drawing's height")
        else:
            x0, y0, x1, y1 = t["box"]
            middle_x, middle_y = (x0 + x1) / 2, (y0 + y1) / 2
            z = 0.0
            for _ in range(2):          # the box carried onto the figure's front where it is, twice (the depth moves it)
                edges = hp.unproject(view, [x0, x1, middle_x, middle_x], [middle_y, middle_y, y1, y0], z, mesh_scale)
                target_box = (edges[0, 0], edges[1, 0], edges[2, 1], edges[3, 1])
                inner = (target_box[0] + 0.2 * (target_box[1] - target_box[0]),
                         target_box[1] - 0.2 * (target_box[1] - target_box[0]),
                         target_box[2] + 0.2 * (target_box[3] - target_box[2]),
                         target_box[3] - 0.2 * (target_box[3] - target_box[2]))
                pixel = (target_box[3] - target_box[2]) / hp.FIELD_PIXELS
                front, _ = hp.depth_maps(every, every_faces, inner, pixel)
                z = float(np.median(front[np.isfinite(front)])) if np.isfinite(front).any() else z
            faces = src["clean"]["faces"][:, ::-1] if t["mirror"] else src["clean"]["faces"]
            tf, report = hp.place_on_body(points, faces, target_box, z, every, every_faces)
            width = float(np.ptp(hh.moved(tf, points)[:, 0])) / (target_box[1] - target_box[0])
            t["height"] = report.pop("height")
            t["report"] = {**report, "width_to_box": round(width, 3)}
            if not 0.75 <= width <= 1.33:
                run.flag(f"{t['name']} is {width:.2f} times as wide as its box at its height: check the box and "
                         "the drawing")
        t["tf"] = tf
        run.say(f"place: {t['name']}: {t['report']}")

    # 7. graft (the rigged model first: its cut figure hides what the picture cannot see of a part when it is painted)
    stage(run, "graft", "Setting the parts in")
    fresh(run, "graft")

    def placed(t, which):
        part = dict(built[t["source"]][which])
        p = part["positions"] @ t["turn"].T
        normals = None if part.get("normals") is None else part["normals"] @ t["turn"].T
        faces = part["faces"]
        if t["mirror"]:
            p, faces, normals = hh.mirror_x(p, faces, normals)
        part.update(positions=hh.moved(t["tf"], p), faces=faces)
        if normals is not None:
            part["normals"] = normals @ (t["tf"][:3, :3] / np.cbrt(np.linalg.det(t["tf"][:3, :3]))).T
        return part

    def graft(parts, part_bones, which, main):
        reports = {}
        new = []
        for t in targets:
            part = placed(t, which)
            if args.boots:
                parts, report = hp.graft_boot(parts, part_bones, t["leg"], t["cut_y"], placed(t, "clean"))
                if main is not None:
                    part["dense"] = hp.boot_weights(part["positions"], main[0], main[1], t["leg"])
            else:
                if "field" not in t:
                    t["field"] = hp.part_field(placed(t, "clean"), t["height"] / hp.FIELD_PIXELS,
                                               hp.GROW_SHARE * t["height"])
                parts, rows, report = hp.graft_on_body(parts, t["field"], hp.BEHIND_SHARE * t["height"])
                if rows is not None and len(rows):
                    part["dense"] = np.repeat(hp.rigid_weights(rows)[None], len(part["positions"]), axis=0)
                elif main is not None:
                    near = hh.borrowed_weights(part["positions"].mean(axis=0, keepdims=True), main[0]["positions"],
                                               main[0]["dense"])
                    part["dense"] = np.repeat(near, len(part["positions"]), axis=0)
                    run.flag(f"{t['name']} replaced nothing of the figure: it moves with the nearest point's bones")
            part_bones = [bones(p) for p in parts]
            new.append(part)
            reports[t["name"]] = report
        return parts, new, reports

    main_dense = None
    if figure[0].get("dense") is not None:
        merged = hh.merge_parts([p for p in figure if p.get("dense") is not None])
        main_dense = (merged, merged["dense"].argmax(axis=1))
    cut_main, new_main, graft_reports = graft(list(figure), figure_bones, "clean", main_dense)
    for t, part in zip(targets, new_main):
        t["placed_clean"] = part
        run.say(f"graft: {t['name']}: {graft_reports[t['name']]}")

    # 8. paint
    stage(run, "paint", "Painting the parts from the picture")
    fresh(run, "paint")
    occluders = (np.concatenate([p["positions"] for p in cut_main]),
                 np.concatenate([np.asarray(p["faces"]) + o for p, o in
                                 zip(cut_main, np.cumsum([0] + [len(p["positions"]) for p in cut_main]))]))

    def paint(t, part, which):
        crop = t.get("crop") or next(x["crop"] for x in targets if x["source"] == t["source"] and "crop" in x)
        if t["mirror"]:   # a mirrored part's own crop: its box in the view, cut and upscaled like its source's
            if "own_crop" not in t:
                t["own_crop"] = crop_part(run, view, t["box"], t["name"], folder)[:2]
            crop = t["own_crop"]
        texture = hf.part_paint(part)
        if texture is None:
            run.flag(f"{t['name']} ({which}) has no paint of its own to start from: left unpainted")
            return part, 0.0
        image = Image.fromarray(np.clip(texture, 0, 255).astype(np.uint8))
        if max(image.size) > PAINT_SIZE:
            image = image.resize((PAINT_SIZE, PAINT_SIZE), Image.LANCZOS)
        settings = photo_paint.Settings(align=False)
        result, weight = photo_paint.paint_texture(np.asarray(image), part["positions"], part["uvs"], part["faces"],
                                                   [hp.crop_view(view, crop[0], crop[1])], mesh_scale,
                                                   hp.FRAME, settings, occluders=occluders)
        share = float((weight > 0.5).mean())
        material = hf.picture_material(photo_paint.encode_png(result), name="HeroPart")
        return {**part, "material": material}, share

    for t in targets:
        t["placed_clean"], share = paint(t, t["placed_clean"], "clean")
        t["report"]["painted_by_picture"] = round(share, 3)
        run.say(f"paint: {t['name']}: {share:.0%} of its texels from the picture")

    def write(model: Path, parts, new, target_path: Path):
        written = [hh.finish_part(p) for p in parts] + [hh.finish_part(p, smooth=not args.hard) for p in new]
        target_path.write_bytes(hh.regraft_glb(model.read_bytes(), written))

    result = out / f"{args.model.stem}_{name}.glb"
    write(args.model, cut_main, [t["placed_clean"] for t in targets], result)
    run.say(f"graft: {result.name}: {len(targets)} part(s)")
    raw_painted = {}
    for extra in args.also:
        target_path = out / f"{extra.stem}_{name}.glb"
        if not run.step(f"graft {extra.name}", target_path):
            continue
        _, _, extra_parts = hf.read_parts(extra.read_bytes())
        extra_bones = [bones(p) for p in extra_parts]
        cut, new, _ = graft(extra_parts, extra_bones, "raw", None)
        for index, (t, part) in enumerate(zip(targets, new)):
            if index not in raw_painted:
                raw_painted[index], share = paint(t, part, "raw")
            new[index] = {**part, "material": raw_painted[index]["material"]}
        write(extra, cut, new, target_path)
        run.say(f"graft: {target_path.name}")
        del extra_parts, cut, new

    # 9. check
    stage(run, "check", "Counting the parts' pieces and open edges")
    fresh(run, "check")
    summary = {"model": str(args.model), "picture": str(args.picture), "result": str(result), "name": name,
               "views": None if views_dir is None else str(views_dir), "hard": args.hard, "parts": {}}
    if args.boots and args.top == "auto":
        summary["boot_top"] = boot_top
    for t in targets:
        part = t["placed_clean"]
        lo, hi = part["positions"].min(axis=0), part["positions"].max(axis=0)
        pad = 0.05 * float(np.ptp(part["positions"], axis=0).max())
        near = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, lo[2] - pad, hi[2] + pad)
        before = hd.open_edges(figure, near)
        after = hd.open_edges(cut_main + [x["placed_clean"] for x in targets], near)
        entry = {**t["report"], **graft_reports[t["name"]], "pieces": hp.pieces(part),
                 "edges_where_it_touches_itself": hp.edge_faults(part)["shared_by_more"],
                 "triangles": int(len(part["faces"])), "open_edges_near_before": before,
                 "open_edges_near_after": after, "built_from": str(built[t["source"]]["clean_path"]),
                 "build_touches_itself": built[t["source"]]["build_touches_itself"],
                 "one_closed_skin": built[t["source"]]["solid"], "mirrored": t["mirror"]}
        if args.boots:
            entry["sole_below_picture_feet_px"] = round(hp.sole_below_feet(view, part["positions"], mesh_scale), 2)
            if entry["sole_below_picture_feet_px"] > SOLE_LIMIT_PX:
                run.flag(f"the new {t['side']} boot's sole lies {entry['sole_below_picture_feet_px']:.1f} px below "
                         "the picture's feet: a made-up sole")
        if after > before:
            run.flag(f"{t['name']}: {after - before} more edges stay open round it than before")
        if entry["pieces"] != 1:
            run.flag(f"{t['name']} is {entry['pieces']} pieces")
        summary["parts"][t["name"]] = entry
        run.say(f"check: {t['name']}: {entry}")
    summary["unsure"] = run.unsure
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    record = read_model_record(args.model)
    upscaled = {"license": {"name": "BSD-3-Clause"}, "classification": "commercial-clear", "components": LICENCES}
    write_derived_sidecar(args.model, result, operation="hero_parts", source_record=record,
                          component_records=[*records, upscaled])
    for extra in args.also:
        target_path = out / f"{extra.stem}_{name}.glb"
        write_derived_sidecar(extra, target_path, operation="hero_parts", component_records=[*records, upscaled])
    run.say(f"done: {result}; numbers in {out / (name + '.json')}")
    for note in run.unsure:
        run.say(f"  to check: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
