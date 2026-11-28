#!/usr/bin/env python3
"""Give a rigged figure hands with real fingers: each hand built on its own and grafted on at the wrist.

    .venv/bin/python scripts/hero_hands.py rigged.glb front.png [--out output/hero_hands/NAME]
        [--also finer.glb ...] [--left mirror|own] [--tries 8] [--keep 2] [--seeds 42 7]
        [--hand-faces 20000] [--hand-detail 1] [--hand-paint match|keep] [--colour-miss paint|keep]
        [--look "black fingerless gloves, black cuffs with teal stripes"]

A hand that is a few dozen pixels of a full-body picture comes out of Pixal3D as a mitten. Built on its own it
fills Pixal3D's whole box and keeps its fingers, but only if its picture shows them: in an A-pose picture a hand
hangs beside the thigh, seen from the thumb's side with the fingers stacked, and Pixal3D guesses their number
(six, on the first try). So the hand is redrawn first:

  1. measure  the rig finds both arms by the tree's shape; along each forearm the narrowest section between
              the palm and a cuff (or the forearm) is the wrist. No names, no hand-drawn box.
  2. crop     the hand and its cuff cut from the picture, upscaled 4x with Real-ESRGAN; the picture's hand (the
              figure's old hand past the wrist, projected into the crop) is measured: its median colour, skin or not.
  3. draw     the picture editor (FLUX.2 klein, edit mode) redraws the crop as the same hand seen flat from its
              back, fingers down and apart (a hand that is not skin is asked for as a glove of its colour);
              --tries drawings, kept only when their outline counts five digits and one thumb and their hand's
              colour lies near the picture's (the editor drew the Arkan-Duellant's black glove as a bare hand),
              up to --keep of them. When none keeps the colour, the nearest five-digit drawing is built and the
              hands painted from the picture's hand (--colour-miss paint: on the Arkan-Duellant five separate black
              fingers, where its own hands are a creased mitten), or the figure keeps its own hands (--colour-miss
              keep: exit code 3 and the reason in hands.json).
  4. build    Pixal3D on each kept drawing, --seeds each, kept only when the built hand's outline, seen from
              twelve sides, counts five digits and one thumb, its palm is solid (an armoured hand can come
              out as a plate round an empty inside) and its wrist closes where the figure's cut falls (not open
              plates); the first one that does is the hand.
  5. clean    Pixal3D's own clean-up of that build at --hand-faces: the hand's wrist and size are measured on it.
  6. graft    each arm cut straight through below the cuff, the hand (the left one mirrored from the right, or
              built on its own with --left own) put on at the old hand's place, length and turn, its outer skin
              kept, eased into the wrist and stitched to it as one surface. The figure keeps its skeleton and
              every body weight; the new hand takes the old hand's. Every --also model (the finer models a
              game-ready step bakes paint and bumps from, in the same frame) gets the same hands. Each model's
              hands are as fine as the model round its wrists (--hand-detail; Pixal3D's own clean-up at that
              size), so a game-ready step thins them with the rest of the figure. Their paint is brought to the
              tone of their drawing (--hand-paint match): the clean-up paints a built hand darker and far more
              saturated than its drawing.
  7. check    each grafted hand's digits counted again, on the figure.

Writes <out>/<model>_hands.glb (and <out>/<also>_hands.glb), with hands.json (every number) and a licence
record. Every step skips when its output exists, so a rerun picks up after a fix. Where the code is unsure the
log says UNSURE: that is where a human should look.
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

PYTHON = Path(sys.executable)
GENERATE = REPO / "scripts" / "pixal3d_generate.py"
UPSCALE = REPO / "scripts" / "upscale_esrgan.py"
TRIES, KEEP, SEEDS = 8, 2, (42, 7)   # round 22's test: 2 of 8 drawings had five digits, both built right
HAND_FACES = 20000                   # Pixal3D's clean-up of the whole build; the hand below the cuff keeps ~a quarter
HAND_DETAIL = 1.0                    # each model's new hands as fine as the figure round its wrists
DRAW_SIZE = 1024
LICENCES = [{"component": "Real-ESRGAN x4plus", "license": "BSD-3-Clause", "purpose": "hand crop upscale"}]


def fresh(run: Run, name: str) -> None:
    """Start a step that always runs (it is quick): its notes from an earlier run go, as `Run.step` does for a step
    it runs again, so the closing list says only what this run found."""
    run.current = name
    if run.notes.pop(name, None) is not None:
        run.unsure_file.write_text(json.dumps(run.notes, indent=2), encoding="utf-8")


def needed_parts() -> dict[str, Path]:
    """Everything the steps run, by what to install to get it."""
    import pixal3d_generate

    from image_to_3dlab import pixal3d_native, upscale

    return {"Pixal3D (3D models)": pixal3d_generate.CLI,
            "Pixal3D's own clean-up (Setup: Rebuild Pixal3D)": pixal3d_native.POST_REPLAY,
            "Hero face upscaler (Real-ESRGAN, 67 MB)": upscale.WEIGHTS_FILE}


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


def mask_of(picture: Path):
    """The subject's mask: its own transparency, else the background remover's cut-out."""
    import numpy as np
    from PIL import Image

    from image_to_3dlab import matte

    with Image.open(picture) as opened:
        cut = opened.convert("RGBA") if matte.is_matted(opened) else matte.cut_out(opened.convert("RGB"))[0]
    return np.asarray(cut)[..., 3] > 127


SKIPPED = 3     # the exit code when the figure keeps its own hands (as hero_parts' without boots): the chain skips


def picture_hand(run: Run, side: str, crop: Path, box, cutout: Path, px, py, folder: Path) -> dict | None:
    """The picture's hand colour (`hh.hand_colour`): the crop's pixels inside the figure's old hand past the wrist,
    projected (PX, PY in the picture's pixels), where the cut-out shows the subject. Written beside the crop:
    <side>_hand.png (the crop darkened outside the mask, for a look) and <side>_crop_hand.png with its cut-out
    <side>_crop_hand__matted.png, which `match_paint` takes its tones from with --colour-miss paint."""
    import numpy as np
    from PIL import Image

    from image_to_3dlab import hero_hands as hh

    with Image.open(cutout) as opened:
        subject = np.asarray(opened.convert("RGBA").crop(box))[..., 3] > 127
    with Image.open(crop) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    mask = hh.picture_hand_mask(np.asarray(px) - box[0], np.asarray(py) - box[1], subject)
    Image.fromarray(np.where(mask[..., None], rgb, rgb // 3).astype(np.uint8)).save(folder / f"{side}_hand.png")
    Image.fromarray(rgb).save(folder / f"{side}_crop_hand.png")
    Image.fromarray(np.dstack([rgb, np.where(mask, 255, 0).astype(np.uint8)])).save(
        folder / f"{side}_crop_hand__matted.png")
    colour = hh.hand_colour(rgb[mask])
    if colour is None:
        run.say(f"colour: the {side} hand shows {int(mask.sum())} pixels in the picture, too few to measure; its "
                "drawings are kept by their fingers alone")
    else:
        run.say(f"colour: the picture's {side} hand ({hh.hand_words(colour)}): L*a*b* median {colour['lab']}, "
                f"{colour['skin']:.0%} skin-toned, {colour['pixels']} pixels")
    return colour


def stale(drawing: Path, prompt: str) -> bool:
    """Whether DRAWING was drawn with another prompt (its licence record says which): a rerun after the prompt
    changed (a glove asked for) must not keep the bare hands drawn before."""
    record = drawing.with_suffix(".provenance.json")
    if not drawing.exists() or not record.is_file():
        return False
    return json.loads(record.read_text(encoding="utf-8")).get("prompt") != prompt


def draw_hands(run: Run, side: str, up: Path, folder: Path, args, engine,
               picture: dict | None = None) -> tuple[list[Path], list[Path]]:
    """Up to --keep drawings of the hand seen flat from its back, each with five digits and one thumb and a hand
    whose colour lies near PICTURE's (`hh.colour_check`); and the five-digit ones whose colour missed, nearest first."""
    import image_api
    import numpy as np
    from PIL import Image

    from image_to_3dlab import back_view
    from image_to_3dlab import hero_hands as hh
    from image_to_3dlab.provenance import write_provenance

    reference = folder / f"{side}_reference.png"
    if not reference.exists():
        with Image.open(up) as opened:
            back_view.reference_on_grey(opened.convert("RGBA")).save(reference)
    weights, vision = image_api.resolve_weights(engine), image_api.resolve_vision(engine)
    glove = None if args.look else hh.glove_of(picture)
    prompt = hh.hand_prompt(side, args.look, glove)
    if glove:
        run.say(f"draw: the picture's {side} hand wears a {glove} glove: the drawings are asked to keep it")
    kept, missed = [], []
    for seed in range(1, args.tries + 1):
        drawing = folder / f"{side}_draw_s{seed}.png"
        counted = drawing.with_suffix(".count.json")
        if stale(drawing, prompt):            # drawn with another prompt: it, its count and its builds go
            for old in [*folder.glob(f"{drawing.stem}.*"), *folder.glob(f"{drawing.stem}_*")]:
                old.unlink()
            run.say(f"draw: {drawing.name} was drawn with another prompt; it is drawn again")
        if not drawing.exists():
            settings = image_api.clean_settings({"seed": seed, "width": DRAW_SIZE, "height": DRAW_SIZE},
                                                engine=engine)
            started = time.time()
            command = image_api.build_command(prompt, settings, drawing, weights, reference=reference,
                                              vision=vision, engine=engine)
            run.say(f"draw: {side} hand, seed {seed}")
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                    cwd=str(REPO), check=False)
            if result.returncode or not drawing.exists():
                run.flag(f"the picture editor failed on seed {seed}: {(result.stdout + result.stderr)[-300:]}")
                continue
            record = image_api.provenance(prompt, settings, time.time() - started, drawing, engine)
            record["reference"] = {"picture": str(up), "mode": "edit (hand seen flat from its back)"}
            write_provenance(drawing.with_suffix(".provenance.json"), record)
        count = json.loads(counted.read_text(encoding="utf-8")) if counted.exists() else {}
        if count.get("rule") != "cleft" or "colour" not in count:   # a count from before the thumb was told by its
            mask = mask_of(drawing)                                  # cleft, or before the colour was measured
            if count.get("rule") != "cleft":
                # the prompt puts a right hand's thumb on the picture's right, a left hand's on its left
                count = {**hh.digit_count(mask, thumb_side=side), "rule": "cleft"}
            with Image.open(drawing) as opened:
                rgb = np.asarray(opened.convert("RGB"))
            count["colour"] = hh.hand_colour(rgb[hh.drawing_hand_mask(mask)])
            counted.write_text(json.dumps(count), encoding="utf-8")
        colour = hh.colour_check(picture, count["colour"])
        ok = count["ok"] and colour["ok"]
        run.say(f"draw: {drawing.name} {count['digits']} digits, {count['thumbs']} thumb(s); {colour['why']}"
                + (" - kept" if ok else ""))
        if ok:
            kept.append(drawing)
            if len(kept) >= args.keep:
                break
        elif count["ok"]:
            missed.append((colour["de"], drawing))
    return kept, [drawing for _, drawing in sorted(missed)]


def build_count(raw: Path) -> dict:
    """A build's digits seen all round (`count_built`) and the share of it hollow below the wrist (`hand_wrist`),
    kept beside it as <build>.count.json; "kept" when it counts five digits and one thumb and its palm is solid.
    A count kept from before the hollow check gets it added."""
    from image_to_3dlab import hero_hands as hh

    counted = raw.with_suffix(".count.json")
    count = json.loads(counted.read_text(encoding="utf-8")) if counted.exists() else {}
    if "digits" not in count or "hollow" not in count:
        _, _, part = hh.read_part(raw.read_bytes())
        if "digits" not in count:
            count = hh.count_built(part["positions"], part["faces"], hh.DRAWN_DOWN)
        count["hollow"] = hh.hand_wrist(part["positions"], part["faces"])["hollow"]
        counted.write_text(json.dumps(count), encoding="utf-8")
    return {**count, "kept": bool(count["ok"]) and count["hollow"] <= hh.HOLLOW_LIMIT}


def build_hand(run: Run, side: str, drawings: list[Path], folder: Path, args, fits=None) -> Path | None:
    """The first Pixal3D build of a kept drawing whose outline counts five digits and one thumb, whose palm is not
    a hole (Pixal3D built the Waechter's first gauntlet as a plate round an empty inside) and, with FITS (a build ->
    the share of its cross-section the stitch would close at the figure's cut, `rim_share`), whose wrist closes."""
    from image_to_3dlab import hero_hands as hh

    for drawing in drawings:
        for seed in args.seeds:
            raw = folder / f"{drawing.stem}_b{seed}.glb"
            if not raw.exists():
                run.say(f"build: Pixal3D on {drawing.name}, seed {seed}")
                if not run.run([PYTHON, "-u", GENERATE, drawing, raw, "--seed", str(seed)], required=False):
                    run.flag(f"Pixal3D failed on {drawing.name} seed {seed} (see run.log)")
                    continue
            count = build_count(raw)
            note = (f", {count['hollow']:.0%} of it hollow below the wrist (a hole for a palm)"
                    if not count["kept"] and count["ok"] else "")
            if count["kept"] and fits is not None:
                closed = fits(raw)
                if closed < hh.RIM_LIMIT:
                    count["kept"] = False
                    note = f", its cut at the figure's wrist closes only {closed:.0%} of it (open plates there)"
            run.say(f"build: {raw.name} {count['digits']} digits, {count['thumbs']} thumb(s), "
                    f"per view {count['per_view']}{note}" + (" - kept" if count["kept"] else ""))
            if count["kept"]:
                return raw
    return None


def clean_hand(run: Run, raw: Path, faces: int) -> Path:
    """Pixal3D's own clean-up of the build: a light mesh with one painted atlas, in the build's own frame. A
    clean-up whose bounds stray from the build's (`frame_problem`) is made again finer, doubling up to HAND_FACES: the
    Arkan-Duellant's glove build at 3,000 faces sat 3.7 % off (allowed 3 %), its thin fingers shrunk by the decimation
    (round 32). Off at every size, it is a frame fault and the stage stops."""
    from image_to_3dlab import pixal3d_native as native

    sizes = [faces]
    while sizes[-1] < HAND_FACES:
        sizes.append(min(sizes[-1] * 2, HAND_FACES))
    for size in sizes:
        clean = raw.with_name(f"{raw.stem}_clean{size // 1000}k.glb" if size % 1000 == 0
                              else f"{raw.stem}_clean{size}.glb")
        if clean.exists():
            return clean
        replay = raw.with_name(f"{raw.stem}.replay.glb")
        run.run(native.replay_command(raw.with_suffix(native.DUMP_SUFFIX), replay, size, 2048))
        converted = native.convert_replay_glb(replay.read_bytes())
        problem = native.frame_problem(native.glb_bounds(raw.read_bytes()), native.glb_bounds(converted))
        replay.unlink(missing_ok=True)
        if not problem:
            clean.write_bytes(converted)
            return clean
        run.say(f"clean: {raw.name} at {size} faces does not sit on its build ({problem})"
                + ("; made again finer" if size != sizes[-1] else ""))
    raise SystemExit(f"[hero] the hand's clean-up does not sit on its build: {problem}")


def place(part: dict, plan: dict):
    """(a built hand's own wrist, the transform putting it on the figure's wrist of PLAN) (`hand_wrist`,
    `placement`)."""
    from image_to_3dlab import hero_hands as hh

    own = hh.hand_wrist(part["positions"], part["faces"])
    return own, hh.placement(own["centre"], own["fingers"], own["back"], own["length"], plan["centre"],
                             plan["fingers"], plan["back"], plan["length"])


def hand_of(path: Path, mirrored: bool) -> dict:
    """A built hand as a part, mirrored into the other hand when it is to be."""
    from image_to_3dlab import hero_hands as hh

    _, _, part = hh.read_part(path.read_bytes())
    if mirrored:
        part["positions"], part["faces"], part["normals"] = hh.mirror_x(part["positions"], part["faces"],
                                                                         part.get("normals"))
    return part


def glb_faces(path: Path) -> int:
    """How many triangles a one-part GLB holds, from its document alone."""
    from image_to_3dlab.glb_turn import parse_glb

    doc, _ = parse_glb(path.read_bytes())
    primitive = doc["meshes"][0]["primitives"][0]
    return int(doc["accessors"][primitive["indices"]]["count"]) // 3


def hand_size(run: Run, hand: dict, faces: int | None, rigged: bool) -> Path:
    """The hand build at the size a model needs: Pixal3D's own clean-up at FACES (it paints its own atlas at any
    size), the raw build when that is not finer; without FACES (--hand-detail 0) as before this option: the
    reference clean-up for the rigged model, the raw build for the finer ones."""
    if faces is None:
        return hand["reference"] if rigged else hand["raw"]
    if faces >= hand.setdefault("raw_faces", glb_faces(hand["raw"])):
        return hand["raw"]
    return clean_hand(run, hand["raw"], faces)


def clean_sizes(density: float, hands: dict, detail: float) -> dict:
    """Each side's clean-up size for a model of DENSITY (`hand_faces` over the area its clean-ups spread their faces
    on), one size for every side built from one build: a left hand mirrored from the right asked for 33k faces where
    the right asked for 32k, and the build was cleaned up twice. None for each side with DETAIL 0."""
    from image_to_3dlab import hero_hands as hh

    if detail <= 0:
        return dict.fromkeys(hands)
    wanted = {side: hh.hand_faces(density, hand["area"], detail) for side, hand in hands.items()}
    return {side: max(wanted[other] for other, h in hands.items() if h["raw"] == hand["raw"])
            for side, hand in hands.items()}


def graft_one(run: Run, model: Path, out: Path, bone, plans: dict, hands: dict, detail: float, rigged: bool,
              loaded: dict, paint: str = "match"):
    """MODEL with both hands grafted on, written to OUT, each hand as fine as the model round its wrists times
    DETAIL (`wrist_density`, `clean_sizes`): (the report per side, both hands as one part, the body part, how the
    hands were sized and painted). LOADED keeps each hand build read once. With PAINT "match" the new hands' paint is
    brought to the model's own hands' (`match_paint`)."""
    from image_to_3dlab import hero_hands as hh

    data = model.read_bytes()
    _, _, part = hh.read_part(data)
    model_bone = bone(part)
    density = hh.wrist_density(part["positions"], part["faces"], model_bone, plans)
    sizes = clean_sizes(density, hands, detail)
    put, used = {}, {}
    for side, hand in hands.items():
        path = hand_size(run, hand, sizes[side], rigged)
        if (path, hand["mirrored"]) not in loaded:
            loaded[path, hand["mirrored"]] = hand_of(path, mirrored=hand["mirrored"])
        put[side] = {"part": loaded[path, hand["mirrored"]], "tf": hand["tf"]}
        used[side] = str(path)
    body, both, report = hh.graft_model(part, model_bone, plans, put)
    if paint == "match":
        first = next(iter(hands))
        figure = hh.part_paint(part, hh.old_hand_faces(part["positions"], part["faces"], model_bone, plans))
        source = hands[first].get("paint_from") or drawing_of(hands[first]["raw"])   # the picture's hand when
        toned = match_paint(run, out, put[first]["part"], source, both, figure)        # the drawing missed its colour
    else:
        toned = {"paint": "kept"}
    for side, entry in report.items():         # the build was checked at the reference size; a smaller one may differ
        if entry["rim_share"] < hh.RIM_LIMIT:
            run.flag(f"the {side} hand's stitch in {out.name} closes only {entry['rim_share']:.0%} of its wrist: "
                     "look for a hole at the cuff")
    out.write_bytes(hh.regraft_glb(data, [hh.finish_part(body), hh.finish_part(both, smooth=True)]))
    finer = len(both["faces"]) / max(hh.surface_area(both["positions"], both["faces"]), 1e-18) / max(density, 1e-18)
    run.say(f"graft: {out.name}: body {len(body['faces'])} faces, hands {len(both['faces'])} faces "
            f"({finer:.1f} times as fine as the figure round its wrists; from "
            f"{', '.join(sorted({Path(p).name for p in used.values()}))})")
    sizing = {"figure_density": round(density, 1), "hands_finer": round(finer, 2), "built_from": used, **toned}
    return report, both, body, sizing


def match_paint(run: Run, out: Path, built: dict, drawing: Path, both: dict, figure=None) -> dict:
    """Bring BOTH's paint (the grafted hands) to the tone of the DRAWING they were built from, in place: curves per
    L*a*b* tone (`tone_curves`) from the whole BUILT hand's paint (hand, cuff and sleeve, as drawn) to the drawing's.
    Pixal3D's clean-up paints a built hand darker and far more saturated than its drawing. FIGURE (the figure's own
    hands' paint, when known) is only reported. Returns the medians, for hands.json."""
    import numpy as np

    from image_to_3dlab import hero_hands as hh

    new, ref = hh.part_paint(built), hh.drawing_paint(drawing)
    if new is None or ref is None or min(len(new), len(ref)) < hh.TONE_LEAST_FACES:
        run.say(f"paint: {out.name}: the hands' paint is kept (no painted build or drawing to match it to)")
        return {"paint": "kept, nothing to match"}
    before = hh.part_paint(both)
    both["material"] = hh.retinted(both["material"], hh.tone_curves(new, ref))
    after = hh.part_paint(both)
    named = {"built": before, "matched": after, "drawing": ref}
    if figure is not None and len(figure):
        named["figure"] = figure
    medians = {name: np.round(np.median(hh.to_lab(rgb), axis=0), 1).tolist() for name, rgb in named.items()}
    tail = f", the figure's own hands {medians['figure']})" if "figure" in medians else ")"
    run.say(f"paint: {out.name}: the new hands' L*a*b* median {medians['built']} brought to {medians['matched']} "
            f"(the drawing's {medians['drawing']}{tail}")
    shift = abs(medians["matched"][0] - medians["built"][0])
    if shift > hh.TONE_SHIFT_LIMIT:
        run.flag(f"the new hands' paint in {out.name} moved {shift:.0f} L* to meet their drawing: look at the hands, "
                 "or rerun with --hand-paint keep")
    return {"paint": "matched", "paint_lab_median": medians}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="the figure, rigged (or with --rig, any model in its frame)")
    parser.add_argument("picture", type=Path, help="the front picture it was built from")
    parser.add_argument("--out", type=Path, default=None, help="run folder (default output/hero_hands/<model>)")
    parser.add_argument("--rig", type=Path, default=None, help="the figure's rig, when MODEL is not rigged")
    parser.add_argument("--also", type=Path, nargs="*", default=[],
                        help="more models of the figure in the same frame (the finer ones a game-ready step bakes "
                             "paint and bumps from): each gets the same hands")
    parser.add_argument("--left", choices=("mirror", "own"), default="mirror",
                        help="mirror: the left hand is the right one mirrored (a figure symmetric there); own: "
                             "the left hand drawn and built from its own crop")
    parser.add_argument("--tries", type=int, default=TRIES, help="drawings per hand")
    parser.add_argument("--keep", type=int, default=KEEP, help="drawings with five digits to build from")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="Pixal3D seeds per drawing")
    parser.add_argument("--hand-faces", type=int, default=HAND_FACES,
                        help="the hand build's clean-up the new hand is measured on (with --hand-detail 0 also the "
                             "rigged model's hand)")
    parser.add_argument("--hand-detail", type=float, default=HAND_DETAIL,
                        help="how fine the new hands are against the figure round their wrists: 1 as fine, so a "
                             "game-ready step thins them with the rest; 2 twice as fine; 0 as built")
    parser.add_argument("--hand-paint", choices=("match", "keep"), default="match",
                        help="match: the new hands' paint brought to the tone of the drawing they were built from "
                             "(Pixal3D's clean-up paints a built hand darker and more saturated); keep: as built")
    parser.add_argument("--colour-miss", choices=("paint", "keep"), default="paint",
                        help="when no drawing keeps the picture's hand colour (a glove drawn bare): paint, the nearest "
                             "five-digit drawing is built and the new hands painted from the picture's hand; keep, the "
                             "figure keeps its own hands (exit code 3)")
    parser.add_argument("--look", default=None,
                        help="what the hand wears, in words, for the drawing (default: as in the picture)")
    args = parser.parse_args(argv)

    missing = [name for name, path in needed_parts().items() if not Path(path).exists()]
    engine, problem = picture_engine()
    reasons = ([f"not installed: {', '.join(missing)}. Install them from the Setup page (nothing was downloaded)"]
               if missing else []) + ([problem] if problem else [])
    if reasons:
        print("[hero] cannot start: " + "; ".join(reasons), file=sys.stderr)
        return 2

    import numpy as np
    from PIL import Image
    from rig_check import skeleton

    from image_to_3dlab import hero, matte
    from image_to_3dlab import hero_hands as hh
    from image_to_3dlab.provenance import read_model_record, write_derived_sidecar

    out = (args.out or REPO / "output" / "hero_hands" / args.model.stem[:60]).resolve()
    folder = out / "hands"
    folder.mkdir(parents=True, exist_ok=True)
    run = Run(out)
    run.say(f"hero hands: {args.model} + {args.picture} -> {out}")

    # 1. measure
    stage(run, "measure", "Finding the wrists")
    fresh(run, "measure")
    rig = args.rig or args.model
    parents, heads = skeleton(rig)
    rig_doc, _, rig_part = hh.read_part(rig.read_bytes())
    if rig_part.get("dense") is None:
        raise SystemExit(f"[hero] {rig.name} has no skin: give the figure's rig with --rig")
    joint_index = {rig_doc["nodes"][j].get("name") or f"joint{j}": i
                   for i, j in enumerate(rig_doc["skins"][0]["joints"])}
    rig_bone = rig_part["dense"].argmax(axis=1)

    def bone(part):
        if part.get("dense") is not None:
            return part["dense"].argmax(axis=1)
        return hero.regions(part["positions"], rig_part["positions"], rig_bone)

    _, _, figure = hh.read_part(args.model.read_bytes())
    figure_bone = bone(figure)
    arms, frame = hh.arm_bones(parents, heads)
    if set(arms) != set(hh.SIDES):
        raise SystemExit(f"[hero] the rig shows {sorted(arms) or 'no'} arm(s) with a hand; both are needed")
    plans = {}
    for side, arm in arms.items():
        plans[side] = hh.figure_wrist(figure["positions"], figure["faces"], figure_bone, joint_index[arm["forearm"]],
                                      [joint_index[b] for b in [arm["hand"], *arm["below"]]],
                                      heads[arm["forearm"]], heads[arm["hand"]], side, frame[:, 2])
        plan = plans[side]
        run.say(f"measure: {side} wrist at {np.round(plan['cut'], 4).tolist()}, hand {plan['length']:.4f} long, "
                f"palm {plan['turn_check_deg']:.0f} deg from the usual turn")
        if not plan["found"]:
            run.flag(f"no wrist showed on the {side} arm's profile; it is cut halfway along the hand bone")
        if plan["turn_check_deg"] > hh.TURN_LIMIT_DEG:
            run.flag(f"the {side} hand's palm faces {plan['turn_check_deg']:.0f} deg away from the thigh: the new "
                     "hand turns its thumb forward as an A-pose hand does; check its turn")

    # 2.-5. crop, draw, build, clean
    sources = ["right"] if args.left == "mirror" else ["right", "left"]
    cutout = out / "picture_cutout.png"
    if run.step("cut-out", cutout):
        matte.cut_out(Image.open(args.picture).convert("RGB"))[0].save(cutout)
    picture = hero.load_picture(cutout)
    full_frame = hero.frame(figure["positions"], hero.picture_box(picture))
    built, raws, records, colours, painted = {}, {}, [], {}, {}
    for side in sources:
        stage(run, "crop", f"Cropping and upscaling the {side} hand")
        points = hh.hand_points(figure["positions"], figure_bone, plans[side])
        px, py = hero.project(points, full_frame)
        box = hh.crop_box(px, py, picture.width, picture.height)
        crop, up = folder / f"{side}_crop.png", folder / f"{side}_up.png"
        run.say(f"crop: {side} hand {box} ({box[2] - box[0]} px)")
        if run.step(f"crop {side}", up) or not crop.exists():
            Image.open(args.picture).convert("RGB").crop(box).save(crop)
        if not up.exists():
            run.run([PYTHON, UPSCALE, crop, up, "--size", str(DRAW_SIZE)])
        past = (points - plans[side]["cut"]) @ plans[side]["axis"] > 0       # the old hand below the wrist
        colours[side] = picture_hand(run, side, crop, box, cutout, px[past], py[past], folder)
        stage(run, "draw", f"Drawing the {side} hand flat, fingers apart")
        drawings, missed = draw_hands(run, side, up, folder, args, engine, colours[side])
        if not drawings and missed:
            why = (f"none of {args.tries} drawings of the {side} hand kept the picture's "
                   f"{hh.hand_words(colours[side])} (the five-digit ones drew {hh.hand_words(_drawn(missed[0]))})")
            if args.colour_miss == "keep":
                run.say(f"hands: skipped: {why}: the figure keeps its own hands")
                (out / "hands.json").write_text(json.dumps({
                    "model": str(args.model), "picture": str(args.picture), "skipped": f"{why}: the figure keeps "
                    "its own hands", "picture_hand": colours, "unsure": run.unsure}, indent=1), encoding="utf-8")
                return SKIPPED
            run.flag(f"{why}: the hands are built from {missed[0].name} and painted from the picture's hand")
            drawings, painted[side] = missed, folder / f"{side}_crop_hand.png"
        if not drawings:
            raise SystemExit(f"[hero] none of {args.tries} drawings of the {side} hand has five digits and one "
                             "thumb: try more (--tries) or describe the hand (--look)")
        stage(run, "build", f"Building the {side} hand on its own (the long step)")
        served = [side] if args.left == "own" else list(plans)       # a mirrored left hand is the right build

        def fits(raw, side=side, served=served):
            reference = clean_hand(run, raw, args.hand_faces)
            shares = []
            for target in served:
                part = hand_of(reference, mirrored=target != side)
                shares.append(hh.rim_share(part, place(part, plans[target])[1], plans[target],
                                           hh.GAP_SHARE * plans[target]["length"]))
            return min(shares)

        raw = build_hand(run, side, drawings, folder, args, fits)
        if raw is None:
            raise SystemExit(f"[hero] no build of the {side} hand counts five digits and one thumb with a solid "
                             "palm and a wrist that closes: try more seeds (--seeds) or drawings (--keep)")
        stage(run, "clean", f"Cleaning up the {side} hand")
        built[side], raws[side] = clean_hand(run, raw, args.hand_faces), raw
        records += [r for r in (read_model_record(raw), _picture_record(raw)) if r]

    # 6. graft
    stage(run, "graft", "Putting the hands on")
    fresh(run, "graft")
    hands, loaded = {}, {}
    for side, plan in plans.items():
        source = side if side in built else sources[0]
        clean = hand_of(built[source], mirrored=source != side)
        own, tf = place(clean, plan)
        scale = float(np.cbrt(np.linalg.det(tf[:3, :3])))
        # each model's clean-up is sized by where this one's faces land: the hand below the cut, put in place
        below, _ = hh.hand_onto(clean, tf, plan, hh.GAP_SHARE * plan["length"])
        area = hh.spread_area(len(below["faces"]), hh.surface_area(below["positions"], below["faces"]),
                              len(clean["faces"]))
        hands[side] = {"tf": tf, "scale": scale, "down": hh.unit(tf[:3, :3] @ hh.DRAWN_DOWN),
                       "reference": built[source], "raw": raws[source], "mirrored": source != side, "area": area,
                       "paint_from": painted.get(source)}
        if not own["found"]:
            run.flag(f"the built {source} hand showed no wrist below its cuff; it is cut halfway")
    result = out / f"{args.model.stem}_hands.glb"
    report, both, body, sizing = graft_one(run, args.model, result, bone, plans, hands, args.hand_detail, True,
                                           loaded, args.hand_paint)
    for side, entry in report.items():
        if not hh.SCALE_LIMITS[0] <= entry["wrist_ratio"] <= hh.SCALE_LIMITS[1]:
            run.flag(f"the new {side} wrist had to be eased {entry['wrist_ratio']}x to meet the arm: check its size")
        if entry["rim_pieces"] > 1 and entry["rim_rest_share"] > 0.05:
            run.flag(f"the {side} cut left {entry['rim_pieces']} pieces ({entry['rim_rest_share']:.0%} of the rim "
                     "beside the wrist): look for holes at the cuff")
        if entry["rims_turn_alike"]:
            run.flag(f"at the {side} wrist one part faces inward: the stitch is turned wrongly")
    also = {}
    for extra in args.also:
        target = out / f"{extra.stem}_hands.glb"
        if run.step(f"graft {extra.name}", target):
            also[extra.name] = graft_one(run, extra, target, bone, plans, hands, args.hand_detail, False, loaded,
                                         args.hand_paint)[3]

    # 7. check
    stage(run, "check", "Counting the grafted hands' fingers")
    fresh(run, "check")
    summary = {"model": str(args.model), "picture": str(args.picture), "result": str(result), "sides": {},
               "hand_detail": args.hand_detail, "sizing": {args.model.name: sizing, **also}, "picture_hand": colours,
               "painted_from_picture": sorted(painted)}
    for index, side in enumerate(plans):
        # framed as its drawing was: the hand with the figure's cuff above it
        hand = hh.merge_parts([hh.side_faces(both, both["side_of_face"], index), hh.arm_stub(body, plans[side])])
        count = hh.count_built(hand["positions"], hand["faces"], hands[side]["down"])
        run.say(f"check: grafted {side} hand {count['digits']} digits, {count['thumbs']} thumb(s), "
                f"per view {count['per_view']}")
        if not count["ok"]:
            run.flag(f"the grafted {side} hand counts {count['digits']} digits and {count['thumbs']} thumb(s)")
        plan = plans[side]
        summary["sides"][side] = {
            "count": count, "scale": round(hands[side]["scale"], 4), **report[side],
            "wrist": np.round(plan["cut"], 5).tolist(), "found": plan["found"],
            "hand_length": round(plan["length"], 5), "turn_check_deg": round(plan["turn_check_deg"], 1),
            "built_from": sizing["built_from"][side], "mirrored": side not in built}
    summary["unsure"] = run.unsure
    (out / "hands.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    record = read_model_record(args.model)
    upscaled = {"license": {"name": "BSD-3-Clause"}, "classification": "commercial-clear", "components": LICENCES}
    write_derived_sidecar(args.model, result, operation="hero_hands", source_record=record,
                          component_records=[*records, upscaled])
    for extra in args.also:
        target = out / f"{extra.stem}_hands.glb"
        write_derived_sidecar(extra, target, operation="hero_hands", component_records=[*records, upscaled])
    run.say(f"done: {result}; numbers in {out / 'hands.json'}")
    for note in run.unsure:
        run.say(f"  to check: {note}")
    return 0


def _drawn(drawing: Path) -> dict | None:
    """A drawing's hand colour as its count recorded it."""
    counted = drawing.with_suffix(".count.json")
    return json.loads(counted.read_text(encoding="utf-8")).get("colour") if counted.is_file() else None


def drawing_of(raw: Path) -> Path:
    """The drawing a hand build was made from: `<side>_draw_s<seed>.png` beside `<side>_draw_s<seed>_b<seed>.glb`."""
    return raw.with_name(f"{raw.stem.rsplit('_b', 1)[0]}.png")


def _picture_record(raw: Path) -> dict | None:
    """The licence record of the drawing a build was made from, as a component record: the classification and the
    output's path where the licence ancestry (`write_derived_sidecar`) reads them. The picture editor's own record
    keeps its output as a bare file name, and the ancestry stopped on it."""
    name = drawing_of(raw).stem
    path = raw.with_name(f"{name}.provenance.json")
    if not path.is_file():
        return None
    picture = json.loads(path.read_text(encoding="utf-8"))
    licence = picture.get("license") or {}
    classification = licence.get("classification", "unverified")
    model = picture.get("model") or {}
    return {"operation": "hand drawing", "license": licence, "classification": classification, "model": model,
            "output": {"path": str(raw.with_name(f"{name}.png")), "classification": classification},
            "components": [{"component": model.get("id", "picture editor"), "license": licence.get("name"),
                            "purpose": "the hand redrawn flat, fingers apart"}],
            "picture_record": picture}


if __name__ == "__main__":
    raise SystemExit(main())
