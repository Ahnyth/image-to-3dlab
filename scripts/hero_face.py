#!/usr/bin/env python3
"""Give a figure a sharper face: the face built again on its own and set in between its hat brim and its collar.

    .venv/bin/python scripts/hero_face.py model.glb front.png [--out output/hero_face/NAME] [--rig rigged.glb]
        [--also finer.glb ...] [--seed 42] [--face-faces 20000]

A face that is a few dozen pixels of a full-body picture gets few of Pixal3D's voxels: a hair strand that is thin in
the picture comes out fat and in front of an eye, and Pixel Match paints the eye's glow onto it. Built on its own
from an upscaled crop, the face fills Pixal3D's whole box and keeps both eyes free. The Hero route swaps the whole
head at the neck, which a big hat or a high collar does not survive; this swaps only the face:

  1. measure  the rig's head bones say where the head is; the figure's front depth map shows how far the face
              reaches from the nose before something stands in front of it (a brim above, a collar below) or it
              turns away (its sides, the chin). The seam runs a little inside that, through skin and hair.
              window.png shows it.
  2. crop     the face with the hair, brim and collar round it, from the picture, upscaled 4x with Real-ESRGAN.
  3. build    Pixal3D on the crop (--seed).
  4. clean    Pixal3D's own clean-up of that build at --face-faces.
  5. paint    Pixel Match of the clean and of the raw build from the crop, so the eyes are the picture's.
  6. place    the build put where the crop came from, then lined up on the figure's face (turns, shifts and a
              little size), on the closest three quarters of the points: the old strand is no match.
  7. graft    the figure cut along the seam: what lies inside it and not clearly behind the new face goes (a strand
              in front of an eye, the eye under it, an inner skin); hat, collar and the hidden rest stay. The new
              face, cut along the same seam, is eased in depth to meet the figure's rim and stitched to it as one
              surface. A rigged figure keeps its skeleton; the new face takes the old face's skin weights. Every
              --also model (the finer ones a game-ready step bakes paint and bumps from, in the same frame) gets the
              same face, from the raw build.

Writes <out>/<model>_face.glb (and <out>/<also>_face.glb), with face.json (every number), window.png and a licence
record. Every step skips when its output exists, so a rerun picks up after a fix. Where the code is unsure the log
says UNSURE: that is where a human should look.
"""
from __future__ import annotations

import argparse
import json
import sys
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
PAINT = REPO / "scripts" / "photo_paint.py"
SEED = 42
FACE_FACES = 20000          # Pixal3D's clean-up of the whole build; the face inside the seam keeps about a fifth
CROP_SIZE = 1024
LICENCES = [{"component": "Real-ESRGAN x4plus", "license": "BSD-3-Clause", "purpose": "face crop upscale"}]
FIT_LIMIT = 0.02            # a line-up whose kept pairs lie further apart than this share of the head's width
                            # put the face somewhere else


def fresh(run: Run, name: str) -> None:
    """Start a step that always runs (it is quick): its notes from an earlier run go, so the closing list says only
    what this run found."""
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


def head_of(parts: list[dict], rig_parts: list[dict] | None):
    """(index of the part holding the head, which of its vertices are head) from the rig's head bones: the model's
    own skin weights, or the nearest vertex of a separate rig's skinned part."""
    import numpy as np

    from image_to_3dlab import hero

    skinned = [p for p in (rig_parts or parts) if p.get("dense") is not None]
    if not skinned:
        raise SystemExit("[hero] the model has no skin: give the figure's rig with --rig")
    rig_points = np.concatenate([p["positions"] for p in skinned])
    rig_bone = np.concatenate([p["dense"].argmax(axis=1) for p in skinned])
    picked = hero.head_bones(rig_bone, rig_points[:, 1])
    best, best_count, best_head = None, -1, None
    for index, part in enumerate(parts):
        bone = (part["dense"].argmax(axis=1) if part.get("dense") is not None and rig_parts is None
                else hero.regions(part["positions"], rig_points, rig_bone))
        head = np.isin(bone, picked)
        if head.sum() > best_count:
            best, best_count, best_head = index, int(head.sum()), head
    return best, best_head, picked


def build_face(run: Run, cutout: Path, raw: Path, seed: int) -> None:
    if run.step(f"build: Pixal3D on the face, seed {seed}", raw):
        run.run([PYTHON, "-u", GENERATE, cutout, raw, "--seed", str(seed)])


def clean_face(run: Run, raw: Path, clean: Path, faces: int) -> None:
    """Pixal3D's own clean-up of the build: a light mesh with one painted atlas, in the build's own frame."""
    from image_to_3dlab import pixal3d_native as native

    if run.step(f"clean: {faces} faces", clean):
        replay = raw.with_name(f"{raw.stem}.replay.glb")
        run.run(native.replay_command(raw.with_suffix(native.DUMP_SUFFIX), replay, faces, 2048))
        converted = native.convert_replay_glb(replay.read_bytes())
        problem = native.frame_problem(native.glb_bounds(raw.read_bytes()), native.glb_bounds(converted))
        if problem:
            raise SystemExit(f"[hero] the face's clean-up does not sit on its build: {problem}")
        clean.write_bytes(converted)
        replay.unlink(missing_ok=True)


def paint_face(run: Run, model: Path, views: Path, out: Path) -> None:
    """Pixel Match from the crop the face was built from: the eyes and brows are the picture's own pixels."""
    if run.step(f"paint: Pixel Match of {model.name}", out):
        run.run([PYTHON, PAINT, model, views, out, "--weights", out.with_suffix(".weights.png")])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="the figure, rigged (or with --rig, any model in its frame)")
    parser.add_argument("picture", type=Path, help="the front picture it was built from")
    parser.add_argument("--out", type=Path, default=None, help="run folder (default output/hero_face/<model>)")
    parser.add_argument("--rig", type=Path, default=None, help="the figure's rig, when MODEL is not rigged")
    parser.add_argument("--also", type=Path, nargs="*", default=[],
                        help="more models of the figure in the same frame (the finer ones a game-ready step bakes "
                             "paint and bumps from): each gets the same face")
    parser.add_argument("--seed", type=int, default=SEED, help="Pixal3D's seed for the face")
    parser.add_argument("--face-faces", type=int, default=FACE_FACES, help="the face build's clean-up size")
    args = parser.parse_args(argv)

    missing = [name for name, path in needed_parts().items() if not Path(path).exists()]
    if missing:
        print(f"[hero] cannot start, not installed: {', '.join(missing)}. Install them from the Setup page "
              "(nothing was downloaded).", file=sys.stderr)
        return 2

    import numpy as np
    from PIL import Image

    from image_to_3dlab import hero, matte
    from image_to_3dlab import hero_face as hf
    from image_to_3dlab import hero_hands as hh
    from image_to_3dlab.provenance import read_model_record, write_derived_sidecar

    out = (args.out or REPO / "output" / "hero_face" / args.model.stem[:60]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    run = Run(out)
    run.say(f"hero face: {args.model} + {args.picture} -> {out}")

    # 1. measure
    stage(run, "measure", "Finding the face")
    fresh(run, "measure")
    _, _, parts = hf.read_parts(args.model.read_bytes())
    rig_parts = hf.read_parts(args.rig.read_bytes())[2] if args.rig else None
    index, head, picked = head_of(parts, rig_parts)
    body = parts[index]
    window = hf.face_window(body["positions"], body["faces"], head)
    hf.window_picture(window).save(out / "window.png")
    bounds = hf.window_bounds(window)
    run.say(f"measure: head bones {picked}, head {window['width']:.4f} wide, brim underside "
            f"{'none' if window['floor'] is None else round(window['floor'], 4)}, seam inside x {bounds[0]:.4f}.."
            f"{bounds[1]:.4f}, y {bounds[2]:.4f}..{bounds[3]:.4f} (window.png)")
    if window["floor"] is None:
        run.flag("no hat brim found above the face: the seam's top follows the hairline's reach; check window.png")

    # 2. crop
    stage(run, "crop", "Cropping and upscaling the face")
    cutout = out / "picture_cutout.png"
    if run.step("cut-out", cutout):
        matte.cut_out(Image.open(args.picture).convert("RGB"))[0].save(cutout)
    picture = hero.load_picture(cutout)
    every = np.concatenate([p["positions"] for p in parts])
    full_frame = hero.frame(every, hero.picture_box(picture))
    corners = np.array([[bounds[0], bounds[2], 0.0], [bounds[1], bounds[3], 0.0]])
    px, py = hero.project(corners, full_frame)
    box = hf.crop_box(px, py, picture.width, picture.height)
    run.say(f"crop: {box} ({box[2] - box[0]} px)")
    if box[2] - box[0] < 64:
        run.flag(f"the face's crop is only {box[2] - box[0]} px: it holds little detail")
    crop, up, face_cutout = out / "face_crop.png", out / "face_up.png", out / "face_cutout.png"
    if run.step("crop + Real-ESRGAN", up):
        Image.open(args.picture).convert("RGB").crop(box).save(crop)
        run.run([PYTHON, UPSCALE, crop, up, "--size", str(CROP_SIZE)])
    if run.step("crop cut-out", face_cutout):
        upscaled = Image.open(up).convert("RGB")
        solid, filled = hf.solid_cutout(np.asarray(upscaled), np.asarray(matte.cut_out(upscaled)[0].convert("RGBA")))
        Image.fromarray(solid, "RGBA").save(face_cutout)
        if filled:
            run.say(f"crop cut-out: {filled} px inside the face that the background remover had cut away were "
                    "put back (glowing eyes look like background to it)")

    # 3.-5. build, clean, paint
    stage(run, "build", "Building the face on its own (the long step)")
    raw = out / f"face_s{args.seed}.glb"
    build_face(run, face_cutout, raw, args.seed)
    stage(run, "clean", "Cleaning up the face")
    clean = out / f"face_s{args.seed}_clean{args.face_faces // 1000}k.glb"
    clean_face(run, raw, clean, args.face_faces)
    stage(run, "paint", "Painting the face from the crop")
    views = raw.with_suffix(".svviews")
    clean_pm, raw_pm = clean.with_name(clean.stem + "_pm.glb"), raw.with_name(raw.stem + "_pm.glb")
    paint_face(run, clean, views, clean_pm)
    if args.also:
        paint_face(run, raw, views, raw_pm)

    # 6. place
    stage(run, "place", "Lining the face up")
    fresh(run, "place")
    _, _, (clean_part,) = hf.read_parts(clean_pm.read_bytes())
    face_picture = hero.load_picture(face_cutout)
    tf = hero.crop_to_full(full_frame, hero.frame(clean_part["positions"], hero.picture_box(face_picture)), box,
                           face_picture.width)
    first = hf.place_face(clean_part, tf)
    new, _ = hf.cut_face(first, window)
    seen = hf.sample(window["depth"], window["box"], window["pixel"], body["positions"], outside=-np.inf,
                     nearest=True)
    on_face = ((hf.window_values(window, body["positions"]) > 0)
               & (body["positions"][:, 2] > seen - hf.BEHIND_SHARE * window["width"]))
    normals = body.get("normals")
    if normals is None:
        from image_to_3dlab.pixal3d_native import smooth_normals

        normals = smooth_normals(body["positions"], body["faces"])
    fine, cost = hf.trimmed_icp(new["positions"], body["positions"][on_face], normals[on_face])
    tf = fine @ tf
    scale = float(np.cbrt(np.linalg.det(tf[:3, :3])))
    moved_by = float(np.linalg.norm(hh.moved(fine, new["positions"]) - new["positions"], axis=1).mean())
    run.say(f"place: lined up on {int(on_face.sum())} points of the figure's face: moved {moved_by:.5f} on average, "
            f"kept pairs {cost:.5f} apart ({cost / window['width']:.1%} of the head's width), size x{scale:.3f}")
    if cost > FIT_LIMIT * window["width"]:
        run.flag(f"the face lines up badly ({cost / window['width']:.1%} of the head's width apart): check the crop")

    # 7. graft
    stage(run, "graft", "Setting the face in")
    fresh(run, "graft")
    record = read_model_record(args.model)
    upscaled = {"license": {"name": "BSD-3-Clause"}, "classification": "commercial-clear", "components": LICENCES}
    face_record = read_model_record(raw)
    summary = {"model": str(args.model), "picture": str(args.picture), "crop": list(box),
               "window": {"bounds": [round(b, 5) for b in bounds], "head_width": window["width"],
                          "brim_underside": window["floor"], "seed_pixel": list(window["seed"])},
               "placement": {"pairs_apart": cost, "scale": scale, "transform": tf.tolist()}, "models": {}}
    for model, source in [(args.model, clean_pm)] + [(extra, raw_pm) for extra in args.also]:
        target_path = out / f"{model.stem}_face.glb"
        data = model.read_bytes()
        _, _, model_parts = hf.read_parts(data)
        _, _, (face_part,) = hf.read_parts(source.read_bytes())
        cut_parts, face_out, report = hf.graft_face(model_parts, window, hf.place_face(face_part, tf))
        written = [hh.finish_part(p) for p in cut_parts] + [hh.finish_part(face_out, smooth=True)]
        target_path.write_bytes(hh.regraft_glb(data, written))
        run.say(f"graft: {target_path.name}: {report}")
        summary["models"][model.name] = {"result": str(target_path), **report}
        if report["rims_turn_alike"]:
            run.flag(f"{model.name}: at the seam one side faces inward: the stitch is turned wrongly")
        if report["ease_max"] > hf.BEHIND_SHARE * window["width"]:
            run.flag(f"{model.name}: the new face had to be eased {report['ease_max']:.4f} to meet the figure's rim "
                     "somewhere: look at the seam")
        write_derived_sidecar(model, target_path, operation="hero_face",
                              source_record=record if model == args.model else None,
                              component_records=[r for r in (face_record, upscaled) if r])
    summary["unsure"] = run.unsure
    (out / "face.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    run.say(f"done: {out / (args.model.stem + '_face.glb')}; numbers in {out / 'face.json'}")
    for note in run.unsure:
        run.say(f"  to check: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
