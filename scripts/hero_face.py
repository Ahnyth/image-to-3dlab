#!/usr/bin/env python3
"""Give a figure a sharper face: the face built again on its own and set in between its hat brim and its collar.

    .venv/bin/python scripts/hero_face.py model.glb front.png [--out output/hero_face/NAME] [--rig rigged.glb]
        [--also finer.glb ...] [--seed 42]

A face that is a few dozen pixels of a full-body picture gets few of Pixal3D's voxels: a hair strand that is thin in
the picture comes out fat and in front of an eye, and Pixel Match paints the eye's glow onto it. Built on its own
from an upscaled crop, the face fills Pixal3D's whole box and keeps both eyes free. The Hero route swaps the whole
head at the neck, which a big hat or a high collar does not survive; this swaps only the face:

  1. measure  the rig's head bones say where the head is; the figure's front depth map shows how far the face
              reaches from the nose before something stands in front of it (a brim above, a collar below) or it
              turns away (its sides, the chin). The seam runs a little inside that, through skin and hair.
              window.png shows it.
  2. crop     the face with the hair, brim and collar round it, from the picture, upscaled 4x with Real-ESRGAN;
              what the background remover cut out inside the face (glowing eyes) is put back.
  3. build    Pixal3D on the crop (--seed).
  4. place    the build's face goes through the build's own camera into the crop and the picture, and the build is
              fitted to where that puts it, then moved in depth onto the figure's face. The new face is the build's
              nearest surface seen from the front (a build can be a hollow shell: its inner wall stays out), painted
              by the crop itself, so the eyes are the picture's own pixels.
  5. graft    the figure cut along the seam: what lies inside it and not clearly behind the new face goes (a strand
              in front of an eye, the eye under it, an inner skin); hat, collar and the hidden rest stay. The new
              face reaches a few pixels past the seam and is tucked just behind the figure's surface there, so the
              figure's cut edge lies over it (Pixal3D's skin and inner wall are one surface, so neither cut gives a
              clean rim to stitch). A rigged figure keeps its skeleton; the new face takes the old face's skin
              weights. Every --also model (the finer ones a game-ready step bakes paint and bumps from, in the same
              frame) gets the same face.

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
SEED = 42
CROP_SIZE = 1024
LICENCES = [{"component": "Real-ESRGAN x4plus", "license": "BSD-3-Clause", "purpose": "face crop upscale"}]
FIT_LIMIT = 0.03            # a new face whose points lie further than this share of the head's width from the old
                            # face (the closest three quarters of them, on average) was put somewhere else


def fresh(run: Run, name: str) -> None:
    """Start a step that always runs (it is quick): its notes from an earlier run go, so the closing list says only
    what this run found."""
    run.current = name
    if run.notes.pop(name, None) is not None:
        run.unsure_file.write_text(json.dumps(run.notes, indent=2), encoding="utf-8")


def needed_parts() -> dict[str, Path]:
    """Everything the steps run, by what to install to get it."""
    import pixal3d_generate

    from image_to_3dlab import upscale

    return {"Pixal3D (3D models)": pixal3d_generate.CLI,
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


def _png(path: Path) -> bytes:
    """A picture as PNG bytes (the face's paint goes into the model as one)."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.open(path).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def build_face(run: Run, cutout: Path, raw: Path, seed: int) -> None:
    if run.step(f"build: Pixal3D on the face, seed {seed}", raw):
        run.run([PYTHON, "-u", GENERATE, cutout, raw, "--seed", str(seed)])


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
    args = parser.parse_args(argv)

    missing = [name for name, path in needed_parts().items() if not Path(path).exists()]
    if missing:
        print(f"[hero] cannot start, not installed: {', '.join(missing)}. Install them from the Setup page "
              "(nothing was downloaded).", file=sys.stderr)
        return 2

    import numpy as np
    from PIL import Image

    from image_to_3dlab import hero, matte, photo_paint
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
    for doubt in hf.window_doubts(window):
        run.flag(f"{doubt}; check window.png")

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

    # 3. build
    stage(run, "build", "Building the face on its own (the long step)")
    raw = out / f"face_s{args.seed}.glb"
    build_face(run, face_cutout, raw, args.seed)

    # 4. place
    stage(run, "place", "Putting the face where the picture has it")
    fresh(run, "place")
    _, _, (built,) = hf.read_parts(raw.read_bytes())
    face_picture = hero.load_picture(face_cutout)
    rough = hero.crop_to_full(full_frame, hero.frame(built["positions"], hero.picture_box(face_picture)), box,
                              face_picture.width)
    # The rough frame is only good enough to say where the build's face is: it takes the camera as straight-on, and
    # Pixal3D's has a 20 degree view, so a face built alone (in front of its box's middle) came out up to a
    # centimetre off. The face's front, seen from the front, goes through the build's own camera into its picture,
    # the crop and the figure's frame, and the build is fitted to where those put it.
    front = hf.face_relief(hf.place_face(built, rough), window, box, face_picture.width, full_frame)
    own = hh.moved(np.linalg.inv(rough), front["positions"][hf.window_values(window, front["positions"]) > 0])
    views, mesh_scale = photo_paint.load_views(raw.with_suffix(".svviews"))
    x, y, _ = photo_paint.project(photo_paint.to_view_space(own, photo_paint.PIXAL3D_FRAME, mesh_scale), views[0])
    crop_px = hf.box_to_box(np.stack([x, y], axis=1), hf.alpha_box(views[0].image),
                            hf.alpha_box(np.asarray(face_picture.convert("RGBA"))))
    tf = hf.similarity_2d(own, hf.crop_to_figure(crop_px, box, face_picture.width, full_frame))
    deeper = hf.depth_fit(window, hf.face_relief(hf.place_face(built, tf), window, box, face_picture.width,
                                                 full_frame))
    tf = deeper @ tf
    scale = float(np.cbrt(np.linalg.det(tf[:3, :3])))
    shift = float(np.linalg.norm(hh.moved(rough, own)[:, :2] - hh.moved(tf, own)[:, :2], axis=1).mean())
    material = hf.picture_material(_png(up))
    face = hf.face_relief(hf.place_face(built, tf), window, box, face_picture.width, full_frame, material)
    seen = hf.sample(window["depth"], window["box"], window["pixel"], body["positions"], outside=-np.inf,
                     nearest=True)
    on_face = ((hf.window_values(window, body["positions"]) > 0)
               & (body["positions"][:, 2] > seen - hf.BEHIND_SHARE * window["width"]))
    cost = hf.pairs_apart(face["positions"][hf.window_values(window, face["positions"]) > 0],
                          body["positions"][on_face])
    run.say(f"place: through the build's camera the face moved {shift:.4f} across the picture from the rough frame, "
            f"and {deeper[2, 3]:+.4f} in depth onto the figure's face, turned by a slope of {deeper[2, 0]:+.3f} "
            f"across and {deeper[2, 1]:+.3f} up; size x{scale:.4f}; its relief lies "
            f"{cost:.5f} from the figure's old face ({cost / window['width']:.1%} of the head's width)")
    if cost > FIT_LIMIT * window["width"]:
        run.flag(f"the new face lies far from the old ({cost / window['width']:.1%} of the head's width): check the "
                 "crop and window.png")

    # 5. graft
    stage(run, "graft", "Setting the face in")
    fresh(run, "graft")
    record = read_model_record(args.model)
    upscaled = {"license": {"name": "BSD-3-Clause"}, "classification": "commercial-clear", "components": LICENCES}
    face_record = read_model_record(raw)
    summary = {"model": str(args.model), "picture": str(args.picture), "crop": list(box),
               "window": {"bounds": [round(b, 5) for b in bounds], "head_width": window["width"],
                          "brim_underside": window["floor"], "seed_pixel": list(window["seed"])},
               "placement": {"apart_from_old_face": cost, "scale": scale, "transform": tf.tolist()}, "models": {}}
    for model in [args.model, *args.also]:
        target_path = out / f"{model.stem}_face.glb"
        data = model.read_bytes()
        _, _, model_parts = hf.read_parts(data)
        cut_parts, face_out, report = hf.graft_face(model_parts, window, face)
        written = [hh.finish_part(p) for p in cut_parts] + [hh.finish_part(face_out, smooth=True)]
        target_path.write_bytes(hh.regraft_glb(data, written))
        run.say(f"graft: {target_path.name}: {report}")
        summary["models"][model.name] = {"result": str(target_path), **report}
        if report["seam_raised_short"] > 0.1:
            run.flag(f"{model.name}: along {report['seam_raised_short']:.0%} of the seam the figure stands more than "
                     f"{hf.EASE_SHARE * window['width']:.4f} in front of the new face (up to "
                     f"{report['seam_gap_max']:.4f}): its cut edge hangs over the face there; look at the seam")
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
