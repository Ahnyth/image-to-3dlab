#!/usr/bin/env python3
"""Mark where an effect was taken out of a picture, as empty nodes in the model built from it.

    python scripts/fx_markers.py model.glb original_front.png effect_free_front.png out.glb [--name flame]
        [--side original_right.png effect_free_right.png]

A flame, a glow or sparks built as geometry look like sheets of paper from the side; a game's own particle
effect moves and glows. Take the effect out of the front picture with the studio's picture editor, build the
model from that picture, then run this: it lines the two pictures up, finds where the effect sat, and writes
one empty node per effect (`fx_<name>_1`, `fx_<name>_2`, ...) where the effect touched it, its size in
`extras.radius` (metres if the model is in metres). From the front alone a node sits on the model's front
surface; give `--side` the right side view with and without the effect (as `make_side_views.py` draws them)
and it sits at the depth the side shows, so a flame on the back of the neck is marked there, not on the face
(`extras.depth_from` says which). An engine imports the nodes as empty objects to attach its effect to. The
mesh, materials and textures are copied unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import fx_markers, matte


def cut_out(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        picture = opened.convert("RGBA") if matte.is_matted(opened) else matte.cut_out(opened.convert("RGB"))[0]
    return np.asarray(picture.convert("RGBA"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="the GLB built from the effect-free picture")
    parser.add_argument("original", type=Path, help="the front picture with the effect")
    parser.add_argument("clean", type=Path, help="the same front picture with the effect taken out")
    parser.add_argument("output", type=Path, help="the GLB to write, with the markers")
    parser.add_argument("--name", default="flame", help="what the effect is, for the nodes' names (default flame)")
    parser.add_argument("--side", nargs=2, type=Path, metavar=("ORIGINAL", "CLEAN"),
                        help="the right side view with and without the effect, for the effects' depth")
    args = parser.parse_args(argv)
    for p in (args.model, args.original, args.clean):
        if not p.is_file():
            raise SystemExit(f"not found: {p}")
    orig, clean = cut_out(args.original), cut_out(args.clean)
    regions = fx_markers.effect_regions(orig, clean)
    if not regions:
        print("[fx-markers] the two pictures show no effect to mark; nothing written")
        return 1
    side = None
    if args.side:
        for p in args.side:
            if not p.is_file():
                raise SystemExit(f"not found: {p}")
        side = (cut_out(args.side[0]), cut_out(args.side[1]))
    out = fx_markers.add_markers(args.model.read_bytes(), orig, clean, name=args.name, side=side)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(out)
    print(f"[fx-markers] {len(regions)} effect(s) marked -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
