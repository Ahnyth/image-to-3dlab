#!/usr/bin/env python3
"""Repeat a band's seen arc round the band (a hat band's runes, a belt's studs), where no picture sees it.

    python scripts/ring_paint.py painted.glb views_dir out.glb [--box X0 Y0 X1 Y1 ...] [--report out.json]

Pixel Match (`photo_paint.py`) copies the picture onto every surface that faces it. A ring round a figure (a hat
band, a belt, a bracelet) faces the front picture only on its front arc: its sides and back keep the generator's
own paint, which redraws the band's pattern as a blur. A ring's pattern goes on round it, so this repeats the arc
the picture sees round the rest of the ring, cut between two of its marks, at the same height on the band.

`painted.glb` is Pixel Match's result, or any later model in the same frame (one mesh; the primitive with the most
surface inside the box is painted, so parts set in later keep theirs). `views_dir` is the folder Pixel Match painted
from (`transforms.json` and the pictures); its first picture is the front, and what the others see (a drawn back)
keeps its paint. `--box` is the band in that picture's pixels, as an image viewer shows them, from its left end to its
right and from its top edge to its bottom edge; give it again for another ring. Without `--box` the band is found in
that picture: a level row of marks evenly spaced round a plain band set off from what lies above and below it
(`find_bands`); with none found nothing is written and the exit code is 3. Only the marks travel: the band's plain
colour follows a slow trend round it, so the picture's painted light leaves no seam where the repeat joins. Only the
base colour image changes; every other byte stays.

The logic lives in `image_to_3dlab/ring_paint.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import ring_paint as rp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="the painted model (Pixel Match's result, or a later one)")
    parser.add_argument("views", type=Path, help="the folder it was painted from (transforms.json + pictures)")
    parser.add_argument("output", type=Path)
    parser.add_argument("--box", type=float, nargs=4, action="append", default=None,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="the band in the first picture's pixels; give it again for another ring (default: "
                             "found in the picture)")
    parser.add_argument("--trusted", type=float, default=rp.TRUSTED,
                        help="how much the picture must be trusted on a texel to count as seen (0..1)")
    parser.add_argument("--report", type=Path, help="also write what was found and repainted as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.model, args.views / "transforms.json"):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")
    started = time.time()
    views, mesh_scale = pp.load_views(args.views)
    boxes, found = args.box, None
    if not boxes:
        found = rp.find_bands(views[0].image)
        boxes = [band["box"] for band in found]
        for band in found:
            print(f"[ring-paint] found a band at {band['box']}: {band['marks']} marks, spacing spread "
                  f"{band['gap_cv']:.2f}", flush=True)
        if not boxes:
            print("[ring-paint] no band whose pattern repeats along it in the picture: nothing repainted", flush=True)
            if args.report:
                args.report.write_text(json.dumps({"found": [], "rings": []}, indent=1), encoding="utf-8")
            return 3
    data, reports = rp.paint_glb(args.model.read_bytes(), views[0], boxes, mesh_scale, trusted=args.trusted,
                                 others=views[1:])
    for report in reports:
        if report.get("skipped"):
            print(f"[ring-paint] box {report['box']}: left alone, {report['skipped']}", flush=True)
        else:
            print(f"[ring-paint] box {report['box']}: ring of radius {report['radius']:.4f} round "
                  f"{[round(v, 4) for v in report['middle']]}, seen arc {report['arc_degrees']:.0f} degrees, "
                  f"{report['repainted']} texels repainted", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    if args.report:
        args.report.write_text(json.dumps(reports if found is None else {"found": found, "rings": reports}, indent=1),
                               encoding="utf-8")
    print(f"[ring-paint] {sum(r['repainted'] for r in reports)} texels in {len(reports)} ring(s), "
          f"{time.time() - started:.0f}s -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
