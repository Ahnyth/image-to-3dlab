#!/usr/bin/env python3
"""Check a finished game figure the way a second reviewer would, and write a report with pictures.

It draws the figure's head, neck and hands up close (found from its skeleton) and measures the
faults a person sees at once: the fingers per hand, a neck that reads as a pale ring, dark
strokes under the jaw, saw teeth along the head's outline, loose flakes, a grafted face's seam
(the figure's cut edge hanging over it, holes in it, see-through beside it), what the light shows
on the head's skin that its paint does not (drawn lit with the figure's own normals and unlit),
open edges, holes in motion (from game_ready's report beside the model) and, given the version it
replaces, where its colour changed and any new patch or crease in the light. Fingers, the neck ring, the face seam and holes are verdicts; strokes, teeth, flakes and
colour changes are hints ("look here", circled), because drawn lines, spiky hair and a new part
look the same to a measure. A hint fails when it got clearly worse than in the version before,
and a colour change fails with --same-look (a rebuild that should only lose faults).

    python scripts/figure_check.py game/figure.glb
    python scripts/figure_check.py new.glb --before old.glb --out check/
    python scripts/figure_check.py rebuilt.glb --before approved.glb --same-look
    python scripts/figure_check.py v1.glb v2.glb v3.glb --series --out check/

`--series` checks each model against the one before it in the list. It writes index.html and
report.json into --out and exits 1 when any model fails a check. The logic lives in
`image_to_3dlab/figure_check.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_to_3dlab import figure_check as fc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", type=Path, nargs="+", help="the game figure(s) to check (GLB)")
    parser.add_argument("--before", type=Path, help="the version the (first) model replaces")
    parser.add_argument("--series", action="store_true",
                        help="check each model against the one before it in the list")
    parser.add_argument("--out", type=Path, help="where to write index.html and report.json "
                        "(default: <model>_check beside the first model)")
    parser.add_argument("--same-look", action="store_true",
                        help="the new version should only lose faults: a colour change fails")
    parser.add_argument("--title", default="Figure check", help="the report's title")
    parser.add_argument("--size", type=int, default=320, help="pixels per close view (default 320)")
    args = parser.parse_args(argv)
    out = args.out or args.models[0].with_name(args.models[0].stem + "_check")
    out.mkdir(parents=True, exist_ok=True)
    checked = []
    for i, model in enumerate(args.models):
        before = args.models[i - 1] if args.series and i else (args.before if i == 0 else None)
        picture = out / f"{model.stem}_colour.png" if before is not None else None
        result = fc.check(model, before=before, size=args.size, colour_picture=picture, same_look=args.same_look)
        checked.append(result)
        print(f"{model.name}: {'FAILED' if result.failed else 'passed'}")
        for r in result.results:
            print(f"  {fc.STATUS_WORDS[r.status]:9} {r.name}: {r.message}")
    page = fc.write_report(checked, out, args.title)
    print(f"report: {page}")
    return 1 if any(c.failed for c in checked) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
