#!/usr/bin/env python3
"""Say where two versions of one figure differ in colour, such as a game model and its source.

A game model is thinned, unwrapped and baked anew, and a check that counts holes in motion does
not see a wrong colour: a bake once put a dark, faceted patch on a cape's back that only a look
at the colour showed. This draws both models from eight sides around the figure in their base
colour, as a game shows them (the nearest face, back faces culled, no light), and names every
patch that differs: which side, how big, darker or lighter, where on the figure. Colour edges
that move a pixel or two (a thinner mesh, new UVs) and the outline's rim do not count.

    python scripts/colour_compare.py OLD.glb NEW.glb
    python scripts/colour_compare.py OLD.glb NEW.glb --picture changed.png

OLD and NEW are two versions of one figure (before and after a change, or a game model and the
model it was baked from). It only reports; it exits 1 when a patch differs, 0 when none does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_to_3dlab import colour_compare as cc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old", type=Path, help="the version to compare against (a GLB)")
    parser.add_argument("new", type=Path, help="the version to check (a GLB)")
    parser.add_argument("--picture", type=Path, help="write old | new | changed for every view with a patch")
    parser.add_argument("--size", type=int, default=384, help="pixels per view (default 384)")
    parser.add_argument("--delta", type=float, default=12.0,
                        help="colour difference that counts, CIE76 delta E (default 12; 2 is just visible)")
    parser.add_argument("--min-share", type=float, default=0.3,
                        help="smallest patch, in percent of the figure in that view (default 0.3)")
    args = parser.parse_args(argv)
    old, new = cc.read_model(args.old), cc.read_model(args.new)
    if args.picture:
        patches = cc.marked_picture(old, new, args.picture, args.size, args.delta, args.min_share / 100)
    else:
        patches = cc.compare(old, new, args.size, args.delta, args.min_share / 100)
    if not patches:
        print(f"no patch differs in colour ({len(cc.VIEWS)} views, delta E above {args.delta:g}, "
              f"at least {args.min_share:g} % of the figure)")
        return 0
    views = sorted({p.view for p in patches})
    print(f"{len(patches)} patch(es) differ in colour, seen from {len(views)} side(s):")
    for p in patches:
        print("  " + p.describe())
    if args.picture:
        print(f"marked: {args.picture}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
