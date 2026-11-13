#!/usr/bin/env python3
"""Make a finished prop game-ready: front to +Z, real size, pivot on the floor, LODs and a collider.

Finish gives a prop its face count and its paint; an engine wants a few things more, and a
vehicle wants all of them:

1. **Front to +Z** (`--front`, where the front points now: +x, -x, +z or -z). glTF's front is
   +Z, and Unity's glTF importers keep it as the object's forward, so a car placed without a
   rotation faces the way it drives.
2. **Real size** (one of `--length`, `--width`, `--height`, in metres; the length runs along
   the front): one uniform scale. Pixal3D builds in a unit cube, so without one a bus stays a
   metre long.
3. **Pivot at the bottom centre**: the floor at y = 0 and the middle of the footprint at the
   origin, so the prop stands where it is placed.
4. **LODs** (`--lods`, the face counts after LOD0, default 2500,1000): each baked from LOD0,
   the way `finish_props.py` bakes a prop sheet's LODs from the original (simplifying LOD0
   blotched the barrel's texture there). LOD0 keeps its atlas, each further LOD gets half of
   the one before, down to 256, and every LOD's maps are compressed as Finish compresses
   them (uncompressed, the ambulance's LOD1 was heavier than its LOD0). Small parts are never
   dropped: a van's wheels are separate parts of about 1.5 % of its faces.
5. **A collider** (`--collider box|hull|none`): `box` is the box around the placed prop, the
   cheapest collider an engine has, also written into the record as centre and size so the
   engine can make it without a mesh; `hull` is a convex hull of LOD0 cut to `--hull-faces`
   triangles (Unity takes at most 255 for a convex mesh collider).

Writes `OUT_DIR/<name>_LOD<n>.glb` (each LOD's node named after it, as `finish_props.py`
names them), `<name>_COL.glb`, `<name>.prop.json` (size, pivot, front, LODs, collider, and
what looks wrong) and carries the licence record on as `<name>.provenance.json`.

    python scripts/prop_ready.py bus_5k.glb out/bus --name bus --height 3.27
    python scripts/prop_ready.py tent_5k.glb out/tent --name tent --front -x --length 10.3 --collider hull

Needs Blender 4.2 or newer; `blender_prop_ready.py` places the prop and makes the hull.
Everything here except `main` is plain maths, tested without Blender.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from itertools import pairwise
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# Degrees to turn about the up axis (counter-clockwise seen from above, +X towards -Z) so
# that the front, pointing this way now, points to +Z.
FRONTS = {"+z": 0.0, "-x": 90.0, "-z": 180.0, "+x": -90.0}
SIZES = ("width", "height", "length")       # the placed prop's x, y and z extents
DEFAULT_LODS = "2500,1000"
CONVEX_LIMIT = 255                          # Unity's triangle limit for a convex MeshCollider
FLOOR_TOLERANCE = 0.005                     # metres; a prop further off the floor floats
# No prop or vehicle a game places by hand is this small or this large; a size outside it
# is a scale left out (Pixal3D's unit cube) or typed wrong.
PLAUSIBLE_METRES = (0.05, 60.0)


def yaw_for_front(front: str) -> float:
    """The turn about the up axis, in degrees, that brings `front` to +Z."""
    key = front.strip().lower()
    if key and key[0] not in "+-":
        key = "+" + key
    if key not in FRONTS:
        raise ValueError(f"front is one of {', '.join(FRONTS)}, got {front!r}")
    return FRONTS[key]


def scale_for(extent, width: float | None = None, height: float | None = None,
              length: float | None = None) -> float:
    """The uniform scale that gives the prop its real size; 1.0 when no size is asked for.

    `extent` is the prop's (x, y, z) size with its front already at +Z: width, height, length.
    """
    asked = {name: value for name, value in zip(SIZES, (width, height, length)) if value is not None}
    if not asked:
        return 1.0
    if len(asked) > 1:
        raise ValueError(f"give one of --width, --height or --length, not {', '.join(asked)}")
    (name, metres), = asked.items()
    current = extent[SIZES.index(name)]
    if current <= 0:
        raise ValueError(f"the prop is flat along its {name}, so it cannot be scaled to {metres} m")
    return metres / current


def floor_translation(low, high) -> tuple[float, float, float]:
    """The move that puts the floor at y = 0 and the middle of the footprint at the origin."""
    return (-(low[0] + high[0]) / 2, -low[1], -(low[2] + high[2]) / 2)


def collider_box(size) -> dict:
    """The box around a placed prop: it stands on the floor, centred on the pivot."""
    width, height, length = (float(v) for v in size)
    return {"centre": [0.0, height / 2, 0.0], "size": [width, height, length]}


def box_glb(box: dict, name: str) -> bytes:
    """A closed, outward-facing box mesh as a GLB: 8 corners, 12 triangles, no material."""
    from compress_glb_textures import build_glb

    (cx, cy, cz), (sx, sy, sz) = box["centre"], box["size"]
    lo = (cx - sx / 2, cy - sy / 2, cz - sz / 2)
    hi = (cx + sx / 2, cy + sy / 2, cz + sz / 2)
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    # corner index = 4*x + 2*y + z, each 0 for low and 1 for high; faces wound outwards
    quads = [(0, 1, 3, 2), (4, 6, 7, 5),   # -x, +x
             (0, 4, 5, 1), (2, 3, 7, 6),   # -y, +y
             (0, 2, 6, 4), (1, 5, 7, 3)]   # -z, +z
    triangles = [index for a, b, c, d in quads for index in (a, b, c, a, c, d)]
    positions = struct.pack("<24f", *(v for corner in corners for v in corner))
    indices = struct.pack("<36H", *triangles)
    binary = positions + indices
    document = {
        "asset": {"version": "2.0", "generator": "prop_ready.py"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
                        {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices),
                         "target": 34963}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
                       "min": [float(v) for v in lo], "max": [float(v) for v in hi]},
                      {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"}],
    }
    return build_glb(document, binary)


def lod_name(name: str, index: int) -> str:
    """A LOD's node name, as `finish_props.py` names them."""
    return f"{name}_LOD{index}"


def lod_plan(lods: list[int], atlas: int) -> list[tuple[int, int, int]]:
    """(index, faces, atlas) for each LOD after LOD0: half the atlas of the one before."""
    from finish_props import MIN_LOD_ATLAS
    return [(index, faces, max(MIN_LOD_ATLAS, atlas >> index))
            for index, faces in enumerate(lods, start=1)]


def parse_lods(text: str) -> list[int]:
    """`--lods`: comma-separated face counts after LOD0, each lighter than the one before."""
    try:
        lods = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise SystemExit(f"--lods wants comma-separated face counts, got {text!r}") from None
    if any(faces <= 0 for faces in lods):
        raise SystemExit("--lods wants face counts above zero")
    if any(later >= earlier for earlier, later in pairwise(lods)):
        raise SystemExit(f"--lods must get lighter from one LOD to the next, got {text!r}")
    return lods


def problems(record: dict) -> list[str]:
    """What looks wrong in a finished prop, one plain sentence each; empty when nothing does."""
    found = []
    size = record.get("size_m") or []
    low, high = PLAUSIBLE_METRES
    if size and (max(size) > high or max(size) < low):
        found.append(f"size {' x '.join(f'{v:.2f}' for v in size)} m is outside {low}-{high} m: "
                     "was --length, --width or --height left out or typed wrong?")
    if abs(record.get("floor_y", 0.0)) > FLOOR_TOLERANCE:
        found.append(f"the prop is {record['floor_y']:.3f} m off the floor")
    triangles = [lod.get("triangles", 0) for lod in record.get("lods", [])]
    for index, (earlier, later) in enumerate(pairwise(triangles), start=1):
        if later >= earlier:
            found.append(f"LOD{index} has {later} triangles, no fewer than LOD{index - 1}'s {earlier}")
    sizes = [lod.get("bytes", 0) for lod in record.get("lods", [])]
    for index, (earlier, later) in enumerate(pairwise(sizes), start=1):
        if earlier and later > earlier:
            found.append(f"LOD{index}'s file is {later / 1e6:.1f} MB, heavier than LOD{index - 1}'s "
                         f"{earlier / 1e6:.1f} MB")
    collider = record.get("collider") or {}
    if collider.get("kind") == "hull" and collider.get("triangles", 0) > CONVEX_LIMIT:
        found.append(f"the hull has {collider['triangles']} triangles; a convex mesh collider takes "
                     f"at most {CONVEX_LIMIT}")
    return found


def blender_command(blender: Path, job: Path) -> list[str]:
    """The headless Blender run of `blender_prop_ready.py` for one job file."""
    return [str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
            "--python", str(SCRIPTS / "blender_prop_ready.py"), "--", str(job)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the finished prop (Finish's GLB): becomes LOD0")
    parser.add_argument("out_dir", type=Path, help="folder for the LODs, the collider and the record")
    parser.add_argument("--name", required=True, help="the prop's name: files and nodes are named after it")
    parser.add_argument("--front", default="+z", help="where the prop's front points now: +x, -x, +z or -z")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--length", type=float, help="real length along the front, metres")
    size.add_argument("--width", type=float, help="real width, metres")
    size.add_argument("--height", type=float, help="real height, metres")
    parser.add_argument("--lods", default=DEFAULT_LODS,
                        help=f"face counts after LOD0, lighter each time (default {DEFAULT_LODS}; '' for none)")
    parser.add_argument("--collider", choices=("box", "hull", "none"), default="box")
    parser.add_argument("--hull-faces", type=int, default=CONVEX_LIMIT,
                        help=f"most triangles for --collider hull (default {CONVEX_LIMIT})")
    parser.add_argument("--blender", type=Path, help="Blender to use (default: I2L_BLENDER, then the usual places)")
    return parser


def _glb_atlas(glb: bytes) -> int:
    """The largest texture side in a GLB, read from the PNG or JPEG headers of its images."""
    from compress_glb_textures import parse_glb

    document, binary = parse_glb(glb)
    largest = 0
    for image in document.get("images", []):
        view = document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        data = binary[start:start + view["byteLength"]]
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            largest = max(largest, *struct.unpack(">II", data[16:24]))
        elif data[:2] == b"\xff\xd8":
            at = 2
            while at + 9 < len(data):
                if data[at] != 0xFF:
                    break
                marker, length = data[at + 1], struct.unpack(">H", data[at + 2:at + 4])[0]
                if marker in (0xC0, 0xC1, 0xC2):
                    h, w = struct.unpack(">HH", data[at + 5:at + 9])
                    largest = max(largest, h, w)
                    break
                at += 2 + length
    return largest


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - runs Blender; its parts are tested
    import subprocess

    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(SCRIPTS.parent))
    from compress_glb_textures import compress
    from finish_props import ANGLE, VOXEL, glb_triangles, write_lod
    from retopo_repaint import bake_command, retopo_command

    from image_to_3dlab.blender import find_blender

    args = build_parser().parse_args(argv)
    yaw = yaw_for_front(args.front)
    lods = parse_lods(args.lods)
    blender = args.blender or find_blender()
    if blender is None:
        raise SystemExit("needs Blender 4.2 or newer: set I2L_BLENDER or pass --blender")
    out = args.out_dir
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    started = time.time()

    lod0 = out / f"{args.name}_LOD0.glb"
    placed_glb = logs / "LOD0.placed.glb"
    hull = out / f"{args.name}_COL.glb"
    atlas = _glb_atlas(args.source.read_bytes()) or 2048
    job = {"source": str(args.source.resolve()), "output": str(placed_glb.resolve()), "yaw": yaw,
           "size": {name: getattr(args, name) for name in SIZES if getattr(args, name) is not None},
           "node": lod_name(args.name, 0),
           "hull": ({"output": str(hull.resolve()), "faces": args.hull_faces, "node": f"{args.name}_COL"}
                    if args.collider == "hull" else None)}
    job_path = logs / "place.job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    log = logs / "place.log"
    with log.open("w", encoding="utf-8") as handle:
        done = subprocess.run(blender_command(blender, job_path), check=False, stdout=handle, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        raise SystemExit(f"placing failed (exit {done.returncode}); see {log}")
    placed = json.loads((logs / "place.result.json").read_text(encoding="utf-8"))
    # Blender writes the textures out again, larger; squeeze them back to Finish's sizes
    squeezed, _report = compress(placed_glb.read_bytes(), max_size=atlas, map_size=min(atlas, 1024),
                                 normal_size=atlas)
    placed_glb.write_bytes(squeezed)
    write_lod(placed_glb, lod0, lod_name(args.name, 0))

    record = {"name": args.name, "source": str(args.source), "front": "+Z", "turned_deg": yaw,
              "scale": placed["scale"], "size_m": placed["size"], "floor_y": placed["low"][1],
              "pivot": "bottom centre", "atlas": atlas,
              "lods": [{"glb": lod0.name, "triangles": glb_triangles(lod0.read_bytes()), "atlas": atlas,
                        "bytes": lod0.stat().st_size}]}
    for index, faces, lod_atlas in lod_plan(lods, atlas):
        print(f"LOD{index}: {faces} faces, {lod_atlas} px", flush=True)
        retopo = logs / f"LOD{index}.retopo.glb"
        detailed = logs / f"LOD{index}.detail.glb"
        target = out / f"{args.name}_LOD{index}.glb"
        for command, name in ((retopo_command(lod0, retopo, faces, lod_atlas, ANGLE, VOXEL, 0.25, 0.65, 1.45,
                                              blender=blender), "retopo"),
                              (bake_command(lod0, retopo, detailed, lod_atlas, blender=blender), "detail")):
            with (logs / f"LOD{index}.{name}.log").open("w", encoding="utf-8") as handle:
                done = subprocess.run(command, check=False, stdout=handle, stderr=subprocess.STDOUT)
            if done.returncode != 0:
                raise SystemExit(f"LOD{index} {name} failed (exit {done.returncode}); see logs/")
        retopo.unlink()
        # the bakes write uncompressed maps: as Finish does, JPEG at this LOD's atlas
        squeezed, _report = compress(detailed.read_bytes(), max_size=lod_atlas,
                                     map_size=min(lod_atlas, 1024), normal_size=lod_atlas)
        detailed.write_bytes(squeezed)
        write_lod(detailed, target, lod_name(args.name, index))
        record["lods"].append({"glb": target.name, "faces": faces, "atlas": lod_atlas,
                               "triangles": glb_triangles(target.read_bytes()), "bytes": target.stat().st_size})

    if args.collider == "box":
        box = collider_box(placed["size"])
        hull.write_bytes(box_glb(box, f"{args.name}_COL"))
        record["collider"] = {"kind": "box", **box, "glb": hull.name}
    elif args.collider == "hull":
        record["collider"] = {"kind": "hull", "glb": hull.name, "triangles": placed["hull_triangles"],
                              "box": collider_box(placed["size"])}
    else:
        record["collider"] = {"kind": "none"}

    licence = args.source.with_suffix(".provenance.json")
    if licence.is_file():
        shutil.copyfile(licence, out / f"{args.name}.provenance.json")
        record["provenance"] = f"{args.name}.provenance.json"
    record["problems"] = problems(record)
    record["seconds"] = round(time.time() - started, 1)
    (out / f"{args.name}.prop.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    size = " x ".join(f"{v:.2f}" for v in record["size_m"])
    print(f"{args.name}: {size} m (w x h x l), front +Z, "
          + ", ".join(f"LOD{i} {lod['triangles']}" for i, lod in enumerate(record["lods"]))
          + f", collider {record['collider']['kind']}", flush=True)
    for line in record["problems"]:
        print(f"check: {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
