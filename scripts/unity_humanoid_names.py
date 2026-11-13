#!/usr/bin/env python3
"""Rename a rigged humanoid GLB's bones to Unity Humanoid or Mixamo names, without Blender.

The auto-rig names its bones `bone_0` to `bone_N`, which a game engine's humanoid avatar
cannot map on its own (Unity's Humanoid, Unreal's and Godot's retargeters all go by name).
This finds each body bone with the same mapper the preset moves use (`kimodo_retarget`,
by tree shape, never by number) and gives it the engine's name: Hips, Spine, Chest, ...
LeftToes. Bones that are not part of the body (hair, a cape, a sword) keep their names,
and so do the fingers of a mitten hand, whose look-alike chains cannot be told apart.

Only joint node names change. The mesh, the skin weights and every move stay exactly as
they are: glTF animations point at bones by index, not by name.

    python scripts/unity_humanoid_names.py output/animate/<run>/<name>_idle.glb named.glb
    python scripts/unity_humanoid_names.py rigged.glb named.glb --style mixamo
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimodo_retarget import map_skintokens_to_soma
from rig_check import Y_UP, skeleton

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
FINGERS = (("Index", "Index"), ("Middle", "Middle"), ("Ring", "Ring"), ("Pinky", "Little"))
PHALANGES = ("Proximal", "Intermediate", "Distal")


def _unity() -> dict[str, str]:
    """SOMA joint (what `map_skintokens_to_soma` returns) -> Unity Humanoid slot name."""
    table = {"Hips": "Hips", "Spine1": "Spine", "Spine2": "Chest", "Chest": "UpperChest",
             "Neck1": "Neck", "Neck2": "Neck", "Head": "Head"}
    for side in ("Left", "Right"):
        table.update({f"{side}Shoulder": f"{side}Shoulder", f"{side}Arm": f"{side}UpperArm",
                      f"{side}ForeArm": f"{side}LowerArm", f"{side}Hand": f"{side}Hand",
                      f"{side}Leg": f"{side}UpperLeg", f"{side}Shin": f"{side}LowerLeg",
                      f"{side}Foot": f"{side}Foot", f"{side}ToeBase": f"{side}Toes"})
        for i, phalanx in enumerate(PHALANGES):
            table[f"{side}HandThumb{i + 1}"] = f"{side}Thumb{phalanx}"
            # SOMA fingers start at the metacarpal (1); SkinTokens' start at the knuckle (2).
            for soma, unity in FINGERS:
                table[f"{side}Hand{soma}{i + 2}"] = f"{side}{unity}{phalanx}"
    return table


def _mixamo() -> dict[str, str]:
    """SOMA joint -> Mixamo bone name (what Mixamo rigs use; Unity and Unreal map them on their own)."""
    table = {"Hips": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Chest": "Spine2",
             "Neck1": "Neck", "Neck2": "Neck", "Head": "Head"}
    for side in ("Left", "Right"):
        table.update({f"{side}Shoulder": f"{side}Shoulder", f"{side}Arm": f"{side}Arm",
                      f"{side}ForeArm": f"{side}ForeArm", f"{side}Hand": f"{side}Hand",
                      f"{side}Leg": f"{side}UpLeg", f"{side}Shin": f"{side}Leg",
                      f"{side}Foot": f"{side}Foot", f"{side}ToeBase": f"{side}ToeBase"})
        for i in range(3):
            table[f"{side}HandThumb{i + 1}"] = f"{side}HandThumb{i + 1}"
            for soma, _ in FINGERS:
                table[f"{side}Hand{soma}{i + 2}"] = f"{side}Hand{soma}{i + 1}"
    return table


STYLES = {"unity": _unity(), "mixamo": _mixamo()}


def engine_names(parents: dict[str, str | None], mapping: dict[str, str], style: str = "unity") -> dict[str, str]:
    """{bone: engine name} for the bones that get one; every other bone keeps its own name.

    A bone is named when it is the first of its chain with that engine role: the mapper hands
    extra bones (hair, a cape) their parent's role, and a two-bone neck both say Neck. A name
    that would still land on more than one bone (four mitten fingers, all "middle") is given
    to none of them, because guessing would put an index finger in the little finger's slot.
    """
    table = STYLES[style]
    role = {bone: table.get(soma) for bone, soma in mapping.items()}
    first = {bone: name for bone, name in role.items() if name and role.get(parents.get(bone)) != name}
    counts = Counter(first.values())
    return {bone: name for bone, name in first.items() if counts[name] == 1}


def rename(glb: bytes, names: dict[str, str]) -> bytes:
    """`glb` with its skin joints renamed per `names`; the binary chunk is kept byte for byte."""
    if len(glb) < 20 or struct.unpack_from("<I", glb, 0)[0] != GLB_MAGIC:
        raise ValueError("not a GLB file")
    version = struct.unpack_from("<I", glb, 4)[0]
    json_len, kind = struct.unpack_from("<II", glb, 12)
    if kind != JSON_CHUNK:
        raise ValueError("GLB has no JSON chunk first")
    doc = json.loads(glb[20:20 + json_len])
    nodes = doc.get("nodes", [])
    for joint in {j for skin in doc.get("skins", []) for j in skin.get("joints", [])}:
        old = nodes[joint].get("name")
        if old in names:
            nodes[joint]["name"] = names[old]
    chunk = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    chunk += b" " * (-len(chunk) % 4)  # chunks are 4-byte aligned; the spec pads JSON with spaces
    rest = glb[20 + json_len:]
    body = struct.pack("<II", len(chunk), JSON_CHUNK) + chunk + rest
    return struct.pack("<III", GLB_MAGIC, version, 12 + len(body)) + body


def main() -> int:  # pragma: no cover - thin CLI over the tested functions
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="rigged (or animated) GLB from the auto-rig")
    parser.add_argument("output", type=Path, help="GLB to write with engine bone names")
    parser.add_argument("--style", choices=sorted(STYLES), default="unity",
                        help="unity: Unity's Humanoid slot names (default); mixamo: Mixamo's names")
    args = parser.parse_args()
    parents, heads = skeleton(args.source)
    mapping, _ = map_skintokens_to_soma(parents, heads, up=Y_UP)
    names = engine_names(parents, mapping, args.style)
    args.output.write_bytes(rename(args.source.read_bytes(), names))
    kept = sorted(set(parents) - set(names), key=lambda b: (len(b), b))
    print(f"{len(names)} of {len(parents)} bones named ({args.style}); kept their names: {', '.join(kept) or 'none'}")
    for bone, name in sorted(names.items(), key=lambda kv: list(parents).index(kv[0])):
        print(f"  {bone:>10} -> {name}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
