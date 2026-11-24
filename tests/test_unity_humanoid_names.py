"""Tests for `scripts/unity_humanoid_names.py`: engine bone names for a SkinTokens humanoid.

The skeletons are the synthetic humanoids `tests/test_kimodo_retarget.py` already builds in
SkinTokens' tree shape (shuffled `bN` names), so the names come from the real body mapper.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import kimodo_retarget as kr
import unity_humanoid_names as uhn
from test_kimodo_retarget import _humanoid, _with_extras


def _names(parents, heads, style="unity"):
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    return uhn.engine_names(parents, mapping, style)


def test_every_body_bone_gets_its_unity_humanoid_name():
    parents, heads, roles = _humanoid()
    names = _names(parents, heads)
    expect = {
        "hips": "Hips", "s1": "Spine", "s2": "Chest", "chest": "UpperChest", "neck": "Neck", "head": "Head",
        "Lsh": "LeftShoulder", "Larm": "LeftUpperArm", "Lfa": "LeftLowerArm", "Lhand": "LeftHand",
        "Rsh": "RightShoulder", "Rarm": "RightUpperArm", "Rfa": "RightLowerArm", "Rhand": "RightHand",
        "Lleg": "LeftUpperLeg", "Lshin": "LeftLowerLeg", "Lfoot": "LeftFoot", "Ltoe": "LeftToes",
        "Rleg": "RightUpperLeg", "Rshin": "RightLowerLeg", "Rfoot": "RightFoot", "Rtoe": "RightToes",
        "Lthumb0": "LeftThumbProximal", "Lthumb2": "LeftThumbDistal",
        "Lindex0": "LeftIndexProximal", "Rmiddle1": "RightMiddleIntermediate", "Lpinky2": "LeftLittleDistal",
    }
    for role, name in expect.items():
        assert names[roles[role]] == name, role


def test_no_name_is_given_twice():
    parents, heads, _ = _humanoid()
    names = _names(parents, heads)
    assert len(set(names.values())) == len(names)


def test_the_mixamo_style_uses_mixamo_names():
    parents, heads, roles = _humanoid()
    names = _names(parents, heads, "mixamo")
    expect = {"hips": "Hips", "s1": "Spine", "s2": "Spine1", "chest": "Spine2", "neck": "Neck",
              "Larm": "LeftArm", "Lfa": "LeftForeArm", "Lleg": "LeftUpLeg", "Lshin": "LeftLeg",
              "Rtoe": "RightToeBase", "Lthumb0": "LeftHandThumb1", "Lindex2": "LeftHandIndex3"}
    for role, name in expect.items():
        assert names[roles[role]] == name, role


def test_extra_bones_keep_their_own_names():
    # Hair, a cape, a skirt, a sword: not body parts, so no engine slot of their own.
    parents, heads, roles = _humanoid()
    parents, heads = _with_extras(parents, heads, roles)
    names = _names(parents, heads)
    for extra in ("hair0", "hair1", "cape0", "cape2", "skirtF", "skirtB1", "strap", "sword", "blade"):
        assert extra not in names
    assert names[roles["hips"]] == "Hips" and names[roles["Rfa"]] == "RightLowerArm"


def test_mitten_hands_name_the_arm_and_leave_the_fingers_alone():
    # SkinTokens leaves out fingers it cannot see; four look-alike chains cannot be told apart
    # as index, middle, ring and little, so none of them is named.
    parents, heads, roles = _humanoid(fingers=False)
    for lr, sx in (("L", -1), ("R", 1)):
        for k in range(4):
            parent = roles[f"{lr}hand"]
            for j in range(2):
                bone = f"{lr}mit{k}_{j}"
                parents[bone] = parent
                heads[bone] = np.array((sx * (0.72 + 0.03 * j), 0.03 - 0.02 * k, 1.45))
                parent = bone
    names = _names(parents, heads)
    assert names[roles["Lhand"]] == "LeftHand" and names[roles["Rhand"]] == "RightHand"
    assert not any(b.startswith(("Lmit", "Rmit")) for b in names)


def test_an_unknown_style_is_refused():
    parents, heads, _ = _humanoid()
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    with pytest.raises(KeyError):
        uhn.engine_names(parents, mapping, "maya")


def _glb(doc: dict, binary: bytes = b"\x01\x02\x03\x04") -> bytes:
    chunk = json.dumps(doc).encode()
    chunk += b" " * (-len(chunk) % 4)
    body = struct.pack("<II", len(chunk), 0x4E4F534A) + chunk + struct.pack("<II", len(binary), 0x004E4942) + binary
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def test_rename_changes_only_the_joint_names_and_keeps_the_binary():
    doc = {"nodes": [{"name": "Body", "mesh": 0}, {"name": "bone_0", "children": [2]}, {"name": "bone_1"}],
           "skins": [{"joints": [1, 2]}], "animations": [{"channels": [{"target": {"node": 2}}]}]}
    out = uhn.rename(_glb(doc), {"bone_0": "Hips", "bone_1": "Spine", "Body": "Hips2"})
    magic, version, length = struct.unpack_from("<III", out, 0)
    assert (magic, version, length) == (0x46546C67, 2, len(out))
    json_len = struct.unpack_from("<I", out, 12)[0]
    assert json_len % 4 == 0
    new = json.loads(out[20:20 + json_len])
    assert [n["name"] for n in new["nodes"]] == ["Body", "Hips", "Spine"]  # the mesh node is not a joint
    assert new["animations"] == doc["animations"]  # channels point at node indices, which do not move
    assert out.endswith(b"\x01\x02\x03\x04")


def test_rename_refuses_a_file_that_is_not_a_glb():
    with pytest.raises(ValueError):
        uhn.rename(b"not a glb at all", {})
