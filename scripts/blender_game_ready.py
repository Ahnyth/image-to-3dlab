#!/usr/bin/env python3
"""The Blender half of `game_ready.py`: rescale, clean, new UVs, one baked atlas, weights, export.

Run (internal): blender --background --python-exit-code 1 --python scripts/blender_game_ready.py -- job.json

`game_ready.py` writes the job (source, output, report, the bone plan and the settings) and
reads the report back. The maths lives in `game_ready.py`, where it is tested without Blender;
this file holds the Blender calls only. The armature is kept: bones are rescaled in place and
every move stays on them, so no move has to be retargeted.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_retopo_bake as rb
import game_ready as gr
from visibility_cull import (
    dense_surface,
    face_samples,
    fibonacci_directions,
    first_air_sealed,
    hidden_from_rays,
    hollows_only,
    poking_out,
    sealed_air,
    shading_roofs,
)

T0 = time.time()


def log(*parts) -> None:
    print("GAME-READY", *parts, flush=True)


def fcurve_slots(action):
    """(collection, fcurve) for every F-curve of an action, layered (Blender 4.4+) or not."""
    layers = getattr(action, "layers", None)
    if layers:
        for layer in layers:
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", []):
                    for fc in list(bag.fcurves):
                        yield bag.fcurves, fc
    elif hasattr(action, "fcurves"):
        for fc in list(action.fcurves):
            yield action.fcurves, fc


def import_source(path: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.gltf(filepath=path, disable_bone_shape=True)
    except TypeError:
        bpy.ops.import_scene.gltf(filepath=path)
    scene = bpy.context.scene
    arms = [o for o in scene.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise SystemExit(f"expected one armature, found {len(arms)}")
    arm = arms[0]
    skinned = [o for o in scene.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]
    if not skinned:
        raise SystemExit("no skinned mesh in the file")
    for o in list(scene.objects):
        if o is not arm and o not in skinned:
            bpy.data.objects.remove(o, do_unlink=True)
    if len(skinned) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in skinned:
            o.select_set(True)
        bpy.context.view_layer.objects.active = skinned[0]
        bpy.ops.object.join()
    return arm, skinned[0]


def swap_in_finer_mesh(arm, ob, path: str):
    """`--mesh-from`: a finer Finish of the same figure becomes the mesh, its skin weights carried over from the
    rigged mesh (nearest face, interpolated; both share the Finish's coordinates). Returns the new mesh object."""
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=path, disable_bone_shape=True)
    except TypeError:
        bpy.ops.import_scene.gltf(filepath=path)
    added = set(bpy.data.objects) - before
    meshes = [o for o in added if o.type == "MESH" and o.data.materials]
    for o in added:
        if o not in meshes:
            bpy.data.objects.remove(o, do_unlink=True)
    if not meshes:
        raise SystemExit(f"--mesh-from: no mesh in {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    fine = bpy.context.view_layer.objects.active
    mw = fine.matrix_world.copy()
    fine.parent = None
    fine.matrix_world = Matrix.Identity(4)
    fine.data.transform(mw)
    arm.data.pose_position = "REST"            # the weights are read off the figure as it stands
    bpy.context.view_layer.update()
    for name in [g.name for g in ob.vertex_groups]:
        fine.vertex_groups.new(name=name)
    mod = fine.modifiers.new("gr_weights", "DATA_TRANSFER")
    mod.object = ob
    mod.use_object_transform = True
    mod.use_vert_data = True
    mod.data_types_verts = {"VGROUP_WEIGHTS"}
    mod.vert_mapping = "POLYINTERP_NEAREST"
    mod.layers_vgroup_select_src = "ALL"
    mod.layers_vgroup_select_dst = "NAME"
    bpy.ops.object.select_all(action="DESELECT")
    fine.select_set(True)
    bpy.context.view_layer.objects.active = fine
    bpy.ops.object.modifier_apply(modifier=mod.name)
    skin = fine.modifiers.new("Armature", "ARMATURE")
    skin.object = arm
    fine.parent = arm
    fine.matrix_parent_inverse = arm.matrix_world.inverted()
    old_mesh = ob.data
    bpy.data.objects.remove(ob, do_unlink=True)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    log(f"mesh from {Path(path).name}: {len(fine.data.polygons)} faces, "
        f"{len(fine.vertex_groups)} weight groups carried over from the rig")
    return fine


def sample_heads(arm, frames, bones):
    """{frame: (n, 3)} world positions of `bones`' heads, posed."""
    out = {}
    for f in frames:
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        out[f] = np.array([(arm.matrix_world @ arm.pose.bones[b].head)[:] for b in bones])
    return out


def rescale(arm, ob, height: float, hips: str):
    """Height `height` with the lowest point on the floor, centred under the hips; returns (S, T)."""
    me = ob.data
    M = np.array(ob.matrix_world)
    co = np.array([v.co[:] for v in me.vertices]) @ M[:3, :3].T + M[:3, 3]
    zmin, zmax = float(co[:, 2].min()), float(co[:, 2].max())
    S = height / (zmax - zmin)
    head = arm.matrix_world @ arm.data.bones[hips].head_local
    origin = Vector((head.x, head.y, zmin))
    T = Matrix.Diagonal((S, S, S, 1.0)) @ Matrix.Translation(-origin)
    log(f"source height {zmax - zmin:.4f}, scale {S:.4f}, origin {tuple(round(c, 4) for c in origin)}")

    mw = ob.matrix_world.copy()
    ob.parent = None
    ob.matrix_world = Matrix.Identity(4)
    me.transform(T @ mw)
    arm.matrix_world = T @ arm.matrix_world
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    scaled = 0
    for action in bpy.data.actions:  # pose locations are in bone units, which just grew by S
        for _, fc in fcurve_slots(action):
            if fc.data_path.endswith(".location"):
                for k in fc.keyframe_points:
                    k.co[1] *= S
                    k.handle_left[1] *= S
                    k.handle_right[1] *= S
                fc.update()
                scaled += 1
    ob.parent = arm
    ob.matrix_parent_inverse = Matrix.Identity(4)
    for m in ob.modifiers:
        if m.type == "ARMATURE":
            m.object = arm
    log("location curves rescaled:", scaled)
    return S, T


def _cut_soles(ob, at: float, height: float, feet=None, clear_splinters: bool = False) -> dict:
    """Cuts off everything under the height `at`, closes each cut with a flat sole facing the floor (its corners take
    the colour and material of the shoe beside them), leaves out the inner wall's second cap inside each sole, and
    drops small bits the cut leaves hanging just above it (a strap of the made-up shoe). With `feet` (the foot and
    toe bones' points) it measures what thin plates the cut left in front of the boot (`game_ready.splinters`); with
    `clear_splinters` it drops such plates (`game_ready.splintered`, `game_ready.plate_faces`) and closes the boot's
    toe where they hung on (`_close_toes`), so the boot keeps its own length."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    faces_before = len(bm.faces)
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=(0.0, 0.0, at), plane_no=(0.0, 0.0, 1.0),
                                 clear_inner=True)
    rim = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge) and e.is_boundary]
    caps = bmesh.ops.holes_fill(bm, edges=rim, sides=0)["faces"]
    inner = gr.nested_caps([np.array([v.co[:2] for v in f.verts]) for f in caps])
    dropped = [f for f, x in zip(caps, inner) if x]
    kept = [f for f, x in zip(caps, inner) if not x]
    if dropped:
        bmesh.ops.delete(bm, geom=dropped, context="FACES_ONLY")
    for f in kept:
        f.normal_update()
        if f.normal.z > 0:
            f.normal_flip()
    soles = bmesh.ops.triangulate(bm, faces=kept, quad_method="BEAUTY", ngon_method="BEAUTY")["faces"]
    sole_set = set(soles)
    layers = list(bm.loops.layers.uv.values())
    for f in soles:
        f.smooth = False
        for e in f.edges:
            beside = [g for g in e.link_faces if g not in sole_set]
            if beside:
                e.smooth = False
                f.material_index = beside[0].material_index
        for loop in f.loops:
            other = next((lo for lo in loop.vert.link_loops if lo.face not in sole_set), None)
            if other is not None:
                for uv in layers:
                    loop[uv].uv = other[uv].uv.copy()
    bm.faces.index_update()
    small = max(60, faces_before // 1000)
    loose = [c for c in shells(bm) if len(c) < small and max(v.co.z for f in c for v in f.verts) < at + 0.04 * height]
    if loose:
        bmesh.ops.delete(bm, geom=list({v for c in loose for f in c for v in f.verts}), context="VERTS")
    bm.normal_update()
    stats = {"soles": len(kept), "inner_caps_left_out": len(dropped), "sole_triangles": len(soles),
             "loose_bits_dropped": len(loose)}
    if feet is not None:
        bm.verts.index_update()
        points = np.array([v.co[:] for v in bm.verts])
        tris = np.array([[lo.vert.index for lo in t] for t in bm.calc_loop_triangles()], dtype=np.int64)
        stats["splinters"] = gr.splinters(points, tris, floor=at, height=height, feet=feet)
        if clear_splinters and gr.splintered(stats["splinters"], height):
            triangles = bm.calc_loop_triangles()
            drop = gr.plate_faces(points, tris, floor=at, height=height, feet=feet)
            whole = {}
            for t, d in zip(triangles, drop):
                whole[t[0].face] = whole.get(t[0].face, True) and bool(d)
            gone = [f for f, d in whole.items() if d]
            bmesh.ops.delete(bm, geom=gone, context="FACES_ONLY")
            bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
            stats["plates_cleared"] = {"faces": len(gone), "toe_faces": _close_toes(bm)}
            bm.verts.index_update()
            points = np.array([v.co[:] for v in bm.verts])
            tris = np.array([[lo.vert.index for lo in t] for t in bm.calc_loop_triangles()], dtype=np.int64)
            stats["splinters_after"] = gr.splinters(points, tris, floor=at, height=height, feet=feet)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return stats


def _close_toes(bm) -> int:
    """Closes the holes clearing the plates left at a boot's toe: each loop of open edges becomes faces wound like the
    surface beside them, with its material, its edges hard, and the colour of one face beside it (the corners it
    shares with that face keep their UVs there, the others take that face's middle: corners taken from any face
    around a vertex reached across UV seams into the glow, D210)."""
    open_edges = [e for e in bm.edges if e.is_boundary]
    if not open_edges:
        return 0
    filled = bmesh.ops.holes_fill(bm, edges=open_edges, sides=0)["faces"]
    filled = bmesh.ops.triangulate(bm, faces=filled, quad_method="BEAUTY", ngon_method="BEAUTY")["faces"]
    new = set(filled)
    layers = list(bm.loops.layers.uv.values())
    for f in filled:
        source = None
        for loop in f.loops:
            across = next((lo for lo in loop.edge.link_loops if lo.face not in new), None)
            if across is not None:
                if across.vert == loop.vert:  # the same direction along the shared edge: wound the other way
                    f.normal_flip()
                source = across.face
                break
        if source is None:  # a fill face inside a bigger fill: the colour of a face around one of its corners
            source = next((lo.face for loop in f.loops for lo in loop.vert.link_loops if lo.face not in new), None)
        f.smooth = False
        for e in f.edges:
            e.smooth = False
        if source is None:
            continue
        f.material_index = source.material_index
        for uv in layers:
            middle = sum((lo[uv].uv for lo in source.loops), source.loops[0][uv].uv * 0) / len(source.loops)
            corners = {lo.vert: lo[uv].uv.copy() for lo in source.loops}
            for loop in f.loops:
                loop[uv].uv = corners.get(loop.vert, middle)
    return len(filled)


def feet_move(me, plan: dict, height: float, T) -> None:
    """Moves a mesh in the figure's frame (floor at 0, `height` tall) by the feet plan, then by `T`."""
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co[:, 2] = gr.feet_heights(co[:, 2] / height, plan) * height
    me.vertices.foreach_set("co", co.ravel())
    me.transform(T)
    me.update()


def fix_feet(arm, ob, plan: dict, s: dict, hips: str, feet_bones=()):
    """The feet plan of `--feet-from` (`game_ready.feet_plan`), right after the rescale: a made-up sole is cut off or
    pressed flat, the bones go with it (a toe bone that sat in the cut part is laid on the new sole), and the figure is
    stood on its new soles and scaled back to its height. A cut chosen by `--feet auto` that leaves thin plates of the
    made-up shoe in front of the boot (measured near `feet_bones`) clears them and closes the toe.
    Returns (stats, the second rescale's T, the bones that moved otherwise than the figure as a whole, which the
    moves' check leaves out, the height it measured the feet at, the plan as done)."""
    me = ob.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    height = float(co.reshape(-1, 3)[:, 2].max())  # the rescale put the floor at 0
    at = plan["at"] * height
    stats = {"mode": plan["mode"], "at_m": round(at, 4), "at": plan["at"], "parts": plan["parts"]}
    feet = [p for name in feet_bones if name in arm.data.bones
            for p in (arm.data.bones[name].head_local[:], arm.data.bones[name].tail_local[:])]
    if plan["mode"] == "cut":
        cut = _cut_soles(ob, at, height, feet or None, clear_splinters=bool(plan.get("auto")))
        stats.update(cut)
        if cut.get("plates_cleared"):
            stats["cleared"] = "the cut left thin plates of the made-up shoe in front of the boot: cleared"
    if plan["mode"] == "squash":
        stats["sole_m"] = round(plan["sole"] * height, 4)
    feet_move(me, plan, height, Matrix.Identity(4))
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    moved = []
    for eb in arm.data.edit_bones:
        if min(eb.head.z, eb.tail.z) < at:
            moved.append(eb.name)
        for end in ("head", "tail"):
            point = getattr(eb, end).copy()
            point.z = float(gr.feet_heights(point.z / height, plan)) * height
            setattr(eb, end, point)
    bpy.ops.object.mode_set(mode="OBJECT")
    S, T = rescale(arm, ob, s["height"], hips)
    stats.update(scale=round(S, 4), bones_moved=moved)
    log("feet:", stats)
    return stats, T, moved, height, plan


def shells(bm):
    bm.faces.ensure_lookup_table()
    done, out = set(), []
    for start in bm.faces:
        if start.index in done:
            continue
        comp, k = [start], 0
        done.add(start.index)
        while k < len(comp):
            for e in comp[k].edges:
                for g in e.link_faces:
                    if g.index not in done:
                        done.add(g.index)
                        comp.append(g)
            k += 1
        out.append(comp)
    return out


def clean_mesh(ob, s: dict) -> dict:
    me = ob.data
    if "custom_normal" in me.attributes:
        me.attributes.remove(me.attributes["custom_normal"])
    bm = bmesh.new()
    bm.from_mesh(me)
    nv0, nf0 = len(bm.verts), len(bm.faces)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    nv1 = len(bm.verts)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-5)
    seen, dups = {}, []
    for f in bm.faces:
        key = tuple(sorted(v.index for v in f.verts))
        if key in seen:
            dups.append(f)
        else:
            seen[key] = f
    if dups:
        bmesh.ops.delete(bm, geom=dups, context="FACES")
    bm.faces.index_update()
    parts = [(c, sum(f.calc_area() for f in c)) for c in shells(bm)]
    parts.sort(key=lambda t: -t[1])
    tiny = [c for c, area in parts[1:] if area * 1e4 < s["tiny_area_cm2"] or len(c) <= s["tiny_faces"]]
    if tiny:
        bmesh.ops.delete(bm, geom=list({v for c in tiny for f in c for v in f.verts}), context="VERTS")
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.normal_update()
    bad = sum(1 for e in bm.edges if len(e.link_faces) == 2 and e.link_loops[0].vert == e.link_loops[1].vert)
    flipped = 0
    for c in shells(bm):
        if not all(len(e.link_faces) == 2 for f in c for e in f.edges):
            continue
        if sum(f.calc_center_median().dot(f.normal) * f.calc_area() for f in c) < 0:
            for f in c:
                f.normal_flip()
            flipped += 1
    if bad:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    limit = math.radians(s["sharp_deg"])
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        e.smooth = not (len(e.link_faces) != 2 or e.calc_face_angle(0.0) > limit)
    stats = {"vertices_in": nv0, "faces_in": nf0, "welded": nv0 - nv1, "duplicate_faces": len(dups),
             "tiny_pieces_deleted": len(tiny), "loose_vertices": len(loose), "inconsistent_winding_edges": bad,
             "inside_out_pieces_fixed": flipped, "pieces": len(parts) - len(tiny),
             "open_edges": sum(1 for e in bm.edges if len(e.link_faces) == 1),
             "non_manifold_edges": sum(1 for e in bm.edges if len(e.link_faces) > 2)}
    bm.to_mesh(me)
    bm.free()
    me.update()
    log("cleaned:", stats)
    return stats


def head_weights(ob, head: list[str]) -> np.ndarray:
    """Per vertex, its skin weight on the head bones (0..1), through `game_ready.head_share`."""
    me = ob.data
    columns = [g.name for g in ob.vertex_groups]
    W = np.zeros((len(me.vertices), max(len(columns), 1)))
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    return gr.head_share(W, columns, head or [])


def decimate_with_head(ob, s: dict, head: list[str]) -> dict:
    """Collapse-thin to `decimate_to` triangles; the head's vertices resist by `head_protect`."""
    me = ob.data
    me.calc_loop_triangles()
    before = len(me.loop_triangles)
    ratio = gr.decimate_ratio(before, s["decimate_to"])
    if ratio >= 1.0:
        return {"decimated": [before, before]}
    share = head_weights(ob, head)
    mod = ob.modifiers.new("gr_decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    if s["head_protect"] > 0 and share.any():
        group = ob.vertex_groups.new(name="gr_head_protect")
        for i in np.nonzero(share)[0]:
            group.add([int(i)], float(share[i]), "REPLACE")
        mod.vertex_group = group.name
        mod.vertex_group_factor = s["head_protect"]
        mod.invert_vertex_group = True   # Blender thins where a group weighs; inverted, the head weighs nothing
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    stale = ob.vertex_groups.get("gr_head_protect")   # the old handle is stale after the apply
    if stale is not None:
        ob.vertex_groups.remove(stale)
    me.calc_loop_triangles()
    after = len(me.loop_triangles)
    log(f"thinned {before} -> {after} triangles (head protect {s['head_protect']:g})")
    return {"decimated": [before, after]}


def _cast_with(tree, eps: float):
    """A ray test for `hidden_from_rays`: whether a ray from a face meets the mesh (its own face skipped)."""
    def cast(origin, direction, face):
        location, _n, index, _d = tree.ray_cast(origin, direction)
        if index == face:
            location, _n, index, _d = tree.ray_cast(location + direction * eps, direction)
        return location is not None
    return cast


def _hidden_at_samples(bm, which: np.ndarray, directions, cast, eps: float, reach: float) -> np.ndarray:
    """Of the faces `which` of `bm`, those from which no ray reaches the outside at any of four points
    (`face_samples`: the centre, then three points toward the corners; a face seen at one stops there)."""
    tris = np.array([[tuple(v.co) for v in bm.faces[int(i)].verts][:3] for i in which]).reshape(-1, 3, 3)
    samples = face_samples(tris, reach)
    hidden = np.ones(len(which), dtype=bool)
    for k in range(samples.shape[1]):
        ask = np.flatnonzero(hidden)
        if not len(ask):
            break
        hidden[ask] = hidden_from_rays([Vector(p) for p in samples[ask, k]], directions,
                                       lambda o, d, j, ask=ask: cast(o, d, int(which[ask[j]])), eps)
    return hidden


def _first_hit_with(tree, eps: float):
    """`first_hit` for `poking_out`: the first face a ray meets within `reach` (its own face skipped), as
    (distance, that face's normal by its winding), or None."""
    def first_hit(origin, direction, face, reach):
        location, normal, index, distance = tree.ray_cast(origin, direction, reach)
        if index == face and location is not None:
            location, normal, index, more = tree.ray_cast(location + direction * eps, direction, reach - distance)
            distance = None if location is None else distance + more
        if location is None:
            return None
        return distance, np.array(tuple(normal))
    return first_hit


def _pokes(bm, s: dict, size: float, eps: float, which=None) -> np.ndarray:
    """Of the faces `which` of `bm` (all by default), those that are inner wall poking out past the outer surface
    (`poking_out`), by their own winding as the mesh stands now."""
    bm.normal_update()
    tree = BVHTree.FromBMesh(bm)
    which = np.arange(len(bm.faces)) if which is None else np.asarray(which, dtype=int)
    tris = np.array([[tuple(loop.vert.co) for loop in bm.faces[int(i)].loops][:3] for i in which]).reshape(-1, 3, 3)
    normals = np.array([tuple(bm.faces[int(i)].normal) for i in which]).reshape(-1, 3)
    vec_cast = _cast_with(tree, eps)
    vec_first = _first_hit_with(tree, eps)
    return poking_out(face_samples(tris, s["hidden_corner_reach"]), normals, fibonacci_directions(s["poke_rays"]),
                      lambda o, d, f: vec_cast(Vector(o), Vector(d), int(which[f])),
                      lambda o, d, f, r: vec_first(Vector(o), Vector(d), int(which[f]), r), s["poke_near"] * size, eps)


def sky_roofs(ob, s: dict) -> np.ndarray:
    """The faces of `ob` (at rest) that may not shade the surface in the sky-light bake (`shading_roofs`): pieces over
    the surface that face it and see the sky behind them; a game shows the surface under them from outside."""
    bm = _world_bmesh(ob.data, ob.matrix_world)
    bm.normal_update()
    size = max(max(v.co[k] for v in bm.verts) - min(v.co[k] for v in bm.verts) for k in range(3))
    eps = size * 1e-5
    tree = BVHTree.FromBMesh(bm)
    tris = np.array([[tuple(loop.vert.co) for loop in f.loops][:3] for f in bm.faces]).reshape(-1, 3, 3)
    normals = np.array([tuple(f.normal) for f in bm.faces]).reshape(-1, 3)
    vec_cast, vec_first = _cast_with(tree, eps), _first_hit_with(tree, eps)
    roofs = shading_roofs(face_samples(tris, s["hidden_corner_reach"]), normals, fibonacci_directions(s["poke_rays"]),
                          lambda o, d, f: vec_cast(Vector(o), Vector(d), f),
                          lambda o, d, f, r: vec_first(Vector(o), Vector(d), f, r), s["roof_near"] * size, eps)
    bm.free()
    return roofs


def _world_bmesh(mesh, matrix):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.transform(matrix)
    bm.faces.ensure_lookup_table()
    return bm


def _pose_on(ob, arm) -> list:
    """Let the armature move the mesh (for posing); returns what to put back with `_pose_off`."""
    skins = [m for m in ob.modifiers if m.type == "ARMATURE"]
    shown = [(m, m.show_viewport) for m in skins]
    for m in skins:
        m.show_viewport = True
    arm.data.pose_position = "POSE"
    return shown


def _pose_off(arm, shown: list) -> None:
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    arm.data.pose_position = "REST"
    for m, was in shown:
        m.show_viewport = was
    bpy.context.view_layer.update()


def _posed_bmesh(ob, arm, moves, total: int):
    """The mesh in world space with the figure posed by `moves` ([(bone, axis, degrees), ...], each bone turned about
    a world axis through its head), or None when a modifier changed the topology (the pose then vouches for nothing)."""
    def rotate(name, axis, degrees):
        if name not in arm.pose.bones:
            return
        bpy.context.view_layer.update()
        pb = arm.pose.bones[name]
        head = arm.matrix_world @ pb.head
        turn = Matrix.Translation(head) @ Matrix.Rotation(math.radians(degrees), 4, Vector(axis)) \
            @ Matrix.Translation(-head)
        pb.matrix = arm.matrix_world.inverted() @ turn @ arm.matrix_world @ pb.matrix
        bpy.context.view_layer.update()

    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    for bone, axis, degrees in moves:
        rotate(bone, axis, degrees)
    evaluated = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    posed = _world_bmesh(evaluated.to_mesh(), ob.matrix_world)
    evaluated.to_mesh_clear()
    if len(posed.faces) != total:
        posed.free()
        return None
    return posed


def _first_front(tree, front_of, origin: Vector, direction: Vector, skip, eps: float, reach: float):
    """The first face a ray meets whose front faces it (back faces are not drawn, so the ray goes on through them),
    skipping the faces `skip` marks: (face, distance) or (-1, inf)."""
    travelled = 0.0
    at = origin
    for _ in range(24):
        location, _n, index, distance = tree.ray_cast(at, direction, reach - travelled)
        if location is None:
            return -1, math.inf
        travelled += distance
        if front_of(index) and not skip[index]:
            return index, travelled
        at = location + direction * eps
        travelled += eps
    return -1, math.inf


def pose_check(ob, arm, s: dict, poses: list, doomed: np.ndarray, total: int, picture: str | None) -> tuple[dict, np.ndarray]:
    """The figure with and without the faces `doomed` marks, posed in each of `poses` and seen from the twelve sides of
    `gr.CHECK_VIEWS`: a ray that first meets a dropped face and, without it, nothing close behind shows a hole. Returns
    the counts and the dropped faces seen through the holes (put back by the caller); the worst view goes to `picture`."""
    res = s["pose_check_res"]
    seen = np.zeros(total, dtype=bool)
    hole_px = views_with_holes = 0
    worst = (0, None, None)
    nothing = np.zeros(total, dtype=bool)
    shown = _pose_on(ob, arm)
    try:
        for name, moves in poses:
            posed = _posed_bmesh(ob, arm, moves, total)
            if posed is None:
                continue
            posed.normal_update()
            tree = BVHTree.FromBMesh(posed)
            normals = np.array([tuple(f.normal) for f in posed.faces])
            coords = np.array([tuple(v.co) for v in posed.verts])
            lo, hi = coords.min(0), coords.max(0)
            center = (lo + hi) / 2
            radius = float(np.linalg.norm(hi - lo)) / 2 * 1.02
            eps = radius * 2e-5
            gap = s["pose_check_gap"] * radius * 2
            posed.free()
            for azimuth, elevation in gr.CHECK_VIEWS:
                origins, direction = gr.view_rays(center, radius, azimuth, elevation, res)
                d = Vector(direction)
                facing = normals @ direction < 0
                front_of = facing.__getitem__
                before_face = np.full(len(origins), -1)
                before_depth = np.full(len(origins), math.inf)
                after_depth = np.full(len(origins), math.inf)
                for k, o in enumerate(origins):
                    origin = Vector(o)
                    face, depth = _first_front(tree, front_of, origin, d, nothing, eps, 4 * radius)
                    before_face[k], before_depth[k] = face, depth
                    if face >= 0 and doomed[face]:
                        after_depth[k] = _first_front(tree, front_of, origin, d, doomed, eps, 4 * radius)[1]
                hole = gr.pose_holes(before_face, before_depth, after_depth, doomed, gap)
                count = int(hole.sum())
                if count:
                    views_with_holes += 1
                    hole_px += count
                    seen[gr.faces_seen_in(before_face, hole)] = True
                    if count > worst[0]:
                        image = np.zeros((res * res, 3), np.float32)
                        image[before_face >= 0] = 0.55
                        image[hole] = (1.0, 0.1, 0.1)
                        worst = (count, f"{name}, {azimuth} deg round, {elevation} deg up", image.reshape(res, res, 3))
    finally:
        _pose_off(arm, shown)
    written = None
    if worst[2] is not None and picture:
        big = np.repeat(np.repeat(worst[2], 3, axis=0), 3, axis=1)   # three times larger, to be read
        gr.write_png(Path(picture), big[::-1])   # write_png takes row 0 as the bottom
        written = picture
    restore = np.flatnonzero(seen)
    report = {"poses": len(poses), "views": len(gr.CHECK_VIEWS), "rays_per_view": res * res, "hole_px": hole_px,
              "views_with_holes": views_with_holes, "restored": int(len(restore)), "worst": worst[1],
              "picture": written}
    return report, restore


def drop_hidden_wall(ob, arm, s: dict, poses: list, check: list | None = None, picture: str | None = None,
                     final=lambda left: True) -> dict:
    """Delete the triangles nobody can see: no ray from them reaches the outside, the air beside them is a
    sealed, compact hollow (the body's inside, not the pocket behind a cape), and they stay hidden in every
    pose of `poses` ([name, [(bone, axis, degrees), ...]]: the figure posed by its own bones)."""
    bm = _world_bmesh(ob.data, ob.matrix_world)
    total = len(bm.faces)
    size = max(max(v.co[k] for v in bm.verts) - min(v.co[k] for v in bm.verts) for k in range(3))
    eps = size * 1e-5
    directions = [Vector(d) for d in fibonacci_directions(s["hidden_rays"])]
    reach = s["hidden_corner_reach"]
    hidden = _hidden_at_samples(bm, np.arange(total), directions, _cast_with(BVHTree.FromBMesh(bm), eps), eps, reach)
    enclosed = int(hidden.sum())
    if enclosed:
        tris = np.array([[tuple(loop.vert.co) for loop in f.loops][:3] for f in bm.faces])
        grid = hollows_only(sealed_air(dense_surface(tris, 0.5 * size / s["hidden_cells"]),
                                       size / s["hidden_cells"], close=s["hidden_close"]))
        which = np.flatnonzero(hidden)
        at = np.array([tuple(bm.faces[int(i)].calc_center_median()) for i in which])
        normal = np.array([tuple(bm.faces[i].normal) for i in which])
        hidden[which] = first_air_sealed(grid, at, normal) | first_air_sealed(grid, at, -normal)
    # inner wall poking out past the outer surface: never drawn (its back faces the sky, no camera sees its front),
    # but it shades the surface under it in the sky-light bake. Asked while the inner wall is still there: once it
    # is gone, the back of a seen face (a collar, an elbow) can look through the opened shell. Like the hidden wall
    # it goes only if every pose keeps it so: a looser rule without the poses opened holes in hair and at the
    # joints (round 9 of the AssetFurnace Support notes).
    poke = _pokes(bm, s, size, eps) & ~hidden
    bm.free()

    vetoed = poke_vetoed = 0
    if (hidden.any() or poke.any()) and poses:
        shown = _pose_on(ob, arm)
        for _name, moves in poses:
            posed = _posed_bmesh(ob, arm, moves, total)
            if posed is None:   # a modifier changed the topology: this pose cannot vouch for anything
                continue
            which = np.flatnonzero(hidden)
            if len(which):
                still = _hidden_at_samples(posed, which, directions, _cast_with(BVHTree.FromBMesh(posed), eps), eps,
                                           reach)
                vetoed += int((~still).sum())
                hidden[which[~still]] = False
            which = np.flatnonzero(poke)
            if len(which):
                still = _pokes(posed, s, size, eps, which)
                poke_vetoed += int((~still).sum())
                poke[which[~still]] = False
            posed.free()
        _pose_off(arm, shown)

    doomed_mask = hidden | poke
    checked = None
    if check and doomed_mask.any() and final(total - int(doomed_mask.sum())):
        # the pose check (#153 of the AssetFurnace Support notes): what a game would show of the dropped faces in
        # other moves is a hole; those faces stay
        checked, restore = pose_check(ob, arm, s, check, doomed_mask, total, picture)
        doomed_mask[restore] = False
        checked["hole_px_left"] = 0
        if len(restore):
            again, _more = pose_check(ob, arm, s, check, doomed_mask, total, None)
            checked["hole_px_left"] = again["hole_px"]
        log(f"pose check: {checked['hole_px']} px of holes in {checked['views_with_holes']} views, "
            f"{checked['restored']} faces put back, {checked['hole_px_left']} px left")
    doomed = np.flatnonzero(doomed_mask)
    _delete_faces(ob, doomed)
    result = {"faces": total, "enclosed": enclosed, "vetoed": vetoed, "poking_out": int(poke.sum()),
              "poking_kept_for_poses": poke_vetoed, "dropped": int(len(doomed)), "left": total - int(len(doomed))}
    if checked is not None:
        result["pose_check"] = checked
    return result


def _delete_faces(ob, faces) -> None:
    """Delete the faces `faces` (indices) of `ob`'s mesh and the vertices left without a face."""
    if not len(faces):
        return
    local = bmesh.new()
    local.from_mesh(ob.data)
    local.faces.ensure_lookup_table()
    bmesh.ops.delete(local, geom=[local.faces[int(i)] for i in faces], context="FACES")
    bmesh.ops.delete(local, geom=[v for v in local.verts if not v.link_faces], context="VERTS")
    local.to_mesh(ob.data)
    local.free()
    ob.data.update()


def thin_and_drop(ob, arm, s: dict, head: list[str], poses: list, check: list | None = None,
                  picture: str | None = None) -> dict:
    """--drop-hidden: thin to a first target above the budget, drop the hidden inner wall, and aim the next try
    by what the drop left, until the figure lands within `drop_slack` of `decimate_to` (at most `drop_tries`).
    Each try starts again from the cleaned, unthinned mesh: thinning what a drop left would spend triangles on
    the drop's open rims."""
    target = s["decimate_to"]
    original = ob.data.copy()
    me = ob.data
    me.calc_loop_triangles()
    before = len(me.loop_triangles)
    first = gr.first_target(target)
    tries = []
    for attempt in range(s["drop_tries"]):
        if attempt:
            spent = ob.data
            ob.data = original.copy()
            bpy.data.meshes.remove(spent)
        thinned = decimate_with_head(ob, dict(s, decimate_to=first), head)["decimated"][1]
        last = attempt == s["drop_tries"] - 1
        drop = drop_hidden_wall(ob, arm, s, poses, check, picture,
                                final=lambda left, last=last: last or gr.drop_settled(left, target, s["drop_slack"]))
        tries.append({"first": first, "thinned": thinned, **{k: v for k, v in drop.items() if k != "pose_check"}})
        log(f"hidden wall, try {attempt + 1}: thinned to {thinned}, {drop['enclosed']} enclosed, "
            f"{drop['vetoed']} kept for the poses, {drop['poking_out']} poking out "
            f"({drop['poking_kept_for_poses']} kept for the poses), {drop['dropped']} dropped, "
            f"{drop['left']} left")
        if "pose_check" in drop or gr.drop_settled(drop["left"], target, s["drop_slack"]):
            break
        first = gr.next_first_target(first, drop["left"], target)
    bpy.data.meshes.remove(original)
    ob.data.calc_loop_triangles()
    out = {"decimated": [before, len(ob.data.loop_triangles)], "hidden_wall": tries}
    if "pose_check" in drop:
        out["pose_check"] = drop["pose_check"]
    return out


def scale_head_islands(ob, factor: float, head: list[str]):
    """In edit mode, after unwrapping and before packing: the head's islands grow by `factor`.
    Returns the head's share of the UV area afterwards, or None when nothing changed."""
    if factor <= 1.0 or not head:
        return None
    me = ob.data
    share_v = head_weights(ob, head)
    bm = bmesh.from_edit_mesh(me)
    uvl = bm.loops.layers.uv.active
    bm.faces.ensure_lookup_table()

    def same(a, b):
        return (a[uvl].uv - b.link_loop_next[uvl].uv).length < 1e-6 and (a.link_loop_next[uvl].uv - b[uvl].uv).length < 1e-6

    loops, island, done, n = [], [], set(), 0
    for start in bm.faces:
        if start.index in done:
            continue
        stack = [start]
        done.add(start.index)
        while stack:
            f = stack.pop()
            for loop in f.loops:
                loops.append(loop)
                island.append(n)
                for other in loop.link_loops:
                    if other.face.index not in done and same(loop, other):
                        done.add(other.face.index)
                        stack.append(other.face)
        n += 1
    uv = np.array([lp[uvl].uv[:] for lp in loops])
    share = np.array([share_v[lp.vert.index] for lp in loops])
    chosen = gr.head_islands(island, share)
    out = gr.scale_islands(uv, island, chosen, factor)
    for lp, (u, v) in zip(loops, out):
        lp[uvl].uv = (u, v)
    bmesh.update_edit_mesh(me)
    area = {}
    for f in bm.faces:
        pts = [lp[uvl].uv for lp in f.loops]
        a = sum(abs((pts[k] - pts[0]).cross(pts[k + 1] - pts[0])) for k in range(1, len(pts) - 1)) / 2
        head_face = np.mean([share_v[v.index] for v in f.verts]) >= 0.5
        area[head_face] = area.get(head_face, 0.0) + a
    total = sum(area.values()) or 1.0
    log(f"head islands: {len(chosen)} grown x{factor:g}, the head now takes {area.get(True, 0.0) / total * 100:.1f} % of the texels")
    return round(area.get(True, 0.0) / total, 4)


def material_image_sizes(o) -> list:
    """Per material slot of `o`, the size of the picture that colours it (the image feeding Base Color, else the
    largest image in the material), or None."""
    sizes = []
    for mat in o.data.materials:
        best = None
        if mat and mat.use_nodes and mat.node_tree:
            nt = mat.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            node = bsdf.inputs["Base Color"].links[0].from_node if bsdf and bsdf.inputs["Base Color"].is_linked else None
            if node is not None and node.type == "TEX_IMAGE" and node.image:
                images = [node.image]
            else:
                images = [n.image for n in nt.nodes if n.type == "TEX_IMAGE" and n.image]
            for img in images:
                w, h = (int(c) for c in img.size[:2])
                if w and h and (best is None or w * h > best[0] * best[1]):
                    best = (w, h)
        sizes.append(best)
    return sizes


def mesh_triangles(o):
    """World positions, triangles as vertex indices and as loop indices (a fan per polygon), the active UVs per
    loop and each triangle's material slot."""
    me = o.data
    nv, nl, npoly = len(me.vertices), len(me.loops), len(me.polygons)
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    M = np.array(o.matrix_world)
    co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    uv = np.zeros(nl * 2)
    if me.uv_layers.active:
        me.uv_layers.active.data.foreach_get("uv", uv)
    loop_vert = np.empty(nl, np.int64)
    me.loops.foreach_get("vertex_index", loop_vert)
    start, total, mat = (np.empty(npoly, np.int64) for _ in range(3))
    me.polygons.foreach_get("loop_start", start)
    me.polygons.foreach_get("loop_total", total)
    me.polygons.foreach_get("material_index", mat)
    tris, poly = [], []
    for k in range(1, int(total.max()) - 1 if npoly else 1):
        m = total > k + 1
        s0 = start[m]
        tris.append(np.stack([s0, s0 + k, s0 + k + 1], 1))
        poly.append(np.nonzero(m)[0])
    loop_tris = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    poly = np.concatenate(poly) if poly else np.zeros(0, np.int64)
    return co, loop_vert[loop_tris], uv.reshape(-1, 2), loop_tris, mat[poly]


def source_head_density(ob, head: list[str], colour_objs=None) -> float:
    """--head-texel auto: texels per cm the figure's colour source holds on the head. The source is `colour_objs`
    (--colour-from, the faces whose centre lies in the box around the head's vertices) or the figure itself before
    its new UVs (its head faces). It stands for what the picture gave: Pixel Match paints the picture into it."""
    share_v = head_weights(ob, head)
    if colour_objs:
        co, _, _, _, _ = mesh_triangles(ob)
        on = co[share_v >= 0.5]
        if not len(on):
            return 0.0
        lo, hi = on.min(0), on.max(0)
    texels = cm2 = 0.0
    for o in colour_objs or [ob]:
        sizes = material_image_sizes(o)
        if not o.data.uv_layers or not any(sizes):
            continue
        co, vert_tris, uv, loop_tris, mat = mesh_triangles(o)
        if colour_objs:
            centre = co[vert_tris].mean(1)
            picked = np.all((centre >= lo) & (centre <= hi), axis=1)
        else:
            picked = share_v[vert_tris].mean(1) >= 0.5
        p = co[vert_tris]
        area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2 * 1e4
        for slot, size in enumerate(sizes):
            pick = np.nonzero(picked & (mat == slot))[0]
            if size is None or not len(pick):
                continue
            d = gr.texel_density(co, vert_tris, uv, loop_tris, size[0], size[1], pick)
            a = float(area[pick].sum())
            texels += d * d * a
            cm2 += a
    return math.sqrt(texels / cm2) if cm2 else 0.0


def pack_uvs(s: dict) -> None:
    bpy.ops.uv.select_all(action="SELECT")
    try:
        bpy.ops.uv.pack_islands(rotate=True, rotate_method="ANY", scale=True, merge_overlap=False,
                                margin_method="ADD", margin=gr.uv_margin(s), shape_method="CONCAVE")
    except (TypeError, RuntimeError) as exc:
        log("concave packing unavailable, plain packing:", exc)
        bpy.ops.uv.pack_islands(rotate=True, margin=gr.uv_margin(s))


def auto_head_texel(ob, s: dict, head: list[str], auto: dict) -> float:
    """--head-texel auto, in edit mode after unwrapping: pack once with the head at factor 1, measure the head's
    texels per cm at the texture size the game loads, and pick the factor that gives it what the colour source
    holds (`game_ready.auto_head_factor`). The real pack follows with the head grown by it."""
    pack_uvs(s)
    share_v = head_weights(ob, head)
    bm = bmesh.from_edit_mesh(ob.data)
    uvl = bm.loops.layers.uv.active
    uv_area = {True: 0.0, False: 0.0}
    head_cm2 = 0.0
    for f in bm.faces:
        pts = [lp[uvl].uv for lp in f.loops]
        a = sum(abs((pts[k] - pts[0]).cross(pts[k + 1] - pts[0])) for k in range(1, len(pts) - 1)) / 2
        on = bool(np.mean([share_v[v.index] for v in f.verts]) >= 0.5)
        uv_area[on] += a
        if on:
            head_cm2 += f.calc_area() * 1e4
    size = s.get("game_texture") or s["atlas"]
    at_one = math.sqrt(uv_area[True] * size * size / head_cm2) if head_cm2 else 0.0
    factor = round(gr.auto_head_factor(auto["source_texels_per_cm"], at_one, uv_area[True], uv_area[False]), 2)
    auto.update(game_texture=size, head_texels_per_cm_at_1=round(at_one, 2), factor=factor)
    log(f"head texel auto: the colour source holds {auto['source_texels_per_cm']:.1f} texels per cm on the head, "
        f"a {size} map gives it {at_one:.1f} at factor 1, so x{factor:g}")
    return factor


def pin_source_uvs(me) -> None:
    """Make every texture read the old UVs by name, before the new layout becomes the active one."""
    me.uv_layers[0].name = "src"
    for mat in me.materials:
        if not (mat and mat.node_tree):
            continue
        nt = mat.node_tree
        for n in list(nt.nodes):
            if n.type == "TEX_IMAGE" and not n.inputs["Vector"].is_linked:
                uv = nt.nodes.new("ShaderNodeUVMap")
                uv.uv_map = "src"
                nt.links.new(uv.outputs["UV"], n.inputs["Vector"])
            elif n.type == "NORMAL_MAP":
                n.uv_map = "src"


def unfold_overlaps(bpy, me, angle: float, margin: float, rounds: int = rb.UV_ROUNDS) -> int | None:
    """Smart UV Project's islands, with only the faces that land on other faces unwrapped again.

    At game_ready's wide angle Smart UV Project folds an island onto itself where the surface
    turns, and the bake then writes one face over another: hidden faces from inside over a
    cape's back gave a dark, faceted patch (black sky light, another colour). The Finish's
    `unwrap_islands` unwraps every island conformally, which here left a third of the atlas
    empty (39 % used instead of 55 %); so the projection's compact islands stay, and only the
    overlapping faces are cut along their islands and unwrapped conformally, a few rounds, the
    last few each an island of its own. Returns the faces still overlapping, or None on a
    Blender whose overlap check crashes (it projects only, as before). Call in edit mode with
    every face selected; the select mode is given back.
    """
    def project():
        bpy.ops.uv.smart_project(angle_limit=angle, island_margin=0.0, area_weight=0.0,
                                 correct_aspect=True, scale_to_bounds=False)

    if rb.unwrap_crashes(bpy.app.version):
        project()
        return None
    settings = bpy.context.scene.tool_settings
    synced, mode = settings.use_uv_select_sync, tuple(settings.mesh_select_mode)
    settings.use_uv_select_sync = True   # UV operators then select mesh faces
    settings.mesh_select_mode = (False, False, True)
    try:
        project()

        def overlapping() -> int:
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.pack_islands(rotate=False, margin=margin)   # islands apart: what overlaps now is a fold
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.uv.select_overlap()
            count, added = rb.overlap_selection(bmesh.from_edit_mesh(me))
            if added:
                bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
            return count

        for _ in range(rounds):
            if not overlapping():
                break
            bpy.ops.uv.seams_from_islands()
            bpy.ops.uv.unwrap(method="CONFORMAL", margin=margin)   # the overlapping faces only
        if overlapping():
            bpy.ops.mesh.mark_seam(clear=False)   # the last few: each an island of its own
            bpy.ops.uv.unwrap(method="CONFORMAL", margin=margin)
        # the unwrap scales its islands on their own; give every island the same texels per square
        # metre again, or the spots that folded come out blockier (or sharper) than the rest
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.average_islands_scale()
        left = overlapping()
        bpy.ops.mesh.select_all(action="SELECT")
        return left
    finally:
        settings.use_uv_select_sync = synced
        settings.mesh_select_mode = mode


def new_uvs(ob, s: dict, head: list[str] | None = None, colour_objs: list | None = None) -> dict:
    me = ob.data
    auto = None
    if s.get("head_texel") == "auto" and head:
        auto = {"source_texels_per_cm": round(source_head_density(ob, head, colour_objs), 2)}
    pin_source_uvs(me)
    layer = me.uv_layers.new(name="UVMap")
    me.uv_layers.active = layer
    layer.active_render = True
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    overlapping = unfold_overlaps(bpy, me, math.radians(s["uv_angle"]), gr.uv_margin(s))
    factor = s.get("head_texel", 1.0)
    if factor == "auto":
        factor = auto_head_texel(ob, s, head or [], auto) if auto else 1.0
    head_uv_share = scale_head_islands(ob, factor, head or [])
    pack_uvs(s)
    bpy.ops.object.mode_set(mode="OBJECT")

    bm = bmesh.new()
    bm.from_mesh(me)
    uvl = bm.loops.layers.uv["UVMap"]
    bm.faces.ensure_lookup_table()

    def same(a, b):
        return (a[uvl].uv - b.link_loop_next[uvl].uv).length < 1e-5 and (a.link_loop_next[uvl].uv - b[uvl].uv).length < 1e-5

    done, islands = set(), 0
    for start in bm.faces:
        if start.index in done:
            continue
        islands += 1
        stack = [start]
        done.add(start.index)
        while stack:
            f = stack.pop()
            for loop in f.loops:
                for other in loop.link_loops:
                    if other.face.index not in done and same(loop, other):
                        done.add(other.face.index)
                        stack.append(other.face)
    hard = seams = 0
    for e in bm.edges:
        if len(e.link_loops) != 2:
            e.smooth = False
            continue
        a, b = e.link_loops
        seam = not same(a, b)
        angle = math.degrees(e.calc_face_angle(0.0))
        e.smooth = not (angle > s["sharp_deg"] or (seam and angle > s["sharp_seam_deg"]))
        hard += not e.smooth
        seams += seam
    bm.to_mesh(me)
    bm.free()
    me.update()
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers["UVMap"].data.foreach_get("uv", uv)
    tris = np.array([(p.loop_start, p.loop_start + k, p.loop_start + k + 1)
                     for p in me.polygons for k in range(1, p.loop_total - 1)])
    loop_vert = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_vert)
    co = np.array([v.co[:] for v in me.vertices])
    atlas = gr.atlas_density(co, uv.reshape(-1, 2), loop_vert[tris], tris, s["atlas"])
    log(f"new UVs: {islands} islands, {atlas['texels_per_cm']:.2f} texels per cm, "
        f"{atlas['atlas_covered'] * 100:.1f} % of the atlas covered, {atlas['texels_twice'] * 100:.2f} % painted "
        f"twice, {hard} hard edges, {seams} seam edges, "
        f"{'unchecked' if overlapping is None else overlapping} faces overlapping")
    out = {"uv_islands": islands, **atlas, "hard_edges": hard, "uv_overlapping_faces": overlapping}
    if head_uv_share is not None:
        out["head_uv_share"] = head_uv_share
    if auto:
        out["head_texel_auto"] = auto
    return out


def setup_cycles(scene) -> str:
    scene.render.engine = "CYCLES"
    device = "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "METAL"):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == kind
                scene.cycles.device = "GPU"
                device = kind
                break
    except Exception as exc:  # no GPU support in this Blender: the CPU bakes too, only slower
        log("GPU setup failed:", exc)
    if device == "CPU":
        scene.cycles.device = "CPU"
    bake = scene.render.bake
    bake.margin = 0           # padding is done in numpy (game_ready.push_out)
    bake.use_clear = True
    bake.use_selected_to_active = False
    bake.target = "IMAGE_TEXTURES"
    return device


def head_mask_attribute(ob, bones: list[str]) -> str | None:
    """A point colour attribute holding each vertex's skin weight on the bones that keep no glow (0..1): the head's
    with --glow-not-on-head, the hands' with --glow-not-on-hands."""
    if not bones:
        return None
    me = ob.data
    index = {g.index for g in ob.vertex_groups if g.name in set(bones)}
    w = np.zeros(len(me.vertices))
    for v in me.vertices:
        w[v.index] = min(1.0, sum(g.weight for g in v.groups if g.group in index))
    attr = me.color_attributes.new(name="gr_head", type="FLOAT_COLOR", domain="POINT")
    attr.data.foreach_set("color", np.repeat(w, 4).astype(np.float32))
    log(f"no-glow mask: {int((w > 0.5).sum())} of {len(w)} vertices on the head or hands asked for")
    return attr.name


def import_colour_model(path: str, T, feet: tuple | None = None, smooth_flat: bool = False):
    """The finer painted model, moved by the same rescale as the figure (it shares the figure's coordinates), and by
    the same feet (`feet`: the plan, the height it was made at and the second rescale). With `smooth_flat` (the model
    the normal map is baked from) a dense mesh that came in shaded flat is shaded smooth (`game_ready.flat_normal_source`)."""
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=path, disable_bone_shape=True)
    except TypeError:
        bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in set(bpy.data.objects) - before if o.type == "MESH" and o.data.materials]
    for o in set(bpy.data.objects) - before:
        if o not in meshes:
            bpy.data.objects.remove(o, do_unlink=True)
    if not meshes:
        raise SystemExit(f"--colour-from: no painted mesh in {path}")
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
        o.data.transform(T @ mw)
        if feet:
            feet_move(o.data, *feet)
        for mod in list(o.modifiers):
            o.modifiers.remove(mod)
        polys = o.data.polygons
        if smooth_flat and gr.flat_normal_source(sum(1 for p in polys if p.use_smooth), len(polys)):
            polys.foreach_set("use_smooth", [True] * len(polys))
            o.data.update()
            log(f"{o.name}: no normals in the file, shaded smooth so its facets stay out of the normal map")
    log(f"colour from {Path(path).name}: {sum(len(o.data.polygons) for o in meshes)} faces")
    return meshes


def emit_base_colour(objs, white: bool = False) -> None:
    """Make the objects' materials glow their base colour (or plain white, for the hit mask)."""
    for o in objs:
        for mat in o.data.materials:
            if mat is None:
                continue
            mat.use_nodes = True
            nt = mat.node_tree
            out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or nt.nodes.new("ShaderNodeOutputMaterial")
            emit = next((n for n in nt.nodes if n.name == "gr_emit"), None)
            if emit is None:
                emit = nt.nodes.new("ShaderNodeEmission")
                emit.name = "gr_emit"
            for link in list(emit.inputs["Color"].links):
                nt.links.remove(link)
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            sock = bsdf.inputs["Base Color"] if bsdf else None
            if white:
                emit.inputs["Color"].default_value = (1, 1, 1, 1)
            elif sock is not None and sock.is_linked:
                nt.links.new(sock.links[0].from_socket, emit.inputs["Color"])
            elif sock is not None:
                emit.inputs["Color"].default_value = tuple(sock.default_value)
            nt.links.new(emit.outputs[0], out.inputs["Surface"])


def bake_atlas(ob, s: dict, work: Path, name: str, head: list[str] | None = None,
               colour_objs: list | None = None, normal_objs: list | None = None) -> tuple[dict, bool]:
    scene = bpy.context.scene
    me = ob.data
    device = setup_cycles(scene)
    size = s["atlas"] * s["bake_scale"]
    mats = []
    for mat in me.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        nt = mat.node_tree
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        base = normal = None
        colour = (0.8, 0.8, 0.8, 1.0)
        if bsdf:
            sock = bsdf.inputs["Base Color"]
            base = sock.links[0].from_socket if sock.is_linked else None
            colour = tuple(sock.default_value)
            nsock = bsdf.inputs["Normal"]
            normal = nsock.links[0].from_socket if nsock.is_linked else None
            if bsdf.inputs["Emission Strength"].default_value > 0 and (bsdf.inputs["Emission Color"].is_linked or
                                                                       max(bsdf.inputs["Emission Color"].default_value[:3]) > 0):
                log("note: the source material glows; its own emission is not carried over, use --glow")
        emit = nt.nodes.new("ShaderNodeEmission")
        diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        if normal is not None:
            nt.links.new(normal, diffuse.inputs["Normal"])
        target = nt.nodes.new("ShaderNodeTexImage")
        for n in nt.nodes:
            n.select = False
        target.select = True
        nt.nodes.active = target
        mats.append({"nt": nt, "out": out, "emit": emit, "diffuse": diffuse, "base": base, "colour": colour,
                     "target": target})

    def bake(kind: str, samples: int, mode: str, **kw):
        img = bpy.data.images.new("bake_" + kind, size, size, alpha=True, float_buffer=True)
        for m in mats:
            m["target"].image = img
        scene.cycles.samples = samples
        t = time.time()
        bpy.ops.object.bake(type=mode, **kw)
        a = np.empty(size * size * 4, np.float32)
        img.pixels.foreach_get(a)
        bpy.data.images.remove(img)
        log(f"baked {kind} in {time.time() - t:.1f} s")
        return a.reshape(size, size, 4)

    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    scene.cycles.use_denoising = False
    for m in mats:  # coverage: which texels belong to an island
        for link in list(m["emit"].inputs["Color"].links):
            m["nt"].links.remove(link)
        m["emit"].inputs["Color"].default_value = (1, 1, 1, 1)
        m["nt"].links.new(m["emit"].outputs[0], m["out"].inputs["Surface"])
    mask = bake("mask", 1, "EMIT")
    for m in mats:  # the colour exactly as painted
        if m["base"] is not None:
            m["nt"].links.new(m["base"], m["emit"].inputs["Color"])
        else:
            m["emit"].inputs["Color"].default_value = m["colour"]
    albedo = bake("albedo", 4, "EMIT")
    reach = {"use_selected_to_active": True, "cage_extrusion": s["transfer_cage_m"],
             "max_ray_distance": s["transfer_reach_m"]}

    def select_finer(objs):
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob

    def drop_finer(objs):  # gone before the sky bake, so it casts no shadow on the figure
        for o in objs:
            me_c = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if me_c.users == 0:
                bpy.data.meshes.remove(me_c)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob

    def bake_normals(objs, hit=None):
        """The finer model's shape in this surface's tangent space, and where its rays hit."""
        select_finer(objs)
        scene.render.bake.normal_space = "TANGENT"
        nrm = bake("normal map", 4, "NORMAL", **reach)
        if hit is None:
            emit_base_colour(objs, white=True)
            hit = bake("normal hits", 1, "EMIT", **reach)[..., 0]
        return nrm, hit

    transfer_share = None
    normal_bake = None
    if colour_objs:  # the finer model's paint wherever a short ray from this surface meets it
        select_finer(colour_objs)
        emit_base_colour(colour_objs)
        transfer = bake("colour transfer", 4, "EMIT", **reach)
        emit_base_colour(colour_objs, white=True)
        hit = bake("transfer hits", 1, "EMIT", **reach)[..., 0]
        inside_now = mask[..., 0] > 0.5
        transfer_share = float(np.clip(hit[inside_now], 0, 1).mean()) if inside_now.any() else 0.0
        albedo = albedo.copy()
        albedo[..., :3] = gr.blend_transfer(albedo[..., :3], transfer[..., :3], hit)
        if normal_objs is colour_objs:
            normal_bake = bake_normals(colour_objs, hit)
        drop_finer(colour_objs)
        log(f"colour transfer: {transfer_share * 100:.1f} % of the atlas from the finer model")
    if normal_objs and normal_bake is None:
        normal_bake = bake_normals(normal_objs)
        drop_finer(normal_objs)
    head_attr = head_mask_attribute(ob, head or [])
    head_bake = None
    if head_attr:  # where the head is, from its skin weights, so the glow can stay off a face
        for m in mats:
            node = m["nt"].nodes.new("ShaderNodeAttribute")
            node.attribute_name = head_attr
            m["nt"].links.new(node.outputs["Color"], m["emit"].inputs["Color"])
        head_bake = bake("head mask", 1, "EMIT")
        me.color_attributes.remove(me.color_attributes[head_attr])
    world = bpy.data.worlds.new("GameReadySky")  # soft light from above, brighter at the zenith
    scene.world = world
    world.use_nodes = True
    wn = world.node_tree
    bg = next(n for n in wn.nodes if n.type == "BACKGROUND")
    coord, split, ramp = wn.nodes.new("ShaderNodeTexCoord"), wn.nodes.new("ShaderNodeSeparateXYZ"), wn.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = -0.35, 1.0
    ramp.inputs["To Min"].default_value, ramp.inputs["To Max"].default_value = 0.30, 1.0
    wn.links.new(coord.outputs["Generated"], split.inputs[0])
    wn.links.new(split.outputs["Z"], ramp.inputs["Value"])
    wn.links.new(ramp.outputs[0], bg.inputs["Strength"])
    bg.inputs[0].default_value = (1, 1, 1, 1)
    # roofs (TODO 103 of the AssetFurnace Support notes): a piece lying over the surface and facing it, its back to
    # the sky, is not drawn from outside, so the light reaches the surface under it; in the bake the shadow and
    # bounce rays go through it, while its own texels are baked as usual
    roofs = sky_roofs(ob, s)
    roof_attr = None
    if roofs.any():
        roof_attr = me.attributes.new("gr_sky_roof", "FLOAT", "FACE")
        roof_attr.data.foreach_set("value", roofs.astype(np.float32))
        roof_attr = roof_attr.name
    for m in mats:
        surface = m["diffuse"].outputs[0]
        if roof_attr:
            nt = m["nt"]
            attr, path, mix = (nt.nodes.new("ShaderNodeAttribute"), nt.nodes.new("ShaderNodeLightPath"),
                               nt.nodes.new("ShaderNodeMix"))
            attr.attribute_name = roof_attr
            either, both = nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath")
            either.operation, both.operation = "MAXIMUM", "MULTIPLY"
            nt.links.new(path.outputs["Is Shadow Ray"], either.inputs[0])
            nt.links.new(path.outputs["Is Diffuse Ray"], either.inputs[1])
            nt.links.new(either.outputs[0], both.inputs[0])
            nt.links.new(attr.outputs["Fac"], both.inputs[1])
            through = nt.nodes.new("ShaderNodeBsdfTransparent")
            shader_mix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(both.outputs[0], shader_mix.inputs["Fac"])
            nt.links.new(m["diffuse"].outputs[0], shader_mix.inputs[1])
            nt.links.new(through.outputs[0], shader_mix.inputs[2])
            nt.nodes.remove(mix)
            surface = shader_mix.outputs[0]
        m["nt"].links.new(surface, m["out"].inputs["Surface"])
    scene.cycles.use_denoising = True
    sky = bake("sky light", 160, "DIFFUSE", pass_filter={"DIRECT", "INDIRECT"})
    if roof_attr:
        me.attributes.remove(me.attributes[roof_attr])
    log(f"sky light: {int(roofs.sum())} roof faces seen through")

    k = s["bake_scale"]
    inside = (mask[..., 0:1] > 0.5).astype(np.float32)
    cov = gr.downsample(inside, k)[..., 0]
    alb = gr.downsample(albedo[..., :3] * inside, k) / np.maximum(cov, 1e-6)[..., None]
    lit = gr.downsample(sky[..., :3] * inside, k).mean(-1) / np.maximum(cov, 1e-6)
    exclude = None
    if head_bake is not None:
        exclude = gr.downsample(head_bake[..., :1] * inside, k)[..., 0] / np.maximum(cov, 1e-6)
    colour, emission, stats = gr.compose(alb, lit, cov, s, exclude=exclude)
    log("colour:", stats)
    p_alb = work / f"{name}_albedo.png"
    gr.write_png(p_alb, colour)
    p_emi = None
    if emission is not None:
        p_emi = work / f"{name}_emission.png"
        gr.write_png(p_emi, emission)
    p_nrm = None
    if normal_bake is not None:
        nrm, nhit = normal_bake
        inside_n = mask[..., 0] > 0.5
        nrm = gr.downsample(nrm[..., :3] * inside_n[..., None], k) / np.maximum(cov, 1e-6)[..., None]
        nhit = gr.downsample((np.clip(nhit, 0, 1) * inside_n)[..., None], k)[..., 0] / np.maximum(cov, 1e-6)
        texels = gr.normal_texels(nrm, nhit)
        texels, have = gr.push_out(texels, cov > 0, s["fill_px"])
        texels[~have] = (0.5, 0.5, 1.0)
        p_nrm = work / f"{name}_normal.png"
        gr.write_png(p_nrm, texels)
        stats["normal_map_hit_share"] = round(float(nhit[cov > 0].mean()) if (cov > 0).any() else 0.0, 4)
        log(f"normal map: {stats['normal_map_hit_share'] * 100:.1f} % of the atlas from the finer model")

    old_images = {n.image for m in mats for n in m["nt"].nodes if n.type == "TEX_IMAGE" and n.image}
    old_mats = [mat for mat in me.materials if mat]
    me.materials.clear()
    for mat in old_mats:
        bpy.data.materials.remove(mat)
    for img in old_images:
        bpy.data.images.remove(img)
    bpy.data.worlds.remove(world)
    me.uv_layers.remove(me.uv_layers["src"])

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    t_alb = nt.nodes.new("ShaderNodeTexImage")
    t_alb.image = bpy.data.images.load(str(p_alb))
    t_alb.image.name = p_alb.stem
    nt.links.new(t_alb.outputs["Color"], bsdf.inputs["Base Color"])
    if p_emi is not None:
        t_emi = nt.nodes.new("ShaderNodeTexImage")
        t_emi.image = bpy.data.images.load(str(p_emi))
        t_emi.image.name = p_emi.stem
        nt.links.new(t_emi.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = s["glow_strength"]
    if p_nrm is not None:
        t_nrm = nt.nodes.new("ShaderNodeTexImage")
        t_nrm.image = bpy.data.images.load(str(p_nrm))
        t_nrm.image.name = p_nrm.stem
        t_nrm.image.colorspace_settings.name = "Non-Color"
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.space = "TANGENT"
        nt.links.new(t_nrm.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = s["roughness"]
    bsdf.inputs["Metallic"].default_value = 0.0
    mat.use_backface_culling = True
    me.materials.append(mat)
    stats["bake_device"] = device
    if transfer_share is not None:
        stats["colour_transfer_share"] = round(transfer_share, 4)
    stats["sky_roofs"] = int(roofs.sum())
    return stats, p_emi is not None


def reweight(ob, arm, plan: dict, s: dict) -> dict:
    me = ob.data
    bones = {b.name for b in arm.data.bones}
    groups = [g.name for g in ob.vertex_groups]
    n = len(me.vertices)
    W = np.zeros((n, len(groups)))
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    deform = [j for j, g in enumerate(groups) if g in bones]
    W, groups = W[:, deform], [groups[j] for j in deform]
    unweighted_source = int((W.sum(1) < 1e-6).sum())
    W, cols = gr.merge_columns(W, groups, plan)
    k = s["max_influences"]
    W = gr.limit_influences(W, k)
    edges = np.array([e.vertices[:] for e in me.edges]).reshape(-1, 2)
    W = gr.smooth_weights(W, edges, s["smooth_iters"], s["smooth_factor"])
    co = np.array([v.co[:] for v in me.vertices])
    W, rigid = gr.rigid_shells(W, gr.shells_from_edges(n, edges), co, s["rigid_share"] * s["height"])
    W = gr.limit_influences(W, k)
    ob.vertex_groups.clear()
    for j, name in enumerate(cols):
        vg = ob.vertex_groups.new(name=name)
        for i in np.nonzero(W[:, j])[0]:
            vg.add([int(i)], float(W[i, j]), "REPLACE")
    count = (W > 0).sum(1)
    stats = {"max_influences": int(count.max()), "influences_histogram": np.bincount(count, minlength=k + 1).tolist(),
             "unweighted": int((count == 0).sum()), "unweighted_source": unweighted_source, "rigid_pieces": rigid}
    log("weights:", stats)
    return stats


def drop_bones(arm, plan: dict) -> None:
    if not plan:
        return
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for name in plan:
        eb = arm.data.edit_bones.get(name)
        if eb:
            arm.data.edit_bones.remove(eb)
    bpy.ops.object.mode_set(mode="OBJECT")
    prefixes = tuple(f'pose.bones["{name}"]' for name in plan)
    removed = 0
    for action in bpy.data.actions:
        for collection, fc in fcurve_slots(action):
            if fc.data_path.startswith(prefixes):
                collection.remove(fc)
                removed += 1
    log(f"bones merged away: {len(plan)}, their curves removed: {removed}")


def main() -> None:
    job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    s, plan = job["settings"], job["merge"]
    work, name = Path(job["work"]), job["name"]
    arm, ob = import_source(job["source"])
    report = {}
    if s.get("mesh_from"):
        ob = swap_in_finer_mesh(arm, ob, s["mesh_from"])
        arm.data.pose_position = "POSE"
    action = arm.animation_data.action if arm.animation_data else None
    frames = []
    if action:
        f0, f1 = (int(round(f)) for f in action.frame_range)
        frames = sorted({f0, (f0 + f1) // 2, f1})
    kept = [b.name for b in arm.data.bones if b.name not in plan]
    before_names = list(kept)
    before = sample_heads(arm, frames, kept)

    S, T = rescale(arm, ob, s["height"], job["hips"])
    arm.data.pose_position = "REST"  # bake the figure as it stands, never in a pose
    for m in ob.modifiers:
        m.show_viewport = m.show_render = False
    report["cleaning"] = clean_mesh(ob, s)
    T_moves, feet = T, None
    if job.get("feet"):
        report["feet"], T2, moved, frame, soles = fix_feet(arm, ob, job["feet"], s, job["hips"],
                                                           job.get("feet_bones") or ())
        report["feet"]["plan"] = soles
        feet = (soles, frame, T2)
        T_moves = T2 @ Matrix.Translation((0.0, 0.0, -gr.feet_drop(soles) * frame)) @ T
        kept = [b for b in kept if b not in moved]
        before = {f: before[f][[i for i, b in enumerate(before_names) if b not in moved]] for f in before}
    head = job.get("all_head_bones") or []
    if s.get("decimate_to") and s.get("drop_hidden"):
        report.update(thin_and_drop(ob, arm, s, head, job.get("veto_poses") or [], job.get("check_poses") or [],
                                    job.get("pose_picture")))
    elif s.get("decimate_to"):
        report.update(decimate_with_head(ob, s, head))
    colour_objs = import_colour_model(s["colour_from"], T, feet) if s.get("colour_from") else None
    report.update(new_uvs(ob, s, head, colour_objs))
    normal_objs = None
    if s.get("normal_from"):
        same = colour_objs and Path(s["normal_from"]).resolve() == Path(s["colour_from"]).resolve()
        normal_objs = colour_objs if same else import_colour_model(s["normal_from"], T, feet, smooth_flat=True)
    report["colour"], glows = bake_atlas(ob, s, work, name, job.get("head_bones"), colour_objs, normal_objs)
    for m in ob.modifiers:
        m.show_viewport = m.show_render = True
    arm.data.pose_position = "POSE"
    report.update(reweight(ob, arm, plan, s))
    drop_bones(arm, plan)

    error = None
    if frames:
        after = sample_heads(arm, frames, kept)
        Tn = np.array(T_moves)
        error = max(float(np.abs(after[f] - (before[f] @ Tn[:3, :3].T + Tn[:3, 3])).max()) for f in frames)
        log(f"moves after the rescale differ by up to {error:.6f} m")
    bpy.context.scene.frame_set(frames[0] if frames else 1)

    me = ob.data
    co = np.array([v.co[:] for v in me.vertices])
    me.calc_loop_triangles()
    loops = len(me.loops)
    lv = np.empty(loops, np.int32)
    me.loops.foreach_get("vertex_index", lv)
    normals = np.array([c.vector[:] for c in me.corner_normals])
    uv = np.empty(loops * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    fps = bpy.context.scene.render.fps
    report.update({
        "triangles": len(me.loop_triangles), "vertices": len(me.vertices),
        "engine_vertices": gr.engine_vertices(lv, normals, uv.reshape(-1, 2)),
        "materials": len(me.materials), "glow": glows, "atlas": s["atlas"],
        "bones": len(arm.data.bones), "scale": round(S, 5),
        "height": round(float(co[:, 2].max() - co[:, 2].min()), 4), "min_z": round(float(co[:, 2].min()), 5),
        "bounds": [np.round(co.min(0), 3).tolist(), np.round(co.max(0), 3).tolist()],
        "clips": [{"name": a.name, "seconds": round((a.frame_range[1] - a.frame_range[0]) / fps, 2)}
                  for a in bpy.data.actions],
        "animation_error_m": None if error is None else round(error, 6),
    })

    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = arm
    options = {"filepath": job["output"], "export_format": "GLB", "use_selection": True, "export_animations": True,
               "export_skins": True, "export_yup": True}
    try:
        bpy.ops.export_scene.gltf(**options, export_all_influences=False)
    except TypeError:
        bpy.ops.export_scene.gltf(**options)
    report["seconds"] = round(time.time() - T0, 1)
    Path(job["report"]).write_text(json.dumps(report, indent=1))
    log(f"done in {report['seconds']:.1f} s")


if __name__ == "__main__":
    main()
