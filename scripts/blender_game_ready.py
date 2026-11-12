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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game_ready as gr

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


def new_uvs(ob, s: dict) -> dict:
    me = ob.data
    pin_source_uvs(me)
    layer = me.uv_layers.new(name="UVMap")
    me.uv_layers.active = layer
    layer.active_render = True
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(s["uv_angle"]), island_margin=0.0, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.select_all(action="SELECT")
    try:
        bpy.ops.uv.pack_islands(rotate=True, rotate_method="ANY", scale=True, merge_overlap=False,
                                margin_method="ADD", margin=gr.uv_margin(s), shape_method="CONCAVE")
    except (TypeError, RuntimeError) as exc:
        log("concave packing unavailable, plain packing:", exc)
        bpy.ops.uv.pack_islands(rotate=True, margin=gr.uv_margin(s))
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
    used = gr.uv_area(uv.reshape(-1, 2), tris)
    log(f"new UVs: {islands} islands, {used * 100:.1f} % of the atlas, {hard} hard edges, {seams} seam edges")
    return {"uv_islands": islands, "atlas_used": round(used, 4), "hard_edges": hard}


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
    """A point colour attribute holding each vertex's skin weight on the head bones (0..1)."""
    if not bones:
        return None
    me = ob.data
    index = {g.index for g in ob.vertex_groups if g.name in set(bones)}
    w = np.zeros(len(me.vertices))
    for v in me.vertices:
        w[v.index] = min(1.0, sum(g.weight for g in v.groups if g.group in index))
    attr = me.color_attributes.new(name="gr_head", type="FLOAT_COLOR", domain="POINT")
    attr.data.foreach_set("color", np.repeat(w, 4).astype(np.float32))
    log(f"head mask: {int((w > 0.5).sum())} of {len(w)} vertices on the head")
    return attr.name


def bake_atlas(ob, s: dict, work: Path, name: str, head: list[str] | None = None) -> tuple[dict, bool]:
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
    for m in mats:
        m["nt"].links.new(m["diffuse"].outputs[0], m["out"].inputs["Surface"])
    scene.cycles.use_denoising = True
    sky = bake("sky light", 160, "DIFFUSE", pass_filter={"DIRECT", "INDIRECT"})

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
    bsdf.inputs["Roughness"].default_value = s["roughness"]
    bsdf.inputs["Metallic"].default_value = 0.0
    mat.use_backface_culling = True
    me.materials.append(mat)
    stats["bake_device"] = device
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
    action = arm.animation_data.action if arm.animation_data else None
    frames = []
    if action:
        f0, f1 = (int(round(f)) for f in action.frame_range)
        frames = sorted({f0, (f0 + f1) // 2, f1})
    kept = [b.name for b in arm.data.bones if b.name not in plan]
    before = sample_heads(arm, frames, kept)

    S, T = rescale(arm, ob, s["height"], job["hips"])
    arm.data.pose_position = "REST"  # bake the figure as it stands, never in a pose
    for m in ob.modifiers:
        m.show_viewport = m.show_render = False
    report["cleaning"] = clean_mesh(ob, s)
    report.update(new_uvs(ob, s))
    report["colour"], glows = bake_atlas(ob, s, work, name, job.get("head_bones"))
    for m in ob.modifiers:
        m.show_viewport = m.show_render = True
    arm.data.pose_position = "POSE"
    report.update(reweight(ob, arm, plan, s))
    drop_bones(arm, plan)

    error = None
    if frames:
        after = sample_heads(arm, frames, kept)
        Tn = np.array(T)
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
