"""The Blender half of `prop_ready.py`: turn, scale and ground a prop, export LOD0, and the hull.

Run (headless): blender --background --factory-startup --python-exit-code 1 --python scripts/blender_prop_ready.py -- JOB.json

The job, written by `prop_ready.py`: `source`, `output`, `yaw` (degrees about the up axis that
bring the front to +Z), `size` (one of width, height or length in metres, or empty), `node`
(LOD0's name), `hull` (output, faces, node) or null, and `parts`. Writes `place.result.json`
beside the job: the scale, the placed size (width, height, length), its lowest and highest
point, the hull's triangles and, with `parts`, each part's name and pivot (and how it moves, when
the model says so in glTF extras: `motion`, `axis`, `out`, `slide`). The maths is
`prop_ready.py`'s, imported here.

With `parts` every part stays its own object with its origin (its pivot: a wheel's axle, a
door's hinge) moved along with it, under a root named `node`; without, everything is baked
into world space as before. A job with `"mode": "decimate"` (`source`, `output`, `targets`:
triangles per part) makes a LOD of a parts model by decimating each part, keeping its UVs.
"""

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prop_ready as pr

HULL_ROUNDS = 6


def meshes():
    return [obj for obj in bpy.data.objects if obj.type == "MESH"]


def bake_transforms(objects):
    """Every mesh in world space with an identity transform, parents and empties gone."""
    for obj in objects:
        world = obj.matrix_world.copy()
        obj.data = obj.data.copy()
        obj.parent = None
        obj.matrix_world = Matrix.Identity(4)
        obj.data.transform(world)
    for obj in [o for o in bpy.data.objects if o.type != "MESH"]:
        bpy.data.objects.remove(obj, do_unlink=True)


def transform(objects, matrix):
    for obj in objects:
        obj.data.transform(matrix)
        obj.data.update()


def keep_parts(objects):
    """Every mesh unparented with its world placement kept, empties gone: the parts' origins stay."""
    for obj in objects:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
    for obj in [o for o in bpy.data.objects if o.type != "MESH"]:
        bpy.data.objects.remove(obj, do_unlink=True)


def move_parts(objects, matrix):
    """Turn, scale or move every part as a whole, its origin with it; the scale goes into the
    mesh so a part's transform stays a plain position and turn, as an engine expects."""
    for obj in objects:
        location, rotation, scale = (matrix @ obj.matrix_world).decompose()
        obj.data = obj.data.copy()
        obj.data.transform(Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0)))
        obj.data.update()
        obj.matrix_world = Matrix.LocRotScale(location, rotation, None)


def world_points(objects):
    for obj in objects:
        world = obj.matrix_world
        for vertex in obj.data.vertices:
            yield world @ vertex.co


def triangles(obj) -> int:
    return sum(len(polygon.vertices) - 2 for polygon in obj.data.polygons)


def motion(obj) -> dict:
    """How a part says it moves (glTF extras, which Blender's importer keeps as custom
    properties): `motion` and its `axis`, `out` and `slide`, still in the model's own axes;
    `prop_ready.moving_parts` turns them with the prop."""
    found = {}
    if isinstance(obj.get("motion"), str):
        found["motion"] = obj["motion"]
        for key in ("axis", "out", "slide"):
            if key in obj:
                found[key] = [float(value) for value in obj[key]]
    return found


def gltf_point(vector):
    """A Blender point in glTF axes: Blender (x, y, z) is glTF (x, z, -y)."""
    return [vector.x, vector.z, -vector.y]


def bounds(objects):
    """Lowest and highest corner, in glTF axes (x, y up, z towards the front)."""
    low = Vector((math.inf,) * 3)
    high = Vector((-math.inf,) * 3)
    for point in world_points(objects):
        for axis in range(3):
            low[axis] = min(low[axis], point[axis])
            high[axis] = max(high[axis], point[axis])
    # Blender is Z up with the front at -Y; glTF is Y up with the front at +Z
    return (low.x, low.z, -high.y), (high.x, high.z, -low.y)


def hull_object(objects, faces: int, name: str):
    """A convex hull of every mesh, cut to at most `faces` triangles."""
    bm = bmesh.new()
    for point in world_points(objects):
        bm.verts.new(point)
    result = bmesh.ops.convex_hull(bm, input=bm.verts[:])
    bmesh.ops.delete(bm, geom=[g for g in result["geom_interior"] + result["geom_unused"]
                               if isinstance(g, bmesh.types.BMVert)], context="VERTS")
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for _ in range(HULL_ROUNDS):
        count = len(obj.data.polygons)
        if count <= faces:
            break
        modifier = obj.modifiers.new("cut", "DECIMATE")
        modifier.ratio = max(0.01, faces / count * 0.97)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        # a cut hull can dent inwards; take the hull of what is left, which only shrinks it
        result = bmesh.ops.convex_hull(bm, input=bm.verts[:])
        bmesh.ops.delete(bm, geom=[g for g in result["geom_interior"] + result["geom_unused"]
                                   if isinstance(g, bmesh.types.BMVert)], context="VERTS")
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(obj.data)
        bm.free()
    return obj


def export(path: str, objects) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True,
                              export_yup=True, export_apply=False)


def decimate(job) -> None:
    """A LOD of a parts model: each part decimated to its share, UVs and pivots kept."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=job["source"])
    for obj in meshes():
        target = job["targets"].get(obj.name)
        count = triangles(obj)
        if target is None or count == 0 or target >= count:
            continue
        modifier = obj.modifiers.new("lod", "DECIMATE")
        modifier.ratio = target / count
        modifier.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    export(job["output"], list(bpy.data.objects))
    print("PROP_READY::decimated " + json.dumps({obj.name: triangles(obj) for obj in meshes()}), flush=True)


def main() -> None:
    job_path = Path(sys.argv[sys.argv.index("--") + 1])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("mode") == "decimate":
        decimate(job)
        return
    parts = bool(job.get("parts"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=job["source"])
    if any(obj.type == "ARMATURE" for obj in bpy.data.objects):
        raise SystemExit("prop_ready is for props; a rigged model goes through game_ready.py")
    objects = meshes()
    if not objects:
        raise SystemExit("no mesh in " + job["source"])
    if parts:
        keep_parts(objects)
        place = move_parts
    else:
        bake_transforms(objects)
        place = transform

    # the same angle turns glTF about +Y and Blender about +Z (see prop_ready.FRONTS)
    place(objects, Matrix.Rotation(math.radians(job["yaw"]), 4, "Z"))
    low, high = bounds(objects)
    extent = [h - lo for lo, h in zip(low, high)]
    scale = pr.scale_for(extent, **job["size"])
    place(objects, Matrix.Scale(scale, 4))
    low, high = bounds(objects)
    x, y, z = pr.floor_translation(low, high)
    place(objects, Matrix.Translation((x, -z, y)))   # glTF (x, y, z) is Blender (x, -z, y)
    low, high = bounds(objects)

    result = {"scale": scale, "low": list(low), "high": list(high),
              "size": [h - lo for lo, h in zip(low, high)], "hull_triangles": None}
    if parts:
        result["parts"] = [{"name": obj.name, "pivot": gltf_point(obj.matrix_world.translation), **motion(obj)}
                           for obj in objects]
        root = bpy.data.objects.new(job["node"], None)
        bpy.context.scene.collection.objects.link(root)
        for obj in objects:
            world = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world
        export(job["output"], [root, *objects])
    else:
        for index, obj in enumerate(objects):
            obj.name = job["node"] if index == 0 else f"{job['node']}_{index}"
        export(job["output"], objects)
    if job.get("hull"):
        hull = hull_object(objects, job["hull"]["faces"], job["hull"]["node"])
        result["hull_triangles"] = len(hull.data.polygons)
        export(job["hull"]["output"], [hull])
    (job_path.parent / "place.result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PROP_READY::" + json.dumps(result), flush=True)


main()
