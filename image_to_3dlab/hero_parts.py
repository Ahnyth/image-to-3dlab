"""Hero parts: a small part of a figure built on its own at full resolution and set into it, as pure functions.

Pixal3D spends its detail on its whole box, so a part that is a few dozen pixels of a full-body picture comes out
coarse: the Revolverheld's holstered pistol as a flat slab with the pistol's outline standing off his thigh, the
Zauberer's boots as smooth tubes with their straps and buckles only painted on (2026-11-22). Cropped, redrawn by the
picture editor as the part alone (complete, from the side that shows its shape) and built on its own, the part fills
Pixal3D's whole box and keeps its shape. This module holds the maths that sets such a part into the figure;
`scripts/hero_parts.py` runs the steps. It generalises the Hero hands (`hero_hands.py`), whose cut, weights and GLB
writing it reuses, to two kinds of seam:

- a part that sits on the body (a holstered pistol, a pouch, a book at the belt): placed by its box in the front
  picture, its back settled on the body behind the slab Pixal3D made of it (`body_behind`, `settle`). Inside its
  outline the figure loses whatever lies in front of the part's back (the slab, a strap crossing it); what lies further
  back (the thigh) stays; every outline the cut leaves is closed, and the part moves rigidly with the bones the slab
  followed (`graft_on_body`, `rigid_weights`).
- a part that ends a limb (a boot): it stands on the figure's floor with its sole flat, as long as the old foot and
  turned as it was (`place_boot`, `level_sole`); the leg is cut straight across just under the new boot's top, the old
  boot below goes and the cut is closed inside the new one (`graft_boot`); the boot takes the old leg's weights point by
  point, so it bends at the ankle (`boot_weights`).

Either is painted by the figure's own picture where that sees it, through the camera the figure was built with
(`staging`, `crop_view`), and by its own build elsewhere, moved to the picture's tones (`photo_paint.paint_texture`).
A hard part (a pistol, a buckle) can have its nearly flat pieces made flat and their edges sharp (`harden`); never a
face or cloth.

Models face +Z (glTF's front) with Y up, as Pixal3D builds them.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import ndimage

from image_to_3dlab import hero_face as hf
from image_to_3dlab import hero_hands as hh
from image_to_3dlab import hero_head as hd
from image_to_3dlab import photo_paint as pp

FRAME = pp.PIXAL3D_FRAME     # a Pixal3D model's axes against its run's cameras
UP = np.array([0.0, 1.0, 0.0])
FAR_SHARE = 1000.0           # a figure without its run's cameras is seen from this many of its heights away (straight on)
TOUCH_SHARE = 0.1            # a settled part's back touches the body over this share of its pixels ...
TUCK_SHARE = 0.01            # ... sinking this share of the part's height into it there, so no air shows under its edge
BEHIND_SHARE = 0.015         # the figure loses what lies in front of the part's back less this share of its height
GROW_SHARE = 0.015           # the part's outline grows this share of its height before it cuts the figure
REMNANT_SHARE = 0.2          # a piece a cut left within this share of the part's height round it is a remnant of the slab
THIN_SHARE = 0.25            # a layer thinner than this share of the part's height is a slab, not the body: never the
                             # part's own depth, which a build from one drawing can get wrong (the Revolverheld's pistol
                             # leaned 20 degrees and came out half as deep as tall, so his thigh counted as a slab too)
TILT_LIMITS_DEG = (30.0, 20.0)   # a part is turned at most this far to lie along the body: its lean, its turn round it
FIELD_PIXELS = 320           # the part's footprint is read this many pixels over its height
CRUMB_SHARE = 0.002          # a build's separate piece with fewer faces than this share of it is a crumb
BOOT_FOOT_SHARE = 0.3        # a boot's foot is its lowest this share of its height ...
HEADING_LIMIT_DEG = 35.0     # ... turned at most this far from straight ahead to follow the old foot
LEVEL_LIMIT_DEG = 10.0       # a built sole is laid flat when it leans at most this much
SOLE_BAND_SHARE = 0.04       # a sole's contact points: the lowest this share of the boot's height, cell by cell
LEG_REACH = 1.8              # the old boot is cut within this many shin radii of the leg's axis ...
FOOT_REACH = 1.15            # ... or within this share past the old foot's own farthest point
BOOT_OVERLAP_SHARE = 0.08    # the leg is cut this share of the new boot's height under its top, inside it
PLANE_LINK_DEG = 4.0         # neighbouring faces whose smoothed normals differ by less than this lie on one surface ...
PLANE_CREASE_DEG = 20.0      # ... unless their own normals differ by this much: an edge, never smoothed across
PLANE_MODE_DEG = 6.0         # a curved surface's flattest piece: the faces this close to its commonest normal
PLANE_SPREAD_DEG = 3.5       # a plane's faces lie this close to its normal (root mean square): a strip of a tube does not
PLANE_FACET_DEG = 20.0       # two planes meeting at less than this are facets of a curved surface, not planes
PLANE_GROW_DEG = 12.0        # a plane takes in the faces along its edge this close to it ...
PLANE_GROW_ROUNDS = 8        # ... ring by ring, at most this many rings
PLANE_AREA_SHARE = 0.004     # a plane covers at least this share of the part's surface
PLANE_FLAT_SHARE = 0.006     # and its points lie this close to it (95 per cent of them, a share of the part's size)
SMOOTH_ROUNDS = 3            # face normals are smoothed over this many rings of neighbours first (a build's facets)


# --- The figure's front camera ----------------------------------------------------------------------------------------

def subject_box(alpha: np.ndarray, threshold: int = 128) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of a picture's subject: the box round its alpha above THRESHOLD, x1 and y1 one past its last
    pixel (as Pillow's getbbox, which the Pixal3D run used to frame it)."""
    ys, xs = np.nonzero(np.asarray(alpha) > threshold)
    if not len(xs):
        raise ValueError("the picture shows no subject (its cut-out is empty)")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def staging(picture_box, view_box) -> tuple[float, np.ndarray]:
    """(scale, offset) taking a picture's pixels to its run's view: view = scale * picture + offset.

    A Pixal3D run frames the picture's subject on a square of its own (`pixal3d_generate.pad_view`: cropped to the
    subject, scaled, centred), so the two subjects' boxes give the map. A part is marked in the picture a person sees;
    it is placed and painted through the view, whose camera is the one the figure was built with: the straight-on box
    frame of the Hero route lay 14 pixels off at the Revolverheld's holster and 31 at his head."""
    pw, ph = picture_box[2] - picture_box[0], picture_box[3] - picture_box[1]
    vw, vh = view_box[2] - view_box[0], view_box[3] - view_box[1]
    scale = (pw * vw + ph * vh) / float(pw * pw + ph * ph)
    middle = np.array([(picture_box[0] + picture_box[2]) / 2, (picture_box[1] + picture_box[3]) / 2])
    target = np.array([(view_box[0] + view_box[2]) / 2, (view_box[1] + view_box[3]) / 2])
    return float(scale), target - scale * middle


def staged_box(box, stage) -> tuple[float, float, float, float]:
    """A picture's box (x0, y0, x1, y1) in its run's view (`staging`)."""
    scale, offset = stage
    return (box[0] * scale + offset[0], box[1] * scale + offset[1], box[2] * scale + offset[0],
            box[3] * scale + offset[1])


def from_view_space(points: np.ndarray, frame=FRAME, mesh_scale: float = 1.0) -> np.ndarray:
    """The inverse of `photo_paint.to_view_space`: points of a view's space in the model's own frame."""
    perm, signs = frame
    p = np.asarray(points, dtype=np.float64) * mesh_scale
    out = np.zeros_like(p)
    for i in range(3):
        out[:, perm[i]] = signs[i] * p[:, i]
    return out


def view_pixels(view: pp.View, points: np.ndarray, mesh_scale: float = 1.0):
    """(x, y, depth) of a model's points in VIEW's pixels (pixel centres at +0.5)."""
    return pp.project(pp.to_view_space(np.asarray(points, dtype=np.float64), FRAME, mesh_scale), view)


def unproject(view: pp.View, x, y, z: float, mesh_scale: float = 1.0) -> np.ndarray:
    """The model's points on the plane Z = z that VIEW sees at pixels (x, y): a part's box in the picture carried to
    the depth the part stands at."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    height, width = view.image.shape[:2]
    cx, cy = view.centre if view.centre is not None else (width / 2.0, height / 2.0)
    f = view.focal_px
    along = np.stack([(x - cx) / f, -(y - cy) / f, -np.ones_like(x)], axis=1) @ view.cam_to_world[:3, :3].T
    origin = view.cam_to_world[:3, 3]
    perm, signs = FRAME
    axis = list(perm).index(2)                      # the view's axis that holds the model's Z
    t = (signs[axis] * z / mesh_scale - origin[axis]) / along[:, axis]
    return from_view_space(origin + t[:, None] * along, FRAME, mesh_scale)


def far_view(points: np.ndarray, picture, distance: float = FAR_SHARE) -> pp.View:
    """A camera for a figure without its run's cameras: the picture's subject framing the model's height as the Hero
    route's straight-on frame does (`hero.frame`), seen from DISTANCE of its heights away. Its pixels are the picture's
    (PICTURE, a Pillow image, cut out or on the studio's grey)."""
    from image_to_3dlab import hero

    p = np.asarray(points, dtype=np.float64)
    centre, per_px, box_centre = hero.frame(p, hero.picture_box(picture))
    eye = centre + np.array([0.0, 0.0, distance * float(np.ptp(p[:, 1]))])

    def axis(v):
        return pp.to_view_space(np.asarray([v], dtype=np.float64), FRAME)[0]

    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = axis([1, 0, 0]), axis([0, 1, 0]), axis([0, 0, 1]), axis(eye)
    focal = (eye[2] - centre[2]) / per_px
    image = np.asarray(picture.convert("RGBA"))
    return pp.View(image, c2w, float(2 * np.arctan(image.shape[1] / 2 / focal)), "front (box frame)", focal=focal,
                   centre=(float(box_centre[0]) + 0.5, float(box_centre[1]) + 0.5))


def crop_view(view: pp.View, box, image: np.ndarray) -> pp.View:
    """VIEW's camera for a crop of its picture: BOX (x0, y0, x1, y1, square, in the view's pixels) cut out and scaled
    to IMAGE (the upscaled crop), so the crop's pixels paint where the whole picture's would, only sharper."""
    height, width = view.image.shape[:2]
    cx, cy = view.centre if view.centre is not None else (width / 2.0, height / 2.0)
    k = np.asarray(image).shape[1] / float(box[2] - box[0])
    return pp.View(np.asarray(image), view.cam_to_world, view.fov_x, view.name + " (crop)", focal=k * view.focal_px,
                   centre=(k * (cx - box[0]), k * (cy - box[1])))


# --- A part on the body -------------------------------------------------------------------------------------------------

def depth_maps(positions: np.ndarray, faces: np.ndarray, box, pixel: float) -> tuple[np.ndarray, np.ndarray]:
    """(front, back): a mesh seen straight from the front over BOX (x0, x1, y0, y1), every pixel's nearest and farthest
    surface (Z); -inf and +inf where nothing is."""
    rows, cols = hf.canvas(box, pixel)
    pixel_id, z, _ = hf.pixel_hits(positions, faces, box, pixel)
    front, back = np.full(rows * cols, -np.inf), np.full(rows * cols, np.inf)
    np.maximum.at(front, pixel_id, z)
    np.minimum.at(back, pixel_id, z)
    return front.reshape(rows, cols), back.reshape(rows, cols)


def _body_front(z: np.ndarray, nz: np.ndarray, thin: float) -> float:
    """Along one line of sight (Z sorted near to far, NZ the crossed faces' normals along Z): the front of the first
    solid at least THIN deep, else the nearest surface. A line through an edge crosses both its faces at one depth;
    that counts once."""
    depth, front, last = 0, None, None
    for zi, ni in zip(z, nz):
        if last is not None and np.sign(ni) == np.sign(last[1]) and abs(zi - last[0]) <= 1e-6 * max(thin, 1e-9):
            continue
        last = (zi, ni)
        if ni > 0:                       # facing the camera: a solid starts
            if depth == 0:
                front = zi
            depth += 1
        elif depth > 0:                  # facing away: it ends
            depth -= 1
            if depth == 0 and front - zi >= thin:
                return float(front)
    return float(z[0])


def body_behind(positions: np.ndarray, faces: np.ndarray, box, pixel: float, thin: float) -> np.ndarray:
    """The body's front (Z) in every pixel of a front view over BOX, behind any layer thinner than THIN.

    Pixal3D builds a part it saw as a few pixels as a thin slab standing off the body (the Revolverheld's holster, a
    centimetre in front of his thigh), and a new part's back belongs on the thigh, not on the slab. Along each pixel's
    line of sight a surface facing the camera starts a solid and one facing away ends it; the first solid at least
    THIN deep is the body. Where none is, the nearest surface stands in; -inf where nothing is."""
    rows, cols = hf.canvas(box, pixel)
    out = np.full(rows * cols, -np.inf)
    pixel_id, z, face = hf.pixel_hits(positions, faces, box, pixel)
    if not len(pixel_id):
        return out.reshape(rows, cols)
    nz = hf.face_normals(positions, faces)[face, 2]
    order = np.lexsort((-z, pixel_id))
    pixel_id, z, nz = pixel_id[order], z[order], nz[order]
    starts = np.flatnonzero(np.r_[True, pixel_id[1:] != pixel_id[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(pixel_id)]):
        out[pixel_id[start]] = _body_front(z[start:end], nz[start:end], thin)
    return out.reshape(rows, cols)


def place_in_box(points: np.ndarray, target, front: float) -> np.ndarray:
    """The 4x4 transform putting a built part (POINTS, turned as it sits on the figure) into TARGET (x0, x1, y0, y1 in
    the figure's frame: its box in the picture carried to its depth): scaled so its height fills the box, centred on
    it, its nearest point at FRONT."""
    p = np.asarray(points, dtype=np.float64)
    lo, hi = p.min(axis=0), p.max(axis=0)
    k = (target[3] - target[2]) / max(hi[1] - lo[1], 1e-12)
    tf = np.diag([k, k, k, 1.0])
    tf[:3, 3] = [(target[0] + target[1]) / 2 - k * (lo[0] + hi[0]) / 2,
                 (target[2] + target[3]) / 2 - k * (lo[1] + hi[1]) / 2, front - k * hi[2]]
    return tf


def fit_tilt(back: np.ndarray, body: np.ndarray, box, pixel: float, limits=TILT_LIMITS_DEG,
             keep: float = 0.75) -> np.ndarray:
    """The turn (3x3, about the part's own middle) that lays a part's back along the body behind it.

    BACK and BODY are front views over BOX (the part's farthest surface; the body's front, `body_behind`). Their gap is
    fitted by a plane over the pixels both have (the closest KEEP share of them, so a strap or the coat's edge does not
    pull it), and its slope is taken out by turning the part about the x axis (its lean toward or away from the body:
    a build from one drawing often leans, the Revolverheld's pistol by 20 degrees) and the y axis (the body's curve
    round it), each at most LIMITS degrees."""
    both = np.isfinite(back) & np.isfinite(body)
    if both.sum() < 10:
        return np.eye(3)
    r, c = np.nonzero(both)
    x = box[0] + (c + 0.5) * pixel
    y = box[3] - (r + 0.5) * pixel
    gap = (back - body)[both]
    design = np.c_[np.ones(len(x)), x, y]
    use = np.ones(len(x), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(design[use], gap[use], rcond=None)[0]
        miss = np.abs(design @ coef - gap)
        use = miss <= np.quantile(miss, keep)
    pitch = np.radians(np.clip(-np.degrees(np.arctan(coef[2])), -limits[0], limits[0]))
    yaw = np.radians(np.clip(np.degrees(np.arctan(coef[1])), -limits[1], limits[1]))
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)], [0.0, np.sin(pitch), np.cos(pitch)]])
    about_y = np.array([[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]])
    return about_y @ about_x


def settle(part_back: np.ndarray, body: np.ndarray, touch: float = TOUCH_SHARE, tuck: float = 0.0) -> float:
    """How far to move a part in depth (+ toward the camera) so its back lies on the body.

    PART_BACK and BODY are front views of the same pixels (the part's farthest surface; the body's front,
    `body_behind`). The part goes back until TOUCH of its back's pixels lie TUCK inside the body and the rest in front of
    it: where the body is nearest it touches, so no air shows under its edge, and nowhere does it sink deeper. 0 when
    they share no pixel."""
    both = np.isfinite(part_back) & np.isfinite(body)
    if not both.any():
        return 0.0
    return float(-tuck - np.quantile(part_back[both] - body[both], touch))


def place_on_body(points: np.ndarray, faces: np.ndarray, target, front: float, body_points: np.ndarray,
                  body_faces: np.ndarray) -> tuple[np.ndarray, dict]:
    """The 4x4 transform setting a built part (POINTS, FACES, turned as it sits on the figure) onto the figure (BODY_POINTS,
    BODY_FACES: every part of the model together), and a report.

    It fills TARGET (its box in the picture carried to its depth) with its nearest point at FRONT (`place_in_box`), is
    turned along the body behind the slab Pixal3D made of it (`body_behind`, `fit_tilt`), and settles with its back on
    that body (`settle`): the picture says where the part is and how big, the figure how deep."""
    p = np.asarray(points, dtype=np.float64)
    tf = place_in_box(p, target, front)
    height = float(np.ptp(hh.moved(tf, p)[:, 1]))
    pixel = height / FIELD_PIXELS

    def maps(transform):
        moved = hh.moved(transform, p)
        lo, hi = moved.min(axis=0), moved.max(axis=0)
        footprint = (lo[0], hi[0], lo[1], hi[1])
        _, back = depth_maps(moved, faces, footprint, pixel)
        body = body_behind(body_points, body_faces, footprint, pixel, THIN_SHARE * height)
        return moved, footprint, back, np.where(np.isfinite(back), body, -np.inf)

    moved, footprint, back, body = maps(tf)
    turn = fit_tilt(back, body, footprint, pixel)
    centre = moved.mean(axis=0)
    about = np.eye(4)
    about[:3, :3] = turn
    about[:3, 3] = centre - turn @ centre
    tf = about @ tf
    moved, footprint, back, body = maps(tf)
    dz = settle(np.where(np.isfinite(back), back, np.nan), np.where(np.isfinite(body), body, np.nan),
                tuck=TUCK_SHARE * height)
    tf[2, 3] += dz
    return tf, {"scale": round(float(np.cbrt(np.linalg.det(tf[:3, :3]))), 5), "front_from_picture": round(front, 5),
                "lean_deg": round(float(np.degrees(np.arctan2(turn[2, 1], turn[1, 1]))), 2),
                "turn_deg": round(float(np.degrees(np.arctan2(turn[0, 2], turn[0, 0]))), 2),
                "settled_by": round(dz, 5), "height": height,
                "body_found": round(float(np.isfinite(body).sum() / max(np.isfinite(back).sum(), 1)), 3)}


def part_field(part: dict, pixel: float, grow: float) -> dict:
    """A placed part's footprint seen from the front, what it replaces on the figure is measured against: "box" and
    "pixel" of the picture, "mask" (its outline, holes filled: the figure's slab behind a trigger guard goes too),
    "field" (the signed distance to that outline grown by GROW, positive inside) and "back" (its farthest surface,
    carried past the outline to the grown one, -inf beyond)."""
    p = np.asarray(part["positions"], dtype=np.float64)
    lo, hi = p.min(axis=0), p.max(axis=0)
    margin = grow + 4 * pixel
    box = (lo[0] - margin, hi[0] + margin, lo[1] - margin, hi[1] + margin)
    front, back = depth_maps(p, part["faces"], box, pixel)
    mask = ndimage.binary_fill_holes(np.isfinite(front))
    steps = int(round(grow / pixel))
    grown = ndimage.binary_dilation(mask, iterations=steps) if steps > 0 else mask
    carried = hf.spread(np.where(np.isfinite(back), back, -np.inf), steps + 2)
    return {"box": box, "pixel": pixel, "mask": mask, "field": hf.signed_distance(grown, pixel), "back": carried,
            "front": front}


def field_values(field: dict, points: np.ndarray) -> np.ndarray:
    """The signed distance of POINTS (seen from the front) to the part's grown outline, positive inside.

    A point off the field's picture is at least its distance from the picture's edge outside: a fixed far value put the
    cut on a long edge reaching out of the picture a billionth from its inner end, and the new point welded onto that
    end and doubled the faces round it. A point right on the outline counts as outside: the figure is cut twice along
    it (what goes, and what stays behind the part), and a point both cuts kept doubled them too."""
    p = np.asarray(points, dtype=np.float64)
    values = hf.sample(field["field"], field["box"], field["pixel"], p, outside=np.nan)
    off = np.isnan(values)
    if off.any():
        x0, x1, y0, y1 = field["box"]
        dx = np.maximum(np.maximum(x0 - p[off, 0], p[off, 0] - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - p[off, 1], p[off, 1] - y1), 0.0)
        values[off] = -(np.hypot(dx, dy) + 4 * field["pixel"])
    return np.where(np.abs(values) < 1e-9, -1e-9, values)


def open_loops(part: dict, keep) -> list[np.ndarray]:
    """The closed outlines of PART's open edges (one vertex id per place, `hero_head.cycles`), those whose every point
    KEEP (a function of points) accepts."""
    p = np.asarray(part["positions"], dtype=np.float64)
    f = np.asarray(part["faces"], dtype=np.int64)
    if not len(f):
        return []
    group = hh.weld_groups(p)
    first = np.zeros(group.max() + 1, dtype=np.int64)
    first[group[::-1]] = np.arange(len(group))[::-1]
    welded = group[f]
    directed = np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]])
    directed = directed[directed[:, 0] != directed[:, 1]]
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    single = directed[counts[inverse.reshape(-1)] == 1]
    loops = [first[loop] for loop in hd.cycles(single)]
    return [loop for loop in loops if np.all(keep(p[loop]))]


def drop_inside(part: dict, inside) -> tuple[dict, int]:
    """PART without the pieces that lie wholly where INSIDE (a function of points) says and have an open edge: what a
    cut left of the thing it took away (a slab's back that lay behind the new part's)."""
    p = np.asarray(part["positions"], dtype=np.float64)
    f = np.asarray(part["faces"], dtype=np.int64)
    if not len(f):
        return part, 0
    group = hh.weld_groups(p)
    label = hh.components(f, group)
    within = np.ones(label.max() + 1, bool)
    np.logical_and.at(within, label, np.asarray(inside(p[f].reshape(-1, 3))).reshape(-1, 3).all(axis=1))
    welded = group[f]
    edges = np.sort(np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]]), axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    open_face = (counts[inverse.reshape(-1)] == 1).reshape(3, -1).any(axis=0)
    opened = np.zeros(label.max() + 1, bool)
    opened[label[open_face]] = True
    gone = within & opened
    if not gone.any():
        return part, 0
    return hh.keep_faces(part, ~gone[label])[0], int(gone.sum())


def graft_on_body(parts: list[dict], field: dict, behind: float) -> tuple[list[dict], np.ndarray | None, dict]:
    """PARTS (one model of the figure, part by part, in its frame) with what a placed part replaces cut out.

    Inside the part's outline (FIELD, `part_field`) whatever lies in front of the part's back less BEHIND goes: the slab
    Pixal3D made of it, a strap crossing it. What lies further back (the thigh) stays whole. The cut follows the outline
    (triangles split on it, `hero_hands.cut_part_values`), so it is a clean line under the part's edge; pieces the cut
    opened that lie wholly round the part go (a slab's back behind the new part's, a flap of it below); every outline the cut leaves
    round the part is closed by a fan (`hero_head.cap_loops`), so no edge stays open. Returns (the cut parts, the skin
    weights of the faces that went as rows, or None for a model without, report)."""
    box, pixel = field["box"], field["pixel"]
    out, rows = [], []
    report = {"faces_removed": 0, "pieces_dropped": 0, "outlines_closed": 0, "parts": []}

    x0, x1, y0, y1 = box
    margin = REMNANT_SHARE * (y1 - y0)

    def near_part(points):
        # what a cut left of the slab: inside the outline (its back behind the new part's), or beside it where the slab
        # reached past the new part (2 cm below the Revolverheld's holster, a flap the cut left standing); the body, the
        # coat and the belt reach further and stay
        return ((points[:, 0] > x0 - margin) & (points[:, 0] < x1 + margin) & (points[:, 1] > y0 - margin)
                & (points[:, 1] < y1 + margin))

    def opened(points):
        # an outline the graft opened reaches inside, and nowhere further out than a face it split reached (a
        # 48-point outline round the Revolverheld's thigh ran 3 pixels out at one end)
        values = field_values(field, points)
        return np.full(len(points), bool(values.max() > 0) and bool(values.min() > -reach))

    for part in parts:
        p = np.asarray(part["positions"], dtype=np.float64)
        f = np.asarray(part["faces"], dtype=np.int64)
        values = field_values(field, p)
        touched = (values[f] > 0).any(axis=1)
        if not touched.any():
            out.append(part)
            report["parts"].append({"cut": False})
            continue
        middle = p[f].mean(axis=1)
        back = hf.sample(field["back"], box, pixel, middle, outside=-np.inf, nearest=True)
        cuttable = touched & np.isfinite(back) & (middle[:, 2] > back - behind)
        gone = cuttable & (values[f] > 0).all(axis=1)
        if part.get("dense") is not None and gone.any():
            rows.append(np.asarray(part["dense"])[np.unique(f[gone])])
        # Every face the outline crosses is split, whether it goes or stays: cut only where it goes, the thigh's
        # face next to a slab's kept the whole edge the slab's had split, and a crack opened along the outline
        # (110 open edges round each of the Revolverheld's pistols, 2026-11-22).
        cut, _ = hh.cut_part_values(part, values)
        if (~cuttable).any():
            stays, _ = hh.keep_faces(part, ~cuttable)
            inside, _ = hh.cut_part_values(stays, -field_values(field, stays["positions"]))
            if len(inside["faces"]):
                cut = hh.merge_parts([cut, inside])
        cut, dropped = drop_inside(cut, near_part)
        crossed = f[(values[f] > 0).any(axis=1) & (values[f] <= 0).any(axis=1)]
        reach = float(np.linalg.norm(p[crossed] - p[np.roll(crossed, 1, axis=1)], axis=2).max()) if len(crossed) else pixel
        cut, closed = hd.cap_loops(cut, open_loops(cut, opened))
        report["faces_removed"] += int(gone.sum())
        report["pieces_dropped"] += dropped
        report["outlines_closed"] += closed
        report["parts"].append({"cut": True, "faces_removed": int(gone.sum()), "outlines_closed": closed})
        out.append(cut)
    return out, (np.concatenate(rows) if rows else None), report


def rigid_weights(rows: np.ndarray) -> np.ndarray:
    """One row of skin weights for every point of a hard part: the mean of ROWS (the weights of what it replaced), so
    it moves as one piece with the bones the slab followed and never bends."""
    mean = np.asarray(rows, dtype=np.float64).mean(axis=0)
    return mean / max(float(mean.sum()), 1e-12)


# --- A part at a limb's end ---------------------------------------------------------------------------------------------

# A boot is drawn from its outer side in profile, its toe to the picture's right (the right boot) or left (the left
# one), and Pixal3D builds the drawing's view facing +Z: these turn the build to stand on the figure's foot, toe
# forward (+Z), its outer side out (the figure's right is -X).
BOOT_TURN = {"right": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
             "left": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])}


def leg_bones(parents: dict[str, str | None], heads: dict[str, np.ndarray]) -> tuple[dict[str, dict], np.ndarray]:
    """({side: {"thigh", "shin", "foot", "below"}}, the figure's frame) from a rig's skeleton, found by the tree's shape
    with the mapper the preset moves use, as `hero_hands.arm_bones` finds the arms. SIDE is the figure's own."""
    if str(hh.SCRIPTS) not in sys.path:
        sys.path.insert(0, str(hh.SCRIPTS))
    from kimodo_retarget import map_skintokens_to_soma

    mapping, frame = map_skintokens_to_soma(parents, heads, up=UP)
    children: dict[str, list[str]] = {}
    for bone, parent in parents.items():
        if parent is not None:
            children.setdefault(parent, []).append(bone)

    def first(role: str) -> str | None:
        named = [b for b, r in mapping.items() if r == role and mapping.get(parents.get(b)) != role]
        return named[0] if named else None

    legs = {}
    for side in hh.SIDES:
        prefix = side.capitalize()
        shin, foot = first(f"{prefix}Shin"), first(f"{prefix}Foot")
        if shin is None or foot is None:
            continue
        below, stack = [], list(children.get(foot, []))
        while stack:
            bone = stack.pop()
            below.append(bone)
            stack.extend(children.get(bone, []))
        legs[side] = {"thigh": first(f"{prefix}Leg"), "shin": shin, "foot": foot, "below": sorted(below)}
    return legs, frame


def _across(points: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """How far POINTS lie from the line through TOP and BOTTOM (a leg's axis), across it."""
    axis = hh.unit(np.asarray(bottom, dtype=np.float64) - top)
    rel = np.asarray(points, dtype=np.float64) - top
    return np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)


def leg_plan(positions: np.ndarray, bone: np.ndarray, shin: int, foot: list[int], knee: np.ndarray,
             ankle: np.ndarray) -> dict:
    """Where a figure's old boot is: {"floor" (its sole's lowest point), "foot" (its points below the ankle), "knee",
    "ankle", "reach" (how far from the leg's axis the old boot is cut), "bones"}. SHIN and FOOT (the foot bone and every
    bone below it) name the leg; KNEE and ANKLE are the rig's joints."""
    p = np.asarray(positions, dtype=np.float64)
    leg = np.isin(np.asarray(bone), [shin, *foot])
    if not leg.any():
        raise ValueError("no vertex of the figure belongs to the leg's bones")
    knee, ankle = np.asarray(knee, dtype=np.float64), np.asarray(ankle, dtype=np.float64)
    shaft = leg & (p[:, 1] < knee[1]) & (p[:, 1] > ankle[1])
    radius = float(np.percentile(_across(p[shaft], knee, ankle), 90)) if shaft.any() else 0.0
    foot_points = p[leg & (p[:, 1] <= ankle[1])]
    if not len(foot_points):
        raise ValueError("the leg's bones hold nothing below the ankle: no foot to stand a boot on")
    reach = max(LEG_REACH * radius, FOOT_REACH * float(_across(foot_points, knee, ankle).max()))
    return {"floor": float(foot_points[:, 1].min()), "foot": foot_points, "knee": knee, "ankle": ankle,
            "reach": reach, "radius": radius, "bones": [int(shin), *map(int, foot)]}


def foot_heading(points: np.ndarray) -> np.ndarray:
    """A foot's heading seen from above, (x, z) of unit length: its points' longest direction, toward +Z."""
    flat = np.asarray(points, dtype=np.float64)[:, [0, 2]]
    flat = flat - flat.mean(axis=0)
    _, vectors = np.linalg.eigh(flat.T @ flat)
    heading = vectors[:, -1]
    return heading if heading[1] >= 0 else -heading


def _yaw(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])


def _foot_box(points: np.ndarray, heading: np.ndarray) -> tuple[float, np.ndarray]:
    """(length along HEADING, middle (x, z)) of a foot's points, from their 1st to 99th percentile."""
    flat = np.asarray(points, dtype=np.float64)[:, [0, 2]]
    across = np.array([heading[1], -heading[0]])
    along, side = flat @ heading, flat @ across
    a0, a1 = np.percentile(along, [1, 99])
    s0, s1 = np.percentile(side, [1, 99])
    return float(a1 - a0), heading * (a0 + a1) / 2 + across * (s0 + s1) / 2


def place_boot(points: np.ndarray, old_foot: np.ndarray, floor: float, height: float | None = None,
               limit: float = HEADING_LIMIT_DEG) -> np.ndarray:
    """The 4x4 transform standing a built boot (POINTS, turned toe forward: `BOOT_TURN`) in a figure's old boot: HEIGHT
    tall from sole to top when the picture says how tall (its top in the front view), else as long from heel to toe as
    the old foot (OLD_FOOT, its points below the ankle, a guess of Pixal3D's, and the drawing's foot may be short for
    its shaft: the Zauberer's boot came out 28 per cent too tall that way); turned about the up axis to the old foot's
    heading (at most LIMIT degrees), its foot's middle over the old one's, its lowest point on FLOOR."""
    p = np.asarray(points, dtype=np.float64)
    heading = foot_heading(old_foot)
    yaw = float(np.clip(np.degrees(np.arctan2(heading[0], heading[1])), -limit, limit))
    turn = _yaw(yaw)
    old_length, old_middle = _foot_box(old_foot, heading)
    if height is not None:
        k = height / max(float(np.ptp(p[:, 1])), 1e-12)
    else:
        low = p[p[:, 1] < p[:, 1].min() + BOOT_FOOT_SHARE * np.ptp(p[:, 1])]
        k = old_length / max(_foot_box(low, np.array([0.0, 1.0]))[0], 1e-12)
    moved = (p * k) @ turn.T
    low = moved[moved[:, 1] < moved[:, 1].min() + BOOT_FOOT_SHARE * np.ptp(moved[:, 1])]
    _, new_middle = _foot_box(low, np.array([np.sin(np.radians(yaw)), np.cos(np.radians(yaw))]))
    tf = np.eye(4)
    tf[:3, :3] = k * turn
    tf[:3, 3] = [old_middle[0] - new_middle[0], floor - moved[:, 1].min(), old_middle[1] - new_middle[1]]
    return tf


def level_sole(points: np.ndarray, limit: float = LEVEL_LIMIT_DEG) -> np.ndarray:
    """The 4x4 turn (about the boot's lowest point) laying a built boot's sole flat: the lowest point of every cell of a
    grid seen from below, those within SOLE_BAND_SHARE of the height of the lowest, fitted by a plane, whose tilt is
    taken out when it is at most LIMIT degrees. A drawing seen a little from above builds a sole that rests on its heel
    or its toe alone."""
    p = np.asarray(points, dtype=np.float64)
    height = float(np.ptp(p[:, 1]))
    lo = p.min(axis=0)
    cell = max(float(np.ptp(p[:, [0, 2]], axis=0).max()) / 24, 1e-12)
    keys = np.floor((p[:, [0, 2]] - lo[[0, 2]]) / cell).astype(np.int64)
    order = np.lexsort((p[:, 1], keys[:, 1], keys[:, 0]))
    first = order[np.r_[True, np.any(np.diff(keys[order], axis=0) != 0, axis=1)]]
    low = p[first]
    low = low[low[:, 1] < lo[1] + 3 * SOLE_BAND_SHARE * height]
    if len(low) < 3:
        return np.eye(4)
    design = np.c_[low[:, 0], low[:, 2], np.ones(len(low))]
    use = np.ones(len(low), bool)
    for _ in range(3):                     # the contact cells: the arch and the toe's lift drop out
        coef = np.linalg.lstsq(design[use], low[use, 1], rcond=None)[0]
        miss = low[:, 1] - design @ coef
        use = miss <= np.quantile(miss, 0.6) + 1e-12
    normal = hh.unit(np.array([-coef[0], 1.0, -coef[1]]))
    tilt = float(np.degrees(np.arccos(np.clip(normal @ UP, -1.0, 1.0))))
    if tilt < 1e-6 or tilt > limit:
        return np.eye(4)
    turn = hh.rotation_between(normal, np.array([1.0, 0.0, 0.0]) - normal[0] * normal, UP, np.array([1.0, 0.0, 0.0]))
    centre = p[np.argmin(p[:, 1])]
    tf = np.eye(4)
    tf[:3, :3] = turn
    tf[:3, 3] = centre - turn @ centre
    return tf


def boot_rim(points: np.ndarray, sectors: int = 16, band: float = 0.15) -> float:
    """The lowest point of a placed boot's top rim: the highest point of its top BAND of its height in each of SECTORS
    slices round its shaft, the lowest of those. A shaft's top often slants (the Zauberer's drawn boots ran lower at the
    back); a leg cut under its highest point stood above the back of the rim as a ledge."""
    p = np.asarray(points, dtype=np.float64)
    high = p[p[:, 1] > p[:, 1].max() - band * float(np.ptp(p[:, 1]))]
    middle = high[:, [0, 2]].mean(axis=0)
    angle = np.arctan2(high[:, 0] - middle[0], high[:, 2] - middle[1])
    slot = np.clip(((angle + np.pi) / (2 * np.pi) * sectors).astype(int), 0, sectors - 1)
    tops = np.full(sectors, -np.inf)
    np.maximum.at(tops, slot, high[:, 1])
    return float(tops[np.isfinite(tops)].min())


def tuck_inside(points: np.ndarray, boot: np.ndarray, boot_faces: np.ndarray, low: float, high: float,
                keep: float = 0.96, rows: int = 24, slices: int = 24, samples: int = 40000) -> np.ndarray:
    """POINTS of the old leg with those between LOW (the cut) and HIGH (the new boot's top) pulled inside the new
    boot (BOOT, BOOT_FACES; its surface sampled SAMPLES times): each to KEEP of the boot's own radius at its height and
    angle round the boot's middle there, when it lies further out. The old boot's top stood wider than the new boot at
    the back and showed over its rim as a ledge; above HIGH, and where the boot has no surface (over a rim that runs
    lower), nothing moves."""
    p = np.array(points, dtype=np.float64)
    tri = np.asarray(boot, dtype=np.float64)[np.asarray(boot_faces, dtype=np.int64)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(tri), size=samples, p=area / area.sum())
    u, v = rng.random(samples), rng.random(samples)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    b = tri[pick, 0] + u[:, None] * (tri[pick, 1] - tri[pick, 0]) + v[:, None] * (tri[pick, 2] - tri[pick, 0])
    band = (p[:, 1] >= low) & (p[:, 1] <= high)
    near = b[(b[:, 1] >= low) & (b[:, 1] <= high)]
    if not band.any() or len(near) < 3:
        return p
    edges = np.linspace(low, high, rows + 1)

    def row_of(y):
        return np.clip(np.searchsorted(edges, y, side="right") - 1, 0, rows - 1)

    boot_row, old_row = row_of(near[:, 1]), row_of(p[band, 1])
    middle = np.full((rows, 2), np.nan)
    for r in np.unique(boot_row):
        middle[r] = near[boot_row == r][:, [0, 2]].mean(axis=0)
    centre = middle[boot_row]

    def slice_of(q, c):
        return np.clip(((np.arctan2(q[:, 0] - c[:, 0], q[:, 2] - c[:, 1]) + np.pi) / (2 * np.pi) * slices).astype(int),
                       0, slices - 1)

    radius = np.full((rows, slices), np.nan)
    reach = np.hypot(near[:, 0] - centre[:, 0], near[:, 2] - centre[:, 1])
    where = (boot_row, slice_of(near, centre))
    np.fmax.at(radius, where, reach)
    old = p[band]
    old_centre = middle[old_row]
    known = np.isfinite(old_centre[:, 0])
    out = old.copy()
    if known.any():
        c = old_centre[known]
        q = old[known]
        r_old = np.hypot(q[:, 0] - c[:, 0], q[:, 2] - c[:, 1])
        r_boot = radius[old_row[known], slice_of(q, c)]
        pull = np.isfinite(r_boot) & (r_old > keep * r_boot)
        scale = np.where(pull, keep * r_boot / np.maximum(r_old, 1e-12), 1.0)
        q = q.copy()
        q[:, 0] = c[:, 0] + (q[:, 0] - c[:, 0]) * scale
        q[:, 2] = c[:, 1] + (q[:, 2] - c[:, 1]) * scale
        out[known] = q
    p[band] = out
    return p


def graft_boot(parts: list[dict], bones: list[np.ndarray], plan: dict, cut_y: float,
               boot: dict | None = None) -> tuple[list[dict], dict]:
    """PARTS (one model of the figure, part by part; BONES the main bone of every vertex of each) with one old boot cut
    off: the leg (PLAN, `leg_plan`) cut straight across at CUT_Y, just under the new boot's rim; whatever of the leg's
    bones lies below that within its reach goes (a coat hanging beside it stays: other bones, or out of reach), and the
    cut's outlines are closed by fans inside the new boot. With the placed new BOOT (positions, faces) the old leg
    between the cut and the boot's top is tucked inside it (`tuck_inside`). Returns (parts, report)."""
    out = []
    report = {"faces_removed": 0, "outlines_closed": 0, "points_tucked": 0}
    knee, ankle, reach = plan["knee"], plan["ankle"], plan["reach"]

    def low_and_near(points):
        return (points[:, 1] < cut_y + 1e-9) & (_across(points, knee, ankle) < reach)

    for part, bone in zip(parts, bones):
        p = np.asarray(part["positions"], dtype=np.float64)
        f = np.asarray(part["faces"], dtype=np.int64)
        on_leg = np.isin(np.asarray(bone), plan["bones"])[f].any(axis=1)
        middle = p[f].mean(axis=1)
        cuttable = on_leg & (_across(middle, knee, ankle) < reach) & (p[f][:, :, 1].min(axis=1) < cut_y)
        if not cuttable.any():
            out.append(part)
            continue
        gone = cuttable & (p[f][:, :, 1].max(axis=1) < cut_y)
        marked = {**part, "marks": np.isin(np.asarray(bone), plan["bones"]).astype(float)[:, None]}
        cut, _ = hh.cut_part(marked, np.array([0.0, cut_y, 0.0]), -UP, cuttable)
        cut, _ = drop_inside(cut, low_and_near)
        loops = open_loops(cut, lambda q: (np.abs(q[:, 1] - cut_y) < 1e-6) & (_across(q, knee, ankle) < reach))
        cut, closed = hd.cap_loops(cut, loops)
        if boot is not None:
            q = np.asarray(cut["positions"], dtype=np.float64)
            leg = (np.asarray(cut["marks"])[:, 0] > 0.5) & (_across(q, knee, ankle) < reach)
            top = float(np.asarray(boot["positions"])[:, 1].max())
            tucked = tuck_inside(q[leg], boot["positions"], boot["faces"], cut_y, top)
            report["points_tucked"] += int(np.any(tucked != q[leg], axis=1).sum())
            q[leg] = tucked
            cut = {**cut, "positions": q}
        report["faces_removed"] += int(gone.sum())
        report["outlines_closed"] += closed
        out.append({k: v for k, v in cut.items() if k != "marks"})
    return out, report


def sole_below_feet(view: pp.View, points: np.ndarray, mesh_scale: float = 1.0, band: float = 0.01) -> float:
    """How far a boot's lowest point lies below the picture's feet where it stands, in the view's pixels (positive:
    below them): the foot rule's question, whether a sole was made up below the pictures' feet (as the game-ready
    step's --feet asks of a whole figure), asked of one new boot. Its sole is its lowest BAND of its height; the
    picture's feet are its subject's lowest pixel in the columns the sole covers. NaN when the picture has none there."""
    p = np.asarray(points, dtype=np.float64)
    x, y, _ = view_pixels(view, p, mesh_scale)
    alpha = np.asarray(view.image)[..., 3] > 127
    columns = np.clip(np.floor(x).astype(np.int64), 0, alpha.shape[1] - 1)
    sole = p[:, 1] <= p[:, 1].min() + band * float(np.ptp(p[:, 1]))
    bottoms = [np.flatnonzero(alpha[:, c]).max() + 1 for c in np.unique(columns[sole]) if alpha[:, c].any()]
    return float(y[sole].max() - max(bottoms)) if bottoms else float("nan")


def boot_weights(points: np.ndarray, part: dict, bone: np.ndarray, plan: dict) -> np.ndarray:
    """A new boot's skin weights, point by point: those of the nearest point of the old leg below the knee (the old
    boot and the trousers above it), so the toe moves with the toes' bone, the heel with the foot's and the shaft with
    the shin's, and the boot bends at the ankle as the old one did."""
    p = np.asarray(part["positions"], dtype=np.float64)
    old = (np.isin(np.asarray(bone), plan["bones"]) & (p[:, 1] < plan["knee"][1])
           & (_across(p, plan["knee"], plan["ankle"]) < plan["reach"]))
    return hh.borrowed_weights(points, p[old], np.asarray(part["dense"])[old])


# --- Hard edges ---------------------------------------------------------------------------------------------------------

def _adjacent_faces(wfaces: np.ndarray) -> np.ndarray:
    """Pairs of faces that share an edge (one shared by exactly two faces)."""
    m = len(wfaces)
    edges = np.sort(np.concatenate([wfaces[:, [0, 1]], wfaces[:, [1, 2]], wfaces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(m), 3)
    _, edge_id, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    edge_id = edge_id.reshape(-1)
    shared = counts[edge_id] == 2
    order = np.argsort(edge_id[shared], kind="stable")
    return owner[shared][order].reshape(-1, 2)


def _angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)))


def tidy_planes(label: np.ndarray, pairs: np.ndarray, normal: np.ndarray, planes: np.ndarray,
                rounds: int = 2, close: float = PLANE_GROW_DEG) -> np.ndarray:
    """LABEL (a plane id per face, -1 off every plane) with its planes' edges tidied, ROUNDS times: a face off a plane
    whose neighbours (PAIRS of faces sharing an edge) lie on one plane more than on anything else, and whose own normal
    is within CLOSE degrees of it, joins it; a plane face with fewer neighbours on its plane than off every plane (or
    none on it) leaves it, while a face in a corner, between its plane and others, stays. On a
    build's clean-up the plane's edge ran round single triangles, and the hard shading drew it as a notched line."""
    label = np.asarray(label).copy()
    m = len(label)
    for _ in range(rounds):
        votes = np.zeros((m, len(planes) + 1))
        for here, there in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
            np.add.at(votes, (here, label[there] + 1), 1.0)
        best = np.argmax(votes[:, 1:], axis=1)
        strong = votes[:, 1:][np.arange(m), best] > np.maximum(votes[:, 0], 1.0)
        fits = np.einsum("ij,ij->i", normal, planes[best, :3]) > np.cos(np.radians(close))
        own = np.where(label >= 0, votes[np.arange(m), label + 1], 0.0)
        new = label.copy()
        new[(label < 0) & strong & fits] = best[(label < 0) & strong & fits]
        new[(label >= 0) & ((own == 0) | (own < votes[:, 0]))] = -1
        if np.array_equal(new, label):
            break
        label = new
    return label


def face_planes(positions: np.ndarray, faces: np.ndarray, rounds: int = SMOOTH_ROUNDS,
                link: float = PLANE_LINK_DEG, crease: float = PLANE_CREASE_DEG, mode: float = PLANE_MODE_DEG,
                spread: float = PLANE_SPREAD_DEG, facet: float = PLANE_FACET_DEG, least: float = PLANE_AREA_SHARE,
                flat: float = PLANE_FLAT_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """(a plane id per face, -1 off every plane; the planes as rows (nx, ny, nz, offset)): a hard part's nearly flat
    pieces.

    A build's flat side is a little wavy, so its faces' normals are smoothed over ROUNDS rings of neighbours first, never
    across an edge where they turn by CREASE degrees or more; neighbours whose smoothed normals differ by less than LINK
    degrees lie on one surface. A surface is a plane when its faces lie within SPREAD degrees of its normal (root mean
    square) and its points within FLAT of the part's size of it (95 per cent of them); on a curved one (a grip that
    bends into a flat side) its flattest piece is tried, the faces within MODE degrees of its commonest normal, and a
    tube's strip that wide spreads too much. Two planes that meet at less than FACET degrees are the facets of a curved
    surface (a cylinder of a few dozen sides) and stay smooth. Planes smaller than LEAST of the surface are left out."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    points, wfaces, _ = hh.weld(positions, faces)
    tri = points[wfaces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(2 * area, 1e-18)[:, None]
    label = np.full(len(wfaces), -1, dtype=np.int64)
    if not len(wfaces):
        return label, np.zeros((0, 4))
    pairs = _adjacent_faces(wfaces)
    raw_turn = _angles(normal[pairs[:, 0]], normal[pairs[:, 1]])
    a, b = pairs[raw_turn < crease, 0], pairs[raw_turn < crease, 1]
    smooth = normal.copy()
    for _ in range(rounds):
        total = smooth * area[:, None]
        np.add.at(total, a, smooth[b] * area[b, None])
        np.add.at(total, b, smooth[a] * area[a, None])
        smooth = total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-18)
    linked = pairs[(_angles(smooth[pairs[:, 0]], smooth[pairs[:, 1]]) < link) & (raw_turn < crease)]
    m = len(wfaces)
    size = float(np.ptp(points, axis=0).max()) or 1.0
    total = float(area.sum())

    def pieces(select: np.ndarray) -> np.ndarray:
        """Components of the linked faces among SELECT (a label per face, -1 outside it)."""
        keep = select[linked[:, 0]] & select[linked[:, 1]]
        graph = coo_matrix((np.ones(int(keep.sum())), (linked[keep, 0], linked[keep, 1])), shape=(m, m))
        out = connected_components(graph, directed=False)[1]
        return np.where(select, out, -1)

    planes = []

    def accept(index: np.ndarray) -> bool:
        weight = area[index]
        if weight.sum() < least * total:
            return False
        n = hh.unit((normal[index] * weight[:, None]).sum(axis=0))
        if np.sqrt(np.average(_angles(smooth[index], np.tile(n, (len(index), 1))) ** 2, weights=weight)) > spread:
            return False
        corners = points[np.unique(wfaces[index])]
        centre = corners.mean(axis=0)
        fitted = np.linalg.svd(corners - centre, full_matrices=False)[2][-1]     # the least-squares plane
        n = fitted if fitted @ n >= 0 else -fitted
        if np.percentile(np.abs((corners - centre) @ n), 95) > flat * size:
            return False
        label[index] = len(planes)
        planes.append(np.r_[n, n @ centre])
        return True

    first = pieces(np.ones(m, bool))
    for piece in np.unique(first):
        index = np.flatnonzero(first == piece)
        if area[index].sum() < least * total or accept(index):
            continue
        free = np.zeros(m, bool)
        free[index] = True
        for _ in range(3):                  # its flattest pieces, one commonest normal at a time
            candidates = np.flatnonzero(free)
            if area[candidates].sum() < least * total:
                break
            sample = candidates[np.linspace(0, len(candidates) - 1, min(len(candidates), 400)).astype(int)]
            near = smooth[candidates] @ smooth[sample].T > np.cos(np.radians(mode))
            peak = smooth[sample[np.argmax(area[candidates] @ near)]]
            chosen = np.zeros(m, bool)
            chosen[candidates[smooth[candidates] @ peak > np.cos(np.radians(mode))]] = True
            free &= ~chosen
            sub = pieces(chosen)
            for piece_id in np.unique(sub[sub >= 0]):
                accept(np.flatnonzero(sub == piece_id))
    planes = np.array(planes).reshape(-1, 4)
    if not len(planes):
        return label, planes
    # A plane's edge, where the noise stopped the linking, grows out to where the surface really turns: faces next to
    # it within GROW degrees of it and as close to it as its own points, so its sharp edge runs along the crease.
    middle = tri.mean(axis=1)
    for _ in range(PLANE_GROW_ROUNDS):
        one, two = label[pairs[:, 0]], label[pairs[:, 1]]
        added = False
        for here, there in ((one, pairs[:, 1]), (two, pairs[:, 0])):
            other = label[there]
            take = (here >= 0) & (other < 0)
            if not take.any():
                continue
            plane = planes[here[take]]
            close = ((np.einsum("ij,ij->i", normal[there[take]], plane[:, :3]) > np.cos(np.radians(PLANE_GROW_DEG)))
                     & (np.abs(np.einsum("ij,ij->i", middle[there[take]], plane[:, :3]) - plane[:, 3])
                        < flat * size))
            if close.any():
                label[there[take][close]] = here[take][close]
                added = True
        if not added:
            break
    label = tidy_planes(label, pairs, normal, planes)
    one, two = label[pairs[:, 0]], label[pairs[:, 1]]
    meet = (one >= 0) & (two >= 0) & (one != two)
    gentle = meet & (_angles(planes[np.maximum(one, 0), :3], planes[np.maximum(two, 0), :3]) < facet)
    facets = np.zeros(len(planes), bool)
    facets[one[gentle]] = facets[two[gentle]] = True
    renumber = np.where(facets, -1, np.cumsum(~facets) - 1)
    label = np.where(label >= 0, renumber[np.maximum(label, 0)], -1)
    return label, planes[~facets]


def harden(part: dict, move: float = 2 * PLANE_FLAT_SHARE) -> tuple[dict, dict]:
    """PART (a hard part: a pistol, a buckle; never a face or cloth) with its nearly flat pieces made flat and their
    edges sharp, the clean cut look of a hard surface.

    Every point of a plane (`face_planes`) goes onto it, a point on two onto the line they meet in, a point on three or
    more onto their corner (a least-squares point, so the build's rounded edge becomes a crease), when that moves it
    at most MOVE of the part's size. A plane's faces take its normal and a point on its edge is split, one copy for each
    side, so the edge shades sharp; faces off every plane keep smooth normals among themselves. Returns (the part,
    report)."""
    p = np.asarray(part["positions"], dtype=np.float64)
    f = np.asarray(part["faces"], dtype=np.int64)
    label, planes = face_planes(p, f)
    if not len(planes):
        return part, {"planes": 0, "plane_faces": 0, "faces": int(len(f)), "points_moved": 0, "points_split": 0}
    points, wfaces, group = hh.weld(p, f)
    size = float(np.ptp(points, axis=0).max()) or 1.0
    moved = points.copy()
    corners = np.repeat(np.arange(len(wfaces)), 3)
    on_plane = label[corners] >= 0
    pairs = np.unique(np.stack([wfaces.reshape(-1)[on_plane], label[corners][on_plane]], axis=1), axis=0)
    shifted = 0
    if len(pairs):
        starts = np.flatnonzero(np.r_[True, pairs[1:, 0] != pairs[:-1, 0]])
        for start, end in zip(starts, np.r_[starts[1:], len(pairs)]):
            vertex, own = pairs[start, 0], planes[pairs[start:end, 1]]
            n, d = own[:, :3], own[:, 3]
            lam = 1e-4
            target = np.linalg.solve(n.T @ n + lam * np.eye(3), n.T @ d + lam * points[vertex])
            if np.linalg.norm(target - points[vertex]) <= move * size:
                moved[vertex] = target
                shifted += 1
    tri = moved[wfaces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    off = label < 0
    smooth = np.zeros((len(moved), 3))
    for k in range(3):
        np.add.at(smooth, wfaces[off, k], cross[off])
    smooth /= np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-18)
    keys = np.stack([f.reshape(-1), label[corners]], axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    source = unique[:, 0]
    normals = np.where((unique[:, 1] >= 0)[:, None], planes[np.maximum(unique[:, 1], 0), :3], smooth[group[source]])
    out = {**part, "positions": moved[group[source]], "faces": inverse.reshape(-1, 3), "normals": normals}
    for key in hh.ATTRIBUTES:
        if key != "normals" and part.get(key) is not None:
            out[key] = np.asarray(part[key])[source]
    split = len(unique) - len(np.unique(f))
    return out, {"planes": int(len(planes)), "plane_faces": int((label >= 0).sum()), "faces": int(len(f)),
                 "points_moved": int(shifted), "points_split": int(split)}


# --- Counting -----------------------------------------------------------------------------------------------------------

def pieces(part: dict) -> int:
    """How many separate pieces PART is (faces joined at a shared place, `hero_hands.components`)."""
    f = np.asarray(part["faces"], dtype=np.int64)
    if not len(f):
        return 0
    return int(hh.components(f, hh.weld_groups(np.asarray(part["positions"]))).max() + 1)


def edge_faults(part: dict) -> dict:
    """PART's edges (welded by place) that only one face uses ("open": holes) and that more than two use
    ("shared_by_more": where its surface touches itself), and that second count's share of all its edges. A build whose
    thin wall pinched onto itself in many places (the Zauberer's first boot build: 21,107 such edges, 1.4 per cent; the
    pistol's 0.08) could not be thinned by the game-ready step, which kept 81,000 of its 14,000 triangles."""
    group = hh.weld_groups(np.asarray(part["positions"]))
    f = group[np.asarray(part["faces"], dtype=np.int64)]
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if not len(edges):
        return {"open": 0, "shared_by_more": 0, "share": 0.0}
    _, counts = np.unique(edges, axis=0, return_counts=True)
    more = int((counts > 2).sum())
    return {"open": int((counts == 1).sum()), "shared_by_more": more, "share": round(more / len(counts), 5)}


def drop_crumbs(part: dict, share: float = CRUMB_SHARE) -> tuple[dict, int]:
    """(PART without its crumbs, how many went): separate pieces with fewer than SHARE of its faces, the specks a build
    leaves floating round a part."""
    f = np.asarray(part["faces"], dtype=np.int64)
    label = hh.components(f, hh.weld_groups(np.asarray(part["positions"])))
    counts = np.bincount(label)
    crumb = counts < share * len(f)
    if not crumb.any():
        return part, 0
    return hh.keep_faces(part, ~crumb[label])[0], int(crumb.sum())
