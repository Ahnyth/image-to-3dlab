"""Hero head under the brim: everything of a head between a hat's brim and a collar built again on its own and set in,
as pure functions.

Hero face (`hero_face.py`) swaps only what the front view shows of a face. On the Zauberer (2026-11-22) that left the
coarse full-figure head all round it: the face's seam ran from the eye down the cheek, a thick side strand of the old
head hung between ear and face, and under the brim the old hair tips were open shells with gaps between them through
which the hat's inside showed. The fine head built alone has clean ears, side strands and hair under its brim, so here
the whole head goes, from the brim's underside down into the collar, and the seams sit where nobody sees them: the new
hair reaches up into the hat, the new neck down inside the collar.

A head is unrolled round a vertical axis through it (`cylinder`): every column an angle, every row a height, and along
each horizontal ray from the axis the model's crossings (`cylinder_hits`). From them come the outermost surface (hair,
ears, the face), the skin (the first surface the ray leaves going out) and whatever stands round the head beyond a gap
(a collar, facing the head) (`layers`). A collar is told from a strand standing off the skull by reaching down to the
bottom of the picture, into the body (`collar_mask`). The brim's underside is a plane read from below in a ring round
the head (`brim_plane`), and the hat's opening, where the hair meets it, is found all round from the brim's own faces
(`brim_faces`, `hat_opening`). The region to replace lies under that plane and up to the opening, above a height a
little below the chin, and inside the collar wherever there is one (`head_zone`, `zone_values`).

The new head is a tube round the same axis: its radius in every column and row is the build's outermost head surface,
or its skin behind a collar (`tube_radius`), kept inside the figure's collar, at its own width up into the hat
(`crown_width`), its creases rounded. It is laid out as a grid closed all round (`tube_part`), capped inside the hat and
stitched to the figure's own neck inside the collar (`graft_head`); the hat's opening is closed in the hat's colour
(`opening_cap`, `hat_colour`), so no seam is open.

Models face +Z (glTF's front) with Y up, as Pixal3D builds them.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from image_to_3dlab import hero_face as hf
from image_to_3dlab import hero_hands as hh

FACING_LIMIT = 0.1      # a crossing faces out (or in) when its normal's share along the ray is past this; flatter ones
                        # (a brim's underside, a strand's tip) count only for the outermost surface
GAP_SHARE = 0.04        # a collar's inside lies at least this share of the head's width outside the head's skin
AXIS_BACK = 0.55        # the skull's middle is looked for this share of the head's width behind the nose ...
SKULL_SHARE = 0.3       # ... at this share of the way from the nose up to the brim
AXIS_ROUNDS = 2         # the axis is moved this often to lean through the neck's middle
NECK_DEPTH_SHARE = 0.5  # the neck is followed down at most this share of the head's width below the chin ...
NECK_AREA_LIMIT = 0.7   # ... while its section stays smaller than this share of the skull's ...
NECK_GROWTH = 1.5       # ... and does not grow this much from one pixel to the next (a collar fused with it)
NECK_MARGIN = 2         # the stitch sits this many pixels above the lowest free section of the neck
NECK_LEAST = 0.02       # a section holding less than this share of the skull's is a graze, not a neck
COLLAR_MARGIN_SHARE = 0.03   # the new head stays this share of the head's width inside a collar
COLLAR_GROW = 3         # a collar's mask grows this many pixels, so its own rim and edges are never cut
COLLAR_JUMP_SHARE = 0.05     # a collar's outside runs on up without jumps of this share of the head's width
REACH_SHARE = 0.95      # nothing further than this share of the head's width from the axis is head (hair included)
TOP_SHARE = 0.04        # the new head reaches this share of the head's width above the brim's underside, into the hat
CUT_GAP_PIXELS = 2      # the old head is cut this many pixels under the brim's plane (below its lowest residual)
COVER_BAND_SHARE = 0.3  # under the brim, in a band this share of the head's width deep under the cut, the old head is
                        # measured all round ...
COVER_MARGIN_SHARE = 0.015   # ... and the new head is at least this much wider there, so the cut lies inside it ...
COVER_CAP_SHARE = 0.02  # ... and at most this much more again (a piece of the build's brim hanging lower stood out)
COVER_SPREAD = 3        # ... and a column takes the widest of its neighbours this many columns either side ...
COVER_SMOOTH = 20       # ... and the new head is as wide as the widest of it this many columns either side, softened
COVER_SOFTEN = 15       # ... over this many columns (a closer fit and less softening left ridges round the flare)
FLARE_SMOOTH = 0.01     # the flare under the brim starts from the head's width at the band's foot, softened over this
                        # share of the way round ...
FLARE_FADE = 0.4        # ... and the new head takes it wherever the build is narrower, in full from this share of the band up
FLARE_BLUR_SHARE = 0.05 # up the band the build is softened more and more, at the top over this share of the head's width
SIDE_LIMIT = 0.7        # a surface whose normal points this much up or down is a plate (a brim), not a head's side
STAND_SHARE = 0.06      # a collar's top edge stands out at least this share of the head's width from the head above it
BRIM_RING = (0.75, 1.1)  # the brim's underside is read in this ring round the axis (shares of the head's width) ...
BRIM_RING_NEAR = (0.62, 0.85)  # ... or this one, for a build from a crop that cut its brim short
BRIM_LEAST = 0.4        # at least this share of the ring must hold a brim, or there is none to set the head under
BRIM_KEEP = 0.75        # the plane is fitted on the closest three quarters (a strand hanging into the ring is no brim)
BRIM_DIP = 0.0          # the cut stays under all but this share of the brim's underside where it dips below the plane
                        # (a cut brim's dip, closed by a cap, stood out under the brim as a fin)
BRIM_TURN = 0.5         # a brim's underside faces down at least this much (its normal's height part) and not out from
                        # the axis; it is followed from what the ring reads from below within this share of the head's
BRIM_SEED_SHARE = 0.15  # width of the brim's plane
OPENING_GAP_PIXELS = 2  # the old head is cut out up to this many pixels over the hat's opening (where its hair meets
OPENING_RIM_SHARE = 0.02    # the brim) and this share of the head's width beyond it, the brim's innermost ring with it
OPENING_LEAST = 0.5     # a brim found round less than this share of the head has no opening to cut up to
OPENING_SPREAD = 0.015  # the opening is taken at its highest and furthest out over this share of the way round
CROWN_EASE = (0.4, 0.75)    # up the band under the brim the new head eases from the build's surface to its own width
                            # between these shares of the band ...
CROWN_READ = 0.3        # ... read over the band from this share of it up to the cut, where the build shows its head ...
CROWN_PERCENTILE = 75   # ... as this percentile of its softened surface (a notch or a stray plate near the build's own
CROWN_SPREAD = 0.15     # brim sets nothing), a running median over this share of the way round, softened after
HAT_RING_SHARE = 0.15   # the hat's colour is read on its brim's underside within this share of the head's width of the
                        # opening ...
HAT_PAINT_SHARE = 0.15  # ... and the figure's paint in it on the head is not taken within this share of it under the cut
BAND_DARK = 30          # on the new neck the figure's own paint this much darker (L*) than its skin is the collar's dark:
BAND_SHADE = 0.85       # it takes the skin's colour this much shaded ...
BAND_SOFT = 15          # ... and from this much darker on it eases toward it, so no ragged edge stands where the two meet
MEMBRANE_RINGS = 4      # an outline the cut leaves at the hat's opening is spanned by a film with this many rings inside
MEMBRANE_LEAST = 8      # ... when it has at least this many points (a smaller one by a fan)
FRAME_SHARE = 0.75      # the picture's pixels are square this far from the axis (the face's front)
MIN_COLUMNS = 16
AXIS_CELLS = 6          # this many cells round the axis count as solid in a column's section ...
SPLIT_ROUNDS = 7        # ... where a cap's fan, split this often, still has no angle to raster
CROP_FACING = (0.35, 0.8)   # the picture paints the new head fully where it faces the camera this much (the cosine),
                            # the build's paint below the first
OCCLUDE_SHARE = 0.02    # the picture shows a point unless the kept figure stands this far in front of it (a brim)
BUILD_PAINT_SHARE = 0.05    # the build's paint is read within this share of the head's width of its surface
PICTURE_WIDTH = 2048    # the new head's picture is this many texels round
SKIN_TURN = 0.55        # the build's skin is believed this share of a half turn either side of the nose (face, ears)
OFFSET_SHARE = 0.05     # the build's paint is moved to the picture's by offsets spread this share of the way round
OFFSET_NEAR_SHARE = 0.02    # the figure's tone is read where its old head lay this close to the new one
BLEND_ROWS = 6          # the head's picture reaches this many rows below the stitch
NECK_EASE_ROWS = 1      # the new head eases onto the figure's neck at the stitch over this many rows
NECK_STEP_SHARE = 0.08  # the new head's neck runs up from the stitch until it stands out this share of the head's
                        # width (a chin, a jaw) ...
NECK_BAND_SHARE = 0.25  # ... at most this share of it over the stitch (the Zauberer's jaw beside his chin is 0.15 to
                        # 0.23 of it over the stitch); the figure's own paint covers it
NECK_ROUND_SHARE = 0.03 # the slope under the chin and the jaw is rounded over this share of the way round ...
EDGE_ROUND_SHARE = 0.02 # ... and the chin's and the jaw's lower edge over this share of the head's width
NECK_COLLAR_LEAST = 0.5 # a collar round at least this share of the head hides the neck's lower part: the new neck
                        # reaches down into it (`head_zone` "low") ...
NECK_LOW_SHARE = 0.3    # ... this share of the head's width under the stitch (the Zauberer's old neck showed down to
                        # 0.24 of it through his collar's open front, painted with his old chin's shadow) ...
NECK_COVER_SHARE = 0.01 # ... at least this share of the head's width round the old neck there (`neck_cover`) ...
NECK_COVER_REACH = 0.08 # ... where that lies within this share of it of the new neck (a collar's inside or a chest
                        # further out stays in front of it) ...
NECK_DIVE_SHARE = 0.05  # ... and its lowest this share of the head's width under the old surface, so it ends inside
                        # the body (at the bottom of the Zauberer's collar front a piece of the build's collar stood 5 mm
                        # out of his shirt there, 2026-11-23)
STRAND_GAP_SHARE = 0.04     # a narrow piece of the build standing off its own skin by more than this share of the
STRAND_WIDE_SHARE = 0.05    # head's width (a strand over the forehead: no wider than this share of it either side)
STRAND_RELIEF_SHARE = 0.02  # is laid onto the skin, standing at most this share of the head's width off it (`press_strands`)
SKIN_DETAIL_SHARE = 0.05    # the skin the picture does not see: the face's colour, shaded by the build's lightness
SKIN_DETAIL_RANGE = (0.6, 1.2)  # against its mean this share of the head's width round, within these bounds ...
SKIN_PLAIN = 0.3            # ... the mean taken over its plain skin, within this share of its usual lightness
FLAP_TURN = (0.3, 0.75)     # on the head's sides (these shares of a half turn from the nose) a piece standing out of
FLAP_WIDE_SHARE = 0.15      # it, no wider than this share of the head's width either side (an ear) ...
FLAP_LEAST_SHARE = 0.04     # ... and more than this share of it out, stands at most FLAP_STAND_SHARE of it out ...
FLAP_STAND_SHARE = 0.12     # ... its rim rounded over FLAP_ROUND_SHARE of it (`round_flaps`)
FLAP_ROUND_SHARE = 0.04
FLAP_BRIM_SHARE = 0.1       # ... no nearer the brim's cut than this share of the head's width (eased in over half of it)
SLIVER_PIXELS = 2       # a piece the cut leaves wholly within this many pixels of the region is a bit of the old head
SPIKE_ROWS = 3          # fewer rows than this of the build standing out more than ...
SPIKE_SHARE = 0.1       # ... this share of the head's width from the rows round them are a plate the rays only grazed
ROUND_SHARE = hf.ROUND_SHARE      # creases rounded over this share of the head's width, as the face's ...
ROUND_JUMP_SHARE = 0.1  # ... but not across a step of this share of it (an ear's rim, the nose's and chin's underside)


# --- The axis -------------------------------------------------------------------------------------------------------

def line_axis(low: tuple[float, float, float], high: tuple[float, float, float]) -> dict:
    """An axis leaning from LOW to HIGH (x, y, z): its middle at height y is `centre(axis, y)`."""
    (x0, y0, z0), (x1, y1, z1) = low, high
    span = y1 - y0 if abs(y1 - y0) > 1e-12 else 1e-12
    return {"x": float(x0), "z": float(z0), "y": float(y0), "kx": float((x1 - x0) / span),
            "kz": float((z1 - z0) / span)}


def centre(axis: dict, heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, z) of the axis's middle at HEIGHTS."""
    h = np.asarray(heights, dtype=np.float64) - axis["y"]
    return axis["x"] + axis["kx"] * h, axis["z"] + axis["kz"] * h


def skin_loop(points: np.ndarray, faces: np.ndarray, height: float, inside: tuple[float, float],
              least: float = 0.0) -> np.ndarray | None:
    """The smallest outline at HEIGHT round the point INSIDE (x, z) whose inside is solid and holds more than LEAST, of a
    welded mesh: a head's or a neck's skin. A hollow build's inner wall turns the other way seen from above and is
    passed over; a collar round the neck is larger; a skull's lowest point grazed by the cut holds nothing. None when no
    solid outline holds the point."""
    up = np.array([0.0, 1.0, 0.0])
    point = np.array([inside[0], height, inside[1]])
    solid = [loop for loop in hh.section_loops(points, faces, np.array([0.0, height, 0.0]), up)
             if hh.loop_area(loop, up, signed=True) < -least and hh.contains(loop, up, point)]
    return min(solid, key=lambda loop: hh.loop_area(loop, up)) if solid else None


def loop_middle(loop: np.ndarray) -> tuple[float, float]:
    """(x, z) of the middle of an outline's area seen from above."""
    x, z = loop[:, 0], loop[:, 2]
    cross = x * np.roll(z, -1) - np.roll(x, -1) * z
    area = cross.sum() / 2
    if abs(area) < 1e-18:
        return float(x.mean()), float(z.mean())
    return (float(((x + np.roll(x, -1)) * cross).sum() / (6 * area)),
            float(((z + np.roll(z, -1)) * cross).sum() / (6 * area)))


def neck_and_axis(vertices: np.ndarray, faces: np.ndarray, start: tuple[float, float], skull: float, chin: float,
                  width: float, pixel: float) -> dict:
    """Where the head's axis runs and how far down the neck stands free of the collar.

    The skull's middle is its skin's middle at height SKULL (between nose and brim), found round START (x, z). Going
    down from the CHIN a pixel at a time, the neck's skin is the smallest solid outline round the axis; where a collar
    fuses with the neck (the Zauberer's, 0.005 under his chin), that outline turns into the collar's and is no longer a
    neck: the last one before is the neck's lowest free section, where the new head is stitched on. The axis leans from
    the neck's middle there to the skull's (his neck sits 0.013 behind his skull's middle: a straight axis through the
    skull ran outside his neck). Returns {"axis", "skull_area", "neck" (its lowest free height, or None), "necks"
    (height, area) for the record}."""
    points, welded, _ = hh.weld(vertices, faces)
    up = np.array([0.0, 1.0, 0.0])
    loop = skin_loop(points, welded, skull, start)
    if loop is None:
        raise ValueError("found no skin round the head's middle: is the model facing +Z?")
    top = loop_middle(loop)
    skull_area = hh.loop_area(loop, up)
    axis = line_axis((top[0], skull, top[1]), (top[0], skull + 1.0, top[1]))
    necks, lowest = [], None
    for _ in range(AXIS_ROUNDS):
        necks, lowest, last = [], None, None
        for height in np.arange(chin, chin - NECK_DEPTH_SHARE * width, -pixel):
            x, z = centre(axis, height)
            ring = skin_loop(points, welded, float(height), (float(x), float(z)), NECK_LEAST * skull_area)
            area = hh.loop_area(ring, up) if ring is not None else np.inf
            if ring is None or area > NECK_AREA_LIMIT * skull_area or (last is not None and area > NECK_GROWTH * last):
                break
            necks.append((float(height), float(area)))
            lowest, last = (float(height), loop_middle(ring)), area
        if lowest is None:
            break
        axis = line_axis((lowest[1][0], lowest[0], lowest[1][1]), (top[0], skull, top[1]))
    return {"axis": axis, "skull_area": float(skull_area), "neck": None if lowest is None else lowest[0],
            "necks": necks}


# --- The head unrolled ----------------------------------------------------------------------------------------------

def cylinder(axis: dict, y_lo: float, y_hi: float, pixel: float, radius: float) -> dict:
    """The picture of a head unrolled round AXIS (`line_axis`): columns are angles from -pi (the back) through 0 (the
    front, +Z), rows are heights from Y_HI down (row 0 at the top), pixels PIXEL square at RADIUS from the axis."""
    cols = max(MIN_COLUMNS, int(round(2 * np.pi * radius / pixel)))
    rows = max(1, int(np.ceil((y_hi - y_lo) / pixel)))
    return {"axis": dict(axis), "y_hi": float(y_hi), "y_lo": float(y_hi - rows * pixel), "pixel": float(pixel),
            "rows": rows, "cols": cols, "turn": 2 * np.pi / cols, "radius": float(radius)}


def polar(frame: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(angle, distance from the axis) of POINTS: angle 0 is the front (+Z), +pi/2 the figure's left (+X)."""
    p = np.asarray(points, dtype=np.float64)
    x, z = centre(frame["axis"], p[:, 1])
    return np.arctan2(p[:, 0] - x, p[:, 2] - z), np.hypot(p[:, 0] - x, p[:, 2] - z)


def unpolar(frame: dict, angle: np.ndarray, radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Points at ANGLE, RADIUS from the axis and HEIGHT (the inverse of `polar`)."""
    angle, radius, height = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (angle, radius, height)))
    x, z = centre(frame["axis"], height)
    return np.stack([x + radius * np.sin(angle), height, z + radius * np.cos(angle)], axis=-1)


def column_angles(frame: dict) -> np.ndarray:
    """The angle at the middle of every column."""
    return -np.pi + (np.arange(frame["cols"]) + 0.5) * frame["turn"]


def row_heights(frame: dict) -> np.ndarray:
    """The height at the middle of every row."""
    return frame["y_hi"] - (np.arange(frame["rows"]) + 0.5) * frame["pixel"]


def cylinder_hits(vertices: np.ndarray, faces: np.ndarray, frame: dict) -> dict:
    """Every crossing of a horizontal ray from the axis with the model, one per pixel and triangle, every layer:
    {"pixel" (row-major id), "radius" (distance from the axis), "facing" (the triangle's normal along the ray: above 0
    it faces away from the axis), "up" (its normal's height part: a brim's underside is near -1), "face"}. Triangles
    round the axis itself (inside the head's hollow, or a hat's top) have no angle and are left out."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if not len(f):
        return {"pixel": np.zeros(0, np.int64), "radius": np.zeros(0), "facing": np.zeros(0), "up": np.zeros(0),
                "face": np.zeros(0, np.int64)}
    angle, radius = polar(frame, v)
    pixel, turn = frame["pixel"], frame["turn"]
    t = angle[f]
    wrap = (t.max(axis=1) - t.min(axis=1)) > np.pi
    t = np.where(wrap[:, None] & (t < 0), t + 2 * np.pi, t)        # a triangle across the back seam, unbroken
    near = (radius[f].min(axis=1) > 2 * pixel) & ((t.max(axis=1) - t.min(axis=1)) < np.pi / 2)
    heights = v[f][:, :, 1]
    keep = near & (heights.max(axis=1) >= frame["y_lo"]) & (heights.min(axis=1) <= frame["y_hi"])
    t, heights, r, index = t[keep], heights[keep], radius[f][keep], np.flatnonzero(keep)
    u = (t + np.pi) / turn * pixel                 # one column is PIXEL wide in this picture
    points = np.stack([u.reshape(-1), heights.reshape(-1), r.reshape(-1)], axis=1)
    triangles = np.arange(len(points)).reshape(-1, 3)
    box = (0.0, 2 * frame["cols"] * pixel, frame["y_lo"], frame["y_hi"])
    pixel_id, depth, which = hf.pixel_hits(points, triangles, box, pixel)
    _, wide = hf.canvas(box, pixel)
    row, col = np.divmod(pixel_id, wide)
    ok = (row >= 0) & (row < frame["rows"])
    row, col, depth, which = row[ok], col[ok] % frame["cols"], depth[ok], index[which[ok]]
    normals = hf.face_normals(v, f[which])
    middle = v[f[which]].mean(axis=1)
    x, z = centre(frame["axis"], middle[:, 1])
    out = np.stack([middle[:, 0] - x, np.zeros(len(middle)), middle[:, 2] - z], axis=1)
    facing = (normals * out).sum(axis=1) / np.maximum(np.linalg.norm(out, axis=1), 1e-12)
    return {"pixel": row * frame["cols"] + col, "radius": depth, "facing": facing, "up": normals[:, 1], "face": which}


def wrap_grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    """MASK grown by PIXELS (a square), its first and last columns neighbours, and the holes it closes filled."""
    if not mask.any():
        return mask.copy()
    cols = mask.shape[1]
    pad = cols // 2
    padded = np.concatenate([mask[:, cols - pad:], mask, mask[:, :pad]], axis=1)
    if pixels > 0:
        padded = ndimage.binary_dilation(padded, structure=np.ones((3, 3), bool), iterations=pixels)
    return ndimage.binary_fill_holes(padded)[:, pad:pad + cols]


def collar_mask(outer: np.ndarray, bottom: int, jump: float, stand: float, grow: int = COLLAR_GROW) -> np.ndarray:
    """Where a collar is the outermost surface round the head (OUTER, the outermost surface's distance from the axis).

    In every column the outermost surface is followed up from row BOTTOM (the neck's lowest free section; a collar
    goes on down into the body) for as long as it runs on without a jump of JUMP from one row to the next. Where it ends
    in a step in toward the axis of STAND or more, the run was a collar: its rim stands in front of the head above it.
    Where it ends in a step out (a chin over the throat between a collar's open fronts) or not at all, it was the head.
    Grown by GROW pixels and its holes filled, so the collar's own rim and edges are in it."""
    rows, cols = outer.shape
    start = min(max(bottom - 1, 1), rows - 1)
    with np.errstate(invalid="ignore"):
        runs_on = np.isfinite(outer[:-1]) & np.isfinite(outer[1:]) & (np.abs(outer[:-1] - outer[1:]) < jump)
    breaks = ~runs_on[:start]                                    # breaks[r]: row r and r + 1 apart
    seen = np.isfinite(outer[start])
    last = np.where(breaks.any(axis=0), start - 1 - np.argmax(breaks[::-1], axis=0), -1)   # the break nearest above
    top = last + 1                                               # the run's top row
    with np.errstate(invalid="ignore"):
        step = np.where(last >= 0, outer[np.maximum(last, 0), np.arange(cols)] - outer[top, np.arange(cols)], np.inf)
    collar_column = seen & np.isfinite(step) & (step <= -stand)
    mask = (np.arange(rows)[:, None] >= top[None, :]) & collar_column[None, :]
    return wrap_grow(mask, grow)


def flat_walls(vertices: np.ndarray, faces: np.ndarray, frame: dict, step: float, cells: int) -> tuple:
    """(column, row, cell) of every place where a vertical line in a column's section (height by distance from the
    axis, STEP a cell) meets the model: the surfaces a horizontal ray only grazes (a collar's rim, a cap) are walls
    there too."""
    corners = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    corners = corners[(corners[:, :, 1].max(axis=1) >= frame["y_lo"]) & (corners[:, :, 1].min(axis=1) <= frame["y_hi"])]
    if not len(corners):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    pixel, turn = frame["pixel"], frame["turn"]
    for _ in range(SPLIT_ROUNDS):         # a cap's fan reaches far round the axis: split it until it does not
        angle, _ = polar(frame, corners.reshape(-1, 3))
        t = angle.reshape(-1, 3)
        span = np.ptp(np.where((np.ptp(t, axis=1) > np.pi)[:, None] & (t < 0), t + 2 * np.pi, t), axis=1)
        wide = span > np.pi / 4
        if not wide.any():
            break
        a, b, c = corners[wide, 0], corners[wide, 1], corners[wide, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        split = np.concatenate([np.stack(q, axis=1) for q in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
        corners = np.concatenate([corners[~wide], split])
    angle, radius = polar(frame, corners.reshape(-1, 3))
    t, radius = angle.reshape(-1, 3), radius.reshape(-1, 3)
    wrap = np.ptp(t, axis=1) > np.pi
    t = np.where(wrap[:, None] & (t < 0), t + 2 * np.pi, t)
    keep = (np.ptp(t, axis=1) < np.pi / 2) & (radius.min(axis=1) < cells * step)
    t, r, y = t[keep], radius[keep], corners[keep][:, :, 1]
    points = np.stack([((t + np.pi) / turn * pixel).reshape(-1), (r / step * pixel).reshape(-1), y.reshape(-1)],
                      axis=1)
    box = (0.0, 2 * frame["cols"] * pixel, 0.0, cells * pixel)
    pixel_id, height, _ = hf.pixel_hits(points, np.arange(len(points)).reshape(-1, 3), box, pixel)
    _, wide = hf.canvas(box, pixel)
    upside, col = np.divmod(pixel_id, wide)
    row = np.floor((frame["y_hi"] - height) / pixel).astype(np.int64)
    ok = (row >= 0) & (row < frame["rows"]) & (upside < cells)
    return col[ok] % frame["cols"], row[ok], cells - 1 - upside[ok]


def outermost_inside(vertices: np.ndarray, faces: np.ndarray, hits: dict, frame: dict, reach: float,
                     step: float) -> np.ndarray:
    """In every pixel, how far in from the axis the outermost solid ends: the distance where, going in from the
    outermost surface, the ray first comes out into air open to the outside (-inf where the outermost solid reaches the
    axis, as a head does; NaN where there is nothing).

    Pixal3D builds every solid as a thin hollow shell, so along one ray a collar is two thin walls with its hollow
    between them, just as a collar and a neck are two thin walls with air between them: the Zauberer's collar has a
    hollow 0.01 wide, the air between it and his neck is 0.012 at the stitch. Only the air between collar and neck is
    open to the outside, upward over the collar's rim; so the air is filled from the outside in each column's section
    (height by distance from the axis, STEP a cell; walls from the model's crossings with HITS' horizontal rays and with
    vertical lines, `flat_walls`) and what it does not reach is a solid's hollow."""
    rows, cols = frame["rows"], frame["cols"]
    cells = int(np.ceil(reach / step)) + 2
    pixel_id, radius, facing = hits["pixel"], hits["radius"], hits["facing"]
    keep = radius < (cells - 1) * step
    pixel_id, radius, facing = pixel_id[keep], radius[keep], facing[keep]
    order = np.lexsort((-radius, pixel_id))                     # by pixel, from the outside in
    pid, r, f = pixel_id[order], radius[order], facing[order]
    nxt_same = np.r_[pid[1:] == pid[:-1], False]
    r_next = np.r_[r[1:], 0.0]
    # thin walls: from a surface facing out to the next one in facing in (a glancing one is a wall of its own)
    wall = np.zeros((cols, rows, cells), np.int16)
    start = np.where(f > -FACING_LIMIT, r, np.nan)
    stop = np.where((f > FACING_LIMIT) & nxt_same, r_next, r)
    ok = np.isfinite(start)
    hi = np.minimum((start[ok] / step).astype(np.int64) + 1, cells - 1)
    lo = np.maximum((stop[ok] / step).astype(np.int64), 0)
    row, col = np.divmod(pid[ok], cols)
    np.add.at(wall, (col, row, lo), 1)
    np.add.at(wall, (col, row, hi), -1)
    solid = np.cumsum(wall, axis=2) > 0
    del wall
    solid[flat_walls(vertices, faces, frame, step, cells)] = True
    solid[:, :, :AXIS_CELLS] = True       # round the axis the head is solid: a cap's fan there has no angle to raster
    outer = np.full(rows * cols, -np.inf)
    np.maximum.at(outer, pixel_id, radius)
    outer = outer.reshape(rows, cols)
    k = np.arange(cells)[None, None, :]
    beyond = (k * step > outer.T[:, :, None]) | ~np.isfinite(outer.T)[:, :, None]   # outside the outermost surface
    air = ~solid
    structure = np.zeros((3, 3, 3), bool)
    structure[1] = ndimage.generate_binary_structure(2, 1)        # within a column only
    label, _ = ndimage.label(air, structure)
    open_air = np.isin(label, np.unique(label[beyond & air]))
    open_air[label == 0] = False
    below = k * step < outer.T[:, :, None]
    first = np.where(open_air & below, k, -1).max(axis=2)        # the outermost open air under the outer surface
    inside = np.where(first >= 0, (first + 1) * step, -np.inf).T
    return np.where(np.isfinite(outer), inside, np.nan)


def fill_nearest(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """VALUES with every pixel not KNOWN given the value of the nearest known one (first and last columns
    neighbours); VALUES as they are when nothing is known."""
    if known.all() or not known.any():
        return values.copy()
    cols = values.shape[1]
    pad = cols // 2
    padded = np.concatenate([values[:, -pad:], values, values[:, :pad]], axis=1)
    known_padded = np.concatenate([known[:, -pad:], known, known[:, :pad]], axis=1)
    _, (r, c) = ndimage.distance_transform_edt(~known_padded, return_indices=True)
    return padded[r, c][:, pad:pad + cols]


# --- The brim ---------------------------------------------------------------------------------------------------------

def brim_plane(vertices: np.ndarray, faces: np.ndarray, middle_xz: tuple[float, float], start: float, width: float,
               pixel: float, ring: tuple[float, float] = BRIM_RING, keep: float = BRIM_KEEP) -> dict | None:
    """The brim's underside near the head as a plane y = a + b (x - ax) + c (z - az): read from below (the lowest
    surface above height START, the nose, where a brim lies over the face) in a RING round MIDDLE_XZ (ax, az; shares of
    the head's WIDTH), fitted on the KEEP share of points that fit best, three times. None when less than BRIM_LEAST of
    the ring holds a surface (no brim, or one too small to set a head under). The Zauberer's brim tilts, 0.243 at the
    back to 0.27 at his left front, so one height for it would cut into the brim on one side and leave old hair under it
    on the other. "below" is how far the brim dips under the plane (all but its lowest BRIM_DIP of points)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    ax, az = middle_xz
    outer = ring[1] * width
    middle = v[f].mean(axis=1)
    above = (v[f][:, :, 1].min(axis=1) > start) & (np.hypot(middle[:, 0] - ax, middle[:, 2] - az) < 1.3 * outer)
    if not above.any():
        return None
    seen = np.stack([v[:, 0], -v[:, 2], -v[:, 1]], axis=1)       # from below: picture up is -z, nearest is lowest
    box = (ax - outer - 2 * pixel, ax + outer + 2 * pixel, -az - outer - 2 * pixel, -az + outer + 2 * pixel)
    depth, _ = hf.depth_map(seen, f[above], box, pixel)
    rows, cols = depth.shape
    x = box[0] + (np.arange(cols) + 0.5) * pixel
    z = -(box[3] - (np.arange(rows) + 0.5) * pixel)
    gx, gz = np.meshgrid(x, z)
    distance = np.hypot(gx - ax, gz - az)
    in_ring = (distance >= ring[0] * width) & (distance <= outer)
    found = in_ring & np.isfinite(depth)
    if found.sum() < BRIM_LEAST * in_ring.sum():
        return None
    px, pz, py = gx[found] - ax, gz[found] - az, -depth[found]
    design = np.stack([np.ones(len(px)), px, pz], axis=1)
    use = np.ones(len(px), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(design[use], py[use], rcond=None)[0]
        miss = py - design @ coef
        use = np.abs(miss) <= np.quantile(np.abs(miss), keep)
    miss = py - design @ coef
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]), "axis": (float(ax), float(az)),
            "below": float(max(0.0, -np.quantile(miss[use], BRIM_DIP))), "spread": float(np.std(miss[use])),
            "share": float(found.sum() / max(in_ring.sum(), 1))}


def plane_height(plane: dict, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The plane's height over the points (X, Z)."""
    ax, az = plane["axis"]
    return plane["a"] + plane["b"] * (np.asarray(x, dtype=np.float64) - ax) + plane["c"] * (
        np.asarray(z, dtype=np.float64) - az)


def face_neighbours(faces: np.ndarray, group: np.ndarray, among: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of faces (indices into AMONG, face ids) that share an edge, the corners welded by GROUP."""
    g = group[np.asarray(faces, dtype=np.int64)[among]]
    edges = np.sort(np.concatenate([g[:, [0, 1]], g[:, [1, 2]], g[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(among)), 3)
    key = edges[:, 0] * (int(group.max()) + 1) + edges[:, 1]
    order = np.argsort(key, kind="stable")
    key, owner = key[order], owner[order]
    same = key[1:] == key[:-1]
    return owner[:-1][same], owner[1:][same]


def brim_faces(vertices: np.ndarray, faces: np.ndarray, plane: dict, start: float, width: float, pixel: float,
               ring: tuple[float, float] = BRIM_RING) -> np.ndarray:
    """Which faces make the brim's underside: those `brim_plane` reads from below in its RING round the plane's axis
    (over height START, within BRIM_SEED_SHARE of the head's WIDTH of the plane), and every face joined to them across
    an edge that faces down too (its normal BRIM_TURN or more downward) and not out from the axis: followed inward to
    where the brim turns into the head (a hair's flare under a brim faces down and out) and out to its rim. The
    Zauberer's brim droops toward its rim, so near his head its underside lies lower than where his hair meets it: the
    region cut up to there must stay inside it (2026-11-23)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    out = np.zeros(len(f), bool)
    if not len(f):
        return out
    ax, az = plane["axis"]
    outer = ring[1] * width
    corners = v[f]
    middle = corners.mean(axis=1)
    normals = hf.face_normals(v, f)
    away = np.stack([middle[:, 0] - ax, np.zeros(len(f)), middle[:, 2] - az], axis=1)
    facing = (normals * away).sum(axis=1) / np.maximum(np.linalg.norm(away, axis=1), 1e-12)
    high = corners[:, :, 1].min(axis=1) > start
    down = high & (normals[:, 1] < -BRIM_TURN) & (facing <= FACING_LIMIT)
    index = np.flatnonzero(high & (np.hypot(away[:, 0], away[:, 2]) < 1.3 * outer))
    if not down.any() or not len(index):
        return out
    seen = np.stack([v[:, 0], -v[:, 2], -v[:, 1]], axis=1)        # from below, as `brim_plane` reads the brim
    box = (ax - outer - 2 * pixel, ax + outer + 2 * pixel, -az - outer - 2 * pixel, -az + outer + 2 * pixel)
    depth, owner = hf.depth_map(seen, f[index], box, pixel)
    rows, cols = depth.shape
    gx, gz = np.meshgrid(box[0] + (np.arange(cols) + 0.5) * pixel, -(box[3] - (np.arange(rows) + 0.5) * pixel))
    distance = np.hypot(gx - ax, gz - az)
    with np.errstate(invalid="ignore"):
        seed = ((owner >= 0) & (distance >= ring[0] * width) & (distance <= outer)
                & (np.abs(-depth - plane_height(plane, gx, gz)) < BRIM_SEED_SHARE * width))
    seeds = np.unique(index[owner[seed]])
    seeds = seeds[down[seeds]]
    if not len(seeds):
        return out
    among = np.flatnonzero(down)
    a, b = face_neighbours(f, hh.weld_groups(v), among)
    _, label = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(among), len(among))),
                                    directed=False)
    out[among[np.isin(label, label[np.searchsorted(among, seeds)])]] = True
    return out


def brim_ceiling(vertices: np.ndarray, faces: np.ndarray, brim: np.ndarray, axis: dict, height: float, reach: float,
                 pixel: float) -> dict:
    """The brim's underside (BRIM, `brim_faces`) seen from straight below round AXIS (its middle at HEIGHT, the brim's)
    out to REACH: {"height" (its lowest surface's height in every pixel, inf where there is none), "box" (x0, x1,
    -z1, -z0: the picture's x is the model's, its y the model's -z), "pixel"}. Rays from the axis graze a brim that is
    nearly flat, so they bound it poorly; from below it is plain."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)[np.asarray(brim, bool)]
    cx, cz = centre(axis, np.array([height]))
    x0, z0 = float(cx[0]), float(cz[0])
    half = reach + 2 * pixel
    box = (x0 - half, x0 + half, -z0 - half, -z0 + half)
    if not len(f):
        return {"height": np.full(hf.canvas(box, pixel), np.inf), "box": box, "pixel": float(pixel)}
    depth, _ = hf.depth_map(np.stack([v[:, 0], -v[:, 2], -v[:, 1]], axis=1), f, box, pixel)
    return {"height": -depth, "box": box, "pixel": float(pixel)}


def hat_opening(ceiling: dict, axis: dict, height: float, angles: np.ndarray,
                reach: float) -> tuple[np.ndarray, np.ndarray]:
    """(height, distance from the axis) of the hat's opening at every one of ANGLES round AXIS: where the brim's
    underside (CEILING, `brim_ceiling`) ends nearest the axis, the hair under it meeting the hat. Read along each angle
    out from the axis (its middle at HEIGHT, the brim's) to REACH, and taken at its highest and furthest out over
    OPENING_SPREAD of the way round, softened: the old head is cut out up to there, so the cut runs through the brim's
    smooth underside and the hat's inside, not along the seam where the hair meets the brim, as jagged as the build's
    mesh there (cut along it, a torn fringe stood round the Zauberer's new head, 2026-11-23). NaN everywhere when the
    brim is found round less than OPENING_LEAST of the head. On the Zauberer the opening lies 0.5 mm under the brim's
    plane at his back and 8 mm over it at his ears, where the hat rises."""
    nothing = np.full(len(angles), np.nan), np.full(len(angles), np.nan)
    pixel, box, lift = ceiling["pixel"], ceiling["box"], ceiling["height"]
    cx, cz = centre(axis, np.array([height]))
    x0, z0 = float(cx[0]), float(cz[0])
    steps = (np.arange(int(reach / pixel)) + 1.0) * pixel
    a = np.asarray(angles, dtype=np.float64)[:, None]
    col = np.floor((x0 + steps[None, :] * np.sin(a) - box[0]) / pixel).astype(np.int64)
    row = np.floor((box[3] + z0 + steps[None, :] * np.cos(a)) / pixel).astype(np.int64)
    inside = (row >= 0) & (row < lift.shape[0]) & (col >= 0) & (col < lift.shape[1])
    lowest = np.where(inside, lift[np.clip(row, 0, lift.shape[0] - 1), np.clip(col, 0, lift.shape[1] - 1)], np.inf)
    found = np.isfinite(lowest)
    first = found.argmax(axis=1)
    has = found.any(axis=1)
    if has.mean() < OPENING_LEAST:
        return nothing
    pick = np.arange(len(a))
    heights = fill_nearest(np.where(has, lowest[pick, first], 0.0)[None, :], has[None, :])[0]
    radii = fill_nearest(np.where(has, steps[first], 0.0)[None, :], has[None, :])[0]
    size = max(3, int(OPENING_SPREAD * len(a)) | 1)
    heights = ndimage.gaussian_filter1d(ndimage.maximum_filter1d(heights, size, mode="wrap"), size / 6, mode="wrap")
    radii = ndimage.gaussian_filter1d(ndimage.maximum_filter1d(radii, size, mode="wrap"), size / 6, mode="wrap")
    return heights, radii


def sample_columns(values: np.ndarray, frame: dict, angle: np.ndarray) -> np.ndarray:
    """VALUES (one per column of FRAME) at ANGLE, between column middles in a straight line, round the back."""
    values = np.asarray(values, dtype=np.float64)
    at = (np.asarray(angle, dtype=np.float64) + np.pi) / frame["turn"] - 0.5
    low = np.floor(at).astype(np.int64)
    t = at - low
    return (1 - t) * values[low % len(values)] + t * values[(low + 1) % len(values)]


# --- The region to replace ------------------------------------------------------------------------------------------

def nose_point(window: dict) -> tuple[float, float, float]:
    """(x, y, z) of the nose the face's window was measured from."""
    box, pixel = window["box"], window["pixel"]
    row, col = window["seed"]
    return (box[0] + (col + 0.5) * pixel, box[3] - (row + 0.5) * pixel, float(window["depth"][row, col]))


def pixel_of(frame: dict, angle: np.ndarray, height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, column) of the pixel holding each (ANGLE, HEIGHT), rows kept in the picture."""
    col = np.floor((np.asarray(angle) + np.pi) / frame["turn"]).astype(np.int64) % frame["cols"]
    row = np.clip(np.floor((frame["y_hi"] - np.asarray(height)) / frame["pixel"]).astype(np.int64), 0,
                  frame["rows"] - 1)
    return row, col


def hit_points(frame: dict, hits: dict) -> np.ndarray:
    """The crossings of `cylinder_hits` as points (at their pixel's middle angle and height)."""
    row, col = np.divmod(hits["pixel"], frame["cols"])
    return unpolar(frame, column_angles(frame)[col], hits["radius"], row_heights(frame)[row])


def slab_values(points: np.ndarray, axis: dict, plane: dict, cut: float, bottom: float, reach: float,
                top: np.ndarray | None = None) -> np.ndarray:
    """How far inside the slab each point lies (positive inside): under the brim's PLANE less CUT (or under TOP, one
    height a point, where that is higher), over BOTTOM (the stitch) and nearer the AXIS than REACH (a number, or a
    picture of the frame's pixels)."""
    p = np.asarray(points, dtype=np.float64)
    x, z = centre(axis, p[:, 1])
    level = plane_height(plane, p[:, 0], p[:, 2]) - cut
    if top is not None:
        level = np.maximum(level, top)
    return np.minimum(np.minimum(level - p[:, 1], p[:, 1] - bottom), reach - np.hypot(p[:, 0] - x, p[:, 2] - z))


def collar_limit(vertices: np.ndarray, faces: np.ndarray, hits: dict, frame: dict, bottom: float, width: float,
                 reach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(how far out from the axis the head reaches in every pixel, the collar's mask, the outermost surface): REACH,
    but inside a collar (`collar_mask`) its inside (`outermost_inside`) less COLLAR_MARGIN_SHARE of the head's
    WIDTH; collar pixels whose inside is not found take the nearest found one."""
    pixel = frame["pixel"]
    outer = np.full(frame["rows"] * frame["cols"], -np.inf)
    np.maximum.at(outer, hits["pixel"], hits["radius"])
    outer = outer.reshape(frame["rows"], frame["cols"])
    above = int(np.searchsorted(-row_heights(frame), -bottom))       # rows from the top down to the stitch
    core = collar_mask(outer, above, COLLAR_JUMP_SHARE * width, STAND_SHARE * width, grow=0)
    collar = wrap_grow(core, COLLAR_GROW)
    inside = outermost_inside(vertices, faces, hits, frame, reach, pixel)
    skin = np.full(frame["rows"] * frame["cols"], np.inf)        # the head's own skin: the first surface leaving it
    out = hits["facing"] > FACING_LIMIT
    np.minimum.at(skin, hits["pixel"][out], hits["radius"][out])
    # where the collar itself is the outermost surface, and its inside lies beyond the head's skin (else the fill
    # leaked through a gap in the walls)
    known = core & np.isfinite(inside) & (inside > skin.reshape(inside.shape) + GAP_SHARE * width)
    limit = np.full(collar.shape, reach)
    if known.any():
        filled = fill_nearest(np.where(known, inside, reach), known)
        # the nearest few pixels' least: at a collar's slanted rim a ray can meet the rim's top before the inside wall
        filled = ndimage.minimum_filter(filled, size=2 * COLLAR_GROW + 1, mode=("nearest", "wrap"))
        limit[collar] = np.minimum(filled[collar] - COLLAR_MARGIN_SHARE * width, reach)
    return limit, collar, outer


def head_zone(vertices: np.ndarray, faces: np.ndarray, window: dict) -> dict:
    """The region of a figure (its mesh in the frame of `hero_face.face_window`'s WINDOW) that the new head replaces, and
    how the new head must lie in it.

    The region (`zone_values`) lies under the brim's underside (`brim_plane`, cut a little below its lowest dip) and,
    where the hat's opening is found all round (`brim_faces`, `hat_opening`), up to just over it and out to its rim
    (and to any old hair standing out under the brim beyond it), over the neck's lowest section free of the collar
    (`neck_and_axis`), and inside the collar wherever one stands round the head (`collar_limit`); elsewhere within
    REACH_SHARE of the head's width from the axis. Hat and collar stay whole, but for the brim's innermost ring.
    Returns the "frame" (the head unrolled round its leaning "axis", `cylinder`), "limit" (in every pixel, how far out
    the region and the new head reach; "side_limit" without the opening's rim), "collar", "outer" (the old outermost
    surface), the brim's "plane" and "brim" faces, the heights ("cut" under the plane, "bottom"), per column the hat's
    "opening" height, its rim ("opening_radius") and the region's "top" over it (all None without an opening), "cover"
    (per column, how far out the old head reaches round the cut: without an opening the new head must be at least
    that wide there, so the old cut lies inside it) and "band" (from where under the cut that holds), the nose, the
    chin and the neck's record; "low", how far down the new neck reaches (into a collar round at least
    NECK_COLLAR_LEAST of the head, NECK_LOW_SHARE of the head's width under the stitch; else the stitch itself), "skin"
    (the old figure's first surface facing out in every pixel) and the face's "window". Raises ValueError when the
    figure has no brim or no neck under its chin."""
    width, pixel = window["width"], window["pixel"]
    if window.get("floor") is None:
        raise ValueError("no hat brim above the face: the head under a brim needs one (the face alone needs none)")
    nose = nose_point(window)
    floor = window["floor"]
    chin = window["box"][3] - (window["rows"][1] + 0.5) * pixel
    skull = nose[1] + SKULL_SHARE * (floor - nose[1])
    neck = neck_and_axis(vertices, faces, (window["centre_x"], nose[2] - AXIS_BACK * width), skull, chin, width, pixel)
    axis = neck["axis"]
    if neck["neck"] is None:
        raise ValueError("found no neck under the chin standing free of the collar: the new head has nowhere to be "
                         "stitched on")
    bottom = neck["neck"] + NECK_MARGIN * pixel
    reach = REACH_SHARE * width
    plane = brim_plane(vertices, faces, centre(axis, floor), nose[1], width, pixel)
    if plane is None:
        raise ValueError("found no brim round the head to set it under")
    cut = CUT_GAP_PIXELS * pixel + plane["below"]
    turn = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    rim = unpolar({"axis": axis}, turn, np.full(64, reach), np.full(64, floor))
    y_hi = float(plane_height(plane, rim[:, 0], rim[:, 2]).max()) + (TOP_SHARE + 0.1) * width
    # the hat's opening: where the hair meets the brim's underside, the old head is cut up to it (its hair flaring out
    # under the brim stood round the Zauberer's new head as a shelf, covered by a new head as wide: a mushroom from below)
    brim = brim_faces(vertices, faces, plane, nose[1], width, pixel)
    ceiling = brim_ceiling(vertices, faces, brim, axis, plane["a"], 1.2 * reach, pixel)
    angles = column_angles(cylinder(axis, bottom, y_hi, pixel, FRAME_SHARE * width))
    opening, opening_radius = hat_opening(ceiling, axis, plane["a"], angles, 1.2 * reach)
    if np.isfinite(opening).any():
        y_hi = max(y_hi, float(np.nanmax(opening)) + (TOP_SHARE + 0.05) * width)
    frame = cylinder(axis, bottom - BLEND_ROWS * pixel, y_hi, pixel, FRAME_SHARE * width)
    hits = cylinder_hits(vertices, faces, frame)
    side_limit, collar, outer = collar_limit(vertices, faces, hits, frame, bottom, width, reach)
    low = bottom
    if collar.any(axis=0).mean() >= NECK_COLLAR_LEAST:
        # a collar hides the neck's lower part: the new neck reaches down into it (`neck_cover`), so nothing of the old
        # neck shows through the collar's open front. The Zauberer's did, down to 0.24 of his head's width, painted
        # with his old chin's shadow: a dark zigzag and a pale chevron under the new chin (2026-11-23)
        low = bottom - NECK_LOW_SHARE * width
        shallow = side_limit
        frame = cylinder(axis, low - BLEND_ROWS * pixel, y_hi, pixel, FRAME_SHARE * width)
        hits = cylinder_hits(vertices, faces, frame)
        side_limit, collar, outer = collar_limit(vertices, faces, hits, frame, bottom, width, reach)
        # over the stitch the region stays as the frame down to the stitch measured it (its rows are the same): lower
        # down the Zauberer's air between collar and head opened to the outside, his collar's inside was measured 10
        # to 15 mm further out round his back over the stitch, the cut took it away and the fans closing that showed
        # as pink-white stripes beside his jaw (2026-11-23)
        over = row_heights(frame)[:len(shallow)] > bottom
        side_limit[:len(shallow)][over] = shallow[over]
    skin = np.full(frame["rows"] * frame["cols"], np.inf)      # the old figure's skin: the first surface facing out
    leaving = hits["facing"] > FACING_LIMIT
    np.minimum.at(skin, hits["pixel"][leaving], hits["radius"][leaving])
    limit = side_limit
    found = bool(np.isfinite(opening).all())
    if found:
        # over the old cut the region reaches out to the opening's rim, and to any of the old hair standing out under
        # the brim beyond it (on the Zauberer a bulge of it at his left temple): the brim beyond them stays whole
        side = unpolar(frame, column_angles(frame), np.full(frame["cols"], 0.6 * reach), np.full(frame["cols"], floor))
        level = plane_height(plane, side[:, 0], side[:, 2]) - cut
        row, col = np.divmod(hits["pixel"], frame["cols"])
        height = row_heights(frame)[row]
        hair = ((height > level[col]) & (height < opening[col] + OPENING_GAP_PIXELS * pixel) & ~brim[hits["face"]]
                & (hits["up"] < SIDE_LIMIT) & (hits["radius"] < side_limit[row, col]))    # not a brim's top
        stands = np.zeros(frame["cols"])
        np.maximum.at(stands, col[hair], hits["radius"][hair])
        size = max(3, int(OPENING_SPREAD * frame["cols"]) | 1)
        stands = ndimage.gaussian_filter1d(ndimage.maximum_filter1d(stands, size, mode="wrap"), size / 6, mode="wrap")
        opening_radius = np.maximum(opening_radius, stands) + OPENING_RIM_SHARE * width
        over = row_heights(frame)[:, None] > level[None, :] - 2 * pixel
        limit = np.where(over, np.minimum(side_limit, opening_radius[None, :]), side_limit)
    # how far out the old head reaches in a band under the brim: its sides do, not the brim's underside (a tilted brim
    # faces the rays a little too); above the cut it is the old head inside the hat, which stays and must be covered
    points = hit_points(frame, hits)
    radius = hits["radius"]
    row, col = np.divmod(hits["pixel"], frame["cols"])
    edge = unpolar(frame, column_angles(frame), np.full(frame["cols"], 0.6 * reach), np.full(frame["cols"], floor))
    band = plane_height(plane, edge[:, 0], edge[:, 2]) - cut - COVER_BAND_SHARE * width
    level = plane_height(plane, points[:, 0], points[:, 2]) - cut
    # only what stays of the old head counts, over the cut: its hair hanging lower goes with the cut, and taken in it
    # pushed the new head out a fifth of the head's width further than needed (a puffy roll under the brim)
    near = ((points[:, 1] > level - pixel) & (points[:, 1] < level + cut + TOP_SHARE * width)
            & (radius < limit[row, col]) & (np.abs(hits["up"]) < SIDE_LIMIT))
    cover = np.zeros(frame["cols"])
    np.maximum.at(cover, col[near], radius[near])
    cover = ndimage.maximum_filter1d(cover, 2 * COVER_SPREAD + 1, mode="wrap")
    if (cover > 0).any():
        cover[cover <= 0] = np.median(cover[cover > 0])
    # the new head is as wide as the widest of it round each column, softened (`tube_radius`): the old hair's spikes
    # stood out of a closer fit as fins, and so did the caps closing their cut
    cover = ndimage.gaussian_filter1d(ndimage.maximum_filter1d(cover, 2 * COVER_SMOOTH + 1, mode="wrap"),
                                      COVER_SOFTEN, mode="wrap")
    return {"frame": frame, "axis": axis, "limit": limit, "side_limit": side_limit, "collar": collar,
            "outer": np.where(collar, -np.inf, outer), "skin": skin.reshape(frame["rows"], frame["cols"]),
            "low": float(low), "window": window,
            "plane": plane, "band": band, "cut": float(cut), "bottom": float(bottom), "cover": cover,
            "width": float(width), "pixel": float(pixel), "nose": nose, "chin": float(chin), "neck": neck,
            "reach": float(reach), "brim": brim, "opening": opening if found else None,
            "opening_radius": opening_radius if found else None,
            "top": opening + OPENING_GAP_PIXELS * pixel if found else None}


def zone_values(zone: dict, points: np.ndarray) -> np.ndarray:
    """How far inside the region to replace each point lies (positive inside, negative outside): the least of its
    depth under the cut (or under the region's top over the hat's opening, zone "top", where that is higher), its
    height over the stitch and its distance inside the region's reach from the axis (zone "limit": inside a collar,
    its inside less a margin; over the old cut, the opening's rim)."""
    p = np.asarray(points, dtype=np.float64)
    frame = zone["frame"]
    angle, _ = polar(frame, p)
    row, col = pixel_of(frame, angle, p[:, 1])
    top = None if zone.get("top") is None else sample_columns(zone["top"], frame, angle)
    return slab_values(p, zone["axis"], zone["plane"], zone["cut"], zone["bottom"], zone["limit"][row, col], top)


# --- The new head -----------------------------------------------------------------------------------------------------

def build_radius(zone: dict, vertices: np.ndarray, faces: np.ndarray) -> dict:
    """The new head's radius in every pixel of the zone's picture, from the build placed in the figure's frame: the
    outermost surface of its head (hair, ears, the face; inside its own collar, its neck), never past the figure's
    collar (zone "limit"). The build's own brim is left out by its own brim plane (`brim_plane`; a face crop cuts the
    brim, and the Zauberer's build carries a piece of it lower than his hat's on one side), its own collar the way the
    figure's is (`collar_limit`). Pixels it leaves empty (under the build's brim, where its crop ended) take the nearest
    known one. A plate the rays only graze (fewer than SPIKE_ROWS rows standing out more than SPIKE_SHARE of the head's
    width: a piece of that brim under its plane, a strand's flat side) is left out, and so is any surface facing up or
    down (past SIDE_LIMIT) further out than the old head reached under the brim (zone "cover"): on the Zauberer that
    piece of brim reached out to the region's edge at the back, and carried up into the hat it stood out of the new
    head as a comb of thin plates (2026-11-22). Returns
    {"radius", "known", "plane", "skin" (which of its faces make that outermost surface: its paint is read there, not on
    the inner wall of its hollow shell just behind), "inner" (the build's own skin in every pixel, the first surface
    facing out: a strand standing off the forehead lies further out with air between, `press_strands`)}."""
    frame, width, pixel = zone["frame"], zone["width"], zone["pixel"]
    middle = centre(zone["axis"], zone["plane"]["a"])
    plane = (brim_plane(vertices, faces, middle, zone["nose"][1], width, pixel)
             or brim_plane(vertices, faces, middle, zone["nose"][1], width, pixel, BRIM_RING_NEAR) or zone["plane"])
    cut = CUT_GAP_PIXELS * pixel + plane["below"]
    hits = cylinder_hits(vertices, faces, frame)
    points = hit_points(frame, hits)
    under = slab_values(points, zone["axis"], plane, cut, -np.inf, zone["reach"]) > 0
    hits = {key: value[under] for key, value in hits.items()}
    own, _, _ = collar_limit(vertices, faces, hits, frame, zone["bottom"], width, zone["reach"])
    pixel_id, radius = hits["pixel"], hits["radius"]
    valid = radius < np.minimum(own, zone["limit"]).reshape(-1)[pixel_id]
    if zone.get("cover") is not None:       # a plate further out than the old head ever reached is the build's brim
        reach = zone["cover"][pixel_id % frame["cols"]] + COVER_MARGIN_SHARE * width
        valid &= (np.abs(hits["up"]) < SIDE_LIMIT) | (radius < reach)
    outer = np.full(frame["rows"] * frame["cols"], -np.inf)
    np.maximum.at(outer, pixel_id[valid], radius[valid])
    skin = np.zeros(len(np.asarray(faces)), bool)          # the faces the outermost surface is made of
    skin[hits["face"][valid & (radius >= outer[pixel_id])]] = True
    inner = np.full(frame["rows"] * frame["cols"], np.inf)    # its own skin: the first surface facing out
    leaving = valid & (hits["facing"] > FACING_LIMIT)
    np.minimum.at(inner, pixel_id[leaving], radius[leaving])
    outer = outer.reshape(frame["rows"], frame["cols"])
    opened = ndimage.grey_opening(outer, size=(SPIKE_ROWS, 1), mode="nearest")   # an empty pixel counts as low
    with np.errstate(invalid="ignore"):
        outer = np.where(outer - opened > SPIKE_SHARE * width, -np.inf, outer)
    known = np.isfinite(outer)
    filled = fill_nearest(np.where(known, outer, 0.0), known)
    inner = inner.reshape(frame["rows"], frame["cols"])
    return {"radius": filled, "known": known, "plane": plane, "skin": skin,
            "inner": np.where(known & np.isfinite(inner), np.minimum(inner, filled), filled)}


def wrap_round(radius: np.ndarray, sigma: float, jump: float) -> np.ndarray:
    """`hero_face.round_relief` on a picture unrolled round an axis: its first and last columns neighbours."""
    cols = radius.shape[1]
    pad = min(cols, int(np.ceil(4 * sigma)) + hf.CLIFF_PIXELS + 1)
    padded = np.concatenate([radius[:, cols - pad:], radius, radius[:, :pad]], axis=1)
    out = hf.round_relief(padded, np.isfinite(padded), sigma, jump)
    return out[:, pad:pad + cols]


def tube_radius(zone: dict, built: dict) -> np.ndarray:
    """The new head's radius in every pixel of the zone's picture: the build's (`build_radius`) where it has its head,
    and above that (under the build's own brim, which a face crop cuts lower than the hat's) its highest row carried on
    up; creases rounded over ROUND_SHARE of the head's width (`wrap_round`). Where the old head is cut up to the hat's
    opening (zone "opening") the new head keeps its own width under the brim (`crown_width`), eased into over
    CROWN_EASE of the band. Without an opening the old head stays over the cut, and in the band under the brim the
    new head is at least as wide as the old head there (zone "cover": the widest of it round each column, softened,
    and reached in full at the cut), so the old head's cut and the caps closing it lie inside the new one and no gap
    shows between hat and hair. Either way not wider than the old head there by more than COVER_CAP_SHARE of the
    head's width (a drooping piece of the build's brim stood out there as a comb of plates). The figure's own surface
    there is its old hair: taken as it is, its spikes stood out of the new head as fins (the Zauberer, 2026-11-22).
    Under the chin and the jaw the neck slopes down to the stitch (`neck_fill`)."""
    frame, width, pixel = zone["frame"], zone["width"], zone["pixel"]
    radius = np.array(built["radius"], dtype=np.float64)
    known = built["known"]
    if built.get("inner") is not None:
        radius = press_strands(zone, radius, np.asarray(built["inner"], dtype=np.float64), known)
    rows = np.arange(frame["rows"])[:, None]
    first = np.where(known.any(axis=0), known.argmax(axis=0), frame["rows"] - 1)[None, :]
    highest = np.take_along_axis(radius, np.clip(first, 0, frame["rows"] - 1), axis=0)
    radius = np.where(rows < first, highest, radius)
    radius = wrap_round(radius, ROUND_SHARE * width / pixel, ROUND_JUMP_SHARE * width)
    heights = row_heights(frame)[:, None]
    turn = column_angles(frame)
    edge = unpolar(frame, turn, zone["cover"], np.full(len(turn), zone["plane"]["a"]))
    level = (plane_height(zone["plane"], edge[:, 0], edge[:, 2]) - zone["cut"])[None, :]
    band = zone["band"][None, :]
    s = np.clip((heights - band) / np.maximum(level - band, pixel), 0, 1)
    least = zone["cover"][None, :] + 1.5 * COVER_MARGIN_SHARE * width
    # up the band the build's own surface is softened more and more, so the step where it ran into its own brim (a notch
    # over the forehead) and its lumps do not reach the brim; under it a smooth flare from the head's width at the
    # band's foot to the old head's at the cut: where the build is narrower the new head is that flare, so it widens
    # evenly up to the hat instead of in a puffy roll just under it (the Zauberer, 2026-11-23)
    soft = blur_up(radius, s, FLARE_BLUR_SHARE * width / pixel)
    foot_row = np.clip(np.floor((frame["y_hi"] - zone["band"]) / pixel).astype(np.int64), 0, frame["rows"] - 1)
    foot = ndimage.gaussian_filter1d(radius[foot_row, np.arange(frame["cols"])], FLARE_SMOOTH * frame["cols"],
                                     mode="wrap")[None, :]
    ease = s * s * (3 - 2 * s)
    # at most a little wider than the old head there, or than the new one at the band's foot where that is wider
    most = np.maximum(least, foot) + COVER_CAP_SHARE * width
    if zone.get("opening") is not None:
        # the old head is cut up to the hat's opening and the opening closed (`graft_head`): nothing of it is left to
        # cover, so the new head keeps its own width up into the hat. Widened to the old hair flaring out under the
        # brim, the Zauberer's stood under his hat as a shelf a centimetre wider than his head, a mushroom seen from
        # below and behind (2026-11-23)
        crown = np.minimum(crown_width(soft, built["known"], s)[None, :], most)
        take = np.clip((s - CROWN_EASE[0]) / (CROWN_EASE[1] - CROWN_EASE[0]), 0, 1)
        radius = soft + take * take * (3 - 2 * take) * (crown - soft)
    else:
        flare = foot + ease * np.maximum(least - foot, 0)
        take = np.clip(s / FLARE_FADE, 0, 1)
        radius = soft + take * take * (3 - 2 * take) * np.maximum(flare - soft, 0)
    radius = np.minimum(radius, radius + ease * np.minimum(most - radius, 0))
    # the ears rounded last: before, they narrowed the head's own width under the brim read from the band (`crown_width`),
    # and round the narrower head the hat's opening film showed its up-facing parts from below (see-through slits)
    return neck_cover(zone, round_flaps(zone, neck_fill(zone, radius)))


def press_strands(zone: dict, radius: np.ndarray, inner: np.ndarray, known: np.ndarray) -> np.ndarray:
    """RADIUS (the build's outermost surface, `build_radius`) with every narrow piece standing off the build's own skin
    (INNER) laid onto it: where the outermost surface stands more than STRAND_GAP_SHARE of the head's width off the
    skin, with air between (a strand of hair in front of the forehead), and no square STRAND_WIDE_SHARE of it either
    side fits in it (a lock of hair is wider and keeps its volume), it stands at most STRAND_RELIEF_SHARE of it off the
    skin, never under what lies round it. Unrolled round the head such a strand became a fin: on the Zauberer one stood
    8 mm off his forehead over his right eye, 4 mm wide, and its sides read as a dark notch (2026-11-23). An ear seen
    edge-on stands off the head with no air behind it along the rays, and stays. Under the brim (in the band where the
    head's own width there is read, zone "band", or FLAP_BRIM_SHARE of the head's width under its cut) only the face's
    strands are laid down (FLAP_TURN's first share either side of the nose): beside and behind it the hair runs up into
    the hat's opening, and taken in there it narrowed the head under the brim, round which the opening's film showed
    its up-facing parts from below, see-through slits in the gallery."""
    width, pixel = zone["width"], zone["pixel"]
    cols = radius.shape[1]
    allowed = np.ones(radius.shape, bool)
    if zone.get("nose") is not None and zone.get("plane") is not None:
        nose, _ = polar(zone["frame"], np.asarray(zone["nose"], dtype=np.float64)[None])
        turn = np.abs(np.angle(np.exp(1j * (column_angles(zone["frame"]) - nose[0]))))
        top = zone["plane"]["a"] - zone.get("cut", 0.0) - FLAP_BRIM_SHARE * width
        foot = np.minimum(np.asarray(zone["band"], dtype=np.float64), top) if zone.get("band") is not None else top
        high = row_heights(zone["frame"])[:, None] > np.broadcast_to(foot, (zone["frame"]["cols"],))[None, :]
        allowed = ~(high & (turn > FLAP_TURN[0] * np.pi)[None, :])
    size = 2 * max(1, int(round(STRAND_WIDE_SHARE * width / pixel))) + 1
    pad = min(cols, size)
    padded = np.concatenate([radius[:, cols - pad:], radius, radius[:, :pad]], axis=1)
    opened = ndimage.grey_opening(padded, size=(size, size), mode="nearest")[:, pad:pad + cols]
    gap = radius - inner
    with np.errstate(invalid="ignore"):
        strand = (np.asarray(known, bool) & np.isfinite(inner) & (gap > STRAND_GAP_SHARE * width)
                  & (radius - opened > STRAND_GAP_SHARE * width))
    out = np.array(radius, dtype=np.float64)
    if strand.any():
        strand = wrap_grow(strand, 2) & np.isfinite(inner) & allowed     # its edges too, where the gap closes
        pressed = np.maximum(inner + np.clip(gap, 0, STRAND_RELIEF_SHARE * width), opened)
        out = np.where(strand, np.minimum(out, pressed), out)
    # what is left of a strand's tip, a blade fewer than SPIKE_ROWS pixels thin standing out of the head, goes down to
    # what lies round it: two such stood behind the Zauberer's right ear, 3 to 8 mm out (2026-11-23)
    across = ndimage.grey_opening(out, size=(SPIKE_ROWS, 1), mode="nearest")
    out = np.where(allowed & (out - across > STRAND_GAP_SHARE * width), across, out)
    pad = min(cols, SPIKE_ROWS)
    padded = np.concatenate([out[:, cols - pad:], out, out[:, :pad]], axis=1)
    along = ndimage.grey_opening(padded, size=(1, SPIKE_ROWS), mode="nearest")[:, pad:pad + cols]
    return np.where(allowed & (out - along > STRAND_GAP_SHARE * width), along, out)


def round_flaps(zone: dict, radius: np.ndarray) -> np.ndarray:
    """RADIUS with the pieces standing out of the head's sides rounded: on its sides (FLAP_TURN from the nose), between
    the chin and the brim's cut, whatever stands more than FLAP_LEAST_SHARE of the head's width out of the head round it
    (its grey opening, FLAP_WIDE_SHARE either side: an ear) stands at most FLAP_STAND_SHARE of it out, its rim rounded
    over FLAP_ROUND_SHARE of it, eased into the head round it. Unrolled round the head an ear is a fin whose rim the
    rounding keeps as a wall (a step past ROUND_JUMP_SHARE), and the Zauberer's stood as a flat oval plate with a hard
    rim (2026-11-23); the face's own steps (the nose, the chin) lie outside FLAP_TURN."""
    frame, width, pixel = zone["frame"], zone["width"], zone["pixel"]
    if zone.get("nose") is None:
        return radius
    nose, _ = polar(frame, np.asarray(zone["nose"], dtype=np.float64)[None])
    turn = np.abs(np.angle(np.exp(1j * (column_angles(frame) - nose[0]))))
    heights = row_heights(frame)
    side = (turn > FLAP_TURN[0] * np.pi) & (turn < FLAP_TURN[1] * np.pi)
    # never right under the brim: the hair there runs up into the hat's opening, and taken in it opened slits round
    # the opening's film that the gallery showed as see-through from below (533 pixels against 113, 2026-11-23)
    under = zone["plane"]["a"] - zone["cut"] - FLAP_BRIM_SHARE * width
    fade = np.clip((under - heights) / (0.5 * FLAP_BRIM_SHARE * width), 0, 1)
    rows = (heights > zone.get("chin", zone["bottom"])) & (fade > 0)
    cols = radius.shape[1]
    size = 2 * max(1, int(round(FLAP_WIDE_SHARE * width / pixel))) + 1
    pad = min(cols, size)
    padded = np.concatenate([radius[:, cols - pad:], radius, radius[:, :pad]], axis=1)
    base = ndimage.grey_opening(padded, size=(size, size), mode="nearest")[:, pad:pad + cols]
    stand = radius - base
    flap = (stand > FLAP_LEAST_SHARE * width) & rows[:, None] & side[None, :]
    if not flap.any():
        return radius
    sigma = FLAP_ROUND_SHARE * width / pixel
    limited = base + np.minimum(stand, FLAP_STAND_SHARE * width)
    pad = min(cols, int(np.ceil(4 * sigma)) + 1)
    padded = np.concatenate([limited[:, cols - pad:], limited, limited[:, :pad]], axis=1)
    soft = ndimage.gaussian_filter(padded, sigma, mode="nearest")[:, pad:pad + cols]
    near = wrap_grow(flap, int(np.ceil(2 * sigma))).astype(np.float64)
    padded = np.concatenate([near[:, cols - pad:], near, near[:, :pad]], axis=1)
    weight = np.clip(2 * ndimage.gaussian_filter(padded, sigma, mode="nearest")[:, pad:pad + cols], 0, 1)
    weight = weight * fade[:, None]
    # rounded from the ear's side only: its corners cut, the head round it as it was
    return weight * np.minimum(soft, limited) + (1 - weight) * radius


def neck_cover(zone: dict, radius: np.ndarray) -> np.ndarray:
    """RADIUS (a picture of the zone's frame) under the stitch, where the new neck reaches down into a collar (zone
    "low"): at least NECK_COVER_SHARE of the head's width outside the old figure's skin (zone "skin") wherever that lies
    within NECK_COVER_REACH of it (the old neck standing free under the stitch, and its front between the collar's open
    fronts), as a smooth envelope, so nothing of the old neck, its old paint and its ledges shows through it; a collar's
    inside or a chest further out stays in front of the new neck, which never reaches past the zone's limit there (a
    collar's inside less a margin) nor keeps what stands narrow out of the build's neck. Its lowest NECK_DIVE_SHARE of
    the head's width lies under the old surface, so it ends inside the body. RADIUS as it is without a deep neck."""
    low, bottom = zone.get("low", zone["bottom"]), zone["bottom"]
    if low >= bottom or zone.get("skin") is None:
        return radius
    frame, width, pixel = zone["frame"], zone["width"], zone["pixel"]
    heights = row_heights(frame)[:, None] * np.ones((1, frame["cols"]))
    skin = np.asarray(zone["skin"], dtype=np.float64)
    under = (heights <= bottom + pixel) & (heights >= low - BLEND_ROWS * pixel)
    limit = np.asarray(zone["limit"], dtype=np.float64)
    # under the stitch the build's neck without what stands narrow out of it: a piece of its own collar's edge stood
    # out of the Zauberer's new neck as a violet chip beside the collar's front (2026-11-23)
    cols = radius.shape[1]
    size = 2 * max(1, int(round(STRAND_WIDE_SHARE * width / pixel))) + 1
    pad = min(cols, size)
    padded = np.concatenate([radius[:, cols - pad:], radius, radius[:, :pad]], axis=1)
    opened = ndimage.grey_opening(padded, size=(size, size), mode="nearest")[:, pad:pad + cols]
    radius = np.where(under & (heights < bottom - pixel), np.minimum(radius, opened), radius)
    with np.errstate(invalid="ignore"):
        near = (under & np.isfinite(skin) & (skin < radius + NECK_COVER_REACH * width)
                & (skin + NECK_COVER_SHARE * width <= limit))      # a collar's inside is no neck to cover
    need = np.where(near, skin + NECK_COVER_SHARE * width, -np.inf)
    out = np.array(radius, dtype=np.float64)
    if near.any():
        grown = ndimage.maximum_filter(need, size=5, mode=("nearest", "wrap"))
        have = np.isfinite(grown)
        num = ndimage.gaussian_filter(np.where(have, grown, 0.0), 2.0, mode=("nearest", "wrap"))
        den = ndimage.gaussian_filter(have.astype(np.float64), 2.0, mode=("nearest", "wrap"))
        soft = np.divide(num, den, out=np.full(num.shape, -np.inf), where=den > 1e-6)
        out = np.where(have, np.maximum(out, np.maximum(soft, need)), out)
    out = np.where(under, np.minimum(out, limit), out)          # never out through the collar
    # under the old surface at the bottom, eased from over it to under it over the band's top third
    depth = np.clip((low + NECK_DIVE_SHARE * width - heights) / (NECK_DIVE_SHARE * width / 3), 0, 1)
    with np.errstate(invalid="ignore"):
        dive = (heights < low + NECK_DIVE_SHARE * width) & np.isfinite(skin)
    return np.where(dive, np.minimum(out, skin + NECK_COVER_SHARE * width * (1 - 2 * depth)), out)


def crown_width(soft: np.ndarray, known: np.ndarray, share: np.ndarray) -> np.ndarray:
    """The new head's own width under the brim, one per column: its softened surface (SOFT, `blur_up`) where the build
    shows its head (KNOWN) in the band from CROWN_READ of it up to the cut (SHARE, 0 at the band's foot, 1 at the
    cut), its CROWN_PERCENTILE, a running median over CROWN_SPREAD of the way round and softened. Columns with none
    take the nearest. The build's surface right under its own brim is no guide: on the Zauberer a notch where its hair
    ended stood 1.4 cm inside his head on one side, and a stray plate of its brim 1 cm out on the other."""
    upper = known & (share > CROWN_READ) & (share < 1)
    crown = np.full(soft.shape[1], np.nan)
    have = upper.any(axis=0)
    for column in np.flatnonzero(have):
        crown[column] = np.percentile(soft[upper[:, column], column], CROWN_PERCENTILE)
    if not have.any():                  # nothing shown in the band: the surface at the cut, as it is
        top = np.clip(np.argmin(np.abs(share - 1), axis=0), 0, soft.shape[0] - 1)
        crown, have = soft[top, np.arange(soft.shape[1])], np.ones(soft.shape[1], bool)
    crown = fill_nearest(np.where(have, crown, 0.0)[None, :], have[None, :])[0]
    size = max(3, int(CROWN_SPREAD * len(crown)) | 1)
    return ndimage.gaussian_filter1d(ndimage.median_filter(crown, size, mode="wrap"), size / 6, mode="wrap")


def blur_up(radius: np.ndarray, share: np.ndarray, most: float) -> np.ndarray:
    """RADIUS (a picture unrolled round an axis, its first and last columns neighbours) softened by a Gaussian whose
    width grows with SHARE (0..1 per pixel) up to MOST pixels."""
    widths = most * np.array([0.0, 0.125, 0.25, 0.5, 1.0])
    cols = radius.shape[1]
    pad = min(cols, int(np.ceil(3 * most)) + 1)
    padded = np.concatenate([radius[:, cols - pad:], radius, radius[:, :pad]], axis=1)
    levels = np.stack([padded if w <= 0 else ndimage.gaussian_filter(padded, w, mode=("nearest", "wrap"))
                       for w in widths])[:, :, pad:pad + cols]
    at = np.clip(np.asarray(share, dtype=np.float64), 0, 1) ** 2 * (len(widths) - 1)
    low = np.minimum(np.floor(at).astype(np.int64), len(widths) - 2)
    t = at - low
    rows, columns = np.indices(radius.shape)
    return (1 - t) * levels[low, rows, columns] + t * levels[low + 1, rows, columns]


def neck_fill(zone: dict, radius: np.ndarray) -> np.ndarray:
    """RADIUS (a picture of the zone's frame) with the pocket under the chin and jaw filled: in each column, through
    the new head's neck (`neck_band`), at least the straight line from the stitch up to where the neck ends (a chin's
    or a jaw's edge), never past the region's limit (a collar). A narrow neck under a wide jaw left a pocket that the
    game figure's few triangles spanned in the air, and the bake painted them from the collar's dark inside: a beard
    under the Zauberer's chin (2026-11-22)."""
    frame = zone["frame"]
    heights = row_heights(frame)
    above = heights >= zone["bottom"]
    if above.sum() < 2:
        return radius
    sub, h = radius[above], heights[above]
    band = neck_band(unpolar(frame, column_angles(frame)[None, :], sub, h[:, None]), zone)
    top = np.clip(len(h) - 1 - band.sum(axis=0), 0, len(h) - 1)     # the row just over the neck, column by column
    rise = sub[top, np.arange(sub.shape[1])] - sub[-1]
    # round it from column to column, so a narrow chin's underside runs into the jaw beside it instead of standing
    # under it as a squared tab (the Zauberer's game figure, 2026-11-22). Where the head over the neck is narrower (a
    # nape under the hair) the line runs in to it: held at the stitch's width up to the band's top, the Zauberer's neck
    # ended there in a shelf 3 mm deep all round the back (2026-11-23)
    sigma = NECK_ROUND_SHARE * sub.shape[1]
    over = ndimage.gaussian_filter1d(h[top], sigma, mode="wrap")
    rise = ndimage.gaussian_filter1d(rise, sigma, mode="wrap")
    share = np.clip((h[:, None] - h[-1]) / np.maximum(over - h[-1], frame["pixel"])[None, :], 0, 1)
    line = np.minimum(sub[-1][None, :] + share * rise[None, :], zone["limit"][above])
    filled = np.where(band, np.maximum(sub, line), sub)
    # the chin's and the jaw's lower edge rounded: where the neck's slope meets them they stood as a squared block seen
    # from below (the build's chin has a flat underside, 0.18 to 0.28 of the head's width deep), and a game figure's
    # few triangles made it blockier still (the Zauberer, 2026-11-23). Only taken in, never out (a corner pointing out)
    tall = EDGE_ROUND_SHARE * zone["width"] / frame["pixel"]
    wide = tall * frame["pixel"] / max(float(np.median(sub[np.isfinite(sub)])) * frame["turn"], 1e-12)
    pad = min(sub.shape[1], int(np.ceil(4 * wide)) + 1)
    padded = np.concatenate([filled[:, -pad:], filled, filled[:, :pad]], axis=1)
    soft = ndimage.gaussian_filter(padded, (tall, wide), mode=("nearest", "wrap"))[:, pad:pad + sub.shape[1]]
    near = (np.arange(len(h))[:, None] >= (top - 2 * tall)[None, :]) & band.any(axis=0)[None, :]
    out = np.array(radius, dtype=np.float64)
    out[above] = np.where(near, np.minimum(filled, soft), filled)
    return out


def sample_map(frame: dict, values: np.ndarray, angle: np.ndarray, height: np.ndarray) -> np.ndarray:
    """VALUES (a picture of FRAME) at ANGLE and HEIGHT, between pixel middles in a straight line (columns wrap)."""
    col = (np.asarray(angle) + np.pi) / frame["turn"] - 0.5
    row = (frame["y_hi"] - np.asarray(height)) / frame["pixel"] - 0.5
    padded = np.concatenate([values[:, -1:], values, values[:, :1]], axis=1)
    return ndimage.map_coordinates(padded, [row, col + 1], order=1, mode="nearest")


def tube_top(zone: dict, angle: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """The height the new head reaches up to at ANGLE: TOP_SHARE of the head's width over the brim's plane where the
    head's side (RADIUS) meets it, or over the hat's opening (zone "top") where that is higher, inside the hat."""
    rim = unpolar(zone["frame"], angle, radius, np.full(len(angle), zone["plane"]["a"]))
    top = plane_height(zone["plane"], rim[:, 0], rim[:, 2])
    if zone.get("top") is not None:
        top = np.maximum(top, sample_columns(zone["top"], zone["frame"], angle))
    return top + TOP_SHARE * zone["width"]


def tube_low(zone: dict) -> float:
    """How far down the new head reaches: into a collar (zone "low", `head_zone`), else to the stitch."""
    return float(zone.get("low", zone["bottom"]))


def tube_part(zone: dict, radius: np.ndarray, step: float, bottom: tuple[np.ndarray, np.ndarray] | None = None,
              close_bottom: bool = False) -> dict:
    """The new head laid out as a grid round the axis, every STEP apart (about square), closed all round, from the
    zone's bottom (`tube_low`) up into the hat (`tube_top`), capped there, its faces out. RADIUS is `tube_radius`'s
    picture. BOTTOM (angles, radii of the figure's own neck at the stitch) puts the lowest row onto it (NECK_EASE_ROWS:
    on the Zauberer the stitch lies 1.2 mm under the chin, and easing six rows pulled the chin in), so the stitch is a
    band of one row: the chin's underside. CLOSE_BOTTOM caps the lowest row too (a neck reaching down into a collar,
    stitched to nothing). Texture coordinates run once round (u, the back at 0 and 1) and down (v, the top at 0); the
    back's column is doubled for that. Returns a part with "bottom_ids" (the lowest row round, in the order the angle
    grows) and "cap" (the caps' faces)."""
    frame = zone["frame"]
    low = tube_low(zone)
    mean = float(np.median(radius[np.isfinite(radius)]))
    cols = max(MIN_COLUMNS, int(round(2 * np.pi * mean / step)))
    turn = -np.pi + np.arange(cols + 1) * (2 * np.pi / cols)
    mid = sample_map(frame, radius, turn, np.full(cols + 1, zone["bottom"] + 0.5 * (frame["y_hi"] - zone["bottom"])))
    top = tube_top(zone, turn, mid)
    span = float(np.max(top) - low)
    rows = max(2, int(round(span / step)))
    s = np.linspace(0.0, 1.0, rows + 1)
    height = low + s[:, None] * (top[None, :] - low)
    angle = np.broadcast_to(turn[None, :], height.shape)
    r = sample_map(frame, radius, angle.reshape(-1), height.reshape(-1)).reshape(height.shape)
    if bottom is not None:
        angles, radii = bottom
        order = np.argsort(angles)
        ring = np.interp(turn, angles[order], radii[order], period=2 * np.pi)
        ease = np.clip(1 - np.arange(rows + 1) / NECK_EASE_ROWS, 0, 1)[:, None]
        r = ease * ring[None, :] + (1 - ease) * r
    points = unpolar(frame, angle, r, height).reshape(-1, 3)
    ids = np.arange((rows + 1) * (cols + 1)).reshape(rows + 1, cols + 1)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    u = np.broadcast_to(np.arange(cols + 1)[None, :] / cols, height.shape)
    v = np.broadcast_to(1.0 - s[:, None], height.shape)
    uvs = np.stack([u.reshape(-1), v.reshape(-1)], axis=1)
    rim = ids[-1, :-1]
    middle = np.append(points[rim].mean(axis=0), [])[None, :]
    centre_id = len(points)
    cap = np.stack([np.full(cols, centre_id), rim, np.roll(rim, -1)], axis=1)
    cap[-1, 2] = ids[-1, -1]                                   # the doubled back column closes the last one
    extra, extra_uvs = [middle], [[0.5, 0.0]]
    if close_bottom:
        ring = ids[0, :-1]
        under = np.stack([np.full(cols, centre_id + 1), np.roll(ring, -1), ring], axis=1)
        under[-1, 1] = ids[0, -1]
        cap = np.concatenate([cap, under])
        extra.append(points[ring].mean(axis=0)[None, :])
        extra_uvs.append([0.5, 1.0])
    return {"positions": np.concatenate([points] + extra), "faces": np.concatenate([faces, cap]),
            "uvs": np.concatenate([uvs, extra_uvs]), "bottom_ids": ids[0, :-1], "cap": len(cap),
            "grid": (rows + 1, cols + 1)}


# --- Setting it in --------------------------------------------------------------------------------------------------

def cycles(edges: np.ndarray) -> list[np.ndarray]:
    """The closed outlines of a cut (EDGES as pairs of ids), each as its ids in walking order, walked the way most of its
    own edges point. The edges are followed whatever way each points: a cut through a face whose winding is the wrong
    way round (a build's clean-up leaves a few) gives an edge against its neighbours, and the Zauberer's neck outline
    broke into two open runs where three such edges lay. Runs that do not close are left out."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    around: dict[int, list[tuple[int, int]]] = {}
    for k, (a, b) in enumerate(edges.tolist()):
        if a != b:
            around.setdefault(a, []).append((b, k))
            around.setdefault(b, []).append((a, k))
    used = np.zeros(len(edges), bool)
    out = []
    for first in range(len(edges)):
        a, b = int(edges[first, 0]), int(edges[first, 1])
        if used[first] or a == b:
            continue
        used[first] = True
        loop, agree, here = [a], 1, b
        while here != a:
            loop.append(here)
            step = next(((n, k) for n, k in around[here] if not used[k]), None)
            if step is None:
                break
            n, k = step
            used[k] = True
            agree += int(edges[k, 0] == here)
            here = n
        if here == a and len(loop) >= 3:
            out.append(np.array(loop if agree * 2 >= len(loop) else loop[::-1], dtype=np.int64))
    return out


def drop_slivers(part: dict, rim: np.ndarray, zone: dict) -> tuple[dict, np.ndarray, int]:
    """PART as the cut left it (RIM, the cut's edges) without the pieces that touch the cut and lie wholly within
    SLIVER_PIXELS of the region: bits of the old head that reached just past its edge. The Zauberer's chin dipped 0.2 mm
    under the stitch, and its tip stayed under the new chin as a patch of the old paint (2026-11-22). Returns (part,
    rim, how many pieces went)."""
    f = np.asarray(part["faces"], dtype=np.int64)
    if not len(f):
        return part, rim, 0
    label = hh.components(f, hh.weld_groups(np.asarray(part["positions"])))
    values = zone_values(zone, part["positions"])[f]
    least, most = np.full(label.max() + 1, np.inf), np.full(label.max() + 1, -np.inf)
    np.minimum.at(least, label, values.min(axis=1))
    np.maximum.at(most, label, values.max(axis=1))
    sliver = (least > -SLIVER_PIXELS * zone["pixel"]) & (most > -0.5 * zone["pixel"])
    if not sliver.any():
        return part, rim, 0
    kept, rim = hh.keep_faces(part, ~sliver[label], rim)
    return kept, rim, int(sliver.sum())


def cap_loops(part: dict, loops: list[np.ndarray]) -> tuple[dict, int]:
    """PART with every outline of LOOPS (vertex ids in the order its own faces' boundary runs, `cycles`) closed by a fan
    from its middle, facing as the cut surface's own faces do across it. Returns (the part, how many were closed). A cut
    through a hollow build leaves an outline on its skin and one on its inner wall: closed, both stay watertight."""
    if not loops:
        return part, 0
    vertices = np.asarray(part["positions"], dtype=np.float64)
    added_points, added_faces, middles = [], [], []
    for ids in loops:
        centre_id = len(vertices) + len(added_points)
        added_points.append(int(ids[0]))
        middles.append(vertices[ids].mean(axis=0))
        added_faces.append(np.stack([np.full(len(ids), centre_id), np.roll(ids, -1), ids], axis=1))
    out = dict(part)
    out["positions"] = np.concatenate([vertices, np.array(middles)])
    for key in hh.ATTRIBUTES:
        if part.get(key) is not None:
            values = np.asarray(part[key])
            out[key] = np.concatenate([values, values[added_points]])
    out["faces"] = np.concatenate([np.asarray(part["faces"])] + added_faces)
    return out, len(loops)


def membrane(outline: np.ndarray, rings: int = MEMBRANE_RINGS) -> tuple[np.ndarray, np.ndarray]:
    """(points, triangles) of a film spanning OUTLINE (a closed loop's points, in order), as soap spans a wire: the
    outline laid round a circle by its length, RINGS rings of points inside it and its middle, joined as the circle's
    Delaunay triangles, and every inner point moved to the mean of its neighbours (the outline held), so the film is
    smooth and its triangles are small near the outline. The outline's points come first, in order. A fan from the
    middle folds where the outline turns back; triangles from outline point to outline point alone ran across a
    saddle-shaped opening as long slivers (the Zauberer's, 2 cm higher at his ears than at his back, 2026-11-23)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import Delaunay

    outline = np.asarray(outline, dtype=np.float64)
    n = len(outline)
    step = np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1)
    turn = 2 * np.pi * np.r_[0.0, np.cumsum(step)[:-1]] / max(step.sum(), 1e-18)
    flat = [np.stack([np.cos(turn), np.sin(turn)], axis=1)]
    for k in range(1, rings + 1):
        share = 1 - k / (rings + 1)
        count = max(3, int(round(n * share)))
        inner = 2 * np.pi * (np.arange(count) + 0.5 * k) / count
        flat.append(share * np.stack([np.cos(inner), np.sin(inner)], axis=1))
    flat.append(np.zeros((1, 2)))
    flat = np.concatenate(flat)
    triangles = Delaunay(flat).simplices.astype(np.int64)
    a, b, c = flat[triangles[:, 0]], flat[triangles[:, 1]], flat[triangles[:, 2]]
    turned = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
    triangles[turned] = triangles[turned][:, ::-1]
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    m = len(flat)
    links = coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                       shape=(m, m)).tocsr()
    degree = np.asarray(links.sum(axis=1)).ravel()
    free = np.arange(n, m)
    lap = (coo_matrix((degree[free], (np.arange(len(free)), np.arange(len(free)))), shape=(len(free), len(free)))
           - links[free][:, free]).tocsc()
    points = np.zeros((m, 3))
    points[:n] = outline
    points[free] = np.stack([spsolve(lap, links[free][:, :n] @ outline[:, k]) for k in range(3)], axis=1)
    # never over the outline nearest it: across a saddle the film rose a few pixels inside its low end, over the cut's
    # outline on the hollow's inner wall there, whose cap then showed through it from below (`lift_inner_caps`)
    from scipy.spatial import cKDTree

    _, near = cKDTree(outline[:, [0, 2]]).query(points[free][:, [0, 2]])
    points[free, 1] = np.minimum(points[free, 1], outline[near, 1])
    return points, triangles


def lift_inner_caps(points: list[np.ndarray], faces: list[np.ndarray], rims: list[int]) -> None:
    """Every cap but the widest (POINTS and FACES, one entry a cap, each's outline the first RIMS points of it) raised,
    in place, where it lies under the widest one as seen from below, to a little over it: a cut through a hollow build
    leaves an outline on its skin and one on its inner wall, and on a saddle-shaped opening the two films crossed, so
    the inner wall's cap (facing up, into the hollow) showed through the outer one from below, a see-through hole in
    the gallery round the Zauberer's head (2026-11-23). Outlines stay where they are."""
    if len(points) < 2:
        return
    start = np.cumsum([0] + [len(p) for p in points])
    spans = [np.ptp(p[:, [0, 2]], axis=0).prod() for p in points]
    widest = int(np.argmax(spans))
    outer, outer_faces = points[widest], faces[widest] - start[widest]
    size = float(np.ptp(outer[:, [0, 2]], axis=0).max())
    pixel, gap = size / 200, size * 0.002
    seen = np.stack([outer[:, 0], -outer[:, 2], -outer[:, 1]], axis=1)
    box = (float(outer[:, 0].min()) - pixel, float(outer[:, 0].max()) + pixel, float(-outer[:, 2].max()) - pixel,
           float(-outer[:, 2].min()) + pixel)
    depth, _ = hf.depth_map(seen, outer_faces, box, pixel)
    for k, cap in enumerate(points):
        if k == widest:
            continue
        inner = cap[rims[k]:]
        floor = -hf.sample(depth, box, pixel, np.stack([inner[:, 0], -inner[:, 2]], axis=1), outside=-np.inf,
                           nearest=True)
        under = np.isfinite(floor) & (inner[:, 1] < floor + gap)
        inner[under, 1] = floor[under] + gap


def opening_cap(part: dict, loops: list[np.ndarray], material=None) -> dict:
    """A part of its own closing every outline of LOOPS (vertex ids of PART, in the order its own faces' boundary runs,
    `cycles`), facing as `cap_loops` would: a film spanning it (`membrane`), or a fan from its middle for a small one.
    Its outline's corners lie at the outlines' places (welded by place, so nothing opens), with their skin weights; its
    inner ones take the nearest corner's; the normals are its faces'. With a MATERIAL (one colour, the hat's) every
    corner takes one texture coordinate; without one it takes the part's own, as `cap_loops` gives them."""
    vertices = np.asarray(part["positions"], dtype=np.float64)
    points, faces, source = [], [], []
    for ids in loops:
        start = sum(len(p) for p in points)
        ring = start + np.arange(len(ids))
        fan = np.stack([np.full(len(ids), start + len(ids)), np.roll(ring, -1), ring], axis=1)
        if len(ids) >= MEMBRANE_LEAST:
            film, triangles = membrane(vertices[ids])
            nearest = np.r_[np.arange(len(ids)), np.argmin(np.linalg.norm(
                film[len(ids):, None, :] - vertices[ids][None, :, :], axis=2), axis=1)]
            points.append(film)
            source.append(ids[nearest])
            triangles = start + triangles
            p = np.concatenate(points)
            fan_points = np.concatenate([p[:start + len(ids)], vertices[ids].mean(axis=0)[None]])
            if (hf.face_normals(p, triangles).sum(axis=0) @ hf.face_normals(fan_points, fan).sum(axis=0)) < 0:
                triangles = triangles[:, ::-1]
            faces.append(triangles)
            continue
        points.append(np.concatenate([vertices[ids], vertices[ids].mean(axis=0)[None]]))
        source.append(np.r_[ids, ids[0]])
        faces.append(fan)
    lift_inner_caps(points, faces, [len(ids) for ids in loops])
    out = {"positions": np.concatenate(points), "faces": np.concatenate(faces)}
    source = np.concatenate(source)
    for key in ("dense", "marks"):
        if part.get(key) is not None:
            out[key] = np.asarray(part[key])[source]
    if material is not None:
        out["uvs"], out["material"] = np.full((len(source), 2), 0.5), material
    elif part.get("uvs") is not None:
        out["uvs"], out["material"] = np.asarray(part["uvs"])[source], part.get("material")
    from image_to_3dlab.pixal3d_native import smooth_normals

    out["normals"] = smooth_normals(out["positions"], out["faces"])
    return out


def colour_material(rgb) -> tuple[dict, bytes, int]:
    """A material of one colour (RGB, 0..255), as `hero_face.picture_material` makes one."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), tuple(int(round(float(c))) for c in rgb)).save(buffer, format="PNG")
    return hf.picture_material(buffer.getvalue(), name="HeroHatOpening")


def hat_colour(part: dict, zone: dict) -> np.ndarray | None:
    """The figure's own paint on its brim's underside round the hat's opening (PART: one model's body as it was): the
    median of the corners of its faces there facing down (BRIM_TURN), within HAT_RING_SHARE of the head's width outside
    the opening (zone "opening", "opening_radius") and of its height. None without paint or an opening."""
    paint = hf.part_paint(part)
    if paint is None or zone.get("opening") is None:
        return None
    v = np.asarray(part["positions"], dtype=np.float64)
    f = np.asarray(part["faces"], dtype=np.int64)
    middle = v[f].mean(axis=1)
    angle, distance = polar(zone["frame"], middle)
    edge = sample_columns(zone["opening_radius"], zone["frame"], angle)
    height = sample_columns(zone["opening"], zone["frame"], angle)
    band = HAT_RING_SHARE * zone["width"]
    near = ((hf.face_normals(v, f)[:, 1] < -BRIM_TURN) & (distance > edge - zone["pixel"])
            & (distance < edge + band) & (np.abs(middle[:, 1] - height) < band))
    if near.sum() < 3:
        return None
    return np.median(hf.paint_at(paint, np.asarray(part["uvs"])[f[near]].reshape(-1, 2)), axis=0)


def winding(loop: np.ndarray, point: np.ndarray) -> int:
    """How many times the outline LOOP goes round POINT seen from above (signed). Unlike the even-odd rule it stays
    right when the outline crosses itself: the Zauberer's neck outline at the stitch takes in his chin's tip, whose
    skin crosses the neck's there."""
    q = np.asarray(loop, dtype=np.float64)
    angle = np.arctan2(q[:, 0] - point[0], q[:, 2] - point[2])
    return int(np.round(np.sum(np.angle(np.exp(1j * (np.roll(angle, -1) - angle)))) / (2 * np.pi)))


def rim_ids(vertices: np.ndarray, rim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(RIM's edges on welded places, one vertex id per place): a UV seam leaves two vertices at one place."""
    group = hh.weld_groups(vertices)
    first = np.zeros(group.max() + 1, dtype=np.int64)
    first[group[::-1]] = np.arange(len(group))[::-1]
    return group[rim], first


def open_edges(parts: list[dict], near: np.ndarray | None = None) -> int:
    """How many edges of PARTS (welded by place) only one face uses: holes and loose rims. NEAR (a box x0, x1, y0, y1,
    z0, z1) counts only the edges whose middle lies in it."""
    positions = np.concatenate([np.asarray(p["positions"], dtype=np.float64) for p in parts])
    offsets = np.cumsum([0] + [len(p["positions"]) for p in parts])
    faces = np.concatenate([np.asarray(p["faces"], dtype=np.int64) + o for p, o in zip(parts, offsets)])
    group = hh.weld_groups(positions)
    f = group[faces]
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    single = unique[counts == 1]
    if near is None:
        return int(len(single))
    points = np.zeros((group.max() + 1, 3))
    points[group] = positions
    middle = points[single].mean(axis=1)
    x0, x1, y0, y1, z0, z1 = near
    inside = ((middle[:, 0] > x0) & (middle[:, 0] < x1) & (middle[:, 1] > y0) & (middle[:, 1] < y1)
              & (middle[:, 2] > z0) & (middle[:, 2] < z1))
    return int(inside.sum())


def zone_box(zone: dict) -> tuple[float, ...]:
    """A box round the region and the new head's top (x0, x1, y0, y1, z0, z1), for counting what is open there."""
    frame = zone["frame"]
    low = tube_low(zone)
    x, z = centre(frame["axis"], np.array([low, frame["y_hi"]]))
    r = zone["reach"] * 1.2
    return (float(x.min() - r), float(x.max() + r), low - 4 * zone["pixel"], frame["y_hi"],
            float(z.min() - r), float(z.max() + r))


def graft_head(parts: list[dict], zone: dict, radius: np.ndarray, step: float) -> tuple[list[dict], dict, dict]:
    """PARTS (one model of the figure, part by part, in the figure's frame) with the zone's old head cut out and the new
    head (`tube_part` of RADIUS every STEP) set in. Returns (the model's parts, the new head part, report).

    The old head is cut out along the region's edges (`zone_values`): under the brim, round the neck's lowest free
    section and inside the collar; the collar and the hat stay whole. At the neck the new head is
    stitched to the figure's own skin (`hero_hands.join_rims`), its lowest rows eased onto it, so the seam is closed;
    every other outline the cut leaves (the old head's inside the hat, a hollow build's inner wall) is closed by a fan
    (`cap_loops`), slivers of the old head just past the region's edge go (`drop_slivers`), and the new head is
    capped inside the hat. With skin weights (dense) the new head takes those of the nearest old point it replaces."""
    frame = zone["frame"]
    up = np.array([0.0, 1.0, 0.0])
    low = tube_low(zone)
    deep = low < zone["bottom"] - 1e-12
    out_parts, body_index, most, old_points, old_dense, rims = [], None, 0, [], [], []
    report = {"parts": [], "slivers_dropped": 0}
    for part in parts:
        values = zone_values(zone, part["positions"])
        f = np.asarray(part["faces"], dtype=np.int64)
        if not (values[f] > 0).any():
            out_parts.append(part)
            rims.append(np.zeros((0, 2), np.int64))
            report["parts"].append({"cut": False})
            continue
        gone = (values[f] > 0).all(axis=1)
        if part.get("dense") is not None:
            # the skin weights of what the new head replaces, and of the old neck its own neck covers in the collar
            reach = values if not deep else np.maximum(values, zone_values({**zone, "bottom": low}, part["positions"]))
            used = np.unique(f[(reach[f] > 0).any(axis=1)])
            old_points.append(np.asarray(part["positions"])[used])
            old_dense.append(np.asarray(part["dense"])[used])
        cut, rim = hh.cut_part_values(part, values)
        cut, rim, slivers = drop_slivers(cut, rim, zone)
        report["slivers_dropped"] += slivers
        report["parts"].append({"cut": True, "faces_removed": int(gone.sum())})
        if gone.sum() > most:
            body_index, most = len(out_parts), int(gone.sum())
        out_parts.append(cut)
        rims.append(rim)
    if body_index is None:
        raise ValueError("nothing of the model lies in the head's region: check the region")
    body, rim = out_parts[body_index], rims[body_index]
    p = np.asarray(body["positions"])
    cx, cz = centre(frame["axis"], zone["bottom"])
    placed, first = rim_ids(p, rim)
    loops = [first[loop] for loop in cycles(placed)]
    middle = np.array([float(cx), zone["bottom"], float(cz)])
    round_neck = [loop for loop in loops if abs(np.median(p[loop, 1]) - zone["bottom"]) < zone["pixel"]
                  and abs(winding(p[loop], middle)) >= 1]
    ids = max(round_neck, key=lambda loop: hh.loop_area(p[loop], up)) if round_neck else np.zeros(0, np.int64)
    loop = p[ids]
    # a neck reaching down into a collar is stitched to nothing: the old neck's outline at the stitch is closed with
    # the others, inside the new neck (`neck_cover`), and the new neck's end is capped deep in the collar
    stitched = len(ids) >= 3 and not deep
    head = tube_part(zone, radius, step, polar(frame, loop) if stitched else None, close_bottom=deep)
    if old_points:
        head["dense"] = hh.borrowed_weights(head["positions"], np.concatenate(old_points), np.concatenate(old_dense))
        if body.get("dense") is None:
            head.pop("dense")
    band = 0
    if stitched:
        head, band, alike = hh.join_rims(body, ids, head, head["bottom_ids"], up)
        report["stitch_turns_alike"] = alike
    rest = [loop for loop in loops if loop is not ids or not stitched]
    opening = []
    if zone.get("top") is not None:
        # the outlines the cut leaves at the hat's opening are closed by a part of their own in the hat's colour: from
        # below, round the slim new head, it reads as the brim's underside running on to the hair
        old = plane_height(zone["plane"], p[:, 0], p[:, 2]) - zone["cut"]
        high = [np.median(p[loop, 1] - old[loop]) > -2 * zone["pixel"] for loop in rest]
        opening = [loop for loop, at in zip(rest, high) if at]
        rest = [loop for loop, at in zip(rest, high) if not at]
    cap = None
    if opening:
        colour = hat_colour(parts[body_index], zone)
        cap = opening_cap(body, opening, None if colour is None else colour_material(colour))
        report["hat_colour"] = None if colour is None else [round(float(c), 1) for c in colour]
    body, closed = cap_loops(body, rest)
    out_parts[body_index] = body
    if cap is not None:
        report["opening_cap"] = len(out_parts)
        out_parts.append(cap)
    report.update({"body_part": body_index, "stitched": bool(stitched), "stitch_faces": int(band),
                   "neck": "into the collar" if deep else ("stitched" if stitched else "open"),
                   "neck_rim_points": int(len(ids)), "outlines_closed": int(closed + len(opening)),
                   "opening_outlines": len(opening), "head_faces": int(len(head["faces"])),
                   "grid": list(head["grid"])})
    head = {k: v for k, v in head.items() if k not in ("bottom_ids", "cap", "grid")}
    report["open_edges_near"] = open_edges(out_parts + [head], zone_box(zone))
    return out_parts, head, report


# --- Paint ------------------------------------------------------------------------------------------------------------

def tube_points(zone: dict, radius: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(points, normals) of the new head at the middle of every texel of its picture SIZE (rows, columns): columns run
    once round from the back (u), rows from the top down (v), as `tube_part` lays its texture out."""
    rows, cols = size
    frame = zone["frame"]
    low = tube_low(zone)
    turn = -np.pi + (np.arange(cols) + 0.5) * (2 * np.pi / cols)
    s = 1.0 - (np.arange(rows) + 0.5) / rows
    mid = sample_map(frame, radius, turn, np.full(cols, zone["bottom"] + 0.5 * (frame["y_hi"] - zone["bottom"])))
    top = tube_top(zone, turn, mid)
    height = low + s[:, None] * (top[None, :] - low)
    angle = np.broadcast_to(turn[None, :], height.shape)
    r = sample_map(frame, radius, angle.reshape(-1), height.reshape(-1)).reshape(height.shape)
    points = unpolar(frame, angle, r, height)
    along = np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)
    down = np.gradient(points, axis=0)
    normals = np.cross(down, along)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    return points, normals


def crop_paint(points: np.ndarray, normals: np.ndarray, crop_rgb: np.ndarray, crop_alpha: np.ndarray,
               crop: tuple[int, int, int, int], frame, seen: tuple[np.ndarray, tuple, float] | None,
               width: float) -> tuple[np.ndarray, np.ndarray]:
    """(colour, weight) of the picture at the new head's texels (POINTS, NORMALS): the crop's own pixels (CROP_RGB,
    upscaled) and how fully they paint each texel: where it faces the camera (CROP_FACING), the picture shows it
    (CROP_ALPHA, the cut-out) and nothing the figure keeps stands in front of it (SEEN: the kept figure's front depth,
    its box and pixel: a brim, a collar); 0 elsewhere."""
    rows, cols = points.shape[:2]
    p = points.reshape(-1, 3)
    uv = hf.picture_uvs(p, crop, crop_rgb.shape[1], frame)
    h, w = crop_rgb.shape[:2]
    x = np.clip((uv[:, 0] * w).astype(np.int64), 0, w - 1)
    y = np.clip((uv[:, 1] * h).astype(np.int64), 0, h - 1)
    in_crop = (uv[:, 0] >= 0) & (uv[:, 0] < 1) & (uv[:, 1] >= 0) & (uv[:, 1] < 1)
    facing = normals.reshape(-1, 3)[:, 2]
    weight = np.clip((facing - CROP_FACING[0]) / (CROP_FACING[1] - CROP_FACING[0]), 0, 1)
    weight = weight * weight * (3 - 2 * weight) * in_crop * (np.asarray(crop_alpha)[y, x] > 0.5)
    if seen is not None:
        depth, box, pixel = seen
        front = hf.sample(depth, box, pixel, p, outside=-np.inf, nearest=True)
        weight = weight * (front < p[:, 2] + OCCLUDE_SHARE * width)
    colour = np.asarray(crop_rgb, dtype=np.float64)[y, x]
    return colour.reshape(rows, cols, 3), weight.reshape(rows, cols)


def own_paint(part: dict, zone: dict, points: np.ndarray) -> np.ndarray:
    """The figure's own paint (PART: one model's body as it was) seen from the head's axis toward each of POINTS (the
    new head's texels): on its outermost surface in the region in that direction (or within a pixel of it), read
    between the corners of its triangle. NaN where it has none there (inside the hat). The figure's paint was laid on
    its whole body at once (Pixel Match), so its hair has the picture's tone all round; the build's own paint (Pixal3D's)
    painted the Zauberer's hair in blue, white and tan patches, a helmet in close views (2026-11-22)."""
    out = np.full(points.shape, np.nan)
    paint = hf.part_paint(part)
    if paint is None or part.get("uvs") is None:
        return out
    frame = zone["frame"]
    v = np.asarray(part["positions"], dtype=np.float64)
    f = np.asarray(part["faces"], dtype=np.int64)
    x0, x1, y0, y1, z0, z1 = zone_box(zone)
    lo, hi = v[f].min(axis=1), v[f].max(axis=1)
    f = f[(hi[:, 0] > x0) & (lo[:, 0] < x1) & (hi[:, 1] > y0) & (lo[:, 1] < y1) & (hi[:, 2] > z0) & (lo[:, 2] < z1)]
    if not len(f):
        return out
    hits = cylinder_hits(v, f, frame)
    at = hit_points(frame, hits)
    keep = np.flatnonzero(zone_values(zone, at) > -zone["pixel"])
    if not len(keep):
        return out
    keep = keep[np.lexsort((-hits["radius"][keep], hits["pixel"][keep]))]     # by pixel, the outermost first
    keep = keep[np.r_[True, hits["pixel"][keep][1:] != hits["pixel"][keep][:-1]]]
    corners = f[hits["face"][keep]]
    a, b, c = v[corners[:, 0]], v[corners[:, 1]], v[corners[:, 2]]
    e0, e1, e2 = b - a, c - a, at[keep] - a
    d00, d01, d11 = (e0 * e0).sum(1), (e0 * e1).sum(1), (e1 * e1).sum(1)
    d20, d21 = (e2 * e0).sum(1), (e2 * e1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-30, d00 * d11 - d01 * d01, 1e-30)
    t1, t2 = np.clip((d11 * d20 - d01 * d21) / den, 0, 1), np.clip((d00 * d21 - d01 * d20) / den, 0, 1)
    scale = np.maximum(t1 + t2, 1.0)
    t1, t2 = t1 / scale, t2 / scale
    uvs = np.asarray(part["uvs"], dtype=np.float64)
    uv = (1 - t1 - t2)[:, None] * uvs[corners[:, 0]] + t1[:, None] * uvs[corners[:, 1]] + t2[:, None] * uvs[corners[:, 2]]
    seen = np.full((frame["rows"] * frame["cols"], 3), np.nan)
    seen[hits["pixel"][keep]] = hf.paint_at(paint, uv)
    flat = points.reshape(-1, 3)
    row, col = pixel_of(frame, polar(frame, flat)[0], flat[:, 1])
    return seen[row * frame["cols"] + col].reshape(points.shape)


def lab(rgb) -> np.ndarray:
    """RGB (0..255, sRGB) as CIE L*a*b* (D65): distances there are about as the eye sees them."""
    c = np.clip(np.asarray(rgb, dtype=np.float64) / 255.0, 0, 1)
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = (c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
           / np.array([0.9505, 1.0, 1.089]))
    g = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * g[..., 1] - 16, 500 * (g[..., 0] - g[..., 1]), 200 * (g[..., 1] - g[..., 2])], axis=-1)


def figure_skin(own: np.ndarray, weight: np.ndarray, points: np.ndarray, zone: dict) -> np.ndarray | None:
    """The figure's own skin colour (OWN, `own_paint`): its median where the picture sees the face (WEIGHT) between the
    chin and the nose; None where it sees too little of it."""
    height = points[..., 1]
    face = np.isfinite(own[..., 0]) & (weight > 0.8) & (height > zone["chin"]) & (height < zone["nose"][1])
    return np.median(own[face], axis=0) if face.sum() >= 3 else None


def nose_turn(points: np.ndarray, zone: dict) -> np.ndarray:
    """How far round the head from the nose each of POINTS lies (radians, 0 to pi)."""
    flat = points.reshape(-1, 3)
    angle, _ = polar(zone["frame"], flat)
    nose, _ = polar(zone["frame"], np.asarray(zone["nose"], dtype=np.float64)[None])
    return np.abs(np.angle(np.exp(1j * (angle - nose[0])))).reshape(points.shape[:-1])


def ear_texels(points: np.ndarray, zone: dict) -> np.ndarray:
    """How surely each texel of the new head (POINTS, its picture's rows by its columns once round) lies on an ear,
    0 to 1: on the head's sides (FLAP_TURN from the nose), over the chin, what stands more than FLAP_LEAST_SHARE of the
    head's width out of the head round it (its grey opening, FLAP_WIDE_SHARE of the width either side), as
    `round_flaps` finds an ear, eased out round it over FLAP_ROUND_SHARE of the width."""
    width = zone["width"]
    _, r = polar(zone["frame"], points.reshape(-1, 3))
    r = r.reshape(points.shape[:2])
    across = max(float(np.median(np.linalg.norm(np.diff(points, axis=1), axis=-1))), 1e-9)
    down = max(float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=-1))), 1e-9)
    size = (2 * max(1, int(round(FLAP_WIDE_SHARE * width / down))) + 1,
            2 * max(1, int(round(FLAP_WIDE_SHARE * width / across))) + 1)
    cols = r.shape[1]
    pad = min(cols, size[1])
    padded = np.concatenate([r[:, cols - pad:], r, r[:, :pad]], axis=1)
    base = ndimage.grey_opening(padded, size=size, mode="nearest")[:, pad:pad + cols]
    turn = nose_turn(points, zone)
    side = (turn > FLAP_TURN[0] * np.pi) & (turn < FLAP_TURN[1] * np.pi)
    core = (r - base > FLAP_LEAST_SHARE * width) & side & (points[..., 1] > zone.get("chin", zone["bottom"]))
    if not core.any():
        return np.zeros(r.shape)
    sigma = (FLAP_ROUND_SHARE * width / down, FLAP_ROUND_SHARE * width / across)
    pad = min(cols, int(np.ceil(4 * sigma[1])) + 1)
    padded = np.concatenate([core[:, cols - pad:], core, core[:, :pad]], axis=1).astype(np.float64)
    return np.clip(2 * ndimage.gaussian_filter(padded, sigma, mode="nearest")[:, pad:pad + cols], 0, 1)


def head_kinds(own: np.ndarray, build: np.ndarray, weight: np.ndarray, points: np.ndarray, zone: dict) -> dict | None:
    """Which texels of the new head (POINTS) are hair and which skin, and the colours to tell them by; None when too
    little is seen to tell. The figure's own (OWN, `own_paint`) skin is read where the picture sees the face under the
    nose (WEIGHT), its hair at the back over the nose; the build's (BUILD) the same way, and the new head's kind by the
    build's paint, each texel as most of its neighbours ("new_hair"). Where the build has no paint near (over its own
    brim, and where the new head stands out of it) the head is hair: taken as skin there, the figure's own hair right
    under the Zauberer's brim was painted in his skin colour, a quarter of that band, and taken as the nearest kind the
    build tells, his forehead's skin was carried up under the brim as a tan band round the front, his ears' up his
    temples (2026-11-23). The build's skin is believed only round the face and the ears ("front", SKIN_TURN): built from
    a front picture it makes up the back of the head, and the Zauberer's came out bald behind his ears (2026-11-23).
    Returns {"skin", "hair" (the figure's), "build_skin", "build_hair", "new_hair", "front", "seen" (where the build has
    paint), "is_hair" (a colour test by hue, the figure's colours by default)}."""
    flat = points.reshape(-1, 3)
    angle, _ = polar(zone["frame"], flat)
    nose, _ = polar(zone["frame"], np.asarray(zone["nose"], dtype=np.float64)[None])
    turn = np.abs(np.angle(np.exp(1j * (angle - nose[0])))).reshape(points.shape[:2])
    height = points[..., 1]
    have = np.isfinite(own[..., 0])
    face = have & (weight > 0.8) & (height > zone["chin"]) & (height < zone["nose"][1])
    back = have & (turn > 0.8 * np.pi) & (height > zone["nose"][1])
    seen = np.isfinite(build[..., 0])
    if face.sum() < 3 or back.sum() < 3 or (face & seen).sum() < 3 or (back & seen).sum() < 3:
        return None

    def hue(rgb):
        return rgb / np.maximum(rgb.sum(axis=-1, keepdims=True), 1e-6)

    skin, hair = np.median(own[face], axis=0), np.median(own[back], axis=0)

    def is_hair(rgb, skin=skin, hair=hair):
        return np.linalg.norm(hue(rgb) - hue(hair), axis=-1) < np.linalg.norm(hue(rgb) - hue(skin), axis=-1)

    # the build told by its own skin and hair (it paints them its own way: the Zauberer's hair blue and white)
    build_skin, build_hair = np.median(build[face & seen], axis=0), np.median(build[back & seen], axis=0)
    built = is_hair(np.nan_to_num(build), build_skin, build_hair)
    votes = ndimage.uniform_filter(np.where(seen, built, False).astype(np.float64), size=(7, 15),
                                   mode=("nearest", "wrap"))
    cast = ndimage.uniform_filter(seen.astype(np.float64), size=(7, 15), mode=("nearest", "wrap"))
    new_hair = np.where(cast > 1e-9, votes > 0.5 * cast, True)
    return {"skin": skin, "hair": hair, "build_skin": build_skin, "build_hair": build_hair, "new_hair": new_hair,
            "front": turn < SKIN_TURN * np.pi, "seen": seen, "is_hair": is_hair}


def build_skin_paint(build: np.ndarray, kinds: dict, skin: np.ndarray, points: np.ndarray,
                     width: float) -> tuple[np.ndarray, np.ndarray]:
    """(the face's SKIN shaded by the build's own paint, where the new head is skin): on the texels KINDS
    (`head_kinds`) call skin round the face, where the build (BUILD, `build_colour`) has paint, the face's skin colour
    times the build's lightness against its mean over SKIN_DETAIL_SHARE of the head's WIDTH round (on its skin alone),
    within SKIN_DETAIL_RANGE. The build painted the head it built, so its shading fits the new shape (an ear's bowl
    darker, its rim lighter); its colours do not (it paints the head's sides orange where the Zauberer's face is pale),
    nor does the figure's own paint, which lay on the old head: it painted his new ears in the old ears' pink,
    misplaced, and the right one in his hat's dark violet (2026-11-23). POINTS are the texels' places."""
    where = ~kinds["new_hair"] & kinds["front"] & kinds["seen"]
    light = np.nan_to_num(build) @ np.array([0.2126, 0.7152, 0.0722])
    if not where.any():
        return np.zeros(build.shape), where
    across = np.median(np.linalg.norm(np.diff(points, axis=1), axis=-1))
    down = np.median(np.linalg.norm(np.diff(points, axis=0), axis=-1))
    sigma = (SKIN_DETAIL_SHARE * width / max(down, 1e-9), SKIN_DETAIL_SHARE * width / max(across, 1e-9))
    # the mean over the plain skin round each texel: an ear's dark bowl left out, or the skin round it came out lighter
    usual = float(np.median(light[where]))
    plain = (where & (np.abs(light - usual) < SKIN_PLAIN * usual)).astype(np.float64)
    num = ndimage.gaussian_filter(light * plain, sigma, mode=("nearest", "wrap"))
    den = ndimage.gaussian_filter(plain, sigma, mode=("nearest", "wrap"))
    mean = np.divide(num, den, out=np.full_like(num, usual), where=den > 1e-3)
    shade = np.clip(light / np.maximum(mean, 1e-6), *SKIN_DETAIL_RANGE)
    return np.clip(np.asarray(skin, dtype=np.float64)[None, None, :] * shade[..., None], 0, 255), where


def same_kind(own: np.ndarray, build: np.ndarray, weight: np.ndarray, points: np.ndarray, zone: dict,
              hat: np.ndarray | None = None) -> np.ndarray:
    """OWN (`own_paint`) kept where it is the kind the new head has there, hair or skin, else the figure's own colour of
    that kind (its median, shaded by the build's paint). The kind is told by the colour's hue (its share of red, green
    and blue): the figure's skin is read where the picture sees the face under the nose (WEIGHT), its hair at the back
    over the nose; the new head's kind by the build's own paint (BUILD), each texel as most of its neighbours. On the
    Zauberer the new hair lay over the old head's ears and cheeks, and their skin stood in it as patches
    (2026-11-22). With the HAT's colour (`hat_colour`), OWN nearer it than to the hair and the skin (in L*a*b*) under
    the brim, within HAT_PAINT_SHARE of the head's width under the old cut, takes the kind's colour too: Pixel Match
    painted the brim's dark onto the Zauberer's hairline, and cut up to the hat's opening his new head wore it as a
    plum strip with white specks all round (2026-11-23)."""
    kinds = head_kinds(own, build, weight, points, zone)
    if kinds is None:
        return own
    skin, hair, new_hair, front, seen = kinds["skin"], kinds["hair"], kinds["new_hair"], kinds["front"], kinds["seen"]
    is_hair, have, height = kinds["is_hair"], np.isfinite(own[..., 0]), points[..., 1]
    light = np.nan_to_num(build).sum(axis=-1)
    out = np.array(own)
    hatted = np.zeros(have.shape, bool)
    if hat is not None:
        seen_as = lab(np.nan_to_num(own))
        near_hat = np.linalg.norm(seen_as - lab(hat), axis=-1)
        level = plane_height(zone["plane"], points[..., 0], points[..., 2]) - zone.get("cut", 0.0)
        hatted = (have & (height > level - HAT_PAINT_SHARE * zone["width"])
                  & (near_hat < np.linalg.norm(seen_as - lab(hair), axis=-1))
                  & (near_hat < np.linalg.norm(seen_as - lab(skin), axis=-1)))
    for kind, colour in ((new_hair, hair), (~new_hair & front, skin), (~new_hair & ~front, hair)):
        wrong = have & kind & (is_hair(np.nan_to_num(own)) != (colour is hair))
        if colour is hair and kind is not new_hair:
            wrong = np.zeros(have.shape, bool)       # behind the face the build's skin is not believed (`front`)
        wrong |= hatted & kind
        if wrong.any():
            usual = np.median(light[kind & seen]) if (kind & seen).any() else 1.0
            shade = np.clip(light[wrong] / max(usual, 1e-6), 0.85, 1.15)
            out[wrong] = np.clip(colour[None, :] * shade[:, None], 0, 255)
    # the hat's colour over all the head's own paint in its column is left unknown, so the figure's paint under it is
    # carried up there (`carry_up`): one flat colour stood there as a band
    fine = have & ~hatted
    first = np.where(fine.any(axis=0), fine.argmax(axis=0), own.shape[0])[None, :]
    out[hatted & (np.arange(own.shape[0])[:, None] < first)] = np.nan
    return out


def carry_up(values: np.ndarray) -> np.ndarray:
    """VALUES (a picture of the new head: rows from the top down, columns round, channels; NaN where unknown) with the
    unknown texels over each column's highest known one filled from the known ones under it, mirrored (the band under
    them read upward): the figure's own paint carried up where the new head reaches into the hat over the old head's
    cut. The figure has no surface there, and the build's paint that stood in painted a tan band with streaks right
    under the Zauberer's brim (2026-11-23)."""
    out = np.array(values, dtype=np.float64)
    known = np.isfinite(out[..., 0])
    rows, cols = known.shape
    has = known.any(axis=0)
    top = np.where(has, known.argmax(axis=0), rows)[None, :]
    r = np.arange(rows)[:, None]
    every = np.arange(cols)[None, :]
    source = np.clip(2 * top - 1 - r, 0, rows - 1)
    source = np.where(known[source, every], source, np.minimum(top, rows - 1))
    above = (r < top) & has[None, :]
    out[above] = out[source, every][above]
    return out


def build_colour(points: np.ndarray, build: dict | None, width: float) -> np.ndarray:
    """The build's own paint at the new head's texels (POINTS): at its nearest point on the build's outer skin (BUILD:
    positions, uvs, paint), within BUILD_PAINT_SHARE of the head's WIDTH; NaN elsewhere or without a build."""
    colour = np.full(points.shape, np.nan)
    if build is None:
        return colour
    from scipy.spatial import cKDTree

    flat = colour.reshape(-1, 3)
    distance, nearest = cKDTree(build["positions"]).query(points.reshape(-1, 3), workers=-1)
    near = distance < BUILD_PAINT_SHARE * width
    flat[near] = hf.paint_at(build["paint"], np.asarray(build["uvs"])[nearest[near]])
    return flat.reshape(points.shape)


def tube_paint(points: np.ndarray, normals: np.ndarray, crop_rgb: np.ndarray, crop_alpha: np.ndarray,
               crop: tuple[int, int, int, int], frame, build: dict | None, seen: tuple[np.ndarray, tuple, float] | None,
               width: float) -> np.ndarray:
    """The new head's picture (RGB, 0..255) at its texels' POINTS: the crop (the picture itself, upscaled: the eyes are
    the picture's own pixels) where the head faces the camera, the picture shows it (CROP_ALPHA, the cut-out) and
    nothing the figure keeps stands in front of it (SEEN: the kept figure's front depth, its box and pixel: a brim, a
    collar); the build's own paint elsewhere (BUILD: positions, uvs, its paint picture), moved to the picture's where
    both are seen (`smooth_offsets`) and eased between over the facing. Texels neither reaches stay NaN
    (`figure_paint` fills them from each model's own paint)."""
    rows, cols = points.shape[:2]
    crop_colour, weight = crop_paint(points, normals, crop_rgb, crop_alpha, crop, frame, seen, width)
    crop_colour, weight = crop_colour.reshape(-1, 3), weight.reshape(-1)
    colour = build_colour(points, build, width).reshape(-1, 3)
    known = np.isfinite(colour[:, 0])
    both = known & (weight > 0.5)
    if both.any():          # the build paints its own way (the Zauberer's skin peach where the picture's is pale)
        shift = smooth_offsets((crop_colour - np.nan_to_num(colour)).reshape(rows, cols, 3), both.reshape(rows, cols))
        colour = colour + shift.reshape(-1, 3)
    out = np.where(known[:, None], (1 - weight[:, None]) * np.nan_to_num(colour) + weight[:, None] * crop_colour,
                   crop_colour)
    painted = (known | (weight > 0)).reshape(rows, cols)
    out = np.where(painted[..., None], out.reshape(rows, cols, 3), np.nan)
    return np.clip(out, 0, 255)


def neck_band(points: np.ndarray, zone: dict) -> np.ndarray:
    """Which texels of the new head's picture (POINTS, `tube_points`: rows from the top down) make its neck: in each
    column from the stitch up while the head stays within NECK_STEP_SHARE of its width of the lowest texel's distance
    from the axis, or faces down (a chin's underside goes with the neck), at most NECK_BAND_SHARE of the width over the
    stitch. The picture shows the throat there in the chin's shadow, often the shirt's dark opening under it, and the
    build's paint copies it: on the Zauberer both painted a dark beard under the new chin (2026-11-22). A neck reaching
    down into a collar (`tube_low`) has its band from the stitch up too: under it the picture and the build paint it."""
    _, r = polar(zone["frame"], points.reshape(-1, 3))
    r = r.reshape(points.shape[:2])
    height = points[..., 1]
    up = height >= zone["bottom"] - 0.5 * zone.get("pixel", 0.0)
    base = stitch_rows(up)
    lowest = r[base, np.arange(r.shape[1])][None, :]
    near = np.ones(r.shape, bool)
    near[:-1] = (r[:-1] < lowest + NECK_STEP_SHARE * zone["width"]) | (r[:-1] - r[1:] > height[:-1] - height[1:])
    near &= height - zone["bottom"] < NECK_BAND_SHARE * zone["width"]
    return np.logical_and.accumulate((near | ~up)[::-1], axis=0)[::-1] & up


def stitch_rows(up: np.ndarray) -> np.ndarray:
    """In every column of a picture of the new head (rows from the top down), the lowest row of UP (at or over the
    stitch): the last row when the head ends at the stitch."""
    rows = up.shape[0]
    return np.where(up.any(axis=0), rows - 1 - np.argmax(up[::-1], axis=0), rows - 1)


def figure_paint(picture: np.ndarray, points: np.ndarray, part: dict, zone: dict,
                 crop: tuple[np.ndarray, ...] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """(the new head's picture for one model, the offset its own part got): PICTURE (`tube_paint`, NaN where neither the
    picture nor the build reaches: under the build's own brim, where the figure's own head stands in) moved to the
    model's own tone (`paint_offset`), and its empty texels and its neck (`neck_band`: the stitch then shows no change
    of paint) painted from the model's own old head nearest each one (POINTS, the texels' places), else from the
    nearest painted texel; up the neck that paint eases into the head's just over it. With CROP (`crop_paint`: the
    picture's colour and weight, and the build's own paint, `build_colour`, to tell hair from skin) the head takes the picture only where it sees it, moved to the model's tone read
    there, and round it the model's own paint (`own_paint`); PICTURE stands in only where that has none. A model with
    no paint keeps the picture as it is, those texels filled the same way."""
    band = neck_band(points, zone)
    if crop is None:
        out = np.array(picture, dtype=np.float64)
        out[band] = np.nan
        offset = paint_offset(part, zone, out)
        if offset is not None:
            out = out + np.clip(offset, -hf.PAINT_LIMIT, hf.PAINT_LIMIT)
    else:                   # the picture where it sees the head, the figure's own paint round it (`own_paint`)
        colour, weight = np.asarray(crop[0], dtype=np.float64), np.asarray(crop[1], dtype=np.float64)[..., None]
        # over the old head's cut the picture shows the hat, not the head: its brim's shadow and the band's runes
        # stood there as a dark band with teal flecks over the Zauberer's forehead (2026-11-23)
        level = plane_height(zone["plane"], points[..., 0], points[..., 2]) - zone["cut"]
        weight = weight * np.clip((level - points[..., 1]) / zone["pixel"], 0, 1)[..., None]
        offset = paint_offset(part, zone, np.where(weight > 0.5, colour, np.nan))
        shift = 0.0 if offset is None else np.clip(offset, -hf.PAINT_LIMIT, hf.PAINT_LIMIT)
        raw = np.asarray(crop[2] if len(crop) > 2 else picture, dtype=np.float64)     # the build's own paint
        hat = hat_colour(part, zone) if zone.get("opening") is not None else None
        seen_own = own_paint(part, zone, points)
        skin = figure_skin(seen_own, weight[..., 0], points, zone) if "chin" in zone and "nose" in zone else None
        own = carry_up(same_kind(seen_own, raw, weight[..., 0], points, zone, hat=hat))
        kinds = head_kinds(seen_own, raw, weight[..., 0], points, zone) if skin is not None else None
        if kinds is not None:
            # the new head's skin in the face's colour as the picture paints it, shaded by the build's own paint (its
            # ears' bowls, the jaw's shadow) ...
            height = points[..., 1]
            face = (weight[..., 0] > 0.8) & (height > zone["chin"]) & (height < zone["nose"][1])
            painted = np.median((colour + shift)[face], axis=0) if face.sum() >= 3 else skin
            toned, skin_at = build_skin_paint(raw, kinds, painted, points, zone["width"])
            skin_at &= height < level                   # up in the hat, over the old head's cut, it is hair
            own = np.where(skin_at[..., None], toned, own)
            if zone.get("nose") is not None:
                # ... and on an ear the picture paints no skin: it sees the ear edge-on, and what it shows there is a
                # strand or the brim's shadow (on the Zauberer's right ear, 2026-11-23). Elsewhere on the head's sides
                # it stays: gated on all the skin off the face, a lock laid onto his temple turned skin and the gate's
                # edge stood as a hard line down both temples (2026-11-23)
                weight = weight * (1 - ear_texels(points, zone) * skin_at)[..., None]
        out = np.where(np.isfinite(own[..., :1]), weight * (colour + shift) + (1 - weight) * own,
                       np.asarray(picture, dtype=np.float64) + shift)
        out[band] = np.nan
    empty = ~np.isfinite(out[..., 0])
    paint = hf.part_paint(part)
    if empty.any() and paint is not None and part.get("uvs") is not None:
        from scipy.spatial import cKDTree

        p = np.asarray(part["positions"], dtype=np.float64)
        old = zone_values(zone, p) > -zone["pixel"]
        if old.any():
            distance, nearest = cKDTree(p[old]).query(points[empty], workers=-1)
            near = distance < BUILD_PAINT_SHARE * zone["width"]
            fill = np.full((int(empty.sum()), 3), np.nan)
            fill[near] = hf.paint_at(paint, np.asarray(part["uvs"])[old][nearest[near]])
            out[empty] = fill
    known = np.isfinite(out[..., 0])
    if not known.all() and known.any():
        out = np.stack([fill_nearest(np.nan_to_num(out[..., k]), known) for k in range(3)], axis=-1)
    if crop is not None and skin is not None:
        # beside the jaw and behind it the figure's own neck lies in its collar's dark, and taken onto the new neck, now
        # reaching up to the jaw's edge, it stood under the Zauberer's ears as a black mark (2026-11-23). A neck reaching
        # down into a collar is neck under the stitch too, and the band eases from the paint there: where the picture
        # showed the collar's dark, it stood under his jaw as a black ragged blotch (v19l, 2026-11-24). Ragged, because
        # a texel just under the line was taken and its neighbour just over it kept: the shaded skin eases in instead
        neck = band | (points[..., 1] < zone["bottom"] - 0.5 * zone.get("pixel", 0.0))
        below = lab(skin)[0] - lab(out)[..., 0]
        lift = (np.clip((below - BAND_SOFT) / (BAND_DARK - BAND_SOFT), 0, 1) * neck)[..., None]
        out = (1 - lift) * out + lift * np.asarray(skin, dtype=np.float64) * BAND_SHADE
    # the neck eases from the figure's own paint at the stitch to the head's just over it (a chin's skin), so seen
    # from below the chin's underside is skin, not a block of the figure's shaded throat (the Zauberer, 2026-11-22)
    rows, cols = band.shape
    base = stitch_rows(points[..., 1] >= zone["bottom"] - 0.5 * zone.get("pixel", 0.0))
    top = np.clip(base - band.sum(axis=0), 0, rows - 1)              # the texel just over the neck, column by column
    height = points[..., 1]
    over = height[top, np.arange(cols)][None, :]
    ease = np.clip((height - zone["bottom"]) / np.maximum(over - zone["bottom"], 1e-12), 0, 1)[..., None]
    start = out
    under = np.minimum(base + 1, rows - 1)
    deep = (under > base) & (height[under, np.arange(cols)] < zone["bottom"] - 0.5 * zone.get("pixel", 0.0))
    if deep.any():
        # a neck reaching down into a collar is painted under the stitch by the picture and the build: the band eases
        # from that paint up, not from the figure's old neck, which lies in the collar's shadow. Eased from the old
        # neck, the new one stood under the Zauberer's chin as a pale ring with a hard top edge (2026-11-23)
        start = np.where(deep[None, :, None], out[under, np.arange(cols)][None, :, :], out)
    out = np.where(band[..., None], (1 - ease) * start + ease * out[top, np.arange(cols)][None, :, :], out)
    return np.clip(out, 0, 255), offset


def smooth_offsets(offsets: np.ndarray, known: np.ndarray, share: float = OFFSET_SHARE) -> np.ndarray:
    """OFFSETS (rows, columns, 3) known in KNOWN texels spread smoothly over the whole picture (a picture unrolled
    round the head: its first and last columns neighbours): each texel takes the Gaussian-weighted mean of the known
    ones round it, SHARE of the picture's width wide, and where none lie near, their median."""
    cols = known.shape[1]
    sigma = share * cols
    pad = min(cols, int(3 * sigma))
    weight = np.concatenate([known[:, cols - pad:], known, known[:, :pad]], axis=1).astype(np.float64)
    spread = ndimage.gaussian_filter(weight, sigma, mode=("nearest", "wrap"))[:, pad:pad + cols]
    median = np.median(offsets[known], axis=0)
    out = np.empty_like(offsets, dtype=np.float64)
    for k in range(offsets.shape[-1]):
        value = np.where(known, offsets[..., k], 0.0)
        padded = np.concatenate([value[:, cols - pad:], value, value[:, :pad]], axis=1)
        total = ndimage.gaussian_filter(padded, sigma, mode=("nearest", "wrap"))[:, pad:pad + cols]
        mean = np.divide(total, spread, out=np.zeros_like(total), where=spread > 1e-6)
        sure = np.clip(spread / 0.05, 0, 1)
        out[..., k] = sure * mean + (1 - sure) * median[k]
    return out


def paint_offset(part: dict, zone: dict, picture: np.ndarray) -> np.ndarray | None:
    """The figure's own paint less the new head's PICTURE (RGB, the median) where the old head lay close to the new one
    (zone "radius", the new head's: within
    OFFSET_NEAR_SHARE of the head's width): a model's paint is often lighter or darker than the picture (the Zauberer's
    models 5 to 60 levels, D245), and the new head should keep the figure's tone. Read over the whole head, not at the
    stitch alone: there the picture's throat lies in the collar's shadow. None when the model carries no paint there."""
    paint = hf.part_paint(part)
    if paint is None or part.get("uvs") is None:
        return None
    p = np.asarray(part["positions"], dtype=np.float64)
    inside = zone_values(zone, p) > 0
    if inside.sum() < 3:
        return None
    frame = zone["frame"]
    radius = zone["radius"]
    angle, r = polar(frame, p[inside])
    rows, cols = picture.shape[:2]
    low = tube_low(zone)
    turn = -np.pi + (np.arange(cols) + 0.5) * (2 * np.pi / cols)
    mid = sample_map(frame, radius, turn, np.full(cols, zone["bottom"] + 0.5 * (frame["y_hi"] - zone["bottom"])))
    top = tube_top(zone, turn, mid)
    col = np.floor((angle + np.pi) / (2 * np.pi) * cols).astype(np.int64) % cols
    s = (p[inside, 1] - low) / np.maximum(top[col] - low, 1e-12)
    row = np.clip(np.floor((1 - s) * rows).astype(np.int64), 0, rows - 1)
    new = picture[row, col].astype(np.float64)
    near = ((np.abs(sample_map(frame, radius, angle, p[inside, 1]) - r) < OFFSET_NEAR_SHARE * zone["width"])
            & np.isfinite(new[:, 0]))
    if near.sum() < 3:
        return None
    own = hf.paint_at(paint, np.asarray(part["uvs"])[inside][near])
    return np.median(own - new[near], axis=0)


def zone_picture(zone: dict):
    """The region unrolled round the head, for a human to check: the old outermost surface in grey (lighter is further
    from the axis), the collar in red, the cut under the brim in yellow and the stitch in green."""
    from PIL import Image

    frame = zone["frame"]
    outer = np.where(np.isfinite(zone["outer"]), zone["outer"], zone["limit"])
    grey = 40 + 200 * np.clip(outer / max(zone["reach"], 1e-12), 0, 1)
    rgb = np.stack([grey] * 3, axis=-1)
    rgb[zone["collar"]] = rgb[zone["collar"]] * 0.5 + np.array([200, 40, 40]) * 0.5
    turn = column_angles(frame)
    edge = unpolar(frame, turn, zone["cover"], np.full(len(turn), zone["plane"]["a"]))
    level = plane_height(zone["plane"], edge[:, 0], edge[:, 2]) - zone["cut"]
    row = np.clip(((frame["y_hi"] - level) / frame["pixel"]).astype(np.int64), 0, frame["rows"] - 1)
    rgb[row, np.arange(frame["cols"])] = [255, 220, 0]
    stitch = int(np.clip((frame["y_hi"] - zone["bottom"]) / frame["pixel"], 0, frame["rows"] - 1))
    rgb[stitch] = [40, 220, 60]
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
