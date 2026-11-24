"""The back picture for a front + back build: what Qwen is asked, and what it is shown.

Qwen-Image draws the back in edit mode from the front picture. Two rules found the hard
way (2026-10-10 to 10-12, eight characters):

- **Name only what you want.** Naming a thing summons it, so the prompt never says what to
  leave out. That cuts both ways: naming parts summons them even when the subject has none
  (2026-10-29), so the prompt names no parts at all, only that the back holds what the
  reference holds, with the same outline. It works for any subject that way.
- **Show the subject on plain grey.** The reference is the front's cut-out laid on flat
  light grey, so Qwen redraws the subject and not a backdrop, and the result cuts out
  cleanly.

Same pose, size and place in the frame matter as much as the drawing: Pixal3D reads the
two pictures through cameras exactly half a turn apart.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

BACK_PROMPT = (
    "The same subject as in the reference image, seen from directly behind: the camera has "
    "moved 180 degrees around it. Same pose, same proportions, same size and same position "
    "in the frame. The view shows the back of exactly the parts that are in the reference "
    "and ends exactly where the reference ends. The outline matches the reference: the same "
    "width at every height, and every part at the same height and the same size as in the "
    "reference. Keep the colours, materials and design consistent with the reference. Flat "
    "even lighting, plain light grey background."
)

# The side for a four-view build (`pixal3d_generate.py --right --left`), by the same rules: no part
# is named, only the turn, the way the subject faces, and that its whole depth shows. A front and a
# back picture say nothing about how long a subject is from front to back (an animal, a vehicle),
# so a build from them alone comes out short or doubles what it cannot place (a second pair of
# ears, a third pair of legs); a side picture says it. Tested 2026-11-15 on two pets, a person and a
# van against two other wordings: "seen from its right side, the camera moved 90 degrees" drew a fox
# almost from the front and the van at three quarters; asking for the side view of a turnaround
# sheet gave a true profile for all four, and kept a round creature round.
SIDE_PROMPT = (
    "The side view from a turnaround sheet of the same subject as in the reference image: seen "
    "exactly from its right side, in strict profile, its front pointing to the right edge of the "
    "picture and its rear to the left edge, its whole length from front to rear visible, nothing of "
    "its front turned to the camera. Same pose, same proportions, same size and same height in the "
    "frame. Keep the colours, materials and design consistent with the reference. Flat even "
    "lighting, plain light grey background."
)

# A picture that already shows its subject from the side (a vehicle drawn side-on, as Create asks for
# one) has its profile as the "front": asked for its right side, the engine draws that same profile
# again, and the four-view build takes the subject's length for its depth too. The scissor lift of
# 2026-11-24 came out with a near-square footprint that way. Its turned view is then its end.
END_PROMPT = (
    "The front view from a turnaround sheet of the same subject as in the reference image, which "
    "shows it from its side: seen exactly head-on from its front end, its front facing the camera, "
    "its length running straight away from the camera so that only its width shows, nothing of its "
    "side visible. Same proportions, same size and same height in the frame. Keep the colours, "
    "materials and design consistent with the reference. Flat even lighting, plain light grey "
    "background."
)

# Two outlines this alike (`same_view`) are one view drawn twice, not a turn.
SAME_VIEW = 0.85
SILHOUETTE = 96         # outlines are compared at this many pixels square
GROUND_STEP = 30        # a pixel this far from the ground's colour (any channel) belongs to the subject

# The backdrop the reference is laid on: light grey, like the backgrounds our prompts ask for.
GREY = 236


def mirror_side(right: Image.Image) -> Image.Image:
    """The left side of a subject that is the same on both sides: its right side, mirrored.

    Drawing both sides lets them disagree on the subject's length (two sides of one van came out
    508 and 691 px long, 2026-11-13); a mirror cannot.
    """
    return right.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def reference_on_grey(cut_out: Image.Image) -> Image.Image:
    """The cut-out front composited on plain grey, as an RGB picture for Qwen to read."""
    rgba = cut_out.convert("RGBA")
    backdrop = Image.new("RGBA", rgba.size, (GREY, GREY, GREY, 255))
    return Image.alpha_composite(backdrop, rgba).convert("RGB")


def silhouette(picture: Image.Image) -> tuple[np.ndarray, float] | None:
    """The subject's outline in PICTURE, cropped to its box and scaled to SILHOUETTE square, with
    the box's width over its height; None when nothing stands out. A cut-out's alpha says what the
    subject is; otherwise what differs from the ground, the border's median colour, by GROUND_STEP."""
    if picture.mode in ("RGBA", "LA") and np.asarray(picture.getchannel("A")).min() < 128:
        mask = np.asarray(picture.getchannel("A")) > 127
    else:
        rgb = np.asarray(picture.convert("RGB"), dtype=np.int16)
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        mask = np.abs(rgb - np.median(border, axis=0)).max(axis=-1) > GROUND_STEP
    if mask.sum() < 16:
        return None
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    box = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    square = Image.fromarray(box.astype(np.uint8) * 255).resize((SILHOUETTE, SILHOUETTE), Image.Resampling.NEAREST)
    return np.asarray(square) > 127, box.shape[1] / box.shape[0]


def same_view(a: Image.Image, b: Image.Image) -> float:
    """How alike two pictures' outlines are, 0 to 1: the overlap of their silhouettes (B as it is or
    mirrored, whichever fits better) times how alike their boxes' width-to-height ratios are. A side
    drawn from a front that already showed the side scores near 1; a true turn scores low."""
    found_a, found_b = silhouette(a), silhouette(b)
    if found_a is None or found_b is None:
        return 0.0
    (mask_a, ratio_a), (mask_b, ratio_b) = found_a, found_b

    def overlap(m):
        return (mask_a & m).sum() / max((mask_a | m).sum(), 1)
    return float(max(overlap(mask_b), overlap(mask_b[:, ::-1])) * min(ratio_a, ratio_b) / max(ratio_a, ratio_b))
