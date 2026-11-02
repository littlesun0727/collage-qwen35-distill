"""Local crop expansion; never resample a source or ask a model to repair it."""
import math

MIN_PIXEL_COVERAGE = 0.75


def expand_crop(crop, source_size, target_size, *, contain=False):
    """Expand around the chosen region, shifting inward at image boundaries.

    Cover consumes at least target * coverage on BOTH axes after aspect fitting.
    Contain only needs enough pixels for the smaller fitted dimension.
    An undersized original is used as far as possible, without rejection.
    """
    w, h = source_size
    l, t, r, b = crop
    cw, ch = max(1, int(r*w)-int(l*w)), max(1, int(b*h)-int(t*h))
    need_w, need_h = (math.ceil(v*MIN_PIXEL_COVERAGE) for v in target_size)
    factor = (min if contain else max)(need_w/cw, need_h/ch)
    if factor <= 1:
        return list(crop)
    nw, nh = min(w, math.ceil(cw*factor)), min(h, math.ceil(ch*factor))
    x = min(max(0, math.floor((l+r)*w/2-nw/2)), w-nw)
    y = min(max(0, math.floor((t+b)*h/2-nh/2)), h-nh)
    # Include the requested crop even when fractional boundaries round outward.
    return [min(l, x/w), min(t, y/h), max(r, (x+nw)/w), max(b, (y+nh)/h)]
