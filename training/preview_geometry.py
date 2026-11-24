"""Photo outline primitives from the collage-recreate v10 review renderer."""
import math

import numpy as np


def _heart_outline(steps=80):
    """A soft-notched Bezier heart with a small, rounded directional tip.

    Smooth tangent joins replace the separate circular lobes and broad bottom
    cap. Both rendering and axis fitting consume this same normalized outline.
    Keep the 240-point contract: notch at 0, bottom tip at 120, clockwise.
    """
    segments = [
        ((.5, .18), (.535, .18), (.57, 0.), (.76, 0.)),
        ((.76, 0.), (.91, 0.), (1., .11), (1., .28)),
        ((1., .28), (1., .56), (.68, .84), (.525, .98)),
        ((.525, .98), (.515, .989), (.507, 1.), (.5, 1.)),
    ]
    t = np.linspace(0., 1., steps, endpoint=False)[:, None]
    half = np.vstack([
        (1-t)**3*np.array(a) + 3*(1-t)**2*t*np.array(b)
        + 3*(1-t)*t**2*np.array(c) + t**3*np.array(d)
        for a, b, c, d in segments
    ] + [np.array([[.5, 1.]])])
    along = np.concatenate([[0.], np.cumsum(np.hypot(*np.diff(half, axis=0).T))])
    marks = np.linspace(0., along[-1], 121)
    right = np.column_stack([np.interp(marks, along, half[:, i]) for i in (0, 1)])
    left = right[-1:0:-1].copy()
    left[:, 0] = 1. - left[:, 0]
    points = np.vstack([right[:-1], left])
    low, high = points.min(0), points.max(0)
    return [tuple(p) for p in (points-low)/(high-low)]


def unit_outline(kind, star_points=5, inner_ratio=.5):
    """Renderer outline in the unit box (y downward). Drawing and tilt fitting both use it."""
    if kind == 'star':
        count = star_points
        return [(.5+.5*(1 if i%2 == 0 else inner_ratio)*math.sin(i*math.pi/count),
                 .5-.5*(1 if i%2 == 0 else inner_ratio)*math.cos(i*math.pi/count)) for i in range(count*2)]
    if kind == 'heart':
        return _heart_outline()
    if kind == 'ellipse':
        t = np.linspace(0, 2*math.pi, 360, endpoint=False)
        return list(zip(.5+.5*np.cos(t), .5+.5*np.sin(t)))
    raise ValueError('No unit outline for shape ' + str(kind))


def rounded_quad_outline(points, radii):
    """Circular fillets in pixel space. A zero radius marks a straight clipping corner."""
    pts = np.asarray(points, dtype=float)
    prev = np.roll(pts, 1, axis=0) - pts
    following = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(following, axis=1)
    if np.min(lengths) < 1e-6:
        return pts.tolist()
    prev /= np.roll(lengths, 1)[:, None]
    following /= lengths[:, None]
    half_angles = np.arccos(np.clip(np.sum(prev * following, axis=1), -1, 1)) / 2
    short_side = min((lengths[0] + lengths[2]) / 2, (lengths[1] + lengths[3]) / 2)
    radius = short_side * np.asarray(radii, dtype=float)
    tangents = radius / np.maximum(np.tan(half_angles), 1e-8)
    # Two adjacent arcs must fit on their shared edge, including the semicircle of a capsule.
    scale = min(1., float(np.min(lengths / np.maximum(tangents + np.roll(tangents, -1), 1e-8))))
    points_out = []
    for i, vertex in enumerate(pts):
        if radius[i] <= 0:
            points_out.append(tuple(vertex))
            continue
        t = tangents[i] * scale
        center = vertex + (prev[i] + following[i]) * t / (2 * max(np.cos(half_angles[i])**2, 1e-12))
        a = vertex + prev[i] * t - center
        b = vertex + following[i] * t - center
        start = math.atan2(a[1], a[0])
        sweep = math.atan2(a[0]*b[1] - a[1]*b[0], np.dot(a, b))
        # Bound the chord error at render resolution. Also retain a useful contour for overlap tests.
        r = radius[i] * scale
        max_step = min(math.pi / 8, 2 * math.acos(max(-1., 1 - .2 / max(r, .2))))
        count = max(2, math.ceil(abs(sweep) / max_step))
        angles = np.linspace(start, start + sweep, count + 1)
        points_out.extend(zip(center[0] + r * np.cos(angles), center[1] + r * np.sin(angles)))
    return points_out


def _finite_point(p):
    return (isinstance(p, (list, tuple)) and len(p) == 2 and all(
        not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in p))


def _wrap(angle):
    """Degrees into [-180, 180)."""
    return (angle + 180) % 360 - 180


def axis_angle(shape, axis):
    """Clockwise rotation (degrees, y down) implied by the two axis points; 0 = unrotated.

    heart: notch -> tip points down when unrotated. star: center -> top tip points up.
    ellipse: only the long-axis direction matters, folded into [-45, 45).
    """
    if len(axis) != 2 or not all(_finite_point(p) for p in axis):
        raise ValueError('axis needs two finite coordinate pairs')
    (x0, y0), (x1, y1) = axis
    dx, dy = x1 - x0, y1 - y0
    if math.hypot(dx, dy) < 2:
        raise ValueError('axis points are too close to define a direction')
    angle = math.degrees(math.atan2(dy, dx))
    if shape == 'heart':
        return _wrap(angle - 90)
    if shape == 'star':
        return _wrap(angle + 90)
    if shape == 'ellipse':
        return (angle + 45) % 90 - 45
    raise ValueError('axis does not apply to shape ' + str(shape))


def _solve_box(outline, angle, box, fixed=None):
    """Find (cx, cy, w, h) whose rotated outline has the given horizontal bounding box.

    Each side of the bounding box is set by one outline vertex, so four linear equations in
    (cx, cy, w, h) hold once the extreme vertices are known. Those vertices depend only weakly on
    (w, h); iterate until they stop changing. `fixed` pins one dimension (ellipse long axis)
    where the box alone cannot separate w from h.
    """
    pts = np.asarray(outline, float) - .5
    u, v = pts[:, 0], pts[:, 1]
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    left, top, right, bottom = box
    w, h = max(right-left, 2.), max(bottom-top, 2.)
    last = None
    for _ in range(60):
        x = c*w*u - s*h*v
        y = s*w*u + c*h*v
        ia, ib, ic, idx = int(x.argmin()), int(x.argmax()), int(y.argmin()), int(y.argmax())
        if (ia, ib, ic, idx) == last:
            break
        last = (ia, ib, ic, idx)
        rows = [[c*(u[ib]-u[ia]), -s*(v[ib]-v[ia])], [s*(u[idx]-u[ic]), c*(v[idx]-v[ic])]]
        rhs = [right-left, bottom-top]
        if fixed:
            rows.append([1., 0.] if fixed[0] == 'w' else [0., 1.])
            rhs.append(fixed[1])
        solution = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        w, h = float(solution[0]), float(solution[1])
        if w < 2 or h < 2:
            raise ValueError('Shape axis and bounding box do not describe a usable shape')
    x = c*w*u - s*h*v
    y = s*w*u + c*h*v
    cx = (left - x.min() + right - x.max())/2
    cy = (top - y.min() + bottom - y.max())/2
    fitted = [cx + x.min(), cy + y.min(), cx + x.max(), cy + y.max()]
    short = max(min(right-left, bottom-top), 1.)
    residual = max(abs(a-b) for a, b in zip(fitted, box))/short
    return (cx, cy, w, h), residual


def window_outline(shape, box, size, *, corners, points, axis, radius, star_points):
    """Return a closed window's vertices in display pixels."""
    left, top, right, bottom = box
    quad = corners or [(left, top), (right, top), (right, bottom), (left, bottom)]
    if shape == 'rectangle':
        if not radius:
            return quad
        radii = [radius] * 4
        if corners:
            tolerance = max(.004 * min(size), 2)
            for i, (p, q) in enumerate(zip(quad, quad[1:] + quad[:1])):
                if any(max(abs(p[d] - edge), abs(q[d] - edge)) <= tolerance
                       for d, extent in enumerate(size) for edge in (0, extent)):
                    radii[i] = radii[(i + 1) % 4] = 0
        return rounded_quad_outline(quad, radii)
    if shape == 'other':
        return points
    if shape not in ('circle', 'ellipse', 'heart', 'star'):
        return []
    unit = unit_outline('ellipse' if shape == 'circle' else shape, star_points)
    w, h = right - left, bottom - top
    cx, cy = (left + right) / 2, (top + bottom) / 2
    angle = 0
    if shape == 'circle':
        w = h = min(w, h)
    elif axis or shape == 'star':
        try:
            angle = axis_angle(shape, axis) if axis else 0
            if abs(angle) < .5:
                angle = 0
            fixed = None
            if shape == 'ellipse' and axis:
                dx, dy = np.subtract(axis[1], axis[0])
                along = math.degrees(math.atan2(dy, dx))
                fixed = ('w' if abs(_wrap(2 * (along - angle))) < 90 else 'h', math.dist(*axis))
            (cx, cy, w, h), residual = _solve_box(unit, angle, box, fixed)
            if residual > .3:
                return []
        except ValueError:
            return []
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    local = [((u - .5) * w, (v - .5) * h) for u, v in unit]
    return [(cx + c*x - s*y, cy + s*x + c*y) for x, y in local]
