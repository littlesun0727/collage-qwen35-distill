"""Convert visible inner corners into the renderer's unrotated rectangle."""
import math

CORNER_NAMES = ('top_left', 'top_right', 'bottom_right', 'bottom_left')


def fit_rectangle(corners):
    if set(corners) != set(CORNER_NAMES):
        raise ValueError('Rectangle needs four named inner corners')
    points = [corners[k] for k in CORNER_NAMES]
    if any(not isinstance(p, list) or len(p) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in p) for p in points):
        raise ValueError('Corners must be finite coordinate pairs')
    edges = [(points[(i+1)%4][0]-p[0], points[(i+1)%4][1]-p[1]) for i,p in enumerate(points)]
    if any(math.hypot(*e)<2 for e in edges) or any(
            e[0]*edges[(i+1)%4][1]-e[1]*edges[(i+1)%4][0] <= 0 for i,e in enumerate(edges)):
        raise ValueError('Corners must form a non-degenerate clockwise convex quadrilateral')
    hx,hy = ((edges[0][i]-edges[2][i])/2 for i in (0,1))
    vx,vy = ((edges[1][i]-edges[3][i])/2 for i in (0,1))
    angle = math.atan2(hy-vx, hx+vy)
    c,s = math.cos(angle),math.sin(angle)
    width,height = hx*c+hy*s,-vx*s+vy*c
    if min(width,height)<2:
        raise ValueError('Inner rectangle has no usable area')
    cx,cy = (sum(p[i] for p in points)/4 for i in (0,1))
    fitted = [[cx+c*x-s*y,cy+s*x+c*y] for x,y in
              [(-width/2,-height/2),(width/2,-height/2),(width/2,height/2),(-width/2,height/2)]]
    residual = max(math.dist(a,b) for a,b in zip(points,fitted))/min(width,height)
    if residual>.3:
        raise ValueError('Corners cannot be represented by a rotated rectangle; strong perspective or wrong corners')
    placement = {'bbox':[round(cx-width/2),round(cy-height/2),round(cx+width/2),round(cy+height/2)],
                 'rotation':round(math.degrees(angle),4)}
    review = {'input_corners':corners,'fitted_corners':fitted,'placement':placement,
              'relative_corner_error':round(residual,6),'warnings':[]}
    if residual>.05:
        review['warnings'].append('Corner fit differs by more than 5% of the short side; inspect perspective or corner estimates')
    return placement,review


def save_preview(run, reviews):
    from PIL import Image, ImageDraw
    with Image.open(run/'prepared/reference.png') as raw:
        image=raw.convert('RGB')
    draw=ImageDraw.Draw(image)
    for row in reviews:
        if 'input_corners' not in row:
            continue
        points=[tuple(row['input_corners'][k]) for k in CORNER_NAMES]
        fit=[tuple(p) for p in row['fitted_corners']]
        draw.line(points+[points[0]],fill='#ff4050',width=3)
        draw.line(fit+[fit[0]],fill='#00cbea',width=2)
        for x,y in points:draw.ellipse((x-4,y-4,x+4,y+4),fill='#ff4050')
        draw.text(points[0],row['photo_id'],fill='#ffff00')
    folder=run/'previews';folder.mkdir(exist_ok=True)
    image.save(folder/'window-geometry.png')
