"""Explicit ruled paper and repeated design motifs; no whole-template reuse."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from overlays import stroke_path


def photo_extent(photo):
    l,t,r,b=photo['bbox'];angle=math.radians(photo.get('rotation',0))
    c,s=abs(math.cos(angle)),abs(math.sin(angle));w,h=r-l,b-t;cx,cy=(l+r)/2,(t+b)/2
    return cx-(c*w+s*h)/2,cy-(s*w+c*h)/2,cx+(c*w+s*h)/2,cy+(s*w+c*h)/2


def draw_patterns(image, patterns, obj, reference, photos):
    notes=[];w,h=image.size
    for spec in patterns:
        kind=spec['kind'];sx,sy=spec['spacing'];px,py=spec['phase']
        # Repeating an oversized intermediate canvas is bounded by the output.
        sx=max(2,sx);sy=max(2,sy)
        angle=math.radians(spec['angle']);c,s=math.cos(angle),math.sin(angle)
        # Rotation about the object's centre; phase is measured in pattern space.
        def point(x,y):return (w/2+c*x-s*y,h/2+s*x+c*y)
        span=math.hypot(w,h)/2+max(sx,sy)*2
        ink=Image.new('RGBA',image.size);draw=ImageDraw.Draw(ink)
        color=spec['color'];width=max(1,round(spec['width']));tile=None
        if kind=='reference_tile':
            box=spec.get('sample_bbox',[])
            if len(box)==4 and 0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1:
                rw,rh=reference.size;l,t,r,b=[round(v*(rw if i%2==0 else rh)) for i,v in enumerate(box)]
                contaminated=any(max(l,box[0])<min(r,box[2]) and max(t,box[1])<min(b,box[3])
                                 for box in map(photo_extent,photos))
                if not contaminated and r-l>=2 and b-t>=2:
                    tile=reference.crop((l,t,r,b)).convert('RGBA').resize((max(2,round(sx)),max(2,round(sy))),Image.Resampling.LANCZOS)
            if tile is None:
                kind=spec.get('fallback_motif','none')
                notes.append('重复纹样取样区域无效或与照片相交，已使用声明的程序纹样兜底')
        if kind in ['lines','grid']:
            dash=spec.get('dash')
            for y in np.arange(-span+py%sy,span,sy):
                stroke_path(draw,[point(-span,y),point(span,y)],color,width,dash,'round' if dash else 'butt')
            if kind=='grid':
                for x in np.arange(-span+px%sx,span,sx):
                    stroke_path(draw,[point(x,-span),point(x,span)],color,width,dash,'round' if dash else 'butt')
        elif kind in ['dots','cherry','reference_tile']:
            motif=spec.get('motif_size',min(sx,sy)*.25)
            rotated=tile.rotate(-spec['angle'],Image.Resampling.BICUBIC,expand=True) if tile is not None else None
            for y in np.arange(-span+py%sy,span,sy):
                for x in np.arange(-span+px%sx,span,sx):
                    xx,yy=point(x,y)
                    if not (-max(sx,sy)<=xx<=w+max(sx,sy) and -max(sx,sy)<=yy<=h+max(sx,sy)):continue
                    if kind=='reference_tile':
                        ink.alpha_composite(rotated,(round(xx-rotated.width/2),round(yy-rotated.height/2)))
                    elif kind=='dots':
                        radius=max(.5,width/2);draw.ellipse((xx-radius,yy-radius,xx+radius,yy+radius),fill=color)
                    else:
                        # A recognisable two-fruit motif, with measured size/phase.
                        radius=motif*.19
                        a=point(x-motif*.2,y+motif*.15);b=point(x+motif*.2,y+motif*.24);top=point(x+motif*.08,y-motif*.35)
                        for fruit in [a,b]:
                            draw.line([fruit,top],fill='#587444',width=max(1,round(motif*.055)))
                            draw.ellipse((fruit[0]-radius,fruit[1]-radius,fruit[0]+radius,fruit[1]+radius),fill=color)
                        leaf=[top,point(x+motif*.38,y-motif*.43),point(x+motif*.24,y-motif*.19)]
                        draw.polygon(leaf,fill='#587444')
        ink.putalpha(ImageChops.multiply(ink.getchannel('A'),image.getchannel('A')))
        image=Image.alpha_composite(image,ink)
    return image,notes
