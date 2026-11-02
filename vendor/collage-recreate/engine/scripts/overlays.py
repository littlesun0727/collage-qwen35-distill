"""Inexpensive local primitives and conservative reference extraction."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
from text import render_text
from paths import smooth_points


def intersect(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def stroke_path(draw,points,color,width,dash=None,cap='butt'):
    """One phase across all segments; round caps are a drawing primitive."""
    phase=0.;period=sum(dash) if dash else None
    for p,q in zip(points,points[1:]):
        dx,dy=q[0]-p[0],q[1]-p[1];length=math.hypot(dx,dy)
        if not length:continue
        at=0.
        while at<length-1e-8:
            active=not dash or phase<dash[0]
            step=length-at if not dash else min(length-at,(dash[0] if active else period)-phase)
            if step<1e-8:phase=0.;continue
            if active:
                a=(p[0]+dx*at/length,p[1]+dy*at/length);b=(p[0]+dx*(at+step)/length,p[1]+dy*(at+step)/length)
                draw.line([a,b],fill=color,width=width)
                if cap=='round':
                    radius=width/2
                    for x,y in [a,b]:draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
            at+=step
            if dash:phase=(phase+step)%period


def frame_image(size,style,color,width):
    """Closed, inward outline; radius is a fraction of the outer short side."""
    factor=1 if style.get('dash') and not style.get('radius') else 4;w,h=size
    width=min(width,min(size));radius=min(min(size)/2,min(size)*style.get('radius',0))
    im=Image.new('RGBA',(w*factor,h*factor));d=ImageDraw.Draw(im)
    if not style.get('dash'):
        d.rounded_rectangle((0,0,w*factor-1,h*factor-1),radius=radius*factor,
                            outline=color,width=max(1,round(width*factor)))
    else:
        left=top=width*factor/2;right=max(left,w*factor-1-left);bottom=max(top,h*factor-1-top)
        r=max(0,min(radius*factor-width*factor/2,(right-left)/2,(bottom-top)/2))
        if r:
            points=[]
            for cx,cy,start in [(right-r,top+r,-90),(right-r,bottom-r,0),
                                (left+r,bottom-r,90),(left+r,top+r,180)]:
                for angle in np.linspace(start,start+90,max(3,math.ceil(r/2))):
                    a=math.radians(angle);points.append((cx+r*math.cos(a),cy+r*math.sin(a)))
            points.append(points[0])
        else:points=[(left,top),(right,top),(right,bottom),(left,bottom),(left,top)]
        stroke_path(d,points,color,max(1,round(width*factor)),[v*factor for v in style['dash']],style.get('line_cap','butt'))
    return im.resize(size,Image.Resampling.LANCZOS)


def draw_overlay(obj,size,reference=None,photos=(),run=None,model=None):
    style=obj.get('style',{});method=obj.get('method','local');note=''
    if method=='extract':
        from extraction import extract
        try:return extract(obj,reference,photos,size,run,model)
        except ValueError as exc:note=str(exc);method='placeholder'
    shape=style.get('shape')
    if obj['kind']=='background':shape=shape or 'rectangle'
    if method=='placeholder' or not shape:
        im=Image.new('RGBA',size,(240,230,245,95));d=ImageDraw.Draw(im)
        d.rectangle((0,0,size[0]-1,size[1]-1),outline='#9B59B6',width=2)
        d.line((0,0,*size),fill='#9B59B6',width=1)
        d.line((0,size[1],size[0],0),fill='#9B59B6',width=1)
        return im,{'quality':'placeholder','method':'placeholder','note':note or 'Complex overlay not yet produced; select for generation if important'}
    w,h=size;im=Image.new('RGBA',size);d=ImageDraw.Draw(im)
    fill=style.get('fill','#F0E4C8' if shape in ['paper','tape'] else '#FFFFFF')
    stroke=style.get('stroke');sw=max(1,round(style.get('stroke_width',2)));pad=sw/2 if stroke else 0
    box=(pad,pad,max(pad,w-1-pad),max(pad,h-1-pad));rng=np.random.default_rng(style.get('seed',42))
    if shape in ['rectangle','paper','tape']:
        if shape in ['paper','tape'] and style.get('edge','straight')=='torn':
            amp=max(1,min(w,h)*.025);xs=np.linspace(0,w-1,18);ys=np.linspace(0,h-1,18)
            points=[(float(x),float(rng.uniform(0,amp))) for x in xs]+[(float(w-1-rng.uniform(0,amp)),float(y)) for y in ys]+[(float(x),float(h-1-rng.uniform(0,amp))) for x in xs[::-1]]+[(float(rng.uniform(0,amp)),float(y)) for y in ys[::-1]]
            d.polygon(points,fill=fill)
        else:d.rectangle(box,fill=fill,outline=stroke,width=sw)
    elif shape=='rounded_rectangle':d.rounded_rectangle(box,radius=min(w,h)*style.get('radius',.12),fill=fill,outline=stroke,width=sw)
    elif shape=='ellipse':d.ellipse(box,fill=fill,outline=stroke,width=sw)
    elif shape=='frame':
        width=sw if 'stroke_width' in style else max(1,round(min(w,h)*.06))
        im=frame_image(size,style,stroke or fill,width)
    elif shape=='paperclip':
        points=[(.7,.82),(.7,.2),(.67,.12),(.6,.08),(.4,.08),(.33,.12),(.3,.2),(.3,.86),(.35,.94),(.45,.97),(.6,.97),(.8,.9),(.85,.8),(.85,.25),(.8,.18),(.7,.15),(.55,.15),(.48,.2),(.48,.73)]
        stroke_path(d,[(x*(w-1),y*(h-1)) for x,y in points],stroke or fill,sw,cap='round')
    elif shape=='rays':
        for angle in np.linspace(0,2*math.pi,style.get('sides',10),endpoint=False):
            points=[(w/2+math.cos(angle)*w*.5*r,h/2+math.sin(angle)*h*.5*r) for r in [style.get('inner_ratio',.45),.92]]
            stroke_path(d,points,stroke or fill,sw,cap=style.get('line_cap','round'))
    elif shape=='heart':
        points=[]
        for t in np.linspace(0,2*math.pi,120):
            x=16*math.sin(t)**3;y=13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t)
            points.append(((x+17)/34*(w-1),(13-y)/30*(h-1)))
        d.polygon(points,fill=fill)
    elif shape=='star':
        n=style.get('sides',5);inner=style.get('inner_ratio',.45)
        points=[(w/2+(w/2-pad)*(.99 if i%2==0 else inner)*math.sin(i*math.pi/n),h/2-(h/2-pad)*(.99 if i%2==0 else inner)*math.cos(i*math.pi/n)) for i in range(n*2)]
        d.polygon(points,fill=fill)
    elif shape in ['line','arrow','polyline','polygon','curve']:
        points=[(p[0]*(w-1),p[1]*(h-1)) for p in style.get('points',[[.05,.5],[.95,.5]])]
        if shape=='curve':
            factor=4;large=Image.new('RGBA',(w*factor,h*factor));ld=ImageDraw.Draw(large)
            sampled=smooth_points([(x*factor,y*factor) for x,y in points])
            stroke_path(ld,sampled,stroke or fill,max(1,round(sw*factor)),[v*factor for v in style['dash']] if style.get('dash') else None,style.get('line_cap','round'))
            im=large.resize((w,h),Image.Resampling.LANCZOS);d=ImageDraw.Draw(im)
        elif shape=='polygon':d.polygon(points,fill=fill)
        else:
            stroke_path(d,points,stroke or fill,sw,style.get('dash'),style.get('line_cap','butt'))
            if shape=='arrow':
                x,y=points[-1];px,py=points[-2];a=math.atan2(y-py,x-px);length=min(w,h)*.45
                d.polygon([(x,y),(x-length*math.cos(a-.6),y-length*math.sin(a-.6)),(x-length*math.cos(a+.6),y-length*math.sin(a+.6))],fill=stroke or fill)
    texture=style.get('texture','paper' if shape=='paper' else 'none')
    if texture in ['paper','grain']:
        a=np.asarray(im).copy();noise=rng.normal(0,3 if texture=='paper' else 7,(h,w,1));a[:,:,:3]=np.clip(a[:,:,:3].astype(float)+noise,0,255).astype('uint8');im=Image.fromarray(a)
    elif texture in ['dots','stripes','grid']:
        pattern=Image.new('RGBA',size);pd=ImageDraw.Draw(pattern);step=max(8,round(min(w,h)/10));ink=stroke or '#FFFFFF'
        for y in range(step//2,h,step):
            if texture=='dots':
                for x in range(step//2,w,step):pd.ellipse((x-step*.1,y-step*.1,x+step*.1,y+step*.1),fill=ink)
            else:pd.line((0,y,w,y),fill=ink,width=1)
        if texture=='grid':
            for x in range(step//2,w,step):pd.line((x,0,x,h),fill=ink,width=1)
        pattern.putalpha(Image.fromarray((np.asarray(pattern.getchannel('A'),dtype='float32')*np.asarray(im.getchannel('A'))/255).astype('uint8')));im=Image.alpha_composite(im,pattern)
    if style.get('patterns'):
        from patterns import draw_patterns
        im,pattern_notes=draw_patterns(im,style['patterns'],obj,reference,photos)
        if pattern_notes:note='；'.join(pattern_notes)
    if style.get('window'):
        from effects import rounded_mask
        window=style['window'];left,top,right,bottom=window['bbox']
        if not (0<=left<right<=1 and 0<=top<bottom<=1):
            raise ValueError('window.bbox must be a positive normalized rectangle')
        left,top,right,bottom=round(left*w),round(top*h),round(right*w),round(bottom*h)
        opening=Image.new('L',size)
        inner=(max(1,right-left),max(1,bottom-top))
        opening.paste(rounded_mask(inner,min(inner)*window.get('radius',0)),(left,top))
        im.putalpha(ImageChops.multiply(im.getchannel('A'),ImageChops.invert(opening)))
    if obj.get('text'):
        ts={**style,'fill':style.get('stroke','#171717')};lettering,_=render_text(obj['text'],(max(1,round(w*.88)),max(1,round(h*.8))),ts)
        im.alpha_composite(lettering,((w-lettering.width)//2,(h-lettering.height)//2))
    return im,{'quality':'approximate','method':'local','shape':shape,'note':note or 'Local primitive; visual similarity requires review'}
