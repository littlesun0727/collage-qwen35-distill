"""Fixed font selection and measured text; no per-task drawing scripts."""
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont

FONT_ROOT=Path(os.environ.get('COLLAGE_FONT_ROOT', 'C:/Windows/Fonts'))
FONTS={'sans':'arial','serif':'times','mono':'cour','hand':'comic','cjk':'msyh','cjk-hand':'simkai'}


def font_file(style,text):
    name=style.get('font','cjk' if any(ord(c)>255 for c in text) else 'sans')
    if any(ord(c)>255 for c in text) and name not in ['cjk','cjk-hand']:name='cjk'
    base=FONTS[name]
    if name=='cjk':name_file='msyhbd.ttc' if style.get('bold') else 'msyh.ttc'
    elif name=='cjk-hand':name_file='simkai.ttf'
    else:
        suffix=('bi' if style.get('bold') and style.get('italic') else 'bd' if style.get('bold') else 'i' if style.get('italic') else '')
        if base=='comic' and suffix=='bi':suffix='z'
        name_file=base+suffix+'.ttf'
    path=FONT_ROOT/name_file
    if not path.exists():raise ValueError(f'Font unavailable: {path}; configure COLLAGE_FONT_ROOT for installed fonts')
    return str(path)


def render_text(text,size,style):
    """Fit explicit lines to the box, spreading spare height between the lines."""
    text=text.replace('\r\n','\n').replace('\r','\n');lines=text.split('\n')
    out=Image.new('RGBA',size);d=ImageDraw.Draw(out)
    path=font_file(style,text);stroke=round(style.get('stroke_width',0))
    spacing_ratio=style.get('line_spacing',1);align=style.get('align','center')
    requested=style.get('font_size');hi=max(1,min(2000,round(requested or 2000)));lo=1;chosen=None
    while lo<=hi:
        px=(lo+hi)//2;f=ImageFont.truetype(path,px)
        # Font ascent + descent is its single-spaced line box; using only px
        # can make ascenders and descenders touch in handwriting/CJK fonts.
        advance=sum(f.getmetrics())*spacing_ratio
        bounds=[d.textbbox((0,0),line or 'Ag',font=f,anchor='ls',stroke_width=stroke) for line in lines]
        widths=[b[2]-b[0] if line else 0 for line,b in zip(lines,bounds)]
        top=min(b[1]+i*advance for i,b in enumerate(bounds))
        bottom=max(b[3]+i*advance for i,b in enumerate(bounds))
        width=max(widths);height=bottom-top
        if width<=size[0] and height<=size[1]:
            chosen=(f,bounds,widths,top,width,height,advance,px);lo=px+1
        else:hi=px-1
    if chosen is None:raise ValueError('Text cannot fit even at 1px')
    f,bounds,widths,top,width,height,advance,px=chosen
    spacing_mode='explicit' if 'line_spacing' in style else 'single'
    if len(lines)>1 and 'line_spacing' not in style:
        # Width can limit the font before the paragraph fills its height. Use
        # that spare space between lines, keeping standard single spacing as
        # the minimum. A common glyph envelope keeps every line inside the box.
        ink_height=max(b[3] for b in bounds)-min(b[1] for b in bounds)
        advance=max(advance,(size[1]-ink_height)/(len(lines)-1))
        top=min(b[1]+i*advance for i,b in enumerate(bounds))
        height=max(b[3]+i*advance for i,b in enumerate(bounds))-top
        spacing_mode='fit_bbox'
    x=(size[0]-width)/2 if align=='center' else 0 if align=='left' else size[0]-width
    y=(size[1]-height)/2-top
    for i,(line,b,line_width) in enumerate(zip(lines,bounds,widths)):
        offset=(width-line_width)/2 if align=='center' else 0 if align=='left' else width-line_width
        d.text((x+offset-b[0],y+i*advance),line,font=f,anchor='ls',fill=style.get('fill','#171717'),
               stroke_width=stroke,stroke_fill=style.get('stroke','#FFFFFF'))
    return out,{'font':path,'font_size_px':px,'requested_font_size_px':requested,
                'line_count':len(lines),'line_spacing':round(advance/sum(f.getmetrics()),4),
                'line_spacing_mode':spacing_mode,'line_advance_px':round(advance,3),
                'align':align,'fit_scale':round(px/requested,4) if requested else None,
                'text':text,'text_rendered':True,'font_match':'approximate'}
