"""Shared masks/effects. Symmetric padding preserves the object's rotation pivot."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops


def rounded_mask(size, radius):
    factor=4;w,h=size
    mask=Image.new('L',(w*factor,h*factor))
    ImageDraw.Draw(mask).rounded_rectangle((0,0,w*factor-1,h*factor-1),radius=max(0,min(radius,min(size)/2))*factor,fill=255)
    return mask.resize(size,Image.Resampling.LANCZOS)


def clip_round(im,radius):
    result=im.copy();result.putalpha(ImageChops.multiply(im.getchannel('A'),rounded_mask(im.size,radius)))
    return result


def expanded_outline(im,width,color):
    if width<=0:return im,0
    pad=math.ceil(width)+2
    out=Image.new('RGBA',(im.width+2*pad,im.height+2*pad));out.alpha_composite(im,(pad,pad))
    radius=max(1,round(width))
    alpha=out.getchannel('A').filter(ImageFilter.MaxFilter(radius*2+1))
    backing=Image.new('RGBA',out.size,color)
    backing.putalpha(ImageChops.multiply(backing.getchannel('A'),alpha))
    return Image.alpha_composite(backing,out),pad


def photo_layout(size,style):
    card=style.get('card')
    if card is None:return (0,0,*size)
    w,h=size
    padding=card.get('padding',[min(size)*.05,min(size)*.05,min(size)*.18,min(size)*.05])
    top,right,bottom,left=[round(v) for v in padding]
    if min(padding)<0 or left+right>=w or top+bottom>=h:
        raise ValueError('card.padding leaves no photo window')
    return left,top,w-right,h-bottom


def photo_card(im,size,style):
    card=style.get('card')
    if card is None:return im
    out=Image.new('RGBA',size,card.get('fill','#F5F2E9'))
    if card.get('radius',0):out=clip_round(out,card['radius'])
    left,top,_,_=photo_layout(size,style);out.alpha_composite(im,(left,top))
    return out


def shadow_layer(alpha,spec):
    """Canvas-space offset: shadow never contributes to customer-photo visibility."""
    blur=spec.get('blur',8);dx,dy=map(round,spec.get('offset',[3,5]))
    mask=alpha.filter(ImageFilter.GaussianBlur(blur)) if blur else alpha
    shifted=Image.new('L',alpha.size);shifted.paste(mask,(dx,dy))
    shadow=Image.new('RGBA',alpha.size,spec.get('color','#00000040'))
    shadow.putalpha(ImageChops.multiply(shadow.getchannel('A'),shifted))
    return shadow


def resolve_style(obj):
    style=dict(obj.get('style',{}));appearance=obj.get('appearance')
    if appearance=='rounded_photo':style.setdefault('corner_radius',min(obj['bbox'][2]-obj['bbox'][0],obj['bbox'][3]-obj['bbox'][1])*.08)
    if appearance=='polaroid':
        style.setdefault('card',{});style.setdefault('shadow',{})
    return style


def scale_style(style,scale):
    from copy import deepcopy
    s=deepcopy(style)
    for key in ['font_size','stroke_width','outline_width','corner_radius']:
        if key in s:s[key]*=scale
    if 'dash' in s:s['dash']=[v*scale for v in s['dash']]
    for pattern in s.get('patterns',[]):
        for key in ['spacing','phase','dash']:
            if key in pattern:pattern[key]=[v*scale for v in pattern[key]]
        for key in ['width','motif_size']:
            if key in pattern:pattern[key]*=scale
    if 'card' in s:
        if 'padding' in s['card']:s['card']['padding']=[v*scale for v in s['card']['padding']]
        if 'radius' in s['card']:s['card']['radius']*=scale
    if 'shadow' in s:
        # Resolve defaults before scaling, including omitted components.
        s['shadow']={'offset':[v*scale for v in s['shadow'].get('offset',[3,5])],
                     'blur':s['shadow'].get('blur',8)*scale,'color':s['shadow'].get('color','#00000040')}
    return s
