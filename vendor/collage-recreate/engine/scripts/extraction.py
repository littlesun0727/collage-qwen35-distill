"""Recover reference pixels locally; geometry overlap is a warning, not segmentation."""
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from common import sha, save, event
from photos import cutout


def overlap_area(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def grabcut(im):
    """Fallback when no ONNX weights are provided. Padding keeps edge subjects eligible."""
    import cv2
    rgb=np.asarray(im.convert('RGB'));h,w=rgb.shape[:2]
    if min(w,h)<8:raise ValueError('Reference crop too small for foreground separation')
    border=np.concatenate([rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]])
    pad=max(3,round(min(w,h)*.04));background=np.median(border,axis=0).astype('uint8')
    expanded=np.empty((h+2*pad,w+2*pad,3),dtype='uint8');expanded[:]=background;expanded[pad:pad+h,pad:pad+w]=rgb
    mask=np.zeros(expanded.shape[:2],dtype='uint8');bg=np.zeros((1,65),np.float64);fg=bg.copy()
    cv2.setRNGSeed(42)
    cv2.grabCut(expanded,mask,(pad,pad,w,h),bg,fg,3,cv2.GC_INIT_WITH_RECT)
    alpha=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype('uint8')[pad:pad+h,pad:pad+w]
    result=im.convert('RGBA');result.putalpha(Image.fromarray(alpha));return result


def extract(obj,reference,photos,size,run=None,model=None):
    started=time.monotonic();box=obj['bbox'];style=obj.get('style',{});area=(box[2]-box[0])*(box[3]-box[1])
    overlap=max([overlap_area(box,p['bbox'])/area for p in photos] or [0])
    mode=style.get('extract_mode','auto')
    if mode=='auto':mode='color' if style.get('extract_background') else 'foreground'
    source=reference.crop(box).convert('RGBA');folder=Path(run)/'assets/extractions' if run else None
    if folder:
        folder.mkdir(parents=True,exist_ok=True);source_path=folder/(obj['id']+'-reference.png');source.save(source_path)
        event(run,'extraction_started',object_id=obj['id'],mode=mode,photo_bbox_overlap=round(overlap,4))
    try:
        if mode=='crop':
            if overlap>.03:raise ValueError('Raw crop overlaps photos; use foreground or color separation')
            result=source
        elif mode=='color':
            if not style.get('extract_background'):raise ValueError('Color separation requires extract_background')
            array=np.asarray(source,dtype='float32');key=np.array(ImageColor.getrgb(style['extract_background'])[:3])
            distance=np.max(np.abs(array[:,:,:3]-key),axis=2);tol=style.get('extract_tolerance',35)
            array[:,:,3]*=np.clip((distance-tol)/20,0,1);result=Image.fromarray(array.astype('uint8'))
        elif model and folder:
            result=cutout(source,{'sha256':sha(source_path)},model,folder/'.cache');mode='birefnet'
        else:
            result=grabcut(source);mode='grabcut'
        alpha=np.asarray(result.getchannel('A'));fraction=float(np.mean(alpha>127))
        if fraction<.003 or int(np.count_nonzero(alpha>127))<8:raise ValueError('No usable foreground recovered')
        if mode!='crop' and fraction>.995 and overlap>.03:
            raise ValueError('Separation kept almost the entire crop over photos; contamination unresolved')
        borders=np.concatenate([alpha[0],alpha[-1],alpha[:,0],alpha[:,-1]])
        note='Recovered reference pixels; visually inspect remaining reference person/background and sticker outline'
        if overlap>.03:note+='; photo bbox overlaps, but actual foreground was separated locally'
        metadata={'quality':'needs_review','method':'extract','resource_type':'reference_extraction','extraction_mode':mode,'reference_bbox':box,'photo_bbox_overlap':round(overlap,4),'foreground_fraction':round(fraction,4),'border_foreground_fraction':round(float(np.mean(borders>127)),4),'note':note}
        if folder:
            result.save(folder/(obj['id']+'-foreground.png'));result.getchannel('A').save(folder/(obj['id']+'-mask.png'))
            metadata.update(reference_crop=str(source_path),foreground=str(folder/(obj['id']+'-foreground.png')),mask=str(folder/(obj['id']+'-mask.png')))
            save(folder/(obj['id']+'.json'),metadata);event(run,'extraction_finished',object_id=obj['id'],elapsed_seconds=round(time.monotonic()-started,3),foreground_fraction=round(fraction,4))
        return result.resize(size,Image.Resampling.LANCZOS),metadata
    except Exception as exc:
        if folder:
            save(folder/(obj['id']+'.json'),{'quality':'placeholder','mode':mode,'error':str(exc),'reference_crop':str(source_path)})
            event(run,'extraction_failed',object_id=obj['id'],elapsed_seconds=round(time.monotonic()-started,3),error=str(exc))
        raise ValueError('Local reference separation failed: '+str(exc)) from exc


def sheets(run,records):
    """One batch contact sheet, alongside whole-composition review; no unit review loop."""
    selected=[r for r in records if r['metadata'].get('reference_crop')]
    from text import font_file
    paths=[];font=ImageFont.truetype(font_file({'font':'sans'},''),16)
    for start in range(0,len(selected),6):
        sheet=Image.new('RGB',(1040,900),'#dddddd');d=ImageDraw.Draw(sheet)
        for index,r in enumerate(selected[start:start+6]):
            x=(index%2)*520;y=(index//2)*300;d.text((x+8,y+8),r['id'],font=font,fill='black')
            for j,key in enumerate(['reference_crop','foreground']):
                with Image.open(r['metadata'][key]) as im:tile=im.convert('RGBA');tile.thumbnail((248,248))
                left=x+8+j*256;top=y+38
                for cy in range(0,248,16):
                    for cx in range(0,248,16):
                        color='#eeeeee' if (cx//16+cy//16)%2 else '#bbbbbb'
                        d.rectangle((left+cx,top+cy,left+min(cx+15,247),top+min(cy+15,247)),fill=color)
                sheet.paste(tile,(left+(248-tile.width)//2,top+(248-tile.height)//2),tile.getchannel('A'))
        path=Path(run)/'previews'/f'extractions-{start//6+1}.jpg';sheet.save(path,quality=92);paths.append(str(path))
    return paths
