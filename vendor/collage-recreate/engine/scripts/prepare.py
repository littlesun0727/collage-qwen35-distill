from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
from common import save, sha, now, timed

EXTENSIONS = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}


def prepare(reference, materials, run, width=1200, instructions='', cutout_model=None):
    run=Path(run).resolve(); reference=Path(reference).resolve()
    if run.exists(): raise ValueError('prepare requires a new task directory; use existing input.json to continue')
    sources=[Path(p).resolve() for p in materials]
    if any(run.is_relative_to(p) for p in sources) or reference.is_relative_to(run):
        raise ValueError('Task outputs must be outside source directories')
    if not 128 <= width <= 4096: raise ValueError('Output width must be 128..4096')
    files=sorted({p.resolve() for root in sources for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS})
    if not files: raise ValueError('No customer photos found')
    run.mkdir(parents=True); prepared=run/'prepared';prepared.mkdir()
    with timed(run,'prepare'):
        with Image.open(reference) as raw:
            ref=ImageOps.exif_transpose(raw).convert('RGB')
        ref.save(prepared/'reference.png')
        assets=[]; warnings=[]; seen=set()
        for path in files:
            try:
                digest=sha(path)
                if digest in seen: continue
                with Image.open(path) as raw: im=ImageOps.exif_transpose(raw).convert('RGB')
                # Compact task-local IDs; full hashes still handle deduplication and integrity.
                aid=f'material_{len(assets):02d}';thumb=im.copy();thumb.thumbnail((260,220))
                thumb.save(prepared/(aid+'.jpg'))
                assets.append({'id':aid,'file':str(path),'sha256':digest,'size':list(im.size),'thumbnail':str(prepared/(aid+'.jpg'))})
                seen.add(digest)
            except (OSError,ValueError) as exc: warnings.append({'file':str(path),'error':str(exc)})
        if not assets: raise ValueError('No decodable customer photos')
        from text import font_file
        sheets=[];font=ImageFont.truetype(font_file({'font':'sans'},''),15)
        for offset in range(0,len(assets),12):
            sheet=Image.new('RGB',(1120,810),'#eeeeee');d=ImageDraw.Draw(sheet)
            for i,a in enumerate(assets[offset:offset+12]):
                x=(i%4)*280;y=(i//4)*270
                with Image.open(a['thumbnail']) as im:sheet.paste(im,(x+(280-im.width)//2,y+5))
                d.text((x+8,y+230),a['id'],font=font,fill='black')
                d.text((x+8,y+251),str(a['size']),font=font,fill='black')
            p=prepared/f'contact-sheet-{offset//12+1}.jpg';sheet.save(p);sheets.append(str(p))
        catalog={'assets':assets,'contact_sheets':sheets,'warnings':warnings};save(prepared/'catalog.json',catalog)
        metadata={'schema_version':'collage-input-v1','created_at':now(),'reference':{'file':str(prepared/'reference.png'),'sha256':sha(prepared/'reference.png')},'original_reference':{'file':str(reference),'sha256':sha(reference)},'reference_size':list(ref.size),'output_width':width,'materials':[str(p) for p in sources],'instructions':instructions,'cutout_model':str(Path(cutout_model).resolve()) if cutout_model else None}
        save(run/'input.json',metadata)
    return {'run':str(run),'reference_size':list(ref.size),'reference':str(prepared/'reference.png'),'contact_sheets':sheets,'asset_count':len(assets),'warnings':warnings}


def boxes(run, analysis):
    run=Path(run)
    with Image.open(run/'prepared/reference.png') as ref: im=ref.convert('RGB')
    from text import font_file
    d=ImageDraw.Draw(im);font=ImageFont.truetype(font_file({'font':'sans'},''),max(13,im.width//65))
    colors={'background':'#964b00','photo':'#00bb00','frame':'#aa33cc','text':'#ff2222','overlay':'#0077ff'}
    for o in analysis['objects']:
        box=o['bbox'];color=colors[o['kind']]
        if o.get('rotation',0):
            import math
            l,t,r,b=box;cx,cy=(l+r)/2,(t+b)/2
            angle=math.radians(o['rotation']);c,s=math.cos(angle),math.sin(angle)
            points=[(cx+(x-cx)*c-(y-cy)*s,cy+(x-cx)*s+(y-cy)*c)
                    for x,y in [(l,t),(r,t),(r,b),(l,b)]]
            d.line(points+[points[0]],fill=color,width=max(2,im.width//500))
        else:
            d.rectangle(box,outline=color,width=max(2,im.width//500))
        d.text((box[0]+2,box[1]+2),o['id'],fill=color,font=font,stroke_width=1,stroke_fill='white')
    path=run/'previews/analysis-boxes.png';path.parent.mkdir(exist_ok=True);im.save(path)
    return str(path)
