"""Export extracted alpha geometry and visual comparisons."""
import html
import json
import math
import base64
from io import BytesIO
from PIL import Image
from collage.storage import save
from . import service as reveal
VERSION = "layer-dataset-v2-chain"


def write_svg(run, analysis, records):
    """Embed separate cropped PNG elements, draggable in browsers and editors."""
    width, height = analysis['reference_size']
    objects = {o['id']: o for o in analysis['objects']}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    for ident in analysis['layer_order']:
        box = objects[ident]['bbox']
        l, t, r, b = box
        with Image.open(run/records[ident]['full_layer']) as layer:
            crop = layer.crop((l, t, r, b))
            if 'file' in records[ident]:
                crop.save(run/records[ident]['file'])
            buffer = BytesIO(); crop.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        label = html.escape(ident, quote=True)
        parts.append(f'<g id="{label}" inkscape:groupmode="layer" inkscape:label="{label}" class="asset"><title>{label}</title><image x="{l}" y="{t}" width="{r-l}" height="{b-t}" href="data:image/png;base64,{encoded}"/></g>')
    parts.append('''<style>.asset{cursor:move}svg{touch-action:none}</style>
<script><![CDATA[
const svg=document.documentElement;let drag=null;
function point(e){return new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse())}
svg.addEventListener('pointerdown',e=>{const g=e.target.closest('.asset');if(!g)return;
const p=point(e);drag={g,p,x:Number(g.dataset.x||0),y:Number(g.dataset.y||0)};svg.setPointerCapture(e.pointerId);e.preventDefault()});
svg.addEventListener('pointermove',e=>{if(!drag)return;const p=point(e),x=drag.x+p.x-drag.p.x,y=drag.y+p.y-drag.p.y;
drag.g.dataset.x=x;drag.g.dataset.y=y;drag.g.setAttribute('transform',`translate(${x} ${y})`)});
for(const event of ['pointerup','pointercancel','lostpointercapture'])svg.addEventListener(event,()=>drag=null);
]]></script></svg>''')
    (run/'reference.svg').write_text('\n'.join(parts), encoding='utf-8')

def export_dataset(run):
    manifest=reveal.read(run/'manifest.json')
    if 'aug' not in manifest.get('variants', {}):
        raise ValueError('Export requires aug layers')
    manifest['variants'] = {'aug': manifest['variants']['aug']}
    save(run/'manifest.json', manifest)
    analysis=reveal.read(run/('source-analysis.json' if (run/'source-analysis.json').exists() else 'analysis.json'))
    by_id={x['id']:x for x in analysis['objects']}
    geometry=reveal.read(run/'geometry-review.json') if (run/'geometry-review.json').exists() else {}
    source=Image.open(run/'input/reference.png').convert('RGB')
    records=[];sections=[];empty=[]
    for variant,data in manifest['variants'].items():
        width,height=data['size'];folder=run/'assets'/variant;folder.mkdir(parents=True,exist_ok=True)
        for ident in manifest['stack_order']:
            image=Image.open(run/f'layers/{variant}/{ident}.png').convert('RGBA')
            box=image.getchannel('A').getbbox()
            if box is None and ident == 'layer_0':
                box = (0, 0, width, height)
            if box is None:
                empty.append({'id':ident,'variant':variant});continue
            crop=folder/f'{ident}.png';image.crop(box).save(crop)
            records.append({'id':ident,'variant':variant,'kind':by_id.get(ident,{}).get('kind','background'),'file':str(crop.relative_to(run)).replace('\\','/'),'full_layer':f'layers/{variant}/{ident}.png','canvas_size':[width,height],'crop_bbox':list(box),'source_object':by_id.get(ident),'z_index':manifest['stack_order'].index(ident)})
    compare=run/'comparison';compare.mkdir(exist_ok=True)
    for record in records:
        record['geometry_review'] = geometry.get(f"{record['variant']}/{record['id']}", {})
    for ident in manifest['stack_order']:
        ims=[Image.open(run/f'layers/{v}/{ident}.png').convert('RGBA') for v in ('base','aug') if v in manifest['variants']]
        boxes=[im.getchannel('A').getbbox() for im in ims];boxes=[b for b in boxes if b]
        if not boxes:continue
        box=(min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes))
        w,h=ims[0].size
        original_box=(math.floor(box[0]*source.width/w),math.floor(box[1]*source.height/h),math.ceil(box[2]*source.width/w),math.ceil(box[3]*source.height/h))
        crops=[source.crop(original_box)]+[im.crop(box) for im in ims]
        types=['original']+[v for v in ('base','aug') if v in manifest['variants']]
        cells=[]
        for name,crop in zip(types,crops):
            filename=f'{ident}-{name}.png';crop.save(compare/filename)
            cells.append(f'<div class="cell"><p>{name} · {crop.width} × {crop.height} <a href="{filename}" target="_blank" rel="noopener">打开原图</a></p><div class="view"><img style="--pixel-width:{box[2]-box[0]}px;--pixel-height:{box[3]-box[1]}px" src="{filename}"></div></div>')
        sections.append(f'<h2>{html.escape(ident)}</h2><div class="row">'+''.join(cells)+'</div>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原图 / base / aug</title><style>body{background:#222;color:white;font:14px system-ui;margin:24px}.row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.cell{min-width:0}.view{display:flex;align-items:center;justify-content:center;height:420px;overflow:hidden;background:conic-gradient(#aaa 25%,#ddd 0 50%,#aaa 0 75%,#ddd 0) 0 0/16px 16px}img{display:block;width:var(--pixel-width);height:var(--pixel-height);max-width:100%;max-height:100%;object-fit:contain}.pixels .view{display:block;overflow:auto}.pixels img{max-width:none;max-height:none}a{color:#9df}.cell a{float:right}label{display:block;margin:18px 0}@media(max-width:700px){body{margin:12px}.row{grid-template-columns:1fr}.view{height:360px}}</style><h1>原图 crop / base / aug</h1><p>默认完整显示素材，三列对应同一区域。勾选 100% 可检查细节；打开原图可查看文件原始分辨率。透明图层不包含原背景。</p><a href="../index.html">整图叠合预览</a><label><input type="checkbox" onchange="document.body.classList.toggle('pixels',this.checked)"> 按接口像素 100% 显示（大图可滚动）</label>'''+''.join(sections)
    (compare/'index.html').write_text(page,encoding='utf-8')
    save(run/'dataset.json',{'schema_version':VERSION,'reference':'input/reference.png','analysis':'analysis.json','manifest':'manifest.json','missing_ids':manifest['missing_ids'],'empty_layers':empty,'visual_review':'pending','records':records})
    (run/'dataset.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    return len(records)

