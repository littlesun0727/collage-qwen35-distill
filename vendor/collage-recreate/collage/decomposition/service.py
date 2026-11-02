"""Dataset-specific assembly; transport is shared with the v9 engine."""
import json
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collage import contracts  # registers the engine module path
from collage.storage import read, save
from reveal import fetch_batch, reserve_submission


def submit_and_fetch(run, key, timeout):
    analysis = read(run/'input/analysis.json')
    objects = analysis['objects']
    if not 1 <= len(objects) <= 20 or any(o['kind'] == 'background' for o in objects):
        raise ValueError('Reveal requires 1-20 foreground objects')
    plan = {'object_ids': [o['id'] for o in objects],
            'background': {'id': 'layer_0', 'source': 'api_output_index_0'}}
    if (run/'request.json').exists() and read(run/'request.json') != plan:
        raise ValueError('Batch changed; use a new run directory')
    save(run/'request.json', plan)
    task = run/'raw'
    budget = run.parents[1]
    reserve_submission(task, budget, limit=read(budget/'submission-budget.json')['limit'])
    for attempt in range(3):
        try:
            fetch_batch(task, run/'input/reference.png', objects, key, timeout)
            break
        except PermissionError:
            # Resume only a known submitted task; never risk a second submission
            # if saving the submission receipt itself failed.
            if attempt == 2 or not (task/'submit_response.json').exists():
                raise
            time.sleep(.5 * (attempt + 1))


def assemble(run):
    analysis=read(run/'input/analysis.json');response=read(run/'raw/query_response.json');out=response['output']
    plan=read(run/'request.json')
    by_id={o['id']:o for o in analysis['objects']}
    objects=[by_id[oid] for oid in plan['object_ids']];mapping=out['boxes_mapping_index']
    assert len(mapping)==len(set(mapping)) and all(type(i) is int and 0<=i<len(objects) for i in mapping)
    ids=[objects[i]['id'] for i in mapping]
    order=[oid for oid in analysis['layer_order'] if oid in ids]
    original=Image.open(run/'input/reference.png').convert('RGB')
    manifest={'task_id':response['task_id'],'model':'ecommerce_layer','version':'v1.1','requested_count':len(objects),'returned_count':len(mapping),'missing_ids':[o['id'] for o in objects if o['id'] not in ids],'reference_size':list(original.size),'layer_order':order,'generation_seconds':response.get('generation_time'),'charge_amount':response.get('charge_amount'),'variants':{}}
    manifest.update(schema_version='reveal-lab-result-v2',background=plan.get('background',{'id':'layer_0','source':'api_output_index_0','replaces_input_ids':[]}),stack_order=['layer_0']+order,total_layer_count=1+len(mapping))
    for variant in ['base','aug']:
        urls=out.get('layers_'+variant,[])
        if not urls:continue
        if len(urls)!=len(mapping)+1:raise RuntimeError('Unexpected background/mapping count; cannot safely assume layer indices')
        dest=run/'layers'/variant;dest.mkdir(parents=True,exist_ok=True)
        bg=Image.open(run/f'raw/layers_{variant}_00.png').convert('RGBA');bg.save(dest/'layer_0.png')
        images={};entries=[]
        for j,index in enumerate(mapping,1):
            obj=objects[index];im=Image.open(run/f'raw/layers_{variant}_{j:02d}.png').convert('RGBA')
            assert im.size==bg.size
            im.save(dest/(obj['id']+'.png'));images[obj['id']]=im
            alpha=im.getchannel('A');bounds=alpha.getbbox()
            entries.append({'id':obj['id'],'kind':obj['kind'],'description':obj['description'],'requested_bbox':obj['bbox'],'visible_bbox':bounds,'alpha_range':list(alpha.getextrema()),'visible_fraction':round(float((np.asarray(alpha)>0).mean()),6),'file':f'layers/{variant}/{obj["id"]}.png'})
        composite=bg.copy();only=Image.new('RGBA',bg.size)
        for oid in order:
            composite=Image.alpha_composite(composite,images[oid]);only=Image.alpha_composite(only,images[oid])
        composite.save(run/f'reconstructed-{variant}.png');only.save(run/f'objects-only-{variant}.png')
        # Compare with the service's resized input if available, avoiding JPEG/resize ambiguity.
        target_path=run/'raw/resized_image.png'
        reference=Image.open(target_path).convert('RGB') if target_path.exists() else original.resize(bg.size,Image.Resampling.LANCZOS)
        assert reference.size==bg.size
        error=np.abs(np.asarray(composite.convert('RGB'),dtype=float)-np.asarray(reference,dtype=float))
        difference=Image.fromarray(np.clip(error*4,0,255).astype('uint8'));difference.save(run/f'difference-{variant}-4x.png')
        comparison=Image.new('RGB',(bg.width*3,bg.height));comparison.paste(reference,(0,0));comparison.paste(composite,(bg.width,0));comparison.paste(difference,(bg.width*2,0));comparison.save(run/f'comparison-{variant}.jpg',quality=95)
        manifest['variants'][variant]={'size':list(bg.size),'background_file':f'layers/{variant}/layer_0.png','background_layer':{'id':'layer_0','kind':'background','file':f'layers/{variant}/layer_0.png'},'layers':entries,'mean_absolute_rgb_error':round(float(error.mean()),4),'fraction_pixels_with_error_over_20':round(float((error.max(axis=2)>20).mean()),4)}
        entries_for_sheet=[{'id':'layer_0','visible_bbox':bg.getchannel('A').getbbox()}]+entries
        images['layer_0']=bg
        sheet=Image.new('RGB',(1000,((len(entries_for_sheet)+3)//4)*260),'#f4f4ef');pen=ImageDraw.Draw(sheet)
        try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
        except OSError:font=ImageFont.load_default()
        for n,entry in enumerate(entries_for_sheet):
            x=(n%4)*250;y=(n//4)*260
            for yy in range(y+6,y+228,12):
                for xx in range(x+8,x+240,12):
                    pen.rectangle([xx,yy,min(xx+11,x+239),min(yy+11,y+227)],fill='#aaaeb0' if ((xx-x)//12+(yy-y)//12)%2 else '#e9e9e9')
            im=images[entry['id']];bounds=entry['visible_bbox']
            if bounds:
                thumb=im.crop(bounds);thumb.thumbnail((220,210));sheet.paste(thumb,(x+125-thumb.width//2,y+117-thumb.height//2),thumb)
            pen.text((x+8,y+235),entry['id'],font=font,fill='#233e31')
        sheet.save(run/f'contact-sheet-{variant}.png')
    save(run/'manifest.json',manifest)
    create_html(run,manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='variants'},ensure_ascii=False),flush=True)

def create_html(run,manifest):
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reveal 整图拆解实验</title><style>*{box-sizing:border-box}body{margin:24px;background:#eeeee8;color:#263c33;font:14px system-ui}h1{font-size:24px}button,select{padding:8px;font:inherit}.controls{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}.compare{display:grid;grid-template-columns:1fr 1fr 260px;gap:20px;align-items:start}.frame{position:relative;background:conic-gradient(#bbb 25%,#eee 0 50%,#bbb 0 75%,#eee 0) 0 0/20px 20px;aspect-ratio:3/4}.frame img{width:100%;display:block}.frame .layer{position:absolute;inset:0;height:100%}.list label{display:block;padding:8px;background:#fff;margin:3px 0}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-top:24px}.card{background:#fff;padding:12px}.card canvas{max-width:100%;background:conic-gradient(#bbb 25%,#eee 0 50%,#bbb 0 75%,#eee 0) 0 0/16px 16px}.card p{font-size:12px;overflow-wrap:anywhere}a{color:#287559}.hint{color:#68776c;line-height:1.7}@media(max-width:800px){.compare{grid-template-columns:1fr 1fr}.list{grid-column:1/-1}}</style><h1>整图拆解 → 图层复原</h1><div class="hint">背景直接采用接口返回的 layer_0，不再提交 photo_base 框。其余照片、文字和装饰按 JSON 顺序叠加。左侧为原图，右侧为复原结果，可逐层开关。</div><div class="controls"><select id="variant"><option value="base">base 原始层</option><option value="aug">aug 增强层</option></select><label><input id="residual" type="checkbox" checked>显示背景 layer_0</label><button id="all">显示全部元素</button><button id="none">隐藏全部元素</button></div><div class="compare"><section><h3>原图</h3><div class="frame"><img src="input/reference.png"></div></section><section><h3>图层叠合</h3><div class="frame" id="stack"></div></section><aside><h3>图层开关</h3><div id="list" class="list"></div><p id="stats" class="hint"></p><a href="manifest.json">实验结果 JSON</a></aside></div><h2>独立元素 · 按透明边界预览</h2><div class="gallery" id="gallery"></div><script>const manifest=__MANIFEST__;let generation=0;const visible=new Set(manifest.layer_order);function draw(){const gen=++generation,v=document.querySelector('#variant').value,data=manifest.variants[v],stack=document.querySelector('#stack'),list=document.querySelector('#list'),gallery=document.querySelector('#gallery');stack.innerHTML=list.innerHTML=gallery.innerHTML='';function layer(src,show){const im=new Image();im.src=src;im.className='layer';im.style.display=show?'block':'none';stack.append(im)}layer(data.background_file,document.querySelector('#residual').checked);const bgCard=document.createElement('div');bgCard.className='card';const bgImg=new Image();bgImg.src=data.background_file;bgImg.style.width='100%';const bgText=document.createElement('p');bgText.textContent='layer_0 · 背景';const bgLink=document.createElement('a');bgLink.href=data.background_file;bgLink.textContent='查看背景层';bgCard.append(bgImg,bgText,bgLink);gallery.append(bgCard);for(const id of manifest.layer_order){const item=data.layers.find(l=>l.id===id);layer(item.file,visible.has(id));const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=visible.has(id);check.onchange=()=>{check.checked?visible.add(id):visible.delete(id);draw()};label.append(check,document.createTextNode(' '+id));list.prepend(label);const card=document.createElement('div');card.className='card';const c=document.createElement('canvas');c.width=220;c.height=220;const im=new Image();im.onload=()=>{if(gen!==generation)return;const b=item.visible_bbox;if(!b)return;const w=b[2]-b[0],h=b[3]-b[1],s=Math.min(210/w,210/h);c.getContext('2d').drawImage(im,b[0],b[1],w,h,(220-w*s)/2,(220-h*s)/2,w*s,h*s)};im.src=item.file;const text=document.createElement('p');text.textContent=id+' · '+item.kind;const link=document.createElement('a');link.href=item.file;link.textContent='查看完整透明图层';card.append(c,text,link);gallery.append(card)}document.querySelector('#stats').textContent=`请求 ${manifest.requested_count} 个前景元素，返回 ${manifest.returned_count} 个，加背景 layer_0 共 ${manifest.total_layer_count} 层；服务输出 ${data.size.join(' × ')}。RGB 平均绝对误差 ${data.mean_absolute_rgb_error} / 255（仅用于比较复原，不代表分层质量）。`}document.querySelector('#variant').onchange=draw;document.querySelector('#residual').onchange=draw;document.querySelector('#all').onclick=()=>{manifest.layer_order.forEach(id=>visible.add(id));draw()};document.querySelector('#none').onclick=()=>{visible.clear();draw()};draw();</script></html>'''
    if 'base' not in manifest['variants']:
        page = page.replace('<option value="base">base ???</option>', '')
    page=page.replace('aspect-ratio:3/4',f'aspect-ratio:{manifest["reference_size"][0]}/{manifest["reference_size"][1]}')
    (run/'index.html').write_text(page.replace('__MANIFEST__',json.dumps(manifest,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')

