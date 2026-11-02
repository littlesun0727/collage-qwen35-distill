"""Explicit mask edits. Optional model workers never run on normal admission."""
import hashlib
import os
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageColor, ImageFilter
from common import ROOT, read, save, sha, fingerprint, verify_source


def selection(size, edit):
    mask=Image.new('L',size);d=ImageDraw.Draw(mask)
    if 'bbox' in edit:
        l,t,r,b=map(round,edit['bbox']);d.rectangle((l,t,r-1,b-1),fill=255)
    else:d.polygon([tuple(map(round,p)) for p in edit['polygon']],fill=255)
    return mask


def tools_config(run):
    path=Path(run)/'repair-tools.json'
    if not path.exists():path=ROOT/'repair-tools.local.json'
    return read(path) if path.exists() else {}


def backend_signature(config, kind):
    fields=['python','ocr_det','ocr_rec'] if kind=='ocr' else ['python','lama_model']
    files={}
    for field in fields:
        value=config.get(field)
        if not value:raise RuntimeError('Local repair tool not configured: '+field)
        p=Path(value)
        if not p.exists():raise RuntimeError('Local repair tool missing: '+field)
        files[field]={str(f.relative_to(p)) if p.is_dir() else f.name:sha(f)
                      for f in (sorted(p.rglob('*')) if p.is_dir() else [p]) if f.is_file()}
    return fingerprint([files,config.get('environment_id'),sha(ROOT/'scripts/repair_worker.py')])


def worker(run, kind, image, mask=None):
    config=tools_config(run);signature=backend_signature(config,kind)
    key=fingerprint([kind,signature,hashlib.sha256(image.tobytes()).hexdigest(),
                     hashlib.sha256(mask.tobytes()).hexdigest() if mask else None])
    folder=Path(run)/'assets/repairs'/key;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json'
    if receipt.exists():
        data=read(receipt)
        if data.get('key')==key:
            verify_source(data['output'])
            return read(data['output']['file']) if kind=='ocr' else Image.open(data['output']['file']).convert('RGB')
    image.save(folder/'input.png')
    if mask:mask.save(folder/'mask.png')
    job={'kind':kind,'image':str(folder/'input.png'),'mask':str(folder/'mask.png'),
         'output':str(folder/('output.json' if kind=='ocr' else 'output.png')),'config':config}
    save(folder/'job.json',job)
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    env=dict(os.environ);env.update(PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True',HF_HUB_OFFLINE='1',
                                  PADDLE_PDX_CACHE_HOME=str(folder/'paddle-cache'))
    result=subprocess.run([config['python'],'-X','utf8',str(ROOT/'scripts/repair_worker.py'),str(folder/'job.json')],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180,env=env,creationflags=flags)
    (folder/'worker.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode or not Path(job['output']).is_file():raise RuntimeError('Local '+kind+' worker failed; see '+str(folder/'worker.log'))
    save(receipt,{'key':key,'output':{'file':job['output'],'sha256':sha(job['output'])}})
    return read(job['output']) if kind=='ocr' else Image.open(job['output']).convert('RGB')


def text_mask(image, allowed, records, edit):
    def norm(t):return ''.join(t.split()).casefold()
    expected={norm(t) for t in edit.get('texts',[])};found=set();mask=Image.new('L',image.size)
    selected=[]
    for record in records:
        value=norm(record['text'])
        if record['confidence']<.8 or (not edit.get('all_text') and value not in expected):continue
        poly=selection(image.size,{'polygon':record['polygon']})
        # Require the whole OCR region within the authorized selection, never cut a neighbouring word.
        if ImageChops.subtract(poly,allowed).getbbox():continue
        mask=ImageChops.lighter(mask,poly);found.add(value);selected.append(record)
    if expected-found:raise RuntimeError('Expected text not confidently detected: '+', '.join(sorted(expected-found)))
    if not selected:raise RuntimeError('No confident text in authorized selection')
    pad=round(edit.get('padding',2))
    if pad:mask=mask.filter(ImageFilter.MaxFilter(pad*2+1))
    return ImageChops.multiply(mask,allowed),selected


def apply_edits(run, image, oid, plan):
    result=image.copy();records=[];errors=[]
    for number,edit in enumerate(e for e in plan.get('edits',[]) if e['id']==oid):
        before=result.copy();mask=selection(image.size,edit);op=edit['operation'];record={'edit':edit}
        try:
            if not mask.getbbox():raise RuntimeError('Empty repair selection')
            if op=='remove_text':
                # OCR sees a neutral matte instead of arbitrary RGB under transparent pixels.
                rgb=Image.new('RGB',result.size,'white');rgb.paste(result,mask=result.getchannel('A'))
                mask,detected=text_mask(result,mask,worker(run,'ocr',rgb),edit)
                record['detected_text']=detected
            if op=='erase':result.putalpha(ImageChops.multiply(result.getchannel('A'),ImageChops.invert(mask)))
            else:
                if op=='fill' or (op=='remove_text' and edit.get('backend')=='fill'):
                    replacement=Image.new('RGB',result.size,ImageColor.getrgb(edit['fill'])[:3])
                else:
                    rgb=Image.new('RGB',result.size,'white');rgb.paste(result,mask=result.getchannel('A'))
                    replacement=worker(run,'lama',rgb,mask)
                    if replacement.size!=result.size:raise RuntimeError('Inpainting changed canvas size')
                alpha=result.getchannel('A');rgb=Image.composite(replacement,result.convert('RGB'),mask)
                result=rgb.convert('RGBA');result.putalpha(alpha)
            changed=np.any(np.asarray(before)!=np.asarray(result),axis=2)
            if np.any(changed & (np.asarray(mask)==0)):raise RuntimeError('Repair changed pixels outside selection')
            if op!='erase' and before.getchannel('A').tobytes()!=result.getchannel('A').tobytes():raise RuntimeError('Repair changed carrier alpha')
            folder=Path(run)/'assets/repairs/masks';folder.mkdir(parents=True,exist_ok=True)
            file=folder/f'{oid}-{number}.png';mask.save(file)
            record.update(status='applied',mask={'file':str(file),'sha256':sha(file)},changed_pixels=int(changed.sum()))
        except (RuntimeError,ValueError,OSError,subprocess.TimeoutExpired) as exc:
            result=before;errors.append(str(exc));record.update(status='failed',error=str(exc))
        records.append(record)
    return result,records,errors
