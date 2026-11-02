"""Deterministic manual editor commands. Never invokes agents or extraction."""
import argparse
import json
from pathlib import Path
import shutil
import time
from common import read,save,sha,fingerprint,locked,verify_source
from scene import load_scene
from editor_scene import apply_edits,capabilities,transform
from render import render
from revision import folder,commit_revision,recover_commit


def copy_run(run,candidate):
    shutil.copytree(run,candidate,ignore=shutil.ignore_patterns('chat','observability','.write.lock','events.jsonl','sdk-timing.json','editor_layers'))


def editor_document(candidate):
    scene=read(candidate/'scene.json');result=read(candidate/'result.json');caps=capabilities(scene)
    layers={r['id']:r for r in result['editor_layers']};objects=[]
    for identifier in scene['layer_order']:
        if identifier not in layers:continue
        obj=next(o for o in scene['objects'] if o['id']==identifier)
        crop_source=None
        if caps[identifier]['crop']:
            from PIL import Image,ImageOps
            from photos import cutout
            with Image.open(verify_source(obj['source'])) as raw:source=ImageOps.exif_transpose(raw).convert('RGBA')
            if obj.get('mode')=='cutout':
                photo_record=next(r for r in result['resources'] if r['id']==identifier)
                source=cutout(source,obj['source'],scene.get('cutout_model'),Path(photo_record['file']).parent/'.cache')
                source=source.crop(source.getchannel('A').point(lambda v:255 if v>8 else 0).getbbox())
            crop_source=candidate/'editor_layers'/(identifier+'-crop-source.png')
            source.thumbnail((1400,1400));source.save(crop_source)
        objects.append({'id':identifier,'label':obj.get('label',identifier),'kind':obj['kind'],
                        'bbox':obj['bbox'],'rotation':obj.get('rotation',0),'transform':transform(obj),'text':obj.get('text',''),
                        'source_crop':obj.get('binding',{}).get('source_crop',[0,0,1,1]),
                        'window_crop':obj.get('window_crop',[0,0,1,1]),
                        'crop_source_file':str(crop_source) if crop_source else None,
                        'capabilities':caps[identifier],'file':layers[identifier].get('unclipped_file',layers[identifier]['file'])})
    return {'reference_size':scene['reference_size'],'canvas_size':scene['canvas_size'],
            'objects':objects,'final_file':str(candidate/'final.png')}


def execute(run,action,request):
    run=Path(run);root=folder(run,request['request_id']);candidate=root/'candidate'
    if action not in ['export','preview','save']:raise ValueError('Unknown editor action')
    if action=='save' and (root/'committed.json').exists():
        return commit_revision(run,request['request_id'],None,manual=True)
    marker=read(run/'chat/commit.json') if (run/'chat/commit.json').exists() else {}
    if marker.get('status')=='promoting':
        if marker.get('request_id')!=request['request_id']:raise ValueError('另一轮保存需要先恢复')
        recover_commit(run)
    if action=='save' and marker.get('request_id')==request['request_id'] and marker.get('status')=='done':
        return commit_revision(run,request['request_id'],None,manual=True)
    result=read(run/'result.json');scene=load_scene(run)
    if result.get('render_id')!=request['base_render_id'] or sha(run/'scene.json')!=request['base_scene_sha256']:
        raise ValueError('基础版本已变化，请保留草稿并重新载入最新成图')
    saved=read(root/'candidate.json') if (root/'candidate.json').exists() else {}
    if saved and saved.get('plan_sha256')!=fingerprint(request):raise ValueError('请求编号已被其他修改使用')
    if not saved:
        edited=apply_edits(scene,request['changes'],request.get('layer_order')) if action!='export' else scene
        if candidate.exists():
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe editor candidate')
            shutil.rmtree(candidate)
        root.mkdir(parents=True,exist_ok=True);started=time.monotonic()
        if action=='export':
            # Export reads immutable asset files in place; no full task/cache copy.
            candidate.mkdir()
            for name in ['analysis.json','bindings.json','photo-windows.json','input.json','prepared/catalog.json','result.json','final.png','reveal-config.json']:
                source=run/name
                if source.exists():
                    target=candidate/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        else:copy_run(run,candidate)
        save(candidate/'scene.json',edited);save(root/'plan.json',request)
        copied=time.monotonic()
        render(candidate,publish_result=False,capture_layers=action!='save',editor_export=action=='export')
        save(root/'editor-timing.json',{'copy_seconds':round(copied-started,3),'render_seconds':round(time.monotonic()-copied,3)})
        saved={'base_render_id':result['render_id'],'plan_sha256':fingerprint(request),'origin':'manual',
               'final_sha256':sha(candidate/'final.png'),'scene_sha256':sha(candidate/'scene.json')}
        save(root/'candidate.json',saved)
    if action=='save':return commit_revision(run,request['request_id'],None,manual=True)
    return editor_document(candidate)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--action',required=True)
    parser.add_argument('--request',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    with locked(Path(args.run)):
        result=execute(Path(args.run),args.action,read(args.request))
        save(Path(args.output),result)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'error':str(exc)},ensure_ascii=True));raise SystemExit(1)
