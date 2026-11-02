"""First-preview pipeline with optional Reveal extraction and yibu generation."""
import argparse
import json
import os
import sys
from pathlib import Path
from common import read, timed, locked, save
from collage.config import Config


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--reference',required=True);a.add_argument('--materials',nargs='+',required=True);a.add_argument('--run',required=True);a.add_argument('--width',type=int,default=1200);a.add_argument('--instructions',default='');a.add_argument('--cutout-model')
    a=sub.add_parser('prepare-live',help='Prepare mixed photos and native-frame Live videos')
    a.add_argument('--reference',required=True);a.add_argument('--materials',nargs='+',required=True);a.add_argument('--run',required=True)
    a.add_argument('--width',type=int,default=1200);a.add_argument('--instructions',default='');a.add_argument('--cutout-model')
    a.add_argument('--progress-file')
    a=sub.add_parser('motion-render',help='Render Live motion after freezing a static cover')
    a.add_argument('--run',required=True);a.add_argument('--request-id');a.add_argument('--base-render-id')
    a=sub.add_parser('progress',help='Record an actual Codex stage observation')
    a.add_argument('--run',required=True);a.add_argument('--stage',type=int,choices=range(1,7),required=True)
    a.add_argument('--status',choices=['running','waiting','blocked','complete','skipped','failed'],required=True)
    a.add_argument('--summary',required=True);a.add_argument('--analysis-only',action='store_true')
    for name in ['validate','build','reveal-plan','render','apply','review','generate','recover','screen','revision-prepare','revision-commit','revision-recover']:
        a=sub.add_parser(name);a.add_argument('--run',required=True)
        a.add_argument('--brief',action='store_true',help='Print only paths and outcome; complete evidence remains in result.json')
        if name=='validate':a.add_argument('--analysis-only',action='store_true',help='Validate reference analysis before customer binding')
        if name in ['apply','review','recover','screen','revision-prepare','revision-commit']:a.add_argument('--file',required=True)
        if name=='screen':a.add_argument('--rescreen',action='store_true',help='Reassess current cached candidates and retain the previous screening receipt')
        if name.startswith('revision-'):a.add_argument('--request-id',required=True)
        if name in ['build','reveal-plan']:
            a.add_argument('--reveal',action='store_true',help='Enable remote Reveal requests for missing complex assets')
            a.add_argument('--reveal-cache',help='Existing downloaded Reveal sample folder; offline unless --reveal is also set')
            a.add_argument('--reveal-key-file',default=os.getenv('COLLAGE_REVEAL_ENV',Config.reveal_key_file))
            a.add_argument('--reveal-timeout',type=int,default=360)
            a.add_argument('--reveal-max-requests',type=int,default=int(os.getenv('COLLAGE_REVEAL_MAX_REQUESTS','16')),
                           help='Total submission limit for full-image material and background-cleaning batches')
            a.add_argument('--reveal-padding',type=float,default=.1,help='Reveal request padding fraction (default .1)')
            a.add_argument('--reveal-padding-mode',choices=['capped','ratio'],default='capped',help='Equal capped padding (default), or historical per-axis ratio padding')
            a.add_argument('--no-reveal',action='store_true')
            a.add_argument('--reveal-layout',choices=['full','grouped','legacy'],default='grouped',help='Spatial crops (default, at most 3 groups), full image, or historical requests')
        if name=='generate':
            a.add_argument('--ids',nargs='+',required=True);a.add_argument('--credentials',default=os.getenv('COLLAGE_YIBU_CREDENTIALS',Config.yibu_credentials));a.add_argument('--allow-remote',action='store_true');a.add_argument('--timeout',type=int,default=300);a.add_argument('--workers',type=int,default=2);a.add_argument('--dry-run',action='store_true');a.add_argument('--group',action='store_true',help='Generate selected overlay/text members as one fused unit')
    args=p.parse_args()
    try:
        if args.command=='prepare-live':
            from live_media import prepare_live
            result=prepare_live(args.reference,args.materials,args.run,args.width,args.instructions,args.cutout_model,progress_file=args.progress_file)
        elif args.command=='motion-render':
            from motion_render import execute
            result=execute(args.run,args.request_id,args.base_render_id)
        elif args.command=='progress':
            from observation import checkpoint
            with locked(Path(args.run)):
                result=checkpoint(args.run,args.stage,args.status,args.summary,args.analysis_only)
        elif args.command=='prepare':
            from prepare import prepare
            result=prepare(args.reference,args.materials,args.run,args.width,args.instructions,args.cutout_model)
        else:
            run=Path(args.run).resolve()
            if not (run/'input.json').exists():raise ValueError('Run prepare first')
            with locked(run),timed(run,args.command):
                if getattr(args,'file',None):
                    from observation import evidence
                    evidence(run,args.file)
                if args.command=='revision-recover':
                    from revision import recover_commit
                    marker=read(run/'chat/commit.json') if (run/'chat/commit.json').exists() else {}
                    if marker and marker.get('request_id')!=args.request_id:raise ValueError('Commit belongs to another request')
                    recover_commit(run);result={'recovered':True}
                elif args.command in ['revision-prepare','revision-commit']:
                    from revision import prepare_revision,commit_revision
                    function=prepare_revision if args.command=='revision-prepare' else commit_revision
                    result=function(run,args.request_id,read(args.file))
                elif args.command=='validate':
                    from validate import analysis_check, bindings_check
                    from prepare import boxes
                    analysis=analysis_check(read(run/'analysis.json'),read(run/'input.json'))
                    result={'analysis_valid':True,'boxes':boxes(run,analysis)}
                    if (run/'photo-windows.json').exists():
                        from photo_windows import check, previews
                        plan=check(read(run/'photo-windows.json'),analysis)
                        result['photo_windows_valid']=True
                        result['photo_window_previews']=[str(p) for p in previews(run,analysis,plan)]
                    if not args.analysis_only and (run/'bindings.json').exists():
                        bindings_check(read(run/'bindings.json'),analysis,read(run/'prepared/catalog.json'));result['bindings_valid']=True
                elif args.command=='reveal-plan':
                    from validate import analysis_check
                    from effects import resolve_style
                    from reveal_assets import targets
                    from reveal_plan import plan,preview
                    a=analysis_check(read(run/'analysis.json'),read(run/'input.json'))
                    for o in a['objects']:o['style']=resolve_style(o)
                    if args.reveal_layout=='legacy':raise ValueError('Plan preview supports grouped or full')
                    result=preview(read(run/'input.json')['reference']['file'],plan(a,targets(a),args.reveal_padding,args.reveal_layout,args.reveal_padding_mode,args.reveal_max_requests if args.reveal_layout=='full' else None),run/'previews/reveal-plan')
                elif args.command=='build':
                    from scene import compile_scene
                    from render import render
                    config=None
                    if args.no_reveal and (args.reveal or args.reveal_cache):raise ValueError('Conflicting Reveal options')
                    if args.no_reveal:config={'enabled':False}
                    elif args.reveal or args.reveal_cache:
                        config={'enabled':True,'remote':args.reveal,'cache':str(Path(args.reveal_cache).resolve()) if args.reveal_cache else None,'key_file':args.reveal_key_file,'timeout':args.reveal_timeout,'padding':args.reveal_padding,'padding_mode':args.reveal_padding_mode,'layout':args.reveal_layout,'max_requests':args.reveal_max_requests}
                    compile_scene(run,config);result=render(run)
                elif args.command=='render':
                    from render import render
                    result=render(run)
                elif args.command=='recover':
                    from recovery import recover
                    result=recover(run,args.file)
                elif args.command=='screen':
                    from screening import screen
                    result=screen(run,args.file,rescreen=args.rescreen)
                elif args.command=='apply':
                    from scene import apply_review
                    from render import render
                    apply_review(run,args.file);result=render(run)
                elif args.command=='review':
                    from render import accept_review
                    result=accept_review(run,args.file)
                elif args.command=='generate':
                    from generation.generate import generate
                    result=generate(run,args)
        if getattr(args,'brief',False) and isinstance(result,dict) and 'render_id' in result:
            result={k:result[k] for k in ['status','render_id','incomplete_objects','asset_gate_summary','extraction_sheets','review_regions','renders_verified'] if k in result}
        print(json.dumps(result,ensure_ascii=True,indent=2));return 0
    except Exception as exc:
        print(json.dumps({'status':'failed','error':str(exc),'type':type(exc).__name__},ensure_ascii=True));return 2


if __name__=='__main__':sys.exit(main())
