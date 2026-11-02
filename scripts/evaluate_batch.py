"""Evaluate raw-first or final outputs against supplied GT, retaining failed cases."""
import argparse
import hashlib
import json
from pathlib import Path
from eval.io import Record, load_records, read_structure, load_occlusions
from eval.score import evaluate_dataset, markdown_report, parse_record

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def final_prediction(sample_id, root, case):
    path=root/'cases'/case['id']/'structure.json'
    source={'id':sample_id,'filename':case['relative_path'],'workflow_status':case['status'],'path':str(path)}
    if case['status']!='complete':
        source.update(missing=True,reason='workflow_failed; no deliverable prediction')
        return None,source
    record=read_structure(sample_id,path)
    if not parse_record(record).valid:
        raise ValueError('Completed workflow has missing/invalid final structure: '+str(path))
    source.update(sha256=record.source_hash,repaired=case.get('evidence',{}).get('repaired',False))
    return record,source

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--id-map',type=Path,help='JSON object mapping image relative filenames to GT IDs; default filenames are IDs')
    p.add_argument('--stage',choices=['raw_first','final_accepted'],required=True)
    p.add_argument('--occlusions',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    batch=read(a.batch/'batch-summary.json')
    if 'elapsed_seconds_this_invocation' not in batch:
        p.error('Batch must be finished')
    gt,metadata=load_records(a.gt)
    mapping=read(a.id_map) if a.id_map else {}
    predictions,sources={},[]
    seen=set()
    for case in batch['cases']:
        sid=mapping.get(case['relative_path'],case['relative_path'])
        if sid not in gt or sid in seen:
            p.error('Unmatched or duplicate GT ID: '+sid+'; supply --id-map')
        seen.add(sid)
        if a.stage=='final_accepted':
            rec,src=final_prediction(sid,a.batch,case)
        else:
            paths=list((a.batch/'cases'/case['id']).glob('workflow/vlm/analyze/*/response-0.json'))
            if len(paths)>1:
                p.error('Ambiguous first response for '+sid)
            src={'id':sid,'missing':not paths}
            rec=None
            if paths:
                d=read(paths[0]); content=d.get('content')
                raw_hash=hashlib.sha256(paths[0].read_bytes()).hexdigest()
                rec=Record(sid,content,source=str(paths[0]),source_hash=raw_hash)
                src.update(path=str(paths[0]),sha256=raw_hash,finish_reason=d.get('finish_reason'))
        if rec is not None:
            predictions[sid]=rec
        sources.append(src)
    summary,samples=evaluate_dataset(gt,predictions,occlusions=load_occlusions(a.occlusions,gt),prediction_stage=a.stage)
    summary['run']={'batch':str(a.batch.resolve()),'gt_input':metadata,'prediction_stage':a.stage,'sources':sources}
    if metadata.get('annotation_status') in ('teacher_prediction_unreviewed','mixed_candidate_annotations_unreviewed'):
        summary['warnings'].append('Unreviewed teacher annotations measure agreement, not verified accuracy.')
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (a.out/'per_sample.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False,allow_nan=False)+'\n' for s in samples),encoding='utf-8')
    (a.out/'report.md').write_text(markdown_report(summary),encoding='utf-8')
    print(a.out/'report.md')

if __name__=='__main__':
    main()
