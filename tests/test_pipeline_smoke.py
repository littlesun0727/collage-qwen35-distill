"""Exercise real analyze/compile/render and request evidence using a fake transport."""
import json
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import run_covers

def test_batch_pipeline_and_resume(tmp_path):
    source=tmp_path/'images'; source.mkdir()
    Image.new('RGB',(300,500),'white').save(source/'one.png')
    root=tmp_path/'project'
    model=root/'models/Qwen3.5-4B'; model.mkdir(parents=True)
    (model/'download-manifest.json').write_text(json.dumps({'model':'Qwen/Qwen3.5-4B','revision':'test'}))
    output=tmp_path/'run'
    repo=Path(__file__).resolve().parents[1]/'vendor/collage-recreate'
    answer={'background':{'type':'solid_color','color':'#FFFFFF'},
        'objects':[{'id':'photo_1','kind':'photo','bbox':[101,203,789,907],'description':'A rectangular photograph.'}],
        'layer_order':['photo_1']}
    calls=[]
    def fake(payload):
        calls.append(payload)
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(answer)}}],
                'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150}}
    argv=['run_covers.py','--repo',str(repo),'--input',str(source),'--output',str(output),'--model','Qwen3.5-4B','--workers','1']
    with patch.object(run_covers,'ROOT',root),patch.object(run_covers,'infer',side_effect=fake),patch('sys.argv',argv):
        run_covers.main()
        run_covers.main()
    assert len(calls)==1
    assert calls[0]['chat_template_kwargs']=={'enable_thinking':False}
    assert 'reference_aspect_ratio' not in json.dumps(calls[0])
    assert 'reference_size' not in json.dumps(calls[0])
    s=json.loads((output/'batch-summary.json').read_text())
    assert s['complete']==1 and s['failed']==0
    case=output/'cases'/s['cases'][0]['id']
    analysis=json.loads((case/'analysis.json').read_text())
    box=next(o['bbox'] for o in analysis['objects'] if o['id']=='photo_1')
    assert box==[30,102,237,454]
    assert (case/'analysis.bbox.png').exists()
    assert (output/'index.html').exists()
