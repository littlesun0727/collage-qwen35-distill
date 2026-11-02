"""Optional PaddleOCR / TorchScript LaMa worker, local weights only, no network."""
import json
import os
import socket
import sys
from pathlib import Path


def offline(*args,**kwargs):raise RuntimeError('Repair workers are offline; provision local weights first')


def main():
    job=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));cfg=job['config']
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    os.environ.setdefault('PADDLE_PDX_CACHE_HOME',str(Path(job['output']).parent/'paddle-cache'))
    os.environ.setdefault('MPLCONFIGDIR',str(Path(job['output']).parent/'matplotlib-cache'))
    socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
    import numpy as np
    from PIL import Image
    im=Image.open(job['image']).convert('RGB')
    if job['kind']=='ocr':
        import torch  # Load its Windows DLLs before Paddle's overlapping runtime libraries.
        # Paddle's dataset import ignores PADDLE_PDX_CACHE_HOME. Its documented
        # read-only-home fallback uses tempfile; route that fallback to this job.
        # Do not change HOME/USERPROFILE or filesystem permissions.
        import tempfile
        cache=Path(os.environ['PADDLE_PDX_CACHE_HOME']);cache.mkdir(parents=True,exist_ok=True)
        tempfile.tempdir=str(cache)
        original_access=os.access
        home=os.path.normcase(os.path.expanduser('~'))
        os.access=lambda p,mode,*a,**kw: False if mode==os.W_OK and os.path.normcase(os.fspath(p))==home else original_access(p,mode,*a,**kw)
        try:import paddle
        finally:os.access=original_access
        from paddleocr import PaddleOCR
        ocr=PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det',text_detection_model_dir=cfg['ocr_det'],
            text_recognition_model_name='PP-OCRv5_mobile_rec',text_recognition_model_dir=cfg['ocr_rec'],
            use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,
            device='cpu',enable_mkldnn=False,cpu_threads=2)
        records=[]
        for page in ocr.predict(np.asarray(im)[:,:,::-1]):
            for poly,text,score in zip(page['rec_polys'],page['rec_texts'],page['rec_scores']):
                records.append({'polygon':np.asarray(poly).tolist(),'text':text,'confidence':float(score)})
        Path(job['output']).write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    elif job['kind']=='lama':
        import torch
        torch.set_num_threads(2)
        model=torch.jit.load(cfg['lama_model'],map_location='cpu').eval()
        mask=np.asarray(Image.open(job['mask']).convert('L'),dtype=np.float32)/255
        rgb=np.asarray(im,dtype=np.float32)/255
        ph=(-im.height)%8;pw=(-im.width)%8
        rgb=np.pad(rgb,((0,ph),(0,pw),(0,0)),mode='symmetric')
        mask=np.pad(mask,((0,ph),(0,pw)),mode='symmetric')
        with torch.inference_mode():
            output=model(torch.from_numpy(rgb.transpose(2,0,1)[None].copy()),torch.from_numpy((mask>0).astype('float32')[None,None]))
        rgb=np.clip(output[0].permute(1,2,0).cpu().numpy()*255,0,255).astype('uint8')
        Image.fromarray(rgb[:im.height,:im.width]).save(job['output'])
    else:raise ValueError('Unknown local repair kind')


if __name__=='__main__':main()
