"""Customer RGB always comes from a pinned source. Masks are never image substitutes."""
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageColor
from common import sha, fingerprint, save, read, verify_source
from effects import clip_round, expanded_outline, photo_layout, photo_card
from photo_windows import apply_window, subject_stroke


def resolve_cutout_model(model):
    """Resolve archived tasks after relocating the shared model into this project."""
    if model and not Path(model).is_file() and Path(model).name == 'birefnet-lite-fp32.onnx':
        root = Path(__file__).resolve().parents[2]
        relocated = root.with_name(root.name+'-output')/'models/birefnet-lite-fp32.onnx'
        if relocated.is_file():
            return str(relocated)
    return model


def fit_image(im, size, center=(.5,.5), contain=False, margin=0):
    if margin:
        if contain:
            tile = fit_image(im, size, center, contain=True)
            return ImageOps.expand(tile, margin, fill=0)
        ratio = max(size[0]/im.width, size[1]/im.height)
        x = (im.width-size[0]/ratio)*center[0]-margin/ratio
        y = (im.height-size[1]/ratio)*center[1]-margin/ratio
        return im.transform((size[0]+2*margin, size[1]+2*margin), Image.Transform.AFFINE,
                            (1/ratio, 0, x, 0, 1/ratio, y), Image.Resampling.BICUBIC)
    if contain:
        tile=ImageOps.contain(im,size,Image.Resampling.LANCZOS)
        result=Image.new('RGBA',size);result.alpha_composite(tile,((size[0]-tile.width)//2,(size[1]-tile.height)//2));return result
    return ImageOps.fit(im,size,Image.Resampling.LANCZOS,centering=tuple(center))


def feather_alpha(size, amount):
    w,h=size;y,x=np.mgrid[0:h,0:w].astype(np.float32)
    distance=np.minimum.reduce([x+.5,w-x-.5,y+.5,h-y-.5])
    v=np.clip(distance/max(1,amount*min(size)),0,1);v=v*v*(3-2*v)
    return Image.fromarray(np.rint(v*255).astype('uint8'))


def fit_subject_mask(alpha, binding, size, margin=0):
    """Use the same source coordinates and cover transform as the retained RGB."""
    crop = binding.get('source_crop', [0,0,1,1]); w,h = alpha.size
    alpha = alpha.crop((int(crop[0]*w),int(crop[1]*h),
                        max(1,int(crop[2]*w)),max(1,int(crop[3]*h))))
    if binding.get('mirror_x'):
        alpha = ImageOps.mirror(alpha)
    return fit_image(alpha, size, binding.get('crop_center',[.5,.5]), margin=margin)


@lru_cache(maxsize=2)
def session(model, model_sha):
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    # The default CPU arena retains large activation allocations between Live
    # frames. Concurrent static and motion workers can exhaust Windows commit.
    opts.enable_cpu_mem_arena=False
    opts.enable_mem_pattern=False
    return ort.InferenceSession(model,sess_options=opts,providers=['CPUExecutionProvider'])


def cutout(im, source, model, cache):
    model = resolve_cutout_model(model)
    if not model or not Path(model).is_file(): raise ValueError('cutout needs existing BiRefNet FP32 ONNX weights (--cutout-model in prepare)')
    model_sha=sha(model);key=fingerprint([source['sha256'],model_sha,'v5-birefnet-1'])
    path=Path(cache)/(key+'.png');manifest=path.with_suffix('.json')
    if path.exists() and manifest.exists() and sha(path)==read(manifest)['sha256']:
        with Image.open(path) as raw:return raw.convert('RGBA')
    # Validated offline mattes may be reused across tasks selecting the same photo.
    from collage.paths import OUTPUT_ROOT
    shared=OUTPUT_ROOT/'cache/cutouts'/(key+'.png');receipt=shared.with_suffix('.json')
    record=read(receipt) if receipt.exists() else {}
    if (shared.is_file() and record.get('source_sha256')==source['sha256']
            and record.get('model_sha256')==model_sha and sha(shared)==record.get('sha256')):
        with Image.open(shared) as raw:
            result=raw.convert('RGBA')
        if result.size!=im.size:raise ValueError('Shared cutout dimensions differ from source')
        return result
    # ONNX native errors inherit Exception directly, not RuntimeError. Normalize
    # only the external engine boundary so render can fall back for this photo.
    try:
        engine=session(str(model),model_sha);inp=engine.get_inputs()[0]
    except Exception as exc:
        raise RuntimeError('BiRefNet session failed: '+str(exc)) from exc
    if inp.type!='tensor(float)':raise ValueError('BiRefNet weights must be FP32')
    h,w=[v if isinstance(v,int) and v>0 else 1024 for v in inp.shape[2:]]
    tensor=np.asarray(im.convert('RGB').resize((w,h)),dtype=np.float32)/255
    tensor=(tensor-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    try:
        prediction=engine.run(None,{inp.name:np.ascontiguousarray(tensor.transpose(2,0,1)[None])})
    except Exception as exc:
        raise RuntimeError('BiRefNet inference failed: '+str(exc)) from exc
    alpha=np.asarray(prediction[-1]).squeeze()
    if alpha.ndim!=2 or not np.isfinite(alpha).all():raise ValueError('Invalid segmentation output')
    if alpha.min()<0 or alpha.max()>1:alpha=1/(1+np.exp(-np.clip(alpha,-50,50)))
    alpha=Image.fromarray(np.clip(alpha,0,1).astype('float32')).resize(im.size,Image.Resampling.BILINEAR)
    a=np.rint(np.clip(np.asarray(alpha),0,1)*255).astype('uint8')
    a=np.rint(a.astype('float32')*np.asarray(im.getchannel('A'))/255).astype('uint8')
    fg=im.copy();fg.putalpha(Image.fromarray(a))
    bounds=fg.getchannel('A').point(lambda x:255 if x>8 else 0).getbbox()
    if not bounds:raise ValueError('Empty customer cutout')
    path.parent.mkdir(exist_ok=True,parents=True);fg.save(path);save(manifest,{'sha256':sha(path),'source_sha256':source['sha256'],'model_sha256':model_sha})
    return fg


def make_photo(obj, source, binding, size, run, model):
    path=verify_source(source)
    with Image.open(path) as raw: im=ImageOps.exif_transpose(raw).convert('RGBA')
    subject_alpha = None
    spec = obj.get('photo_window_spec')
    shared=obj.get('shared_photo')
    if shared and shared.get('subject_anchor') and not obj.get('_skip_subject_anchor'):
        from photo_links import anchor_subject
        foreground=cutout(im,source,model,Path(run)/'assets/photos/.cache')
        obj=anchor_subject(obj,foreground.getchannel('A'))
        shared=obj['shared_photo']
        if obj['mode']=='cutout':im=foreground
        elif subject_stroke(spec):subject_alpha=foreground.getchannel('A')
    elif obj['mode']=='cutout':
        im=cutout(im,source,model,Path(run)/'assets/photos/.cache')
        if not shared:
            bounds=im.getchannel('A').point(lambda x:255 if x>8 else 0).getbbox();im=im.crop(bounds)
    elif subject_stroke(spec):
        # Keep original RGB, alpha and coordinates; only borrow the foreground mask.
        subject_alpha = cutout(im,source,model,Path(run)/'assets/photos/.cache').getchannel('A')
        if subject_alpha.size != im.size:
            raise ValueError('Subject mask dimensions do not match the source photo')
    if not shared and obj['mode'] != 'cutout':
        from collage.crop_quality import expand_crop
        style = obj.get('style', {})
        bounds = (0, 0, *size) if spec else photo_layout(size, style)
        target = (bounds[2]-bounds[0], bounds[3]-bounds[1])
        binding = {**binding, 'source_crop': expand_crop(
            binding.get('source_crop', [0,0,1,1]), im.size, target)}
    crop=binding.get('source_crop',[0,0,1,1]);w,h=im.size
    if not shared:
        im=im.crop((int(crop[0]*w),int(crop[1]*h),max(1,int(crop[2]*w)),max(1,int(crop[3]*h))))
        if binding.get('mirror_x'):im=ImageOps.mirror(im)
    style=obj.get('style',{})
    left,top,right,bottom=(0,0,*size) if obj.get('photo_window_spec') else photo_layout(size,style)
    window=(right-left,bottom-top)
    placement_size=window
    # Linked subjects share the base photo's source transform. Their analysis
    # bbox describes the reference silhouette, not a clip for a new customer's
    # silhouette. Sample the visible canvas when no fixed window was requested.
    canvas_aligned=bool(shared and obj['mode']=='cutout' and spec and
                        not spec['window_enabled'] and obj.get('_render_canvas_size'))
    sample_obj=obj
    if canvas_aligned:
        window=tuple(obj['_render_canvas_size']);scale=obj['_render_scale']
        sample_obj={**obj,'bbox':[0,0,window[0]/scale,window[1]/scale],'rotation':0}
    from window_drawing import sampling_margin
    margin = sampling_margin(window, spec)
    if shared:
        from photo_links import shared_tile
        im=shared_tile(sample_obj,im,window,obj['_render_scale'], margin=margin)
    else:im=fit_image(im,window,binding.get('crop_center',[.5,.5]),obj['mode']=='cutout', margin=margin)
    clipped_by_base_bottom=False
    if obj.get('photo_window_spec'):
        if subject_alpha is not None:
            subject_alpha = shared_tile(obj,subject_alpha,window,obj['_render_scale'], margin=margin) if shared else fit_subject_mask(subject_alpha, binding, window, margin=margin)
        im,content_alpha=apply_window(im,obj['photo_window_spec'],subject_alpha, source_margin=margin,
                                      placement_size=placement_size if canvas_aligned else None)
        if shared and obj['mode']=='cutout':
            from photo_links import clip_linked_bottom
            im,content_alpha=clip_linked_bottom(sample_obj,im,obj['_render_scale'],content_alpha)
            clipped_by_base_bottom=not im.getchannel('A').getbbox()
    else:
        if style.get('corner_radius',0):im=clip_round(im,style['corner_radius'])
        if obj['mode']=='feather':
            a=np.asarray(im.getchannel('A'),dtype='float32')*np.asarray(feather_alpha(window,style.get('feather',.08)))/255
            im.putalpha(Image.fromarray(np.rint(a).astype('uint8')))
        # Keep actual customer pixels distinct from paper and outlines for visibility.
        content_alpha=Image.new('L',size);content_alpha.paste(im.getchannel('A'),(left,top))
        im=photo_card(im,size,style)
        im,pad=expanded_outline(im,style.get('outline_width',3 if obj['mode']=='cutout' else 0),style.get('outline_color','#FFFFFF'))
        if pad:
            mask=Image.new('L',im.size);mask.paste(content_alpha,(pad,pad));content_alpha=mask
    if not im.getchannel('A').getbbox() and not clipped_by_base_bottom:raise ValueError('Photo output is entirely transparent')
    content_file=Path(run)/'assets/photos'/(obj['id']+'-content-mask.png');content_file.parent.mkdir(parents=True,exist_ok=True);content_alpha.save(content_file)
    metadata={'resource_type':'customer_photo','source':source,'mode':obj['mode'],'binding':binding,'pixel_operation':'bound-source-resize-crop-mask; no generated RGB','content_mask':str(content_file),'content_mask_sha256':sha(content_file)}
    if obj.get('photo_window_spec'):metadata['photo_window_spec']=obj['photo_window_spec']
    if canvas_aligned:metadata['canvas_aligned']=True
    if clipped_by_base_bottom:
        metadata.update(quality='needs_review', clipped_by_base_bottom=True,
                        note='Linked foreground is entirely below the base crop bottom; kept transparent')
    return im,metadata
