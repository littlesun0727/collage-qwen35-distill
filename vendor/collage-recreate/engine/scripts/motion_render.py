"""Post-cover motion production. Never rewrites the approved static scene/PNG."""
from collage.object_roles import is_customer_photo, is_photo_background
import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
import time
import uuid
# Import the package runtime settings before NumPy initializes BLAS workers.
from common import read, sha, now, locked, verify_source, fingerprint
import numpy as np
from PIL import Image, ImageOps, ImageChops
from motion_io import save
from live_media import timeline, frame_index, decode_needed, encode_frames, verify_encoded, NativeFrameReader
from photos import cutout, fit_image, feather_alpha, fit_subject_mask, resolve_cutout_model
from effects import photo_layout, photo_card, clip_round, expanded_outline, scale_style, shadow_layer
from editor_scene import translate,clip_window
from photo_windows import apply_window, needs_subject_mask, subject_stroke


def reuse_matte(run, sequence_key, index, source_sha, model_sha, destination):
    """Reuse only a published, hash-verified matte of the identical frame/model."""
    pattern = f'*/chat/motion/cache/alpha/{sequence_key}/{index:05d}.json'
    for receipt in Path(run).parent.glob(pattern):
        source = receipt.with_suffix('.png')
        if source == destination:
            continue
        try:
            record = read(receipt)
            if (record.get('source_sha256') != source_sha
                    or record.get('model_sha256') != model_sha
                    or sha(source) != record.get('sha256')):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha(destination) != record['sha256']:
                continue
            save(destination.with_suffix('.json'), record)
            return True
        except (OSError, ValueError, KeyError):
            continue
    return False


class Progress:
    def __init__(self, path):
        self.path = path
        self.started = time.monotonic()
        self.data = {'status': 'running', 'stages': {}, 'started_at': now()}

    def write(self):
        self.data['elapsed_seconds'] = round(time.monotonic()-self.started, 3)
        save(self.path, self.data)

    @contextmanager
    def stage(self, name):
        start = time.monotonic()
        self.name = name
        self.data['stage'] = name
        row = self.data['stages'][name] = {'status': 'running', 'started_at': now()}
        self.write()
        try:
            yield
        except Exception:
            row['status'] = 'failed'
            raise
        else:
            row['status'] = 'complete'
        finally:
            row['elapsed_seconds'] = round(time.monotonic()-start, 3)
            self.write()

    def count(self, done, total, **detail):
        self.data['stages'][self.name].update(done=done, total=total, **detail)
        self.write()


def fixed_photo_layer(obj, image, anchor, scene, subject_alpha=None):
    """Cover geometry stays fixed. Unframed cutouts may move beyond cover alpha bounds."""
    scale = scene['canvas_size'][0]/scene['reference_size'][0]
    box = [round(v*scale) for v in obj['bbox']]
    size = (max(1, box[2]-box[0]), max(1, box[3]-box[1]))
    style = scale_style(obj['style'], scale)
    binding = obj['binding']
    crop = binding.get('source_crop', [0, 0, 1, 1])
    left, top, right, bottom = (0,0,*size) if obj.get('photo_window_spec') else photo_layout(size, style)
    window = (right-left, bottom-top)
    if not obj.get('shared_photo') and obj['mode'] != 'cutout':
        from collage.crop_quality import expand_crop
        crop = expand_crop(crop, image.size, window)
        binding = {**binding, 'source_crop': crop}
    from window_drawing import sampling_margin
    margin = sampling_margin(window, obj.get('photo_window_spec'))
    pad = 0
    canvas_aligned = bool(obj.get('shared_photo') and obj['mode']=='cutout' and
                          obj.get('photo_window_spec') and not obj['photo_window_spec']['window_enabled'])
    if obj.get('shared_photo'):
        from photo_links import shared_tile
        if anchor and obj['shared_photo'].get('subject_anchor'):
            # Pin both layers to the same first-frame silhouette, not the full frame.
            shared={**obj['shared_photo'], 'subject_bounds':[
                v/(image.width if i%2==0 else image.height) for i,v in enumerate(anchor)]}
            obj={**obj,'shared_photo':shared}
        placement_size = window
        sample_obj = obj
        if canvas_aligned:
            window = tuple(scene['canvas_size'])
            sample_obj = {**obj, 'bbox':[0,0,window[0]/scale,window[1]/scale], 'rotation':0}
            margin = sampling_margin(window, obj['photo_window_spec'])
        im=shared_tile(sample_obj,image,window,scale,margin=margin)
        if subject_alpha is not None:subject_alpha=shared_tile(sample_obj,subject_alpha,window,scale,margin=margin)
        tile,_=apply_window(im,obj['photo_window_spec'],subject_alpha,source_margin=margin,
                            placement_size=placement_size if canvas_aligned else None)
        from photo_links import clip_linked_bottom
        tile,_=clip_linked_bottom(sample_obj,tile,scale)
    elif obj.get('photo_window_spec'):
        if obj['mode'] == 'cutout':
            x0,y0,x1,y1 = anchor; width,height = x1-x0,y1-y0
        else:
            x0,y0 = 0,0; width,height = image.size
        im = image.crop((x0+int(crop[0]*width), y0+int(crop[1]*height),
                         x0+max(1,int(crop[2]*width)), y0+max(1,int(crop[3]*height))))
        if binding.get('mirror_x'):im = ImageOps.mirror(im)
        im = fit_image(im, window, binding.get('crop_center',[.5,.5]),obj['mode']=='cutout', margin=margin)
        spec = obj['photo_window_spec']
        if spec['subject']['mode'] == 'preserve' and subject_stroke(spec):
            if subject_alpha is None or subject_alpha.size != image.size:
                raise ValueError('Live subject stroke requires a source-aligned subject mask')
            subject_alpha = fit_subject_mask(subject_alpha, binding, window, margin=margin)
        tile,_ = apply_window(im,spec,subject_alpha,source_margin=margin)
    elif obj['mode'] == 'cutout':
        x0, y0, x1, y1 = anchor
        width, height = x1-x0, y1-y0
        bound = (x0+int(crop[0]*width), y0+int(crop[1]*height),
                 x0+max(1, int(crop[2]*width)), y0+max(1, int(crop[3]*height)))
        # Pillow contain's rounded dimensions match the original cover scaling.
        probe = ImageOps.contain(Image.new('RGBA', (bound[2]-bound[0], bound[3]-bound[1])), window)
        sx, sy = probe.width/(bound[2]-bound[0]), probe.height/(bound[3]-bound[1])
        unrestricted = crop == [0, 0, 1, 1] and not style.get('card') and not style.get('corner_radius')
        if unrestricted:
            full = image.resize((max(1, round(image.width*sx)), max(1, round(image.height*sy))), Image.Resampling.LANCZOS)
            x = left+(window[0]-probe.width)//2-round(bound[0]*sx)
            y = top+(window[1]-probe.height)//2-round(bound[1]*sy)
            if binding.get('mirror_x'):
                full = ImageOps.mirror(full)
                center = left+(window[0]-probe.width)//2
                x = center+probe.width-(x-center)-full.width
            extent=full.getchannel('A').getbbox()
            if not extent:
                raise ValueError('Empty mapped person frame')
            # Include every nonzero alpha pixel, but avoid outlining huge empty margins.
            # Only transparent padding changes; the source-to-cover mapping stays fixed.
            pad=max(0,-x-extent[0],-y-extent[1],x+extent[2]-size[0],y+extent[3]-size[1])+2
            tile = Image.new('RGBA', (size[0]+2*pad, size[1]+2*pad))
            tile.alpha_composite(full, (x+pad, y+pad))
        else:
            im = image.crop(bound)
            if binding.get('mirror_x'):
                im = ImageOps.mirror(im)
            im = fit_image(im, window, contain=True)
            if style.get('corner_radius'):
                im = clip_round(im, style['corner_radius'])
            tile = photo_card(im, size, style)
    else:
        width, height = image.size
        im = image.crop((int(crop[0]*width), int(crop[1]*height),
                         max(1, int(crop[2]*width)), max(1, int(crop[3]*height))))
        if binding.get('mirror_x'):
            im = ImageOps.mirror(im)
        im = fit_image(im, window, binding.get('crop_center', [.5, .5]))
        if style.get('corner_radius'):
            im = clip_round(im, style['corner_radius'])
        if obj['mode'] == 'feather':
            im.putalpha(ImageChops.multiply(im.getchannel('A'), feather_alpha(window, style.get('feather', .08))))
        tile = photo_card(im, size, style)
    if not obj.get('photo_window_spec'):
        tile, _ = expanded_outline(tile, style.get('outline_width', 3 if obj['mode']=='cutout' else 0), style.get('outline_color', '#FFFFFF'))
    if style.get('opacity', 1) != 1:
        tile.putalpha(tile.getchannel('A').point(lambda v: round(v*style['opacity'])))
    if obj.get('rotation') and not canvas_aligned:
        tile = tile.rotate(-obj['rotation'], Image.Resampling.BICUBIC, expand=True)
    layer = Image.new('RGBA', tuple(scene['canvas_size']))
    if canvas_aligned:box=[0,0,*scene['canvas_size']]
    layer.alpha_composite(tile, (round((box[0]+box[2]-tile.width)/2), round((box[1]+box[3]-tile.height)/2)))
    if obj.get('photo_window'):
        with Image.open(verify_source(obj['photo_window'])) as raw:
            layer.putalpha(ImageChops.multiply(layer.getchannel('A'), raw.convert('L').resize(layer.size, Image.Resampling.LANCZOS)))
    shadow = shadow_layer(layer.getchannel('A'), style['shadow']) if 'shadow' in style else None
    if shadow is not None:
        layer = Image.alpha_composite(shadow, layer)
    layer=clip_window(layer,obj,scale)
    return translate(layer, obj, scale) if obj.get('editor_transform') else layer


def make_plan(run, scene, result):
    manifest = read(run/'media/manifest.json')
    videos = manifest['videos']
    if not videos:
        raise ValueError('This task has no Live videos')
    by_asset = {v['asset_id']: v for v in videos}
    if len(by_asset) != len(videos):
        raise ValueError('Ambiguous Live poster mappings')
    for video in videos:
        verify_source(video)
        verify_source(video['poster'])
    slots = []
    for obj in scene['objects']:
        if not is_customer_photo(obj):
            continue
        video = by_asset.get(obj.get('binding', {}).get('asset_id'))
        if not video:
            # Binding schema calls the identifier asset, depending on producer version.
            video = next((v for v in videos if v['poster']['sha256'] == obj['source']['sha256']), None)
        if video:
            if obj['source']['sha256'] != video['poster']['sha256']:
                raise ValueError('Live video does not match the frozen cover source')
            slots.append({'object_id': obj['id'], 'video_id': video['id'], 'mode': obj['mode']})
    if not slots:
        raise ValueError('The cover has no selected Live posters; bind Live photos before finalizing the cover')
    model_record = None
    moving_ids = {slot['object_id'] for slot in slots}
    if any(obj['id'] in moving_ids and needs_subject_mask(obj) for obj in scene['objects']):
        model = resolve_cutout_model(scene.get('cutout_model'))
        if not model or not Path(model).is_file():
            raise ValueError('Live cutout/subject stroke requires the pinned BiRefNet model')
        model_record = {'file': str(model), 'sha256': sha(model)}
    return {'schema_version': 'collage-motion-v1', 'base_render_id': result['render_id'],
            'base_scene_sha256': sha(run/'scene.json'), 'base_final_sha256': sha(run/'final.png'),
            'duration_us': min(3_000_000, max(v['duration_us'] for v in videos)),
            'time_policy': 'native_pts_union', 'audio': 'muted', 'slots': slots,
            'supplied_videos': [v['id'] for v in videos], 'videos': videos, 'cutout_model': model_record}


def execute(run, identifier=None, base_render_id=None):
    folder=Path(run)/'chat/motion'
    folder.mkdir(parents=True,exist_ok=True)
    with locked(folder):
        return _execute(run,identifier,base_render_id)


def _execute(run, identifier=None, base_render_id=None):
    run = Path(run).resolve()
    identifier = identifier or 'motion-'+uuid.uuid4().hex
    if not re.fullmatch(r'motion-[a-zA-Z0-9_-]{8,72}', identifier):
        raise ValueError('Invalid motion request identifier')
    root = run/'chat/motion/versions'/identifier
    root.mkdir(parents=True, exist_ok=True)
    completed = root/'result.json'
    if completed.exists():
        result = read(completed)
        if base_render_id and base_render_id != result['base_render_id']:
            raise ValueError('Request identifier belongs to another cover')
        verify_source(result['video'])
        return result
    progress = Progress(root/'progress.json')
    try:
        with progress.stage('prepare'):
            from scene import load_scene
            from manual_edit import copy_run
            from render import render
            with locked(run):
                scene = load_scene(run)
                result = read(run/'result.json')
                if not result.get('exported') or result['scene_sha256'] != sha(run/'scene.json') or result['final_sha256'] != sha(run/'final.png'):
                    raise ValueError('Finalize the static cover before motion rendering')
                if base_render_id and result['render_id'] != base_render_id:
                    raise ValueError('Cover changed; refresh before rendering motion')
                # Preserve visual gaps in the existing cover rather than blocking encoding.
                cover_review=result.get('visual_review') or {}
                cover_issues={'incomplete_objects':list(result.get('incomplete_objects',[])),
                              'visual_verdict':cover_review.get('verdict'),
                              'checked_entire_composition':cover_review.get('checked_entire_composition'),
                              'summary':cover_review.get('summary')}
                plan = make_plan(run, scene, result)
                saved_plan = root/'motion.json'
                if saved_plan.exists() and read(saved_plan) != plan:
                    raise ValueError('Request identifier belongs to another motion plan')
                save(saved_plan, plan)
                # Keep full task copies out of the deeply nested version folder:
                # VLM hashes and Reveal assets otherwise exceed Windows MAX_PATH.
                # The unique sibling snapshot also avoids copying itself recursively.
                candidate = run.parent/'.motion-snapshots'/uuid.uuid4().hex[:16]
                save(root/'snapshot.json', {'run': str(run), 'snapshot': str(candidate)})
                copy_run(run, candidate)
            render(candidate, publish_result=False, capture_layers=True)
            if sha(candidate/'final.png') != plan['base_final_sha256']:
                raise ValueError('Renderer no longer reproduces the frozen cover; finalize it first')
            captured = read(candidate/'result.json')['editor_layers']
            by_obj = {o['id']: o for o in scene['objects']}
            slot_map = {s['object_id']: s for s in plan['slots']}
            if not set(slot_map).issubset({r['id'] for r in captured}):
                raise ValueError('A Live photo is fused into a static layer')
            selected = [v for v in plan['videos'] if any(s['video_id']==v['id'] for s in plan['slots'])]
            by_video = {v['id']: v for v in selected}
            points = timeline(selected, plan['duration_us'])
            needed = {v['id']: {frame_index(v, t) for t in points} for v in selected}
            save(root/'timeline.json', {'at_us': points, 'duration_us': plan['duration_us'],
                                      'unique_source_frames': {key: sorted(value) for key, value in needed.items()}})
        cache = run/'chat/motion/cache'
        decoded = {}
        masked_ids = {s['video_id'] for s in plan['slots'] if needs_subject_mask(by_obj[s['object_id']])}
        streamed = [v for v in selected if v['id'] not in masked_ids]
        with progress.stage('decode'):
            decoded_count = 0
            decode_total = sum(len(needed[key]) for key in masked_ids)
            progress.count(0,decode_total,skipped=not decode_total,streamed_videos=len(streamed),
                           streamed_frames=sum(len(needed[v['id']]) for v in streamed))
            for video in selected:
                if video['id'] not in masked_ids:continue
                decoded[video['id']] = decode_needed(video, needed[video['id']], cache/'frames',
                    lambda done, total: progress.count(decoded_count+done, decode_total, video_id=video['id'],
                                                        source_done=done,source_total=total))
                decoded_count += len(decoded[video['id']])
                progress.count(decoded_count,decode_total,video_id=video['id'])
        mattes = {}; anchors = {}; inference_rows = []; warnings = []
        model = resolve_cutout_model(scene.get('cutout_model'))
        with progress.stage('matte'):
            total = sum(len(needed[key]) for key in masked_ids); done = 0
            progress.count(0,total)
            for video in selected:
                key = video['id']
                if key not in masked_ids:
                    continue
                mattes[key] = {}
                model_hash = sha(model) if model else ''
                previous_area = None
                for index, record in sorted(decoded[key].items()):
                    start = time.monotonic()
                    sequence_key = fingerprint([video['sha256'], model_hash, 'native-alpha-v1'])[:32]
                    destination = cache/'alpha'/sequence_key/f'{index:05d}.png'
                    receipt = destination.with_suffix('.json')
                    old = read(receipt) if receipt.exists() else {}
                    hit = bool(old.get('source_sha256')==record['sha256'] and destination.exists() and sha(destination)==old.get('sha256'))
                    if not hit:
                        hit = reuse_matte(run, sequence_key, index, record['sha256'], model_hash, destination)
                    if hit:
                        with Image.open(destination) as raw:
                            foreground = raw.convert('RGBA')
                    else:
                        with Image.open(record['file']) as raw:
                            foreground = cutout(raw.convert('RGBA'), record, model, cache/'birefnet')
                        verify_source(plan['cutout_model'])
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        foreground.save(destination)
                        save(receipt, {'source_sha256': record['sha256'], 'sha256': sha(destination),
                                       'video_sha256': video['sha256'], 'pts': record['pts'], 'model_sha256': model_hash})
                    bounds = foreground.getchannel('A').point(lambda v: 255 if v>8 else 0).getbbox()
                    if not bounds:
                        raise ValueError(f'Empty matte in {video["name"]} frame {index}')
                    area = float(np.asarray(foreground.getchannel('A'), dtype=np.float32).sum()/255)
                    if previous_area and abs(area-previous_area)/previous_area > .25:
                        warnings.append({'video_id': key, 'frame': index, 'reason': 'matte_area_jump'})
                    previous_area = area
                    if index == 0:
                        anchors[key] = bounds
                    mattes[key][index] = str(destination)
                    inference_rows.append({'video_id': key, 'index': index, 'pts': record['pts'],
                                           'elapsed_seconds': round(time.monotonic()-start, 3), 'cache_hit': hit,
                                           'alpha_area': round(area), 'bounds': bounds})
                    save(root/'matting.json', {'frames': inference_rows, 'warnings': warnings, 'anchors': anchors})
                    done += 1; progress.count(done, total, video_id=key, frame=index, cache_hit=hit)
                save(root/'matting.json', {'frames': inference_rows, 'warnings': warnings, 'anchors': anchors})
            if not total:
                progress.count(0, 0, skipped=True)
        scale = scene['canvas_size'][0]/scene['reference_size'][0]
        layers = []
        # Merge only consecutive static layers, preserving their relation to all video layers.
        static = Image.new('RGBA', tuple(scene['canvas_size']))
        for row in captured:
            oid = row['id']
            if oid in slot_map:
                layers.append(static); layers.append(oid)
                static = Image.new('RGBA', tuple(scene['canvas_size']))
            else:
                with Image.open(row['file']) as raw:
                    layer = raw.convert('RGBA')
                if by_obj[oid].get('editor_transform'):
                    layer = translate(layer, by_obj[oid], scale)
                static = Image.alpha_composite(static, layer)
        layers.append(static)
        composition_seconds = 0
        recent = {}
        def frames():
            nonlocal composition_seconds
            for number, at in enumerate(points):
                start = time.monotonic()
                canvas = Image.new('RGBA', tuple(scene['canvas_size']), 'white')
                for layer in layers:
                    if isinstance(layer, str):
                        slot = slot_map[layer]; key = slot['video_id']
                        index = frame_index(by_video[key], at)
                        cached = recent.get(layer)
                        if not cached or cached[0] != index:
                            subject_alpha = None
                            if key in readers:
                                image=readers[key].get(index)
                            else:
                                path = mattes[key][index] if slot['mode']=='cutout' else decoded[key][index]['file']
                                with Image.open(path) as raw:
                                    image = raw.convert('RGBA')
                                spec = by_obj[layer].get('photo_window_spec')
                                if spec and spec['subject']['mode'] == 'preserve' and subject_stroke(spec):
                                    with Image.open(mattes[key][index]) as raw:
                                        subject_alpha = raw.getchannel('A')
                            dynamic = fixed_photo_layer(by_obj[layer], image, anchors.get(key), scene,
                                                        subject_alpha=subject_alpha)
                            recent[layer] = (index, dynamic)
                        layer = recent[layer][1]
                    canvas = Image.alpha_composite(canvas, layer)
                image = canvas.convert('RGB')
                if number == 0:
                    image.save(root/'composed-first.png')
                    # Keep the exact finalized cover at t=0; subsequent geometry is pinned to it.
                    with Image.open(candidate/'final.png') as raw:
                        image = raw.convert('RGB')
                if number in {0, len(points)//2, len(points)-1}:
                    image.save(root/f'frame-{number:04d}.png')
                composition_seconds += time.monotonic()-start
                yield image
        with progress.stage('compose_encode'), ExitStack() as stack:
            readers={v['id']:stack.enter_context(NativeFrameReader(v)) for v in streamed}
            video_path = root/'final.mp4'
            encoding_started=time.monotonic()
            encode_frames(video_path, tuple(scene['canvas_size']), points,
                          plan['duration_us'], frames(), progress.count, validate=False)
            progress.data['stages']['compose_encode']['composition_seconds'] = round(composition_seconds, 3)
            progress.data['stages']['compose_encode']['encoding_mux_seconds'] = round(max(0,time.monotonic()-encoding_started-composition_seconds),3)
            progress.data['stages']['compose_encode']['stream_decode_seconds']=round(sum(r.elapsed_seconds for r in readers.values()),3)
            progress.data['stages']['compose_encode']['stream_decoded_frames']=sum(r.decoded_frames for r in readers.values())
        with progress.stage('validate'):
            metadata=verify_encoded(video_path,points,plan['duration_us'])
            for video in plan['videos']:
                verify_source(video)
            if plan.get('cutout_model'):
                verify_source(plan['cutout_model'])
            with Image.open(root/'composed-first.png') as dynamic, Image.open(candidate/'final.png') as cover:
                difference=np.abs(np.asarray(dynamic,dtype=np.int16)-np.asarray(cover,dtype=np.int16))
            alignment={'mean_absolute_error':round(float(difference.mean()),6),
                       'max_channel_error':int(difference.max()),'pixels_over_8':int((difference.max(2)>8).sum())}
            save(root/'cover-alignment.json',alignment)
            if alignment['mean_absolute_error']>1:
                warnings.append({'reason':'cover_alignment_needs_review','detail':alignment})
            output = {'schema_version': 'collage-motion-result-v1', 'id': identifier,
                      'status': 'needs_review', 'visual_verified': False, 'at': now(),
                      'cover_issues': cover_issues,
                      'base_render_id': plan['base_render_id'], 'base_scene_sha256': plan['base_scene_sha256'],
                      'base_final_sha256': plan['base_final_sha256'], 'duration_us': metadata['duration_us'],
                      'frame_count': metadata['frame_count'], 'time_policy': plan['time_policy'], 'audio': 'muted',
                      'video': {'file': str(video_path), 'sha256': sha(video_path)},
                      'canvas_size': scene['canvas_size'], 'encoded_size': metadata['size'],
                      'unique_source_frames': sum(len(v) for v in needed.values()),
                      'streamed_source_frames':sum(len(needed[v['id']]) for v in streamed),
                      'cached_source_frames':sum(len(v) for v in decoded.values()),
                      'matte_frames': len(inference_rows), 'matte_cache_hits': sum(r['cache_hit'] for r in inference_rows),
                      'warnings': warnings, 'cover_alignment':alignment,'timing': progress.data}
        progress.data['status'] = 'completed'
        progress.data['elapsed_seconds'] = round(time.monotonic()-progress.started,3)
        output['timing'] = progress.data
        with locked(run):
            current = read(run/'result.json')
            output['superseded_at_completion'] = current.get('render_id') != plan['base_render_id']
            save(completed, output)
            save(run/'chat/motion/latest.json', {'id': identifier})
        save(progress.path,progress.data)
        return output
    except Exception as exc:
        progress.data.update(status='failed', error=str(exc)); progress.write()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--request-id')
    parser.add_argument('--base-render-id')
    args = parser.parse_args()
    try:
        result = execute(args.run, args.request_id, args.base_render_id)
        print(json.dumps(result, ensure_ascii=True))
    except Exception as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=True))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
