"""Optional Live adapter. Static prepare receives only photos and pinned posters."""
from pathlib import Path
from fractions import Fraction
import bisect
import uuid
import time
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from common import read, sha, verify_source
from motion_io import save

MICROSECOND = Fraction(1, 1_000_000)
VIDEO_EXTENSIONS = {'.mp4', '.mov'}


def av_module():
    try:
        import av
        return av
    except ImportError as exc:
        raise ValueError('Live media requires requirements.txt') from exc


def oriented(frame):
    # PyAV exposes the display matrix's counterclockwise angle; Pillow uses CCW.
    image = frame.to_image().convert('RGBA')
    angle = frame.rotation
    if angle % 90:
        raise ValueError('Only right-angle video display rotation is supported')
    return image.rotate(angle, expand=True) if angle else image


def inspect_video(path, poster=None, progress=None):
    """Decode native PTS once. Audio duration never sets the video duration."""
    path = Path(path).resolve()
    source_hash = sha(path)
    av = av_module()
    rows = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('No video stream')
        stream = container.streams.video[0]
        if stream.width * stream.height > 20_000_000:
            raise ValueError('Video exceeds 20 million pixels')
        if stream.duration and stream.duration * stream.time_base > 60:
            raise ValueError('Live source must be at most 60 seconds')
        first = None
        reported=0
        for index, frame in enumerate(container.decode(stream)):
            if index >= 7200 or frame.pts is None:
                raise ValueError('Unsupported frame count or missing video timestamps')
            timestamp = Fraction(frame.pts) * frame.time_base
            if first is None:
                first = timestamp
                cover = oriented(frame)
                size, rotation = list(cover.size), frame.rotation
                if poster:
                    Path(poster).parent.mkdir(parents=True, exist_ok=True)
                    info = PngInfo(); info.add_text('live_source_sha256', source_hash)
                    cover.convert('RGB').save(poster, pnginfo=info)
            displayed = [frame.height,frame.width] if frame.rotation % 180 else [frame.width,frame.height]
            if frame.rotation != rotation or displayed != size:
                raise ValueError('Video changes resolution or orientation during playback')
            at = round((timestamp-first) / MICROSECOND)
            if at < 0 or (rows and at <= rows[-1]['at_us']) or at > 60_000_000:
                raise ValueError('Video timestamps must increase within 60 seconds')
            rows.append({'index': index, 'pts': frame.pts, 'at_us': at,
                         'duration_us': round(frame.duration * frame.time_base / MICROSECOND)})
            if progress and (index==0 or time.monotonic()-reported>=1):
                progress(index+1,stream.frames or None);reported=time.monotonic()
        if not rows:
            raise ValueError('Empty video')
        last = rows[-1]
        if last['duration_us'] <= 0:
            end = round((stream.start_time or 0) * stream.time_base / MICROSECOND
                        + (stream.duration or 0) * stream.time_base / MICROSECOND
                        - first / MICROSECOND)
            last['duration_us'] = end-last['at_us']
        if last['duration_us'] <= 0:
            raise ValueError('Cannot determine final frame duration')
        for left, right in zip(rows, rows[1:]):
            left['duration_us'] = right['at_us']-left['at_us']
        duration = last['at_us']+last['duration_us']
        if not 0 < duration <= 60_000_000:
            raise ValueError('Invalid Live duration')
        return {'file': str(path), 'sha256': source_hash, 'duration_us': duration,
                'size': size, 'rotation': rotation, 'time_base': str(stream.time_base),
                'average_rate': str(stream.average_rate), 'frames': rows,
                'frame_count': len(rows), 'has_audio': bool(container.streams.audio)}


def prepare_live(reference, materials, run, width=1200, instructions='', cutout_model=None, progress_file=None):
    from prepare import prepare
    from common import now
    started=time.monotonic()
    def publish(stage,**details):
        if progress_file:save(progress_file,{'stage':stage,'updated_at':now(),'elapsed_seconds':round(time.monotonic()-started,1),**details})
    run = Path(run).resolve()
    if run.exists():
        raise ValueError('prepare-live requires a new task directory')
    roots = [Path(p).resolve() for p in materials]
    if any(run.is_relative_to(p) for p in roots):
        raise ValueError('Task outputs must be outside source directories')
    videos = sorted({p.resolve() for root in roots for p in root.rglob('*')
                     if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS})
    if not videos:
        publish('contact_sheets')
        result=prepare(reference, materials, run, width, instructions, cutout_model)
        publish('complete',assets=result['asset_count']);return result
    # Sibling cache is outside the not-yet-created run; original prepare stays intact.
    cache = run.parent / '.live-media' / (run.name+'-'+uuid.uuid4().hex[:12])
    records = []
    for number,video in enumerate(videos):
        details={'done':number,'total':len(videos),'current':video.name}
        publish('video',**details)
        digest = sha(video)
        if any(r['sha256'] == digest for r in records):
            continue
        poster = cache / (digest + '.png')
        record = inspect_video(video, poster,progress=lambda done,total:publish('video',**details,frames_decoded=done,frames_total=total))
        record.update(id='live_'+digest[:24], name=video.name,
                      poster={'file': str(poster), 'sha256': sha(poster)}, poster_at_us=0)
        records.append(record)
    poster_hashes = [r['poster']['sha256'] for r in records]
    if len(set(poster_hashes)) != len(poster_hashes):
        raise ValueError('Different videos have identical posters; choose distinct cover frames first')
    publish('contact_sheets',done=len(videos),total=len(videos))
    result = prepare(reference, [*materials, str(cache)], run, width, instructions, cutout_model)
    assets_by_hash = {asset['sha256']: asset['id'] for asset in read(run/'prepared/catalog.json')['assets']}
    for record in records:
        record['asset_id'] = assets_by_hash[record['poster']['sha256']]
    save(run/'media/manifest.json', {'schema_version': 'collage-live-v1', 'videos': records,
                                    'audio': 'muted', 'poster_policy': 'first_native_frame'})
    result['live_count'] = len(records)
    publish('complete',assets=result['asset_count'],videos=len(records))
    return result


def timeline(videos, duration_us):
    """Union of native source transitions, looping each source without resampling."""
    points = {0}
    for video in videos:
        duration = video['duration_us']
        if duration <= 0:
            raise ValueError('Invalid source duration')
        for offset in range(0, duration_us, duration):
            points.update(offset+row['at_us'] for row in video['frames']
                          if offset+row['at_us'] < duration_us)
    return sorted(points)


def frame_index(video, at_us):
    return bisect.bisect_right([r['at_us'] for r in video['frames']],
                              at_us % video['duration_us'])-1


class NativeFrameReader:
    """Sequential native frames, retaining only the current image per video.

    A backwards request means a short source has looped; reopen it and decode
    from the beginning so VFR timestamps and frame identities remain exact.
    """
    def __init__(self, video):
        verify_source(video)
        self.video=video;self.container=None;self.frames=None;self.index=-1;self.image=None
        self.decoded_frames=0;self.converted_frames=0;self.elapsed_seconds=0

    def _restart(self):
        self.close()
        self.container=av_module().open(self.video['file'])
        self.frames=enumerate(self.container.decode(video=0));self.index=-1

    def __enter__(self):
        self._restart();return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.container is not None:self.container.close()
        self.container=None;self.frames=None;self.image=None

    def get(self, index):
        if type(index) is not int or not 0<=index<len(self.video['frames']):raise ValueError('Invalid native frame index')
        if self.container is None:raise RuntimeError('Native frame reader is closed')
        if index==self.index:return self.image
        start=time.monotonic()
        try:
            if index<self.index:self._restart()
            for current,frame in self.frames:
                self.decoded_frames+=1
                if current>=len(self.video['frames']) or frame.pts!=self.video['frames'][current]['pts'] or frame.rotation!=self.video['rotation']:
                    raise ValueError('Decoded frame metadata changed')
                displayed=[frame.height,frame.width] if frame.rotation%180 else [frame.width,frame.height]
                if displayed!=self.video['size']:raise ValueError('Decoded frame dimensions changed')
                if current==index:
                    # Match the existing lossless RGB PNG path, including alpha policy.
                    self.image=oriented(frame).convert('RGB').convert('RGBA');self.index=current
                    self.converted_frames+=1;return self.image
            raise ValueError('Video ended before required frames')
        finally:self.elapsed_seconds+=time.monotonic()-start


def decode_needed(video, indices, cache, progress=lambda *args: None):
    verify_source(video)
    av = av_module()
    folder = Path(cache)/video['sha256']
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder/'frames.json'
    saved = read(manifest) if manifest.exists() else {}
    result = {}
    missing = set()
    for index in sorted(indices):
        record = saved.get(str(index), {})
        valid_identity = record.get('index')==index and record.get('pts')==video['frames'][index]['pts'] and record.get('source_sha256')==video['sha256']
        if valid_identity and Path(record['file']).exists() and sha(record['file']) == record['sha256']:
            result[index] = record
        else:
            missing.add(index)
    if missing:
        with av.open(video['file']) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index > max(missing):
                    break
                if index not in missing:
                    continue
                if frame.pts != video['frames'][index]['pts'] or frame.rotation != video['rotation']:
                    raise ValueError('Decoded frame metadata changed')
                path = folder/f'{index:05d}.png'
                info = PngInfo(); info.add_text('live_source_sha256', video['sha256'])
                oriented(frame).convert('RGB').save(path, pnginfo=info,compress_level=1)
                record = {'file': str(path), 'sha256': sha(path), 'index': index,
                          'pts': frame.pts, 'source_sha256': video['sha256']}
                result[index] = saved[str(index)] = record
                save(manifest, saved)
                progress(len(result), len(indices))
    if len(result) != len(indices):
        raise ValueError('Video ended before required frames')
    return result


def encode_frames(path, size, points, duration_us, images, progress=lambda *args: None, validate=True):
    """Explicit packet durations include the clipped final frame; no CFR conversion."""
    av = av_module()
    width, height = size
    encoded_size = (width+width%2, height+height%2)
    durations = {at: end-at for at, end in zip(points, points[1:]+[duration_us])}
    with av.open(str(path), 'w', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264')
        stream.width, stream.height = encoded_size
        stream.pix_fmt = 'yuv420p'
        stream.time_base = stream.codec_context.time_base = MICROSECOND
        stream.codec_context.max_b_frames = 0
        stream.options = {'crf': '18', 'preset': 'fast', 'tune': 'zerolatency'}
        def mux(packet):
            at = round(packet.pts * packet.time_base / MICROSECOND)
            packet.duration = round(durations[at] * MICROSECOND / packet.time_base)
            container.mux(packet)
        for index, (at, image) in enumerate(zip(points, images)):
            if image.size != (width, height):
                raise ValueError('Motion frame size changed')
            if encoded_size != image.size:
                padded = Image.new('RGB', encoded_size, 'white')
                padded.paste(image, (0, 0)); image = padded
            frame = av.VideoFrame.from_image(image.convert('RGB'))
            frame.pts, frame.time_base = at, MICROSECOND
            for packet in stream.encode(frame):
                mux(packet)
            progress(index+1, len(points))
        for packet in stream.encode():
            mux(packet)
    return verify_encoded(path,points,duration_us) if validate else {'size':list(encoded_size)}


def verify_encoded(path, points, duration_us):
    metadata = inspect_video(path)
    if metadata['frame_count'] != len(points) or metadata['duration_us'] != duration_us:
        raise ValueError('Encoded timing differs from the native-frame timeline')
    if [f['at_us'] for f in metadata['frames']] != points or metadata['has_audio']:
        raise ValueError('Encoded timestamps/audio differ from the motion plan')
    return metadata
