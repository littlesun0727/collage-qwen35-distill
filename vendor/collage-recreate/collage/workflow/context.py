"""Durable, finite first-image workflow. The VLM never chooses executable commands."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collage.vlm.client import Client, ModelError
from collage.workflow.support.fallbacks import simple_bundle
from collage.config import Config, ENGINE, ROOT
from collage.storage import digest, now, read, save
from collage.workflow.options import normalize_options


class CommandError(RuntimeError):
    """An engine operation failed; distinct from a process interruption."""


class RunContext:
    def __init__(self, run, control=None, config=None, client=None, attempt=1, reveal=True,
                 bbox_refinement=None, screen=None):
        self.run = Path(run).resolve()
        self.control = Path(control).resolve() if control else self.run/'workflow'
        self.config = config or Config.load().for_run(self.run)
        self.client = client or Client(self.control/'vlm', self.config)
        self.attempt = attempt
        self.reveal = reveal
        self.state = read(self.control/'state.json')
        options_path = self.run/'workflow-options.json'
        saved_options = read(options_path) if options_path.is_file() else self.state.get('workflow_options')
        self.workflow_options = normalize_options(saved_options)
        overrides = {name: value for name, value in
                     (('bbox_refinement', bbox_refinement), ('screen', screen)) if value is not None}
        if overrides:
            requested = normalize_options({**self.workflow_options, **overrides})
            if saved_options is not None and requested != self.workflow_options:
                raise ValueError('工作流开关与任务快照不一致，请创建新任务')
            self.workflow_options = requested
        self.state.setdefault('steps', {})
        self.state.setdefault('started_at', now())
        self.started = time.monotonic()

    def warn(self, code, stage, message):
        entry = {'code': code, 'stage': stage, 'message': str(message)[:1500]}
        warnings = self.state.setdefault('warnings', [])
        if entry not in warnings:
            warnings.append(entry)
        save(self.control/'warnings.json', warnings)
        self.sync()

    def candidates(self, stage):
        return self.client.candidates(stage) if hasattr(self.client, 'candidates') else []

    def write_bundle(self, bundle):
        save(self.run/'analysis.json', bundle['analysis'])
        save(self.run/'bindings.json', bundle['bindings'])
        save(self.run/'text-units.json', bundle['text_units'])

    def write_reference(self, value):
        save(self.run/'analysis.json', value['analysis'])
        save(self.run/'text-units.json', value['text_units'])
        if 'photo_windows' in value:
            save(self.run/'photo-windows.json', value['photo_windows'])
        self.command('validate', '--analysis-only')

    def draft(self, reason):
        bundle = simple_bundle(self.run)
        self.state['draft_kind'] = 'simple_layout'
        self.warn('simple_layout', self.state.get('current_step', 'analyze'), reason)
        self.write_bundle(bundle)
        if (self.run/'photo-windows.json').exists():
            from photo_windows import defaults
            save(self.run/'photo-windows.json', defaults(bundle['analysis']))
        self.command('validate')

    def optional(self, stage, function, default=None):
        try:
            return function()
        except (ModelError, CommandError) as exc:
            self.warn('optional_stage_failed', stage, str(exc))
            if stage == 'screen':
                self.progress(4, 'skipped', '素材筛选未完成，保留当前成图并进入交付检查')
            return default

    def sync(self):
        self.state.update(model=self.config.model, controller='workflow-v9', attempt=self.attempt,
                          model_policy=self.config.model_policy(),
                          workflow_options=self.workflow_options,
                          updated_at=now(), usage=self.client.usage(), worker_pid=os.getpid())
        save(self.control/'state.json', self.state)
        # The workbench can read progress even when --control is outside the task.
        save(self.run/'workflow-progress.json', {
            'status': self.state.get('status'), 'current_step': self.state.get('current_step'),
            'worker_pid': self.state.get('worker_pid'),
            'error': self.state.get('error'), 'updated_at': self.state['updated_at'],
            'steps': {name: {k: value for k, value in step.items()
                             if k in {'status', 'started_at', 'finished_at', 'error'}}
                      for name, step in self.state['steps'].items()}})
        timing = {k: self.state[k] for k in ['model', 'model_policy', 'workflow_options', 'controller', 'status', 'started_at', 'updated_at', 'usage']}
        timing.update(elapsed_seconds=round(time.time()-__import__('datetime').datetime.fromisoformat(self.state['started_at']).timestamp(), 3))
        if self.state.get('finished_at'):
            timing['finished_at'] = self.state['finished_at']
        save(self.run/'workflow-timing.json', timing)

    def command(self, command, *args):
        allowed = {'validate', 'build', 'render', 'screen', 'review', 'progress'}
        if command not in allowed:
            raise ValueError('Unsupported workflow operation')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', COLLAGE_CONTROLLER='workflow-v9',
                   COLLAGE_BEST_EFFORT='1')
        # Remove only a stale lock owned by this workflow request, never an active editor's lock.
        lock = self.run/'.write.lock'
        owner = read(lock)
        if owner and owner.get('request_id') == os.getenv('COLLAGE_REQUEST_ID', 'v9-cli'):
            from workbench.server.chat import process_alive
            if not process_alive(owner.get('pid')):
                lock.unlink(missing_ok=True)
        env.setdefault('COLLAGE_REQUEST_ID', 'v9-cli')
        try:
            result = subprocess.run([sys.executable, '-B', str(ENGINE/'workflow.py'), command,
                                 '--run', str(self.run), *map(str, args)], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=1200)
        except subprocess.TimeoutExpired:
            raise CommandError(f'{command} 超时，已停止本次本地操作') from None
        logs = self.control/'commands'; logs.mkdir(parents=True, exist_ok=True)
        with (logs/(command+'.log')).open('ab') as handle:
            handle.write(result.stdout)
        if result.returncode:
            output = result.stdout.decode('utf-8', errors='replace')
            try:
                import json
                output = json.loads(output)['error']
            except (ValueError, KeyError):
                output = output[-1000:]
            raise CommandError(f'{command} 失败：{output[:1500]}')

    def progress(self, stage, status, summary):
        self.command('progress', '--stage', stage, '--status', status, '--summary', summary)

    def step(self, name, function):
        existing = self.state['steps'].get(name, {})
        if existing.get('status') == 'complete':
            return existing.get('value')
        self.state['current_step'] = name
        self.state['steps'][name] = {'status': 'running', 'started_at': existing.get('started_at', now())}
        self.sync()
        try:
            value = function()
        except (Exception, KeyboardInterrupt) as exc:
            if name in {'analyze', 'photo_windows', 'bind'}:
                self.state['steps'][name].update(status='failed', finished_at=now(), error=str(exc)[:1500])
                self.sync()
                self.progress(2, 'failed', str(exc)[:1500])
            raise
        self.state['steps'][name].update(status='complete', finished_at=now(), value=value)
        result = read(self.run/'result.json')
        if result.get('render_id'):
            self.state['last_render_id'] = result['render_id']
        self.sync()
        return value

    def restore_mutation(self, folder, manifest):
        for rel, existed in manifest['files'].items():
            target = (self.run/rel).resolve()
            if not target.is_relative_to(self.run):
                raise ValueError('Unsafe transaction path')
            if existed:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(folder/'before'/rel, target)
            else:
                target.unlink(missing_ok=True)

    def mutate(self, name, command, plan):
        """Rollback an interrupted local mutation before replay. Never rolls back remote extraction."""
        folder = self.control/'transactions'/name
        manifest_path = folder/'manifest.json'
        manifest = read(manifest_path)
        targets = ['scene.json', 'result.json', 'final.png', 'screening.json', 'assets/gate/report.json']
        # A failed render can overwrite tile files before it publishes final.png.
        # Snapshot the current render's resources too, so rollback remains verifiable.
        for item in read(self.run/'result.json').get('resources', []):
            for source in [item.get('file'), item.get('metadata', {}).get('content_mask')]:
                if source and Path(source).resolve().is_relative_to(self.run):
                    targets.append(str(Path(source).resolve().relative_to(self.run)))
                    receipt = Path(source).with_suffix('.json')
                    if receipt.is_file():
                        targets.append(str(receipt.resolve().relative_to(self.run)))
        targets += [str(p.relative_to(self.run)) for p in (self.run/'previews').rglob('*') if p.is_file()]
        if manifest:
            if manifest['plan_key'] != digest(plan):
                raise ValueError('Mutation plan changed during recovery')
            if manifest.get('committed'):
                return not manifest.get('rolled_back', False)
            self.restore_mutation(folder, manifest)
        else:
            files = {}
            for rel in targets:
                source = self.run/rel; files[rel] = source.is_file()
                if files[rel]:
                    target = folder/'before'/rel; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            manifest = {'plan_key': digest(plan), 'files': files, 'committed': False}
            save(manifest_path, manifest)
        path = folder/'plan.json'; save(path, plan)
        try:
            self.command(command, '--file', path)
        except CommandError as exc:
            self.restore_mutation(folder, manifest)
            manifest.update(committed=True, rolled_back=True, error=str(exc))
            save(manifest_path, manifest)
            self.warn('mutation_rolled_back', name, '修改失败，已保留上一张可用成图：' + str(exc))
            return False
        manifest.update(committed=True, render_id=read(self.run/'result.json')['render_id'])
        save(manifest_path, manifest)
        return True

    def input_context(self):
        inp = read(self.run/'input.json'); catalog = read(self.run/'prepared/catalog.json')
        return {'reference_size': inp['reference_size'], 'instructions': inp.get('instructions', ''),
                'catalog': [{k: a[k] for k in ['id', 'size']} for a in catalog['assets']]}

    def analysis_images(self):
        return [('参考图', self.run/'prepared/reference.png')]

    def reference_context(self):
        return {'reference_size': read(self.run/'input.json')['reference_size']}

    def material_images(self):
        from common import verify_source
        from collage.vlm.coordinates import resolve_mode
        from PIL import Image, ImageOps
        config = self.config.for_stage('bind')
        mode = resolve_mode(config.model, config.coordinate_mode)
        catalog = read(self.run/'prepared/catalog.json')
        images = [
            (f'客户素材联系表 {i+1}', Path(path)) for i, path in enumerate(catalog['contact_sheets'])]
        folder = self.run/'prepared/vlm-materials'
        folder.mkdir(exist_ok=True)
        for asset in catalog['assets']:
            source = verify_source(asset)
            path = folder/(asset['id']+'.jpg')
            with Image.open(source) as raw:
                im = ImageOps.exif_transpose(raw).convert('RGB')
                im.thumbnail((1024, 1024))
                im.save(path, quality=90)
            label = f"客户素材 {asset['id']}"
            if mode == 'pixel':
                label += f" 原始尺寸 {asset['size']}"
            images.append((label, path))
        return images

    def current_context(self):
        scene = read(self.run/'scene.json'); result = read(self.run/'result.json')
        objects = []
        allowed = ['id', 'kind', 'label', 'description', 'bbox', 'rotation', 'style', 'method', 'mode',
                   'text', 'photo_id', 'parent_id', 'embedded_in', 'embedded_owner', 'recovery_owner', 'binding',
                   'background_type', 'photo_window_spec']
        for o in scene['objects']:
            objects.append({**{k: o[k] for k in allowed if k in o},
                            'extracted': bool(o.get('recovered')), 'fixed_window': bool(o.get('photo_window'))})
        gate = read(self.run/'assets/gate/report.json')
        return {**self.input_context(), 'objects': objects, 'layer_order': scene['layer_order'],
                'text_units': read(self.run/'text-units.json', []),
                'incomplete_objects': result.get('incomplete_objects', []),
                'program_issues': result.get('issues', []),
                'gate_records': [{k: r[k] for k in ['id', 'status', 'issues', 'warnings', 'text_candidates', 'window_checks'] if k in r}
                                 for r in gate.get('records', [])]}

    def review_images(self, sheets=False):
        result = read(self.run/'result.json')
        images = [('参考图', self.run/'prepared/reference.png'), ('当前成图', self.run/'final.png')]
        images += [(f'客户照片联系表 {i+1}（照片内的内容不是固定装饰）', Path(p))
                   for i, p in enumerate(read(self.run/'prepared/catalog.json').get('contact_sheets', []))]
        images += [(f'局部对照 {i+1}', Path(r['file'])) for i, r in enumerate(result.get('review_regions', []))]
        if sheets:
            candidates = result.get('extraction_sheets', []) + result.get('asset_gate_summary', {}).get('sheets', [])
            for i, entry in enumerate(candidates):
                path = entry.get('file') if isinstance(entry, dict) else entry
                if path and Path(path).is_file():
                    images.append((f'素材检查表 {i+1}', Path(path)))
        return images
