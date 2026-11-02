"""Read this file first: the complete, ordered first-image workflow.

Each handler below is implemented in stages/<name>.py. RunContext owns durable
state, engine commands and artifact I/O, not model instructions or stage logic.
"""

from collage.storage import digest, exclusive, file_sha, now, read, save
from collage.workflow.context import RunContext
from collage.workflow.options import normalize_options
from collage.workflow.stages import analyze, photo_windows, boxes, bind, build, screen, deliver

STAGES = (
    ('analyze', '识别参考图对象、归属和层级', True),
    ('photo_windows', '确定照片摆放、窗口与效果', True),
    ('boxes', '校准对象范围（可选）', True),
    ('bind', '客户照片选材与裁切', True),
    ('build', '提取固定素材并合成', False),
    ('screen', '检查提取相框的原照片残留（可选，默认关闭）', True),
    ('register', '校验文件、来源并交付', False),
)


class Workflow(RunContext):
    # The same handlers can be exercised individually without running later stages.
    analyze = analyze.execute
    photo_windows = photo_windows.execute
    apply_boxes = boxes.apply
    boxes = boxes.execute
    bind = bind.execute
    build = build.execute
    screen = screen.execute
    register = deliver.execute

    def run_stages(self):
        """The execution order, branching and local apply steps in one place."""
        self.progress(1, 'complete', '输入素材已准备完成')
        self.step('analyze', self.analyze)
        self.step('photo_windows', self.photo_windows)

        if self.workflow_options['bbox_refinement']:
            decision = self.step('boxes', lambda: self.optional('boxes', self.boxes,
                {'decision': 'keep', 'summary': '范围校准未完成，沿用已校验的分析'}))
            self.step('boxes_apply', lambda: self.apply_boxes(decision))
        else:
            for name in ('boxes', 'boxes_apply'):
                if self.state['steps'].get(name, {}).get('status') != 'complete':
                    self.state['steps'][name] = {'status': 'skipped', 'finished_at': now(),
                                                 'summary': 'bbox 校准已关闭，沿用当前分析框'}
            self.sync()

        self.step('bind', self.bind)
        self.step('build', self.build)
        if self.workflow_options['screen']:
            decision = self.step('screen', lambda: self.optional('screen', self.screen))
            if decision and decision['decisions']:
                self.step('screen_apply', lambda: self.mutate('screen', 'screen', decision))
            if decision:
                self.progress(4, 'complete', '素材筛选完成')
        else:
            for name in ('screen', 'screen_apply'):
                if self.state['steps'].get(name, {}).get('status') != 'complete':
                    self.state['steps'][name] = {'status': 'skipped', 'finished_at': now(),
                                                 'summary': '相框审核已关闭，直接使用提取素材'}
            self.sync()
            self.progress(4, 'skipped', '相框审核已关闭，直接使用提取素材并进入交付检查')
        self.step('register', self.register)

    def execute(self):
        """Validate the task snapshot, resume completed steps, then run the pipeline."""
        with exclusive(self.run/'.workflow.lock'):
            if self.state.get('controller') not in {None, 'workflow-v9'}:
                raise ValueError('此工程只续跑自身创建的任务；已有任务请在原工程中继续')
            if self.state.get('binding_contract') not in {None, bind.VERSION}:
                raise ValueError('照片绑定协议不兼容，请使用原版本续跑或创建新任务')
            options_path = self.run/'workflow-options.json'
            for previous in (self.state.get('workflow_options'), read(options_path) if options_path.is_file() else None):
                if previous is not None and normalize_options(previous) != self.workflow_options:
                    raise ValueError('工作流开关与任务快照不一致，请创建新任务')
            policy = self.config.model_policy()
            if not self.config.policy_matches(self.state.get('model_policy')):
                raise ValueError('阶段模型配置已改变，请使用原配置接续或创建新任务')
            snapshot = read(self.run/'workflow-models.json')
            if snapshot and not self.config.policy_matches(snapshot):
                raise ValueError('任务模型快照与运行配置不一致，请创建新任务')
            identity = digest({'input': file_sha(self.run/'input.json'),
                               'catalog': file_sha(self.run/'prepared/catalog.json'),
                               'model': self.config.model, 'reveal': self.reveal})
            if self.state.get('identity') not in {None, identity}:
                raise ValueError('任务输入、模型或提取模式已改变，请创建新任务')
            previous = read(self.run/'result.json')
            interrupted = any(s.get('status') == 'running' for s in self.state['steps'].values())
            if self.state.get('last_render_id') and previous.get('render_id') != self.state['last_render_id'] and not interrupted:
                raise ValueError('当前成图已被后续编辑改变，不能续跑首版')
            if self.state.get('status') == 'finished':
                if previous.get('final_sha256') != file_sha(self.run/'final.png') or previous.get('scene_sha256') != file_sha(self.run/'scene.json'):
                    raise ValueError('交付文件已改变')
                self.sync()
                return previous

            save(options_path, self.workflow_options)
            if not snapshot:
                save(self.run/'workflow-models.json', policy)
            self.state.update(identity=identity, status='running', error=None, binding_contract=bind.VERSION)
            self.state.pop('finished_at', None)
            self.sync()
            try:
                self.run_stages()
                self.state.update(status='finished', finished_at=now())
                self.sync()
                return read(self.run/'result.json')
            except (Exception, KeyboardInterrupt) as exc:
                self.state.update(status='failed', error=str(exc)[:1500], finished_at=now())
                self.sync()
                raise
