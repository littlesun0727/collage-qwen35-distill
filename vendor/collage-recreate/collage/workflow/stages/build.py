"""阶段 5：分析、窗口、绑定 -> 素材提取与本地合成；不调用结构化 VLM。"""

from collage.storage import read
from collage.workflow.context import CommandError


def execute(ctx):
    ctx.progress(3, 'running', '按固定计划提取装饰并本地合成')
    needed = any(o.get('method') == 'extract' for o in read(ctx.run/'analysis.json')['objects'])
    try:
        if needed and ctx.reveal:
            ctx.command('build', '--reveal', '--reveal-layout', 'full', '--reveal-key-file', ctx.config.reveal_key_file)
        else:
            ctx.command('build', '--no-reveal')
    except CommandError as exc:
        ctx.warn('build_fallback', 'build', '制作失败，尝试保留布局的本地合成：' + str(exc))
        try:
            ctx.command('build', '--no-reveal')
        except CommandError as local_exc:
            ctx.draft('布局无法完成渲染，改为简化拼图：' + str(local_exc))
            ctx.command('build', '--no-reveal')
    for item in read(ctx.run/'result.json').get('resources', []):
        if item.get('metadata', {}).get('fallback'):
            ctx.warn('asset_fallback', 'build', item['id'] + '：' + item['metadata'].get('note', '素材已降级'))
    ctx.progress(3, 'complete', '首版已合成，进入素材筛选' if ctx.workflow_options['screen']
                 else '首版已合成，已直接使用提取素材')
