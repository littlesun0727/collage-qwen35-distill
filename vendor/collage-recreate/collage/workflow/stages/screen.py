"""阶段 6：提取相框候选 -> 保留、承托或替代决策。仅审核相框原照片残留。"""

from copy import deepcopy
from collage import contracts, prompts
from collage.storage import read, save


def execute(ctx):
    result = read(ctx.run/'result.json')
    records = read(ctx.run/'assets/gate/report.json').get('records', [])
    if not records:
        ctx.progress(4, 'skipped', '无远端提取素材，跳过筛选')
        return
    context = ctx.current_context()
    gate_ids = {r['id'] for r in records}
    screenable = sorted(o['id'] for o in context['objects'] if o['kind'] == 'frame'
                        and o['id'] in gate_ids)
    if not screenable:
        ctx.progress(4, 'skipped', '没有可由素材筛选节点处理的对象')
        return
    context['screenable_ids'] = screenable
    schema = deepcopy(contracts.SCREEN)
    schema['properties']['decisions']['items']['properties']['id'] = {'enum': screenable}
    ctx.progress(4, 'running', '仅检查相框原照片残留；其他装饰默认保留')
    def validate(answer):
        from screening import screen, retain_uncertain_frames
        answer, _ = retain_uncertain_frames(answer, records)
        fixed_text = {u['carrier_id'] for u in read(ctx.run/'text-units.json', [])
                      if u['treatment'] == 'fixed_artwork'}
        if ctx.state.get('text_inventory_advisory'):
            fixed_text.update(o['id'] for o in context['objects'] if o.get('method') == 'extract')
        if any(d['id'] in fixed_text and d['action'] == 'draw' for d in answer['decisions']):
            raise ValueError('完整文字贴纸不能降级为空白纸或通用字体；请保留可用候选或报告缺口')
        plan = {'render_id': result['render_id'], **answer}
        path = ctx.control/'screen-check.json'; save(path, plan)
        screen(ctx.run, path, dry_run=True)
    answer = ctx.client.ask('screen', prompts.render('screen'), context, ctx.review_images(True), schema, validate)
    from screening import retain_uncertain_frames
    answer, notes = retain_uncertain_frames(answer, records)
    for note in notes:ctx.warn('frame_retained_or_fallback', 'screen', note)
    return {'render_id': result['render_id'], **answer}
