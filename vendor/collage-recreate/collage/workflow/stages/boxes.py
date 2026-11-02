"""阶段 3（可选）：对象裁片 -> 范围校准建议 -> 本地应用；不改变对象身份。"""

from collage import contracts, prompts
from collage.storage import read, save, now


def execute(ctx):
    from collage.workflow.support import box_geometry as box_refinement
    from collage.workflow.support.reference_cleanup import normalize_boxes
    notes = []
    bundle = {'analysis': read(ctx.run/'analysis.json'),
              'text_units': read(ctx.run/'text-units.json', [])}
    if (ctx.run/'photo-windows.json').exists():
        # The VLM carries windows through unchanged; validation reconciles
        # fade coverage deterministically after its geometry corrections.
        bundle['photo_windows'] = read(ctx.run/'photo-windows.json')
    folder = ctx.control/'boxes-refinement'
    context, images, manifest = box_refinement.prepare(ctx.run, folder, bundle)
    if not context['targets']:
        answer = {'decision': 'keep', 'summary': '没有需要局部精修的前景对象。'}
        save(ctx.control/'boxes-decision.json', answer)
        return answer

    def merged(local):
        notes.clear()
        answer, unresolved, mapping = box_refinement.merge(local, bundle, manifest)
        notes.extend(unresolved)
        if answer['decision'] == 'replace':notes.extend(normalize_boxes(answer, bundle))
        contracts.validate_box_decision(answer, bundle, ctx.run, geometry_warnings=notes)
        return answer, mapping

    from collage.vlm.response_normalization import normalize_boxes_answer
    normalization = {'normalize_answer': normalize_boxes_answer} if (ctx.run/'structure.json').exists() else {}
    local = ctx.client.ask('boxes', prompts.render('boxes'), context, images,
                            box_refinement.schema([row['id'] for row in manifest['crops']]),
                            lambda value: merged(value), **normalization)
    answer, mapping = merged(local)
    save(folder/'local-answer.json', local)
    save(folder/'mapping.json', {'objects': mapping, 'unresolved': notes})
    # The corrected bundle is the stage output; persist before mutating inputs.
    save(ctx.control/'boxes-decision.json', answer)
    if notes:
        save(ctx.control/'boxes-cleanup.json', {'at': now(), 'notes': notes})
        for note in notes:ctx.warn(note['code'], 'boxes', note['message'])
    return answer

def apply(ctx, answer):
    if answer['decision'] == 'replace':
        current = {'analysis': read(ctx.run/'analysis.json'),
                   'text_units': read(ctx.run/'text-units.json', [])}
        if (ctx.run/'photo-windows.json').exists():
            current['photo_windows'] = read(ctx.run/'photo-windows.json')
        contracts.validate_box_decision(answer, current, ctx.run, geometry_warnings=[])
        ctx.write_reference(answer)
        if 'photo_windows' in answer:
            from photo_windows import previews
            previews(ctx.run, answer['analysis'], answer['photo_windows'])
    ctx.progress(2, 'running', '参考分析及叠框已完成，准备一次完成照片选材与裁切')
