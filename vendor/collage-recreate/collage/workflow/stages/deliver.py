"""阶段 7：校验成图文件和客户照片来源，登记交付；不冒充整图视觉审核。"""

from collage.storage import read, save


def execute(ctx):
    """Validate and deliver the first image without claiming a visual review."""
    ctx.progress(5, 'running', '检查首版文件与客户照片来源')
    result = read(ctx.run/'result.json')
    issues = {}
    for oid in result.get('incomplete_objects', []):
        issues.setdefault(oid, '程序检测到素材缺失、占位或照片不可见')
    if not result.get('customer_photos_verified'):
        raise ValueError('客户照片来源校验失败，禁止交付')
    warnings = ctx.state.get('warnings', [])
    suffix = '；简化拼图草稿，尚未复刻参考布局。' if ctx.state.get('draft_kind') else ''
    if issues:
        suffix += f'；程序检测到 {len(issues)} 个未完成对象。'
    if warnings:
        suffix += f'；流程保留 {len(warnings)} 项告警，待优化。'
    review = {'schema_version': 'collage-review-v1', 'render_id': result['render_id'],
              'checked_entire_composition': False, 'verdict': 'needs_changes',
              'summary': '首版已生成，未做整图视觉复核；可在画布继续编辑。' + suffix,
              'items': [{'id': oid, 'action': 'unresolved', 'reason': reason} for oid, reason in issues.items()]}
    save(ctx.run/'review.json', review)
    # This engine command only checks files and registers the delivery record; no VLM call.
    ctx.command('review', '--file', ctx.run/'review.json')
    result = read(ctx.run/'result.json')
    category = 'simple_draft' if ctx.state.get('draft_kind') else 'with_issues' if issues or warnings else 'unreviewed'
    result['status'] = 'preview' if category == 'unreviewed' else 'needs_changes'
    result['workflow_delivery'] = {'mode': 'best_effort', 'render_id': result['render_id'],
        'category': category, 'review_status': 'skipped', 'warnings': warnings, 'draft_kind': ctx.state.get('draft_kind')}
    save(ctx.run/'result.json', result)
    ctx.progress(5, 'complete', review['summary'])
    ctx.progress(6, 'complete', '首版已交付，未做整图视觉复核；可进入画布编辑')
