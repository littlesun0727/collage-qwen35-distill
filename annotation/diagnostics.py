"""Explainable checks. Geometry warnings are hypotheses, never visual verdicts."""
import hashlib
import json
import re

from eval.schema import parse_structure

CHECKS = ('background', 'elements', 'kinds', 'bbox', 'relations', 'layer_order', 'occlusion_pairs')


def overlap(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa, bb = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    area = x*y
    return area/(aa+bb-area) if aa+bb-area else 0, area/min(aa, bb) if min(aa, bb)>0 else 0


def diagnose(s, review=None):
    review = review or {}
    parsed = parse_structure(s)
    objects = s.get('objects', [])
    by_id = {o.get('id'): o for o in objects if isinstance(o, dict)}
    ranks = {v: i for i, v in enumerate(s.get('layer_order', []))}
    issues = []

    def add(code, ids, severity, title, detail, advice, action=None):
        evidence = [by_id.get(i) for i in ids]
        if 'order' in code:
            evidence.append(s.get('layer_order'))
        if not ids:
            evidence.append(s.get('background'))
        fingerprint = hashlib.sha256(json.dumps([code, ids, detail, evidence], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
        issues.append(dict(key=fingerprint, code=code, ids=ids, severity=severity, title=title,
                           detail=detail, advice=advice, action=action,
                           acknowledged=fingerprint in review.get('acknowledged_issues', [])))

    for error in parsed.errors:
        match = re.search(r'objects\[(\d+)\]', error)
        ids = [objects[int(match[1])].get('id')] if match and isinstance(objects[int(match[1])], dict) else []
        if 'bbox' in error:
            title, advice = '元素框无效', '选中元素重新画框，确保框有宽度和高度，并位于图片内。'
        elif 'photo_id' in error:
            title, advice = '相框关联未填写或指向无效对象', '在元素面板勾选这个相框实际承载的照片；若只是描边，请重新判断类别。'
        elif 'linked_to' in error:
            title, advice = '出框来源无效', '在元素面板重新选择同一照片的来源；独立贴纸不需要出框关联。'
        elif 'layer_order' in error:
            title, advice = '图层清单与元素不一致', '使用“补齐图层清单”，再检查实际上下关系。'
        elif 'background' in error:
            title, advice = '底图设置不完整', '打开背景设置，选择底图类型并补充颜色或说明。'
        else:
            title, advice = '结构字段需要修正', '定位元素后检查类别和关联；详细字段见下方依据。'
        add('schema-'+error, ids, 'error', title, error, advice,
            'repair_order' if 'layer_order' in error else None)
    owners = {}
    valid = [o for o, p in zip(objects, parsed.objects) if p.box is not None and p.id]
    for o in valid:
        oid, kind = o['id'], o.get('kind')
        if kind == 'frame':
            targets = o.get('photo_id', [])
            if isinstance(targets, list):
                for target in targets:
                    if target in owners:
                        add('two-frames', [oid, owners[target], target], 'error', '同一照片绑定了两个框体',
                            '正文规则要求一张内口照片最多归属一个 frame。', '取消不属于这个照片窗口的关联。')
                    owners[target] = oid
                    t = by_id.get(target)
                    if t in valid and overlap(o['bbox'], t['bbox'])[1] == 0:
                        add('frame-apart', [oid, target], 'warning', '相框与关联照片完全不相交',
                            '两者的框没有交集，可能选错了照片或定位偏了。', '对照高亮的两个元素，重选相框内的照片或调整框。')
            desc = o.get('description', '')
            if re.search('描边|虚线|轮廓线|手绘框', desc) and not re.search('纸卡|相纸|信封|胶片|播放器|厚纸', desc):
                add('outline-or-frame', [oid]+(targets if isinstance(targets, list) else []), 'warning',
                    '这是独立框体，还是照片的描边？', '描述提到了线条边框；prompt 的正文与示例在这里存在口径冲突。',
                    '有独立纸卡、材质或一体设计才保留为相框；如果含字样、附属设计也要整体判断。单纯沿照片轮廓的线条并入照片说明，再删除多余框体。')
        target = o.get('linked_to')
        if target in by_id:
            t = by_id[target]
            if t.get('linked_to'):
                add('linked-chain', [oid, target], 'error', '出框来源又是一个出框前景',
                    'linked_to 应直接指向原始照片，不形成出框链或循环。', '把来源改成保留完整摄影环境的那张照片。')
            if ranks.get(oid, -1) < ranks.get(target, -1):
                add('foreground-order', [target, oid], 'error', '出框前景位于来源照片下方',
                    '当前图层把前景放在来源的下面。', '将前景移到来源照片上方，再核对它遮住了谁。', 'foreground_above')
            if t in valid and overlap(o['bbox'], t['bbox'])[1] == 0:
                add('linked-apart', [oid, target], 'warning', '出框与来源没有相交',
                    '连续出框通常在原照片窗口处有衔接，这两个框完全分离。', '若只是同一个人不同姿态的两张照片，应取消出框关联。')
        b = o['bbox']
        if min(b[2]-b[0], b[3]-b[1]) < 3:
            add('thin-box', [oid], 'warning', '框非常细，确认是否漏了主体',
                '框的一条边小于画布对应尺寸的 0.3%。', '放大检查：若确实是细线则保留，否则拖动框边。')
    for i, a in enumerate(valid):
        for b in valid[i+1:]:
            if a.get('kind') != b.get('kind') or a.get('linked_to') == b['id'] or b.get('linked_to') == a['id']:
                continue
            iou, _ = overlap(a['bbox'], b['bbox'])
            if iou > .85:
                add('near-duplicate', [a['id'], b['id']], 'warning', '两个同类元素的框几乎重合',
                    f'框的 IoU 为 {iou:.0%}。这只是几何疑点，也可能是正常叠放。',
                    '分别点选高亮元素，核对是否重复标注；确实是不同素材则点“已核对，保留”。')
    # These are targeted manual review tasks, explicitly distinct from errors.
    for o in valid:
        target = o.get('linked_to')
        if target:
            add('review-breakout', [o['id']]+([target] if target in by_id else []), 'hint',
                '确认这个出框主体与来源连续',
                '当前把这个元素标成同一张照片延伸出的前景；这决定它是否共用客户素材。',
                '看接缝处的姿态、衣服和身体是否连续。若是同一人不同照片，取消出框来源；确实连续则保留。')
        if o.get('kind')=='frame' and len(o.get('photo_id', []))>1:
            add('review-multiwindow', [o['id']]+o['photo_id'], 'hint',
                f'一个框体关联了 {len(o["photo_id"])} 张照片',
                '多窗口框体允许这种关联；仅仅挨在一起的照片不能都归给它。',
                '逐一核对高亮照片是否确实由同一个完整框体承载；取消误连的照片。')
    obscured = [o for o in valid if o.get('kind')=='photo' and re.search('被.*遮挡|被.*覆盖', o.get('description',''))]
    if obscured:
        o = obscured[0]
        add('review-occluded-window', [o['id']], 'hint', '这张照片有遮挡，确认框没有缩成残片',
            o.get('description', '')[:190],
            '保留摄影环境的照片应框完整窗口；抠图则框完整主体组合。有证据的遮挡部分仍属于原单元，不要只框露出来的碎片。')
    if s.get('background', {}).get('type')=='photographic':
        add('review-background', [], 'hint', '确认底图是一张连续的全幅照片',
            s.get('background', {}).get('description', '')[:180],
            '检查四角和间隙的场景是否连续。若只是几张照片拼满画布，应各自保留为照片，背景选“没有独立全画布底图”；固定设计底图应选择保留原样。')
    return issues


def candidates(s):
    valid = [o for o, p in zip(s.get('objects', []), parse_structure(s).objects) if p.box]
    result = []
    for i, a in enumerate(valid):
        for b in valid[i+1:]:
            _, fraction = overlap(a['bbox'], b['bbox'])
            if fraction < .08:
                continue
            related = a.get('linked_to') == b['id'] or b.get('linked_to') == a['id'] or a['id'] in b.get('photo_id', []) or b['id'] in a.get('photo_id', [])
            result.append({'ids': [a['id'], b['id']], 'overlap_fraction': round(fraction, 3),
                           'reason': '已有相框 / 出框关联' if related else '两个框有交集，实际像素未必互相遮挡',
                           'priority': (1 if related else 0) + fraction})
    return sorted(result, key=lambda r: -r['priority'])
