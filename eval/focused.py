"""Six-question report, following the 2026-11-03 archive's v3 definitions.

One-to-one, group-aware IoU >= .5 matching; fixed GT cohorts and per-image
means. This module does not change the legacy v2 CLI or fabricate a repair pass.
"""
from collections import Counter

from .matching import match
from .schema import parse_structure, validate_occlusions

REFERENCE_URL = (
    'https://github.com/littlesun0727/reports-achrive/blob/main/reports/'
    '2026-11-03/collage-demo/collage-qwen35-distill/runs/eval/'
    'covers137-six-metrics-auto-occlusions-20261102/index.html'
)
GROUPS = {'photo': {'photo'}, 'fixed': {'frame', 'overlay', 'text'}}


def ratio(a, b):
    return a / b if b else None


def overlap_pairs(gt):
    """Freeze bbox positive-area overlap + GT layer order (an automatic proxy)."""
    if not gt.valid:
        raise ValueError('Cannot derive order pairs from invalid GT: ' + '; '.join(gt.errors))
    objects = {obj.id: obj for obj in gt.objects}
    pairs = []
    for i, lower in enumerate(gt.order):
        a = objects[lower].box
        for upper in gt.order[i + 1:]:
            b = objects[upper].box
            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                pairs.append([lower, upper])
    return pairs


def evaluate_focused(sample_id, gt, pred, pairs):
    if not gt.valid:
        raise ValueError(f'Invalid GT {sample_id}: ' + '; '.join(gt.errors))
    validate_occlusions(pairs, gt)
    result = {'id': sample_id, 'present': pred.present, 'structure_valid': pred.valid,
              'errors': pred.errors, 'matches': []}
    g_to_p, p_to_g = {}, {}
    gu, pu = gt.unique_ids(), pred.unique_ids()
    for group, kinds in GROUPS.items():
        gs = [o for o in gt.objects if o.kind in kinds]
        ps = [o for o in pred.objects if o.kind in kinds]
        matches = match(gs, ps, threshold=.5)
        result[group] = {'gt_count': len(gs), 'pred_count': len(ps),
                         'matched_count': len(matches), 'extra_predictions': len(ps) - len(matches),
                         'recall': ratio(len(matches), len(gs)),
                         'matched_mean_iou': ratio(sum(v for _, _, v in matches), len(matches))}
        for g, p, iou in matches:
            go, po = gs[g], ps[p]
            result['matches'].append({'gt_id': go.id, 'pred_id': po.id, 'iou': iou, 'group': group})
            # Geometry can still score duplicate-ID boxes; references cannot.
            if gu.get(go.id) == go.index and pu.get(po.id) == po.index:
                g_to_p[go.index] = po.index
                p_to_g[po.index] = go.index

    counts = Counter(pred.order)
    rank = {oid: i for i, oid in enumerate(pred.order) if counts[oid] == 1}
    outcomes = Counter()
    order_details = []
    for lower, upper in pairs or []:
        a, b = g_to_p.get(gu[lower]), g_to_p.get(gu[upper])
        if a is None or b is None:
            outcome = 'unmatched_endpoint'
        else:
            lo, hi = pred.objects[a].id, pred.objects[b].id
            if lo not in rank or hi not in rank:
                outcome = 'missing_or_duplicate_order_id'
            else:
                outcome = 'correct' if rank[lo] < rank[hi] else 'reversed'
        outcomes[outcome] += 1
        order_details.append({'lower': lower, 'upper': upper, 'outcome': outcome})
    result['layer_order'] = {'gt_count': len(pairs or []), 'correct': outcomes['correct'],
                             'rate': ratio(outcomes['correct'], len(pairs or [])),
                             'pairs_available': pairs is not None, 'details': order_details}

    def mapped_target(edge):
        if not edge.syntax_valid:
            return None
        if (pred.background.get('type') == gt.background.get('type') == 'photographic'
                and edge.target == pred.background.get('id')):
            return ('background', gt.background.get('id'))
        idx = pu.get(edge.target)
        if idx in p_to_g:
            return ('object', gt.objects[p_to_g[idx]].id)
        return None

    frames = [o for o in gt.objects if o.kind == 'frame']
    connected = correct = 0
    frame_details = []
    matched_frames = set()
    for obj in frames:
        idx = g_to_p.get(obj.index)
        edges = []
        if idx is not None and pred.objects[idx].kind == 'frame':
            matched_frames.add(idx)
            edges = [e for e in pred.edges['photo_id'] if e.source == idx]
        linked = any(e.syntax_valid for e in edges)
        expected = {('object', e.target) for e in gt.edges['photo_id'] if e.source == obj.index}
        actual = [mapped_target(e) for e in edges]
        exact = bool(expected) and len(actual) == len(expected) and set(actual) == expected
        connected += linked
        correct += exact
        frame_details.append({'gt_id': obj.id, 'connected': linked, 'correct': exact})
    result['frame_links'] = {'gt_count': len(frames), 'connected': connected, 'correct': correct,
                             'connected_rate': ratio(connected, len(frames)),
                             'correct_rate': ratio(correct, len(frames)), 'details': frame_details}

    linked_gt = gt.edges['linked_to']
    linked_correct = 0
    linked_details = []
    matched_linked = set()
    for edge in linked_gt:
        idx = g_to_p.get(edge.source)
        edges = [e for e in pred.edges['linked_to'] if e.source == idx] if idx is not None else []
        if edges:
            matched_linked.add(idx)
        target = ('background' if gt.background.get('type') == 'photographic'
                  and edge.target == gt.background.get('id') else 'object', edge.target)
        ok = len(edges) == 1 and mapped_target(edges[0]) == target
        linked_correct += ok
        linked_details.append({'gt_id': gt.objects[edge.source].id, 'correct': ok})
    result['linked_to'] = {'gt_count': len(linked_gt), 'correct': linked_correct,
                           'rate': ratio(linked_correct, len(linked_gt)), 'details': linked_details}
    result['diagnostics'] = {
        'extra_photo': result['photo']['extra_predictions'],
        'extra_fixed': result['fixed']['extra_predictions'],
        'extra_frame_claims': sum(o.kind == 'frame' and o.index not in matched_frames for o in pred.objects),
        'extra_linked_to_claims': sum(e.source not in matched_linked for e in pred.edges['linked_to']),
        'invalid_kind_predictions': sum(o.kind not in GROUPS['photo'] | GROUPS['fixed'] for o in pred.objects),
        'order_outcomes': dict(outcomes),
    }
    return result


def summarize_focused(samples):
    def mean(group, key):
        values = [s[group][key] for s in samples if s[group][key] is not None]
        return {'mean': ratio(sum(values), len(values)), 'eligible_images': len(values),
                'scored_images': len(values), 'total_images': len(samples)}

    def total(group, key):
        return sum(s[group][key] for s in samples)

    summary = {
        'version': 'collage-structure-eval-v3-preview',
        'output_usability': {
            'first_pass_core_valid_rate': ratio(sum(s['structure_valid'] for s in samples), len(samples)),
            'final_core_valid_rate': None, 'prediction_stage': 'raw_first',
            'total_images': len(samples), 'valid_images': sum(s['structure_valid'] for s in samples),
            'predictions_present': sum(s['present'] for s in samples),
        },
    }
    for group in GROUPS:
        summary[group] = {key: mean(group, key) for key in ('recall', 'matched_mean_iou')}
        summary[group].update({key: total(group, key) for key in ('gt_count', 'matched_count', 'extra_predictions')})
    for group in ('layer_order', 'linked_to'):
        summary[group] = {'rate': mean(group, 'rate'), 'gt_count': total(group, 'gt_count'),
                          'correct': total(group, 'correct')}
    summary['frame_links'] = {key: mean('frame_links', key) for key in ('connected_rate', 'correct_rate')}
    summary['frame_links'].update({key: total('frame_links', key) for key in ('gt_count', 'connected', 'correct')})
    summary['diagnostics'] = {key: total('diagnostics', key) for key in
                              ('extra_photo', 'extra_fixed', 'extra_frame_claims',
                               'extra_linked_to_claims', 'invalid_kind_predictions')}
    outcomes = Counter()
    for sample in samples:
        outcomes.update(sample['diagnostics']['order_outcomes'])
    summary['diagnostics']['order_outcomes'] = dict(outcomes)
    return summary
