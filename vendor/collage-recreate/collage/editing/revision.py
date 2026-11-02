"""Data-only natural-language revision planner and visual reviewer."""

import json
from collage import contracts
from collage import prompts
from collage.vlm.client import Client
from collage.config import Config


def infer(mode, root, context, images, client=None):
    # root = <task>/chat/requests/<request-id>; edits inherit the task's subsequent stage.
    task_root = root.parents[2] if root.parent.name == 'requests' and root.parent.parent.name == 'chat' else root
    client = client or Client(root/'vlm', Config.load().for_run(task_root))
    if mode == 'plan':
        ids = {o['id'] for o in context['objects']}
        assets = {a['id'] for a in context.get('catalog', [])}
        def validate(answer):
            changed = [c['id'] for c in answer['changes']]
            removed = answer['remove_ids']
            if answer['decision'] != 'edit':
                if changed or removed or answer['layer_order']:
                    raise ValueError('Non-edit decisions cannot mutate objects')
                return
            if len(set(changed)) != len(changed) or len(set(removed)) != len(removed):
                raise ValueError('Duplicate change/remove ID')
            if not set(changed+removed) <= ids or set(changed) & set(removed):
                raise ValueError('Unknown or conflicting object IDs')
            for change in answer['changes']:
                if not change['changes']:
                    raise ValueError('Empty changes')
                if 'asset_id' in change['changes'] and change['changes']['asset_id'] not in assets:
                    raise ValueError('Unknown customer asset')
            order = answer['layer_order']
            if order and (len(order) != len(ids-set(removed)) or set(order) != ids-set(removed)):
                raise ValueError('layer_order must list every retained object once')
        answer = client.ask('edit_plan', prompts.render('edit_plan'), context, images, contracts.EDIT, validate)
        # Keep the proven queue adapter contract, after validating real JSON changes.
        answer['changes'] = [{'id': c['id'], 'reason': c['reason'],
                              'changes_json': json.dumps(c['changes'], ensure_ascii=False)} for c in answer['changes']]
        return answer
    if mode == 'review':
        scene = context['actual_scene']
        answer = client.ask('edit_review', prompts.render('edit_review'),
                            context, images, contracts.FINAL_REVIEW,
                            lambda a: contracts.validate_review(a, scene, allow_adjustments=False))
        issues = {i['id']: i['reason'] for i in answer['issues']}
        for oid in context.get('incomplete_objects', []):
            issues.setdefault(oid, '程序检测到未解决素材')
        answer['issues'] = [{'id': k, 'reason': v} for k, v in issues.items()]
        if issues:
            answer['verdict'] = 'needs_changes'
        return answer
    raise ValueError('Unsupported model operation')
