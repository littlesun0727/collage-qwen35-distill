"""Frame photo lists, with read compatibility for legacy scalar and parent links."""


def photo_ids(carrier):
    value = carrier.get('photo_id', [])
    if isinstance(value, str):
        return [value]
    return value if isinstance(value, list) else []


def carrier_photos(carrier, objects):
    ids = photo_ids(carrier)
    return [p for p in objects if p.get('kind') == 'photo'
            and (p['id'] in ids or p.get('parent_id') == carrier['id'])]


def frame_issues(objects):
    by_id = {o['id']: o for o in objects}
    owners = {}
    issues = []
    for frame in objects:
        if frame.get('kind') != 'frame':
            continue
        ids = photo_ids(frame)
        if not ids or any(not isinstance(pid, str) or by_id.get(pid, {}).get('kind') != 'photo'
                          for pid in ids):
            issues.append({'id': frame['id'], 'fields': ['photo_id'],
                           'message': 'frame photo_id must list existing photo IDs'})
            continue
        for photo in carrier_photos(frame, objects):
            previous = owners.setdefault(photo['id'], frame['id'])
            if previous != frame['id']:
                issues.append({'id': frame['id'], 'fields': ['photo_id'],
                               'message': photo['id'] + ': photo cannot belong to multiple frames'})
    return issues
