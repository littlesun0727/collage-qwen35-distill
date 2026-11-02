"""Object roles shared by analysis, selection and both renderers."""


def is_photo_background(obj):
    return obj.get('kind') == 'background' and obj.get('background_type') == 'photo'


def is_customer_photo(obj):
    return obj.get('kind') == 'photo' or is_photo_background(obj)


def fill_background_canvas(analysis):
    """Only explicit backgrounds receive full-canvas placement."""
    changed = []
    for obj in analysis['objects']:
        if obj.get('kind') != 'background':
            continue
        box = [0, 0, *analysis['reference_size']]
        if obj['bbox'] != box or obj.get('rotation', 0):
            changed.append(obj['id'])
        obj.update(bbox=box, rotation=0)
    return changed
