"""Structural checks for complete lettering/label units, separate from the engine schema."""

from jsonschema import Draft202012Validator


def validate(units, analysis, geometry_warnings=None):
    from collage.contracts import TEXT_UNITS
    Draft202012Validator(TEXT_UNITS).validate(units)
    objects = {item['id']: item for item in analysis['objects']}
    width, height = analysis['reference_size']
    order = analysis['layer_order']
    seen, represented, carriers = set(), set(), set()

    def fail(unit, message):
        raise ValueError(f"text_units.{unit['id']}: {message}")

    def geometry_issue(unit, message):
        if geometry_warnings is None:
            fail(unit, message)
        geometry_warnings.append({
            'object': unit['id'], 'code': 'text_unit_geometry',
            'message': f"text_units.{unit['id']}：文字单元与载体范围不一致，保留原框供后续复核；{message}",
            'bbox': unit['bbox'], 'carrier_id': unit['carrier_id'],
        })

    def covers(outer, inner):
        return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]

    def normalized(text):
        return ''.join(c for c in (text or '').casefold() if c.isalnum())

    for unit in units:
        if unit['id'] in seen:
            fail(unit, 'duplicate unit ID')
        seen.add(unit['id'])
        left, top, right, bottom = unit['bbox']
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            fail(unit, 'bbox must be a nonempty complete unit inside the reference')
        text = objects.get(unit['text_id'])
        carrier = objects.get(unit['carrier_id'])
        route = unit['treatment']
        if route == 'fixed_artwork':
            if unit['edit_requested']:
                fail(unit, 'fixed extraction cannot satisfy requested wording changes')
            if unit['text_id'] is not None:
                fail(unit, 'fixed lettering uses content as OCR metadata, not a second text object')
            if not carrier or carrier['kind'] not in {'overlay', 'frame'} or carrier.get('method') != 'extract':
                fail(unit, 'fixed artwork needs an extract overlay/frame for the complete lettering and backing')
            if not covers(carrier['bbox'], unit['bbox']):
                geometry_issue(unit, 'extract bbox omits part of the declared complete text unit')
            lettering = normalized(unit['content'])
            for item in objects.values():
                if item['kind'] != 'text':
                    continue
                linked = item.get('embedded_in') == carrier['id'] or item.get('parent_id') == carrier['id']
                a, b = item['bbox'], carrier['bbox']
                overlap = min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
                if linked or (overlap and lettering and normalized(item.get('text')) == lettering):
                    fail(unit, 'extracted lettering would be rendered twice; keep OCR only in text_units')
        else:
            if not text or text['kind'] != 'text' or text.get('embedded_in'):
                fail(unit, 'editable text needs an independent text object')
            if text['id'] in represented:
                fail(unit, 'one text object cannot represent multiple units')
            represented.add(text['id'])
            if unit['decorative_lettering']:
                fail(unit, 'distinctive lettering cannot be replaced with a generic font')
            if not covers(unit['bbox'], text['bbox']):
                geometry_issue(unit, 'complete unit bbox must include its text object')
            if route == 'plain_text':
                if unit['has_backing'] or unit['carrier_id'] is not None:
                    fail(unit, 'text with backing must use editable_label or fixed_artwork')
            else:
                if not unit['edit_requested']:
                    fail(unit, 'fixed labels with backing must be extracted unless a wording change is requested')
                if not unit['has_backing']:
                    fail(unit, 'editable_label must declare its backing')
                if not carrier or carrier['kind'] != 'overlay' or carrier.get('method', 'local') != 'local':
                    fail(unit, 'editable_label needs a local blank backing, not an extracted old inscription')
                if text.get('parent_id') != carrier['id']:
                    fail(unit, 'editable text must reference its backing through parent_id')
                if order.index(carrier['id']) >= order.index(text['id']):
                    fail(unit, 'backing must be below its text in layer_order')
                if not covers(unit['bbox'], carrier['bbox']):
                    geometry_issue(unit, 'complete unit bbox must include its backing')
        if carrier:
            if carrier['id'] in carriers:
                fail(unit, 'one complete label needs one unit record, not separate records for each word')
            carriers.add(carrier['id'])
    missing = {item['id'] for item in objects.values() if item['kind'] == 'text'} - represented
    if missing:
        raise ValueError('Every independent text object needs a text_units decision: ' + ', '.join(sorted(missing)))
