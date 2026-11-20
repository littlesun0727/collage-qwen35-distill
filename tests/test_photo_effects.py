from copy import deepcopy

import pytest

from eval.photo_effects import evaluate_effects, summarize_effects
from eval.schema import parse_structure


def photo(ident, **fields):
    return dict(id=ident, kind='photo', bbox=[0, 0, 200, 200], description='photo') | fields


def structure(*objects):
    return parse_structure(dict(background={'type': 'transparent'}, objects=list(objects),
                                layer_order=[o['id'] for o in objects]))


def summary(gt, pred):
    sample = evaluate_effects(gt, pred, (720, 1280))
    return summarize_effects([sample], 'raw_first')


def test_removed_effects_and_missing_photo():
    left = photo('a', stroke={'color': '#FFFFFF', 'width': .01}, popout=[10, 10, 210, 180])
    right = photo('b', mode='cutout', bbox=[400, 400, 550, 550])
    gt = structure(left, right)
    own = summary(gt, gt)
    stripped = summary(gt, structure(photo('different_id')))
    assert own['quality']['popout.iou']['mean'] == 1
    assert stripped['valid_rate'] == 1
    assert stripped['detection']['popout']['end_to_end']['f1'] == 0
    cutout = stripped['detection']['mode.cutout']
    assert cutout['end_to_end']['fn'] == 1 and cutout['matched']['recall'] is None
    missing = summary(gt, parse_structure(None, present=False))
    assert missing['missing'] == 1 and missing['valid_rate'] == 0


def test_illegal_field_and_false_positive():
    gt = structure(photo('p'))
    bad = summary(gt, structure(photo('p', mode='bare', stroke={}, linked_to='ghost')))
    assert bad['valid_rate'] == 0
    assert bad['detection']['stroke']['end_to_end']['fp'] == 1
    assert bad['detection']['popout']['end_to_end']['f1'] is None


@pytest.mark.parametrize('shape,expected', [('ellipse', 0), ('heart', 180)])
def test_axis_direction(shape, expected):
    original = photo('a', shape=shape, axis=[[10, 10], [50, 60]])
    reversed_axis = deepcopy(original)
    reversed_axis['axis'].reverse()
    quality = summary(structure(original), structure(reversed_axis))['quality']
    assert quality['axis.angle_error_deg']['mean'] == pytest.approx(expected)


def test_explicit_five_point_star_matches_default():
    result = summary(structure(photo('p', shape='star')), structure(photo('p', shape='star', star_points=5)))
    assert result['detection']['star_points']['predicted'] == 0
    assert result['quality']['star_points.accuracy']['mean'] == 1
