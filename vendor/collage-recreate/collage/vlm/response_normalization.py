"""Recognize redundant JSON containers without changing their business content."""

from copy import deepcopy


def unwrap_object(value, required_keys):
    result = deepcopy(value)
    if (isinstance(result, list) and len(result) == 1
            and isinstance(result[0], dict) and set(required_keys) <= result[0].keys()):
        return result[0]
    return result


def normalize_boxes_answer(value):
    return unwrap_object(value, {'movement_unit', 'objects'})
