"""Project business contracts to GPT's strict subset, then restore optional fields."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from math import prod
from jsonschema import Draft202012Validator


MISSING = object()


class Impossible(ValueError):
    pass


def merge(left, right):
    """Conjoin the finite object constraints used by the workflow."""
    if left is False or right is False:
        return False
    if left is True:
        return deepcopy(right)
    if right is True:
        return deepcopy(left)
    result = deepcopy(left)
    for key, value in right.items():
        if key not in result:
            result[key] = deepcopy(value)
        elif key == 'properties':
            result[key] = {**result[key], **{
                name: merge(result[key].get(name, {}), field) for name, field in value.items()}}
        elif key == 'required':
            result[key] = list(dict.fromkeys(result[key] + value))
        elif key in {'enum', 'type'}:
            a = result[key] if isinstance(result[key], list) else [result[key]]
            b = value if isinstance(value, list) else [value]
            common = [item for item in a if item in b]
            if not common:
                raise Impossible('Conflicting field constraints')
            result[key] = common[0] if key == 'type' and len(common) == 1 else common
        elif key in {'minimum', 'exclusiveMinimum', 'minItems', 'minLength'}:
            result[key] = max(result[key], value)
        elif key in {'maximum', 'exclusiveMaximum', 'maxItems', 'maxLength'}:
            result[key] = min(result[key], value)
        elif key in {'description', 'title'}:
            result[key] = value
        elif result[key] != value:
            raise Impossible('Conflicting schema constraint: ' + key)
    if 'const' in result and not Draft202012Validator(result).is_valid(result['const']):
        raise Impossible('Constant does not satisfy field constraints')
    if any(result.get('properties', {}).get(name) is False for name in result.get('required', [])):
        raise Impossible('A required field cannot be forbidden')
    return result


def branches(schema):
    """Turn conditional rules over finite discriminators into explicit alternatives."""
    base = {key: deepcopy(value) for key, value in schema.items() if key != 'allOf'}
    rules = schema['allOf']
    names = []
    for rule in rules:
        if set(rule) - {'if', 'then', 'else'} or 'if' not in rule:
            raise ValueError('GPT Schema 适配只接受有限枚举字段上的条件约束')
        condition = rule['if']
        if set(condition) - {'properties', 'required'}:
            raise ValueError('GPT Schema 条件包含未适配的规则')
        names.extend(condition.get('properties', {}))
    names = list(dict.fromkeys(names))
    domains = []
    for name in names:
        field = base.get('properties', {}).get(name, {})
        values = [field['const']] if 'const' in field else field.get('enum')
        if not isinstance(values, list) or not values:
            raise ValueError('GPT Schema 条件字段必须具有有限 enum：' + name)
        domains.append(values + ([MISSING] if name not in base.get('required', []) else []))
    if not domains or prod(map(len, domains)) > 64:
        raise ValueError('GPT Schema 条件分支数量超出适配范围')
    results = []
    for values in product(*domains):
        assignment = {name: value for name, value in zip(names, values) if value is not MISSING}
        branch = deepcopy(base)
        try:
            for rule in rules:
                selected = 'then' if Draft202012Validator(rule['if']).is_valid(assignment) else 'else'
                branch = merge(branch, rule.get(selected, {}))
            for name, value in zip(names, values):
                if value is MISSING:
                    if name in branch.get('required', []):
                        raise Impossible('Required discriminator cannot be absent')
                    branch['properties'][name] = {'type': 'null'}
                else:
                    branch['properties'][name] = merge(branch['properties'][name], {'const': value})
        except Impossible:
            continue
        results.append(branch)
    if not results:
        raise ValueError('GPT Schema 条件没有可满足的分支')
    return results


def schema_children(node, transform):
    result = deepcopy(node)
    for group in ('properties', '$defs', 'definitions'):
        if isinstance(result.get(group), dict):
            result[group] = {key: transform(value) for key, value in result[group].items()}
    for key in ('items', 'additionalProperties'):
        if isinstance(result.get(key), dict):
            result[key] = transform(result[key])
    for key in ('anyOf', 'oneOf', 'allOf', 'prefixItems'):
        if isinstance(result.get(key), list):
            result[key] = [transform(value) for value in result[key]]
    return result


def project(schema):
    """Keep conditional branches; represent absent optional fields with null."""
    def convert(node):
        if not isinstance(node, dict):
            return node
        if 'allOf' in node:
            return {'anyOf': [convert(branch) for branch in branches(node)]}
        if set(node) & {'if', 'then', 'else', 'not', 'oneOf', 'dependentSchemas', 'dependentRequired'}:
            raise ValueError('GPT Schema 包含未适配的组合约束')
        result = schema_children(node, convert)
        for key in ('$schema', '$id'):
            result.pop(key, None)
        if isinstance(result.get('const'), (list, dict)):
            # Exact reference_size stays in the prompt and original local validator.
            result.pop('const')
        if 'type' not in result and ('enum' in result or 'const' in result):
            values = result.get('enum', [result.get('const')])
            types = list(dict.fromkeys(
                'null' if v is None else 'boolean' if isinstance(v, bool) else
                'integer' if isinstance(v, int) else 'number' if isinstance(v, float) else
                'string' if isinstance(v, str) else 'array' if isinstance(v, list) else 'object'
                for v in values))
            result['type'] = types[0] if len(types) == 1 else types
        if isinstance(result.get('properties'), dict):
            original_required = node.get('required', [])
            for name, field in result['properties'].items():
                if field is False:
                    if name in original_required:
                        raise ValueError('GPT Schema 必填字段不能同时禁止出现')
                    result['properties'][name] = {'type': 'null'}
                    continue
                if name not in original_required and not Draft202012Validator(field).is_valid(None):
                    result['properties'][name] = {'anyOf': [field, {'type': 'null'}]}
            result['required'] = list(result['properties'])
            result['additionalProperties'] = False
        return result
    return deduplicate(convert(schema))


def deduplicate(schema):
    """Reuse large repeated branches so style fields do not inflate the API schema."""
    def identity(node):
        return json.dumps(node, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    counts = Counter()
    def count(node):
        if isinstance(node, dict):
            counts[identity(node)] += 1
            schema_children(node, count)
        return node
    count(schema)
    definitions = {}
    def compact(node, inline=False):
        if not isinstance(node, dict):
            return node
        key = identity(node)
        if not inline and len(key) > 600 and counts[key] > 1:
            name = 'shared_' + sha256(key.encode()).hexdigest()[:16]
            if name not in definitions:
                definitions[name] = schema_children(node, compact)
            return {'$ref': '#/$defs/' + name}
        return schema_children(node, compact)
    result = compact(schema, inline=True)
    if definitions:
        result['$defs'] = {**result.get('$defs', {}), **definitions}
    return result


def restore(value, original):
    """Remove only synthetic optional nulls; retain genuine nullable business values."""
    if not isinstance(original, dict):
        return value
    if 'anyOf' in original:
        candidates = [restore(value, branch) for branch in original['anyOf']]
        return next((v for v, branch in zip(candidates, original['anyOf'])
                     if Draft202012Validator(branch).is_valid(v)), candidates[0])
    if isinstance(value, dict):
        fields = original.get('properties', {})
        result = {}
        for name, item in value.items():
            field = fields.get(name, {})
            if (name in fields and item is None and name not in original.get('required', [])
                    and not Draft202012Validator(field).is_valid(None)):
                continue
            result[name] = restore(item, field)
        return result
    if isinstance(value, list):
        return [restore(item, original.get('items', {})) for item in value]
    return value
