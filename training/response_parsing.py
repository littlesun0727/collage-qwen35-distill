"""Shared transport-wrapper parsing for preview rendering and scoring."""
import re


PARSING_POLICY = 'empty_think_and_outer_json_fence_v1'


def unwrap_response(response):
    """Remove only empty thinking and a complete outer JSON code fence."""
    text = response.strip()
    wrappers = []
    empty_think = re.match(r'^<think>\s*</think>\s*', text)
    if empty_think:
        text = text[empty_think.end():]
        wrappers.append('空 think 标签')
    fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
        wrappers.append('JSON 代码围栏')
    return text, wrappers
