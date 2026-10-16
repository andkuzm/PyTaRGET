import textwrap


def normalize_indentation(code):
    """
    Remove only what does not change the meaning of a repaired Python snippet: trailing whitespace on
    every line, blank lines around the snippet, and the margin shared by all its lines.
    The relative indentation between lines is kept and tabs are not converted to spaces.
    """
    lines = [line.rstrip() for line in code.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return textwrap.dedent("\n".join(lines))


def exact_match_flags(candidates, target):
    """(strict match, indentation-normalized match) of a case: does any candidate equal the target?"""
    normalized_target = normalize_indentation(target)
    strict = any(candidate == target for candidate in candidates)
    normalized = any(normalize_indentation(candidate) == normalized_target for candidate in candidates)
    return strict, normalized
