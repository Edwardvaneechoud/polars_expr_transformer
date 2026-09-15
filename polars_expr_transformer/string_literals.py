"""Reading and writing string literals without ``eval``.

A formula is untrusted text: it arrives from a filter box, a formula node or a
shared link. Its literals therefore must never be handed to ``eval``, which
would turn a crafted quote into arbitrary Python. Everything here goes through
``ast.literal_eval`` on a literal this module built itself, so the worst a
hostile token can do is raise.
"""

import ast

QUOTE_CHARS = ("'", '"')


def is_string_literal(token: str) -> bool:
    """True if the token is a quoted string as produced by the tokenizer."""
    return len(token) > 1 and token[0] in QUOTE_CHARS and token[-1] == token[0]


def _escape_for_double_quotes(inner: str) -> str:
    """Escape the bare double quotes in a literal's body, leaving escapes alone.

    Walking the body two characters at a time is what keeps ``\\\\"`` (an escaped
    backslash followed by a bare quote) from being mistaken for ``\\"``.
    """
    out = []
    i = 0
    n = len(inner)
    while i < n:
        char = inner[i]
        if char == "\\":
            if i + 1 < n:
                out.append(inner[i : i + 2])
                i += 2
            else:
                out.append("\\\\")
                i += 1
            continue
        out.append('\\"' if char == '"' else char)
        i += 1
    return "".join(out)


def requote_double(token: str) -> str:
    """Rewrite a quoted token as an equivalent double-quoted Python literal."""
    if not is_string_literal(token):
        return token
    return '"' + _escape_for_double_quotes(token[1:-1]) + '"'


def parse_string_literal(token: str) -> str:
    """Return the value of a quoted token, honouring backslash escapes.

    Raises:
        ValueError: If the token is not a string literal, or its body is not a
            literal Python string once requoted.
    """
    if not is_string_literal(token):
        raise ValueError(f"Not a string literal: {token!r}")
    value = ast.literal_eval(requote_double(token))
    if not isinstance(value, str):
        raise ValueError(f"Not a string literal: {token!r}")
    return value


def parse_number_literal(token: str) -> int | float:
    """Return the value of a numeric token.

    Raises:
        ValueError: If the token is not a literal number.
    """
    value = ast.literal_eval(token)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Not a number literal: {token!r}")
    return value


def parse_literal(token: str) -> str | int | float:
    """Return the value of a literal token: a quoted string, or a number.

    Numbers are accepted here too because the classifier files any token it
    cannot recognise under ``string``, and ``eval`` used to give meaning to
    forms ``float()`` rejects, such as ``0x1f``.

    Raises:
        ValueError: If the token is not a literal string or number.
    """
    if is_string_literal(token):
        return parse_string_literal(token)
    return parse_number_literal(token)


def render_string_literal(value: str) -> str:
    """Render a Python string as source code, preferring double quotes."""
    rendered = repr(value)
    if rendered.startswith("'") and '"' not in value:
        rendered = '"' + rendered[1:-1] + '"'
    return rendered
