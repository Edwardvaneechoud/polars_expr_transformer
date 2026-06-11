"""Early syntax validation on the raw expression string.

The preprocessor rewrites the conditional keywords (if/then/else/elseif/endif)
into parenthesized function syntax before parsing. A structural mistake in the
user's expression (e.g. a misspelled 'if') therefore used to surface as a
confusing error about parentheses the user never typed. This module validates
the expression exactly as the user wrote it — before any rewriting — so errors
can report precise character positions into the original input.

Scanning skips string literals, ``[column]`` references and ``//`` comments.
Quote handling mirrors the ``("[^"]*"|'[^']*')`` regexes used throughout
``preprocess.py``: a quote runs to the next identical quote character, with no
escape handling, and the state persists across newlines. (``remove_comments``
processes quotes per line instead; the divergence only matters for inputs with
unterminated quotes, which fail downstream anyway.)

Known pre-existing quirk, intentionally untouched here: keyword rewriting in
``preprocess.py`` does not protect ``[column]`` references, so a column literally
named e.g. ``[then]`` is mangled by ``mark_special_tokens``. This scanner skips
``[...]``, so it neither masks nor worsens that.
"""

from typing import List, Tuple

from polars_expr_transformer.exceptions import ExpressionSyntaxError

_KEYWORDS = ("if", "then", "else", "elseif", "endif")
_TEMPLATE = "if <condition> then <value> else <value> endif"

# Conditional-frame states: waiting for 'then', inside the then-value,
# inside the else-value.
_COND, _THEN, _ELSE = "cond", "then", "else"


def _is_word_char(ch: str) -> bool:
    # Matches the \b semantics of the keyword regexes in preprocess.py
    # (Python's re is unicode-aware, so any alphanumeric counts).
    return ch.isalnum() or ch == "_"


def _scan_events(expression: str) -> List[Tuple[str, int, str]]:
    """Scan the raw expression into ordered ('paren'|'word', index, value) events.

    Content inside string literals, [column] references and // comments is
    skipped entirely.
    """
    events = []
    word_start = None
    i = 0
    n = len(expression)

    def flush_word(end: int):
        nonlocal word_start
        if word_start is not None:
            events.append(("word", word_start, expression[word_start:end]))
            word_start = None

    while i < n:
        ch = expression[i]
        if ch in ('"', "'"):
            flush_word(i)
            closing = expression.find(ch, i + 1)
            i = n if closing == -1 else closing + 1
        elif ch == "[":
            flush_word(i)
            closing = expression.find("]", i + 1)
            i = n if closing == -1 else closing + 1
        elif ch == "/" and i + 1 < n and expression[i + 1] == "/":
            flush_word(i)
            newline = expression.find("\n", i)
            i = n if newline == -1 else newline + 1
        elif ch in ("(", ")"):
            flush_word(i)
            events.append(("paren", i, ch))
            i += 1
        elif _is_word_char(ch):
            if word_start is None:
                word_start = i
            i += 1
        else:
            flush_word(i)
            i += 1
    flush_word(n)
    return events


def _validate_parentheses(expression: str, events: List[Tuple[str, int, str]]):
    open_positions = []
    for kind, idx, val in events:
        if kind != "paren":
            continue
        if val == "(":
            open_positions.append(idx)
        elif not open_positions:
            raise ExpressionSyntaxError(
                f"Unbalanced parentheses: ')' at position {idx + 1} has no matching '('.",
                expression=expression,
                position=idx,
                hint="Remove the extra ')' or add a '(' earlier in the expression.",
            )
        else:
            open_positions.pop()
    if open_positions:
        first = open_positions[0]
        if len(open_positions) == 1:
            message = f"Unbalanced parentheses: '(' at position {first + 1} is never closed."
        else:
            message = (
                f"Unbalanced parentheses: {len(open_positions)} '(' are never closed; "
                f"the first is at position {first + 1}."
            )
        raise ExpressionSyntaxError(
            message,
            expression=expression,
            position=first,
            hint="Add a matching ')'.",
        )


def _validate_conditional_structure(expression: str, events: List[Tuple[str, int, str]]):
    depth = 0
    # Stack of open if-blocks: [if_position, paren_depth_at_if, state]
    frames = []

    for kind, idx, val in events:
        if kind == "paren":
            depth += 1 if val == "(" else -1
            continue

        lowered = val.lower()
        if lowered not in _KEYWORDS:
            continue
        if val not in _KEYWORDS:
            raise ExpressionSyntaxError(
                f"Found '{val}' at position {idx + 1}. Keywords are case-sensitive: "
                f"use '{lowered}' instead of '{val}'.",
                expression=expression,
                position=idx,
            )

        keyword = val
        pos = idx + 1
        if keyword == "if":
            frames.append([idx, depth, _COND])
            continue

        if not frames:
            raise ExpressionSyntaxError(
                f"Found '{keyword}' at position {pos}, but there is no 'if' before it.",
                expression=expression,
                position=idx,
                hint=f"Every condition starts with 'if': {_TEMPLATE}. "
                "Check that 'if' is present and spelled correctly.",
            )

        frame = frames[-1]
        if_pos, if_depth, state = frame
        if depth > if_depth:
            raise ExpressionSyntaxError(
                f"Found '{keyword}' at position {pos} inside parentheses opened "
                f"after the 'if' at position {if_pos + 1}.",
                expression=expression,
                position=idx,
                hint=f"Close the parenthesis before '{keyword}'; "
                "parentheses cannot cut across if/then/else.",
            )
        if depth < if_depth:
            raise ExpressionSyntaxError(
                f"The 'if' at position {if_pos + 1} starts inside parentheses "
                f"that are closed before '{keyword}' at position {pos}.",
                expression=expression,
                position=idx,
                hint="Keep the whole if ... endif block inside the same parentheses.",
            )

        if keyword == "then":
            if state == _COND:
                frame[2] = _THEN
            elif state == _THEN:
                raise ExpressionSyntaxError(
                    f"Found 'then' at position {pos}, but the 'if' at position "
                    f"{if_pos + 1} already has a 'then'.",
                    expression=expression,
                    position=idx,
                    hint="Use 'elseif <condition> then <value>' to add another branch.",
                )
            else:
                raise ExpressionSyntaxError(
                    f"Found 'then' at position {pos} after 'else'.",
                    expression=expression,
                    position=idx,
                    hint="'else' takes a value directly, without 'then'.",
                )
        elif keyword == "elseif":
            if state == _THEN:
                frame[2] = _COND
            elif state == _COND:
                raise ExpressionSyntaxError(
                    f"Found 'elseif' at position {pos}, but the 'if' at position "
                    f"{if_pos + 1} is missing its 'then'.",
                    expression=expression,
                    position=idx,
                    hint="Write 'then <value>' after the if-condition first.",
                )
            else:
                raise ExpressionSyntaxError(
                    f"Found 'elseif' at position {pos} after 'else'.",
                    expression=expression,
                    position=idx,
                    hint="'elseif' branches must come before the final 'else'.",
                )
        elif keyword == "else":
            if state == _THEN:
                frame[2] = _ELSE
            elif state == _COND:
                raise ExpressionSyntaxError(
                    f"Found 'else' at position {pos}, but the 'if' at position "
                    f"{if_pos + 1} is missing its 'then'.",
                    expression=expression,
                    position=idx,
                    hint=f"Write it as: {_TEMPLATE}.",
                )
            else:
                raise ExpressionSyntaxError(
                    f"Found a second 'else' at position {pos} for the 'if' at "
                    f"position {if_pos + 1}.",
                    expression=expression,
                    position=idx,
                    hint="An 'if' can have only one 'else'. "
                    "Use 'elseif <condition> then <value>' for more branches.",
                )
        elif keyword == "endif":
            if state == _ELSE:
                frames.pop()
            elif state == _COND:
                raise ExpressionSyntaxError(
                    f"Found 'endif' at position {pos}, but the 'if' at position "
                    f"{if_pos + 1} is missing its 'then'.",
                    expression=expression,
                    position=idx,
                    hint=f"Write it as: {_TEMPLATE}.",
                )
            else:
                raise ExpressionSyntaxError(
                    f"Found 'endif' at position {pos}, but the 'if' at position "
                    f"{if_pos + 1} has no 'else' branch.",
                    expression=expression,
                    position=idx,
                    hint=f"Conditionals require an 'else': {_TEMPLATE}.",
                )

    if frames:
        if_pos, _, state = frames[-1]
        if state == _COND:
            raise ExpressionSyntaxError(
                f"The 'if' at position {if_pos + 1} is missing its 'then'.",
                expression=expression,
                position=if_pos,
                hint=f"Write it as: {_TEMPLATE}.",
            )
        if state == _THEN:
            raise ExpressionSyntaxError(
                f"The 'if' at position {if_pos + 1} is missing 'else' and 'endif'.",
                expression=expression,
                position=if_pos,
                hint=f"Conditionals require an 'else': {_TEMPLATE}.",
            )
        raise ExpressionSyntaxError(
            f"The 'if' at position {if_pos + 1} is missing its closing 'endif'.",
            expression=expression,
            position=if_pos,
            hint="End every conditional with 'endif'.",
        )


def validate_expression_syntax(expression: str) -> None:
    """Validate parenthesis balance and if/then/else/endif structure.

    Operates on the raw expression exactly as the user wrote it, so error
    messages can point at precise character positions.

    Args:
        expression: The expression string to validate.

    Raises:
        ExpressionSyntaxError: If parentheses are unbalanced or conditional
            keywords are misplaced, with the offending position and a hint.
    """
    events = _scan_events(expression)
    _validate_parentheses(expression, events)
    _validate_conditional_structure(expression, events)
