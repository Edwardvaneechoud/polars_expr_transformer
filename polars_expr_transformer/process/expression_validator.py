"""Early syntax validation on the raw expression string.

The preprocessor rewrites the conditional keywords (if/then/else/elseif/endif)
into parenthesized function syntax before parsing. A structural mistake in the
user's expression (e.g. a misspelled 'if') therefore used to surface as a
confusing error about parentheses the user never typed. This module validates
the expression exactly as the user wrote it — before any rewriting — so errors
can report precise character positions into the original input.

Scanning skips ``//`` comments, string literals and ``[column]`` references,
mirroring the pipeline's own handling so the validator never rejects an
expression the pipeline would have accepted:

- Comments are located with :func:`find_comment_spans`, the exact per-line
  algorithm of ``preprocess.remove_comments`` (which also calls it).
- String literals mirror the ``("[^"]*"|'[^']*')`` regexes used throughout
  ``preprocess.py``: a quote runs to the next identical quote character, with
  no escape handling, and the state persists across newlines.
- ``[column]`` references are quote-aware like ``preprocess.parse_pl_cols``:
  a ``]`` inside a quoted run does not terminate the reference.

The case-sensitivity check (rejecting e.g. ``IF``/``Then``) is intentionally
stricter than the old pipeline, which never recognized mixed-case keywords and
in some positions silently dropped them from the parsed expression. A clear
error is preferable to silently ignoring part of the user's input.

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


def find_comment_spans(expression: str) -> List[Tuple[int, int]]:
    """Find the (start, end) spans of // comments in the expression.

    This is the single source of truth for comment detection: it implements
    the per-line scan of ``preprocess.remove_comments`` (which calls it), so
    the validator and the pipeline can never disagree about what is a comment.
    Quote state resets on every line; a // inside a quoted run on its line is
    not a comment. Spans run to the end of the line, excluding the newline.
    """
    spans = []
    offset = 0
    for line in expression.split("\n"):
        inside_single_quote = False
        inside_double_quote = False
        pos = 0
        while pos < len(line):
            char = line[pos]
            if char == "'" and not inside_double_quote:
                inside_single_quote = not inside_single_quote
            elif char == '"' and not inside_single_quote:
                inside_double_quote = not inside_double_quote
            elif (
                char == "/"
                and pos + 1 < len(line)
                and line[pos + 1] == "/"
                and not inside_single_quote
                and not inside_double_quote
            ):
                spans.append((offset + pos, offset + len(line)))
                break
            pos += 1
        offset += len(line) + 1
    return spans


def _scan_events(expression: str) -> List[Tuple[str, int, str]]:
    """Scan the raw expression into ordered ('paren'|'word', index, value) events.

    Content inside // comments, string literals and [column] references is
    skipped entirely. Comments are masked first (using the same per-line
    algorithm as preprocess.remove_comments), matching the pipeline, where
    comment removal runs before everything else.
    """
    events = []
    word_start = None
    i = 0
    n = len(expression)

    masked = bytearray(n)
    for start, end in find_comment_spans(expression):
        for k in range(start, end):
            masked[k] = 1

    def flush_word(end: int):
        nonlocal word_start
        if word_start is not None:
            events.append(("word", word_start, expression[word_start:end]))
            word_start = None

    while i < n:
        if masked[i]:
            flush_word(i)
            i += 1
            continue
        ch = expression[i]
        if ch in ('"', "'"):
            flush_word(i)
            j = i + 1
            while j < n and (masked[j] or expression[j] != ch):
                j += 1
            i = n if j >= n else j + 1
        elif ch == "[":
            flush_word(i)
            # Quote-aware like parse_pl_cols: a ']' inside a quoted run does
            # not terminate the column reference.
            j = i + 1
            quote = None
            while j < n:
                c = expression[j]
                if masked[j]:
                    pass
                elif quote is not None:
                    if c == quote:
                        quote = None
                elif c in ('"', "'"):
                    quote = c
                elif c == "]":
                    break
                j += 1
            i = n if j >= n else j + 1
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
