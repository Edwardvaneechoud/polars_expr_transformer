import re
from copy import deepcopy
from typing import Callable, List, Tuple

from polars_expr_transformer.process.expression_validator import (
    find_comment_spans,
    validate_expression_syntax,
)


def replace_double_spaces(func_string: str) -> str:
    """
    Replace all double spaces in the input string with single spaces.

    Args:
        func_string: The string to process.

    Returns:
        The processed string with double spaces replaced by single spaces.
    """
    while '  ' in func_string:
        func_string = func_string.replace('  ', ' ')
    return func_string


def remove_comments(input_string: str) -> str:
    """
    Remove comments starting with // from each line of the input string.
    Preserves string literals, only removing comments outside of quotes.

    Args:
        input_string: The input string that may contain comments.

    Returns:
        A string with all comments removed, preserving the rest of the content.
    """
    parts = []
    last = 0
    for start, end in find_comment_spans(input_string):
        parts.append(input_string[last:start])
        last = end
    parts.append(input_string[last:])
    return ''.join(parts)


def normalize_whitespace(input_string: str) -> str:
    """
    Normalize whitespace in the input string by replacing newlines and tabs with spaces
    and ensuring no double spaces exist.

    Args:
        input_string: The string to normalize.

    Returns:
        A string with normalized whitespace.
    """
    # Replace newlines with spaces
    result = input_string.replace('\n', ' ')
    # Replace tabs with spaces
    result = result.replace('\t', ' ')
    # Remove double spaces
    return replace_double_spaces(result)


def _find_quote_end(input_string: str, start: int) -> int:
    """
    Find the closing quote of the literal opening at ``start``.

    Mirrors the ``("[^"]*"|'[^']*')`` regexes this module used to rely on:
    a quote runs to the next identical quote character, with no escape
    handling, and an unterminated quote is not a literal at all.

    Args:
        input_string: The string being scanned.
        start: Index of the opening quote character.

    Returns:
        The index of the closing quote, or -1 if the literal is unterminated.
    """
    return input_string.find(input_string[start], start + 1)


def _find_bracket_end(input_string: str, start: int) -> int:
    """
    Find the ``]`` closing the column reference opening at ``start``.

    Quote-aware like :func:`parse_pl_cols`: a ``]`` inside a quoted run does not
    terminate the reference.

    Args:
        input_string: The string being scanned.
        start: Index of the opening ``[``.

    Returns:
        The index of the closing bracket, or -1 if the reference is unclosed.
    """
    quote_char = None
    for pos in range(start + 1, len(input_string)):
        char = input_string[pos]
        if quote_char is not None:
            if char == quote_char:
                quote_char = None
        elif char in "\"'":
            quote_char = char
        elif char == ']':
            return pos
    return -1


def split_protected_spans(input_string: str) -> List[str]:
    """
    Split a string into alternating unprotected and protected parts.

    Protected parts are quoted string literals and ``[column]`` references —
    the two kinds of span that carry the user's own text and must survive the
    rewrites that only apply to ordinary expression syntax (keyword marking,
    operator spacing, whitespace removal). Without the bracket half of this, a
    column named ``[if Flag]`` came out of :func:`mark_special_tokens` as
    ``[$if$( Flag]``.

    Even indices hold unprotected text and odd indices hold the protected
    spans, mirroring ``re.split`` with a capturing group, so callers can
    rewrite ``parts[::2]`` and join the parts back together.

    Args:
        input_string: The string to split.

    Returns:
        A list of parts, always of odd length, starting and ending with
        (possibly empty) unprotected text.
    """
    parts = []
    unprotected = ''
    pos = 0

    while pos < len(input_string):
        char = input_string[pos]
        if char in "\"'":
            end = _find_quote_end(input_string, pos)
        elif char == '[':
            end = _find_bracket_end(input_string, pos)
        else:
            end = -1

        if end == -1:
            unprotected += char
            pos += 1
            continue

        parts.append(unprotected)
        parts.append(input_string[pos:end + 1])
        unprotected = ''
        pos = end + 1

    parts.append(unprotected)
    return parts


def replace_outside_protected_spans(input_string: str, transform: Callable[[str], str]) -> str:
    """
    Apply a transformation to everything but string literals and ``[columns]``.

    Args:
        input_string: The string to process.
        transform: A function rewriting one run of unprotected text.

    Returns:
        The processed string, with protected spans passed through untouched.
    """
    parts = split_protected_spans(input_string)
    parts[::2] = [transform(part) for part in parts[::2]]
    return ''.join(parts)


def add_spaces_around_logical_operators(input_string: str) -> str:
    """
    Add spaces around logical operators (and, or) in the input string,
    but only outside of string literals and [column] references. Normalizes
    operators to lowercase.

    Args:
        input_string: The string to process.

    Returns:
        A string with spaces added around logical operators outside of
        protected spans.
    """
    def add_spaces(part: str) -> str:
        # Add spaces around 'and' and 'or' operators using word boundaries (case-insensitive)
        # and normalize to lowercase
        part = re.sub(r'\b(and|or)\b', lambda m: f' {m.group(1).lower()} ', part, flags=re.IGNORECASE)
        return replace_double_spaces(part)

    return replace_outside_protected_spans(input_string, add_spaces)


def mark_special_tokens(input_string: str) -> str:
    """
    Mark special tokens (if, else, endif, elseif, then) with $ markers
    and format them for further processing.

    Args:
        input_string: The string to process.

    Returns:
        A string with special tokens marked and formatted.
    """
    # Add $ markers around special tokens
    result = add_additions_outside_of_quotes(input_string, '$', 'if', 'else', 'endif', 'elseif', 'then')

    # Replace marked tokens with their formatted versions
    result = replace_values_outside_of_quotes(result, replacements=[
        ('$if$', '$if$('),
        ('$else$', ')$else$('),
        ('$endif$', ')$endif$'),
        ('$elseif$', ')$elseif$('),
        ('$then$', ')$then$(')
    ])

    return result


def standardize_equality_operators(input_string: str) -> str:
    """
    Standardize equality operators by replacing == with = outside of string literals.

    Args:
        input_string: The string to process.

    Returns:
        A string with standardized equality operators.
    """
    return replace_value_outside_of_quotes(input_string, '==', '=')


def preserve_logical_operators_with_markers(input_string: str) -> str:
    """
    Replace logical operators with special markers to preserve them during
    whitespace removal. Handles both uppercase and lowercase operators.

    Args:
        input_string: The string to process.

    Returns:
        A string with logical operators replaced by markers.
    """
    # Use case-insensitive matching and normalize to lowercase markers
    return replace_outside_protected_spans(
        input_string,
        lambda part: re.sub(r'\s+(and|or)\s+', lambda m: f' __{m.group(1).lower()}__ ', part, flags=re.IGNORECASE),
    )


def restore_logical_operators(input_string: str) -> str:
    """
    Restore logical operators from special markers.

    Args:
        input_string: The string with logical operator markers.

    Returns:
        A string with logical operators restored.
    """
    result = input_string.replace('__and__', ' and ')
    result = result.replace('__or__', ' or ')
    return result


def add_additions_outside_of_quotes(func_string: str, addition: str, *args) -> str:
    """
    Add additions outside quoted substrings and [column] references.

    Args:
        func_string: The string to process.
        addition: The addition to add outside of protected spans.
        *args: Additional arguments specifying the values to add the addition to.

    Returns:
        The processed string with additions added outside of protected spans.
    """
    return replace_outside_protected_spans(func_string, lambda part: replace_values(part, addition, *args))


def replace_value_outside_of_quotes(func_string: str, val: str, replace: str) -> str:
    """
    Replace a value with another value outside quoted substrings and [column]
    references in the input string.

    Args:
        func_string: The string to process.
        val: The value to replace.
        replace: The value to replace with.

    Returns:
        The processed string with the value replaced outside of protected spans.
    """
    return replace_outside_protected_spans(' ' + func_string + ' ', lambda part: part.replace(val, replace))


def replace_values_outside_of_quotes(func_string: str, replacements: List[Tuple[str, str]]) -> str:
    """
    Replace multiple values with corresponding replacements outside quoted
    substrings and [column] references in the input string.

    Args:
        func_string: The string to process.
        replacements: A list of tuples where each tuple contains a value to replace and its replacement.

    Returns:
        The processed string with values replaced outside of protected spans.
    """
    def replace_all(part: str) -> str:
        for old_val, new_val in replacements:
            part = part.replace(old_val, new_val)
        return part

    return replace_outside_protected_spans(func_string, replace_all)

def replace_values(part_string: str, addition: str, *args) -> str:
    """
    Add an addition around specified values in the input substring.

    Args:
        part_string: The substring to process.
        addition: The addition to add around specified values.
        *args: Values to add the addition to.

    Returns:
        The processed substring with additions added around specified values.
    """
    for arg in args:
        part_string = re.sub(rf'\b{arg}\b', f'{addition}{arg}{addition}', part_string)
    return part_string


def parse_pl_cols(func_string: str) -> str:
    """
    Parse Polars column expressions in the input string and replace them with appropriate Polars expressions.

    This function identifies column references in square brackets (e.g., [column_name]) and
    converts them to Polars column expressions (e.g., pl.col("column_name")).

    Args:
        func_string: The string containing Polars column expressions.

    Returns:
        The processed string with Polars column expressions replaced.
    """
    func_op = []
    func_string = deepcopy(func_string)
    cur_string = func_string
    pos = 0
    inside_quotes = False
    quote_char = ''
    length = len(cur_string)

    while pos < length:
        char = cur_string[pos]
        if char in "\"'":
            if inside_quotes:
                if char == quote_char:
                    inside_quotes = False
                    quote_char = ''
            else:
                inside_quotes = True
                quote_char = char
        elif char == '[' and not inside_quotes:
            start = pos
            end = pos
            while end < length:
                end += 1
                if cur_string[end] in "\"'":
                    if inside_quotes:
                        if cur_string[end] == quote_char:
                            inside_quotes = False
                            quote_char = ''
                    else:
                        inside_quotes = True
                        quote_char = cur_string[end]
                elif cur_string[end] == ']' and not inside_quotes:
                    break

            if end < length and cur_string[end] == ']':
                val = cur_string[start + 1:end]
                if ',' not in val:
                    func_op.append((start + 1, end))
                pos = end
            else:
                break
        pos += 1

    col_rename = set((f'pl.col("{func_string[_s:_e]}")', func_string[_s - 1:_e + 1]) for _s, _e in func_op)
    for new_val, old_val in col_rename:
        func_string = func_string.replace(old_val, new_val)
    return func_string


def remove_unwanted_characters(func_string: str) -> str:
    """
    Remove unwanted characters outside quoted substrings and [column]
    references in the input string, while preserving special markers.

    This function removes whitespace and other unnecessary characters while
    ensuring that special markers like __and__ and __or__ are preserved.

    Args:
        func_string: The string to process.

    Returns:
        The processed string with unwanted characters removed outside of
        protected spans.
    """
    def strip_whitespace(part: str) -> str:
        # Save any special markers before removing whitespace
        special_markers = {}
        marker_count = 0

        # Find all special markers (like __and__, __or__)
        for marker in ["__and__", "__or__"]:
            while marker in part:
                unique_id = f"__MARKER_{marker_count}__"
                part = part.replace(marker, unique_id, 1)
                special_markers[unique_id] = marker
                marker_count += 1

        # Remove all whitespace
        part = "".join(part.split())

        # Restore the special markers
        for unique_id, marker in special_markers.items():
            part = part.replace(unique_id, marker)

        return part

    return replace_outside_protected_spans(func_string, strip_whitespace)


def preprocess(input_function: str) -> str:
    """
    Preprocess an input function string by applying a series of transformations
    to standardize its format for further processing.

    This function performs the following steps:
    1. Validates parentheses and if/then/else/endif structure on the raw input
    2. Removes comments (text starting with // to the end of line)
    3. Normalizes whitespace (replaces newlines with spaces, removes double spaces)
    4. Adds spaces around logical operators (and, or)
    5. Marks and formats special tokens (if, else, endif, elseif, then)
    6. Standardizes equality operators (== becomes =)
    7. Converts column references ([column]) to Polars expressions
    8. Preserves logical operators during whitespace removal
    9. Removes unwanted whitespace and characters
    10. Restores logical operators with proper spacing

    Args:
        input_function: The function string to preprocess.

    Returns:
        The preprocessed function string ready for tokenization and parsing.

    Raises:
        ExpressionSyntaxError: If parentheses are unbalanced or conditional
            keywords (if/then/else/elseif/endif) are misplaced or missing.
    """
    validate_expression_syntax(input_function)

    input_function = remove_comments(input_function)

    input_function = normalize_whitespace(input_function)

    input_function = add_spaces_around_logical_operators(input_function)

    input_function = mark_special_tokens(input_function)

    input_function = standardize_equality_operators(input_function)

    input_function = parse_pl_cols(input_function)

    input_function = preserve_logical_operators_with_markers(input_function)

    input_function = remove_unwanted_characters(input_function)

    input_function = restore_logical_operators(input_function)

    return input_function