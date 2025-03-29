import re
from copy import deepcopy
from typing import List, Tuple


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
    lines = input_string.split('\n')
    lines_without_comments = []

    for line in lines:
        # Find position of // but only if it's not inside a string
        pos = 0
        inside_single_quote = False
        inside_double_quote = False
        comment_pos = -1

        while pos < len(line):
            char = line[pos]

            if char == "'" and not inside_double_quote:
                inside_single_quote = not inside_single_quote
            elif char == '"' and not inside_single_quote:
                inside_double_quote = not inside_double_quote
            elif char == '/' and pos + 1 < len(line) and line[
                pos + 1] == '/' and not inside_single_quote and not inside_double_quote:
                comment_pos = pos
                break

            pos += 1

        # Remove comment part if found
        if comment_pos != -1:
            line = line[:comment_pos]

        lines_without_comments.append(line)

    return '\n'.join(lines_without_comments)


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


def add_spaces_around_logical_operators(input_string: str) -> str:
    """
    Add spaces around logical operators (and, or) in the input string,
    but only outside of string literals.

    Args:
        input_string: The string to process.

    Returns:
        A string with spaces added around logical operators outside of quotes.
    """
    parts = re.split(r'("[^"]*"|\'[^\']*\')', input_string)

    # Only process parts outside quotes (even indices)
    for i in range(0, len(parts), 2):
        # Add spaces around 'and' and 'or' operators using word boundaries
        parts[i] = re.sub(r'\b(and|or)\b', r' \1 ', parts[i])
        parts[i] = replace_double_spaces(parts[i])

    return ''.join(parts)


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
    whitespace removal.

    Args:
        input_string: The string to process.

    Returns:
        A string with logical operators replaced by markers.
    """
    parts = re.split(r'("[^"]*"|\'[^\']*\')', input_string)

    # Only process parts outside quotes (even indices)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'\s+(and|or)\s+', r' __\1__ ', parts[i])

    return ''.join(parts)


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
    Add additions outside quoted substrings in the input string.

    Args:
        func_string: The string to process.
        addition: The addition to add outside of quotes.
        *args: Additional arguments specifying the values to add the addition to.

    Returns:
        The processed string with additions added outside of quotes.
    """
    parts = re.split(r'("[^"]*"|\'[^\']*\')', func_string)
    parts[::2] = [replace_values(v, addition, *args) for v in parts[::2]]
    return ''.join(parts)


def replace_value_outside_of_quotes(func_string: str, val: str, replace: str) -> str:
    """
    Replace a value with another value outside quoted substrings in the input string.

    Args:
        func_string: The string to process.
        val: The value to replace.
        replace: The value to replace with.

    Returns:
        The processed string with the value replaced outside of quotes.
    """
    parts = re.split(r"""("[^"]*"|'[^']*')""", ' ' + func_string + ' ')
    parts[::2] = [v.replace(val, replace) for v in parts[::2]]
    return ''.join(parts)


def replace_values_outside_of_quotes(func_string: str, replacements: List[Tuple[str, str]]) -> str:
    """
    Replace multiple values with corresponding replacements outside quoted substrings in the input string.

    Args:
        func_string: The string to process.
        replacements: A list of tuples where each tuple contains a value to replace and its replacement.

    Returns:
        The processed string with values replaced outside of quotes.
    """
    # Split the string by quoted parts
    parts = re.split(r'("[^"]*"|\'[^\']*\')', func_string)

    # Process only the parts outside quotes (even indices)
    for i in range(0, len(parts), 2):
        for old_val, new_val in replacements:
            parts[i] = parts[i].replace(old_val, new_val)

    # Join the parts back together
    return ''.join(parts)

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
    Remove unwanted characters outside quoted substrings in the input string,
    while preserving special markers.

    This function removes whitespace and other unnecessary characters while
    ensuring that special markers like __and__ and __or__ are preserved.

    Args:
        func_string: The string to process.

    Returns:
        The processed string with unwanted characters removed outside of quotes.
    """
    parts = re.split(r"""("[^"]*"|'[^']*')""", func_string)

    # Process parts outside quotes (even indices)
    for i in range(0, len(parts), 2):
        # Save any special markers before removing whitespace
        special_markers = {}
        marker_count = 0

        # Find all special markers (like __and__, __or__)
        for marker in ["__and__", "__or__"]:
            while marker in parts[i]:
                unique_id = f"__MARKER_{marker_count}__"
                parts[i] = parts[i].replace(marker, unique_id, 1)
                special_markers[unique_id] = marker
                marker_count += 1

        # Remove all whitespace
        parts[i] = "".join(parts[i].split())

        # Restore the special markers
        for unique_id, marker in special_markers.items():
            parts[i] = parts[i].replace(unique_id, marker)

    return "".join(parts)


def preprocess(input_function: str) -> str:
    """
    Preprocess an input function string by applying a series of transformations
    to standardize its format for further processing.

    This function performs the following steps:
    1. Removes comments (text starting with // to the end of line)
    2. Normalizes whitespace (replaces newlines with spaces, removes double spaces)
    3. Adds spaces around logical operators (and, or)
    4. Marks and formats special tokens (if, else, endif, elseif, then)
    5. Standardizes equality operators (== becomes =)
    6. Converts column references ([column]) to Polars expressions
    7. Preserves logical operators during whitespace removal
    8. Removes unwanted whitespace and characters
    9. Restores logical operators with proper spacing

    Args:
        input_function: The function string to preprocess.

    Returns:
        The preprocessed function string ready for tokenization and parsing.
    """
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
