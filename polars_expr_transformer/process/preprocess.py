import re
from copy import deepcopy
from typing import List, Tuple


def replace_double_spaces(func_string: str):
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


def add_additions_outside_of_quotes(func_string: str, addition: str, *args):
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


def replace_value_outside_of_quotes(func_string: str, val: str, replace: str):
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


def replace_values_outside_of_quotes(func_string: str, replacements: List[Tuple[str, str]]):
    """
    Replace multiple values with corresponding replacements outside quoted substrings in the input string.

    Args:
        func_string: The string to process.
        replacements: A list of tuples where each tuple contains a value to replace and its replacement.

    Returns:
        The processed string with values replaced outside of quotes.
    """
    for _old, _new in replacements:
        func_string = func_string.replace(_old, _new)
    return func_string


def replace_values(part_string: str, addition: str, *args):
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


def parse_pl_cols(func_string):
    """
    Parse Polars column expressions in the input string and replace them with appropriate Polars expressions.

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
    Remove unwanted characters outside quoted substrings in the input string.

    Args:
        func_string: The string to process.

    Returns:
        The processed string with unwanted characters removed outside of quotes.
    """
    parts = re.split(r"""("[^"]*"|'[^']*')""", func_string)
    parts[::2] = map(lambda s: "".join(s.split()), parts[::2])  # outside quotes
    return "".join(parts)


def preprocess(input_function: str):
    """
    Preprocess the input function string by applying a series of transformations.

    Args:
        input_function: The function string to preprocess.

    Returns:
        The preprocessed function string.
    """
    input_function = input_function.replace('\n', ' ')
    input_function = replace_double_spaces(input_function)
    input_function = add_additions_outside_of_quotes(input_function, '$', 'if', 'else', 'endif', 'elseif', 'then')
    input_function = replace_values_outside_of_quotes(input_function, replacements=[('$if$', '$if$('),
                                                                                    ('$else$', ')$else$('),
                                                                                    ('$endif$', ')$endif$'),
                                                                                    ('$elseif$', ')$elseif$('),
                                                                                    ('$then$', ')$then$(')])
    input_function = replace_value_outside_of_quotes(input_function, '==', '=')
    # input_function = standardize_quotes(input_function)
    input_function = parse_pl_cols(input_function)
    input_function = remove_unwanted_characters(input_function)
    return input_function
