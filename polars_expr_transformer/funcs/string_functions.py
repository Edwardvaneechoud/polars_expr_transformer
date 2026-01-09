import polars as pl
import polars_ds as pds
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col
from polars_expr_transformer.funcs.utils import PlStringType, PlIntType
from functools import partial


def concat(*text_parts) -> pl.Expr:
    """
    Combines multiple text values into a single text.

    For example, concat("Hello", " ", "World") would return "Hello World".

    Parameters:
    - text_parts: The texts you want to combine

    Returns:
    - The combined text
    """
    columns = [create_fix_col(c) if not is_polars_expr(c) else c for c in text_parts]
    return pl.concat_str(columns)


def count_match(text: PlStringType, pattern: str) -> pl.Expr:
    """
    Counts how many times a pattern appears in text.

    For example, count_match("banana", "a") would return 3.

    Parameters:
    - text: The text to search in
    - pattern: The pattern to search for

    Returns:
    - The number of matches found
    """
    if isinstance(text, pl.Expr):
        return text.str.count_matches(pattern)
    return pl.lit(text).str.count_matches(pattern)


def length(text: PlStringType) -> pl.Expr:
    """
    Counts the number of characters in text.

    For example, length("hello") would return 5.

    Parameters:
    - text: The text to measure

    Returns:
    - The number of characters
    """
    if isinstance(text, str):
        return pl.lit(len(text))
    text: pl.Expr
    return text.str.len_chars()


def uppercase(text: PlStringType) -> pl.Expr:
    """
    Converts text to ALL UPPERCASE.

    For example, uppercase("Hello World") would return "HELLO WORLD".

    Parameters:
    - text: The text to convert

    Returns:
    - The uppercase text
    """
    if isinstance(text, pl.Expr):
        return text.str.to_uppercase()
    return pl.lit(text.__str__().upper())


def titlecase(text: PlStringType) -> pl.Expr:
    """
    Converts Text To Title Case, Where Each Word Is Capitalized.

    For example, titlecase("hello world") would return "Hello World".

    Parameters:
    - text: The text to convert

    Returns:
    - The title case text
    """
    if isinstance(text, pl.Expr):
        return text.str.to_titlecase()
    return pl.lit(text.__str__().title())


def lowercase(text: PlStringType) -> pl.Expr:
    """
    Converts text to all lowercase.

    For example, lowercase("Hello World") would return "hello world".

    Parameters:
    - text: The text to convert

    Returns:
    - The lowercase text
    """
    if isinstance(text, pl.Expr):
        return text.str.to_lowercase()
    return pl.lit(text.__str__().lower())


def left(text: PlStringType, num_chars: pl.Expr | int) -> pl.Expr:
    """
    Gets a specified number of characters from the beginning of text.

    For example, left("Hello World", 5) would return "Hello".

    Parameters:
    - text: The text to extract from
    - num_chars: How many characters to take from the beginning

    Returns:
    - The extracted text
    """
    if is_polars_expr(text):
        if is_polars_expr(num_chars):
            return text.str.slice(0, num_chars)
        else:
            return text.str.slice(0, num_chars)
    elif is_polars_expr(num_chars):
        return pl.lit(text).str.slice(0, num_chars)
    else:
        return pl.lit(text[:num_chars])


def right(text: PlStringType, num_chars: PlIntType) -> pl.Expr:
    """
    Gets a specified number of characters from the end of text.

    For example, right("Hello World", 5) would return "World".

    Parameters:
    - text: The text to extract from
    - num_chars: How many characters to take from the end

    Returns:
    - The extracted text
    """
    if is_polars_expr(text):
        if is_polars_expr(num_chars):
            return text.str.slice(pl.Expr.mul(pl.lit(-1), num_chars))
        else:
            return text.str.slice(pl.lit(-num_chars))
    elif is_polars_expr(num_chars):
        return pl.lit(text).str.slice(pl.Expr.mul(pl.lit(-1), num_chars))
    else:
        return pl.lit(text[-num_chars:])


def __apply_replace(row, replace_by=None):
    v = list(row.values())
    main_str = v[0]
    other_str = v[1] if len(v) > 1 else None
    replace_str = v[2] if len(v) > 2 else replace_by
    return main_str.replace(other_str, replace_str)


def replace(text: PlStringType, find_text: PlStringType, replace_with: PlStringType) -> pl.Expr:
    """
    Replaces specific text with different text.

    For example, replace("Hello world", "world", "friend") would return "Hello friend".

    Parameters:
    - text: The original text where replacements will be made
    - find_text: The text you want to find and replace
    - replace_with: The new text that will replace the found text

    Returns:
    - The text after replacement
    """
    if not is_polars_expr(text):
        text = pl.lit(text)
    return text.str.replace_many(find_text, replace_with).cast(pl.Utf8)


def find_position(text: PlStringType, sub: PlStringType) -> pl.Expr:
    """
    Find the position of a substring within a string.

    Parameters:
    - text: The text in which to find the position of the substring. Can be an expression or any other value.
    - sub (Any): The substring to find the position of. Can be an expression or any other value.

    Returns:
    - the position of the text

    Note: If `s` or `sub` is not a pl expression, it will be converted into one.
    """
    text = text if is_polars_expr(text) else create_fix_col(text)
    sub = sub if is_polars_expr(sub) else create_fix_col(sub)
    return text.str.find(sub, literal=True, strict=False)


def pad_left(text: PlStringType, length: int, pad_character: str = " ") -> pl.Expr:
    """
    Adds characters to the beginning of text to reach a specific length.

    For example, pad_left("123", 5, "0") would return "00123".

    Parameters:
    - text: The text you want to pad
    - length: How long you want the final text to be
    - pad_character: What character to use for padding (default is a space)

    Returns:
    - The padded text
    """
    s = text if is_polars_expr(text) else create_fix_col(text)
    return s.str.pad_start(length, pad_character)


def pad_right(text: PlStringType, length: int, pad_character: str = " ") -> pl.Expr:
    """
    Adds characters to the end of text to reach a specific length.

    For example, pad_right("123", 5, "0") would return "12300".

    Parameters:
    - text: The text you want to pad
    - length: How long you want the final text to be
    - pad_character: What character to use for padding (default is a space)

    Returns:
    - The padded text
    """
    s = text if is_polars_expr(text) else create_fix_col(text)
    return s.str.pad_end(length, pad_character)


def trim(text: PlStringType) -> pl.Expr:
    """
    Removes spaces from both the beginning and end of text.

    For example, trim("  hello world  ") would return "hello world".

    Parameters:
    - text: The text you want to trim

    Returns:
    - The trimmed text
    """
    s = text if is_polars_expr(text) else create_fix_col(text)
    return s.str.strip_chars_end().str.strip_chars_start()


def left_trim(text: PlStringType) -> pl.Expr:
    """
    Removes spaces from the beginning of text.

    For example, left_trim("  hello world  ") would return "hello world  ".

    Parameters:
    - text: The text you want to trim

    Returns:
    - The trimmed text
    """
    s = text if is_polars_expr(text) else create_fix_col(text)
    return s.str.strip_chars_start()


def right_trim(text: PlStringType) -> pl.Expr:
    """
    Removes spaces from the end of text.

    For example, right_trim("hello world  ") would return "hello world".

    Parameters:
    - text: The text you want to trim

    Returns:
    - The trimmed text
    """
    s = text if is_polars_expr(text) else create_fix_col(text)
    return s.str.strip_chars_end()


def __get_similarity_method(how: str) -> callable:
    match how:
        case 'levenshtein':
            return partial(pds.str_leven, return_sim=True)
        case 'jaro':
            return pds.str_jaro
        case 'jaro_winkler':
            return pds.str_jw
        case 'damerau_levenshtein':
            return partial(pds.str_d_leven, return_sim=True)
        case 'hamming':
            return pds.str_hamming
        case 'fuzzy':
            return pds.str_fuzz
        case 'optimal_string_alignment':
            return pds.str_osa
        case 'sorensen_dice':
            return pds.str_sorensen_dice
        case 'jaccard':
            return pds.str_jaccard
        case _:
            raise ValueError(f"Unknown similarity method: {how}\n\n"
                             f"Possible options are: "
                             f"levenshtein, jaro, jaro_winkler, damerau_levenshtein, "
                             f"hamming, fuzzy, optimal_string_alignment, "
                             f"sorensen_dice, jaccard")


def string_similarity(text1: PlStringType, text2: PlStringType, method: str = 'levenshtein') -> pl.Expr:
    """
    Measures how similar two texts are to each other, on a scale of 0 to 1.

    A value of 1 means the texts are identical, while 0 means they are completely different.

    For example, string_similarity("apple", "appl") might return 0.8.

    Parameters:
    - text1: The first text to compare
    - text2: The second text to compare
    - method: Which comparison method to use (default is 'levenshtein')
      Available methods include:
      - 'levenshtein': Good general-purpose similarity measure
      - 'jaro': Good for short strings like names
      - 'jaro_winkler': Similar to jaro but gives higher scores to strings that match at the beginning
      - 'fuzzy': Finds approximate matches, good for typos

    Returns:
    - A similarity score between 0 and 1
    """
    similarity_method = partial(__get_similarity_method(method), parallel=True)
    v1 = text1 if is_polars_expr(text1) else pl.lit(text1)
    v2 = text2 if is_polars_expr(text2) else pl.lit(text2)
    return similarity_method(v1, v2)


def mid(text: PlStringType, start: PlIntType, num_chars: PlIntType) -> pl.Expr:
    """
    Extracts a portion of text from the middle starting at a specified position.

    For example, mid("Hello World", 0, 5) would return "Hello".

    Parameters:
    - text: The text to extract from
    - start: The starting position (0-based index)
    - num_chars: How many characters to extract

    Returns:
    - The extracted text
    """
    t = text if is_polars_expr(text) else pl.lit(text)
    s = start if is_polars_expr(start) else start
    n = num_chars if is_polars_expr(num_chars) else num_chars
    return t.str.slice(s, n)


def substring(text: PlStringType, start: PlIntType, num_chars: PlIntType) -> pl.Expr:
    """
    Extracts a portion of text starting at a specified position (alias for mid).

    For example, substring("Hello World", 6, 5) would return "World".

    Parameters:
    - text: The text to extract from
    - start: The starting position (0-based index)
    - num_chars: How many characters to extract

    Returns:
    - The extracted text
    """
    return mid(text, start, num_chars)


def starts_with(text: PlStringType, prefix: PlStringType) -> pl.Expr:
    """
    Checks if text starts with a specific prefix.

    For example, starts_with("Hello World", "Hello") would return True.

    Parameters:
    - text: The text to check
    - prefix: The prefix to look for at the start

    Returns:
    - True if the text starts with the prefix, False otherwise
    """
    t = text if is_polars_expr(text) else pl.lit(text)
    p = prefix if is_polars_expr(prefix) else prefix
    return t.str.starts_with(p)


def ends_with(text: PlStringType, suffix: PlStringType) -> pl.Expr:
    """
    Checks if text ends with a specific suffix.

    For example, ends_with("Hello World", "World") would return True.

    Parameters:
    - text: The text to check
    - suffix: The suffix to look for at the end

    Returns:
    - True if the text ends with the suffix, False otherwise
    """
    t = text if is_polars_expr(text) else pl.lit(text)
    s = suffix if is_polars_expr(suffix) else suffix
    return t.str.ends_with(s)


def reverse(text: PlStringType) -> pl.Expr:
    """
    Reverses the characters in text.

    For example, reverse("Hello") would return "olleH".

    Parameters:
    - text: The text to reverse

    Returns:
    - The reversed text
    """
    t = text if is_polars_expr(text) else pl.lit(text)
    return t.str.reverse()


def repeat(text: PlStringType, count: PlIntType) -> pl.Expr:
    """
    Repeats text a specified number of times.

    For example, repeat("ab", 3) would return "ababab".

    Parameters:
    - text: The text to repeat
    - count: How many times to repeat the text

    Returns:
    - The repeated text
    """
    t = text if is_polars_expr(text) else pl.lit(text)
    # Polars doesn't have a direct repeat method, so we use a workaround
    if is_polars_expr(count):
        # For dynamic count, we need to use concat_str with a list
        return pl.concat_str([t] * 100).str.slice(0, t.str.len_chars() * count)
    else:
        return pl.concat_str([t] * count)


def split(text: PlStringType, delimiter: str) -> pl.Expr:
    """
    Splits text into a list using a delimiter.

    For example, split("a,b,c", ",") would return ["a", "b", "c"].

    Parameters:
    - text: The text to split
    - delimiter: The character(s) to split on

    Returns:
    - A list of text parts
    """
    t = text if is_polars_expr(text) else pl.lit(text)
    return t.str.split(delimiter)
