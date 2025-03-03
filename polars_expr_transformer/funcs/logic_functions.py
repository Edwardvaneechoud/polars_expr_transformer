# logic functions

import polars as pl

from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col
from typing import Any
from polars_expr_transformer.funcs.utils import PlStringType


def equals(value1: Any, value2: Any) -> pl.Expr:
    """
    Checks if two values are equal to each other.

    For example, equals("apple", "apple") would return True.

    Parameters:
    - value1: The first value to compare
    - value2: The second value to compare

    Returns:
    - True if the values are equal, False otherwise
    """
    s = value1 if is_polars_expr(value1) else create_fix_col(value1)
    t = value2 if is_polars_expr(value2) else create_fix_col(value2)
    return s.eq(t)


def is_empty(value: pl.Expr) -> pl.Expr:
    """
    Checks if a value is empty (null/missing).

    For example, is_empty(null) would return True.

    Parameters:
    - value: The value to check

    Returns:
    - True if the value is empty, False otherwise
    """
    return value.is_null()


def is_not_empty(value: pl.Expr) -> pl.Expr:
    """
    Checks if a value contains something (is not null/missing).

    For example, is_not_empty("apple") would return True.

    Parameters:
    - value: The value to check

    Returns:
    - True if the value contains something, False if it's empty
    """
    return value.is_not_null()


def does_not_equal(value1: Any, value2: Any):
    """
    Checks if two values are different from each other.

    For example, does_not_equal("apple", "orange") would return True.

    Parameters:
    - value1: The first value to compare
    - value2: The second value to compare

    Returns:
    - True if the values are different, False if they're the same
    """
    s = value1 if is_polars_expr(value1) else create_fix_col(value1)
    t = value2 if is_polars_expr(value2) else create_fix_col(value2)
    return pl.Expr.eq(s, t).not_()


def _not(value: Any) -> pl.Expr:
    """
    Reverses a True/False value.

    For example, _not(True) would return False.

    Parameters:
    - value: The True/False value to reverse

    Returns:
    - The opposite value (True becomes False, False becomes True)
    """
    if not is_polars_expr(value):
        value = pl.lit(value)
    return pl.Expr.not_(value)


def is_string(value: Any) -> pl.Expr:
    """
    Checks if a value is text (a string).

    For example, is_string("apple") would return True, but is_string(123) would return False.

    Parameters:
    - value: The value to check

    Returns:
    - True if the value is text, False otherwise
    """
    if is_polars_expr(value):
        dtype = pl.select(value).dtypes[0]
        return pl.lit(dtype.is_(pl.Utf8))
    return pl.lit(isinstance(value, str))


def contains(text: PlStringType, search_for: Any) -> pl.Expr:
    """
    Checks if some text contains a specific pattern.

    For example, contains("hello world", "world") would return True.

    Parameters:
    - text: The text to search in
    - search_for: The pattern to look for

    Returns:
    - True if the pattern is found in the text, False otherwise
    """
    if isinstance(text, pl.Expr):
        return text.str.contains(search_for)
    else:
        if isinstance(search_for, pl.Expr):
            return pl.lit(text).str.contains(search_for)
        else:
            return pl.lit(search_for in text)


def _in(value: Any, collection: PlStringType) -> pl.Expr:
    """
    Checks if a value exists within a larger text.

    For example, _in("world", "hello world") would return True.

    Parameters:
    - value: The value to search for
    - collection: The text to search in

    Returns:
    - True if the value is found in the collection, False otherwise
    """
    return contains(collection, value)