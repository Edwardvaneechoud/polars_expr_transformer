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


def coalesce(*values) -> pl.Expr:
    """
    Returns the first non-null value from a list of values.

    For example, coalesce(null, null, "default") would return "default".

    Parameters:
    - values: Multiple values to check in order

    Returns:
    - The first non-null value, or null if all values are null
    """
    if len(values) == 0:
        raise ValueError("coalesce requires at least one argument")

    exprs = [v if is_polars_expr(v) else pl.lit(v) for v in values]
    return pl.coalesce(exprs)


def ifnull(value: Any, default: Any) -> pl.Expr:
    """
    Returns a default value if the input is null.

    For example, ifnull(null, "default") would return "default".

    Parameters:
    - value: The value to check
    - default: The value to return if the first value is null

    Returns:
    - The original value if not null, otherwise the default value
    """
    value_expr = value if is_polars_expr(value) else pl.lit(value)
    default_expr = default if is_polars_expr(default) else pl.lit(default)
    return pl.coalesce([value_expr, default_expr])


def nvl(value: Any, default: Any) -> pl.Expr:
    """
    Returns a default value if the input is null (alias for ifnull).

    For example, nvl(null, "default") would return "default".

    Parameters:
    - value: The value to check
    - default: The value to return if the first value is null

    Returns:
    - The original value if not null, otherwise the default value
    """
    return ifnull(value, default)


def nullif(value1: Any, value2: Any) -> pl.Expr:
    """
    Returns null if the two values are equal, otherwise returns the first value.

    For example, nullif(5, 5) would return null, but nullif(5, 3) would return 5.

    Parameters:
    - value1: The value to potentially return
    - value2: The value to compare against

    Returns:
    - null if value1 equals value2, otherwise value1
    """
    v1 = value1 if is_polars_expr(value1) else pl.lit(value1)
    v2 = value2 if is_polars_expr(value2) else pl.lit(value2)
    return pl.when(v1.eq(v2)).then(pl.lit(None)).otherwise(v1)


def between(value: Any, min_val: Any, max_val: Any) -> pl.Expr:
    """
    Checks if a value is between a minimum and maximum value (inclusive).

    For example, between(5, 1, 10) would return True.

    Parameters:
    - value: The value to check
    - min_val: The minimum value (inclusive)
    - max_val: The maximum value (inclusive)

    Returns:
    - True if the value is between min and max (inclusive), False otherwise
    """
    v = value if is_polars_expr(value) else pl.lit(value)
    min_v = min_val if is_polars_expr(min_val) else pl.lit(min_val)
    max_v = max_val if is_polars_expr(max_val) else pl.lit(max_val)
    return v.ge(min_v).and_(v.le(max_v))


def greatest(*values) -> pl.Expr:
    """
    Returns the largest value from a list of values.

    For example, greatest(1, 5, 3) would return 5.

    Parameters:
    - values: Multiple values to compare

    Returns:
    - The largest value
    """
    if len(values) == 0:
        raise ValueError("greatest requires at least one argument")

    exprs = [v if is_polars_expr(v) else pl.lit(v) for v in values]
    return pl.max_horizontal(exprs)


def least(*values) -> pl.Expr:
    """
    Returns the smallest value from a list of values.

    For example, least(1, 5, 3) would return 1.

    Parameters:
    - values: Multiple values to compare

    Returns:
    - The smallest value
    """
    if len(values) == 0:
        raise ValueError("least requires at least one argument")

    exprs = [v if is_polars_expr(v) else pl.lit(v) for v in values]
    return pl.min_horizontal(exprs)
