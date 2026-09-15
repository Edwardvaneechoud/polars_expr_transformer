# logic functions

import polars as pl

from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col, as_expr
from typing import Any
from polars_expr_transformer.funcs.utils import PlStringType


def equals(value1: Any, value2: Any) -> pl.Expr:
    """
    Checks if two values are equal to each other.

    For example, equals([status], "shipped") would return true when [status] is "shipped".

    Parameters:
    - value1: The column or value to compare
    - value2: The column or value to compare it against

    Returns:
    - true if both values are equal, otherwise false
    """
    s = value1 if is_polars_expr(value1) else create_fix_col(value1)
    t = value2 if is_polars_expr(value2) else create_fix_col(value2)
    return s.eq(t)


def is_empty(value: pl.Expr) -> pl.Expr:
    """
    Checks if a value is empty (missing).

    For example, is_empty([email]) would return true when [email] is empty.

    Parameters:
    - value: The column to check

    Returns:
    - true if the value is empty, otherwise false
    """
    return value.is_null()


def is_not_empty(value: pl.Expr) -> pl.Expr:
    """
    Checks if a value contains something (is not missing).

    For example, is_not_empty([discount]) would return false when [discount] is empty.

    Parameters:
    - value: The column to check

    Returns:
    - true if the value contains something, false if it is empty
    """
    return value.is_not_null()


def does_not_equal(value1: Any, value2: Any):
    """
    Checks if two values are different from each other.

    For example, does_not_equal([status], "cancelled") would return true when [status] is "shipped".

    Parameters:
    - value1: The column or value to compare
    - value2: The column or value to compare it against

    Returns:
    - true if the values are different, false if they are the same
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
    Checks if a literal value is text (a string). This works on fixed values,
    not on column references.

    For example, is_string("hello") would return true.

    Parameters:
    - value: The value to check

    Returns:
    - true if the value is text, otherwise false
    """
    if is_polars_expr(value):
        dtype = pl.select(value).dtypes[0]
        return pl.lit(dtype.is_(pl.Utf8))
    return pl.lit(isinstance(value, str))


def contains(text: PlStringType, search_for: Any) -> pl.Expr:
    """
    Checks if some text contains a specific pattern.

    For example, contains([product], "Laptop") would return true when [product] is "Laptop Pro 15".

    Parameters:
    - text: The column or value to search in
    - search_for: The pattern to look for

    Returns:
    - true if the pattern is found in the text, otherwise false
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


def _not_in(value: Any, collection: PlStringType) -> pl.Expr:
    """
    Checks if a value does not exist within a larger text.

    For example, _not_in("moon", "hello world") would return True.

    Parameters:
    - value: The value to search for
    - collection: The text to search in

    Returns:
    - True if the value is not found in the collection, False otherwise
    """
    return contains(collection, value).not_()


def _list(*values) -> list:
    """
    Collects the members written between the parentheses of an `in ( ... )` list.

    This is the collection operand of the membership operators; it is never called
    directly from a formula. Members stay as raw Python values, or as expressions
    when a member references a column, which is what lets `_is_in` pick its lowering.

    For example, _list("a", "b") would return ["a", "b"].

    Parameters:
    - values: The members of the list

    Returns:
    - The members as a list
    """
    return list(values)


def _is_in(value: Any, collection: list) -> pl.Expr:
    """
    Checks if a value is one of the members of a list.

    The `collection: list` annotation is load-bearing: it keeps `_standardize_args`
    from wrapping the list itself in `pl.lit()`. Polars rejects a plain list that
    holds expressions, so a collection containing one is imploded with `concat_list`
    instead. An empty collection cannot take that route, as `concat_list` requires
    at least one expression.

    For example, _is_in([status], ["shipped", "packed"]) would return true when
    [status] is "shipped".

    Parameters:
    - value: The column or value to look for
    - collection: The members to look for it among

    Returns:
    - true if the value is one of the members, otherwise false
    """
    if not collection:
        return as_expr(value).is_in([])
    if any(is_polars_expr(member) for member in collection):
        return as_expr(value).is_in(pl.concat_list([as_expr(m) for m in collection]))
    return as_expr(value).is_in(list(collection))


def _is_not_in(value: Any, collection: list) -> pl.Expr:
    """
    Checks if a value is none of the members of a list.

    For example, _is_not_in([status], ["failed", "cancelled"]) would return true
    when [status] is "shipped".

    Parameters:
    - value: The column or value to look for
    - collection: The members to look for it among

    Returns:
    - true if the value is none of the members, otherwise false
    """
    return _is_in(value, collection).not_()


def coalesce(*values) -> pl.Expr:
    """
    Returns the first non-empty value from a list of values.

    For example, coalesce([discount], 0) would return 0 when [discount] is empty.

    Parameters:
    - values: The columns or values to check in order

    Returns:
    - The first non-empty value, or null if all values are empty
    """
    if len(values) == 0:
        raise ValueError("coalesce requires at least one argument")

    exprs = [v if is_polars_expr(v) else pl.lit(v) for v in values]
    return pl.coalesce(exprs)


def ifnull(value: Any, default: Any) -> pl.Expr:
    """
    Returns a default value if the input is empty.

    For example, ifnull([discount], 0) would return 0 when [discount] is empty.

    Parameters:
    - value: The column or value to check
    - default: The column or value to return if the first value is empty

    Returns:
    - The original value if not empty, otherwise the default value
    """
    value_expr = value if is_polars_expr(value) else pl.lit(value)
    default_expr = default if is_polars_expr(default) else pl.lit(default)
    return pl.coalesce([value_expr, default_expr])


def nvl(value: Any, default: Any) -> pl.Expr:
    """
    Returns a default value if the input is empty (alias for ifnull).

    For example, nvl([email], "no email") would return "no email" when [email] is empty.

    Parameters:
    - value: The column or value to check
    - default: The column or value to return if the first value is empty

    Returns:
    - The original value if not empty, otherwise the default value
    """
    return ifnull(value, default)


def nullif(value1: Any, value2: Any) -> pl.Expr:
    """
    Returns null if the two values are equal, otherwise returns the first value.

    For example, nullif([status], "cancelled") would return null when [status] is "cancelled".

    Parameters:
    - value1: The column or value to return
    - value2: The value to compare against

    Returns:
    - null if both are equal, otherwise the first value
    """
    v1 = value1 if is_polars_expr(value1) else pl.lit(value1)
    v2 = value2 if is_polars_expr(value2) else pl.lit(value2)
    return pl.when(v1.eq(v2)).then(pl.lit(None)).otherwise(v1)


def between(value: Any, min_val: Any, max_val: Any) -> pl.Expr:
    """
    Checks if a value is between a minimum and maximum value (inclusive).

    For example, between([age], 30, 40) would return true when [age] is 35.

    Parameters:
    - value: The column or value to check
    - min_val: The column or value used as the minimum (inclusive)
    - max_val: The column or value used as the maximum (inclusive)

    Returns:
    - true if the value is between min and max (inclusive), otherwise false
    """
    v = value if is_polars_expr(value) else pl.lit(value)
    min_v = min_val if is_polars_expr(min_val) else pl.lit(min_val)
    max_v = max_val if is_polars_expr(max_val) else pl.lit(max_val)
    return v.ge(min_v).and_(v.le(max_v))


def greatest(*values) -> pl.Expr:
    """
    Returns the largest value from a list of values.

    For example, greatest([price], 100) would return 100 when [price] is 79.99.

    Parameters:
    - values: The columns or values to compare

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

    For example, least([price], 100) would return 100 when [price] is 249.99.

    Parameters:
    - values: The columns or values to compare

    Returns:
    - The smallest value
    """
    if len(values) == 0:
        raise ValueError("least requires at least one argument")

    exprs = [v if is_polars_expr(v) else pl.lit(v) for v in values]
    return pl.min_horizontal(exprs)
