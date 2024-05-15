from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col
from typing import Any, Dict
import polars as pl
from polars_expr_transformer.funcs.utils import PlStringType, PlIntType


def equals(s: Any, t: Any) -> pl.Expr:
    """
    Check if two expressions or values are equal.

    Parameters:
    - s (Any): The first expression or value to compare. Can be a pl expression or any other value.
    - t (Any): The second expression or value to compare. Can be a pl expression or any other value.

    Returns:
    - pl.Expr: A pl expression representing the equality of `s` and `t`.

    Note: If `s` or `t` is not a pl expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_col(s)
    t = t if is_polars_expr(t) else create_fix_col(t)
    return s.eq(t)


def is_empty(s: pl.Expr) -> pl.Expr:
    """
    Check if a given expression is empty.

    Parameters:
    - s (pl.Expr): The expression to check. Must be a pl expression.

    Returns:
    - pl.Expr: A pl expression representing whether `s` is empty.

    Note: If `s` is not a pl expression, a ValueError will be raised.
    """
    return s.is_null()


def is_not_empty(s: pl.Expr) -> pl.Expr:
    """
    Check if a given expression is empty.

    Parameters:
    - s (pl.Expr): The expression to check. Must be a pl expression.

    Returns:
    - pl.Expr: A pl expression representing whether `s` is empty.

    Note: If `s` is not a pl expression, a ValueError will be raised.
    """
    return s.is_not_null()


def does_not_equal(s: Any, t: Any):
    s = s if is_polars_expr(s) else create_fix_col(s)
    t = t if is_polars_expr(t) else create_fix_col(t)
    return pl.Expr.eq(s, t).is_not()


def _not(s: Any) -> pl.Expr:
    """
    Negate a given expression.

    Parameters:
    - s (pl.Expr): The expression to negate. Must be a pl expression.

    Returns:
    - pl.Expr: A pl expression representing the negation of `s`.

    Note: If `s` is not a pl expression, a ValueError will be raised.
    """
    if not is_polars_expr(s):
        s = pl.lit(s)
    return pl.Expr.not_(s)


def is_string(val: Any) -> pl.Expr:
    """
    Check if a given expression or value is a string.

    Parameters:
    - val (Any): The expression or value to check. Can be a pl expression or any other value.

    Returns:
    - pl.Expr: A pl expression representing whether `val` is a string.

    Note: If `val` is a pl expression, its dtype will be checked. Otherwise, Python's isinstance will be used.
    """
    if is_polars_expr(val):
        dtype = pl.select(val).dtypes[0]
        return pl.lit(dtype.is_(pl.Utf8))
    return pl.lit(isinstance(val, str))


def contains(base: PlStringType, pattern: Any) -> pl.Expr:
    """
    Check if a pattern is contained within a base string or expression.

    Parameters:
    - base (Union[Expr, str]): The base string or expression where the pattern is to be searched.
    - pattern (Union[Expr, str]): The pattern string or expression to search for within the base.

    Returns:
    - Expr: An expression indicating whether the pattern is contained within the base.

    Note: 
    - If both base and pattern are expressions, a custom function is used for performance optimization.
    - If one of them is a string and the other is an expression, built-in string matching is used.
    - If both are strings, Python's native 'in' operator is used and the result is wrapped as an expression.
    """
    def _contain(_row: Dict):
        base_val, pattern_val = _row.values()
        return pattern_val in base_val

    if isinstance(base, pl.Expr):
        if isinstance(pattern, pl.Expr):
            return pl.struct([base, pattern]).map_elements(lambda r: _contain(r), return_dtype=pl.Boolean)
        else:
            return base.str.contains(pattern)
    else:
        if isinstance(pattern, pl.Expr):
            return pl.struct([pattern]).map_elements(lambda x: next(iter(x.values())) in base)
        else:
            return pl.lit(pattern in base)


def _in(pattern: Any, base: PlStringType) -> pl.Expr:
    return contains(base, pattern)