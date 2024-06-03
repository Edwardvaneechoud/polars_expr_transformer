import polars as pl
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col

string_type = pl.Expr | str


def negation(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply negation to a Polars expression representing a numeric value.

    This function takes a numeric expression from the Polars library and
    returns its negated value. It is specifically designed for use with
    Polars expressions that contain numeric data types.

    Args:
        v (pl.NUMERIC_DTYPES): A Polars expression of a numeric data type.

    Returns:
        pl.Expr: A Polars expression representing the negated value of the
                 input expression.

    Example:
        >>> df = pl.DataFrame({'numbers': [1, -2, 3]})
        >>> df.select(negation(pl.col('numbers')))
        shape: (3, 1)
        ┌─────────┐
        │ numbers │
        │ ---     │
        │ i64     │
        ╞═════════╡
        │ -1      │
        ├─────────┤
        │ 2       │
        ├─────────┤
        │ -3      │
        └─────────┘
    """
    return pl.Expr.__neg__(v)


def log(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply the natural logarithm to a Polars expression representing a numeric value.

    This function takes a numeric expression from the Polars library and
    returns the natural logarithm of its value. It is specifically designed
    for use with Polars expressions that contain numeric data types.

    Args:
        v (pl.NUMERIC_DTYPES): A Polars expression of a numeric data type.

    Returns:
        pl.Expr: A Polars expression representing the natural logarithm of the
                 input expression.

    Example:
        >>> df = pl.DataFrame({'numbers': [1, 2, 3]})
        >>> df.select(log(pl.col('numbers')))
        shape: (3, 1)
        ┌─────────┐
        │ numbers │
        │ ---     │
        │ f64     │
        ╞═════════╡
        │ 0       │
        ├─────────┤
        │ 0.693   │
        ├─────────┤
        │ 1.099   │
        └─────────┘
    """
    return pl.Expr.log(v)


def exp(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply the exponential function to a Polars expression representing a numeric value.
    This function takes a numeric expression from the Polars library and returns the exponential value of its value.
    It is specifically designed for use with Polars expressions that contain numeric data types.
    Args:
        v (pl.NUMERIC_DTYPES): A Polars expression of a numeric data type.
    Returns:
        pl.Expr: A Polars expression representing the exponential value of the input expression.
    Example:
        >>> df = pl.DataFrame({'numbers': [1, 2, 3]})
        >>> df.select(exp(pl.col('numbers')))
        shape: (3, 1)
        ┌─────────┐
        │ numbers │
        │ ---     │
        │ f64     │
        ╞═════════╡
        │ 2.718   │
        ├─────────┤
        │ 7.389   │
        ├─────────┤
        │ 20.086  │
        └─────────┘
    """

    return pl.Expr.exp(v)


def sqrt(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply the square root function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.sqrt(v)
    else:
        return pl.lit(v).sqrt()


def abs(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply the absolute function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.abs(v)
    else:
        return pl.lit(v).abs()


def sin(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply the sine function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.sin(v)
    else:
        return pl.lit(v).sin()


def cos(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply the cosine function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.cos(v)
    else:
        return pl.lit(v).cos()


def tan(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply the tangent function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.tan(v)
    else:
        return pl.lit(v).tan()


def asin(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    ...


def ceil(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply the ceiling function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.ceil(v)
    else:
        return pl.lit(v).ceil()


def round(v: pl.NUMERIC_DTYPES, decimals: int = None) -> pl.Expr:
    """
    Apply the rounding function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.round(v, decimals)
    else:
        return pl.lit(v).round(decimals)


def floor(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply the floor function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.floor(v)
    else:
        return pl.lit(v).floor()


def tanh(v: pl.NUMERIC_DTYPES) -> pl.Expr:
    """
    Apply the hyperbolic tangent function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.tanh(v)
    else:
        return pl.lit(v).tanh()


def negative() -> int:
    return -1
