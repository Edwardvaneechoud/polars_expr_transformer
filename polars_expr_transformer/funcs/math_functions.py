import polars as pl
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col, PlNumericType

string_type = pl.Expr | str


def negation(v: PlNumericType) -> pl.Expr:
    """
    Apply negation to a Polars expression representing a numeric value.

    This function takes a numeric expression from the Polars library and
    returns its negated value. It is specifically designed for use with
    Polars expressions that contain numeric data types.

    Args:
        v (PlNumericType): A Polars expression of a numeric data type.

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
    if is_polars_expr(v):
        return pl.Expr.neg(v)
    else:
        return pl.lit(v).neg()


def log(v: PlNumericType) -> pl.Expr:
    """
    Apply the natural logarithm to a Polars expression representing a numeric value.

    This function takes a numeric expression from the Polars library and
    returns the natural logarithm of its value. It is specifically designed
    for use with Polars expressions that contain numeric data types.

    Args:
        v (PlNumericType): A Polars expression of a numeric data type.

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


def exp(v: PlNumericType) -> pl.Expr:
    """
    Apply the exponential function to a Polars expression representing a numeric value.
    This function takes a numeric expression from the Polars library and returns the exponential value of its value.
    It is specifically designed for use with Polars expressions that contain numeric data types.
    Args:
        v (PlNumericType): A Polars expression of a numeric data type.
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
    if is_polars_expr(v):
        return pl.Expr.exp(v)
    else:
        return pl.lit(v).exp()

def sqrt(v: PlNumericType) -> pl.Expr:
    """
    Apply the square root function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.sqrt(v)
    else:
        return pl.lit(v).sqrt()


def abs(v: PlNumericType) -> pl.Expr:
    """
    Apply the absolute function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.abs(v)
    else:
        return pl.lit(v).abs()


def sin(v: PlNumericType) -> pl.Expr:
    """
    Apply the sine function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.sin(v)
    else:
        return pl.lit(v).sin()


def cos(v: PlNumericType) -> pl.Expr:
    """
    Apply the cosine function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.cos(v)
    else:
        return pl.lit(v).cos()


def tan(v: PlNumericType) -> pl.Expr:
    """
    Apply the tangent function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.tan(v)
    else:
        return pl.lit(v).tan()


def asin(v: PlNumericType) -> pl.Expr:
    ...


def ceil(v: PlNumericType) -> pl.Expr:
    """
    Apply the ceiling function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.ceil(v)
    else:
        return pl.lit(v).ceil()


def round(v: PlNumericType, decimals: int = None) -> pl.Expr:
    """
    Apply the rounding function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.round(v, decimals)
    else:
        return pl.lit(v).round(decimals)


def floor(v: PlNumericType) -> pl.Expr:
    """
    Apply the floor function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.floor(v)
    else:
        return pl.lit(v).floor()


def tanh(v: PlNumericType) -> pl.Expr:
    """
    Apply the hyperbolic tangent function to a Polars expression representing a numeric value.
    """
    if is_polars_expr(v):
        return pl.Expr.tanh(v)
    else:
        return pl.lit(v).tanh()


def negative() -> int:
    return -1


def random_int(from_value: int = 0, to_value: int = 2):
    """
    Generate a random integer between two values.
    Args:
        from_value: Number to start the range from.
        to_value: Number to end the range at.

    Returns:
        A random integer between the two values.
    """
    return pl.int_range(from_value, to_value).sample(n=pl.len(), with_replacement=True)
