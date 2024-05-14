import polars as pl


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


def negative() -> int:
    return -1

