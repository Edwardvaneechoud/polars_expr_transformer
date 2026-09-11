import polars as pl
from typing import Any
import os
from polars.datatypes.group import NUMERIC_DTYPES


PlStringType = pl.Expr | str
PlIntType = pl.Expr | int
PlNumericType = NUMERIC_DTYPES


def is_polars_expr(v: Any) -> bool:
    return isinstance(v, pl.Expr)


def create_fix_col(val: Any) -> pl.Expr:
    return pl.lit(val)


def as_expr(value: Any) -> pl.Expr:
    """Return the value unchanged if it is already an expression, else a literal."""
    return value if is_polars_expr(value) else pl.lit(value)


def create_fix_date_col(s: Any) -> pl.Expr:
    return pl.lit(s).str.to_datetime()
