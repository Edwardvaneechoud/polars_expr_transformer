import polars as pl
from typing import Any
import os


PlStringType = pl.Expr | str
PlIntType = pl.Expr | int


def is_polars_expr(v: Any) -> bool:
    return isinstance(v, pl.Expr)


def create_fix_col(val: Any) -> pl.Expr:
    return pl.lit(val)


def create_fix_date_col(s: Any) -> pl.Expr:
    return pl.col(s).str.to_datetime()
