import polars as pl
from typing import Annotated, Any
import os
from polars.datatypes.group import NUMERIC_DTYPES


PlStringType = pl.Expr | str
PlIntType = pl.Expr | int
PlNumericType = NUMERIC_DTYPES

PlDateType = Annotated[pl.Expr | str, "date-text"]
"""Marks a parameter that expects a date, so text reaching it is parsed first.

Accepts the same values as ``PlStringType`` — the metadata only makes the alias
distinguishable from it, so ``process/schema_coercion.py`` can find the date
parameters of a function from its signature instead of from a hand-kept table
that would drift as date functions are added.
"""

DATE_TEXT_FORMATS = (
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d",
)
"""Formats tried, in order, when date text is parsed automatically.

Only unambiguous ISO-8601 layouts are listed: a layout like ``01/10/2005`` is
left out on purpose, because guessing between day-first and month-first would
silently produce a wrong date. Text matching none of these becomes null, and
``to_date``/``to_datetime`` with an explicit format remain the way to read it.
"""


def is_polars_expr(v: Any) -> bool:
    return isinstance(v, pl.Expr)


def create_fix_col(val: Any) -> pl.Expr:
    return pl.lit(val)


def as_expr(value: Any) -> pl.Expr:
    """Return the value unchanged if it is already an expression, else a literal."""
    return value if is_polars_expr(value) else pl.lit(value)


def create_fix_date_col(s: Any) -> pl.Expr:
    return pl.lit(s).str.to_datetime()
