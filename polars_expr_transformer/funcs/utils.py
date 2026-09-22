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


STRING_CAST_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
"""How a Datetime is rendered when a string function reads it as text.

``cast(pl.String)`` would render ``2026-09-24 10:00:00.000000``, so a formula
written against what the value looks like -- ``ends_with([ordered_at],
"10:00:00")`` -- would never match. Seconds resolution is what the value reads
as, so that is what the string functions see. ``format_date`` remains the way
to ask for any other layout, fractional seconds included.
"""


def _series_to_string(s: pl.Series) -> pl.Series:
    """Render one batch as text, whatever dtype it turns out to hold."""
    if s.dtype == pl.String:
        return s
    if s.dtype == pl.Datetime:
        return s.dt.to_string(STRING_CAST_DATETIME_FORMAT)
    return s.cast(pl.String)


def as_string_expr(value: Any) -> pl.Expr:
    """Return the value as a text expression, casting it first if it is not text.

    Every method in the ``.str`` namespace refuses a non-String column outright,
    so ``contains([order_date], "2026-09")`` used to raise ``expected String
    type, got: date`` before the function body could do anything useful. Going
    through here first makes such a formula mean what it reads like.

    The dispatch has to happen on a real Series rather than while the expression
    is being built, because an expression is built against no frame:
    ``simple_function_to_expr`` cannot know whether ``[order_date]`` holds a
    Date, an Int64 or text already. A String input is handed back untouched, so
    formulas that already worked keep their behaviour exactly.
    """
    if isinstance(value, str):
        return pl.lit(value)
    return as_expr(value).map_batches(_series_to_string, return_dtype=pl.String)
