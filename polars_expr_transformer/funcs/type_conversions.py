import polars as pl
from typing import Any, Optional
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col, PlStringType



def to_string(value: PlStringType) -> pl.Expr:
    """
    Converts a column or value to text.

    For example, to_string([age]) would return "30" when [age] is 30.

    Parameters:
    - value: The column or value to convert to text

    Returns:
    - The text representation
    """
    if isinstance(value, pl.Expr):
        return value.cast(str)
    return pl.lit(value.__str__())


def to_date(text: PlStringType, date_format: str = "%Y-%m-%d") -> pl.Expr:
    """
    Parses a text column or value into a date.

    For example, to_date("2023-05-15") would return the date 2023-05-15.

    Parameters:
    - text: The text column or value to parse as a date
    - date_format: How to interpret the date text (default is "%Y-%m-%d")
      Common format codes:
      - %Y: Four-digit year (e.g., 2023)
      - %m: Two-digit month (01-12)
      - %d: Two-digit day (01-31)
      - %b: Month abbreviation (Jan, Feb)
      - %B: Full month name (January, February)

    Returns:
    - The date value
    """
    text = text if is_polars_expr(text) else create_fix_col(text)
    return text.str.to_date(date_format, strict=False)


def to_datetime(s: PlStringType, date_format: str = "%Y-%m-%d %H:%M:%S") -> pl.Expr:
    """
    Parses a text column or value into a datetime.

    For example, to_datetime("2023-05-15 14:30:00") would return the datetime 2023-05-15 14:30:00.

    Parameters:
    - s: The text column or value to parse as a datetime
    - date_format: How to interpret the datetime text (default is "%Y-%m-%d %H:%M:%S")

    Returns:
    - The datetime value
    """
    s = s if is_polars_expr(s) else create_fix_col(s)
    return s.str.to_datetime(date_format, strict=False)


def to_integer(value: Any) -> pl.Expr:
    """
    Converts a column or value to a whole number, truncating any decimal places.

    For example, to_integer([price]) would return 19 when [price] is 19.99.

    Parameters:
    - value: The column or value to convert to an integer

    Returns:
    - The integer value (decimal places are truncated)
    """
    if is_polars_expr(value):
        return value.cast(pl.Int64)
    return pl.lit(int(value))


def to_float(value: Any) -> pl.Expr:
    """
    Converts a column or value to a number with decimal places.

    For example, to_float([quantity]) would return 3.0 when [quantity] is 3.

    Parameters:
    - value: The column or value to convert to a floating-point number

    Returns:
    - The floating-point number
    """
    if is_polars_expr(value):
        return value.cast(pl.Float64)
    return pl.lit(float(value))


def to_number(value: Any) -> pl.Expr:
    """
    Converts a column or value to a number (same as to_float).

    For example, to_number([quantity]) would return 3.0 when [quantity] is 3.

    Parameters:
    - value: The column or value to convert to a number

    Returns:
    - The numeric value
    """
    return to_float(value)


def to_boolean(value: Any) -> pl.Expr:
    """
    Converts a column or value to true or false. Non-zero numbers and text like "true", "yes",
    "t" or "y" become true; zero and text like "false", "no", "f" or "n" become false.

    For example, to_boolean([quantity]) would return true when [quantity] is 1.

    Parameters:
    - value: The column or value to convert to a boolean

    Returns:
    - The boolean value (true or false)
    """
    if is_polars_expr(value):

        str_value = value.cast(pl.Utf8).str.to_lowercase()

        is_numeric_pattern = r"^-?\d+(\.\d+)?$"

        is_zero_pattern = r"^-?0(\.0*)?$"

        return (
            # Check for true-like strings
            pl.when(str_value.is_in(["true", "yes", "1", "t", "y"]))
            .then(pl.lit(True))
            .when(str_value.is_in(["false", "no", "0", "f", "n"]) | str_value.str.contains(is_zero_pattern))
            .then(pl.lit(False))
            .when(str_value.str.contains(is_numeric_pattern))
            .then(pl.lit(True))
            # Default case
            .otherwise(pl.lit(False))
        )

    # Handle literal values
    if isinstance(value, str):
        value_lower = value.lower()
        return pl.lit(value_lower in ["true", "yes", "1", "t", "y"])
    return pl.lit(bool(value))


def to_decimal(value: Any, precision: Optional[int] = None) -> pl.Expr:
    """
    Converts a column or value to a decimal number rounded to a fixed number of decimal places.

    For example, to_decimal([price], 2) would return 19.99 when [price] is 19.987.

    Parameters:
    - value: The column or value to convert to a decimal
    - precision: How many decimal places to keep (default is None, which keeps all decimal places)

    Returns:
    - The decimal number
    """
    expr = to_float(value)
    if precision is not None:
        return expr.round(precision)
    return expr
