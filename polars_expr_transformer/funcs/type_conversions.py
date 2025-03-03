import polars as pl
from typing import Any, Optional
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col, PlStringType



def to_string(value: PlStringType) -> pl.Expr:
    """
    Converts any value to text.

    For example, to_string(123) would return "123".

    Parameters:
    - value: The value to convert to text

    Returns:
    - The text representation
    """
    if isinstance(value, pl.Expr):
        return value.cast(str)
    return pl.lit(value.__str__())



def to_date(text: PlStringType, date_format: str = "%Y-%m-%d") -> pl.Expr:
    """
    Converts text to a date value.

    For example, to_date("2023-01-15") would return a date value for January 15, 2023.

    Parameters:
    - text: The text to convert to a date
    - date_format: Instructions for how to interpret the date text (default is year-month-day)
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
    Convert a string to a datetime.

    Parameters:
    - s (Any): The string to convert to a datetime. Can be a pl expression or any other value.
    - format (str): The format of the datetime string. Default is "%Y-%m-%d %H:%M:%S".

    Returns:
    - pl.Expr: A pl expression representing the converted datetime.

    Note: If `s` is not a pl expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_col(s)
    return s.str.to_datetime(date_format, strict=False)


def to_integer(value: Any) -> pl.Expr:
    """
    Converts a value to a whole number.

    For example, to_integer("123") would return 123, and to_integer(45.67) would return 45.

    Parameters:
    - value: The value to convert to an integer

    Returns:
    - The integer value (decimal places are truncated)
    """
    if is_polars_expr(value):
        return value.cast(pl.Int64)
    return pl.lit(int(value))


def to_float(value: Any) -> pl.Expr:
    """
    Converts a value to a number with decimal places.

    For example, to_float("123.45") would return 123.45.

    Parameters:
    - value: The value to convert to a floating-point number

    Returns:
    - The floating-point number
    """
    if is_polars_expr(value):
        return value.cast(pl.Float64)
    return pl.lit(float(value))


def to_number(value: Any) -> pl.Expr:
    """
    Converts a value to a number (same as to_float).

    For example, to_number("123.45") would return 123.45.

    Parameters:
    - value: The value to convert to a number

    Returns:
    - The numeric value
    """
    return to_float(value)


def to_boolean(value: Any) -> pl.Expr:
    """
    Converts a value to True or False.

    For example:
    - to_boolean(1) returns True
    - to_boolean(0) returns False
    - to_boolean("t") returns True
    - to_boolean("f") returns False
    - to_boolean("true") returns True
    - to_boolean("false") returns False
    - to_boolean("yes") returns True
    - to_boolean("no") returns False

    Parameters:
    - value: The value to convert to a boolean

    Returns:
    - A Polars expression that will evaluate to True or False
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
    Converts a value to a decimal number with fixed precision.

    For example, to_decimal("123.456789", 2) would return 123.46.

    Parameters:
    - value: The value to convert to a decimal
    - precision: How many decimal places to keep (if None, keeps all decimal places)

    Returns:
    - The decimal number
    """
    expr = to_float(value)
    if precision is not None:
        return expr.round(precision)
    return expr
