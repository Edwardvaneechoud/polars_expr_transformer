import polars as pl
from typing import Any
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col, create_fix_date_col
from datetime import datetime
from polars_expr_transformer.funcs.utils import PlStringType, PlIntType


def now() -> pl.Expr:
    """
    Get the current timestamp.

    Returns:
    - pl.Expr: A FlowFile expression representing the current timestamp.
    """
    return pl.lit(datetime.now())


def today() -> pl.Expr:
    """
    Get the current date.

    Returns:
    - pl.Expr: A FlowFile expression representing the current date.
    """
    return pl.lit(datetime.today())


def year(date_value: Any) -> pl.Expr:
    """
    Extract the year from a date or timestamp.

    Args:
        date_value: The date or timestamp to extract the year from. Can be a FlowFile expression or any other value.

    Returns:
        pl.Expr: A Polars expression representing the year extracted from the input.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.year()


def month(date_value: Any) -> pl.Expr:
    """
    Extract the month from a date or timestamp.

    Args:
        date_value: The date or timestamp to extract the month from. Can be a FlowFile expression or any other value.

    Returns:
        pl.Expr: A Polars expression representing the month extracted from the input.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_col(date_value).str.to_datetime()
    return date_value.dt.month()


def day(date_value: PlStringType) -> pl.Expr:
    """
    Extract the day from a date or timestamp.

    Args:
        date_value: The date or timestamp to extract the day from. Can be a FlowFile expression or any other value.

    Returns:
        pl.Expr: A Polars expression representing the day extracted from the input.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.day()


def hour(date_value: PlStringType) -> pl.Expr:
    """
    Extract the hour from a timestamp.

    Args:
        date_value: The timestamp or Polars expression to extract the hour from.

    Returns:
        pl.Expr: A Polars expression representing the extracted hour.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.hour()


def minute(date_value: PlStringType) -> pl.Expr:
    """
    Extract the minute from a timestamp.

    Args:
        date_value: The timestamp or Polars expression to extract the minute from.

    Returns:
        pl.Expr: A Polars expression representing the extracted minute.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.minute()


def second(date_value: PlStringType) -> pl.Expr:
    """
    Extract the second from a timestamp.

    Args:
        date_value: The timestamp or Polars expression to extract the second from.

    Returns:
        pl.Expr: A Polars expression representing the extracted second.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.second()


def add_days(date_value: PlStringType, days: PlIntType) -> pl.Expr:
    """
    Add a specified number of days to a date or timestamp.

    Args:
        date_value: The date or Polars expression to add days to.
        days (PlIntType): The number of days to add.

    Returns:
        pl.Expr: A Polars expression representing the date after adding the specified number of days.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    days = days if is_polars_expr(days) else create_fix_col(days)
    return date_value + pl.duration(days=days)


def add_years(date_value: PlStringType, years: PlIntType) -> pl.Expr:
    """
    Add a specified number of years to a date or timestamp.

    Args:
        date_value: The date or Polars expression to add years to.
        years (PlIntType): The number of years to add.

    Returns:
        pl.Expr: A Polars expression representing the date after adding the specified number of years.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    years = years if is_polars_expr(years) else create_fix_col(years)
    return date_value + pl.duration(days=years * 365)


def add_hours(date_value: PlStringType, hours: PlIntType) -> pl.Expr:
    """
    Add a specified number of hours to a timestamp.

    Args:
        date_value: The timestamp or Polars expression to add hours to.
        hours (PlIntType): The number of hours to add.

    Returns:
        pl.Expr: A Polars expression representing the timestamp after adding the specified number of hours.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    hours = hours if is_polars_expr(hours) else create_fix_col(hours)
    return date_value + pl.duration(hours=hours)


def add_minutes(date_value: PlStringType, minutes: PlIntType) -> pl.Expr:
    """
    Add a specified number of minutes to a timestamp.

    Args:
        date_value: The timestamp or Polars expression to add minutes to.
        minutes (PlIntType): The number of minutes to add.

    Returns:
        pl.Expr: A Polars expression representing the timestamp after adding the specified number of minutes.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    minutes = minutes if is_polars_expr(minutes) else create_fix_col(minutes)
    return date_value + pl.duration(minutes=minutes)


def add_seconds(date_value: PlStringType, seconds: PlIntType) -> pl.Expr:
    """
    Add a specified number of seconds to a timestamp.

    Args:
        date_value: The timestamp or Polars expression to add seconds to.
        seconds (PlIntType): The number of seconds to add.

    Returns:
        pl.Expr: A Polars expression representing the timestamp after adding the specified number of seconds.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    seconds = seconds if is_polars_expr(seconds) else create_fix_col(seconds)
    return date_value + pl.duration(seconds=seconds)


def datetime_diff_seconds(date_value1: PlStringType, date_value2: PlStringType) -> pl.Expr:
    """
    Calculate the difference in seconds between two timestamps.

    Args:
        date_value1: The first timestamp or Polars expression.
        date_value2: The second timestamp or Polars expression.

    Returns:
        pl.Expr: A Polars expression representing the difference in seconds between the two timestamps.

    Note: If the inputs are not Polars expressions, they will be converted into ones.
    """
    date_value1 = date_value1 if is_polars_expr(date_value1) else create_fix_date_col(date_value1)
    date_value2 = date_value2 if is_polars_expr(date_value2) else create_fix_date_col(date_value2)
    return (date_value1 - date_value2).dt.total_seconds()


def datetime_diff_nanoseconds(date_value1: PlStringType, date_value2: PlStringType) -> pl.Expr:
    """
    Calculate the difference in nanoseconds between two timestamps.

    Args:
        date_value1: The first timestamp or Polars expression.
        date_value2: The second timestamp or Polars expression.

    Returns:
        pl.Expr: A Polars expression representing the difference in nanoseconds between the two timestamps.

    Note: If the inputs are not Polars expressions, they will be converted into ones.
    """
    date_value1 = date_value1 if is_polars_expr(date_value1) else create_fix_date_col(date_value1)
    date_value2 = date_value2 if is_polars_expr(date_value2) else create_fix_date_col(date_value2)
    return (date_value1 - date_value2).dt.total_nanoseconds()


def date_diff_days(date_value1: PlStringType, date_value2: PlStringType) -> pl.Expr:
    """
    Calculate the difference in days between two dates.

    Args:
        date_value1: The first date or Polars expression.
        date_value2: The second date or Polars expression.

    Returns:
        pl.Expr: A Polars expression representing the difference in days between the two dates.

    Note: If the inputs are not Polars expressions, they will be converted into ones.
    """
    date_value1 = date_value1 if is_polars_expr(date_value1) else create_fix_date_col(date_value1)
    date_value2 = date_value2 if is_polars_expr(date_value2) else create_fix_date_col(date_value2)
    return (date_value1 - date_value2).dt.total_days()


def date_trim(date_value: Any, part: str) -> pl.Expr:
    """
    Trim a date to a specified part. For example, trimming 2023-01-12 12:34:56.123 to 'day' will return 2023-01-12 00:00:00.000.

    Args:
        date_value: The date or timestamp to trim. Can be a FlowFile expression or any other value.
        part: The part of the date to trim to. Can be 'year', 'month', 'day', 'hour', 'minute', or 'second'.

    Returns:
        pl.Expr: A Polars expression representing the date trimmed to the specified part.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if isinstance(date_value, pl.Expr) else pl.col(date_value)

    if part == 'year':
        return date_value.dt.truncate('1y')
    elif part == 'month':
        return date_value.dt.truncate('1mo')
    elif part == 'day':
        return date_value.dt.truncate('1d')
    elif part == 'hour':
        return date_value.dt.truncate('1h')
    elif part == 'minute':
        return date_value.dt.truncate('1min')
    elif part == 'second':
        return date_value.dt.truncate('1s')
    else:
        raise ValueError(
            f"Invalid part '{part}' specified. Must be 'year', 'month', 'day', 'hour', 'minute', or 'second'.")


def date_truncate(date_value: Any, truncate_by: str) -> pl.Expr:
    """
    Truncate a date to a specified part. For example, truncating 2023-01-12 12:34:56.123 by '1day' will return 2023-01-12 00:00:00.000.

    Args:
        date_value: The date or timestamp to truncate. Can be a FlowFile expression or any other value.
        truncate_by: The part of the date to truncate by. Can be 'Nyear', 'Nmonth', 'Nday', 'Nhour', 'Nminute', or 'Nsecond'.

    Returns:
        pl.Expr: A Polars expression representing the truncated date.

    Note: If `date_value` is not a Polars expression, it will be converted into one.
    """
    date_value = date_value if isinstance(date_value, pl.Expr) else pl.col(date_value)
    return date_value.dt.truncate(truncate_by)
