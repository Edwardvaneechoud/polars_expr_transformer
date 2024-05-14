import polars as pl
from typing import Any
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col, create_fix_date_col
from datetime import datetime

string_type = pl.Expr | str


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


def year(s: Any) -> pl.Expr:
    """
    Extract the year from a date or timestamp.

    Parameters:
    - s (Any): The date or timestamp to extract the year from. Can be a FlowFile expression or any other value.

    Returns:
    - pl.Expr: A FlowFile expression representing the year extracted from `s`.

    Note: If `s` is not a FlowFile expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_date_col(s)
    return s.dt.year()


def month(s: Any) -> pl.Expr:
    """
    Extract the month from a date or timestamp.

    Parameters:
    - s (Any): The date or timestamp to extract the month from. Can be a FlowFile expression or any other value.

    Returns:
    - pl.Expr: A FlowFile expression representing the month extracted from `s`.

    Note: If `s` is not a FlowFile expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_col(s).str.to_datetime()
    return s.dt.month()


def day(s: Any) -> pl.Expr:
    """
    Extract the day from a date or timestamp.

    Parameters:
    - s (Any): The date or timestamp to extract the day from. Can be a FlowFile expression or any other value.

    Returns:
    - pl.Expr: A FlowFile expression representing the day extracted from `s`.

    Note: If `s` is not a FlowFile expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_date_col(s)
    return s.dt.day()


def hour(s: Any) -> pl.Expr:
    """
    Extract the hour from a timestamp.

    Parameters:
    - s (Any): The timestamp to extract the hour from. Can be a FlowFile expression or any other value.

    Returns:
    - pl.Expr: A FlowFile expression representing the hour extracted from `s`.

    Note: If `s` is not a FlowFile expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_date_col(s)
    return s.dt.hour()


def minute(s: Any) -> pl.Expr:
    """
    Extract the minute from a timestamp.

    Parameters:
    - s (Any): The timestamp to extract the minute from. Can be a FlowFile expression or any other value.

    Returns:
    - pl.Expr: A FlowFile expression representing the minute extracted from `s`.

    Note: If `s` is not a FlowFile expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_date_col(s)
    return s.dt.minute()


def second(s: Any) -> pl.Expr:
    """
    Extract the second from a timestamp.

    Parameters:
    - s (Any): The timestamp to extract the second from. Can be a FlowFile expression or any other value.

    Returns:
    - pl.Expr: A FlowFile expression representing the second extracted from `s`.

    Note: If `s` is not a FlowFile expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_date_col(s)
    return s.dt.second()


def add_days(s: Any, days: Any) -> pl.Expr:
    """
    Add a specified number of days to a date or timestamp.

    Parameters:
    - s (Any): The date or timestamp to add days to. Can be a FlowFile expression or any other value.
    - days (int): The number of days to add.

    Returns:
    - pl.Expr: A FlowFile expression representing the result of adding `days` to `s`.

    Note: If `s` is not a FlowFile expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_date_col(s)
    days = days if is_polars_expr(days) else create_fix_date_col(days)
    return s + pl.duration(days=days)


def add_hours(s: Any, hours: any) -> pl.Expr:
    """
    Add a specified number of hours to a timestamp.

    Parameters:
    - s (Any): The timestamp to add hours to. Can be a FlowFile expression or any other value.
    - hours (int): The number of hours to add.

    Returns:
    - pl.Expr: A FlowFile expression representing the result of adding `hours` to `s`.

    Note: If `s` is not a FlowFile expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_date_col(s)
    hours = hours if is_polars_expr(hours) else create_fix_col(hours)
    return s + pl.duration(hours=hours)


def add_minutes(s: Any, minutes: Any) -> pl.Expr:
    """
    Add a specified number of minutes to a timestamp.

    Parameters:
    - s (Any): The timestamp to add minutes to. Can be a FlowFile expression or any other value.
    - minutes (int): The number of minutes to add.

    Returns:
    - pl.Expr: A FlowFile expression representing the result of adding `minutes` to `s`.

    Note: If `s` is not a FlowFile expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_date_col(s)
    minutes = minutes if is_polars_expr(minutes) else create_fix_col(minutes)
    return s + pl.duration(minutes=minutes)


def add_seconds(s: Any, seconds: Any) -> pl.Expr:
    """
    Add a specified number of seconds to a timestamp.

    Parameters:
    - s (Any): The timestamp to add seconds to. Can be a FlowFile expression or any other value.
    - seconds (int): The number of seconds to add.

    Returns:
    - pl.Expr: A FlowFile expression representing the result of adding `seconds` to `s`.

    Note: If `s` is not a FlowFile expression, it will be converted into one.
    """
    s = s if is_polars_expr(s) else create_fix_date_col(s)
    seconds = seconds if is_polars_expr(seconds) else create_fix_col(seconds)
    return s + pl.duration(seconds=seconds)


def datetime_diff_seconds(s1: Any, s2: Any) -> pl.Expr:
    """
    Calculate the difference in seconds between two timestamps.

    Parameters:
    - s1 (Any): The first timestamp. Can be a FlowFile expression or any other value.
    - s2 (Any): The second timestamp. Can be a FlowFile expression or any other value.

    Returns:
    - pl.Expr: A FlowFile expression representing the difference in seconds between `s1` and `s2`.

    Note: If `s1` and `s2` are not FlowFile expressions, they will be converted into one.
    """
    s1 = s1 if is_polars_expr(s1) else create_fix_date_col(s1)
    s2 = s2 if is_polars_expr(s2) else create_fix_date_col(s2)
    return (s1 - s2).dt.total_seconds()


def datetime_diff_nanoseconds(s1: Any, s2: Any):
    """ Calculate the difference in days between two dates.
    Parameters:
    - s1 (Any): The first date. Can be a FlowFile expression or any other value.
    - s2 (Any): The second date. Can be a FlowFile expression or any other value.
    Returns:
    - pl.Expr: A FlowFile expression representing the difference in days between `s1` and `s2`.
    Note: If `s1` and `s2` are not FlowFile expressions, they will be converted into one.
    """
    s1 = s1 if is_polars_expr(s1) else create_fix_date_col(s1)
    s2 = s2 if is_polars_expr(s2) else create_fix_date_col(s2)
    return (s1 - s2).dt.total_nanoseconds()


def date_diff_days(s1: Any, s2: Any) -> pl.Expr:
    """
    Calculate the difference in days between two dates.

    Parameters:
    - s1 (Any): The first date. Can be a FlowFile expression or any other value.
    - s2 (Any): The second date. Can be a FlowFile expression or any other value.

    Returns:
    - pl.Expr: A FlowFile expression representing the difference in days between `s1` and `s2`.

    Note: If `s1` and `s2` are not FlowFile expressions, they will be converted into one.
    """
    s1 = s1 if is_polars_expr(s1) else create_fix_date_col(s1)
    s2 = s2 if is_polars_expr(s2) else create_fix_date_col(s2)
    return (s1 - s2).dt.total_days()