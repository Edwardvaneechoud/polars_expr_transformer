import polars as pl
from typing import Any
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col, create_fix_date_col
from datetime import datetime
from polars_expr_transformer.funcs.utils import PlStringType, PlIntType


def now() -> pl.Expr:
    """
    Gets the current date and time.

    For example, now() might return "2023-05-15 14:30:25".

    Returns:
    - The current date and time
    """
    return pl.lit(datetime.now())


def today() -> pl.Expr:
    """
    Gets the current date.

    For example, today() might return "2023-05-15".

    Returns:
    - The current date
    """
    return pl.lit(datetime.today())


def year(date_value: Any) -> pl.Expr:
    """
    Gets the year from a date.

    For example, year("2023-05-15") would return 2023.

    Parameters:
    - date_value: The date to extract the year from

    Returns:
    - The year as a number
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.year()


def month(date_value: Any) -> pl.Expr:
    """
    Gets the month from a date.

    For example, month("2023-05-15") would return 5.

    Parameters:
    - date_value: The date to extract the month from

    Returns:
    - The month as a number (1-12)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_col(date_value).str.to_datetime()
    return date_value.dt.month()


def day(date_value: PlStringType) -> pl.Expr:
    """
    Gets the day from a date.

    For example, day("2023-05-15") would return 15.

    Parameters:
    - date_value: The date to extract the day from

    Returns:
    - The day of the month as a number (1-31)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.day()


def hour(date_value: PlStringType) -> pl.Expr:
    """
    Gets the hour from a time.

    For example, hour("2023-05-15 14:30:25") would return 14.

    Parameters:
    - date_value: The date and time to extract the hour from

    Returns:
    - The hour as a number (0-23)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.hour()


def minute(date_value: PlStringType) -> pl.Expr:
    """
    Gets the minute from a time.

    For example, minute("2023-05-15 14:30:25") would return 30.

    Parameters:
    - date_value: The date and time to extract the minute from

    Returns:
    - The minute as a number (0-59)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.minute()


def second(date_value: PlStringType) -> pl.Expr:
    """
    Gets the second from a time.

    For example, second("2023-05-15 14:30:25") would return 25.

    Parameters:
    - date_value: The date and time to extract the second from

    Returns:
    - The second as a number (0-59)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.second()


def add_days(date_value: PlStringType, days: PlIntType) -> pl.Expr:
    """
    Adds a number of days to a date.

    For example, add_days("2023-05-15", 5) would return "2023-05-20".

    Parameters:
    - date_value: The starting date
    - days: How many days to add

    Returns:
    - The new date
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    days = days if is_polars_expr(days) else create_fix_col(days)
    return date_value + pl.duration(days=days)


def add_years(date_value: PlStringType, years: PlIntType) -> pl.Expr:
    """
    Adds a number of years to a date.

    For example, add_years("2023-05-15", 1) would return "2024-05-15".

    Parameters:
    - date_value: The starting date
    - years: How many years to add

    Returns:
    - The new date
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    years = years if is_polars_expr(years) else create_fix_col(years)
    return date_value + pl.duration(days=years * 365)


def add_hours(date_value: PlStringType, hours: PlIntType) -> pl.Expr:
    """
    Adds a number of hours to a date and time.

    For example, add_hours("2023-05-15 14:30:00", 3) would return "2023-05-15 17:30:00".

    Parameters:
    - date_value: The starting date and time
    - hours: How many hours to add

    Returns:
    - The new date and time
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    hours = hours if is_polars_expr(hours) else create_fix_col(hours)
    return date_value + pl.duration(hours=hours)


def add_minutes(date_value: PlStringType, minutes: PlIntType) -> pl.Expr:
    """
    Adds a number of minutes to a date and time.

    For example, add_minutes("2023-05-15 14:30:00", 15) would return "2023-05-15 14:45:00".

    Parameters:
    - date_value: The starting date and time
    - minutes: How many minutes to add

    Returns:
    - The new date and time
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    minutes = minutes if is_polars_expr(minutes) else create_fix_col(minutes)
    return date_value + pl.duration(minutes=minutes)


def add_seconds(date_value: PlStringType, seconds: PlIntType) -> pl.Expr:
    """
    Adds a number of seconds to a date and time.

    For example, add_seconds("2023-05-15 14:30:00", 30) would return "2023-05-15 14:30:30".

    Parameters:
    - date_value: The starting date and time
    - seconds: How many seconds to add

    Returns:
    - The new date and time
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    seconds = seconds if is_polars_expr(seconds) else create_fix_col(seconds)
    return date_value + pl.duration(seconds=seconds)


def datetime_diff_seconds(date1: PlStringType, date2: PlStringType) -> pl.Expr:
    """
    Calculates the number of seconds between two dates and times.

    For example, datetime_diff_seconds("2023-05-15 14:30:00", "2023-05-15 14:29:00") would return 60.

    Parameters:
    - date1: The first date and time
    - date2: The second date and time

    Returns:
    - The number of seconds between the two dates and times
    """
    date_value1 = date1 if is_polars_expr(date1) else create_fix_date_col(date1)
    date_value2 = date2 if is_polars_expr(date2) else create_fix_date_col(date2)
    return (date_value1 - date_value2).dt.total_seconds()


def datetime_diff_nanoseconds(date1: PlStringType, date2: PlStringType) -> pl.Expr:
    """
    Calculates the number of nanoseconds between two dates and times.

    For very precise time measurements, this function shows the exact difference down to billionths of a second.

    Parameters:
    - date1: The first date and time
    - date2: The second date and time

    Returns:
    - The number of nanoseconds between the two dates and times
    """
    date_value1 = date1 if is_polars_expr(date1) else create_fix_date_col(date1)
    date_value2 = date2 if is_polars_expr(date2) else create_fix_date_col(date2)
    return (date_value1 - date_value2).dt.total_nanoseconds()


def date_diff_days(date1: PlStringType, date2: PlStringType) -> pl.Expr:
    """
    Calculates the number of days between two dates.

    For example, date_diff_days("2023-05-15", "2023-05-10") would return 5.

    Parameters:
    - date1: The first date
    - date2: The second date

    Returns:
    - The number of days between the two dates
    """
    date_value1 = date1 if is_polars_expr(date1) else create_fix_date_col(date1)
    date_value2 = date2 if is_polars_expr(date2) else create_fix_date_col(date2)
    return (date_value1 - date_value2).dt.total_days()


def date_trim(date_value: Any, part: str) -> pl.Expr:
    """
    Removes the smaller parts of a date or time.

    For example, date_trim("2023-05-15 14:30:25", "day") would return "2023-05-15 00:00:00".

    Parameters:
    - date_value: The date and time to trim
    - part: Which part to keep ('year', 'month', 'day', 'hour', 'minute', or 'second')
      - 'year': Keeps only the year (resets month to January, day to 1, time to midnight)
      - 'month': Keeps year and month (resets day to 1, time to midnight)
      - 'day': Keeps year, month, and day (resets time to midnight)
      - 'hour': Keeps date and hour (resets minutes and seconds to 0)
      - 'minute': Keeps date, hour, and minute (resets seconds to 0)
      - 'second': Keeps date and time (resets milliseconds to 0)

    Returns:
    - The trimmed date and time
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
    Rounds a date down to the nearest specified unit.

    For example, date_truncate("2023-05-15 14:30:25", "1day") would return "2023-05-15 00:00:00".

    Parameters:
    - date_value: The date and time to truncate
    - truncate_by: The time unit to round down to (like "1day", "2hours", "15minutes")
      Some examples:
      - "1year": Round to the start of the year
      - "3months": Round to the nearest 3-month boundary
      - "1week": Round to the start of the week
      - "12hours": Round to the nearest 12-hour boundary

    Returns:
    - The truncated date and time
    """
    date_value = date_value if isinstance(date_value, pl.Expr) else pl.col(date_value)
    return date_value.dt.truncate(truncate_by)