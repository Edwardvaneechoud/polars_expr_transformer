import polars as pl
from typing import Any
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col, create_fix_date_col
from datetime import datetime
from polars_expr_transformer.funcs.utils import PlStringType, PlIntType


def now() -> pl.Expr:
    """
    Gets the current date and time.

    For example, now() would return the current date and time.

    Returns:
    - The current date and time
    """
    return pl.lit(datetime.now())


def today() -> pl.Expr:
    """
    Gets the current date.

    For example, today() would return the current date.

    Returns:
    - The current date
    """
    return pl.lit(datetime.today())


def year(date_value: Any) -> pl.Expr:
    """
    Gets the year from a date.

    For example, year([hire_date]) would return 2021 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to extract the year from

    Returns:
    - The year as a number
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.year()


def month(date_value: Any) -> pl.Expr:
    """
    Gets the month from a date.

    For example, month([hire_date]) would return 3 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to extract the month from

    Returns:
    - The month as a number (1-12)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_col(date_value).str.to_datetime()
    return date_value.dt.month()


def day(date_value: PlStringType) -> pl.Expr:
    """
    Gets the day of the month from a date.

    For example, day([hire_date]) would return 15 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to extract the day from

    Returns:
    - The day of the month as a number (1-31)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.day()


def hour(date_value: PlStringType) -> pl.Expr:
    """
    Gets the hour from a datetime.

    For example, hour([order_date]) would return 10 when [order_date] is 2024-01-15 10:30:45.

    Parameters:
    - date_value: The datetime column or expression to extract the hour from

    Returns:
    - The hour as a number (0-23)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.hour()


def minute(date_value: PlStringType) -> pl.Expr:
    """
    Gets the minute from a datetime.

    For example, minute([order_date]) would return 30 when [order_date] is 2024-01-15 10:30:45.

    Parameters:
    - date_value: The datetime column or expression to extract the minute from

    Returns:
    - The minute as a number (0-59)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.minute()


def second(date_value: PlStringType) -> pl.Expr:
    """
    Gets the second from a datetime.

    For example, second([order_date]) would return 45 when [order_date] is 2024-01-15 10:30:45.

    Parameters:
    - date_value: The datetime column or expression to extract the second from

    Returns:
    - The second as a number (0-59)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.second()


def add_days(date_value: PlStringType, days: PlIntType) -> pl.Expr:
    """
    Adds a number of days to a date.

    For example, add_days([hire_date], 5) would return 2021-03-20 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to add days to
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

    For example, add_years([hire_date], 1) would return 2022-03-15 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to add years to
    - years: How many years to add

    Returns:
    - The new date
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    years = years if is_polars_expr(years) else create_fix_col(years)
    return date_value + pl.duration(days=years * 365)


def add_hours(date_value: PlStringType, hours: PlIntType) -> pl.Expr:
    """
    Adds a number of hours to a datetime.

    For example, add_hours([order_date], 3) would return 2024-01-15 13:30:00 when [order_date] is 2024-01-15 10:30:00.

    Parameters:
    - date_value: The datetime column or expression to add hours to
    - hours: How many hours to add

    Returns:
    - The new date and time
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    hours = hours if is_polars_expr(hours) else create_fix_col(hours)
    return date_value + pl.duration(hours=hours)


def add_minutes(date_value: PlStringType, minutes: PlIntType) -> pl.Expr:
    """
    Adds a number of minutes to a datetime.

    For example, add_minutes([order_date], 15) would return 2024-01-15 10:45:00 when [order_date] is 2024-01-15 10:30:00.

    Parameters:
    - date_value: The datetime column or expression to add minutes to
    - minutes: How many minutes to add

    Returns:
    - The new date and time
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    minutes = minutes if is_polars_expr(minutes) else create_fix_col(minutes)
    return date_value + pl.duration(minutes=minutes)


def add_seconds(date_value: PlStringType, seconds: PlIntType) -> pl.Expr:
    """
    Adds a number of seconds to a datetime.

    For example, add_seconds([order_date], 30) would return 2024-01-15 10:30:30 when [order_date] is 2024-01-15 10:30:00.

    Parameters:
    - date_value: The datetime column or expression to add seconds to
    - seconds: How many seconds to add

    Returns:
    - The new date and time
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    seconds = seconds if is_polars_expr(seconds) else create_fix_col(seconds)
    return date_value + pl.duration(seconds=seconds)


def datetime_diff_seconds(date1: PlStringType, date2: PlStringType) -> pl.Expr:
    """
    Calculates the number of seconds between two datetimes (the first minus the second).

    For example, datetime_diff_seconds([end], [start]) would return 3600 when [start] is 2024-01-15 10:00:00 and [end] is 2024-01-15 11:00:00.

    Parameters:
    - date1: The datetime column or expression to subtract from
    - date2: The datetime column or expression to subtract

    Returns:
    - The number of seconds between the two datetimes
    """
    date_value1 = date1 if is_polars_expr(date1) else create_fix_date_col(date1)
    date_value2 = date2 if is_polars_expr(date2) else create_fix_date_col(date2)
    return (date_value1 - date_value2).dt.total_seconds()


def datetime_diff_nanoseconds(date1: PlStringType, date2: PlStringType) -> pl.Expr:
    """
    Calculates the number of nanoseconds between two datetimes (the first minus the second), for very precise time measurements.

    For example, datetime_diff_nanoseconds([end], [start]) would return 1000000000 when [start] is 2024-01-15 10:00:00 and [end] is 2024-01-15 10:00:01.

    Parameters:
    - date1: The datetime column or expression to subtract from
    - date2: The datetime column or expression to subtract

    Returns:
    - The number of nanoseconds between the two datetimes
    """
    date_value1 = date1 if is_polars_expr(date1) else create_fix_date_col(date1)
    date_value2 = date2 if is_polars_expr(date2) else create_fix_date_col(date2)
    return (date_value1 - date_value2).dt.total_nanoseconds()


def date_diff_days(date1: PlStringType, date2: PlStringType) -> pl.Expr:
    """
    Calculates the number of days between two dates (the first minus the second).

    For example, date_diff_days(today(), [hire_date]) would return 30 when [hire_date] is 30 days before today.

    Parameters:
    - date1: The date column or expression to subtract from
    - date2: The date column or expression to subtract

    Returns:
    - The number of days between the two dates
    """
    date_value1 = date1 if is_polars_expr(date1) else create_fix_date_col(date1)
    date_value2 = date2 if is_polars_expr(date2) else create_fix_date_col(date2)
    return (date_value1 - date_value2).dt.total_days()


def date_trim(date_value: Any, part: str) -> pl.Expr:
    """
    Removes the smaller parts of a date or time.

    For example, date_trim([order_date], "day") would return 2024-01-15 00:00:00 when [order_date] is 2024-01-15 10:30:45.

    Parameters:
    - date_value: The datetime column or expression to trim
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
        return date_value.dt.truncate('1m')
    elif part == 'second':
        return date_value.dt.truncate('1s')
    else:
        raise ValueError(
            f"Invalid part '{part}' specified. Must be 'year', 'month', 'day', 'hour', 'minute', or 'second'.")


def date_truncate(date_value: Any, truncate_by: str) -> pl.Expr:
    """
    Rounds a date down to the nearest specified unit.

    For example, date_truncate([order_date], "1d") would return 2024-01-15 00:00:00 when [order_date] is 2024-01-15 10:30:45.

    Parameters:
    - date_value: The datetime column or expression to truncate
    - truncate_by: The time unit to round down to (like "1d", "2h", "15m")
      Some examples:
      - "1y": Round to the start of the year
      - "3mo": Round to the nearest 3-month boundary
      - "1w": Round to the start of the week
      - "12h": Round to the nearest 12-hour boundary

    Returns:
    - The truncated date and time
    """
    date_value = date_value if isinstance(date_value, pl.Expr) else pl.col(date_value)
    return date_value.dt.truncate(truncate_by)


def add_months(date_value: PlStringType, months: PlIntType) -> pl.Expr:
    """
    Adds a number of months to a date.

    For example, add_months([hire_date], 2) would return 2021-05-15 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to add months to
    - months: How many months to add

    Returns:
    - The new date
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    months = months if is_polars_expr(months) else pl.lit(months)
    return date_value.dt.offset_by(pl.concat_str([months.cast(pl.Utf8), pl.lit("mo")]))


def add_weeks(date_value: PlStringType, weeks: PlIntType) -> pl.Expr:
    """
    Adds a number of weeks to a date.

    For example, add_weeks([hire_date], 2) would return 2021-03-29 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to add weeks to
    - weeks: How many weeks to add

    Returns:
    - The new date
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    weeks = weeks if is_polars_expr(weeks) else create_fix_col(weeks)
    return date_value + pl.duration(weeks=weeks)


def week(date_value: PlStringType) -> pl.Expr:
    """
    Gets the ISO week number from a date (1-53).

    For example, week([hire_date]) would return 11 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to extract the week from

    Returns:
    - The week number as a number (1-53)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.week()


def weekday(date_value: PlStringType) -> pl.Expr:
    """
    Gets the day of the week from a date (1=Monday, 7=Sunday).

    For example, weekday([hire_date]) would return 1 when [hire_date] is 2021-03-15, a Monday.

    Parameters:
    - date_value: The date column or expression to extract the weekday from

    Returns:
    - The day of the week as a number (1-7)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.weekday()


def dayofweek(date_value: PlStringType) -> pl.Expr:
    """
    Gets the day of the week from a date (alias for weekday, 1=Monday, 7=Sunday).

    For example, dayofweek([hire_date]) would return 1 when [hire_date] is 2021-03-15, a Monday.

    Parameters:
    - date_value: The date column or expression to extract the day of week from

    Returns:
    - The day of the week as a number (1-7)
    """
    return weekday(date_value)


def quarter(date_value: PlStringType) -> pl.Expr:
    """
    Gets the quarter from a date (1-4).

    For example, quarter([hire_date]) would return 1 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to extract the quarter from

    Returns:
    - The quarter as a number (1-4)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.quarter()


def dayofyear(date_value: PlStringType) -> pl.Expr:
    """
    Gets the day of the year from a date (1-366).

    For example, dayofyear([hire_date]) would return 74 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to extract the day of year from

    Returns:
    - The day of the year as a number (1-366)
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.ordinal_day()


def format_date(date_value: PlStringType, date_format: str = "%Y-%m-%d") -> pl.Expr:
    """
    Formats a date as text using a specified format.

    For example, format_date([hire_date], "%B %d, %Y") would return "March 15, 2021" when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to format
    - date_format: The output format string (default is year-month-day)
      Common format codes:
      - %Y: Four-digit year (e.g., 2023)
      - %m: Two-digit month (01-12)
      - %d: Two-digit day (01-31)
      - %B: Full month name (January, February)
      - %b: Month abbreviation (Jan, Feb)
      - %A: Full weekday name (Monday, Tuesday)
      - %H: Hour (00-23)
      - %M: Minute (00-59)
      - %S: Second (00-59)

    Returns:
    - The formatted date as text
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.to_string(date_format)


def end_of_month(date_value: PlStringType) -> pl.Expr:
    """
    Gets the last day of the month for a given date.

    For example, end_of_month([hire_date]) would return 2021-03-31 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to get the end of month for

    Returns:
    - The last day of the month
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.month_end()


def start_of_month(date_value: PlStringType) -> pl.Expr:
    """
    Gets the first day of the month for a given date.

    For example, start_of_month([hire_date]) would return 2021-03-01 when [hire_date] is 2021-03-15.

    Parameters:
    - date_value: The date column or expression to get the start of month for

    Returns:
    - The first day of the month
    """
    date_value = date_value if is_polars_expr(date_value) else create_fix_date_col(date_value)
    return date_value.dt.month_start()