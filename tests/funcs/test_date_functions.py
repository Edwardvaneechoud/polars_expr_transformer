import pytest
import polars as pl
from datetime import datetime, date, timedelta
from polars_expr_transformer.funcs.date_functions import (
    now, today, year, month, day, hour, minute, second,
    add_days, add_years, add_hours, add_minutes, add_seconds,
    add_months, add_weeks,
    datetime_diff_seconds, datetime_diff_nanoseconds, date_diff_days,
    date_trim, date_truncate,
    week, weekday, dayofweek, quarter, dayofyear,
    format_date, end_of_month, start_of_month,
)


# Mock datetime for testing
@pytest.fixture
def mock_datetime(monkeypatch):
    fixed_datetime = datetime(2023, 5, 15, 14, 30, 25)

    class MockDatetime:
        @classmethod
        def now(cls):
            return fixed_datetime

        @classmethod
        def today(cls):
            return fixed_datetime.replace(hour=0, minute=0, second=0)

    monkeypatch.setattr("polars_expr_transformer.funcs.date_functions.datetime", MockDatetime)
    return fixed_datetime


def test_now(mock_datetime):
    result = now()
    assert isinstance(result, pl.Expr)

    # Create a test dataframe to evaluate the expression
    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("now"))

    assert evaluated["now"][0] == mock_datetime


def test_today(mock_datetime):
    result = today()
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("today"))

    expected_date = mock_datetime.replace(hour=0, minute=0, second=0)
    assert evaluated["today"][0] == expected_date


def test_year():
    # Test with string input
    result = year("2023-05-15")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("year"))
    assert evaluated["year"][0] == 2023

    # Test with Polars expression
    df = pl.DataFrame({"date": [datetime(year=2023, month=5, day=15), datetime(year=2022, month=5, day=15)]})
    result = year(pl.col("date"))
    evaluated = df.select(result.alias("year"))
    assert evaluated["year"][0] == 2023
    assert evaluated["year"][1] == 2022


def test_month():
    # Test with string input
    result = month("2023-05-15")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("month"))
    assert evaluated["month"][0] == 5

    # Test with Polars expression
    df = pl.DataFrame({"date": [datetime(year=2023, month=5, day=15), datetime(year=2022, month=12, day=15)]})
    result = month(pl.col("date"))
    evaluated = df.select(result.alias("month"))
    assert evaluated["month"][0] == 5
    assert evaluated["month"][1] == 12


def test_day():
    # Test with string input
    result = day("2023-05-15")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("day"))
    assert evaluated["day"][0] == 15

    # Test with Polars expression
    df = pl.DataFrame({"date": [datetime(year=2023, month=5, day=15), datetime(year=2022, month=12, day=31)]})
    result = day(pl.col("date"))
    evaluated = df.select(result.alias("day"))
    assert evaluated["day"][0] == 15
    assert evaluated["day"][1] == 31


def test_hour():
    # Test with string input
    result = hour("2023-05-15 14:30:25")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("hour"))
    assert evaluated["hour"][0] == 14

    # Test with Polars expression
    df = pl.DataFrame({"dt": [datetime(2023, 5, 15, 14, 30, 25), datetime(2023, 5, 15, 0, 0, 0)]})
    result = hour(pl.col("dt"))
    evaluated = df.select(result.alias("hour"))
    assert evaluated["hour"][0] == 14
    assert evaluated["hour"][1] == 0


def test_minute():
    # Test with string input
    result = minute("2023-05-15 14:30:25")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("minute"))
    assert evaluated["minute"][0] == 30

    # Test with Polars expression
    df = pl.DataFrame({"dt": [datetime(2023, 5, 15, 14, 30, 25), datetime(2023, 5, 15, 14, 0, 0)]})
    result = minute(pl.col("dt"))
    evaluated = df.select(result.alias("minute"))
    assert evaluated["minute"][0] == 30
    assert evaluated["minute"][1] == 0


def test_second():
    # Test with string input
    result = second("2023-05-15 14:30:25")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("second"))
    assert evaluated["second"][0] == 25

    # Test with Polars expression
    df = pl.DataFrame({"dt": [datetime(2023, 5, 15, 14, 30, 25), datetime(2023, 5, 15, 14, 30, 0)]})
    result = second(pl.col("dt"))
    evaluated = df.select(result.alias("second"))
    assert evaluated["second"][0] == 25
    assert evaluated["second"][1] == 0


def test_add_days():
    # Test with string input
    result = add_days("2023-05-15", 5)
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2023, 5, 20)

    # Test with Polars expression
    df = pl.DataFrame({"date": [datetime(2023, 5, 15), datetime(2023, 12, 30)]})
    result = add_days(pl.col("date"), 5)
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2023, 5, 20)
    assert evaluated["result"][1] == datetime(2024, 1, 4)


def test_add_years():
    # add_years uses 365-day approximation
    result = add_years("2023-05-15", 1)
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("result"))
    # 365 days from 2023-05-15 = 2024-05-14 (2024 is a leap year)
    assert evaluated["result"][0] == datetime(2024, 5, 14)

    # Test with Polars expression
    df = pl.DataFrame({"date": [datetime(2023, 1, 1)]})
    result = add_years(pl.col("date"), 2)
    evaluated = df.select(result.alias("result"))
    # 730 days from 2023-01-01 = 2024-12-31
    assert evaluated["result"][0] == datetime(2024, 12, 31)


def test_add_hours():
    # Test with string input
    result = add_hours("2023-05-15 14:30:00", 3)
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2023, 5, 15, 17, 30, 0)

    # Test with Polars expression
    df = pl.DataFrame({"dt": [datetime(2023, 5, 15, 23, 0, 0)]})
    result = add_hours(pl.col("dt"), 3)
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2023, 5, 16, 2, 0, 0)


def test_add_minutes():
    # Test with string input
    result = add_minutes("2023-05-15 14:30:00", 15)
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2023, 5, 15, 14, 45, 0)

    # Test with Polars expression
    df = pl.DataFrame({"dt": [datetime(2023, 5, 15, 14, 50, 0)]})
    result = add_minutes(pl.col("dt"), 20)
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2023, 5, 15, 15, 10, 0)


def test_add_seconds():
    # Test with string input
    result = add_seconds("2023-05-15 14:30:00", 30)
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2023, 5, 15, 14, 30, 30)

    # Test with Polars expression
    df = pl.DataFrame({"dt": [datetime(2023, 5, 15, 14, 30, 50)]})
    result = add_seconds(pl.col("dt"), 20)
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2023, 5, 15, 14, 31, 10)


def test_add_months():
    # Test with string input
    result = add_months("2023-05-15", 2)
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2023, 7, 15)

    # Test with Polars expression
    df = pl.DataFrame({"date": [datetime(2023, 11, 15)]})
    result = add_months(pl.col("date"), 3)
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2024, 2, 15)


def test_add_weeks():
    # Test with string input
    result = add_weeks("2023-05-15", 2)
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2023, 5, 29)

    # Test with Polars expression
    df = pl.DataFrame({"date": [datetime(2023, 12, 25)]})
    result = add_weeks(pl.col("date"), 1)
    evaluated = df.select(result.alias("result"))
    assert evaluated["result"][0] == datetime(2024, 1, 1)


def test_datetime_diff_seconds():
    # Test with string inputs
    result = datetime_diff_seconds("2023-05-15 14:31:00", "2023-05-15 14:30:00")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("diff"))
    assert evaluated["diff"][0] == 60

    # Test with Polars expressions
    df = pl.DataFrame({
        "dt1": [datetime(2023, 5, 15, 15, 0, 0)],
        "dt2": [datetime(2023, 5, 15, 14, 0, 0)],
    })
    result = datetime_diff_seconds(pl.col("dt1"), pl.col("dt2"))
    evaluated = df.select(result.alias("diff"))
    assert evaluated["diff"][0] == 3600


def test_datetime_diff_nanoseconds():
    # Test with string inputs
    result = datetime_diff_nanoseconds("2023-05-15 14:31:00", "2023-05-15 14:30:00")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("diff"))
    assert evaluated["diff"][0] == 60_000_000_000

    # Test with Polars expressions
    df = pl.DataFrame({
        "dt1": [datetime(2023, 5, 15, 14, 30, 1)],
        "dt2": [datetime(2023, 5, 15, 14, 30, 0)],
    })
    result = datetime_diff_nanoseconds(pl.col("dt1"), pl.col("dt2"))
    evaluated = df.select(result.alias("diff"))
    assert evaluated["diff"][0] == 1_000_000_000


def test_date_diff_days():
    # Test with string inputs
    result = date_diff_days("2023-05-15", "2023-05-10")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("diff"))
    assert evaluated["diff"][0] == 5

    # Test with Polars expressions
    df = pl.DataFrame({
        "d1": [datetime(2023, 5, 15)],
        "d2": [datetime(2023, 1, 1)],
    })
    result = date_diff_days(pl.col("d1"), pl.col("d2"))
    evaluated = df.select(result.alias("diff"))
    assert evaluated["diff"][0] == 134


def test_date_trim():
    df = pl.DataFrame({"dt": [datetime(2023, 5, 15, 14, 30, 25)]})

    # Trim to day
    result = date_trim(pl.col("dt"), "day")
    evaluated = df.select(result.alias("trimmed"))
    assert evaluated["trimmed"][0] == datetime(2023, 5, 15, 0, 0, 0)

    # Trim to month
    result = date_trim(pl.col("dt"), "month")
    evaluated = df.select(result.alias("trimmed"))
    assert evaluated["trimmed"][0] == datetime(2023, 5, 1, 0, 0, 0)

    # Trim to year
    result = date_trim(pl.col("dt"), "year")
    evaluated = df.select(result.alias("trimmed"))
    assert evaluated["trimmed"][0] == datetime(2023, 1, 1, 0, 0, 0)

    # Trim to hour
    result = date_trim(pl.col("dt"), "hour")
    evaluated = df.select(result.alias("trimmed"))
    assert evaluated["trimmed"][0] == datetime(2023, 5, 15, 14, 0, 0)

    # Trim to minute
    result = date_trim(pl.col("dt"), "minute")
    evaluated = df.select(result.alias("trimmed"))
    assert evaluated["trimmed"][0] == datetime(2023, 5, 15, 14, 30, 0)

    # Trim to second
    result = date_trim(pl.col("dt"), "second")
    evaluated = df.select(result.alias("trimmed"))
    assert evaluated["trimmed"][0] == datetime(2023, 5, 15, 14, 30, 25)

    # Invalid part raises ValueError
    with pytest.raises(ValueError, match="Invalid part"):
        date_trim(pl.col("dt"), "invalid")


def test_date_truncate():
    df = pl.DataFrame({"dt": [datetime(2023, 5, 15, 14, 30, 25)]})

    # Truncate to 1 day
    result = date_truncate(pl.col("dt"), "1d")
    evaluated = df.select(result.alias("truncated"))
    assert evaluated["truncated"][0] == datetime(2023, 5, 15, 0, 0, 0)

    # Truncate to 1 hour
    result = date_truncate(pl.col("dt"), "1h")
    evaluated = df.select(result.alias("truncated"))
    assert evaluated["truncated"][0] == datetime(2023, 5, 15, 14, 0, 0)

    # Test with string column name
    result = date_truncate("dt", "1mo")
    evaluated = df.select(result.alias("truncated"))
    assert evaluated["truncated"][0] == datetime(2023, 5, 1, 0, 0, 0)


def test_week():
    # Test with string input
    result = week("2023-01-15")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("week"))
    assert evaluated["week"][0] == 2

    # Test with Polars expression
    df = pl.DataFrame({
        "date": [datetime(2023, 1, 1), datetime(2023, 6, 21), datetime(2023, 12, 31)]
    })
    result = week(pl.col("date"))
    evaluated = df.select(result.alias("week"))
    assert evaluated["week"][0] == 52  # 2023-01-01 is week 52 of 2022 (ISO)
    assert evaluated["week"][1] == 25
    assert evaluated["week"][2] == 52


def test_weekday():
    # 2023-05-15 is a Monday
    result = weekday("2023-05-15")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("weekday"))
    assert evaluated["weekday"][0] == 1  # Monday = 1

    # Test multiple days
    df = pl.DataFrame({
        "date": [
            datetime(2023, 5, 15),  # Monday
            datetime(2023, 5, 17),  # Wednesday
            datetime(2023, 5, 21),  # Sunday
        ]
    })
    result = weekday(pl.col("date"))
    evaluated = df.select(result.alias("weekday"))
    assert evaluated["weekday"][0] == 1  # Monday
    assert evaluated["weekday"][1] == 3  # Wednesday
    assert evaluated["weekday"][2] == 7  # Sunday


def test_dayofweek():
    # dayofweek is an alias for weekday
    df = pl.DataFrame({"date": [datetime(2023, 5, 15)]})  # Monday
    result_weekday = weekday(pl.col("date"))
    result_dayofweek = dayofweek(pl.col("date"))

    eval_wd = df.select(result_weekday.alias("wd"))
    eval_dow = df.select(result_dayofweek.alias("dow"))
    assert eval_wd["wd"][0] == eval_dow["dow"][0]
    assert eval_dow["dow"][0] == 1  # Monday


def test_quarter():
    # Test with string input
    result = quarter("2023-05-15")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("quarter"))
    assert evaluated["quarter"][0] == 2

    # Test all four quarters
    df = pl.DataFrame({
        "date": [
            datetime(2023, 1, 15),   # Q1
            datetime(2023, 4, 15),   # Q2
            datetime(2023, 8, 15),   # Q3
            datetime(2023, 11, 15),  # Q4
        ]
    })
    result = quarter(pl.col("date"))
    evaluated = df.select(result.alias("quarter"))
    assert evaluated["quarter"][0] == 1
    assert evaluated["quarter"][1] == 2
    assert evaluated["quarter"][2] == 3
    assert evaluated["quarter"][3] == 4


def test_dayofyear():
    # Test with string input
    result = dayofyear("2023-02-01")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("doy"))
    assert evaluated["doy"][0] == 32

    # Test with Polars expression
    df = pl.DataFrame({
        "date": [datetime(2023, 1, 1), datetime(2023, 12, 31)]
    })
    result = dayofyear(pl.col("date"))
    evaluated = df.select(result.alias("doy"))
    assert evaluated["doy"][0] == 1
    assert evaluated["doy"][1] == 365


def test_format_date():
    # Test with default format
    result = format_date("2023-05-15")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("formatted"))
    assert evaluated["formatted"][0] == "2023-05-15"

    # Test with custom format
    result = format_date("2023-05-15 14:30:00", "%Y/%m/%d %H:%M")
    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("formatted"))
    assert evaluated["formatted"][0] == "2023/05/15 14:30"

    # Test with Polars expression
    df = pl.DataFrame({"date": [datetime(2023, 5, 15, 14, 30, 0)]})
    result = format_date(pl.col("date"), "%d-%m-%Y")
    evaluated = df.select(result.alias("formatted"))
    assert evaluated["formatted"][0] == "15-05-2023"


def test_end_of_month():
    # Test with string input
    result = end_of_month("2023-05-15")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("eom"))
    assert evaluated["eom"][0].day == 31

    # Test with Polars expression - various months
    df = pl.DataFrame({
        "date": [
            datetime(2023, 2, 10),   # February (non-leap)
            datetime(2024, 2, 10),   # February (leap year)
            datetime(2023, 4, 15),   # April (30 days)
        ]
    })
    result = end_of_month(pl.col("date"))
    evaluated = df.select(result.alias("eom"))
    assert evaluated["eom"][0].day == 28
    assert evaluated["eom"][1].day == 29
    assert evaluated["eom"][2].day == 30


def test_start_of_month():
    # Test with string input
    result = start_of_month("2023-05-15")
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"dummy": [1]})
    evaluated = df.select(result.alias("som"))
    assert evaluated["som"][0].day == 1

    # Test with Polars expression
    df = pl.DataFrame({
        "date": [datetime(2023, 5, 15), datetime(2023, 12, 25)]
    })
    result = start_of_month(pl.col("date"))
    evaluated = df.select(result.alias("som"))
    assert evaluated["som"][0] == datetime(2023, 5, 1)
    assert evaluated["som"][1] == datetime(2023, 12, 1)
