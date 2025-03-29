import pytest
import polars as pl
from datetime import datetime, timedelta
from polars_expr_transformer.funcs.date_functions import (
    now, today, year, month, day
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