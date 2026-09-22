"""String functions reading a column that does not hold text.

Every ``.str`` method refuses a non-String column outright, so a filter as
ordinary as ``contains([order_date], "2026-09")`` used to raise instead of
running. The string functions now render such a column as text first: a Date as
``YYYY-MM-DD``, a Datetime as ``YYYY-MM-DD HH:MM:SS``, and anything else the
way ``cast(pl.String)`` would. Text columns are handed through untouched.
"""

import datetime as dt

import polars as pl
import pytest

from polars_expr_transformer import simple_function_to_expr


DATE = dt.date(2026, 9, 24)
DATETIME = dt.datetime(2026, 9, 24, 10, 0, 0)


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "d": [DATE, None],
            "ts": [DATETIME, None],
            "i": [1234, None],
            "f": [12.5, None],
            "b": [True, None],
            "s": ["  Amsterdam  ", None],
        }
    )


def evaluate(df, expr_str):
    return df.select(simple_function_to_expr(expr_str).alias("r"))["r"].to_list()


class TestReportedCase:
    """The filters from the Flowfile report, which used to raise."""

    def test_contains_on_a_date_column(self, df):
        assert df.filter(simple_function_to_expr('contains([d], "2026-09")')).height == 1

    def test_left_on_a_datetime_column(self, df):
        assert df.filter(simple_function_to_expr('left([ts], 4) = "2026"')).height == 1


class TestDateColumns:
    @pytest.mark.parametrize(
        "expr_str,expected",
        [
            ('contains([d], "2026-09")', [True, None]),
            ("left([d], 4)", ["2026", None]),
            ("right([d], 2)", ["24", None]),
            ('starts_with([d], "2026")', [True, None]),
            ('ends_with([d], "-24")', [True, None]),
            ("length([d])", [10, None]),
            ("uppercase([d])", ["2026-09-24", None]),
            ('substring([d], 5, 2)', ["09", None]),
            ('replace([d], "-", "/")', ["2026/09/24", None]),
        ],
    )
    def test_renders_as_yyyy_mm_dd(self, df, expr_str, expected):
        assert evaluate(df, expr_str) == expected


class TestDatetimeColumns:
    @pytest.mark.parametrize(
        "expr_str,expected",
        [
            ('contains([ts], "2026-09")', [True, None]),
            ("left([ts], 10)", ["2026-09-24", None]),
            ("right([ts], 8)", ["10:00:00", None]),
            ("length([ts])", [19, None]),
        ],
    )
    def test_renders_as_yyyy_mm_dd_hh_mm_ss(self, df, expr_str, expected):
        assert evaluate(df, expr_str) == expected

    def test_ends_with_the_clock_time(self, df):
        """``cast(pl.String)`` would append ``.000000`` and miss this."""
        assert evaluate(df, 'ends_with([ts], "10:00:00")') == [True, None]

    def test_right_gives_the_clock_time(self, df):
        assert evaluate(df, 'right([ts], 8) = "10:00:00"') == [True, None]

    def test_sub_second_precision_is_dropped(self):
        """A Datetime reads as its clock time, whatever it stores underneath."""
        precise = pl.DataFrame(
            {"ts": [dt.datetime(2026, 9, 24, 10, 0, 0, 123456)]}
        )
        assert evaluate(precise, "to_string(length([ts]))") == ["19"]
        assert evaluate(precise, 'ends_with([ts], "10:00:00")') == [True]


class TestNumericAndBooleanColumns:
    @pytest.mark.parametrize(
        "expr_str,expected",
        [
            ('contains([i], "23")', [True, None]),
            ("left([i], 2)", ["12", None]),
            ("right([i], 2)", ["34", None]),
            ("length([i])", [4, None]),
            ('pad_left([i], 8, "0")', ["00001234", None]),
            ('contains([f], "12.5")', [True, None]),
            ('contains([b], "true")', [True, None]),
            ("uppercase([b])", ["TRUE", None]),
        ],
    )
    def test_renders_the_way_a_cast_would(self, df, expr_str, expected):
        assert evaluate(df, expr_str) == expected


class TestMembershipOperators:
    """``in`` lowers to ``contains``, so it gets the same treatment."""

    def test_in_against_a_date_column(self, df):
        assert evaluate(df, '[d] in "shipped 2026-09-24 by air"') == [True, None]

    def test_not_in_against_an_int_column(self, df):
        assert evaluate(df, '[i] not in "no digits here"') == [True, None]


class TestTextColumnsAreUntouched:
    @pytest.mark.parametrize(
        "expr_str,expected",
        [
            ("trim([s])", ["Amsterdam", None]),
            ("left_trim([s])", ["Amsterdam  ", None]),
            ("right_trim([s])", ["  Amsterdam", None]),
            ("uppercase([s])", ["  AMSTERDAM  ", None]),
            ("length([s])", [13, None]),
            ('contains([s], "Amsterdam")', [True, None]),
        ],
    )
    def test_existing_formulas_keep_their_behaviour(self, df, expr_str, expected):
        assert evaluate(df, expr_str) == expected

    def test_text_that_looks_like_a_timestamp_keeps_its_fraction(self):
        """The cast is chosen per dtype, so text is never reformatted."""
        stamps = pl.DataFrame({"s": ["2026-09-24 10:00:00.123", "other"]})
        assert evaluate(stamps, 'ends_with([s], ".123")') == [True, False]
        assert evaluate(stamps, 'contains([s], "10:00:00.123")') == [True, False]


class TestStillWorksLazily:
    def test_filter_on_a_lazy_frame(self, df):
        result = df.lazy().filter(simple_function_to_expr('contains([d], "2026-09")'))
        assert result.collect().height == 1

    def test_schema_is_known_without_collecting(self, df):
        lf = df.lazy().select(simple_function_to_expr("left([ts], 4)").alias("year"))
        assert lf.collect_schema()["year"] == pl.String
