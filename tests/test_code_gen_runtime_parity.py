"""Generated code must compute what the formula computes.

Each formula is evaluated twice over the same frame: once through
``simple_function_to_expr`` and once by ``eval``-ing ``to_polars_code``. Both
must give identical rows and dtypes. The frame holds the values the conversion
functions treat specially: nulls, "yes"/"no" text, number text, zeros, ISO and
non-ISO dates, and text that parses as nothing.
"""

import datetime
import hashlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_expr_transformer import (
    simple_function_to_expr,
    to_flowframe_code,
    to_polars_code,
)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "text": ["yes", "No", "TRUE", "f", "Y", "1", "0", "0.00", "-0", "00",
                     "-3.5", "2", "maybe", "", " yes", None],
            "int": [1, 0, -2, None, 5, 0, 3, 1, 0, 7, 10, -1, 0, 2, 1, None],
            "float": [1.5, 0.0, -0.0, None, 2.0, -3.25, 1e20, float("nan"), 0.5, 0.0,
                      None, 1.0, -1.0, 0.001, 100.0, 3.0],
            "flag": [True, False, None, True, False, True, None, False, True, False,
                     True, None, False, True, False, True],
            "date": ["2023-05-15", "15/05/2023", "bad", None, "2023-12-31", "2023-02-30",
                     "2023-01-01 10:00:00", "20230101", "2023-1-5", "", "2024-02-29",
                     "05/15/2023", "2023-05-15T00:00:00", " 2023-05-15", "1999-01-01", "2023/05/15"],
            "datetime": ["2023-05-15 14:30:00", "2023-05-15", "bad", None, "2023-05-15T14:30:00",
                         "15/05/2023 14:30", "2023-05-15 14:30", "2023-12-31 23:59:59", "x", "",
                         "2023-05-15 14:30:00.123", "2023-05-15 25:00:00", "15/05/2023 09:05",
                         "2000-01-01 00:00:00", "2023-05-15 14:30:00 ", "01/01/2000 00:00"],
        }
    )


def _generated_expr(formula: str, **kwargs) -> pl.Expr:
    code = to_polars_code(formula, **kwargs)
    return eval(code, {"pl": pl, "datetime": datetime, "hashlib": hashlib})


def assert_same_result(df: pl.DataFrame, formula: str, **kwargs) -> None:
    expected = df.select(simple_function_to_expr(formula, **kwargs).alias("result"))
    actual = df.select(_generated_expr(formula, **kwargs).alias("result"))
    assert_frame_equal(actual, expected)


CONVERSION_FORMULAS = [
    "to_boolean([text])",
    "to_boolean([int])",
    "to_boolean([float])",
    "to_boolean([flag])",
    "to_boolean('yes')",
    "to_boolean('No')",
    "to_boolean('2')",
    "to_boolean('maybe')",
    "to_boolean(0)",
    "to_boolean(1.5)",
    "to_boolean(null)",
    "to_boolean(true)",
    "to_date([date])",
    'to_date([date], "%d/%m/%Y")',
    'to_date([date], "%m/%d/%Y")',
    'to_date([date], "%Y%m%d")',
    "to_date('2023-05-15')",
    "to_date('15/05/2023')",
    "to_date('bad')",
    "to_datetime([datetime])",
    'to_datetime([datetime], "%d/%m/%Y %H:%M")',
    'to_datetime([datetime], "%Y-%m-%dT%H:%M:%S")',
    'to_datetime([date], "%Y-%m-%d")',
    "to_datetime('2023-05-15 14:30:00')",
    "to_datetime('bad')",
]


@pytest.mark.parametrize("formula", CONVERSION_FORMULAS)
def test_conversion_matches_the_function(df, formula):
    assert_same_result(df, formula)


@pytest.mark.parametrize(
    "formula",
    [
        "to_boolean(lowercase([text]))",
        "to_boolean([int] + 1)",
        "to_boolean(to_string([int]))",
        "if to_boolean([text]) then 'on' else 'off' endif",
        "to_boolean([text]) and to_boolean([int])",
        "year(to_date([date]))",
        'format_date(to_date([date], "%d/%m/%Y"), "%Y-%m-%d")',
        "add_days(to_date([date]), 1)",
        "date_diff_days(to_date([date]), to_date('2023-01-01'))",
        "hour(to_datetime([datetime]))",
        "coalesce(to_date([date]), to_date([date], \"%d/%m/%Y\"))",
        "to_date(concat(to_string(year(to_date([date]))), '-01-01'))",
        "is_empty(to_datetime([datetime]))",
    ],
)
def test_conversion_inside_other_expressions_matches(df, formula):
    assert_same_result(df, formula)


@pytest.mark.parametrize(
    "formula",
    ["to_date([date])", "year(to_date([date]))", "to_boolean([text])", "to_datetime([datetime])"],
)
def test_parity_holds_with_a_schema(df, formula):
    assert_same_result(df, formula, schema=df.schema)


@pytest.mark.parametrize(
    "formula",
    ["to_boolean([text])", "to_date([date])", 'to_date([date], "%d/%m/%Y")', "to_datetime([datetime])"],
)
def test_flowframe_code_is_the_same_expression(formula):
    pl_code = to_polars_code(formula)
    ff_code = to_flowframe_code(formula)
    assert "pl." not in ff_code
    assert ff_code.replace("ff.", "pl.") == pl_code


class TestGeneratedCode:
    def test_to_date_states_its_default_format_and_is_lenient(self):
        assert to_polars_code("to_date([d])") == 'pl.col("d").str.to_date("%Y-%m-%d", strict=False)'

    def test_to_datetime_states_its_default_format_and_is_lenient(self):
        assert to_polars_code("to_datetime([d])") == (
            'pl.col("d").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)'
        )

    def test_an_explicit_format_is_kept(self):
        assert to_polars_code('to_date([d], "%d/%m/%Y")') == (
            'pl.col("d").str.to_date("%d/%m/%Y", strict=False)'
        )

    def test_to_boolean_is_not_a_cast(self):
        code = to_polars_code("to_boolean([b])")
        assert "cast(pl.Boolean)" not in code
        assert code.startswith('pl.when(pl.col("b").cast(pl.Utf8).str.to_lowercase().is_in(')
        assert code.endswith(".otherwise(pl.lit(False))")
