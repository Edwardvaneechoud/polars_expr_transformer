"""Date functions reading a text column, and what they emit as code.

A date function lowers to ``.dt.<something>()``, which Polars refuses on a
String column. Given a schema, the text-to-date parse is inserted for such a
column; without one nothing changes, and a column that is already Date or
Datetime is never touched either way.
"""

import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_expr_transformer import (
    simple_function_to_expr,
    to_flowframe_code,
    to_polars_code,
)

TEXT = ["2005-01-10"]
DATE = [dt.date(2005, 1, 10)]
DATETIME = [dt.datetime(2005, 1, 10, 0, 0)]


@pytest.fixture
def text_df():
    return pl.DataFrame({"Join Date": TEXT})


def evaluate(df, expr_str):
    return df.select(simple_function_to_expr(expr_str, schema=df.schema).alias("r"))["r"]


class TestReportedCase:
    def test_format_date_on_text_column(self, text_df):
        result = evaluate(text_df, 'format_date([Join Date], "%A, %B %d, %Y")')
        assert result.to_list() == ["Monday, January 10, 2005"]

    def test_matches_the_explicit_to_date_workaround(self, text_df):
        with_schema = evaluate(text_df, 'format_date([Join Date], "%A")')
        explicit = text_df.select(
            simple_function_to_expr(
                'format_date(to_date([Join Date], "%Y-%m-%d"), "%A")'
            ).alias("r")
        )["r"]
        assert with_schema.to_list() == explicit.to_list()

    def test_without_a_schema_nothing_changes(self, text_df):
        """The no-schema path is deliberately untouched: the dtype is unknowable."""
        with pytest.raises(pl.exceptions.InvalidOperationError):
            text_df.select(simple_function_to_expr('format_date([Join Date], "%A")'))


class TestTextColumns:
    @pytest.mark.parametrize(
        "expr_str,expected",
        [
            ('format_date([Join Date], "%Y/%m/%d")', ["2005/01/10"]),
            ("year([Join Date])", [2005]),
            ("add_days([Join Date], 5)", [dt.datetime(2005, 1, 15)]),
            ("end_of_month([Join Date])", [dt.datetime(2005, 1, 31)]),
        ],
    )
    def test_text_column_is_parsed(self, text_df, expr_str, expected):
        assert evaluate(text_df, expr_str).to_list() == expected

    @pytest.mark.parametrize(
        "text",
        ["2005-01-10", "2005-01-10 13:45:00", "2005-01-10T13:45:00", "2005-01-10 13:45:00.500"],
    )
    def test_iso_layouts_in_the_ladder(self, text):
        df = pl.DataFrame({"Join Date": [text]})
        assert evaluate(df, "year([Join Date])").to_list() == [2005]

    @pytest.mark.parametrize(
        "expr_str",
        [
            'format_date([Join Date], "%A")',
            "year([Join Date])",
            "add_days([Join Date], 5)",
            "end_of_month([Join Date])",
        ],
    )
    def test_unparseable_text_becomes_null_not_an_error(self, expr_str):
        """Following to_date/to_datetime, which already pass strict=False."""
        df = pl.DataFrame({"Join Date": ["not a date", "10/01/2005"]})
        assert evaluate(df, expr_str).to_list() == [None, None]

    @pytest.mark.parametrize(
        "expr_str",
        [
            'format_date([Join Date], "%A")',
            "year([Join Date])",
            "add_days([Join Date], 5)",
            "end_of_month([Join Date])",
        ],
    )
    def test_null_text_stays_null(self, expr_str):
        df = pl.DataFrame({"Join Date": pl.Series([None, "2005-01-10"], dtype=pl.String)})
        assert evaluate(df, expr_str)[0] is None

    def test_a_column_of_only_unparseable_text_does_not_raise(self):
        """Polars' own format inference raises here, so the ladder is explicit."""
        df = pl.DataFrame({"Join Date": ["nope", "also nope"]})
        assert evaluate(df, "year([Join Date])").to_list() == [None, None]

    def test_both_arguments_of_a_two_date_function(self):
        df = pl.DataFrame({"a": ["2005-02-10"], "b": ["2005-01-10"]})
        assert evaluate(df, "date_diff_days([a], [b])").to_list() == [31]


class TestTemporalColumnsAreUntouched:
    """A Date or Datetime column must produce exactly what it produces today."""

    @pytest.mark.parametrize("values,dtype", [(DATE, pl.Date), (DATETIME, pl.Datetime)])
    @pytest.mark.parametrize(
        "expr_str",
        [
            'format_date([Join Date], "%A")',
            "year([Join Date])",
            "add_days([Join Date], 5)",
            "end_of_month([Join Date])",
        ],
    )
    def test_result_is_identical_with_and_without_a_schema(self, values, dtype, expr_str):
        df = pl.DataFrame({"Join Date": pl.Series(values, dtype=dtype)})
        with_schema = df.select(simple_function_to_expr(expr_str, schema=df.schema))
        without_schema = df.select(simple_function_to_expr(expr_str))
        assert_frame_equal(with_schema, without_schema)

    def test_date_column_keeps_its_dtype(self):
        df = pl.DataFrame({"Join Date": pl.Series(DATE, dtype=pl.Date)})
        assert evaluate(df, "add_days([Join Date], 5)").dtype == pl.Date

    def test_code_generation_inserts_no_parse(self):
        schema = {"Join Date": pl.Date}
        assert (
            to_polars_code('format_date([Join Date], "%A")', schema=schema)
            == 'pl.col("Join Date").dt.to_string("%A")'
        )


class TestWhatIsAndIsNotCoerced:
    def test_a_text_literal_needs_no_schema(self):
        """Its type is known outright, so the literal path parses as it says it does."""
        result = pl.select(
            simple_function_to_expr('format_date("2005-01-10", "%A")').alias("r")
        )["r"]
        assert result.to_list() == ["Monday"]

    def test_an_explicit_to_date_is_not_parsed_twice(self, text_df):
        code = to_polars_code(
            'format_date(to_date([Join Date], "%Y-%m-%d"), "%A")', schema=text_df.schema
        )
        assert code == 'pl.col("Join Date").str.to_date("%Y-%m-%d").dt.to_string("%A")'

    def test_a_non_date_parameter_is_left_alone(self, text_df):
        """date_format is a plain format string, not something to parse."""
        code = to_polars_code('format_date([Join Date], "%A")', schema={"Join Date": pl.Date})
        assert '"%A"' in code and "to_datetime" not in code

    def test_a_column_missing_from_the_schema_is_left_alone(self):
        code = to_polars_code("year([other])", schema={"Join Date": pl.String})
        assert code == 'pl.col("other").dt.year()'

    def test_a_non_string_column_is_left_alone(self):
        code = to_polars_code("year([n])", schema={"n": pl.Int64})
        assert code == 'pl.col("n").dt.year()'

    def test_a_schema_that_is_not_a_mapping_is_rejected(self):
        with pytest.raises(TypeError, match="mapping of column name"):
            simple_function_to_expr("year([d])", schema=["d"])

    def test_coercion_reaches_inside_a_conditional(self):
        df = pl.DataFrame({"Join Date": ["2005-01-10"]})
        result = evaluate(df, 'if year([Join Date]) > 2000 then "new" else "old" endif')
        assert result.to_list() == ["new"]


class TestGeneratedCode:
    def test_the_emitted_code_reads_as_polars(self, text_df):
        code = to_polars_code('format_date([Join Date], "%A, %B %d, %Y")', schema=text_df.schema)
        assert code == (
            'pl.coalesce(['
            'pl.col("Join Date").str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False), '
            'pl.col("Join Date").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False), '
            'pl.col("Join Date").str.to_datetime("%Y-%m-%d", strict=False)'
            ']).dt.to_string("%A, %B %d, %Y")'
        )

    def test_flowframe_code_uses_the_ff_prefix(self, text_df):
        code = to_flowframe_code("year([Join Date])", schema=text_df.schema)
        assert code.startswith("ff.coalesce([ff.col(") and code.endswith(").dt.year()")
        assert "pl." not in code

    @pytest.mark.parametrize(
        "expr_str",
        [
            'format_date([Join Date], "%A")',
            "year([Join Date])",
            "add_days([Join Date], 5)",
            "end_of_month([Join Date])",
            "date_diff_days([Join Date], [Join Date])",
        ],
    )
    def test_emitted_code_matches_the_live_expression(self, text_df, expr_str):
        """The two render paths must not disagree about what was parsed."""
        code = to_polars_code(expr_str, schema=text_df.schema)
        from_code = text_df.select(eval(code, {"pl": pl}).alias("r"))
        live = text_df.select(
            simple_function_to_expr(expr_str, schema=text_df.schema).alias("r")
        )
        assert_frame_equal(from_code, live)

    def test_no_unknown_function_warning(self, text_df, recwarn):
        to_polars_code('format_date([Join Date], "%A")', schema=text_df.schema)
        assert [w for w in recwarn if "Unknown function" in str(w.message)] == []
