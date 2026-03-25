"""Tests for the Polars interceptor that generates string DSL from Polars-like code."""

import polars as pl
from polars.testing import assert_frame_equal

from polars_expr_transformer.interceptor import PolarsInterceptor


pi = PolarsInterceptor()


# ──────────────────────────────────────────────
# DSL generation tests
# ──────────────────────────────────────────────

class TestDslGeneration:
    """Test that the interceptor generates correct DSL strings."""

    def test_simple_column(self):
        expr = pi.col("name")
        assert expr.to_dsl() == "[name]"

    def test_literal_string(self):
        expr = pi.lit("hello")
        assert expr.to_dsl() == '"hello"'

    def test_literal_number(self):
        expr = pi.lit(42)
        assert expr.to_dsl() == "42"

    def test_add_columns(self):
        expr = pi.col("a") + pi.col("b")
        assert expr.to_dsl() == "[a] + [b]"

    def test_subtract(self):
        expr = pi.col("a") - pi.col("b")
        assert expr.to_dsl() == "[a] - [b]"

    def test_multiply(self):
        expr = pi.col("a") * pi.lit(2)
        assert expr.to_dsl() == "[a] * 2"

    def test_divide(self):
        expr = pi.col("a") / pi.lit(10)
        assert expr.to_dsl() == "[a] / 10"

    def test_modulo(self):
        expr = pi.col("a") % pi.lit(3)
        assert expr.to_dsl() == "[a] % 3"

    def test_comparison_eq(self):
        expr = pi.col("age") == pi.lit(30)
        assert expr.to_dsl() == "[age] = 30"

    def test_comparison_ne(self):
        expr = pi.col("age") != pi.lit(30)
        assert expr.to_dsl() == "[age] != 30"

    def test_comparison_lt(self):
        expr = pi.col("age") < pi.lit(30)
        assert expr.to_dsl() == "[age] < 30"

    def test_comparison_ge(self):
        expr = pi.col("age") >= pi.lit(30)
        assert expr.to_dsl() == "[age] >= 30"

    def test_string_uppercase(self):
        expr = pi.col("name").str.to_uppercase()
        assert expr.to_dsl() == "uppercase([name])"

    def test_string_lowercase(self):
        expr = pi.col("name").str.to_lowercase()
        assert expr.to_dsl() == "lowercase([name])"

    def test_string_length(self):
        expr = pi.col("name").str.len_chars()
        assert expr.to_dsl() == "length([name])"

    def test_string_starts_with(self):
        expr = pi.col("name").str.starts_with(pi.lit("A"))
        assert expr.to_dsl() == 'starts_with([name], "A")'

    def test_string_contains(self):
        expr = pi.col("name").str.contains(pi.lit("test"))
        assert expr.to_dsl() == 'contains([name], "test")'

    def test_string_replace(self):
        expr = pi.col("name").str.replace(pi.lit("old"), pi.lit("new"))
        assert expr.to_dsl() == 'replace([name], "old", "new")'

    def test_string_trim(self):
        expr = pi.col("name").str.strip_chars()
        assert expr.to_dsl() == "trim([name])"

    def test_string_left_trim(self):
        expr = pi.col("name").str.strip_chars_start()
        assert expr.to_dsl() == "left_trim([name])"

    def test_string_pad_start(self):
        expr = pi.col("code").str.pad_start(5, "0")
        assert expr.to_dsl() == 'pad_left([code], 5, "0")'

    def test_string_reverse(self):
        expr = pi.col("name").str.reverse()
        assert expr.to_dsl() == "reverse([name])"

    def test_string_split(self):
        expr = pi.col("name").str.split(",")
        assert expr.to_dsl() == 'split([name], ",")'

    def test_dt_year(self):
        expr = pi.col("date").dt.year()
        assert expr.to_dsl() == "year([date])"

    def test_dt_month(self):
        expr = pi.col("date").dt.month()
        assert expr.to_dsl() == "month([date])"

    def test_dt_day(self):
        expr = pi.col("date").dt.day()
        assert expr.to_dsl() == "day([date])"

    def test_dt_weekday(self):
        expr = pi.col("date").dt.weekday()
        assert expr.to_dsl() == "weekday([date])"

    def test_dt_quarter(self):
        expr = pi.col("date").dt.quarter()
        assert expr.to_dsl() == "quarter([date])"

    def test_dt_month_end(self):
        expr = pi.col("date").dt.month_end()
        assert expr.to_dsl() == "end_of_month([date])"

    def test_dt_format(self):
        expr = pi.col("date").dt.to_string("%Y/%m/%d")
        assert expr.to_dsl() == 'format_date([date], "%Y/%m/%d")'

    def test_math_abs(self):
        expr = pi.col("val").abs()
        assert expr.to_dsl() == "abs([val])"

    def test_math_sqrt(self):
        expr = pi.col("val").sqrt()
        assert expr.to_dsl() == "sqrt([val])"

    def test_math_round(self):
        expr = pi.col("val").round(2)
        assert expr.to_dsl() == "round([val], 2)"

    def test_math_ceil(self):
        expr = pi.col("val").ceil()
        assert expr.to_dsl() == "ceil([val])"

    def test_math_floor(self):
        expr = pi.col("val").floor()
        assert expr.to_dsl() == "floor([val])"

    def test_math_pow(self):
        expr = pi.col("val").pow(pi.lit(3))
        assert expr.to_dsl() == "power([val], 3)"

    def test_math_log10(self):
        expr = pi.col("val").log(base=10)
        assert expr.to_dsl() == "log10([val])"

    def test_math_sign(self):
        expr = pi.col("val").sign()
        assert expr.to_dsl() == "sign([val])"

    def test_is_null(self):
        expr = pi.col("val").is_null()
        assert expr.to_dsl() == "is_empty([val])"

    def test_is_not_null(self):
        expr = pi.col("val").is_not_null()
        assert expr.to_dsl() == "is_not_empty([val])"

    def test_cast_int(self):
        expr = pi.col("val").cast(pl.Int64)
        assert expr.to_dsl() == "to_integer([val])"

    def test_cast_float(self):
        expr = pi.col("val").cast(pl.Float64)
        assert expr.to_dsl() == "to_float([val])"

    def test_cast_string(self):
        expr = pi.col("val").cast(pl.Utf8)
        assert expr.to_dsl() == "to_string([val])"

    def test_negation(self):
        expr = -pi.col("val")
        assert expr.to_dsl() == "-[val]"

    def test_logical_not(self):
        expr = pi.col("flag").not_()
        assert expr.to_dsl() == "_not([flag])"

    def test_coalesce(self):
        expr = pi.coalesce(pi.col("a"), pi.col("b"), pi.lit(0))
        assert expr.to_dsl() == "coalesce([a], [b], 0)"

    def test_concat_str(self):
        expr = pi.concat_str(pi.col("first"), pi.lit(" "), pi.col("last"))
        assert expr.to_dsl() == 'concat([first], " ", [last])'

    def test_max_horizontal(self):
        expr = pi.max_horizontal(pi.col("a"), pi.col("b"))
        assert expr.to_dsl() == "greatest([a], [b])"

    def test_min_horizontal(self):
        expr = pi.min_horizontal(pi.col("a"), pi.col("b"))
        assert expr.to_dsl() == "least([a], [b])"

    def test_when_then_otherwise(self):
        expr = (
            pi.when(pi.col("age") >= pi.lit(30))
            .then(pi.lit("Senior"))
            .otherwise(pi.lit("Junior"))
        )
        assert expr.to_dsl() == 'if [age] >= 30 then "Senior" else "Junior" endif'

    def test_chained_when(self):
        expr = (
            pi.when(pi.col("score") >= pi.lit(90))
            .then(pi.lit("A"))
            .when(pi.col("score") >= pi.lit(80))
            .then(pi.lit("B"))
            .otherwise(pi.lit("C"))
        )
        assert expr.to_dsl() == 'if [score] >= 90 then "A" elseif [score] >= 80 then "B" else "C" endif'

    def test_operator_precedence(self):
        # [a] + [b] * 2 should not add parens (mul is higher precedence)
        expr = pi.col("a") + pi.col("b") * pi.lit(2)
        assert expr.to_dsl() == "[a] + [b] * 2"

    def test_operator_precedence_parens(self):
        # ([a] + [b]) * 2 — the add is lower precedence, needs parens
        inner = pi.col("a") + pi.col("b")
        expr = inner * pi.lit(2)
        assert expr.to_dsl() == "([a] + [b]) * 2"

    def test_complex_expression(self):
        expr = pi.col("name").str.to_uppercase() + pi.lit(" (") + pi.col("city") + pi.lit(")")
        dsl = expr.to_dsl()
        assert "uppercase([name])" in dsl
        assert "[city]" in dsl

    def test_string_concat_with_lit(self):
        # When + involves a string literal, it becomes concat
        expr = pi.col("first") + pi.lit(" ") + pi.col("last")
        dsl = expr.to_dsl()
        assert "concat" in dsl


# ──────────────────────────────────────────────
# Round-trip tests: interceptor → DSL → Polars → execute
# ──────────────────────────────────────────────

class TestRoundTrip:
    """Test that interceptor-generated DSL actually works with simple_function_to_expr."""

    def test_arithmetic_roundtrip(self):
        df = pl.DataFrame({"a": [10, 20], "b": [3, 5]})
        expr = pi.col("a") + pi.col("b")
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": [13, 25]})
        assert_frame_equal(result, expected)

    def test_multiply_roundtrip(self):
        df = pl.DataFrame({"price": [10.0, 20.0], "qty": [2, 3]})
        expr = pi.col("price") * pi.col("qty")
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": [20.0, 60.0]})
        assert_frame_equal(result, expected)

    def test_uppercase_roundtrip(self):
        df = pl.DataFrame({"name": ["alice", "bob"]})
        expr = pi.col("name").str.to_uppercase()
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": ["ALICE", "BOB"]})
        assert_frame_equal(result, expected)

    def test_lowercase_roundtrip(self):
        df = pl.DataFrame({"name": ["ALICE", "BOB"]})
        expr = pi.col("name").str.to_lowercase()
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": ["alice", "bob"]})
        assert_frame_equal(result, expected)

    def test_length_roundtrip(self):
        df = pl.DataFrame({"name": ["alice", "bob"]})
        expr = pi.col("name").str.len_chars()
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": [5, 3]}).cast({"result": pl.UInt32})
        assert_frame_equal(result, expected)

    def test_abs_roundtrip(self):
        df = pl.DataFrame({"val": [-5, 3, -1]})
        expr = pi.col("val").abs()
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": [5, 3, 1]})
        assert_frame_equal(result, expected)

    def test_comparison_roundtrip(self):
        df = pl.DataFrame({"age": [25, 30, 35]})
        expr = pi.col("age") >= pi.lit(30)
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": [False, True, True]})
        assert_frame_equal(result, expected)

    def test_when_then_roundtrip(self):
        df = pl.DataFrame({"age": [25, 35]})
        expr = (
            pi.when(pi.col("age") >= pi.lit(30))
            .then(pi.lit("Senior"))
            .otherwise(pi.lit("Junior"))
        )
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": ["Junior", "Senior"]})
        assert_frame_equal(result, expected)

    def test_coalesce_roundtrip(self):
        df = pl.DataFrame({"a": [None, 2, None], "b": [10, None, 30]})
        expr = pi.coalesce(pi.col("a"), pi.col("b"))
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": [10, 2, 30]})
        assert_frame_equal(result, expected)

    def test_ceil_roundtrip(self):
        df = pl.DataFrame({"val": [1.2, 2.7, 3.0]})
        expr = pi.col("val").ceil()
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": [2.0, 3.0, 3.0]})
        assert_frame_equal(result, expected)

    def test_floor_roundtrip(self):
        df = pl.DataFrame({"val": [1.9, 2.1, 3.0]})
        expr = pi.col("val").floor()
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": [1.0, 2.0, 3.0]})
        assert_frame_equal(result, expected)

    def test_dt_year_roundtrip(self):
        from datetime import date
        df = pl.DataFrame({"d": [date(2023, 6, 15), date(2024, 1, 1)]})
        expr = pi.col("d").dt.year()
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": [2023, 2024]}).cast({"result": pl.Int32})
        assert_frame_equal(result, expected)

    def test_concat_str_roundtrip(self):
        df = pl.DataFrame({"first": ["Alice", "Bob"], "last": ["Smith", "Jones"]})
        expr = pi.concat_str(pi.col("first"), pi.lit(" "), pi.col("last"))
        result = df.select(expr.to_polars_expr().alias("result"))
        expected = pl.DataFrame({"result": ["Alice Smith", "Bob Jones"]})
        assert_frame_equal(result, expected)

    def test_complex_roundtrip(self):
        """A more complex expression combining multiple operations."""
        df = pl.DataFrame({"price": [100.0, 200.0], "tax_rate": [0.1, 0.2]})
        expr = pi.col("price") * (pi.lit(1) + pi.col("tax_rate"))
        result = df.select(expr.to_polars_expr().alias("total"))
        expected = pl.DataFrame({"total": [110.0, 240.0]})
        assert_frame_equal(result, expected)
