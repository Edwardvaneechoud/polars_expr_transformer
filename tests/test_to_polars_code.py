from polars_expr_transformer import to_polars_code, simple_function_to_expr
import polars as pl
from polars.testing import assert_frame_equal
from pytest import fixture
import pytest


def eval_pl_expr(expr_func_str: str) -> pl.Expr:
    """Evaluates the polars expressions string and returns the expr"""
    try:
        expr = eval(to_polars_code(expr_func_str), {"pl": pl})
    except Exception as e:
        raise Exception(f"Could not evaluate the polars expression:\n\n{e}")
    return expr


@fixture
def main_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            # String columns
            "col_a": ["apple", "banana", "cherry"],
            "my_column": ["x", "y", "z"],
            "name": [" Alice ", "bob", "Charlie"],
            "first": ["John", "Jane", "Alice"],
            "last": ["Doe", "Smith", "Johnson"],
            "str_col": ["1", "2.5", "3"],
            "a": ["a", "b", "c"],
            "b": ["x", "y", "z"],
            # Numeric columns
            "val": [1.0, -2.5, 9.0],
            "price": [10.555, 20.123, 30.999],
            "age": [25, 35, 45],
            "score": [95, 82, 70],
            "num": [100, 200, 300],
            # Date columns (as strings to test to_date conversion)
            "date": ["2023-01-01", "2023-06-15", "2023-12-31"],
        }
    )


def validate_func_expr_str(df: pl.DataFrame, func_expr_str: str) -> None:
    func_df = df.select(simple_function_to_expr(func_expr_str))
    pl_expr = eval_pl_expr(func_expr_str)
    new_df = df.select(pl_expr)
    assert_frame_equal(func_df, new_df)


class TestColumnAndLiterals:
    def test_column_plus_string_literal(self, main_df):
        expr_str = "[col_a] + 'test'"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("col_a") + pl.lit("test")'

    def test_simple_string_literal(self, main_df):
        expr_str = "'hello world'"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.lit("hello world")'

    def test_numeric_literal(self, main_df):
        expr_str = "42"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == "pl.lit(42)"

    def test_boolean_literal(self, main_df):
        expr_str = "true"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == "pl.lit(True)"

    def test_column_reference(self, main_df):
        expr_str = "[my_column]"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("my_column")'


class TestArithmeticOperators:
    def test_addition_two_columns(self, main_df):
        expr_str = "[val] + [num]"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val") + pl.col("num")'

    def test_subtraction(self, main_df):
        expr_str = "[val] - [num]"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val") - pl.col("num")'

    def test_multiplication(self, main_df):
        expr_str = "[val] * [num]"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val") * pl.col("num")'

    def test_division(self, main_df):
        expr_str = "[val] / [num]"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val") / pl.col("num")'

    def test_modulo(self, main_df):
        expr_str = "[val] % [num]"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val") % pl.col("num")'

    def test_mixed_precedence(self, main_df):
        expr_str = "[val] * [num] + [age]"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == '(pl.col("val") * pl.col("num")) + pl.col("age")'

    def test_literal_multiplication(self, main_df):
        expr_str = "2 * 2"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == "pl.lit(2) * pl.lit(2)"

    def test_column_minus_number(self, main_df):
        expr_str = "[val] - 2"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val") - pl.lit(2)'


class TestComparisonOperators:
    def test_greater_than(self, main_df):
        expr_str = "[age] > 10"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("age") > pl.lit(10)'

    def test_less_than(self, main_df):
        expr_str = "[age] < 10"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("age") < pl.lit(10)'

    def test_greater_equal(self, main_df):
        expr_str = "[age] >= 10"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("age") >= pl.lit(10)'

    def test_less_equal(self, main_df):
        expr_str = "[age] <= 10"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("age") <= pl.lit(10)'

    def test_equal(self, main_df):
        expr_str = "[age] = 10"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("age") == pl.lit(10)'

    def test_not_equal(self, main_df):
        expr_str = "[val] != [num]"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val") != pl.col("num")'


class TestLogicalOperators:
    def test_and(self, main_df):
        expr_str = "[age] > 1 and [score] < 100"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == '(pl.col("age") > pl.lit(1)) & (pl.col("score") < pl.lit(100))'

    def test_or(self, main_df):
        expr_str = "[age] > 1 or [score] < 100"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == '(pl.col("age") > pl.lit(1)) | (pl.col("score") < pl.lit(100))'


class TestStringFunctions:
    def test_uppercase(self, main_df):
        expr_str = "uppercase([name])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("name").str.to_uppercase()'

    def test_lowercase(self, main_df):
        expr_str = "lowercase([name])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("name").str.to_lowercase()'

    def test_length(self, main_df):
        expr_str = "length([name])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("name").str.len_chars()'

    def test_trim(self, main_df):
        expr_str = "trim([name])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("name").str.strip_chars()'

    def test_concat(self, main_df):
        expr_str = 'concat([first], " ", [last])'
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.concat_str([pl.col("first"), pl.lit(" "), pl.col("last")])'

    def test_starts_with(self, main_df):
        expr_str = 'starts_with([name], "A")'
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("name").str.starts_with(pl.lit("A"))'

    def test_ends_with(self, main_df):
        expr_str = 'ends_with([name], "y")'
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("name").str.ends_with(pl.lit("y"))'

    def test_contains(self, main_df):
        expr_str = 'contains([name], "ob")'
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("name").str.contains(pl.lit("ob"))'

    def test_replace(self, main_df):
        expr_str = 'replace([name], "bob", "X")'
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("name").str.replace_many(pl.lit("bob"), pl.lit("X"))'

    def test_left(self, main_df):
        expr_str = "left([name], 3)"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("name").str.slice(0, pl.lit(3))'

    def test_reverse(self, main_df):
        expr_str = "reverse([name])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("name").str.reverse()'


class TestMathFunctions:
    def test_abs(self, main_df):
        expr_str = "abs([val])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val").abs()'

    def test_round(self, main_df):
        expr_str = "round([price], 2)"
        result = to_polars_code(expr_str)
        assert result == 'pl.col("price").round(pl.lit(2))'
        with pytest.raises(Exception, match="cannot be interpreted as an integer"):
            validate_func_expr_str(main_df, expr_str)

    def test_ceil(self, main_df):
        expr_str = "ceil([val])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val").ceil()'

    def test_floor(self, main_df):
        expr_str = "floor([val])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val").floor()'

    def test_sqrt(self, main_df):
        expr_str = "sqrt([val])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val").sqrt()'

    def test_power(self, main_df):
        expr_str = "power([val], 2)"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val").pow(pl.lit(2))'


class TestDateFunctions:
    def test_year_from_date(self, main_df):
        expr_str = "year(to_date([date]))"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("date").str.to_date().dt.year()'

    def test_month_from_date(self, main_df):
        expr_str = "month(to_date([date]))"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("date").str.to_date().dt.month()'

    def test_day_from_date(self, main_df):
        expr_str = "day(to_date([date]))"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("date").str.to_date().dt.day()'

    def test_add_days(self, main_df):
        expr_str = "add_days(to_date([date]), 5)"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("date").str.to_date() + pl.duration(days=pl.lit(5))'


class TestConditionals:
    def test_simple_if_else(self, main_df):
        expr_str = "if [age] > 30 then 'Senior' else 'Junior' endif"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert (
            result
            == 'pl.when(pl.col("age") > pl.lit(30)).then(pl.lit("Senior")).otherwise(pl.lit("Junior"))'
        )

    def test_if_elseif_else(self, main_df):
        expr_str = (
            "if [score] >= 90 then 'A' elseif [score] >= 80 then 'B' else 'C' endif"
        )
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == (
            'pl.when(pl.col("score") >= pl.lit(90)).then(pl.lit("A"))'
            '.when(pl.col("score") >= pl.lit(80)).then(pl.lit("B"))'
            '.otherwise(pl.lit("C"))'
        )


class TestNegation:
    def test_negative_column(self, main_df):
        expr_str = "-[val]"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("val").neg()'


class TestTypeConversions:
    def test_to_string(self, main_df):
        expr_str = "to_string([num])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("num").cast(pl.Utf8)'

    def test_to_integer(self, main_df):
        expr_str = "to_integer([str_col])"
        result = to_polars_code(expr_str)
        assert result == 'pl.col("str_col").cast(pl.Int64)'
        with pytest.raises(pl.exceptions.InvalidOperationError):
            validate_func_expr_str(main_df, expr_str)

    def test_to_float(self, main_df):
        expr_str = "to_float([str_col])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("str_col").cast(pl.Float64)'


class TestLogicFunctions:
    def test_coalesce(self, main_df):
        expr_str = "coalesce([first], [last], 'default')"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert (
            result
            == 'pl.coalesce([pl.col("first"), pl.col("last"), pl.lit("default")])'
        )

    def test_is_empty(self, main_df):
        expr_str = "is_empty([first])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("first").is_null()'

    def test_is_not_empty(self, main_df):
        expr_str = "is_not_empty([first])"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("first").is_not_null()'

    def test_between(self, main_df):
        expr_str = "between([age], 1, 100)"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert result == 'pl.col("age").is_between(pl.lit(1), pl.lit(100))'


class TestCombinedExpressions:
    def test_string_concat_with_column(self, main_df):
        expr_str = '[first] + " loves " + [last]'
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert 'pl.col("first")' in result
        assert 'pl.lit(" loves ")' in result
        assert 'pl.col("last")' in result
        assert "+" in result

    def test_nested_function_calls(self, main_df):
        expr_str = "uppercase(concat([first], [last]))"
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)
        assert (
            result
            == 'pl.concat_str([pl.col("first"), pl.col("last")]).str.to_uppercase()'
        )

    def test_complex_condition_with_concat(self, main_df):
        expr_str = (
            'concat("result:", if "li" in [name] then "found" else "not found" endif)'
        )
        validate_func_expr_str(main_df, expr_str)
        result = to_polars_code(expr_str)

        assert "pl.concat_str" in result
        assert "pl.when" in result
        assert ".otherwise" in result
