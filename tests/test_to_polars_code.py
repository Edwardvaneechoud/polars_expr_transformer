from polars_expr_transformer import to_polars_code, simple_function_to_expr, PolarsCodeGenError
from polars_expr_transformer.process.polars_expr_transformer import _validate_polars_code
from polars_expr_transformer.process.models import Func, Classifier
import polars as pl
from polars.testing import assert_frame_equal
from pytest import fixture
import pytest
import warnings
import datetime


def eval_pl_expr(expr_func_str: str) -> pl.Expr:
    """Evaluates the polars expressions string and returns the expr"""
    try:
        expr = eval(to_polars_code(expr_func_str), {"pl": pl, "datetime": datetime})
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
        assert result == 'pl.col("price").round(2)'
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


class TestCodeGenFixes:
    """Tests verifying that previously broken code gen scenarios now produce valid code."""

    def test_round_generates_plain_int(self, main_df):
        """round() should use a plain int, not pl.lit()."""
        result = to_polars_code("round([price], 2)")
        assert ".round(2)" in result
        assert "pl.lit(2)" not in result
        validate_func_expr_str(main_df, result.replace('pl.col("price")', "[price]").replace(".round(2)", ""))
        # Eval the generated code directly
        expr = eval(result, {"pl": pl})
        main_df.select(expr)

    def test_to_decimal_generates_plain_int(self, main_df):
        """to_decimal() should use a plain int for .round(), not pl.lit()."""
        result = to_polars_code("to_decimal([val], 2)")
        assert ".round(2)" in result
        assert ".round(pl.lit(2))" not in result
        expr = eval(result, {"pl": pl})
        main_df.select(expr)

    def test_repeat_generates_plain_int(self):
        """repeat() should use a plain int for list multiplication, not pl.lit()."""
        result = to_polars_code("repeat([name], 3)")
        assert "* 3" in result
        assert "* pl.lit(3)" not in result
        expr = eval(result, {"pl": pl})
        assert isinstance(expr, pl.Expr)

    def test_now_generates_valid_datetime(self):
        """now() should use datetime.datetime.now(), not bare datetime.now()."""
        result = to_polars_code("now()")
        assert "datetime.datetime.now()" in result
        expr = eval(result, {"pl": pl, "datetime": datetime})
        assert isinstance(expr, pl.Expr)

    def test_today_generates_valid_datetime(self):
        """today() should use datetime.datetime.today(), not bare datetime.today()."""
        result = to_polars_code("today()")
        assert "datetime.datetime.today()" in result
        expr = eval(result, {"pl": pl, "datetime": datetime})
        assert isinstance(expr, pl.Expr)


class TestValidation:
    """Tests for eval-based validation in to_polars_code()."""

    def test_validate_polars_code_raises_on_bad_code(self):
        """_validate_polars_code should raise PolarsCodeGenError for invalid code."""
        with pytest.raises(PolarsCodeGenError) as exc_info:
            _validate_polars_code("test_expr", "nonexistent_func(pl.col('x'))")
        err = exc_info.value
        assert err.expression == "test_expr"
        assert err.generated_code == "nonexistent_func(pl.col('x'))"
        assert isinstance(err.eval_error, NameError)

    def test_validate_polars_code_passes_on_valid_code(self):
        """_validate_polars_code should not raise for valid code."""
        _validate_polars_code("test_expr", 'pl.col("x").str.to_uppercase()')

    def test_validate_polars_code_datetime_in_scope(self):
        """_validate_polars_code should have datetime in scope."""
        _validate_polars_code("test_expr", "pl.lit(datetime.datetime.now())")

    def test_valid_expressions_pass_validation(self, main_df):
        """Normal expressions should pass validation without error."""
        result = to_polars_code("uppercase([name])", validate=True)
        assert result == 'pl.col("name").str.to_uppercase()'

    def test_validate_false_skips_validation(self):
        """validate=False should return code without running eval."""
        result = to_polars_code("round([price], 2)", validate=False)
        assert isinstance(result, str)
        assert ".round(2)" in result

    def test_string_similarity_raises_polars_code_gen_error(self):
        """string_similarity generates code referencing pds which is not in eval scope."""
        with pytest.raises(PolarsCodeGenError) as exc_info:
            to_polars_code('string_similarity([names], [other], "levenshtein")')
        err = exc_info.value
        assert "string_similarity" in err.generated_code
        assert isinstance(err.eval_error, NameError)

    def test_polars_code_gen_error_attributes(self):
        """PolarsCodeGenError should expose expression, generated_code, and eval_error."""
        bad_code = "definitely_not_valid()"
        with pytest.raises(PolarsCodeGenError) as exc_info:
            _validate_polars_code("my_expr", bad_code)
        err = exc_info.value
        assert err.expression == "my_expr"
        assert err.generated_code == bad_code
        assert isinstance(err.eval_error, Exception)
        assert "my_expr" in str(err)


class TestUnknownFunctionWarning:
    """Tests for warnings on unknown function fallback in Func.to_polars_code()."""

    def test_unknown_function_emits_warning(self):
        """The fallback code path in Func.to_polars_code() should emit a warning."""
        # Build a Func node with a function name not in FUNCTION_CODE_GEN
        func_ref = Classifier("unknown_test_func")
        arg = Func(func_ref=Classifier("pl.col"), args=[Classifier('"x"')])
        func = Func(func_ref=func_ref, args=[arg])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func.to_polars_code()
            warning_messages = [str(warning.message) for warning in w]
            assert any("unknown_test_func" in msg for msg in warning_messages)
        assert "unknown_test_func" in result
