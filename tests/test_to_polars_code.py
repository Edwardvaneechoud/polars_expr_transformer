from polars_expr_transformer import to_polars_code


class TestColumnAndLiterals:
    def test_column_plus_string_literal(self):
        result = to_polars_code("[col_a] + 'test'")
        assert result == 'pl.col("col_a") + pl.lit("test")'

    def test_simple_string_literal(self):
        result = to_polars_code("'hello world'")
        assert result == 'pl.lit("hello world")'

    def test_numeric_literal(self):
        result = to_polars_code("42")
        assert result == "pl.lit(42)"

    def test_boolean_literal(self):
        result = to_polars_code("true")
        assert result == "pl.lit(True)"

    def test_column_reference(self):
        result = to_polars_code("[my_column]")
        assert result == 'pl.col("my_column")'


class TestArithmeticOperators:
    def test_addition_two_columns(self):
        result = to_polars_code("[a] + [b]")
        assert result == 'pl.col("a") + pl.col("b")'

    def test_subtraction(self):
        result = to_polars_code("[a] - [b]")
        assert result == 'pl.col("a") - pl.col("b")'

    def test_multiplication(self):
        result = to_polars_code("[a] * [b]")
        assert result == 'pl.col("a") * pl.col("b")'

    def test_division(self):
        result = to_polars_code("[a] / [b]")
        assert result == 'pl.col("a") / pl.col("b")'

    def test_modulo(self):
        result = to_polars_code("[a] % [b]")
        assert result == 'pl.col("a") % pl.col("b")'

    def test_mixed_precedence(self):
        """[a] * [b] + [c] should show multiplication nested inside addition."""
        result = to_polars_code("[a] * [b] + [c]")
        assert result == '(pl.col("a") * pl.col("b")) + pl.col("c")'

    def test_literal_multiplication(self):
        result = to_polars_code("2 * 2")
        assert result == "pl.lit(2) * pl.lit(2)"

    def test_column_minus_number(self):
        result = to_polars_code("[a] - 2")
        assert result == 'pl.col("a") - pl.lit(2)'


class TestComparisonOperators:
    def test_greater_than(self):
        result = to_polars_code("[a] > 10")
        assert result == 'pl.col("a") > pl.lit(10)'

    def test_less_than(self):
        result = to_polars_code("[a] < 10")
        assert result == 'pl.col("a") < pl.lit(10)'

    def test_greater_equal(self):
        result = to_polars_code("[a] >= 10")
        assert result == 'pl.col("a") >= pl.lit(10)'

    def test_less_equal(self):
        result = to_polars_code("[a] <= 10")
        assert result == 'pl.col("a") <= pl.lit(10)'

    def test_equal(self):
        result = to_polars_code("[a] = 10")
        assert result == 'pl.col("a") == pl.lit(10)'

    def test_not_equal(self):
        result = to_polars_code("[a] != [b]")
        assert result == 'pl.col("a") != pl.col("b")'


class TestLogicalOperators:
    def test_and(self):
        result = to_polars_code("[a] > 1 and [b] < 5")
        assert result == '(pl.col("a") > pl.lit(1)) & (pl.col("b") < pl.lit(5))'

    def test_or(self):
        result = to_polars_code("[a] > 1 or [b] < 5")
        assert result == '(pl.col("a") > pl.lit(1)) | (pl.col("b") < pl.lit(5))'


class TestStringFunctions:
    def test_uppercase(self):
        result = to_polars_code("uppercase([name])")
        assert result == 'pl.col("name").str.to_uppercase()'

    def test_lowercase(self):
        result = to_polars_code("lowercase([name])")
        assert result == 'pl.col("name").str.to_lowercase()'

    def test_length(self):
        result = to_polars_code("length([name])")
        assert result == 'pl.col("name").str.len_chars()'

    def test_trim(self):
        result = to_polars_code("trim([name])")
        assert result == 'pl.col("name").str.strip_chars()'

    def test_concat(self):
        result = to_polars_code('concat([first], " ", [last])')
        assert result == 'pl.concat_str([pl.col("first"), pl.lit(" "), pl.col("last")])'

    def test_starts_with(self):
        result = to_polars_code('starts_with([name], "A")')
        assert result == 'pl.col("name").str.starts_with(pl.lit("A"))'

    def test_ends_with(self):
        result = to_polars_code('ends_with([name], "z")')
        assert result == 'pl.col("name").str.ends_with(pl.lit("z"))'

    def test_contains(self):
        result = to_polars_code('contains([name], "test")')
        assert result == 'pl.col("name").str.contains(pl.lit("test"))'

    def test_replace(self):
        result = to_polars_code('replace([name], "old", "new")')
        assert result == 'pl.col("name").str.replace_many(pl.lit("old"), pl.lit("new"))'

    def test_left(self):
        result = to_polars_code("left([name], 3)")
        assert result == 'pl.col("name").str.slice(0, pl.lit(3))'

    def test_reverse(self):
        result = to_polars_code("reverse([name])")
        assert result == 'pl.col("name").str.reverse()'


class TestMathFunctions:
    def test_abs(self):
        result = to_polars_code("abs([val])")
        assert result == 'pl.col("val").abs()'

    def test_round(self):
        result = to_polars_code("round([price], 2)")
        assert result == 'pl.col("price").round(pl.lit(2))'

    def test_ceil(self):
        result = to_polars_code("ceil([val])")
        assert result == 'pl.col("val").ceil()'

    def test_floor(self):
        result = to_polars_code("floor([val])")
        assert result == 'pl.col("val").floor()'

    def test_sqrt(self):
        result = to_polars_code("sqrt([val])")
        assert result == 'pl.col("val").sqrt()'

    def test_power(self):
        result = to_polars_code("power([val], 2)")
        assert result == 'pl.col("val").pow(pl.lit(2))'


class TestDateFunctions:
    def test_year_from_date(self):
        result = to_polars_code("year(to_date([date]))")
        assert result == 'pl.col("date").str.to_date().dt.year()'

    def test_month_from_date(self):
        result = to_polars_code("month(to_date([date]))")
        assert result == 'pl.col("date").str.to_date().dt.month()'

    def test_day_from_date(self):
        result = to_polars_code("day(to_date([date]))")
        assert result == 'pl.col("date").str.to_date().dt.day()'

    def test_add_days(self):
        result = to_polars_code("add_days(to_date([date]), 5)")
        assert result == 'pl.col("date").str.to_date() + pl.duration(days=pl.lit(5))'


class TestConditionals:
    def test_simple_if_else(self):
        result = to_polars_code("if [age] > 30 then 'Senior' else 'Junior' endif")
        assert result == 'pl.when(pl.col("age") > pl.lit(30)).then(pl.lit("Senior")).otherwise(pl.lit("Junior"))'

    def test_if_elseif_else(self):
        result = to_polars_code(
            "if [score] >= 90 then 'A' elseif [score] >= 80 then 'B' else 'C' endif"
        )
        assert result == (
            'pl.when(pl.col("score") >= pl.lit(90)).then(pl.lit("A"))'
            '.when(pl.col("score") >= pl.lit(80)).then(pl.lit("B"))'
            '.otherwise(pl.lit("C"))'
        )


class TestNegation:
    def test_negative_column(self):
        result = to_polars_code("-[a]")
        assert result == 'pl.col("a").neg()'


class TestTypeConversions:
    def test_to_string(self):
        result = to_polars_code("to_string([num])")
        assert result == 'pl.col("num").cast(pl.Utf8)'

    def test_to_integer(self):
        result = to_polars_code("to_integer([str_col])")
        assert result == 'pl.col("str_col").cast(pl.Int64)'

    def test_to_float(self):
        result = to_polars_code("to_float([str_col])")
        assert result == 'pl.col("str_col").cast(pl.Float64)'


class TestLogicFunctions:
    def test_coalesce(self):
        result = to_polars_code("coalesce([a], [b], 'default')")
        assert result == 'pl.coalesce([pl.col("a"), pl.col("b"), pl.lit("default")])'

    def test_is_empty(self):
        result = to_polars_code("is_empty([a])")
        assert result == 'pl.col("a").is_null()'

    def test_is_not_empty(self):
        result = to_polars_code("is_not_empty([a])")
        assert result == 'pl.col("a").is_not_null()'

    def test_between(self):
        result = to_polars_code("between([val], 1, 10)")
        assert result == 'pl.col("val").is_between(pl.lit(1), pl.lit(10))'


class TestCombinedExpressions:
    def test_string_concat_with_column(self):
        result = to_polars_code('[a] + " loves " + [b]')
        # The + operator chains: (pl.col("a") + pl.lit(" loves ")) + pl.col("b")
        assert 'pl.col("a")' in result
        assert 'pl.lit(" loves ")' in result
        assert 'pl.col("b")' in result
        assert "+" in result

    def test_nested_function_calls(self):
        result = to_polars_code("uppercase(concat([first], [last]))")
        assert result == 'pl.concat_str([pl.col("first"), pl.col("last")]).str.to_uppercase()'

    def test_complex_condition_with_concat(self):
        result = to_polars_code(
            'concat("result:", if "a" in [a] then "found" else "not found" endif)'
        )
        assert "pl.concat_str" in result
        assert "pl.when" in result
        assert ".otherwise" in result
