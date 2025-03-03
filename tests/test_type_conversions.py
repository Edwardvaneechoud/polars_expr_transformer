import pytest
from polars_expr_transformer.process.polars_expr_transformer import preprocess, simple_function_to_expr
import polars as pl
from polars.testing import assert_frame_equal


def test_to_string():
    df = pl.from_dicts([{'a': 1}, {'a': 2}, {'a': 3}])
    result = df.select(simple_function_to_expr("to_string([a])"))
    expected = pl.DataFrame({'a': ['1', '2', '3']})
    assert result.equals(expected)


def test_to_integer():
    df = pl.DataFrame({'a': [1.1, 2.2, 3.3]})
    result = df.select(simple_function_to_expr("to_integer([a])"))
    expected = pl.DataFrame({'a': [1, 2, 3]})
    assert result.equals(expected)


def test_to_number():
    df = pl.DataFrame({'a': ["1.1", "2.2", "3.3"]})
    result = df.select(simple_function_to_expr("to_number([a])"))
    expected = pl.DataFrame({'a': [1.1, 2.2, 3.3]})
    assert result.equals(expected)


def test_to_float():
    df = pl.DataFrame({'a': ["1.1", "2.2", "3.3"]})
    result = df.select(simple_function_to_expr("to_float([a])"))
    expected = pl.DataFrame({'a': [1.1, 2.2, 3.3]})
    assert result.equals(expected)


def test_float_to_integer():
    df = pl.DataFrame({'a': [1.1, 2.2, 3.3]})
    result = df.select(simple_function_to_expr("to_integer(to_float([a]))"))
    expected = pl.DataFrame({'a': [1, 2, 3]})
    assert result.equals(expected)


def test_error_string_to_integer():
    df = pl.DataFrame({'a': ["1.1", "2.2", "3.3"]})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.select(simple_function_to_expr("to_integer([a])"))


def test_error_string_to_float():
    df = pl.DataFrame({'a': ["a", "2.2", "3.3"]})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.select(simple_function_to_expr("to_float([a])"))


def test_integer_col_to_boolean():
    df = pl.DataFrame({'a': [1, 0, 1]})
    result = df.select(simple_function_to_expr("to_boolean([a])"))
    expected = pl.DataFrame({'literal': [True, False, True]})
    assert result.equals(expected)


def test_string_col_to_boolean():
    df = pl.DataFrame({'a': ["true", "false", "true"]})
    result = df.select(simple_function_to_expr("to_boolean([a])"))
    expected = pl.DataFrame({'literal': [True, False, True]})
    assert result.equals(expected)


def test_string_to_boolean():
    df = pl.DataFrame({'a': ["True", "False", "True"]})
    result = df.select(simple_function_to_expr("to_boolean('True')"))
    expected = pl.DataFrame({'literal': [True]})
    assert result.equals(expected)


def test_integer_to_boolean():
    df = pl.DataFrame({'a': [1]})
    result = df.select(simple_function_to_expr("to_boolean(1)"))
    expected = pl.DataFrame({'literal': [True]})
    assert result.equals(expected)


def test_float_to_boolean():
    df = pl.DataFrame({'a': [1.0]})
    result = df.select(simple_function_to_expr("to_boolean(1.0)"))
    expected = pl.DataFrame({'literal': [True]})
    assert result.equals(expected)


def test_float_col_to_boolean():
    df = pl.DataFrame({'a': [1.0, 0.0, 1.0]})
    result = df.select(simple_function_to_expr("to_boolean([a])"))
    expected = pl.DataFrame({'literal': [True, False, True]})
    assert result.equals(expected)


def test_to_date():
    df = pl.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03']})
    df_with_dates = df.select(pl.col('date').str.to_date())
    func_str = 'to_date(to_string(year([date])) + "-"+ to_string(month([date])) + "-" + to_string(day([date])))'
    result = df_with_dates.select(simple_function_to_expr(func_str))
    expected = df_with_dates
    assert result.equals(expected)


def test_date_from_string():
    df = pl.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03']})
    result = df.select(simple_function_to_expr('to_date([date])'))
    expected = df.select(pl.col('date').str.to_date())
    assert result.equals(expected)
