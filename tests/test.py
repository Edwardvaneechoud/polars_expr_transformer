import pytest
from polars_expr_transformer.process.polars_expr_transformer import preprocess, simple_function_to_expr
import polars as pl

def test_simple_constant_expression():
    df = pl.from_dicts([{'a': 'edward', 'b': 'courtney'}, {'a': 'courtney', 'b': 'edward'}])
    result = df.select(simple_function_to_expr("'hallo world'"))
    expected = pl.DataFrame({'literal': ['hallo world']})
    assert result.equals(expected)

def test_combining_columns_expression():
    df = pl.from_dicts([{'a': 'edward', 'b': 'courtney'}, {'a': 'courtney', 'b': 'edward'}])
    result = df.select(simple_function_to_expr('[a] + " loves " + [b]').alias('literal'))
    expected = pl.DataFrame({'literal': ['edward loves courtney', 'courtney loves edward']})
    assert result.equals(expected)

def test_condition_expression():
    df = pl.from_dicts([{'a': 'edward', 'b': 'courtney'}, {'a': 'courtney', 'b': 'edward'}])
    result = df.select(simple_function_to_expr('"a" in [a]').alias('literal'))
    expected = pl.DataFrame({'literal': [True, False]})
    assert result.equals(expected)

def test_complex_conditional_expression():
    df = pl.from_dicts([{'a': 'edward', 'b': 'courtney'}, {'a': 'courtney', 'b': 'edward'}])
    result = df.select(simple_function_to_expr('concat("result:", if "a" in [a] then "A has been found" else "not found" endif)'))
    expected = pl.DataFrame({'literal': ['result:A has been found', 'result:not found']})
    assert result.equals(expected)

if __name__ == '__main__':
    pytest.main()
