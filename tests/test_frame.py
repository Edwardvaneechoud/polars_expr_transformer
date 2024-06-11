import pytest
from polars_expr_transformer import DataFrame, LazyFrame

def test_dataframe_apply_expression():
    df = DataFrame({'names': ['Alice', 'Bob'], 'surnames': ['Smith', 'Jones']})
    result = df.apply_expression('concat([names], " ", [surnames])', 'full_name')
    expected = DataFrame({'names': ['Alice', 'Bob'], 'surnames': ['Smith', 'Jones'], 'full_name': ['Alice Smith', 'Bob Jones']})
    assert result.frame_equal(expected)

def test_lazyframe_apply_expression():
    df = LazyFrame({'names': ['Alice', 'Bob'], 'surnames': ['Smith', 'Jones']})
    result = df.apply_expression('concat([names], " ", [surnames])', 'full_name').collect()
    expected = DataFrame({'names': ['Alice', 'Bob'], 'surnames': ['Smith', 'Jones'], 'full_name': ['Alice Smith', 'Bob Jones']})
    assert result.frame_equal(expected)

if __name__ == '__main__':
    pytest.main()
