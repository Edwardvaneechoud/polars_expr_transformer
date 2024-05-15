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


def test_nested_if_expression():
    df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [1, 2, 3], 'names': ['ham', 'spam', 'eggs'],
                       'subnames': ['bread', 'sandwich', 'breakfast']})

    func_str = 'if ((1222*2/[a])> 1222) then true else false endif'
    result = df.select(simple_function_to_expr(func_str))
    expected = pl.DataFrame({'literal': [True, False, False]})
    assert result.equals(expected)


def test_date_from_string():
    df = pl.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03']})
    result = df.select(simple_function_to_expr('to_date([date])'))
    expected = df.select(pl.col('date').str.to_date())
    assert result.equals(expected)


def test_get_year_from_date():
    df = pl.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03']})
    result = df.select(simple_function_to_expr('year(to_date([date]))'))
    expected = df.select(pl.col('date').str.to_date().dt.year())
    assert result.equals(expected)


def test_get_month_from_date():
    df = pl.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03']})
    result = df.select(simple_function_to_expr('month(to_date([date]))'))
    expected = df.select(pl.col('date').str.to_date().dt.month())
    assert result.equals(expected)


def test_get_day_from_date():
    df = pl.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03']})
    result = df.select(simple_function_to_expr('day(to_date([date]))'))
    expected = df.select(pl.col('date').str.to_date().dt.day())
    assert result.equals(expected)


def test_add_years():
    df = pl.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03']})
    result = df.select(simple_function_to_expr('add_years(to_date([date]), 1)'))
    expected = pl.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03']}).select(pl.col('date').str.to_date())
    assert result.equals(expected)


def test_add_days():
    df = pl.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03']})
    result = df.select(simple_function_to_expr('add_days(to_date([date]), 1)'))
    expected = pl.DataFrame({'date': ['2021-01-02', '2021-01-03', '2021-01-04']}).select(pl.col('date').str.to_date())
    assert result.equals(expected)


def test_date_diff_days():
    df = pl.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03'] })
    result = df.select(simple_function_to_expr('date_diff_days(to_date([date]), to_date("2021-01-01"))'))
    expected = pl.DataFrame({'date': [0, 1, 2]})
    assert result.equals(expected)


def test_date_diff_days_two_cols():
    df = pl.DataFrame({'date1': ['2021-01-01', '2021-01-02', '2021-01-03'],
                       'date2': ['2021-03-01', '2021-02-02', '2021-01-03']})
    result = df.select(simple_function_to_expr('date_diff_days(to_date([date1]), to_date([date2]))'))
    expected = pl.DataFrame({'date1': [-59, -31, 0]})
    assert result.equals(expected)


def test_count_match():
    df = pl.DataFrame({'names': ['ham', 'spam', 'eggs'],
                       'subnames': ['bread', 'sandwich', 'breakfast']})
    result = df.select(simple_function_to_expr('count_match([names], "a")'))
    expected = pl.DataFrame({'names': [1, 1, 0]})
    assert result.equals(expected)


def test_count_match_two_cols():
    df = pl.DataFrame({'names': ['hama', 'spam', 'eggs'],
                       'subnames': ['bread', 'sandwich', 'breakfast']})
    result = df.select(simple_function_to_expr('count_match(concat([names], [subnames]), "a")'))
    expected = pl.DataFrame({'names': [3, 2, 2]})
    assert result.equals(expected)


def concat_two_cols_plus_sign():
    df = pl.DataFrame({'names': ['hama', 'spam', 'eggs'],
                       'subnames': ['bread', 'sandwich', 'breakfast']})
    result = df.select(simple_function_to_expr('[names] + [subnames]'))
    expected = pl.DataFrame({'names': ['hamabread', 'spamsandwich', 'eggsbreakfast']})
    assert result.equals(expected)


def test_in_functionality():
    df = pl.DataFrame({'names': ['ham', 'spam', 'eggs'],
                       'subnames': ['bread', 'sandwich', 'breakfast']})
    result = df.select(simple_function_to_expr('"a" in [names]'))
    expected = pl.DataFrame({'names': [True, True, False]})
    assert result.equals(expected)


def test_contains_functionality():
    df = pl.DataFrame({'names': ['ham', 'spam', 'eggs'],
                       'subnames': ['bread', 'sandwich', 'breakfast']})
    result = df.select(simple_function_to_expr('contains([names], "a")'))
    expected = pl.DataFrame({'names': [True, True, False]})
    assert result.equals(expected)


def test_contains_two_cols():
    df = pl.DataFrame({'names': ['ham', 'spam', 'eggs'],
                       'subnames': ['bread', 'sandwich', 'breakfast']})
    result = df.select(simple_function_to_expr('contains(concat([names], [subnames]), "a")'))
    expected = pl.DataFrame({'names': [True, True, True]})
    assert result.equals(expected)


def test_contains_compare_columns():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs'],
                       'subnames': ['bread', 'spam', 'breakfast']})
    result = df.select(simple_function_to_expr('contains([names], [subnames])'))
    expected = pl.DataFrame({'names': [False, True, False]})
    assert result.equals(expected)


def test_replace():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs'],
                       'subnames': ['bread', 'spam', 'breakfast']})
    result = df.select(simple_function_to_expr('replace([names], "a", "o")'))
    expected = pl.DataFrame({'names': ['hom', 'sondwich with spom', 'eggs']})
    assert result.equals(expected)


def replace_in_cols():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs'],
                       'subnames': ['bread', 'spam', 'breakfast']})
    result = df.select(simple_function_to_expr('replace([names], "a", [names])'))
    expected = pl.DataFrame({'names': ['hombread', 'sondwich with spom', 'eggso']})
    assert result.equals(expected)


def test_left():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs'],
                       'subnames': ['bread', 'spam', 'breakfast']})
    result = df.select(simple_function_to_expr('left([names], 2)'))
    expected = pl.DataFrame({'names': ['ha', 'sa', 'eg']})
    assert result.equals(expected)


def test_right():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs'],
                       'subnames': ['bread', 'spam', 'breakfast']})
    result = df.select(simple_function_to_expr('right([names], 2)'))
    expected = pl.DataFrame({'names': ['am', 'am', 'gs']})
    assert result.equals(expected)


def test_right_from_col():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs'],
                       'len': [1, 2, 3]})
    result = df.select(simple_function_to_expr('right([names], [len])'))
    expected = pl.DataFrame({'names': ['m', 'am', 'ggs']})
    assert result.equals(expected)


def test_find_position():
    ...


def test_str_length():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    result = df.select(simple_function_to_expr('length([names])'))
    expected = pl.DataFrame({'names': [3, 18, 4]})
    assert result.equals(expected)


def test_str_length_in_line():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    result = df.select(simple_function_to_expr('length("ham")'))
    expected = pl.DataFrame({'literal': [3]})
    assert result.equals(expected)


def test_complex_logic():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs'],
                       'subnames': ['bread', 'spam', 'breakfast']})
    result = df.select(simple_function_to_expr('if contains([names], "a") then "found" else "not found" endif'))
    expected = pl.DataFrame({'literal': ['found', 'found', 'not found']})
    assert result.equals(expected)


if __name__ == '__main__':
    pytest.main()
