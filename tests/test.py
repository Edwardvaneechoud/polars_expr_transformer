import pytest
from polars_expr_transformer.process.polars_expr_transformer import preprocess, simple_function_to_expr
import polars as pl


def test_simple_constant_expression():
    print('logging')
    df = pl.from_dicts([{'a': 'row a', 'b': 'row b'}, {'a': 'row a 1', 'b': 'row b 1'}])
    result = df.select(simple_function_to_expr("'hallo world'"))
    expected = pl.DataFrame({'literal': ['hallo world']})
    assert result.equals(expected)


def test_subtract_and_multiplication_expression():
    df = pl.from_dicts([{'a': 12, 'b': 34}, {'a': 56, 'b': 78}])
    result = df.select(simple_function_to_expr('2 * -2'))
    expected = pl.DataFrame({'literal': [-4]})
    assert result.equals(expected)


def test_not_equal_cols():
    df = pl.from_dicts([{'a': 12, 'b': 34}, {'a': 56, 'b': 78}])
    result = df.select(simple_function_to_expr('[a] != [b]'))
    expected = pl.DataFrame({'a': [True, True]})
    assert result.equals(expected)


def test_subtraction_expression_two_columns():
    df = pl.from_dicts([{'a': 12, 'b': 34}, {'a': 56, 'b': 78}])
    result = df.select(simple_function_to_expr('[a]-[b]'))
    expected = pl.DataFrame({'a': [-22, -22]})
    assert result.equals(expected)


def test_subtraction_expression_one_column():
    df = pl.from_dicts([{'a': 12, 'b': 34}, {'a': 56, 'b': 78}])
    result = df.select(simple_function_to_expr('[a]-2'))
    expected = pl.DataFrame({'a': [10, 54]})
    assert result.equals(expected)


def test_negative():
    df = pl.from_dicts([{'a': 12, 'b': 34}, {'a': 56, 'b': 78}])
    result = df.select(simple_function_to_expr('-[a]'))
    expected = pl.DataFrame({'a': [-12, -56]})
    assert result.equals(expected)


def test_combining_columns_expression():
    df = pl.from_dicts([{'a': 'man', 'b': 'woman'}, {'a': 'woman', 'b': 'man'}])
    result = df.select(simple_function_to_expr('[a] + " loves " + [b]').alias('literal'))
    expected = pl.DataFrame({'literal': ['man loves woman', 'woman loves man']})
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
    print('yes')
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


def test_to_string_concat():
    df = pl.DataFrame({'numbers': [1, 2, 3], 'more_numbers': [4, 5, 6]})
    result = df.select(simple_function_to_expr('to_string([numbers]) + to_string([more_numbers])'))
    expected = pl.DataFrame({'numbers': ['14', '25', '36']})
    assert result.equals(expected)


def test_date_func_concat():
    df = pl.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03']})
    df_with_dates = df.select(pl.col('date').str.to_date())
    func_str = 'to_date(to_string(year([date])) + "-"+ to_string(month([date])) + "-" + to_string(day([date])))'
    result = df_with_dates.select(simple_function_to_expr(func_str))
    expected = df_with_dates
    assert result.equals(expected)


def test_ceil():
    df = pl.DataFrame({'numbers': [1.1, 2.2, 3.3]})
    result = df.select(simple_function_to_expr('ceil([numbers])'))
    expected = pl.DataFrame({'numbers': [2, 3, 4]})
    assert result.equals(expected)


def test_floor():
    df = pl.DataFrame({'numbers': [1.1, 2.2, 3.3]})
    result = df.select(simple_function_to_expr('floor([numbers])'))
    expected = pl.DataFrame({'numbers': [1, 2, 3]})
    assert result.equals(expected)


def test_tanh():
    df = pl.DataFrame({'numbers': [1.1, 2.2, 3.3]})
    result = df.select(simple_function_to_expr('tanh([numbers])'))
    expected = df.select(pl.col('numbers').tanh())
    assert result.equals(expected)


def test_sqrt():
    df = pl.DataFrame({'numbers': [1.1, 2.2, 3.3]})
    result = df.select(simple_function_to_expr('sqrt([numbers])'))
    expected = df.select(pl.col('numbers').sqrt())
    assert result.equals(expected)


def test_abs():
    df = pl.DataFrame({'numbers': [1.1, -2.2, 3.3]})
    result = df.select(simple_function_to_expr('abs([numbers])'))
    expected = df.select(pl.col('numbers').abs())
    assert result.equals(expected)


def test_sin():
    df = pl.DataFrame({'numbers': [1.1, 2.2, 3.3]})
    result = df.select(simple_function_to_expr('sin([numbers])'))
    expected = df.select(pl.col('numbers').sin())
    assert result.equals(expected)


def test_cos():
    df = pl.DataFrame({'numbers': [1.1, 2.2, 3.3]})
    result = df.select(simple_function_to_expr('cos([numbers])'))
    expected = df.select(pl.col('numbers').cos())
    assert result.equals(expected)


def test_tan():
    df = pl.DataFrame({'numbers': [1.1, 2.2, 3.3]})
    result = df.select(simple_function_to_expr('tan([numbers])'))
    expected = df.select(pl.col('numbers').tan())
    assert result.equals(expected)


def test_pad_left():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    result = df.select(simple_function_to_expr('pad_left([names], 10, " ")'))
    expected = pl.DataFrame({'names': ['       ham', 'sandwich with spam', '      eggs']})
    assert result.equals(expected)


def test_trim():
    df = pl.DataFrame({'names': ['   ham', 'sandwich with spam   ', 'eggs   ']})
    result = df.select(simple_function_to_expr('trim([names])'))
    expected = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    assert result.equals(expected)


def test_pad_right():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    result = df.select(simple_function_to_expr('pad_right([names], 10, " ")'))
    expected = pl.DataFrame({'names': ['ham       ', 'sandwich with spam', 'eggs      ']})
    assert result.equals(expected)

def test_multiply_if_else():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    result = df.select(simple_function_to_expr('if contains([names], "a") then 10 else 20 endif') * 2)
    expected = pl.DataFrame({'literal': [20, 20, 40]})
    assert result.equals(expected)


def test_if_elseif_else_multiply():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    result = df.select(simple_function_to_expr('if contains([names], "an") then 10 elseif contains([names], "s") then 20 else 30 endif') * 2)
    expected = pl.DataFrame({'literal': [60, 20, 40]})
    assert result.equals(expected)


def test_combination_add():
    sf1 = 'if contains([names], "an") then 10 elseif contains([names], "s") then 20 else 30 endif'
    sf2 = 'if contains([names], "a") then 10 else 20 endif'
    combined = f'({sf1}) + ({sf2})'
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    result = df.select(simple_function_to_expr(combined))
    expected = pl.DataFrame({'literal': [40, 20, 40]})
    assert result.equals(expected)


def test_build_on_combination():
    sf1 = 'if contains([names], "anw") then 10 elseif contains([names], "s") then 20 else 30 endif'
    combined = 'concat("result: ", ' + sf1 + ')'
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    result = df.select(simple_function_to_expr(combined))
    expected = pl.DataFrame({'literal': ['result: 30', 'result: 20', 'result: 20']})
    assert result.equals(expected)


def divide_test():
    func_string = '1+(2/2)'
    result = pl.select(simple_function_to_expr(func_string))
    expected = pl.DataFrame({'literal': [2]})
    assert result.equals(expected)


def divide_test_simple():
    func_string = '2/2'
    result = pl.select(simple_function_to_expr(func_string))
    expected = pl.DataFrame({'literal': [1]})
    assert result.equals(expected)


def detect_date_format():
    ...


if __name__ == '__main__':
    pytest.main()

