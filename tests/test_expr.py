from polars_expr_transformer.process.polars_expr_transformer import simple_function_to_expr
import polars as pl
from polars.testing import assert_frame_equal


def test_simple_constant_expression():
    print('logging')
    df = pl.from_dicts([{'a': 'row a', 'b': 'row b'}, {'a': 'row a 1', 'b': 'row b 1'}])
    result = df.select(simple_function_to_expr("'hallo world'"))
    expected = pl.DataFrame({'literal': ['hallo world']})
    assert result.equals(expected)


def test_not_equal_columns_expression():
    df = pl.from_dicts([{'a': 12, 'b': 34}, {'a': 56, 'b': 78}])
    result = df.select(simple_function_to_expr('[a] != [b]'))
    expected = pl.DataFrame({'a': [True, True]})
    assert result.equals(expected)


def test_subtract_and_multiplication_expression():
    df = pl.from_dicts([{'a': 12, 'b': 34}, {'a': 56, 'b': 78}])
    result = df.select(simple_function_to_expr('2 * -2'))
    expected = pl.DataFrame({'literal': [-4]})
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


def test_str_contains_expr():
    df = pl.DataFrame({'names': ['ham', 'spam', 'eggs']})
    df.select(pl.lit('this is ham').str.contains(pl.col('names')))
    result = df.select(simple_function_to_expr('contains("this is ham", [names])'))
    expected = pl.DataFrame({'literal': [True, False, False]})
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


def test_right_from_literal_and_column():
    df = pl.DataFrame({'len': [1, 2, 3]})
    result = df.select(simple_function_to_expr('right("edward", [len])'))
    expected = pl.DataFrame({'literal': ['d', 'rd', 'ard']})
    assert result.equals(expected)


def test_left_from_literal_and_column():
    df = pl.DataFrame({'len': [1, 2, 3]})
    result = df.select(simple_function_to_expr('left("edward", [len])'))
    expected = pl.DataFrame({'literal': ['e', 'ed', 'edw']})
    assert result.equals(expected)


def test_find_position():
    df = pl.DataFrame({'names': ['ham', 'cheese with ham', 'eggs'],
                       'other': ['hamm', ' with cheese', 'eeggs']})
    result = df.select(simple_function_to_expr('find_position([names], "a")'))
    expected = pl.DataFrame([1, 13, None], schema={"names": pl.UInt32})
    assert_frame_equal(result, expected)


def test_and_contains_combination():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    expr = simple_function_to_expr('contains([names], "a") and contains([names], "m")')


def test_string_similarity_two_columns():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs'],
                       'other': ['hamm', 'sandwhich with cheese', 'eeggs']})
    result = df.select(simple_function_to_expr('string_similarity([names], [other], "levenshtein")'))
    expected = pl.DataFrame([0.75, 0.666667, 0.8], schema=['names'])
    assert_frame_equal(result, expected)


def test_string_similarity_column_value():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    result = df.select(simple_function_to_expr('string_similarity([names], "ham", "levenshtein")'))
    expected = pl.DataFrame([1.0, 0.166667, 0.0], schema=['names'])
    assert_frame_equal(result, expected)


def test_string_similarity_two_values():
    df = pl.DataFrame({'names': ['ham', 'sandwich with spam', 'eggs']})
    result = df.select(simple_function_to_expr('string_similarity("hams", "ham", "levenshtein")'))
    expected = pl.DataFrame([0.75], schema=['literal'])
    assert_frame_equal(result, expected)


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


def divide_test_two_cols():
    df = pl.DataFrame({'from_values': [1, 2, 3],
                       'to_values': [10, 20, 30]})
    result = df.select(simple_function_to_expr('[to_values]/[from_values]'))
    expected = pl.DataFrame({'to_values': [10.0, 10.0, 10.0]})
    assert result.equals(expected)


def divide_test_two_cols_plus_literal():
    df = pl.DataFrame({'from_values': [1, 2, 3],
                       'to_values': [10, 20, 30]})
    result = df.select(simple_function_to_expr('[to_values]/[from_values] + 1'))
    expected = pl.DataFrame({'to_values': [11.0, 11.0, 11.0]})
    assert result.equals(expected)


def divide_test_two_cols_plus_literal_multiply():
    df = pl.DataFrame({'from_values': [1, 2, 3],
                       'to_values': [10, 20, 30]})
    result = df.select(simple_function_to_expr('([to_values]/[from_values] + 1) * 2'))
    expected = pl.DataFrame({'to_values': [22.0, 22.0, 22.0]})
    assert result.equals(expected)


def test_random_int():
    df = pl.DataFrame({'from_values': [1, 2, 3],
                       'to_values': [10, 20, 30]})
    result = df.select(simple_function_to_expr('random_int(1, 3)'))
    min_val, max_val = result['literal'].min(), result.max()[0, 0]
    assert 1 <= min_val <= max_val < 3, 'Expected random integer between 1 and 3'


def test_complex_nested_parentheses():
    """Test expressions with deeply nested parentheses."""
    df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = df.select(simple_function_to_expr('(([a] + [b]) * 2) / ([b] - [a])'))
    expected = pl.DataFrame({'a': [3.333333, 4.666667, 6.0]})
    assert_frame_equal(expected, result)


def test_complexer_nested_parentheses():
    """Test expressions with deeply nested parentheses."""
    df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = df.select(simple_function_to_expr('([a] + [b] * 2) / ([b] - [a])'))
    expected = pl.DataFrame({'a': [3.0, 4.0, 5.0]})
    assert_frame_equal(expected, result)


def test_multiple_logical_operators():
    """Test expressions with multiple logical operators."""
    df = pl.DataFrame({'a': [1, 5, 10], 'b': [2, 5, 8]})
    result = df.select(simple_function_to_expr('[a] < [b] and [a] > 0 and [b] < 10'))
    expected = pl.DataFrame({'a': [True, False, False]})
    assert result.equals(expected)


def test_nested_function_calls():
    """Test deeply nested function calls."""
    df = pl.DataFrame({'nums': [1.23456, 2.34567, 3.45678]})
    result = df.select(simple_function_to_expr('abs(ceil(floor(round([nums], 2))))'))
    expected = pl.DataFrame({'nums': [1.0, 2.0, 3.0]})
    assert result.equals(expected)


def test_string_with_operators():
    """Test strings containing operator-like symbols."""
    df = pl.DataFrame({'names': ['John', 'Mary', 'Bob']})
    result = df.select(simple_function_to_expr('"a+b*c/d" + [names]'))
    expected = pl.DataFrame({'literal': ['a+b*c/dJohn', 'a+b*c/dMary', 'a+b*c/dBob']})
    assert result.equals(expected)


def test_complex_if_condition():
    """Test complex conditions in if statements."""
    df = pl.DataFrame({'a': [1, 4, 10], 'b': [2, 5, 8]})
    result = df.select(simple_function_to_expr('''
        if [a] < [b] and ([a] * 2 > [b] or [b] / 2 < [a]) then 
            [a] * [b] 
        else 
            [a] + [b] 
        endif
    '''))
    expected = pl.DataFrame({'a': [3, 20, 18]})
    assert result.equals(expected)


def test_consecutive_operators():
    """Test handling of consecutive operators."""
    df = pl.DataFrame({'a': [5, 10, 15]})
    result = df.select(simple_function_to_expr('[a] * -1'))
    expected = pl.DataFrame({'a': [-5, -10, -15]})
    assert result.equals(expected)


def test_whitespace_in_expressions():
    """Test expressions with irregular whitespace."""
    df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = df.select(simple_function_to_expr('  [a]    +  [b]   *   2  '))
    expected = pl.DataFrame({'a': [9, 12, 15]})
    assert result.equals(expected)


def test_column_names_with_special_chars():
    """Test column names containing special characters."""
    df = pl.DataFrame({'col.with.dots': [1, 2, 3], 'col-with-dashes': [4, 5, 6]})
    result = df.select(simple_function_to_expr('[col.with.dots] + [col-with-dashes]'))
    expected = pl.DataFrame({'col.with.dots': [5, 7, 9]})
    assert result.equals(expected)


def test_mixed_case_functions():
    """Test functions with mixed case names."""
    df = pl.DataFrame({'text': ['HELLO', 'World', 'MiXeD']})
    result = df.select(simple_function_to_expr('lowercase([text])'))
    expected = pl.DataFrame({'text': ['hello', 'world', 'mixed']})
    assert result.equals(expected)


def test_long_expression():
    """Test a very long expression with multiple operations."""
    df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    long_expr = '''
        ([a] + [b] * 2) / ([c] - [a]) + 
        sqrt(abs([b] - [c])) * 
        if [a] < [b] then 
            ceil([a] / 2) 
        else 
            floor([b] / 2) 
        endif
    '''
    result = df.select(simple_function_to_expr(long_expr))
    # Calculate expected values manually
    expected_values = []
    for row in range(3):
        a, b, c = df['a'][row], df['b'][row], df['c'][row]
        first_part = (a + b * 2) / (c - a)
        sqrt_part = (b - c) ** 0.5 if b > c else (-1 * (c - b)) ** 0.5
        if a < b:
            if_part = -(-a // 2)  # Ceiling division
        else:
            if_part = b // 2  # Floor division
        expected_values.append(first_part + abs(sqrt_part) * if_part)
    expected = pl.DataFrame({'a': expected_values})
    assert_frame_equal(result, expected, rtol=1e-10)


def test_string_with_keywords():
    """Test strings containing keywords like 'and', 'or', 'if', etc."""
    df = pl.DataFrame({'names': ['John', 'Mary', 'Bob']})
    result = df.select(simple_function_to_expr('"This and that or something if else" + [names]'))
    expected = pl.DataFrame({'literal': ['This and that or something if elseJohn',
                                      'This and that or something if elseMary',
                                      'This and that or something if elseBob']})
    assert result.equals(expected)


def test_null_handling():
    """Test expressions with null values."""
    df = pl.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]})
    result = df.select(simple_function_to_expr('[a] + [b]'))
    expected = pl.DataFrame({'a': [5.0, None, None]})
    assert result.equals(expected)


def test_boolean_literals():
    """Test handling of boolean literals."""
    df = pl.DataFrame({'a': [1, 2, 3]})
    result = df.select(simple_function_to_expr('if [a] > 2 then true else false endif'))
    expected = pl.DataFrame({'literal': [False, False, True]})
    assert result.equals(expected)


def test_chained_string_operations():
    """Test chained string operations."""
    df = pl.DataFrame({'text': ['  hello  ', '  WORLD  ', ' Test ']})
    result = df.select(simple_function_to_expr('uppercase(trim([text]))'))
    expected = pl.DataFrame({'text': ['HELLO', 'WORLD', 'TEST']})
    assert result.equals(expected)


def test_remove_comments():
    """Test that comments starting with // are removed."""
    df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    expr = """
        ([a] + [b] * 2) // This adds a and multiplies b by 2
        / ([b] - [a]) + // Division operation
        sqrt(abs([b] - [b]))
    """
    result = df.select(simple_function_to_expr(expr))
    expected = pl.DataFrame({'a': [3.0, 4.0, 5.0]})
    assert_frame_equal(result, expected)


def test_comments_in_quoted_strings():
    """Test that // inside quoted strings are not treated as comments."""
    df = pl.DataFrame({'text': ['hello', 'world', 'test']})
    result = df.select(simple_function_to_expr('concat([text], " // This is not a comment")'))
    expected = pl.DataFrame({'text': ['hello // This is not a comment',
                                      'world // This is not a comment',
                                      'test // This is not a comment']})
    assert result.equals(expected)


def test_comment_at_end_of_expression():
    """Test that comments at the end of expressions are removed."""
    df = pl.DataFrame({'a': [1, 2, 3]})
    result = df.select(simple_function_to_expr('[a] * 2 // Multiply by 2'))
    expected = pl.DataFrame({'a': [2, 4, 6]})
    assert result.equals(expected)


def test_multline_with_comments():
    """Test multiline expressions with comments on different lines."""
    df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    expr = """
    if [a] < [b] // Check if a is less than b
    then 
        [a] * 2 // Double a
    else 
        [b] * 2 // Double b
    endif // End of if statement
    """
    result = df.select(simple_function_to_expr(expr))
    expected = pl.DataFrame({'a': [2, 4, 6]})
    assert result.equals(expected)


def test_comment_within_if_statement():
    """Test comments within if statement conditions and bodies."""
    df = pl.DataFrame({'a': [1, 5, 10], 'b': [2, 5, 8]})
    expr = """
    if [a] <= [b] // This is the condition
    and [a] > 0 // And this is another condition
    then 
        [a] * [b] // Multiply a and b
    else 
        [a] + [b] // Add a and b
    endif
    """
    result = df.select(simple_function_to_expr(expr))
    expected = pl.DataFrame({'a': [2, 25, 18]})
    assert result.equals(expected)


def test_comment_after_string_literal():
    """Test comments after string literals are properly handled."""
    df = pl.DataFrame({'names': ['John', 'Mary', 'Bob']})
    expr = """
    concat("Hello, ", [names]) // Greeting message
    """
    result = df.select(simple_function_to_expr(expr))
    expected = pl.DataFrame({'literal': ['Hello, John', 'Hello, Mary', 'Hello, Bob']})
    assert result.equals(expected)


def test_multiple_comments_same_line():
    """Test that only the first comment marker on a line is considered."""
    df = pl.DataFrame({'a': [1, 2, 3]})
    expr = "[a] * 2 // First comment // Second comment shouldn't be parsed"
    result = df.select(simple_function_to_expr(expr))
    expected = pl.DataFrame({'a': [2, 4, 6]})
    assert result.equals(expected)


def test_complex_expression_with_comments():
    """Test a complex expression with multiple comments."""
    df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    expr = """
    // Comment at the beginning
    ([a] + [b] * 2) // Comment after first term
    / ([c] - [a]) + // Comment after division
    sqrt(abs([b] - [c])) * // Comment after sqrt
    if [a] < [b] // Comment in if condition
    then 
        ceil([a] / 2) // Comment in then clause
    else 
        floor([b] / 2) // Comment in else clause
    endif // Comment at the end
    """
    result = df.select(simple_function_to_expr(expr))

    # Calculate expected values manually
    expected_values = []
    for row in range(3):
        a, b, c = df['a'][row], df['b'][row], df['c'][row]
        first_part = (a + b * 2) / (c - a)
        sqrt_part = (b - c) ** 0.5 if b > c else (-1 * (c - b)) ** 0.5
        if a < b:
            if_part = -(-a // 2)  # Ceiling division
        else:
            if_part = b // 2  # Floor division
        expected_values.append(first_part + abs(sqrt_part) * if_part)

    expected = pl.DataFrame({'a': expected_values})
    assert_frame_equal(result, expected, rtol=1e-10)
