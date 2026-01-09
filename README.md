# Polars Expression Transformer

[![PyPI version](https://badge.fury.io/py/polars-expr-transformer.svg)](https://badge.fury.io/py/polars-expr-transformer)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform string-based expressions into [Polars](https://pola.rs/) DataFrame operations. Write simple, SQL-like expressions and let the library convert them to optimized Polars code.

## Quick Start

```python
import polars as pl
from polars_expr_transformer import simple_function_to_expr

df = pl.DataFrame({
    'first_name': ['John', 'Jane', 'Bob'],
    'last_name': ['Doe', 'Smith', 'Johnson'],
    'age': [30, 25, 45],
    'salary': [50000, 60000, 75000]
})

# Concatenate columns
df.select(simple_function_to_expr('concat([first_name], " ", [last_name])').alias('full_name'))

# Conditional logic
df.select(simple_function_to_expr('if [age] > 30 then "Senior" else "Junior" endif').alias('level'))

# Math operations
df.select(simple_function_to_expr('[salary] * 1.1').alias('new_salary'))

# Combine multiple operations
df.select(simple_function_to_expr('uppercase(left([last_name], 3))').alias('code'))
```

## Installation

```bash
pip install polars-expr-transformer
```

## Why Use This Library?

| Use Case | Recommendation |
|----------|----------------|
| Building applications with user-defined transformations | ✅ **Yes** - Users can write expressions without Python knowledge |
| SQL/Tableau users transitioning to Polars | ✅ **Yes** - Familiar syntax |
| Need a simple expression language for configs | ✅ **Yes** - Easy to serialize and store |
| Writing performance-critical Polars code | ❌ **No** - Use Polars directly |
| Need all Polars features | ❌ **No** - This covers common operations only |

## Expression Syntax

### Column References
Reference DataFrame columns using square brackets:
```python
'[column_name]'           # Reference a column
'[Column With Spaces]'    # Columns with spaces work too
```

### Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `[a] + [b]` |
| `-` | Subtraction | `[a] - 10` |
| `*` | Multiplication | `[price] * [quantity]` |
| `/` | Division | `[total] / [count]` |
| `%` | Modulo | `[value] % 2` |
| `=` or `==` | Equals | `[status] = "active"` |
| `!=` | Not equals | `[type] != "deleted"` |
| `>`, `>=`, `<`, `<=` | Comparisons | `[age] >= 18` |
| `and` | Logical AND | `[a] > 0 and [b] > 0` |
| `or` | Logical OR | `[x] = 1 or [y] = 1` |

### Conditional Expressions

```python
# Simple if-then-else
'if [age] >= 18 then "Adult" else "Minor" endif'

# Multiple conditions with elseif
'if [score] >= 90 then "A" elseif [score] >= 80 then "B" elseif [score] >= 70 then "C" else "F" endif'

# Nested conditions
'if [type] = "A" then (if [value] > 100 then "High A" else "Low A" endif) else "Other" endif'
```

### Comments

```python
# Single-line comments with //
'[column] + 1 // This adds one to the column'

# Multi-line expressions with comments
'''
[price] * [quantity]  // Calculate subtotal
- [discount]          // Apply discount
'''
```

## Available Functions

### String Functions

| Function | Description | Example |
|----------|-------------|---------|
| `concat(a, b, ...)` | Concatenate strings | `concat([first], " ", [last])` |
| `length(text)` | String length | `length([name])` |
| `uppercase(text)` | Convert to uppercase | `uppercase([code])` |
| `lowercase(text)` | Convert to lowercase | `lowercase([email])` |
| `titlecase(text)` | Convert to title case | `titlecase([name])` |
| `left(text, n)` | First n characters | `left([phone], 3)` |
| `right(text, n)` | Last n characters | `right([id], 4)` |
| `mid(text, start, len)` | Substring from position | `mid([code], 2, 3)` |
| `substring(text, start, len)` | Alias for mid | `substring([text], 0, 10)` |
| `trim(text)` | Remove leading/trailing spaces | `trim([input])` |
| `left_trim(text)` | Remove leading spaces | `left_trim([text])` |
| `right_trim(text)` | Remove trailing spaces | `right_trim([text])` |
| `replace(text, find, replace)` | Replace text | `replace([name], ".", "")` |
| `find_position(text, search)` | Find substring position | `find_position([text], "@")` |
| `pad_left(text, len, char)` | Pad string on left | `pad_left([id], 5, "0")` |
| `pad_right(text, len, char)` | Pad string on right | `pad_right([code], 10, " ")` |
| `starts_with(text, prefix)` | Check prefix | `starts_with([url], "https")` |
| `ends_with(text, suffix)` | Check suffix | `ends_with([file], ".csv")` |
| `reverse(text)` | Reverse string | `reverse([text])` |
| `repeat(text, n)` | Repeat string n times | `repeat("*", 5)` |
| `split(text, delimiter)` | Split into list | `split([tags], ",")` |
| `count_match(text, pattern)` | Count occurrences | `count_match([text], "a")` |
| `string_similarity(a, b, method)` | Similarity score (0-1) | `string_similarity([a], [b], "levenshtein")` |

### Math Functions

| Function | Description | Example |
|----------|-------------|---------|
| `abs(n)` | Absolute value | `abs([difference])` |
| `round(n, decimals)` | Round to decimals | `round([price], 2)` |
| `ceil(n)` | Round up | `ceil([value])` |
| `floor(n)` | Round down | `floor([value])` |
| `power(base, exp)` | Exponentiation | `power([x], 2)` |
| `pow(base, exp)` | Alias for power | `pow(2, [n])` |
| `sqrt(n)` | Square root | `sqrt([area])` |
| `log(n)` | Natural logarithm | `log([value])` |
| `log10(n)` | Base-10 logarithm | `log10([value])` |
| `log2(n)` | Base-2 logarithm | `log2([value])` |
| `exp(n)` | e^n | `exp([rate])` |
| `mod(a, b)` | Modulo | `mod([value], 10)` |
| `sign(n)` | Sign (-1, 0, 1) | `sign([change])` |
| `negation(n)` | Negate value | `negation([amount])` |
| `sin(n)`, `cos(n)`, `tan(n)` | Trigonometric | `sin([angle])` |
| `asin(n)`, `acos(n)`, `atan(n)` | Inverse trig | `asin([ratio])` |
| `tanh(n)` | Hyperbolic tangent | `tanh([x])` |
| `random_int(min, max)` | Random integer | `random_int(1, 100)` |

### Date Functions

| Function | Description | Example |
|----------|-------------|---------|
| `now()` | Current datetime | `now()` |
| `today()` | Current date | `today()` |
| `year(date)` | Extract year | `year([created_at])` |
| `month(date)` | Extract month (1-12) | `month([date])` |
| `day(date)` | Extract day (1-31) | `day([date])` |
| `hour(datetime)` | Extract hour (0-23) | `hour([timestamp])` |
| `minute(datetime)` | Extract minute | `minute([time])` |
| `second(datetime)` | Extract second | `second([time])` |
| `week(date)` | ISO week number (1-53) | `week([date])` |
| `weekday(date)` | Day of week (1=Mon, 7=Sun) | `weekday([date])` |
| `dayofweek(date)` | Alias for weekday | `dayofweek([date])` |
| `quarter(date)` | Quarter (1-4) | `quarter([date])` |
| `dayofyear(date)` | Day of year (1-366) | `dayofyear([date])` |
| `add_days(date, n)` | Add days | `add_days([start], 30)` |
| `add_weeks(date, n)` | Add weeks | `add_weeks([date], 2)` |
| `add_months(date, n)` | Add months | `add_months([date], 6)` |
| `add_years(date, n)` | Add years | `add_years([birth], 18)` |
| `add_hours(dt, n)` | Add hours | `add_hours([time], 3)` |
| `add_minutes(dt, n)` | Add minutes | `add_minutes([time], 30)` |
| `add_seconds(dt, n)` | Add seconds | `add_seconds([time], 60)` |
| `date_diff_days(a, b)` | Days between dates | `date_diff_days([end], [start])` |
| `datetime_diff_seconds(a, b)` | Seconds between | `datetime_diff_seconds([a], [b])` |
| `format_date(date, fmt)` | Format as string | `format_date([date], "%Y-%m-%d")` |
| `start_of_month(date)` | First of month | `start_of_month([date])` |
| `end_of_month(date)` | Last of month | `end_of_month([date])` |
| `date_truncate(date, unit)` | Truncate to unit | `date_truncate([dt], "1day")` |

### Logic & Null Handling

| Function | Description | Example |
|----------|-------------|---------|
| `equals(a, b)` | Check equality | `equals([status], "active")` |
| `does_not_equal(a, b)` | Check inequality | `does_not_equal([type], "deleted")` |
| `is_empty(value)` | Check if null | `is_empty([email])` |
| `is_not_empty(value)` | Check if not null | `is_not_empty([phone])` |
| `coalesce(a, b, ...)` | First non-null | `coalesce([nickname], [name], "Unknown")` |
| `ifnull(value, default)` | Replace null | `ifnull([count], 0)` |
| `nvl(value, default)` | Alias for ifnull | `nvl([value], 0)` |
| `nullif(a, b)` | Null if equal | `nullif([value], 0)` |
| `between(val, min, max)` | Range check (inclusive) | `between([age], 18, 65)` |
| `greatest(a, b, ...)` | Maximum value | `greatest([a], [b], [c])` |
| `least(a, b, ...)` | Minimum value | `least([price1], [price2])` |
| `contains(text, search)` | Contains substring | `contains([desc], "sale")` |
| `_in(value, text)` | Value in text | `_in("admin", [roles])` |
| `_not(value)` | Logical NOT | `_not([is_deleted])` |
| `is_string(value)` | Type check | `is_string([field])` |

### Type Conversions

| Function | Description | Example |
|----------|-------------|---------|
| `to_string(value)` | Convert to string | `to_string([id])` |
| `to_integer(value)` | Convert to integer | `to_integer([count])` |
| `to_float(value)` | Convert to float | `to_float([price])` |
| `to_number(value)` | Alias for to_float | `to_number([value])` |
| `to_boolean(value)` | Convert to boolean | `to_boolean([flag])` |
| `to_date(text, format)` | Parse date | `to_date([date_str], "%Y-%m-%d")` |
| `to_datetime(text, format)` | Parse datetime | `to_datetime([ts], "%Y-%m-%d %H:%M:%S")` |
| `to_decimal(value, precision)` | Convert with precision | `to_decimal([amount], 2)` |

## API Reference

### `simple_function_to_expr(expression: str) -> pl.Expr`

Converts a string expression to a Polars expression.

```python
from polars_expr_transformer import simple_function_to_expr

expr = simple_function_to_expr('[price] * [quantity]')
df.select(expr.alias('total'))
```

### `build_func(expression: str) -> Func`

Returns the intermediate function object for inspection/debugging.

```python
from polars_expr_transformer import build_func

func = build_func('concat([a], [b])')
print(func.get_readable_pl_function())  # See the Polars translation
```

### `get_all_expressions() -> List[str]`

Returns a list of all available function names.

```python
from polars_expr_transformer import get_all_expressions

functions = get_all_expressions()
print(functions)  # ['concat', 'length', 'uppercase', ...]
```

### `get_expression_overview() -> List[ExpressionsOverview]`

Returns functions grouped by category with descriptions.

```python
from polars_expr_transformer import get_expression_overview

for category in get_expression_overview():
    print(f"\n{category.category}:")
    for expr in category.expressions:
        print(f"  {expr.name}: {expr.description}")
```

## Error Handling

The library validates expressions and provides helpful error messages:

```python
# Unbalanced parentheses
simple_function_to_expr('((1)')
# ValueError: Unbalanced parentheses: 1 unclosed '(' found

# Unknown function
simple_function_to_expr('unknown_func([col])')
# Raises error with available functions
```

## Built on Polars

This library is built on top of [Polars](https://pola.rs/), a blazingly fast DataFrame library written in Rust. All expressions are converted to native Polars operations, ensuring optimal performance.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests on [GitHub](https://github.com/edwardvaneechoud/polars_expr_transformer).

## License

MIT License - see LICENSE file for details.

## Acknowledgements

Thanks to the Polars team for creating such an amazing library.
