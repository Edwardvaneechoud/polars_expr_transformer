"""
Polars Expression Transformer.

A library for converting string-based expressions into Polars DataFrame operations.
Write simple, SQL-like expressions and let the library convert them to optimized Polars code.

Example:
    >>> import polars as pl
    >>> from polars_expr_transformer import simple_function_to_expr
    >>>
    >>> df = pl.DataFrame({
    ...     'name': ['Alice', 'Bob'],
    ...     'age': [30, 25]
    ... })
    >>>
    >>> # String operations
    >>> df.select(simple_function_to_expr('uppercase([name])').alias('upper_name'))
    >>>
    >>> # Conditional logic
    >>> df.select(simple_function_to_expr(
    ...     'if [age] >= 30 then "Senior" else "Junior" endif'
    ... ).alias('level'))

Functions:
    simple_function_to_expr: Convert a string expression to a Polars expression.
    build_func: Build a Func object for inspection/debugging.
    get_all_expressions: Get a list of all available function names.
    get_expression_overview: Get functions grouped by category with descriptions.
"""

from polars_expr_transformer.main_module import build_func, simple_function_to_expr
from polars_expr_transformer.function_overview import get_all_expressions, get_expression_overview

__all__ = [
    'simple_function_to_expr',
    'build_func',
    'get_all_expressions',
    'get_expression_overview',
]