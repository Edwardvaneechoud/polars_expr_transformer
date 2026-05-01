"""
Polars Expression Transformer - Core transformation module.

This module provides the main functions for converting string-based expressions
into Polars DataFrame operations. It handles the complete pipeline from parsing
to generating executable Polars expressions.

Example:
    >>> from polars_expr_transformer import simple_function_to_expr
    >>> import polars as pl
    >>> df = pl.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 25]})
    >>> expr = simple_function_to_expr('concat([name], " is ", to_string([age]))')
    >>> df.select(expr.alias('description'))
"""

from typing import List, Union
from polars_expr_transformer.process.models import IfFunc, Func, TempFunc, Classifier
from polars_expr_transformer.process.hierarchy_builder import build_hierarchy
from polars_expr_transformer.process.tokenize import tokenize
from polars_expr_transformer.process.token_classifier import classify_tokens
from polars_expr_transformer.process.process_inline import parse_inline_functions
from polars_expr_transformer.process.post_process import (
    post_process_hierarchical_formula,
)
from polars_expr_transformer.process.preprocess import preprocess
from polars_expr_transformer.exceptions import PolarsCodeGenError
import polars as pl
import datetime


def finalize_hierarchy(obj):
    """
    Recursively remove all TempFunc instances from the hierarchical formula.

    Args:
        obj: The object to process (Func, TempFunc, IfFunc, or Classifier)

    Returns:
        The processed object with all TempFunc instances replaced by their arguments,
        or None if a TempFunc has zero arguments
    """
    # Base case: If it's not a TempFunc or doesn't need recursion, return it
    if isinstance(obj, Classifier) or obj is None:
        return obj

    # Handle TempFunc
    if isinstance(obj, TempFunc):
        if len(obj.args) > 1:
            raise Exception("TempFunc should have at most one argument")

        # Case: TempFunc with no arguments - remove it from parent
        if len(obj.args) == 0:
            if hasattr(obj, "parent") and obj.parent:
                # If this TempFunc is in the parent's args list, remove it
                if isinstance(obj.parent, Func):
                    if obj in obj.parent.args:
                        obj.parent.args.remove(obj)
                elif isinstance(obj.parent, IfFunc):
                    if obj.parent.else_val is obj:
                        obj.parent.else_val = None
                elif hasattr(obj.parent, "condition") and obj.parent.condition is obj:
                    obj.parent.condition = None
                elif hasattr(obj.parent, "val") and obj.parent.val is obj:
                    obj.parent.val = None
            # Return None to indicate this TempFunc should be removed
            return None

        # Case: TempFunc with one argument - replace it with its child
        if len(obj.args) == 1:
            child = obj.args[0]

            # Set the parent of the child to be the parent of this TempFunc
            if hasattr(obj, "parent") and obj.parent:
                child.parent = obj.parent

                # Replace this TempFunc in the parent's args list
                if isinstance(obj.parent, Func):
                    for i, arg in enumerate(obj.parent.args):
                        if arg is obj:
                            obj.parent.args[i] = child
                            break
                elif isinstance(obj.parent, IfFunc):
                    if obj.parent.else_val is obj:
                        obj.parent.else_val = child
                elif hasattr(obj.parent, "condition") and obj.parent.condition is obj:
                    obj.parent.condition = child
                elif hasattr(obj.parent, "val") and obj.parent.val is obj:
                    obj.parent.val = child

            # Process the child recursively
            return finalize_hierarchy(child)

    # Process Func objects
    if isinstance(obj, Func):
        # Process all arguments and filter out None values (removed TempFuncs)
        processed_args = []
        for arg in obj.args:
            processed_arg = finalize_hierarchy(arg)
            if processed_arg is not None:
                processed_args.append(processed_arg)
        obj.args = processed_args

    # Process IfFunc objects
    elif isinstance(obj, IfFunc):
        # Process conditions
        for cond in obj.conditions:
            cond.condition = finalize_hierarchy(cond.condition)
            cond.val = finalize_hierarchy(cond.val)

        # Process else_val
        if obj.else_val:
            obj.else_val = finalize_hierarchy(obj.else_val)

    return obj


# Wrapper function to handle the top-level case properly
def remove_temp_funcs(hierarchical_formula):
    """
    Remove all TempFunc instances from the hierarchical formula.

    This wrapper function ensures the top-level formula is properly handled
    and all TempFunc instances are replaced with their actual arguments.

    Args:
        hierarchical_formula: The hierarchical formula to process

    Returns:
        The processed formula with all TempFunc instances removed
    """
    result = finalize_hierarchy(hierarchical_formula)

    # If we did a replacement at the top level, make sure we return the new object
    if result is not hierarchical_formula:
        return result
    return hierarchical_formula


def build_func(func_str: str = 'concat("1", "2")') -> Func:
    """
    Build a Func object from a function string.

    This function takes a string representation of a function, preprocesses it,
    tokenizes it, classifies tokens, builds a hierarchical structure, and parses
    inline operators. The resulting Func object can be inspected or converted
    to a Polars expression.

    Args:
        func_str: The string expression to parse. Supports column references
            like [column_name], functions like concat(), operators (+, -, *, /),
            and conditional expressions (if/then/else/endif).

    Returns:
        A Func object representing the parsed expression tree.

    Example:
        >>> func = build_func('concat([first_name], " ", [last_name])')
        >>> print(func.get_readable_pl_function())
        >>> expr = func.get_pl_func()  # Convert to Polars expression

    Raises:
        ValueError: If parentheses are unbalanced.
        Exception: If the expression syntax is invalid.
    """
    formula = preprocess(func_str)
    raw_tokens = tokenize(formula)
    tokens = classify_tokens(raw_tokens)
    hierarchical_formula = build_hierarchy(tokens)
    parse_inline_functions(hierarchical_formula)

    finalized_hierarchical_formula = finalize_hierarchy(hierarchical_formula)
    hierarchical_formula.get_pl_func()
    return finalized_hierarchical_formula


def test_tokenization(func_str, all_split_vals, all_functions):
    """
    Test the preprocessing and tokenization of a function string.

    Args:
        func_str: The function string to test.
        all_split_vals: Set of all split values.
        all_functions: Dictionary of all functions.

    Returns:
        The tokenized result.
    """
    print(f"Original: {func_str}")
    processed = preprocess(func_str)
    print(f"Preprocessed: {processed}")
    tokens = tokenize(processed)
    print(f"Tokens: {tokens}")
    return tokens


def _validate_polars_code(func_str: str, code: str) -> None:
    """Validate generated Polars code by eval-ing it.

    Builds a scope with ``pl``, ``datetime``, and optionally ``pds``
    (polars_ds), then attempts ``eval(code, scope)``.

    Raises:
        PolarsCodeGenError: If the generated code cannot be evaluated.
    """
    scope = {"pl": pl, "datetime": datetime}
    try:
        eval(code, scope)
    except Exception as e:
        raise PolarsCodeGenError(func_str, code, e) from e


def to_polars_code(func_str: str, validate: bool = True) -> str:
    """
    Convert a string expression to a native Polars Python code string.

    This function parses the expression and generates the equivalent Polars
    Python code as a string, which can be used for learning, debugging,
    or code generation purposes.

    Args:
        func_str: The string expression to convert. Supports the same syntax
            as simple_function_to_expr: column references [col], operators,
            functions, and conditionals.
        validate: If True (default), eval the generated code to verify it
            is syntactically and semantically valid. Raises PolarsCodeGenError
            on failure.

    Returns:
        A string containing valid Polars Python code.

    Raises:
        PolarsCodeGenError: If validate=True and the generated code fails eval.

    Example:
        >>> to_polars_code("[col_a] + 'test'")
        'pl.col("col_a") + pl.lit("test")'

        >>> to_polars_code("uppercase([name])")
        'pl.col("name").str.to_uppercase()'

        >>> to_polars_code("if [age] > 30 then 'Senior' else 'Junior' endif")
        'pl.when(pl.col("age") > pl.lit(30)).then(pl.lit("Senior")).otherwise(pl.lit("Junior"))'
    """
    func = build_func(func_str)
    code = func.to_polars_code()
    if validate:
        _validate_polars_code(func_str, code)
    return code


def to_flowframe_code(func_str: str, validate: bool = True) -> str:
    """
    Convert a string expression to a native FlowFrame Python code string.

    This function works identically to ``to_polars_code`` but generates
    code targeting FlowFrame (``ff.``) instead of Polars (``pl.``).
    FlowFrame exposes the same API as Polars, so the same conversion
    rules apply.

    Validation is performed by generating and evaluating the equivalent
    Polars code — if the Polars code is valid, the FlowFrame code is
    guaranteed to be valid as well (since the APIs are identical).

    Args:
        func_str: The string expression to convert. Supports the same syntax
            as ``to_polars_code``.
        validate: If True (default), generate the equivalent Polars code and
            eval it to verify the expression is valid. Raises
            PolarsCodeGenError on failure.

    Returns:
        A string containing valid FlowFrame Python code.

    Raises:
        PolarsCodeGenError: If validate=True and the generated code fails eval.

    Example:
        >>> to_flowframe_code("[col_a] + 'test'")
        'ff.col("col_a") + ff.lit("test")'

        >>> to_flowframe_code("uppercase([name])")
        'ff.col("name").str.to_uppercase()'
    """
    func = build_func(func_str)
    if validate:
        pl_code = func.to_polars_code()
        _validate_polars_code(func_str, pl_code)
    return func.to_polars_code(prefix="ff")


def simple_function_to_expr(func_str: str) -> pl.expr.Expr:
    """
    Convert a string expression to a Polars expression.

    This is the main entry point for transforming string-based expressions
    into executable Polars operations. The resulting expression can be used
    directly with DataFrame.select(), DataFrame.with_columns(), etc.

    Args:
        func_str: The string expression to convert. Supports:
            - Column references: [column_name]
            - Operators: +, -, *, /, %, =, !=, <, >, <=, >=, and, or
            - Functions: concat(), uppercase(), round(), etc.
            - Conditionals: if [col] > 0 then "positive" else "negative" endif
            - Comments: // This is a comment

    Returns:
        A Polars expression (pl.Expr) that can be used in DataFrame operations.

    Example:
        >>> import polars as pl
        >>> from polars_expr_transformer import simple_function_to_expr
        >>>
        >>> df = pl.DataFrame({'price': [10, 20, 30], 'qty': [2, 3, 1]})
        >>>
        >>> # Simple math
        >>> df.select(simple_function_to_expr('[price] * [qty]').alias('total'))
        >>>
        >>> # String operations
        >>> df.select(simple_function_to_expr('concat("$", to_string([price]))').alias('formatted'))
        >>>
        >>> # Conditional logic
        >>> expr = 'if [price] > 15 then "expensive" else "cheap" endif'
        >>> df.select(simple_function_to_expr(expr).alias('category'))

    Raises:
        ValueError: If parentheses are unbalanced.
        Exception: If the expression syntax is invalid or function is unknown.
    """
    func = build_func(func_str)
    return func.get_pl_func()
