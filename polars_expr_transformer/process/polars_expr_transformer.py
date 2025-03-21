from typing import List, Union
from polars_expr_transformer.process.models import IfFunc, Func, TempFunc
from polars_expr_transformer.process.hierarchy_builder import build_hierarchy
from polars_expr_transformer.process.tokenize import tokenize
from polars_expr_transformer.process.token_classifier import classify_tokens
from polars_expr_transformer.process.process_inline import parse_inline_functions
from polars_expr_transformer.process.post_process import post_process_hierarchical_formula
from polars_expr_transformer.process.preprocess import preprocess
import polars as pl


def finalize_hierarchy(hierarchical_formula: Union[Func, TempFunc, IfFunc]):
    if isinstance(hierarchical_formula, TempFunc):
        if len(hierarchical_formula.args) != 1:
            raise Exception(f'Expected exactly one argument in TempFunc, got {len(hierarchical_formula.args)}')
        arg = hierarchical_formula.args[0]
        if hasattr(hierarchical_formula, 'parent') and hierarchical_formula.parent:
            arg.parent = hierarchical_formula.parent
        return finalize_hierarchy(arg)

    if isinstance(hierarchical_formula, Func):
        _process_args(hierarchical_formula)
    elif isinstance(hierarchical_formula, IfFunc):
        _process_if_func(hierarchical_formula)

    return hierarchical_formula


def _process_args(func: Func):
    i = 0
    while i < len(func.args):
        arg = func.args[i]
        if isinstance(arg, TempFunc):
            if len(arg.args) != 1:
                raise Exception(f'Expected exactly one argument in TempFunc, got {len(arg.args)}')
            child = arg.args[0]
            child.parent = func
            func.args[i] = child
            _ensure_no_temp_funcs(child)
        else:
            _ensure_no_temp_funcs(arg)
        i += 1


def _process_if_func(if_func: IfFunc):
    for cond_val in if_func.conditions:
        if isinstance(cond_val.condition, TempFunc):
            if len(cond_val.condition.args) != 1:
                raise Exception(f'Expected exactly one argument in condition TempFunc')
            child = cond_val.condition.args[0]
            child.parent = cond_val
            cond_val.condition = child
            _ensure_no_temp_funcs(child)
        else:
            _ensure_no_temp_funcs(cond_val.condition)

        if isinstance(cond_val.val, TempFunc):
            if len(cond_val.val.args) != 1:
                raise Exception(f'Expected exactly one argument in value TempFunc')
            child = cond_val.val.args[0]
            child.parent = cond_val
            cond_val.val = child
            _ensure_no_temp_funcs(child)
        else:
            _ensure_no_temp_funcs(cond_val.val)

    if if_func.else_val:
        if isinstance(if_func.else_val, TempFunc):
            if len(if_func.else_val.args) != 1:
                raise Exception(f'Expected exactly one argument in else TempFunc')
            child = if_func.else_val.args[0]
            child.parent = if_func
            if_func.else_val = child
            _ensure_no_temp_funcs(child)
        else:
            _ensure_no_temp_funcs(if_func.else_val)


def _ensure_no_temp_funcs(formula: Union[Func, IfFunc]):
    if isinstance(formula, Func):
        _process_args(formula)
    elif isinstance(formula, IfFunc):
        _process_if_func(formula)



def build_func(func_str: str = 'concat("1", "2")') -> Func:
    """
    Build a Func object from a function string.

    This function takes a string representation of a function, preprocesses it,
    tokenizes it, standardizes the tokens, builds a hierarchical structure from
    the tokens, parses any inline functions, and finally returns the resulting Func object.

    Args:
        func_str: The string representation of the function to build. Defaults to 'concat("1", "2")'.

    Returns:
        The resulting Func object built from the function string.
    """
    func_str =  'round((1+122.3212), 2)'
    formula = preprocess(func_str)
    tokens = tokenize(formula)
    standardized_tokens = classify_tokens(tokens)
    hierarchical_formula = build_hierarchy(standardized_tokens)
    print(hierarchical_formula)
    parse_inline_functions(hierarchical_formula)
    print(hierarchical_formula)
    finalized_hierarchical_formula = finalize_hierarchy(hierarchical_formula)
    print(finalized_hierarchical_formula)
    return finalized_hierarchical_formula


def simple_function_to_expr(func_str: str) -> pl.expr.Expr:
    """
    Convert a simple function string to a Polars expression.

    This function takes a string representation of a function, builds a corresponding
    Func object, and then converts that Func object to a Polars expression.

    Args:
        func_str: The string representation of the function to convert.

    Returns:
        The resulting Polars expression (pl.expr.Expr).
    """
    func = build_func(func_str)
    return func.get_pl_func()
