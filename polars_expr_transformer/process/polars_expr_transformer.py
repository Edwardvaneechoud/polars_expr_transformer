from typing import List, Union
from polars_expr_transformer.process.models import IfFunc, Func, TempFunc
from polars_expr_transformer.process.tree import build_hierarchy
from polars_expr_transformer.process.tokenize import tokenize
from polars_expr_transformer.process.standardize import standardize_tokens
from polars_expr_transformer.process.process_inline import parse_inline_functions
from polars_expr_transformer.process.preprocess import preprocess
import polars as pl


def finalize_hierarchy(hierarchical_formula: Union[Func | TempFunc | IfFunc]):
    """
     Finalize the hierarchical formula by ensuring it has a valid structure.

     This function checks the hierarchical formula structure and ensures that
     a TempFunc has exactly one argument. If the TempFunc has more than one
     argument, an exception is raised. If it has exactly one argument, that
     argument is returned as the final hierarchical formula. If it has no arguments,
     an exception is raised.

     Args:
         hierarchical_formula: The hierarchical formula to finalize, which can be a Func, TempFunc, or IfFunc.

     Returns:
         The finalized hierarchical formula, which is either a Func or an IfFunc.

     Raises:
         Exception: If the TempFunc contains zero or more than one argument.
     """
    if isinstance(hierarchical_formula, TempFunc) and len(hierarchical_formula.args) == 1:
        return hierarchical_formula.args[0]
    elif isinstance(hierarchical_formula, TempFunc) and len(hierarchical_formula.args) > 1:
        raise Exception('Expected only one argument')
    elif isinstance(hierarchical_formula, TempFunc) and len(hierarchical_formula.args) == 0:
        raise Exception('Expected at least one argument')
    return hierarchical_formula


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
    formula = preprocess(func_str)
    tokens = tokenize(formula)
    standardized_tokens = standardize_tokens(tokens)
    hierarchical_formula = build_hierarchy(standardized_tokens)
    parse_inline_functions(hierarchical_formula)
    return finalize_hierarchy(hierarchical_formula)


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
