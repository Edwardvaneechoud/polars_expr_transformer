from typing import List, Union
from polars_expr_transformer.process.models import IfFunc, Func, TempFunc
from polars_expr_transformer.process.tree import build_hierarchy
from polars_expr_transformer.process.tokenize import tokenize
from polars_expr_transformer.process.standardize import standardize_tokens
from polars_expr_transformer.process.process_inline import parse_inline_functions
from polars_expr_transformer.process.preprocess import preprocess
import polars as pl


def finalize_hierarchy(hierarchical_formula: Union[Func | TempFunc | IfFunc]):
    if isinstance(hierarchical_formula, TempFunc) and len(hierarchical_formula.args) == 1:
        return hierarchical_formula.args[0]
    elif isinstance(hierarchical_formula, TempFunc) and len(hierarchical_formula.args) > 1:
        raise Exception('Expected only one argument')
    elif isinstance(hierarchical_formula, TempFunc) and len(hierarchical_formula.args) == 0:
        raise Exception('Expected at least one argument')
    return hierarchical_formula


def build_func(func_str: str = 'concat("1", "2")') -> Func:
    formula = preprocess(func_str)
    tokens = tokenize(formula)
    standardized_tokens = standardize_tokens(tokens)
    hierarchical_formula = build_hierarchy(standardized_tokens)
    parse_inline_functions(hierarchical_formula)
    return finalize_hierarchy(hierarchical_formula)


def simple_function_to_expr(func_str: str) -> pl.expr.Expr:
    func = build_func(func_str)
    return func.get_pl_func()
