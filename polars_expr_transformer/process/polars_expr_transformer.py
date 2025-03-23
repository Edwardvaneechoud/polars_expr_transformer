from typing import List, Union
from polars_expr_transformer.process.models import IfFunc, Func, TempFunc, Classifier
from polars_expr_transformer.process.hierarchy_builder import build_hierarchy
from polars_expr_transformer.process.tokenize import tokenize
from polars_expr_transformer.process.token_classifier import classify_tokens
from polars_expr_transformer.process.process_inline import parse_inline_functions
from polars_expr_transformer.process.post_process import post_process_hierarchical_formula
from polars_expr_transformer.process.preprocess import preprocess
import polars as pl


def finalize_hierarchy(obj):
    """
    Recursively remove all TempFunc instances from the hierarchical formula.

    Args:
        obj: The object to process (Func, TempFunc, IfFunc, or Classifier)

    Returns:
        The processed object with all TempFunc instances replaced by their arguments
    """
    # Base case: If it's not a TempFunc or doesn't need recursion, return it
    if isinstance(obj, Classifier) or obj is None:
        return obj

    # Handle TempFunc
    if isinstance(obj, TempFunc):
        if len(obj.args) != 1:
            raise Exception(f"Expected exactly one argument in TempFunc, got {len(obj.args)}")
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
        for i in range(len(obj.args)):
            obj.args[i] = finalize_hierarchy(obj.args[i])

    # Process IfFunc objects
    elif isinstance(obj, IfFunc):
        for cond in obj.conditions:
            cond.condition = finalize_hierarchy(cond.condition)
            cond.val = finalize_hierarchy(cond.val)
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
    tokenizes it, standardizes the tokens, builds a hierarchical structure from
    the tokens, parses any inline functions, and finally returns the resulting Func object.

    Args:
        func_str: The string representation of the function to build. Defaults to 'concat("1", "2")'.

    Returns:
        The resulting Func object built from the function string.
    """
    # func_str =   'if ((1222*[a])> 1222) then true else false endif'
    formula = preprocess(func_str)
    raw_tokens = tokenize(formula)
    tokens = classify_tokens(raw_tokens)
    hierarchical_formula = build_hierarchy(tokens)
    print(hierarchical_formula)
    # hierarchical_formula = finalize_hierarchy(hierarchical_formula)
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
