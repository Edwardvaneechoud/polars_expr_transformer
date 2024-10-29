from typing import List, Dict, Optional
from polars_expr_transformer.schemas import ExpressionRef, ExpressionsOverview
from polars_expr_transformer.funcs import (
    logic_functions,
    string_functions,
    math_functions,
    special_funcs,
    date_functions
)

MODULE_CATEGORIES = {
    'logic': logic_functions,
    'string': string_functions,
    'math': math_functions,
    'special': special_funcs,
    'date': date_functions
}

_available_expressions: Optional[List[ExpressionsOverview]] = None


def get_expression_overview() -> List[ExpressionsOverview]:
    """Get overview of all expressions organized by category."""
    global _available_expressions

    if _available_expressions is None:
        _available_expressions = [
            ExpressionsOverview(
                expression_type=category,
                expressions=[
                    ExpressionRef(
                        name=name,
                        doc=func.__doc__
                    )
                    for name, func in module.__dict__.items()
                    if callable(func) and not name.startswith('_')
                ]
            )
            for category, module in MODULE_CATEGORIES.items()
        ]

    return _available_expressions


def get_all_expressions() -> List[str]:
    """Get list of all available expression names."""
    return [
        name
        for module in MODULE_CATEGORIES.values()
        for name, func in module.__dict__.items()
        if callable(func) and not name.startswith('_')
    ]