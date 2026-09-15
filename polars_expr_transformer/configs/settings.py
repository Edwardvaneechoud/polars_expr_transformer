import polars as pl
from polars_expr_transformer.funcs import all_functions
from polars_expr_transformer.funcs.logic_functions import does_not_equal
from polars_expr_transformer.funcs.logic_functions import _in
from polars_expr_transformer.funcs.logic_functions import _not_in
operators = {  # get your data out of your code...
    "+": "pl.Expr.add",
    "-": "pl.Expr.sub",
    "*": "pl.Expr.mul",
    "%": "pl.Expr.mod",
    "/": "pl.Expr.truediv",
    "<": "pl.Expr.lt",
    "<=": "pl.Expr.le",
    ">": "pl.Expr.gt",
    ">=": "pl.Expr.ge",
    "==": "pl.Expr.eq",
    "=": "pl.Expr.eq",
    "&": "pl.Expr.and_",
    '|': "pl.Expr.or_",
    '!=': "does_not_equal",
    'and': "pl.Expr.and_",
    'or': "pl.Expr.or_",
    'in': "_in",
    'not in': "_not_in",
    'is_null': "pl.Expr.is_null",
}

aliases = {
    'not': '_not',
}

# Membership operators lower to a different implementation than the one `operators` names
# above, but only when their right-hand side is a parenthesised list: `in`/`not in` keep
# their substring meaning against a plain value.
MEMBERSHIP_OPERATORS = {
    'in': '_is_in',
    'not in': '_is_not_in',
}

_OPERATOR_ROOTS = {"pl": pl, "does_not_equal": does_not_equal, "_in": _in, "_not_in": _not_in}


def _resolve_operator(dotted_name: str):
    """Look up an operator implementation by its dotted name."""
    root, *attrs = dotted_name.split(".")
    obj = _OPERATOR_ROOTS[root]
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


operators_mappings = {v: _resolve_operator(v) for v in operators.values()}
single_word_operators = [op for op in operators if ' ' not in op]
all_split_vals = set(['(', ')', '$if$', '$endif$', '$else$', '$then$','$elseif$', ',', ''] + single_word_operators)
all_split_vals_reversed = [v[::-1] for v in all_split_vals]
funcs = {f'{k}': v for k,v in all_functions.items()}
funcs['pl.col'] = pl.col
funcs['pl.lit'] = pl.lit
funcs.update(operators_mappings)
for alias, ref in aliases.items():
    funcs[alias] = funcs[ref]

PRECEDENCE = {
    'or': 1,
    'and': 2,
    '>': 3, '<': 3, '>=': 3, '<=': 3, '==': 3, '!=': 3, 'in': 3, 'not in': 3,
    '+': 4, '-': 4,
    '*': 5, '/': 5
}
