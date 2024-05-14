import polars as pl
from polars_expr_transformer.funcs import all_functions
from polars_expr_transformer.funcs.logic_functions import does_not_equal
from polars_expr_transformer.funcs.logic_functions import _in
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
    'is_null': "pl.Expr.is_null",
}

aliases = {
    'not': '_not',
}


operators_mappings = {v: eval(v) for v in operators.values()}
all_split_vals = set(['(', ')', '$if$', '$endif$', '$else$', '$then$','$elseif$', ',', ''] + list(operators)+list(operators))
all_split_vals_reversed = [v[::-1] for v in all_split_vals]
funcs = {f'{k}': v for k,v in all_functions.items()}
funcs['pl.col'] = pl.col
funcs['pl.lit'] = pl.lit
funcs.update(operators_mappings)
for alias, ref in aliases.items():
    funcs[alias] = funcs[ref]


PRECEDENCE = {'+': 2, '-': 2, '*': 3, '/': 3, '|': -1, '&': -1, 'and': -1, 'or': -1, 'in': -1}