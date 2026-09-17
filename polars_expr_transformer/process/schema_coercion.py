"""Insert the text-to-date parse that a date function's argument needs.

A date function lowers to ``.dt.<something>()``, which Polars refuses on a
String column. The function bodies in ``funcs/date_functions.py`` each guard
that with ``create_fix_date_col``, but the guard only fires for a raw Python
value: a column reference is already a ``pl.Expr`` and goes straight through,
and the parser wraps a bare literal in ``pl.lit`` before the call (see
``Func._standardize_args``), so through the parser the guard never fires at all.

The parse is inserted here, into the tree, rather than inside the functions, so
that ``get_pl_func``, ``get_readable_pl_function`` and ``to_polars_code`` all
see it and cannot disagree about what was parsed.

An argument is only known to be text in two cases, and nothing else is touched:

* a string literal written in the formula, whose type is known outright;
* a column, when the caller supplied a schema saying that column is String.

Without a schema, columns keep today's behaviour, since
``simple_function_to_expr`` builds an expression against no frame and cannot
know a column's type.
"""

import inspect
from collections.abc import Mapping
from typing import Any, Optional

import polars as pl

from polars_expr_transformer.configs.settings import funcs
from polars_expr_transformer.funcs.utils import PlDateType
from polars_expr_transformer.process.models import (
    Classifier,
    Func,
    IfFunc,
    TempFunc,
)

COERCION_FUNC = "_parse_date_text"

_date_params_cache: dict[str, tuple[int, ...]] = {}


def normalize_schema(schema: Any) -> Optional[dict[str, Any]]:
    """Accept any mapping of column name to dtype, such as ``df.schema``."""
    if schema is None:
        return None
    if isinstance(schema, Mapping):
        return dict(schema)
    raise TypeError(
        "schema must be a mapping of column name to Polars dtype, for example "
        f"df.schema or lf.collect_schema(); got {type(schema).__name__}."
    )


def _date_param_positions(func_name: str) -> tuple[int, ...]:
    """Positions of the parameters a function expects a date in.

    Read from the signature rather than a table, so a date function added later
    takes part simply by annotating its parameter ``PlDateType``.
    """
    if func_name not in _date_params_cache:
        func = funcs.get(func_name)
        try:
            params = inspect.signature(func).parameters.values()
        except (TypeError, ValueError):
            params = ()
        _date_params_cache[func_name] = tuple(
            i for i, p in enumerate(params) if p.annotation == PlDateType
        )
    return _date_params_cache[func_name]


def _func_name(node: Any) -> Optional[str]:
    if isinstance(node, Func) and isinstance(node.func_ref, Classifier):
        return node.func_ref.val
    return None


def _string_literal_node(node: Any) -> bool:
    """Whether a node is a text literal written in the formula.

    Both shapes count: the bare classifier the parser produces, and the
    ``pl.lit`` wrapper ``Func._standardize_args`` may already have put round it.
    """
    if isinstance(node, Classifier):
        return node.val_type == "string"
    if _func_name(node) == "pl.lit" and len(node.args) == 1:
        return _string_literal_node(node.args[0])
    return False


def _string_column_node(node: Any, schema: Optional[dict[str, Any]]) -> bool:
    """Whether a node is a column the schema says holds text."""
    if schema is None or _func_name(node) != "pl.col":
        return False
    if len(node.args) != 1 or not isinstance(node.args[0], Classifier):
        return False
    name = node.args[0].val.strip('"').strip("'")
    if name not in schema:
        return False
    return schema[name] == pl.String


def _needs_parse(node: Any, schema: Optional[dict[str, Any]]) -> bool:
    if _func_name(node) == COERCION_FUNC:
        return False
    return _string_literal_node(node) or _string_column_node(node, schema)


def _wrap(node: Any) -> Func:
    parsed = Func(Classifier(COERCION_FUNC))
    parsed.parent = node.parent
    parsed.add_arg(node)
    return parsed


def coerce_date_arguments(node: Any, schema: Any = None) -> Any:
    """Wrap every text argument of a date function in the date-text parse.

    The tree is rewritten in place and the root is returned for convenience;
    only arguments are ever replaced, so the root keeps its identity.
    """
    return _walk(node, normalize_schema(schema))


def _walk(node: Any, schema: Optional[dict[str, Any]]) -> Any:
    if isinstance(node, (Func, TempFunc)):
        for arg in node.args:
            _walk(arg, schema)
        name = _func_name(node)
        if name is not None:
            for i in _date_param_positions(name):
                if i < len(node.args) and _needs_parse(node.args[i], schema):
                    node.args[i] = _wrap(node.args[i])
    elif isinstance(node, IfFunc):
        for condition in node.conditions:
            _walk(condition.condition, schema)
            _walk(condition.val, schema)
        if node.else_val is not None:
            _walk(node.else_val, schema)
    return node
