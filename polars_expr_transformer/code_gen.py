"""
Code generation module for converting AST nodes to native Polars Python code strings.

This module provides mappings and helpers to transform the internal expression tree
representation into valid, executable Polars Python code.

The ``prefix`` parameter (default ``"pl"``) controls the library qualifier used
in the generated code.  Pass ``"ff"`` to emit FlowFrame code instead.
"""

import ast

from polars_expr_transformer.funcs.utils import DATE_TEXT_FORMATS
from polars_expr_transformer.string_literals import (
    is_string_literal,
    parse_string_literal,
    render_string_literal,
)

# Reverse mapping from internal operator names to Python operator symbols
OPERATOR_SYMBOLS = {
    "pl.Expr.add": "+",
    "pl.Expr.sub": "-",
    "pl.Expr.mul": "*",
    "pl.Expr.truediv": "/",
    "pl.Expr.mod": "%",
    "pl.Expr.lt": "<",
    "pl.Expr.le": "<=",
    "pl.Expr.gt": ">",
    "pl.Expr.ge": ">=",
    "pl.Expr.eq": "==",
    "pl.Expr.and_": "&",
    "pl.Expr.or_": "|",
    "does_not_equal": "!=",
}


_ATOMIC_EXPR_NODES = (
    ast.Call,
    ast.Attribute,
    ast.Name,
    ast.Constant,
    ast.Subscript,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
)


def parenthesize(code: str) -> str:
    """Wrap rendered code in parentheses unless it is already an atomic expression.

    Generated code is assembled by attaching method calls and infix operators to
    already-rendered sub-expressions, and both bind tighter than a bare infix
    expression.  Without this, ``pl.col("a") / pl.col("b")`` followed by
    ``.log()`` attaches the method to ``pl.col("b")`` alone, which is valid
    Python computing the wrong thing.

    Atomicity is decided by parsing the rendered code, so this stays correct for
    any code generator in ``FUNCTION_CODE_GEN``, including ones that render
    infix output themselves.  Parentheses around an already-atomic expression
    would be harmless but are skipped to keep generated code readable.
    """
    try:
        node = ast.parse(code, mode="eval").body
    except SyntaxError:
        return f"({code})"
    if isinstance(node, _ATOMIC_EXPR_NODES):
        return code
    return f"({code})"


def _method_chain(method):
    """Create a code gen function for simple method chaining: {0}.method()"""

    def gen(args, prefix="pl"):
        m = method.replace("pl.", f"{prefix}.") if prefix != "pl" else method
        return f"{args[0]}.{m}"

    return gen


def _method_chain_with_args(method):
    """Create a code gen function for method chaining with args: {0}.method({1}, {2}, ...)"""

    def gen(args, prefix="pl"):
        receiver = args[0]
        rest = ", ".join(args[1:])
        m = method.replace("pl.", f"{prefix}.") if prefix != "pl" else method
        return f"{receiver}.{m}({rest})"

    return gen


def _method_chain_with_literal_args(method):
    """Like ``_method_chain_with_args``, for methods taking plain Python values.

    ``str.to_date``, ``dt.to_string`` and friends take a format as a real
    ``str``, not an expression, so the ``prefix.lit(...)`` the generator puts
    round every literal is unwrapped here.
    """

    def gen(args, prefix="pl"):
        receiver = args[0]
        rest = ", ".join(strip_pl_lit(a, prefix) for a in args[1:])
        m = method.replace("pl.", f"{prefix}.") if prefix != "pl" else method
        return f"{receiver}.{m}({rest})"

    return gen


def _top_level_list(func_name_suffix):
    """Create a code gen function for top-level calls with a list arg: prefix.func([a, b, c])

    ``func_name_suffix`` should be the part after ``pl.``, e.g. ``"concat_str"``.
    """

    def gen(args, prefix="pl"):
        items = ", ".join(args)
        return f"{prefix}.{func_name_suffix}([{items}])"

    return gen


def _template(tmpl):
    """Create a code gen function from a format string template using {0}, {1}, etc.

    Any occurrence of ``pl.`` in the template is replaced at call time with
    the active prefix.
    """

    def gen(args, prefix="pl"):
        rendered = tmpl.format(*args)
        if prefix != "pl":
            rendered = rendered.replace("pl.", f"{prefix}.")
        return rendered

    return gen


def strip_pl_lit(code_str: str, prefix: str = "pl") -> str:
    """Extract the raw value from a prefix.lit() wrapper.

    Examples:
        'pl.lit(2)' -> '2'
        'pl.lit("x")' -> '"x"'
        '42' -> '42'  (no-op if not wrapped)
    """
    wrapper = f"{prefix}.lit("
    if code_str.startswith(wrapper) and code_str.endswith(")"):
        return code_str[len(wrapper) : -1]
    return code_str


def _hex_digest(algorithm):
    """Create a code gen function for a per-element hashlib digest.

    The generated code needs ``hashlib`` in scope, the way ``now()`` needs
    ``datetime``.
    """

    def gen(args, prefix="pl"):
        return (
            f"{args[0]}.cast({prefix}.Utf8).map_elements("
            f'lambda v: hashlib.{algorithm}(v.encode("utf-8")).hexdigest(), '
            f"return_dtype={prefix}.Utf8)"
        )

    return gen


def _encoding_arg(args, prefix, fixed):
    """Resolve the encoding name for encode/decode code gen.

    ``str.encode``/``str.decode`` take a plain Python string, so a
    ``prefix.lit(...)`` wrapper around the argument is unwrapped here.
    """
    if fixed is not None:
        return f'"{fixed}"'
    if len(args) > 1:
        return strip_pl_lit(args[1], prefix)
    return '"base64"'


def _encode(fixed=None):
    """Create a code gen function for encode()/base64_encode()/hex_encode()."""

    def gen(args, prefix="pl"):
        encoding = _encoding_arg(args, prefix, fixed)
        return f"{args[0]}.cast({prefix}.Utf8).str.encode({encoding})"

    return gen


def _decode(fixed=None):
    """Create a code gen function for decode()/base64_decode()/hex_decode()."""

    def gen(args, prefix="pl"):
        encoding = _encoding_arg(args, prefix, fixed)
        return (
            f"{args[0]}.str.decode({encoding}, strict=False)"
            f".cast({prefix}.Utf8)"
        )

    return gen


def _parse_date_text_gen(args, prefix="pl"):
    """Render the automatic date-text parse.

    Both this and ``_parse_date_text`` read their formats from
    ``DATE_TEXT_FORMATS``, so the emitted code cannot drift from what the live
    expression does.
    """
    rungs = ", ".join(
        f'{args[0]}.str.to_datetime("{fmt}", strict=False)' for fmt in DATE_TEXT_FORMATS
    )
    return f"{prefix}.coalesce([{rungs}])"


# Maps function names to code generation functions.
# Each function takes a list of argument code strings and an optional prefix,
# and returns the generated code string.
FUNCTION_CODE_GEN = {
    # String functions
    "uppercase": _method_chain("str.to_uppercase()"),
    "lowercase": _method_chain("str.to_lowercase()"),
    "titlecase": _method_chain("str.to_titlecase()"),
    "length": _method_chain("str.len_chars()"),
    "trim": _method_chain("str.strip_chars()"),
    "left_trim": _method_chain("str.strip_chars_start()"),
    "right_trim": _method_chain("str.strip_chars_end()"),
    "left": _template("{0}.str.slice(0, {1})"),
    "right": _template("{0}.str.slice(-{1})"),
    "mid": _template("{0}.str.slice({1}, {2})"),
    "substring": _template("{0}.str.slice({1}, {2})"),
    "replace": _template("{0}.str.replace_many({1}, {2})"),
    "concat": _top_level_list("concat_str"),
    "starts_with": _template("{0}.str.starts_with({1})"),
    "ends_with": _template("{0}.str.ends_with({1})"),
    "reverse": _method_chain("str.reverse()"),
    "find_position": _template("{0}.str.find({1}, literal=True, strict=False)"),
    "pad_left": _template("{0}.str.pad_start({1}, {2})"),
    "pad_right": _template("{0}.str.pad_end({1}, {2})"),
    "count_match": _template("{0}.str.count_matches({1})"),
    "split": _template("{0}.str.split({1})"),
    "contains": _template("{0}.str.contains({1})"),
    "repeat": lambda args, prefix="pl": (
        f"{prefix}.concat_str([{args[0]}] * {strip_pl_lit(args[1], prefix)})"
    ),
    # Math functions
    "abs": _method_chain("abs()"),
    "round": lambda args, prefix="pl": (
        f"{args[0]}.round({strip_pl_lit(args[1], prefix)})"
        if len(args) > 1
        else f"{args[0]}.round(0)"
    ),
    "ceil": _method_chain("ceil()"),
    "floor": _method_chain("floor()"),
    "sqrt": _method_chain("sqrt()"),
    "log": _method_chain("log()"),
    "log10": _method_chain("log(base=10)"),
    "log2": _method_chain("log(base=2)"),
    "exp": _method_chain("exp()"),
    "power": _method_chain_with_args("pow"),
    "pow": _method_chain_with_args("pow"),
    "mod": _method_chain_with_args("mod"),
    "sign": _method_chain("sign()"),
    "negation": _method_chain("neg()"),
    "sin": _method_chain("sin()"),
    "cos": _method_chain("cos()"),
    "tan": _method_chain("tan()"),
    "asin": _method_chain("arcsin()"),
    "acos": _method_chain("arccos()"),
    "atan": _method_chain("arctan()"),
    "tanh": _method_chain("tanh()"),
    # Date functions
    "_parse_date_text": _parse_date_text_gen,
    "year": _method_chain("dt.year()"),
    "month": _method_chain("dt.month()"),
    "day": _method_chain("dt.day()"),
    "hour": _method_chain("dt.hour()"),
    "minute": _method_chain("dt.minute()"),
    "second": _method_chain("dt.second()"),
    "week": _method_chain("dt.week()"),
    "weekday": _method_chain("dt.weekday()"),
    "dayofweek": _method_chain("dt.weekday()"),
    "quarter": _method_chain("dt.quarter()"),
    "dayofyear": _method_chain("dt.ordinal_day()"),
    "add_days": _template("{0} + pl.duration(days={1})"),
    "add_weeks": _template("{0} + pl.duration(weeks={1})"),
    "add_years": _template("{0} + pl.duration(days={1} * 365)"),
    "add_hours": _template("{0} + pl.duration(hours={1})"),
    "add_minutes": _template("{0} + pl.duration(minutes={1})"),
    "add_seconds": _template("{0} + pl.duration(seconds={1})"),
    "add_months": _template(
        '{0}.dt.offset_by(pl.concat_str([{1}.cast(pl.Utf8), pl.lit("mo")]))'
    ),
    "date_diff_days": _template("({0} - {1}).dt.total_days()"),
    "datetime_diff_seconds": _template("({0} - {1}).dt.total_seconds()"),
    "datetime_diff_nanoseconds": _template("({0} - {1}).dt.total_nanoseconds()"),
    "format_date": _method_chain_with_literal_args("dt.to_string"),
    "end_of_month": _method_chain("dt.month_end()"),
    "start_of_month": _method_chain("dt.month_start()"),
    "date_truncate": _method_chain_with_literal_args("dt.truncate"),
    "date_trim": _method_chain_with_literal_args("dt.truncate"),
    "now": lambda args, prefix="pl": f"{prefix}.lit(datetime.datetime.now())",
    "today": lambda args, prefix="pl": f"{prefix}.lit(datetime.datetime.today())",
    # Logic functions
    "equals": _template("{0}.eq({1})"),
    "is_empty": _method_chain("is_null()"),
    "is_not_empty": _method_chain("is_not_null()"),
    "coalesce": _top_level_list("coalesce"),
    "ifnull": lambda args, prefix="pl": f"{prefix}.coalesce([{args[0]}, {args[1]}])",
    "nvl": lambda args, prefix="pl": f"{prefix}.coalesce([{args[0]}, {args[1]}])",
    "nullif": _template("pl.when({0}.eq({1})).then(pl.lit(None)).otherwise({0})"),
    "between": _template("{0}.is_between({1}, {2})"),
    "greatest": _top_level_list("max_horizontal"),
    "least": _top_level_list("min_horizontal"),
    "_not": _method_chain("not_()"),
    "not": _method_chain("not_()"),
    "_in": _template("{1}.str.contains({0})"),
    "_not_in": _template("{1}.str.contains({0}).not_()"),
    # The _list node renders its own members, as a list or as an imploded concat_list.
    "_is_in": _template("{0}.is_in({1})"),
    "_is_not_in": _template("{0}.is_in({1}).not_()"),
    "is_string": lambda args, prefix="pl": (
        f"{prefix}.lit({args[0]}.dtype == {prefix}.Utf8)"
    ),
    # Type conversions
    "to_string": _method_chain("cast(pl.Utf8)"),
    "to_integer": _method_chain("cast(pl.Int64)"),
    "to_float": _method_chain("cast(pl.Float64)"),
    "to_number": _method_chain("cast(pl.Float64)"),
    "to_boolean": _method_chain("cast(pl.Boolean)"),
    "to_date": _method_chain_with_literal_args("str.to_date"),
    "to_datetime": _method_chain_with_literal_args("str.to_datetime"),
    "to_decimal": lambda args, prefix="pl": (
        f"{args[0]}.cast({prefix}.Float64).round({strip_pl_lit(args[1], prefix)})"
        if len(args) > 1
        else f"{args[0]}.cast({prefix}.Float64)"
    ),
    # Special
    "random_int": _template(
        "pl.int_range({0}, {1}).sample(n=pl.len(), with_replacement=True)"
    ),
    # Hashing
    "hash": _method_chain("hash()"),
    "md5": _hex_digest("md5"),
    "sha1": _hex_digest("sha1"),
    "sha256": _hex_digest("sha256"),
    "sha512": _hex_digest("sha512"),
    # Encoding
    "encode": _encode(),
    "decode": _decode(),
    "base64_encode": _encode("base64"),
    "base64_decode": _decode("base64"),
    "hex_encode": _encode("hex"),
    "hex_decode": _decode("hex"),
}


def format_pl_literal(val_str, val_type, prefix="pl"):
    """Format a raw value string as a literal code string.

    Args:
        val_str: The raw value string (e.g. '"test"', '42', 'true')
        val_type: The classified type ('string', 'number', 'boolean')
        prefix: The library prefix to use (default 'pl')

    Returns:
        A string like 'pl.lit("test")', 'pl.lit(42)', 'pl.lit(True)'
    """
    if val_type == "boolean":
        py_val = "True" if val_str.lower() == "true" else "False"
        return f"{prefix}.lit({py_val})"
    elif val_type == "null":
        return f"{prefix}.lit(None)"
    elif val_type == "string" and is_string_literal(val_str):
        # Re-render from the parsed value so the emitted literal is escaped code,
        # never a verbatim copy of untrusted formula text.
        return f"{prefix}.lit({render_string_literal(parse_string_literal(val_str))})"
    else:
        return f"{prefix}.lit({val_str})"
