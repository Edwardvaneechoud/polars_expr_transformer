"""
Code generation module for converting AST nodes to native Polars Python code strings.

This module provides mappings and helpers to transform the internal expression tree
representation into valid, executable Polars Python code.
"""


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


def _method_chain(method):
    """Create a code gen function for simple method chaining: {0}.method()"""
    return lambda args: f"{args[0]}.{method}"


def _method_chain_with_args(method):
    """Create a code gen function for method chaining with args: {0}.method({1}, {2}, ...)"""
    def gen(args):
        receiver = args[0]
        rest = ", ".join(args[1:])
        return f"{receiver}.{method}({rest})"
    return gen


def _top_level_list(func_name):
    """Create a code gen function for top-level calls with a list arg: pl.func([a, b, c])"""
    def gen(args):
        items = ", ".join(args)
        return f"{func_name}([{items}])"
    return gen


def _template(tmpl):
    """Create a code gen function from a format string template using {0}, {1}, etc."""
    def gen(args):
        return tmpl.format(*args)
    return gen


# Maps function names to code generation functions.
# Each function takes a list of argument code strings and returns the Polars code string.
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
    "concat": _top_level_list("pl.concat_str"),
    "starts_with": _template("{0}.str.starts_with({1})"),
    "ends_with": _template("{0}.str.ends_with({1})"),
    "reverse": _method_chain("str.reverse()"),
    "find_position": _template("{0}.str.find({1}, literal=True, strict=False)"),
    "pad_left": _template("{0}.str.pad_start({1}, {2})"),
    "pad_right": _template("{0}.str.pad_end({1}, {2})"),
    "count_match": _template("{0}.str.count_matches({1})"),
    "split": _template("{0}.str.split({1})"),
    "contains": _template("{0}.str.contains({1})"),
    "repeat": _template("pl.concat_str([{0}] * {1})"),

    # Math functions
    "abs": _method_chain("abs()"),
    "round": _method_chain_with_args("round"),
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
    "add_months": _template("{0}.dt.offset_by(pl.concat_str([{1}.cast(pl.Utf8), pl.lit(\"mo\")]))"),
    "date_diff_days": _template("({0} - {1}).dt.total_days()"),
    "datetime_diff_seconds": _template("({0} - {1}).dt.total_seconds()"),
    "datetime_diff_nanoseconds": _template("({0} - {1}).dt.total_nanoseconds()"),
    "format_date": _template("{0}.dt.to_string({1})"),
    "end_of_month": _method_chain("dt.month_end()"),
    "start_of_month": _method_chain("dt.month_start()"),
    "date_truncate": _template("{0}.dt.truncate({1})"),
    "date_trim": _template("{0}.dt.truncate({1})"),
    "now": lambda args: "pl.lit(datetime.now())",
    "today": lambda args: "pl.lit(datetime.today())",

    # Logic functions
    "equals": _template("{0}.eq({1})"),
    "is_empty": _method_chain("is_null()"),
    "is_not_empty": _method_chain("is_not_null()"),
    "coalesce": _top_level_list("pl.coalesce"),
    "ifnull": lambda args: f"pl.coalesce([{args[0]}, {args[1]}])",
    "nvl": lambda args: f"pl.coalesce([{args[0]}, {args[1]}])",
    "nullif": _template("pl.when({0}.eq({1})).then(pl.lit(None)).otherwise({0})"),
    "between": _template("{0}.is_between({1}, {2})"),
    "greatest": _top_level_list("pl.max_horizontal"),
    "least": _top_level_list("pl.min_horizontal"),
    "_not": _method_chain("not_()"),
    "not": _method_chain("not_()"),
    "_in": _template("{1}.str.contains({0})"),
    "is_string": lambda args: f"pl.lit({args[0]}.dtype == pl.Utf8)",

    # Type conversions
    "to_string": _method_chain("cast(pl.Utf8)"),
    "to_integer": _method_chain("cast(pl.Int64)"),
    "to_float": _method_chain("cast(pl.Float64)"),
    "to_number": _method_chain("cast(pl.Float64)"),
    "to_boolean": _method_chain("cast(pl.Boolean)"),
    "to_date": _method_chain_with_args("str.to_date"),
    "to_datetime": _method_chain_with_args("str.to_datetime"),
    "to_decimal": lambda args: f"{args[0]}.cast(pl.Float64).round({args[1]})" if len(args) > 1 else f"{args[0]}.cast(pl.Float64)",

    # Special
    "random_int": _template("pl.int_range({0}, {1}).sample(n=pl.len(), with_replacement=True)"),
    "string_similarity": _template("pds.str_leven({0}, {1}, return_sim=True)"),
}


def format_pl_literal(val_str, val_type):
    """Format a raw value string as a Polars literal code string.

    Args:
        val_str: The raw value string (e.g. '"test"', '42', 'true')
        val_type: The classified type ('string', 'number', 'boolean')

    Returns:
        A string like 'pl.lit("test")', 'pl.lit(42)', 'pl.lit(True)'
    """
    if val_type == 'boolean':
        py_val = "True" if val_str.lower() == 'true' else "False"
        return f"pl.lit({py_val})"
    elif val_type == 'number':
        return f"pl.lit({val_str})"
    elif val_type == 'string':
        return f"pl.lit({val_str})"
    else:
        return f"pl.lit({val_str})"
