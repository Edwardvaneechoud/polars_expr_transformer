"""
Polars Expression Interceptor.

Provides proxy objects that mimic the Polars API but record operations
and generate equivalent string DSL expressions.

Example:
    >>> from polars_expr_transformer.interceptor import PolarsInterceptor
    >>> pi = PolarsInterceptor()
    >>>
    >>> # Write Polars-like code:
    >>> expr = pi.col("name").str.to_uppercase() + pi.lit(" - ") + pi.col("city")
    >>> print(expr.to_dsl())
    'concat(uppercase([name]), " - ", [city])'
    >>>
    >>> # Convert to a real Polars expression:
    >>> pl_expr = expr.to_polars_expr()
"""

from __future__ import annotations

from typing import Any, Optional

import polars as pl

from polars_expr_transformer.main_module import simple_function_to_expr


class DslNode:
    """Base class for DSL AST nodes."""

    def to_dsl(self) -> str:
        raise NotImplementedError


class ColNode(DslNode):
    """Represents a column reference: [col_name]"""

    def __init__(self, name: str):
        self.name = name

    def to_dsl(self) -> str:
        return f"[{self.name}]"


class LitNode(DslNode):
    """Represents a literal value."""

    def __init__(self, value: Any):
        self.value = value

    def to_dsl(self) -> str:
        if isinstance(self.value, str):
            escaped = self.value.replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        if self.value is None:
            return "null"
        return str(self.value)


class FuncNode(DslNode):
    """Represents a function call: func_name(arg1, arg2, ...)"""

    def __init__(self, func_name: str, args: list[DslNode]):
        self.func_name = func_name
        self.args = args

    def to_dsl(self) -> str:
        args_str = ", ".join(a.to_dsl() for a in self.args)
        return f"{self.func_name}({args_str})"


class OpNode(DslNode):
    """Represents an infix operator: left op right"""

    def __init__(self, op: str, left: DslNode, right: DslNode):
        self.op = op
        self.left = left
        self.right = right

    # Precedence for grouping
    PRECEDENCE = {
        "or": 1,
        "and": 2,
        ">": 3, "<": 3, ">=": 3, "<=": 3, "=": 3, "!=": 3,
        "+": 4, "-": 4,
        "*": 5, "/": 5, "%": 5,
    }

    def _precedence(self) -> int:
        return self.PRECEDENCE.get(self.op, 10)

    def _wrap(self, child: DslNode) -> str:
        dsl = child.to_dsl()
        if isinstance(child, OpNode) and child._precedence() < self._precedence():
            return f"({dsl})"
        return dsl

    def to_dsl(self) -> str:
        return f"{self._wrap(self.left)} {self.op} {self._wrap(self.right)}"


class IfNode(DslNode):
    """Represents an if/then/else expression."""

    def __init__(self, conditions: list[tuple[DslNode, DslNode]], else_val: Optional[DslNode] = None):
        self.conditions = conditions  # list of (condition, value)
        self.else_val = else_val

    def to_dsl(self) -> str:
        parts = []
        for i, (cond, val) in enumerate(self.conditions):
            keyword = "if" if i == 0 else "elseif"
            parts.append(f"{keyword} {cond.to_dsl()} then {val.to_dsl()}")
        if self.else_val is not None:
            parts.append(f"else {self.else_val.to_dsl()}")
        parts.append("endif")
        return " ".join(parts)


class NegNode(DslNode):
    """Represents negation: -expr"""

    def __init__(self, child: DslNode):
        self.child = child

    def to_dsl(self) -> str:
        child_dsl = self.child.to_dsl()
        if isinstance(self.child, OpNode):
            return f"-({child_dsl})"
        return f"-{child_dsl}"


class NotNode(DslNode):
    """Represents logical not: _not(expr)"""

    def __init__(self, child: DslNode):
        self.child = child

    def to_dsl(self) -> str:
        return f"_not({self.child.to_dsl()})"


# ──────────────────────────────────────────────
# Proxy objects that mimic the Polars API
# ──────────────────────────────────────────────

class ExprProxy:
    """Proxy that mimics pl.Expr, recording operations as DSL nodes."""

    def __init__(self, node: DslNode):
        self._node = node

    @property
    def node(self) -> DslNode:
        return self._node

    def to_dsl(self) -> str:
        """Return the string DSL expression."""
        return self._node.to_dsl()

    def to_polars_expr(self) -> pl.Expr:
        """Convert the recorded DSL back into a real Polars expression."""
        return simple_function_to_expr(self.to_dsl())

    def alias(self, name: str) -> ExprProxy:
        """alias is a no-op for DSL generation (just passes through)."""
        return self

    def __repr__(self) -> str:
        return f"ExprProxy({self.to_dsl()})"

    # ── Arithmetic operators ──

    def __add__(self, other: Any) -> ExprProxy:
        return _binop("+", self, other)

    def __radd__(self, other: Any) -> ExprProxy:
        return _binop("+", other, self)

    def __sub__(self, other: Any) -> ExprProxy:
        return _binop("-", self, other)

    def __rsub__(self, other: Any) -> ExprProxy:
        return _binop("-", other, self)

    def __mul__(self, other: Any) -> ExprProxy:
        return _binop("*", self, other)

    def __rmul__(self, other: Any) -> ExprProxy:
        return _binop("*", other, self)

    def __truediv__(self, other: Any) -> ExprProxy:
        return _binop("/", self, other)

    def __rtruediv__(self, other: Any) -> ExprProxy:
        return _binop("/", other, self)

    def __mod__(self, other: Any) -> ExprProxy:
        return _binop("%", self, other)

    def __neg__(self) -> ExprProxy:
        return ExprProxy(NegNode(self._node))

    # ── Comparison operators ──

    def __eq__(self, other: Any) -> ExprProxy:
        return _binop("=", self, other)

    def __ne__(self, other: Any) -> ExprProxy:
        return _binop("!=", self, other)

    def __lt__(self, other: Any) -> ExprProxy:
        return _binop("<", self, other)

    def __le__(self, other: Any) -> ExprProxy:
        return _binop("<=", self, other)

    def __gt__(self, other: Any) -> ExprProxy:
        return _binop(">", self, other)

    def __ge__(self, other: Any) -> ExprProxy:
        return _binop(">=", self, other)

    # ── Logical operators ──

    def __and__(self, other: Any) -> ExprProxy:
        return _binop("and", self, other)

    def __or__(self, other: Any) -> ExprProxy:
        return _binop("or", self, other)

    def __invert__(self) -> ExprProxy:
        return ExprProxy(NotNode(self._node))

    def not_(self) -> ExprProxy:
        return ExprProxy(NotNode(self._node))

    # ── Polars Expr-style methods ──

    def abs(self) -> ExprProxy:
        return ExprProxy(FuncNode("abs", [self._node]))

    def sqrt(self) -> ExprProxy:
        return ExprProxy(FuncNode("sqrt", [self._node]))

    def log(self, base: Optional[float] = None) -> ExprProxy:
        if base == 10:
            return ExprProxy(FuncNode("log10", [self._node]))
        if base == 2:
            return ExprProxy(FuncNode("log2", [self._node]))
        return ExprProxy(FuncNode("log", [self._node]))

    def exp(self) -> ExprProxy:
        return ExprProxy(FuncNode("exp", [self._node]))

    def ceil(self) -> ExprProxy:
        return ExprProxy(FuncNode("ceil", [self._node]))

    def floor(self) -> ExprProxy:
        return ExprProxy(FuncNode("floor", [self._node]))

    def round(self, decimals: int = 0) -> ExprProxy:
        return ExprProxy(FuncNode("round", [self._node, LitNode(decimals)]))

    def pow(self, exponent: Any) -> ExprProxy:
        return ExprProxy(FuncNode("power", [self._node, _to_node(exponent)]))

    def sign(self) -> ExprProxy:
        return ExprProxy(FuncNode("sign", [self._node]))

    def sin(self) -> ExprProxy:
        return ExprProxy(FuncNode("sin", [self._node]))

    def cos(self) -> ExprProxy:
        return ExprProxy(FuncNode("cos", [self._node]))

    def tan(self) -> ExprProxy:
        return ExprProxy(FuncNode("tan", [self._node]))

    def arcsin(self) -> ExprProxy:
        return ExprProxy(FuncNode("asin", [self._node]))

    def arccos(self) -> ExprProxy:
        return ExprProxy(FuncNode("acos", [self._node]))

    def arctan(self) -> ExprProxy:
        return ExprProxy(FuncNode("atan", [self._node]))

    def tanh(self) -> ExprProxy:
        return ExprProxy(FuncNode("tanh", [self._node]))

    def neg(self) -> ExprProxy:
        return ExprProxy(FuncNode("negation", [self._node]))

    def mod(self, other: Any) -> ExprProxy:
        return ExprProxy(FuncNode("mod", [self._node, _to_node(other)]))

    # ── Null handling ──

    def is_null(self) -> ExprProxy:
        return ExprProxy(FuncNode("is_empty", [self._node]))

    def is_not_null(self) -> ExprProxy:
        return ExprProxy(FuncNode("is_not_empty", [self._node]))

    # ── Cast ──

    def cast(self, dtype: Any) -> ExprProxy:
        type_map = {
            pl.Utf8: "to_string",
            pl.String: "to_string",
            pl.Int64: "to_integer",
            pl.Int32: "to_integer",
            pl.Float64: "to_float",
            pl.Float32: "to_float",
            pl.Boolean: "to_boolean",
            pl.Date: "to_date",
            pl.Datetime: "to_datetime",
        }
        func_name = type_map.get(dtype, "to_string")
        return ExprProxy(FuncNode(func_name, [self._node]))

    # ── Namespace accessors ──

    @property
    def str(self) -> StrNamespaceProxy:
        return StrNamespaceProxy(self._node)

    @property
    def dt(self) -> DtNamespaceProxy:
        return DtNamespaceProxy(self._node)


class StrNamespaceProxy:
    """Proxy for pl.Expr.str.* methods."""

    def __init__(self, node: DslNode):
        self._node = node

    def to_uppercase(self) -> ExprProxy:
        return ExprProxy(FuncNode("uppercase", [self._node]))

    def to_lowercase(self) -> ExprProxy:
        return ExprProxy(FuncNode("lowercase", [self._node]))

    def to_titlecase(self) -> ExprProxy:
        return ExprProxy(FuncNode("titlecase", [self._node]))

    def len_chars(self) -> ExprProxy:
        return ExprProxy(FuncNode("length", [self._node]))

    def lengths(self) -> ExprProxy:
        return self.len_chars()

    def slice(self, offset: Any, length: Optional[Any] = None) -> ExprProxy:
        if isinstance(offset, int) and offset == 0 and length is not None:
            return ExprProxy(FuncNode("left", [self._node, _to_node(length)]))
        if length is not None:
            return ExprProxy(FuncNode("mid", [self._node, _to_node(offset), _to_node(length)]))
        return ExprProxy(FuncNode("right", [self._node, _to_node(offset)]))

    def starts_with(self, prefix: Any) -> ExprProxy:
        return ExprProxy(FuncNode("starts_with", [self._node, _to_node(prefix)]))

    def ends_with(self, suffix: Any) -> ExprProxy:
        return ExprProxy(FuncNode("ends_with", [self._node, _to_node(suffix)]))

    def contains(self, pattern: Any) -> ExprProxy:
        return ExprProxy(FuncNode("contains", [self._node, _to_node(pattern)]))

    def replace_many(self, old: Any, new: Any) -> ExprProxy:
        return ExprProxy(FuncNode("replace", [self._node, _to_node(old), _to_node(new)]))

    def replace(self, old: Any, new: Any) -> ExprProxy:
        return ExprProxy(FuncNode("replace", [self._node, _to_node(old), _to_node(new)]))

    def strip_chars_start(self) -> ExprProxy:
        return ExprProxy(FuncNode("left_trim", [self._node]))

    def strip_chars_end(self) -> ExprProxy:
        return ExprProxy(FuncNode("right_trim", [self._node]))

    def strip_chars(self) -> ExprProxy:
        return ExprProxy(FuncNode("trim", [self._node]))

    def reverse(self) -> ExprProxy:
        return ExprProxy(FuncNode("reverse", [self._node]))

    def split(self, by: str) -> ExprProxy:
        return ExprProxy(FuncNode("split", [self._node, _to_node(by)]))

    def pad_start(self, length: int, fill_char: str = " ") -> ExprProxy:
        return ExprProxy(FuncNode("pad_left", [self._node, LitNode(length), LitNode(fill_char)]))

    def pad_end(self, length: int, fill_char: str = " ") -> ExprProxy:
        return ExprProxy(FuncNode("pad_right", [self._node, LitNode(length), LitNode(fill_char)]))

    def count_matches(self, pattern: Any) -> ExprProxy:
        return ExprProxy(FuncNode("count_match", [self._node, _to_node(pattern)]))

    def find(self, pattern: Any) -> ExprProxy:
        return ExprProxy(FuncNode("find_position", [self._node, _to_node(pattern)]))

    def to_date(self, fmt: str = "%Y-%m-%d") -> ExprProxy:
        return ExprProxy(FuncNode("to_date", [self._node, LitNode(fmt)]))

    def to_datetime(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> ExprProxy:
        return ExprProxy(FuncNode("to_datetime", [self._node, LitNode(fmt)]))


class DtNamespaceProxy:
    """Proxy for pl.Expr.dt.* methods."""

    def __init__(self, node: DslNode):
        self._node = node

    def year(self) -> ExprProxy:
        return ExprProxy(FuncNode("year", [self._node]))

    def month(self) -> ExprProxy:
        return ExprProxy(FuncNode("month", [self._node]))

    def day(self) -> ExprProxy:
        return ExprProxy(FuncNode("day", [self._node]))

    def hour(self) -> ExprProxy:
        return ExprProxy(FuncNode("hour", [self._node]))

    def minute(self) -> ExprProxy:
        return ExprProxy(FuncNode("minute", [self._node]))

    def second(self) -> ExprProxy:
        return ExprProxy(FuncNode("second", [self._node]))

    def week(self) -> ExprProxy:
        return ExprProxy(FuncNode("week", [self._node]))

    def weekday(self) -> ExprProxy:
        return ExprProxy(FuncNode("weekday", [self._node]))

    def quarter(self) -> ExprProxy:
        return ExprProxy(FuncNode("quarter", [self._node]))

    def ordinal_day(self) -> ExprProxy:
        return ExprProxy(FuncNode("dayofyear", [self._node]))

    def month_start(self) -> ExprProxy:
        return ExprProxy(FuncNode("start_of_month", [self._node]))

    def month_end(self) -> ExprProxy:
        return ExprProxy(FuncNode("end_of_month", [self._node]))

    def truncate(self, every: str) -> ExprProxy:
        return ExprProxy(FuncNode("date_truncate", [self._node, LitNode(every)]))

    def to_string(self, fmt: str = "%Y-%m-%d") -> ExprProxy:
        return ExprProxy(FuncNode("format_date", [self._node, LitNode(fmt)]))

    def total_days(self) -> ExprProxy:
        return ExprProxy(FuncNode("date_diff_days", [self._node]))

    def total_seconds(self) -> ExprProxy:
        return ExprProxy(FuncNode("datetime_diff_seconds", [self._node]))


# ──────────────────────────────────────────────
# The main interceptor class
# ──────────────────────────────────────────────

class PolarsInterceptor:
    """
    A drop-in proxy for common ``pl.*`` calls that records operations
    and emits the equivalent string DSL understood by polars_expr_transformer.

    Usage:
        >>> pi = PolarsInterceptor()
        >>> expr = pi.col("price") * pi.col("quantity")
        >>> expr.to_dsl()
        '[price] * [quantity]'

        >>> expr = pi.col("name").str.to_uppercase()
        >>> expr.to_dsl()
        'uppercase([name])'

        >>> expr = pi.col("birth_date").dt.year()
        >>> expr.to_dsl()
        'year([birth_date])'

        >>> expr = pi.when(pi.col("age") >= 30).then(pi.lit("Senior")).otherwise(pi.lit("Junior"))
        >>> expr.to_dsl()
        'if [age] >= 30 then "Senior" else "Junior" endif'
    """

    def col(self, name: str) -> ExprProxy:
        """Equivalent to pl.col(name)."""
        return ExprProxy(ColNode(name))

    def lit(self, value: Any) -> ExprProxy:
        """Equivalent to pl.lit(value)."""
        return ExprProxy(LitNode(value))

    def when(self, condition: ExprProxy) -> WhenProxy:
        """Equivalent to pl.when(condition)."""
        return WhenProxy(condition._node)

    def coalesce(self, *exprs: ExprProxy) -> ExprProxy:
        """Equivalent to pl.coalesce(...)."""
        return ExprProxy(FuncNode("coalesce", [_to_node(e) for e in exprs]))

    def concat_str(self, *exprs: ExprProxy) -> ExprProxy:
        """Equivalent to pl.concat_str(...)."""
        args = []
        for e in exprs:
            if isinstance(e, (list, tuple)):
                args.extend(_to_node(x) for x in e)
            else:
                args.append(_to_node(e))
        return ExprProxy(FuncNode("concat", args))

    def max_horizontal(self, *exprs: ExprProxy) -> ExprProxy:
        """Equivalent to pl.max_horizontal(...)."""
        return ExprProxy(FuncNode("greatest", [_to_node(e) for e in exprs]))

    def min_horizontal(self, *exprs: ExprProxy) -> ExprProxy:
        """Equivalent to pl.min_horizontal(...)."""
        return ExprProxy(FuncNode("least", [_to_node(e) for e in exprs]))


class WhenProxy:
    """Proxy for pl.when(...).then(...).otherwise(...)."""

    def __init__(self, condition: DslNode):
        self._conditions: list[tuple[DslNode, DslNode]] = []
        self._current_condition = condition

    def then(self, value: Any) -> WhenThenProxy:
        self._conditions.append((self._current_condition, _to_node(value)))
        return WhenThenProxy(self._conditions)


class WhenThenProxy:
    """Proxy for the state after .then(), before .otherwise() or .when()."""

    def __init__(self, conditions: list[tuple[DslNode, DslNode]]):
        self._conditions = conditions

    def when(self, condition: ExprProxy) -> WhenProxy:
        proxy = WhenProxy(condition._node)
        proxy._conditions = self._conditions
        return proxy

    def otherwise(self, value: Any) -> ExprProxy:
        return ExprProxy(IfNode(self._conditions, _to_node(value)))

    def to_dsl(self) -> str:
        return IfNode(self._conditions).to_dsl()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _to_node(value: Any) -> DslNode:
    """Convert a raw value or ExprProxy to a DslNode."""
    if isinstance(value, ExprProxy):
        return value._node
    if isinstance(value, DslNode):
        return value
    return LitNode(value)


def _binop(op: str, left: Any, right: Any) -> ExprProxy:
    """Create an ExprProxy for a binary operation."""
    # Special case: + between strings/columns means concat
    if op == "+" and _is_string_context(left, right):
        return ExprProxy(FuncNode("concat", [_to_node(left), _to_node(right)]))
    return ExprProxy(OpNode(op, _to_node(left), _to_node(right)))


def _is_string_context(left: Any, right: Any) -> bool:
    """Heuristic: if either side is a plain string literal, treat + as concat."""
    if isinstance(left, str) or isinstance(right, str):
        return True
    if isinstance(left, ExprProxy) and isinstance(left._node, LitNode) and isinstance(left._node.value, str):
        return True
    if isinstance(right, ExprProxy) and isinstance(right._node, LitNode) and isinstance(right._node.value, str):
        return True
    return False
