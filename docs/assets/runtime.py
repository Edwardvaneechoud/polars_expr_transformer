"""Python runtime for the in-browser playground.

This file is loaded into Pyodide by ``app.js`` after the
``polars-expr-transformer`` wheel has been installed. It exposes a single
entry point, ``run_expression``, which receives a JSON payload describing
the dataset and the expression, and returns a JSON document with the
result table plus the generated Polars / FlowFrame code.

The module is plain Python on purpose: the exact same file can be executed
against a local interpreter for testing.
"""

import difflib
import json
import math
import sys
import types

# ``polars_ds`` is a native extension that is not available in the
# Pyodide/WASM runtime. It is only needed by ``string_similarity``, so a
# stub module that raises a friendly error on use is installed before the
# package is imported.
if "polars_ds" not in sys.modules:
    _pds = types.ModuleType("polars_ds")

    def _pds_missing(*args, **kwargs):
        raise NotImplementedError(
            "string_similarity requires the polars-ds extension, "
            "which is not available in the browser runtime"
        )

    for _name in (
        "str_leven",
        "str_jaro",
        "str_jw",
        "str_d_leven",
        "str_hamming",
        "str_fuzz",
        "str_osa",
        "str_sorensen_dice",
        "str_jaccard",
    ):
        setattr(_pds, _name, _pds_missing)
    sys.modules["polars_ds"] = _pds

import polars as pl  # noqa: E402
from polars_expr_transformer import simple_function_to_expr  # noqa: E402
from polars_expr_transformer.process.polars_expr_transformer import (  # noqa: E402
    to_flowframe_code,
    to_polars_code,
)

RESULT_COLUMN = "result"
MAX_ERROR_LENGTH = 600


def _build_df(spec):
    """Build a DataFrame from a {"columns": [{name, dtype, values}]} spec."""
    series = []
    for col in spec["columns"]:
        s = pl.Series(col["name"], col["values"])
        dtype = col.get("dtype", "str")
        if dtype == "int":
            s = s.cast(pl.Int64)
        elif dtype == "float":
            s = s.cast(pl.Float64)
        elif dtype == "bool":
            s = s.cast(pl.Boolean)
        elif dtype == "date":
            s = s.str.to_date("%Y-%m-%d")
        elif dtype == "datetime":
            s = s.str.to_datetime("%Y-%m-%d %H:%M:%S")
        series.append(s)
    return pl.DataFrame(series)


def _fmt(value):
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        value = round(value, 6)
        if value == int(value):
            return str(int(value)) + ".0"
        return str(value)
    return str(value)


def _suggest_column(name, available):
    """Best replacement candidate for a misspelled column name, or None."""
    lower_map = {c.lower(): c for c in available}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    close = difflib.get_close_matches(name, available, n=1, cutoff=0.6)
    return close[0] if close else None


def _missing_columns_error(missing, available):
    if len(missing) == 1:
        lines = [f"Column [{missing[0]}] does not exist in this dataset."]
    else:
        listed = ", ".join(f"[{name}]" for name in missing)
        lines = [f"Columns {listed} do not exist in this dataset."]
    suggestions = []
    for name in missing:
        candidate = _suggest_column(name, available)
        if candidate:
            suggestions.append(f"[{candidate}]")
    if suggestions:
        unique = list(dict.fromkeys(suggestions))
        lines[0] += f" Did you mean {' or '.join(unique)}?"
    lines.append("Available columns: " + ", ".join(f"[{c}]" for c in available))
    return "\n".join(lines)


def _clean_error(exc):
    message = str(exc).strip() or type(exc).__name__
    # Polars appends a resolved-plan dump that is noise in this context.
    marker = "Resolved plan until failure"
    if marker in message:
        message = message.split(marker)[0].strip()
    if len(message) > MAX_ERROR_LENGTH:
        message = message[:MAX_ERROR_LENGTH] + " …"
    return message


def run_expression(payload_json: str) -> str:
    """Evaluate an expression against a dataset spec; both sides are JSON."""
    payload = json.loads(payload_json)
    expr_str = payload["expression"]
    out = {"ok": False, "error": None, "stage": None}

    try:
        expr = simple_function_to_expr(expr_str)
    except Exception as exc:  # parse failure
        out["error"] = _clean_error(exc)
        out["stage"] = "parse"
        return json.dumps(out)

    # Code generation is independent of execution: the generated source is
    # shown even when the expression cannot run on the chosen dataset.
    try:
        out["polars_code"] = to_polars_code(expr_str, validate=False)
        out["flowframe_code"] = to_flowframe_code(expr_str, validate=False)
    except Exception as exc:
        out["codegen_error"] = _clean_error(exc)

    try:
        df = _build_df(payload["dataset"])
        # Report missing columns up front: polars' own ColumnNotFoundError
        # message is just the bare column name, which reads as noise.
        missing = [c for c in expr.meta.root_names() if c not in df.columns]
        if missing:
            out["error"] = _missing_columns_error(missing, df.columns)
            out["stage"] = "execute"
            return json.dumps(out)
        result = df.with_columns(expr.alias(RESULT_COLUMN))
        out["columns"] = [
            {"name": name, "dtype": str(dtype)}
            for name, dtype in result.schema.items()
        ]
        out["rows"] = [[_fmt(v) for v in row] for row in result.iter_rows()]
        out["ok"] = True
    except Exception as exc:  # execution failure (code was still generated)
        out["error"] = _clean_error(exc)
        out["stage"] = "execute"

    return json.dumps(out)
