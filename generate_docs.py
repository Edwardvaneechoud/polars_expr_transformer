"""Generate the function reference data for the documentation site.

Introspects every public expression function exposed by
``polars_expr_transformer`` (the same set returned by
``get_expression_overview``), parses the docstrings into structured data
and writes ``docs/assets/functions.json``. The static documentation /
playground site in ``docs/`` renders its searchable function reference
from that file, so the docs are always generated from the source of
truth: the function docstrings themselves.

Usage:
    python generate_docs.py [output_path]
"""

import inspect
import json
import re
import sys
import tomllib
from pathlib import Path

from polars_expr_transformer.function_overview import MODULE_CATEGORIES

CATEGORY_LABELS = {
    "logic": "Logic & Nulls",
    "string": "String",
    "math": "Math",
    "special": "Special",
    "date": "Date & Time",
    "type_conversions": "Type Conversion",
}

# Human friendly names for the union type aliases used in annotations.
TYPE_ALIASES = {
    "PlStringType": "text",
    "PlIntType": "integer",
    "PlNumericType": "number",
    "Any": "any",
    "Expr": "expression",
    "pl.Expr": "expression",
    "str": "text",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "callable": "function",
}

EXAMPLE_RE = re.compile(
    r"For example,?\s+(?P<call>.+?)\s+(?:would|will|might|returns?)\s+"
    r"(?:return\s+)?(?P<result>.+?)"
    r"(?:\s+when\s+(?P<context>.+?))?\.?\s*$",
    re.IGNORECASE,
)

# Docstring examples use bare literals (e.g. year("2023-05-15")), which the
# library does not auto-parse into temporals, and a `null` literal that the
# language does not have. For those functions a curated, runnable expression
# against the playground's sample datasets is provided instead. The dataset
# key tells the playground which dataset to switch to.
TRY_OVERRIDES = {
    "year": ("year([hire_date])", "employees"),
    "month": ("month([hire_date])", "employees"),
    "day": ("day([hire_date])", "employees"),
    "week": ("week([hire_date])", "employees"),
    "weekday": ("weekday([hire_date])", "employees"),
    "dayofweek": ("dayofweek([hire_date])", "employees"),
    "dayofyear": ("dayofyear([hire_date])", "employees"),
    "quarter": ("quarter([hire_date])", "employees"),
    "add_days": ("add_days([hire_date], 30)", "employees"),
    "add_weeks": ("add_weeks([hire_date], 2)", "employees"),
    "add_months": ("add_months([hire_date], 6)", "employees"),
    "add_years": ("add_years([hire_date], 1)", "employees"),
    "date_diff_days": ("date_diff_days(today(), [hire_date])", "employees"),
    "start_of_month": ("start_of_month([hire_date])", "employees"),
    "end_of_month": ("end_of_month([hire_date])", "employees"),
    "format_date": ('format_date([hire_date], "%B %d, %Y")', "employees"),
    "date_trim": ('date_trim([order_date], "day")', "orders"),
    "date_truncate": ('date_truncate([order_date], "1d")', "orders"),
    "hour": ("hour([order_date])", "orders"),
    "minute": ("minute([order_date])", "orders"),
    "second": ("second([order_date])", "orders"),
    "add_hours": ("add_hours([order_date], 3)", "orders"),
    "add_minutes": ("add_minutes([order_date], 30)", "orders"),
    "add_seconds": ("add_seconds([order_date], 45)", "orders"),
    "datetime_diff_seconds": (
        "datetime_diff_seconds(now(), [order_date])",
        "orders",
    ),
    "datetime_diff_nanoseconds": (
        "datetime_diff_nanoseconds(now(), [order_date])",
        "orders",
    ),
    "sin": ("sin([discount])", "orders"),
    "cos": ("cos([discount])", "orders"),
    "tan": ("tan([discount])", "orders"),
    "asin": ("asin([discount])", "orders"),
    "acos": ("acos([discount])", "orders"),
    "atan": ("atan([discount])", "orders"),
    "coalesce": ('coalesce([email], "no email")', "employees"),
    "ifnull": ("ifnull([discount], 0)", "orders"),
    "nvl": ("nvl([discount], 0)", "orders"),
    "is_empty": ("is_empty([email])", "employees"),
    "is_not_empty": ("is_not_empty([email])", "employees"),
    "nullif": ('nullif([status], "cancelled")', "orders"),
    "to_boolean": ("to_boolean(1)", "employees"),
    "to_datetime": ('to_datetime("2023-05-15 14:30:00")', "employees"),
}

# Functions that work locally but not in the browser playground (they need
# native extensions that have no WebAssembly build).
BROWSER_UNSUPPORTED = {"string_similarity"}

# Minimal stand-ins for the playground datasets, used to verify that every
# "try it" expression actually runs. Columns mirror docs/assets/app.js.
_VALIDATION_FRAMES = {
    "employees": {
        "first_name": ("str", ["John", "Jane"]),
        "last_name": ("str", ["Doe", "Smith"]),
        "age": ("int", [30, 25]),
        "salary": ("float", [50000.0, 60000.0]),
        "department": ("str", ["Sales", "Engineering"]),
        "hire_date": ("date", ["2021-03-15", "2019-07-01"]),
        "email": ("str", ["john.doe@acme.com", None]),
    },
    "orders": {
        "order_id": ("str", ["ORD-0001", "ORD-0002"]),
        "product": ("str", ["Laptop Pro", "Wireless Mouse"]),
        "category": ("str", ["Computers", "Accessories"]),
        "price": ("float", [1299.99, 24.5]),
        "quantity": ("int", [1, 4]),
        "discount": ("float", [0.1, None]),
        "order_date": ("datetime", ["2024-01-15 10:30:00", "2024-01-17 14:05:12"]),
        "status": ("str", ["shipped", "pending"]),
    },
    "events": {
        "event": ("str", ["Kickoff Meeting", "Tech Conference"]),
        "city": ("str", ["Amsterdam", "Lisbon"]),
        "start": ("datetime", ["2024-05-06 09:00:00", "2024-06-18 08:00:00"]),
        "end": ("datetime", ["2024-05-06 10:30:00", "2024-06-20 18:00:00"]),
        "attendees": ("int", [12, None]),
    },
}


def _validation_df(dataset_key: str):
    import polars as pl

    series = []
    for name, (dtype, values) in _VALIDATION_FRAMES[dataset_key].items():
        s = pl.Series(name, values)
        if dtype == "int":
            s = s.cast(pl.Int64)
        elif dtype == "float":
            s = s.cast(pl.Float64)
        elif dtype == "date":
            s = s.str.to_date("%Y-%m-%d")
        elif dtype == "datetime":
            s = s.str.to_datetime("%Y-%m-%d %H:%M:%S")
        series.append(s)
    return pl.DataFrame(series)


def _expression_runs(expression: str, dataset_key: str) -> bool:
    from polars_expr_transformer import simple_function_to_expr

    try:
        expr = simple_function_to_expr(expression)
        _validation_df(dataset_key).with_columns(expr.alias("result"))
        return True
    except Exception:
        return False


def _find_runnable_dataset(expression: str):
    for dataset_key in _VALIDATION_FRAMES:
        if _expression_runs(expression, dataset_key):
            return dataset_key
    return None


def _clean_annotation(annotation) -> str:
    if annotation is inspect.Parameter.empty:
        return "any"
    if isinstance(annotation, str):
        raw = annotation
    elif getattr(annotation, "__name__", None):
        raw = annotation.__name__
    else:
        raw = str(annotation)
    raw = raw.replace("polars.expr.expr.", "pl.").replace("typing.", "")
    parts = re.split(r"\s*\|\s*", raw)
    cleaned = [TYPE_ALIASES.get(p.strip(), p.strip()) for p in parts]
    seen, result = set(), []
    for part in cleaned:
        if part not in seen:
            seen.add(part)
            result.append(part)
    return " | ".join(result)


def _build_signature(func) -> str:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return f"{func.__name__}(...)"
    params = []
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            params.append(f"*{param.name}")
        elif param.default is not inspect.Parameter.empty:
            params.append(f"{param.name}={param.default!r}")
        else:
            params.append(param.name)
    return f"{func.__name__}({', '.join(params)})"


def _parse_docstring(doc: str) -> dict:
    """Split a function docstring into description, example, params, returns.

    The docstrings follow a fixed layout::

        Short description, possibly spanning multiple lines.

        For example, func("a", "b") would return "ab".

        Parameters:
        - name: description

        Returns:
        - description
    """
    result = {
        "description": "",
        "example_text": None,
        "example_call": None,
        "example_result": None,
        "example_context": None,
        "params": [],
        "returns": None,
    }
    if not doc:
        return result

    lines = [line.strip() for line in inspect.cleandoc(doc).splitlines()]
    section = "description"
    description_lines, example_lines = [], []

    for line in lines:
        lowered = line.lower()
        if lowered.startswith("parameters:") or lowered.startswith("parameters"):
            if lowered.rstrip(":") == "parameters":
                section = "params"
                continue
        if lowered.rstrip(":") == "returns":
            section = "returns"
            continue
        if lowered.startswith("for example") and section == "description":
            section = "example"

        if not line:
            if section == "example":
                section = "description"
            continue

        if section == "description":
            description_lines.append(line)
        elif section == "example":
            example_lines.append(line)
        elif section == "params" and line.startswith("-"):
            text = line.lstrip("- ")
            name, _, desc = text.partition(":")
            result["params"].append(
                {"name": name.strip(), "description": desc.strip()}
            )
        elif section == "returns" and line.startswith("-"):
            result["returns"] = line.lstrip("- ").strip()

    result["description"] = " ".join(description_lines)
    if example_lines:
        example_text = " ".join(example_lines)
        result["example_text"] = example_text
        match = EXAMPLE_RE.match(example_text)
        if match:
            result["example_call"] = match.group("call").strip()
            result["example_result"] = match.group("result").strip()
            if match.group("context"):
                result["example_context"] = match.group("context").strip()
    return result


def _parameters_for(func, parsed_params) -> list:
    documented = {p["name"]: p["description"] for p in parsed_params}
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return parsed_params
    params = []
    for param in sig.parameters.values():
        name = param.name
        display = f"*{name}" if param.kind == inspect.Parameter.VAR_POSITIONAL else name
        entry = {
            "name": display,
            "type": _clean_annotation(param.annotation),
            "description": documented.get(name, documented.get(display, "")),
        }
        if param.default is not inspect.Parameter.empty:
            entry["default"] = repr(param.default)
        params.append(entry)
    return params


def _package_version() -> str:
    pyproject = Path(__file__).parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    return data["tool"]["poetry"]["version"]


def build_reference() -> dict:
    version = _package_version()
    categories = []
    total = 0
    for key, module in MODULE_CATEGORIES.items():
        functions = []
        for name, func in sorted(module.__dict__.items()):
            if (
                not callable(func)
                or name.startswith("_")
                or inspect.getmodule(func) is not module
            ):
                continue
            parsed = _parse_docstring(func.__doc__ or "")

            # Prefer the docstring example as the playground "try it"
            # expression; fall back to a curated override when the example
            # does not run against any of the sample datasets.
            try_expression, try_dataset = None, None
            if name not in BROWSER_UNSUPPORTED:
                if parsed["example_call"]:
                    dataset = _find_runnable_dataset(parsed["example_call"])
                    if dataset:
                        try_expression = parsed["example_call"]
                        try_dataset = dataset
                if try_expression is None and name in TRY_OVERRIDES:
                    expression, dataset = TRY_OVERRIDES[name]
                    if _expression_runs(expression, dataset):
                        try_expression, try_dataset = expression, dataset

            functions.append(
                {
                    "name": name,
                    "signature": _build_signature(func),
                    "description": parsed["description"],
                    "example_text": parsed["example_text"],
                    "example_call": parsed["example_call"],
                    "example_result": parsed["example_result"],
                    "example_context": parsed["example_context"],
                    "try_expression": try_expression,
                    "try_dataset": try_dataset,
                    "parameters": _parameters_for(func, parsed["params"]),
                    "returns": parsed["returns"],
                }
            )
        if functions:
            categories.append(
                {
                    "key": key,
                    "label": CATEGORY_LABELS.get(key, key.title()),
                    "functions": functions,
                }
            )
            total += len(functions)
    return {
        "version": version,
        "wheel": f"wheel/polars_expr_transformer-{version}-py3-none-any.whl",
        "total_functions": total,
        "categories": categories,
    }


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parent / "docs" / "assets" / "functions.json"
    )
    reference = build_reference()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reference, indent=2) + "\n")
    print(
        f"Wrote {reference['total_functions']} functions in "
        f"{len(reference['categories'])} categories to {output}"
    )


if __name__ == "__main__":
    main()
