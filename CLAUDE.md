# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A library that compiles SQL/Tableau-like string expressions (e.g. `if [age] > 30 then "Senior" else "Junior" endif`) into native Polars expressions. The public entry points are in `polars_expr_transformer/__init__.py`:

- `simple_function_to_expr(str) -> pl.Expr` — the main API; returns a live Polars expression.
- `build_func(str) -> Func` — returns the parsed expression tree for inspection/debugging.
- `to_polars_code(str) -> str` / `to_flowframe_code(str) -> str` — emit Polars (`pl.`) or FlowFrame (`ff.`) Python code as a string.
- `get_all_expressions()` / `get_expression_overview()` — introspect the function registry.

## Commands

```bash
poetry install                       # core deps only (no polars-ds)
poetry install -E similarity         # include polars-ds for string_similarity()
poetry run pytest tests/ -v          # full suite (matches CI; ~400 tests)
poetry run pytest tests/test_tokenize.py -v          # one file
poetry run pytest tests/test_tokenize.py::test_name  # one test
python generate_docs.py              # regenerate docs/ function reference from docstrings
```

CI (`.github/workflows/ci-cd.yml`) runs `pytest tests/` on Python 3.10–3.13 and publishes to PyPI on `v*` tags (the tag must match `version` in `pyproject.toml`). flake8 is a dev dependency but has no config file and is not run in CI.

## Architecture

### Compilation pipeline (`process/`)

`build_func()` in `process/polars_expr_transformer.py` runs the string through these stages in order:

1. **`preprocess.py`** — normalizes the string, strips `//` comments, and rewrites `if/then/elseif/else/endif` keywords into `$if$`/`$then$`/`$else$`/`$endif$` markers.
2. **`tokenize.py`** — scans the string **in reverse** into raw token strings, respecting quoted strings and `[bracketed column]` names.
3. **`token_classifier.py`** — wraps each token in a `Classifier` and assigns a `val_type` (number, string, operator, function, column, case_when, etc.).
4. **`hierarchy_builder.py`** — builds the tree of `Func` / `IfFunc` / `ConditionVal` / `TempFunc` nodes (handles brackets, commas, and conditional structure).
5. **`process_inline.py`** — converts infix operators (`+`, `>`, `and`, …) into nested function calls using `PRECEDENCE` from `configs/settings.py`.
6. **`finalize_hierarchy()` / `remove_temp_funcs()`** (in `polars_expr_transformer.py`) — strip the transient `TempFunc` scaffolding nodes from the tree.

After the tree is built, output is produced two independent ways:
- **Live evaluation**: each node's `get_pl_func()` builds the actual `pl.Expr`.
- **Code generation**: each node's `to_polars_code(prefix)` emits a code string (`code_gen.py`).

### Tree node model (`process/models.py`)

- `Classifier` — a leaf token (literal, column, operator name).
- `Func` — a function call (`func_ref` + `args`); `pl.col`/`pl.lit` are modeled as `Func`s too.
- `IfFunc` + `ConditionVal` — conditionals; `IfFunc` holds a list of `ConditionVal` (when/then pairs) plus `else_val`.
- `TempFunc` — transient parsing scaffold, removed before evaluation.

Every node implements three parallel methods: `get_pl_func()` (live expr), `get_readable_pl_function()` (human-readable string), and `to_polars_code(prefix="pl")` (code string). When changing tree behavior, keep these three in sync.

### Function registry — how to add a function

User-facing functions live in `funcs/{string,math,date,logic,type_conversions,special}_functions.py`. **To add one, just define a function in the appropriate module** — `funcs/__init__.py` merges every module's `__dict__` into `all_functions`, and `configs/settings.py` exposes it to the parser via the `funcs` dict (which also injects `pl.col`, `pl.lit`, operators, and `aliases`). No registration step is needed.

Conventions for new functions:
- Underscore-prefixed names (`_in`, `__negative`) are internal and excluded from the public overview.
- The **docstring is API**: `function_overview.py` and `generate_docs.py` surface it in the playground/reference. Follow the existing format (one-line summary, a concrete `For example, ...` line, `Parameters:`, `Returns:`).
- Parameter **type annotations drive literal handling**. Types from `funcs/utils.py` (`PlStringType`, `PlIntType`, `PlNumericType`) and `Func._standardize_args` decide when a raw value is auto-wrapped in `pl.lit()`. Annotate params accordingly.
- For code generation to emit correct chained Polars (not a generic `func(args)` fallback that warns), add a mapping entry to `FUNCTION_CODE_GEN` / `OPERATOR_SYMBOLS` in `code_gen.py`.

### Optional `polars-ds` dependency

`string_similarity()` is the only feature requiring `polars-ds`, exposed as the `similarity` extra. polars-ds ships **no WebAssembly wheel**, so it must stay optional for the browser/Pyodide playground. The import is lazy inside `__get_similarity_method` in `string_functions.py` — never import `polars_ds` at module top level.

### Errors

- `ExpressionSyntaxError` (subclass of `ValueError`, in `exceptions.py`) carries `expression`/`position`/`hint` and renders a caret-pointer message. `expression_validator.py` produces these during parsing.
- `PolarsCodeGenError` is raised when `to_polars_code(validate=True)` generates code that fails `eval`.

### Docs / playground

`docs/` is a Pyodide-based browser playground deployed to GitHub Pages (`.github/workflows/deploy-docs.yml`), which runs `generate_docs.py` and builds the wheel into `docs/assets/wheel/`. Because the playground runs in-browser, keep the core import path free of native-only dependencies.
