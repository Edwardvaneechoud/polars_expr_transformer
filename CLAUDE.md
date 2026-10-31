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

CI (`.github/workflows/ci-cd.yml`) runs `pytest tests/` on Python 3.10–3.14 against the locked Polars 1.x, plus one Python 3.13 job on Polars 2 (behaviour that differs between the two must be tested in both), and publishes to PyPI on `v*` tags (the tag must match `version` in `pyproject.toml`). flake8 is a dev dependency but has no config file and is not run in CI.

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

User-facing functions live in `funcs/{string,math,date,logic,type_conversions,special,hashing,encoding}_functions.py`. **To add one, just define a function in the appropriate module** — `funcs/__init__.py` merges every module's `__dict__` into `all_functions`, and `configs/settings.py` exposes it to the parser via the `funcs` dict (which also injects `pl.col`, `pl.lit`, operators, and `aliases`). No registration step is needed.

Conventions for new functions:
- Underscore-prefixed names (`_in`, `__negative`) are internal and excluded from the public overview.
- The **docstring is API**: `function_overview.py` and `generate_docs.py` surface it in the playground/reference. Follow the existing format (one-line summary, a concrete `For example, ...` line, `Parameters:`, `Returns:`).
- Parameter **type annotations drive literal handling**. Types from `funcs/utils.py` (`PlStringType`, `PlIntType`, `PlNumericType`) and `Func._standardize_args` decide when a raw value is auto-wrapped in `pl.lit()`. Annotate params accordingly.
- For code generation to emit correct chained Polars (not a generic `func(args)` fallback that warns), add a mapping entry to `FUNCTION_CODE_GEN` / `OPERATOR_SYMBOLS` in `code_gen.py`. The entry must compute what the function computes — same defaults, same strictness, same null handling — not the nearest Polars method; constants both sides need belong in `funcs/utils.py` (as `DATE_TEXT_FORMATS` and the `to_boolean` text tables do). `tests/test_code_gen_runtime_parity.py` evals generated code against the live function on awkward data; add the function's edge cases there.

### Membership operators (`in` / `not in`)

`in` means two different things, decided by what follows it. Against a plain value it is
substring containment (`_in` / `_not_in`, lowering to `str.contains`); against a parenthesized
list it is set membership (`_is_in` / `_is_not_in`, lowering to `is_in`). The switch happens in
`process_inline.build_operator_tree`, which swaps in the `MEMBERSHIP_OPERATORS` implementation
when the right operand is a `_list` node.

That `_list` node is built by `hierarchy_builder.handle_opening_bracket` when a `(` directly
follows a membership operator. It has to be a named node rather than the anonymous `pl.lit`
grouping used elsewhere, because `handle_seperator` attaches commas to `current_func.parent` —
for a grouping node that is whatever encloses the group, which scatters the members. `_list` is
special-cased in all three `Func` methods in `models.py`, so its members stay unwrapped.

Only a list of text lowers to `is_in`. Polars 2's `is_in` does no supertype casting, so an
Int column against `[1.0, 2.5]`, a Float column against `[1, 2]`, and an Int column against a
`concat_list` holding a Float column all raise, and no dtype is known when the expression is
built. So a list of literal numbers becomes `==` per member under `pl.any_horizontal` (a single
member is a bare `==`), and a list holding an expression becomes a null-safe `==` per member
wrapped in `pl.when(v.is_not_null())`, which keeps `is_in`'s nulls: null value gives null, null
member never matches. `==` finds the supertype and keeps integers exact; the trade-off is that
`[flag] in (1)` on a Boolean column behaves like `[flag] = 1` rather than raising.
`funcs/utils.numeric_members` decides the numeric case for both `_is_in` and
`code_gen.membership_code`. Code gen reads the `_list` node's own members
(`Func._membership_to_polars_code`), not the rendered list text, and emits each literal from its
source text, so `1e400` is never re-rendered as `inf`.

Negative members are plain numbers because `process_inline.normalize_membership_list` folds a
`negation(<number>)` member into a negative literal. The unary-minus decision stays in
`hierarchy_builder.build_hierarchy`; don't add a second rule at the token level — one that
treated every `case_when` token as unary broke `if ... endif - 1`.

`not in` is two words, but preprocessing strips every space, so it arrives as `notin` and comes
apart on the `in` split value; `token_classifier.merge_multiword_operators` merges the pair back.
Multi-word operators are deliberately kept out of `all_split_vals` — the tokenizer's multi-char
splitter picks the first match from that set in arbitrary order, not the longest, so any split
value containing `in` would be mis-split.

### String functions on non-text columns

Every `.str` method refuses a non-String column, so each string function routes its text
argument through `as_string_expr` in `funcs/utils.py` first (`contains` lives in
`logic_functions.py` and does the same; `in`/`not in` lower to it). The cast has to be decided
per batch, inside `map_batches`, because an expression is built against no frame and cannot
know whether `[order_date]` is a Date, an Int64 or already text. Datetime is rendered with
`STRING_CAST_DATETIME_FORMAT` rather than `cast(pl.String)`, which would append `.000000` and
break `ends_with([ts], "10:00:00")`. String input is returned untouched.

This is the one place the three parallel methods deliberately differ: `to_polars_code` still
emits the plain `.str` chain, so generated code stays readable Polars and the code-gen tests
keep asserting it. `to_string` and `format_date` are not routed through the helper.

### Optional `polars-ds` dependency

`string_similarity()` is the only feature requiring `polars-ds`, exposed as the `similarity` extra. polars-ds ships **no WebAssembly wheel**, so it must stay optional for the browser/Pyodide playground. The import is lazy inside `__get_similarity_method` in `string_functions.py` — never import `polars_ds` at module top level.

### Errors

- `ExpressionSyntaxError` (subclass of `ValueError`, in `exceptions.py`) carries `expression`/`position`/`hint` and renders a caret-pointer message. `expression_validator.py` produces these during parsing.
- `PolarsCodeGenError` is raised when `to_polars_code(validate=True)` generates code that fails `eval`.

### Docs / playground

`docs/` is a Pyodide-based browser playground deployed to GitHub Pages (`.github/workflows/deploy-docs.yml`), which runs `generate_docs.py` and builds the wheel into `docs/assets/wheel/`. Because the playground runs in-browser, keep the core import path free of native-only dependencies.
