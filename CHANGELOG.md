# Changelog

Notable changes per release. Releases before 0.6.4 are recorded only in the git
history and the tags on GitHub.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1]

### Fixed

- **`in ( ... )` with numbers works on Polars 2.** `[a] in (-1, -3)` and
  `[a] in (1, 2.5)` raised `InvalidOperationError` on Polars 2, as did a decimal
  column against whole numbers, such as `[price] in (1, 2)`. They now return the
  same rows as on Polars 1.x, on integer, unsigned, float and `Decimal` columns, for `in`
  and `not in` alike.
  - A negative number in the list is now a plain number literal, not a negation
    expression. This kept a list such as `(-1, -3)` from being a plain list of values.
  - A list of numbers is compared member by member with `==`, which finds the
    common type itself. Integers stay exact, including 64-bit IDs. Whole numbers
    in a mixed list are no longer turned into decimals. So `[a] in (1, 0)` now
    means exactly `[a] = 1 or [a] = 0`, including on a Boolean column, where it
    used to raise.
  - A list holding a column, such as `[qty] in ([min_qty], 2.5)`, is compared the
    same way, so the member and value no longer need the same type. Nulls behave
    as before: a null value gives null and a null member never matches.
  - `to_polars_code()` / `to_flowframe_code()` emit the same comparison, as
    `pl.any_horizontal([pl.col("a") == -1, pl.col("a") == -3])`, or a plain
    `pl.col("a") == 1` for one member. Lists of text still generate `is_in`.
    Members are emitted as written, so `1e400` no longer generates invalid code.

## [0.7.0]

### Changed

- **Polars 2 is supported.** The dependency is now `polars>=1.29.0,<3.0.0`. The
  floor rose from 1.8.2 because `replace()` now relies on list patterns in
  `str.replace_many`, which Polars accepts from 1.29.0.

### Fixed

- **`replace()` works on Polars 2.** It passed the find and replacement text to
  `str.replace_many` as flat strings, which Polars 1.x imploded on its behalf
  (with a deprecation warning) and Polars 2 refuses. Both are now imploded
  explicitly, in the expression and in the code `to_polars_code()` /
  `to_flowframe_code()` emit, so results are unchanged on Polars 1.x. On Polars
  older than 1.29, which the Pyodide playground still runs, `replace()` keeps the
  old call.

## [0.6.5]

### Fixed

- **`to_polars_code()` / `to_flowframe_code()` now compute what `to_boolean`,
  `to_date` and `to_datetime` compute.** The generated code used to differ
  from the live expression:

  | Function | Was emitted as | Now emitted as |
  | --- | --- | --- |
  | `to_boolean` | `cast(pl.Boolean)`, which refuses text and keeps null | the `when/then` chain the function builds: `"yes"`/`"no"`, number text and zeros are read, null becomes false |
  | `to_date` | `str.to_date()`, a strict parse with an inferred format | `str.to_date("%Y-%m-%d", strict=False)`, or the format given |
  | `to_datetime` | `str.to_datetime()`, a strict parse with an inferred format | `str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)`, or the format given |

  So text that does not match the format now becomes null in the generated
  code too, instead of failing the whole column. The default formats and the
  `to_boolean` text tables are shared between the functions and the code
  generator, and `tests/test_code_gen_runtime_parity.py` evaluates both on the
  same frame.

## [0.6.4]

### Changed

- **String functions now read non-text columns.** Every function that reaches
  for a `.str` method — `contains`, `left`, `right`, `mid`/`substring`,
  `starts_with`, `ends_with`, `length`, `uppercase`, `lowercase`, `titlecase`,
  `trim`/`left_trim`/`right_trim`, `replace`, `find_position`, `pad_left`,
  `pad_right`, `count_match`, `reverse`, `repeat`, `split` — renders its input
  as text first instead of raising `expected String type, got: date`. The `in`
  and `not in` membership operators lower to `contains` and are covered too.
  A filter such as `contains([order_date], "2026-09")` now runs.

  How a value renders depends on its type:

  | Column type | Reads as |
  | --- | --- |
  | String | unchanged — existing formulas keep their behaviour exactly |
  | Date | `2026-09-24` |
  | Datetime | `2026-09-24 10:00:00` |
  | anything else (Int, Float, Boolean, Categorical, Time, …) | as `cast(pl.String)` renders it |

  Datetime deliberately drops the fractional seconds that `cast(pl.String)`
  would append, so that `ends_with([ordered_at], "10:00:00")` matches what the
  value reads as. `format_date` remains the way to ask for any other layout.

  `to_string` and `format_date` are unchanged.

### Note for callers

`to_polars_code()` / `to_flowframe_code()` still emit the plain `.str` chain,
which assumes a String column. The rendering above happens inside the live
expression built by `simple_function_to_expr()`; pass a non-text column to
generated code and Polars raises as it did before.
