# Changelog

Notable changes per release. Releases before 0.6.4 are recorded only in the git
history and the tags on GitHub.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
