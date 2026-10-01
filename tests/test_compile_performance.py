"""Compile time must grow linearly with the size of the expression tree.

Every ``Func`` node used to evaluate each of its children about three times, so
a chain like ``([a]!=0) & ([a]!=1) & ...`` cost O(3^depth): eight terms took
seconds. Flowfile writes exactly these chains for its "in" / "not in" filters.
"""
import functools
import operator
import time
from unittest.mock import patch

import polars as pl
import pytest

from polars_expr_transformer.process import models
from polars_expr_transformer.process.models import get_types_from_func
from polars_expr_transformer.process.polars_expr_transformer import simple_function_to_expr


def _chain(n_terms: int, op: str) -> str:
    return f" {op} ".join(f"([a]!={i})" for i in range(n_terms))


def _best_compile_ms(expr: str, repeats: int = 5) -> float:
    """Best of several runs, so a slow CI worker does not fail the test on noise."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        simple_function_to_expr(expr)
        best = min(best, time.perf_counter() - start)
    return best * 1000


def _count_evaluations(expr: str) -> int:
    with patch.object(
        models.Func, "get_pl_func", autospec=True, side_effect=models.Func.get_pl_func
    ) as spy:
        simple_function_to_expr(expr)
    return spy.call_count


@pytest.mark.parametrize("op", ["&", "|"])
def test_evaluations_grow_linearly_with_chain_length(op):
    counts = [_count_evaluations(_chain(n, op)) for n in range(2, 9)]
    increments = {b - a for a, b in zip(counts, counts[1:])}
    assert len(increments) == 1, f"node evaluations per term are not constant: {counts}"


def test_eight_term_chain_compiles_under_20ms():
    assert _best_compile_ms(_chain(8, "&")) < 20


@pytest.mark.parametrize("op", ["&", "|"])
def test_fifty_term_chain_compiles_under_100ms(op):
    # Fail in seconds rather than hang for hours if compile time is exponential again.
    assert _best_compile_ms(_chain(8, op), repeats=1) < 20
    assert _best_compile_ms(_chain(50, op)) < 100


@pytest.mark.parametrize(
    "op, combine", [("&", operator.and_), ("|", operator.or_)]
)
def test_chain_matches_hand_written_polars(op, combine):
    df = pl.DataFrame({"a": [0, 3, 7, 8, None, -1]})
    expected = functools.reduce(combine, [pl.col("a") != i for i in range(8)])
    result = df.select(simple_function_to_expr(_chain(8, op)).alias("r"))
    assert result.equals(df.select(expected.alias("r")))


def test_get_types_from_func_matches_pl_col_by_identity():
    def col(name: int):
        pass

    assert get_types_from_func(pl.col) == [str]
    assert get_types_from_func(col) == [int]
