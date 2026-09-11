import hashlib

import polars as pl
import pytest

from polars_expr_transformer import simple_function_to_expr
from polars_expr_transformer.funcs.hashing_functions import (
    hash, md5, sha1, sha256, sha512,
)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame({"name": ["John", None, "Jane"], "age": [30, 25, 40]})


def evaluate(df: pl.DataFrame, expr: pl.Expr) -> list:
    return df.select(expr.alias("result"))["result"].to_list()


@pytest.mark.parametrize(
    "func,algorithm",
    [(md5, "md5"), (sha1, "sha1"), (sha256, "sha256"), (sha512, "sha512")],
)
def test_digest_matches_hashlib(df, func, algorithm):
    result = func(pl.col("name"))
    assert isinstance(result, pl.Expr)
    expected = hashlib.new(algorithm, b"John").hexdigest()
    assert evaluate(df, result)[0] == expected


@pytest.mark.parametrize(
    "func,length", [(md5, 32), (sha1, 40), (sha256, 64), (sha512, 128)]
)
def test_digest_length(df, func, length):
    assert len(evaluate(df, func(pl.col("name")))[0]) == length


def test_digest_keeps_nulls(df):
    assert evaluate(df, sha256(pl.col("name")))[1] is None


def test_digest_of_literal_text(df):
    assert evaluate(df, md5("John"))[0] == hashlib.md5(b"John").hexdigest()


def test_digest_casts_non_string_column(df):
    assert evaluate(df, sha256(pl.col("age")))[0] == hashlib.sha256(b"30").hexdigest()


def test_digest_is_stable_for_equal_values():
    frame = pl.DataFrame({"name": ["John", "John"]})
    first, second = evaluate(frame, sha256(pl.col("name")))
    assert first == second


def test_hash_returns_unsigned_integers(df):
    result = df.select(hash(pl.col("name")).alias("result"))
    assert result.schema["result"] == pl.UInt64


def test_hash_is_equal_for_equal_values():
    frame = pl.DataFrame({"name": ["John", "Jane", "John"]})
    first, second, third = evaluate(frame, hash(pl.col("name")))
    assert first == third
    assert first != second


def test_hash_accepts_numeric_column(df):
    assert all(isinstance(v, int) for v in evaluate(df, hash(pl.col("age"))))


class TestThroughExpressions:
    def test_sha256_expression(self, df):
        expr = simple_function_to_expr("sha256([name])")
        assert evaluate(df, expr)[0] == hashlib.sha256(b"John").hexdigest()

    def test_hash_expression(self, df):
        expr = simple_function_to_expr("hash([name])")
        assert evaluate(df, expr)[0] == evaluate(df, hash(pl.col("name")))[0]

    def test_digest_composes_with_string_functions(self, df):
        expr = simple_function_to_expr("left(uppercase(md5([name])), 8)")
        assert evaluate(df, expr)[0] == hashlib.md5(b"John").hexdigest()[:8].upper()

    def test_digest_inside_conditional(self, df):
        expr = simple_function_to_expr(
            'if is_empty([name]) then "unknown" else sha1([name]) endif'
        )
        result = evaluate(df, expr)
        assert result[0] == hashlib.sha1(b"John").hexdigest()
        assert result[1] == "unknown"
