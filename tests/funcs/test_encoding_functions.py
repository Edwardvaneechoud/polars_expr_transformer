import polars as pl
import pytest

from polars_expr_transformer import simple_function_to_expr
from polars_expr_transformer.funcs.encoding_functions import (
    encode, decode, base64_encode, base64_decode, hex_encode, hex_decode,
)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": ["John", None, "Jane"],
            "age": [30, 25, 40],
            "encoded": ["Sm9obg==", None, "SmFuZQ=="],
            "hexed": ["4a6f686e", None, "4a616e65"],
        }
    )


def evaluate(df: pl.DataFrame, expr: pl.Expr) -> list:
    return df.select(expr.alias("result"))["result"].to_list()


def test_encode_defaults_to_base64(df):
    result = encode(pl.col("name"))
    assert isinstance(result, pl.Expr)
    assert evaluate(df, result)[0] == "Sm9obg=="


def test_encode_hex(df):
    assert evaluate(df, encode(pl.col("name"), "hex"))[0] == "4a6f686e"


def test_encode_casts_non_string_column(df):
    assert evaluate(df, hex_encode(pl.col("age")))[0] == "3330"


def test_encode_keeps_nulls(df):
    assert evaluate(df, base64_encode(pl.col("name")))[1] is None


def test_decode_defaults_to_base64(df):
    assert evaluate(df, decode(pl.col("encoded")))[0] == "John"


def test_decode_hex(df):
    assert evaluate(df, decode(pl.col("hexed"), "hex"))[0] == "John"


def test_decode_of_literal_text(df):
    assert evaluate(df, base64_decode("Sm9obg=="))[0] == "John"


def test_decode_keeps_nulls(df):
    assert evaluate(df, base64_decode(pl.col("encoded")))[1] is None


def test_decode_turns_invalid_input_into_null():
    frame = pl.DataFrame({"encoded": ["Sm9obg==", "not base64!"]})
    assert evaluate(frame, base64_decode(pl.col("encoded"))) == ["John", None]


@pytest.mark.parametrize(
    "encode_func,decode_func",
    [(base64_encode, base64_decode), (hex_encode, hex_decode)],
)
def test_encode_decode_round_trip(df, encode_func, decode_func):
    assert evaluate(df, decode_func(encode_func(pl.col("name")))) == [
        "John",
        None,
        "Jane",
    ]


@pytest.mark.parametrize("func", [encode, decode])
def test_unknown_encoding_raises(func):
    with pytest.raises(ValueError, match="Unknown encoding: rot13"):
        func(pl.col("name"), "rot13")


class TestThroughExpressions:
    def test_encode_expression_with_encoding_argument(self, df):
        expr = simple_function_to_expr('encode([name], "hex")')
        assert evaluate(df, expr)[0] == "4a6f686e"

    def test_round_trip_expression(self, df):
        expr = simple_function_to_expr("base64_decode(base64_encode([name]))")
        assert evaluate(df, expr) == ["John", None, "Jane"]

    def test_encoded_text_can_be_hashed(self, df):
        expr = simple_function_to_expr("length(sha256(base64_encode([name])))")
        assert evaluate(df, expr)[0] == 64
