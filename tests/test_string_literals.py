"""Round-trip tests for string literals across the whole pipeline.

Every case is checked three ways — the live expression, the generated Polars
code, and evaluating that generated code — so a quoting change cannot fix one
path while breaking another.
"""

import pytest

import polars as pl

from polars_expr_transformer import simple_function_to_expr, to_polars_code
from polars_expr_transformer.string_literals import (
    is_string_literal,
    parse_literal,
    parse_number_literal,
    parse_string_literal,
    render_string_literal,
)


LITERAL_CASES = [
    ('"hello"', "hello"),
    ("'hello'", "hello"),
    ('""', ""),
    ("''", ""),
    ('" "', " "),
    ('"it\'s"', "it's"),
    ('\'say "hi"\'', 'say "hi"'),
    ("'\"'", '"'),
    ('"héllo ünicode 日本 🎈"', "héllo ünicode 日本 🎈"),
    ("'héllo ünicode 日本 🎈'", "héllo ünicode 日本 🎈"),
    (r'"a\nb"', "a\nb"),
    (r'"a\tb"', "a\tb"),
    (r"'a\nb'", "a\nb"),
    (r'"a\\b"', "a\\b"),
    (r"'a\\b'", "a\\b"),
    (r"'a\\\" b'", 'a\\" b'),
    ('"100%"', "100%"),
    ('"a,b"', "a,b"),
    ('"[not_a_column]"', "[not_a_column]"),
    ('"1 + 1"', "1 + 1"),
]


def literal_value(expression: str):
    return pl.select(simple_function_to_expr(expression)).to_series()[0]


@pytest.mark.parametrize("token, expected", LITERAL_CASES)
def test_literal_evaluates_to_its_value(token, expected):
    assert literal_value(token) == expected


@pytest.mark.parametrize("token, expected", LITERAL_CASES)
def test_generated_code_round_trips(token, expected):
    code = to_polars_code(token)
    assert code.startswith("pl.lit(")
    assert pl.select(eval(code, {"pl": pl})).to_series()[0] == expected


@pytest.mark.parametrize("token, expected", LITERAL_CASES)
def test_literal_survives_concatenation(token, expected):
    df = pl.DataFrame({"a": ["x"]})
    result = df.select(simple_function_to_expr(f"concat([a], {token})")).to_series()[0]
    assert result == "x" + expected


@pytest.mark.parametrize("token, expected", LITERAL_CASES)
def test_literal_compares_equal_to_itself(token, expected):
    df = pl.DataFrame({"a": [expected, "other"]})
    result = df.select(simple_function_to_expr(f"[a] = {token}")).to_series().to_list()
    assert result == [True, False]


def test_quoted_column_name_is_a_literal_not_a_column():
    df = pl.DataFrame({"a": [1]})
    assert df.select(simple_function_to_expr('"a"')).to_series()[0] == "a"


class TestHelpers:
    @pytest.mark.parametrize("token", ['"a"', "'a'", '""', "''", "'\"'"])
    def test_is_string_literal_accepts_quoted_tokens(self, token):
        assert is_string_literal(token)

    @pytest.mark.parametrize("token", ["a", "'", '"', "", "42", "[a]", "'a\""])
    def test_is_string_literal_rejects_the_rest(self, token):
        assert not is_string_literal(token)

    @pytest.mark.parametrize("token, expected", LITERAL_CASES)
    def test_parse_string_literal(self, token, expected):
        assert parse_string_literal(token) == expected

    @pytest.mark.parametrize("token", ["a", "42", "[a]"])
    def test_parse_string_literal_rejects_non_literals(self, token):
        with pytest.raises(ValueError):
            parse_string_literal(token)

    def test_escaped_delimiter_is_honoured_by_the_helper(self):
        """The tokenizer cannot produce this token, but the helper reads it."""
        assert parse_string_literal("'it\\'s'") == "it's"

    def test_a_body_that_looks_like_code_stays_data(self):
        assert parse_string_literal('"a" + "b"') == 'a" + "b'

    def test_trailing_backslash_does_not_break_the_literal(self):
        assert parse_string_literal("'a\\'") == "a\\"

    @pytest.mark.parametrize(
        "token, expected", [("42", 42), ("-7", -7), ("3.5", 3.5), ("1e3", 1000.0), ("+3", 3)]
    )
    def test_parse_number_literal(self, token, expected):
        assert parse_number_literal(token) == expected

    @pytest.mark.parametrize("token", ["inf", "nan", "true", "x", "1+1"])
    def test_parse_number_literal_rejects_non_numbers(self, token):
        with pytest.raises(ValueError):
            parse_number_literal(token)

    @pytest.mark.parametrize(
        "token, expected",
        [("0x1f", 31), ("0b101", 5), ("0o17", 15), ("1_000", 1000), ('"a"', "a")],
    )
    def test_parse_literal_keeps_the_forms_eval_used_to_accept(self, token, expected):
        assert parse_literal(token) == expected

    @pytest.mark.parametrize("token", ["0x1f", "0b101", "1_000"])
    def test_exotic_number_forms_still_evaluate(self, token):
        assert literal_value(token) == parse_literal(token)

    @pytest.mark.parametrize("_token, value", LITERAL_CASES)
    def test_render_string_literal_round_trips(self, _token, value):
        import ast

        assert ast.literal_eval(render_string_literal(value)) == value

    def test_render_string_literal_prefers_double_quotes(self):
        assert render_string_literal("abc") == '"abc"'
        assert render_string_literal("a\nb") == '"a\\nb"'
