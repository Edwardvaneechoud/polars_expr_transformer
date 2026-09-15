"""A formula is parsed, never executed.

Every public entry point takes untrusted text (a filter box, a formula node, a
shared link), so no input may reach ``eval``/``exec``. These tests pin that
contract with a canary file: if any payload ever executes Python, the canary
appears and the test fails.
"""

import pytest

import polars as pl

from polars_expr_transformer import (
    simple_function_to_expr,
    to_polars_code,
    to_flowframe_code,
    build_func,
)
from polars_expr_transformer.process.polars_expr_transformer import (
    _validate_polars_code,
)
from polars_expr_transformer.process.token_classifier import standardize_quotes
from polars_expr_transformer.exceptions import PolarsCodeGenError


def hostile_payloads(canary) -> list:
    """Formulas that used to (or plausibly could) run Python."""
    return [
        f'__import__("os").system("touch {canary}")',
        f'open("{canary}", "w")',
        f'[a] > 1 and open("{canary}", "w")',
        f'"x" + str(open("{canary}", "w"))',
        f'[a] = "x\\" + open(\\"{canary}\\", \\"w\\") + \\"y"',
        # The live escape: a single-quoted literal carrying a double quote.
        # standardize_quotes used to requote it unescaped, and get_pl_func
        # eval'd the result.
        f"[a] = 'x\" + open(\"{canary}\", \"w\") + \"y'",
        f"'x\" + open(\"{canary}\", \"w\") + \"y'",
        # Same trick aimed at the column-name path in preprocess().
        f'[x" + open("{canary}", "w") + "y]',
        # Backslash runs must not let a quote slip out of the literal.
        f"'x\\\\\" + open(\"{canary}\", \"w\") + \"y'",
        f"concat('a\" + open(\"{canary}\", \"w\") + \"b', 'c')",
        f"if 'a\" + open(\"{canary}\", \"w\") + \"b' == 'q' then 1 else 2 endif",
    ]


@pytest.fixture()
def canary(tmp_path):
    return tmp_path / "canary"


class TestNoPythonExecution:
    @pytest.mark.parametrize("index", range(11))
    def test_simple_function_to_expr_does_not_execute(self, canary, index):
        expression = hostile_payloads(canary)[index]
        try:
            simple_function_to_expr(expression)
        except Exception:
            pass
        assert not canary.exists(), f"{expression!r} executed Python"

    @pytest.mark.parametrize("index", range(11))
    def test_build_func_does_not_execute(self, canary, index):
        expression = hostile_payloads(canary)[index]
        try:
            build_func(expression)
        except Exception:
            pass
        assert not canary.exists(), f"{expression!r} executed Python"

    @pytest.mark.parametrize("index", range(11))
    def test_to_polars_code_does_not_execute(self, canary, index):
        expression = hostile_payloads(canary)[index]
        for validate in (False, True):
            try:
                to_polars_code(expression, validate=validate)
            except Exception:
                pass
            assert not canary.exists(), f"{expression!r} executed Python"

    @pytest.mark.parametrize("index", range(11))
    def test_to_flowframe_code_does_not_execute(self, canary, index):
        expression = hostile_payloads(canary)[index]
        for validate in (False, True):
            try:
                to_flowframe_code(expression, validate=validate)
            except Exception:
                pass
            assert not canary.exists(), f"{expression!r} executed Python"

    def test_generated_code_for_hostile_literal_is_a_plain_string(self, canary):
        """The payload survives as data: a single string literal, nothing else."""
        expression = f"'x\" + open(\"{canary}\", \"w\") + \"y'"
        code = to_polars_code(expression)
        assert code == f'pl.lit(\'x" + open("{canary}", "w") + "y\')'
        value = pl.select(simple_function_to_expr(expression)).to_series()[0]
        assert value == f'x" + open("{canary}", "w") + "y'
        assert not canary.exists()


class TestValidatorRejectsArbitraryCode:
    """_validate_polars_code must never run code outside its declared scope."""

    def test_rejects_call_to_unknown_name(self, canary):
        with pytest.raises(PolarsCodeGenError) as exc_info:
            _validate_polars_code("expr", f'open("{canary}", "w")')
        assert isinstance(exc_info.value.eval_error, NameError)
        assert not canary.exists()

    def test_rejects_dunder_import(self, canary):
        with pytest.raises(PolarsCodeGenError):
            _validate_polars_code("expr", f'__import__("os").system("touch {canary}")')
        assert not canary.exists()

    def test_rejects_dunder_attribute_traversal(self):
        with pytest.raises(PolarsCodeGenError):
            _validate_polars_code("expr", "pl.__class__.__mro__")

    def test_rejects_free_name_inside_a_lambda(self, canary):
        """The hash functions generate lambdas, so the body is walked too."""
        with pytest.raises(PolarsCodeGenError) as exc_info:
            _validate_polars_code(
                "expr", f'pl.col("a").map_elements(lambda v: open("{canary}", "w"))'
            )
        assert isinstance(exc_info.value.eval_error, NameError)
        assert not canary.exists()

    def test_rejects_comprehension(self):
        with pytest.raises(PolarsCodeGenError):
            _validate_polars_code("expr", "[x for x in pl.Series([1])]")

    def test_rejects_walrus_and_statements(self):
        with pytest.raises(PolarsCodeGenError):
            _validate_polars_code("expr", "import os")

    def test_allows_the_generated_dialect(self):
        _validate_polars_code("expr", 'pl.col("x").str.to_uppercase()')
        _validate_polars_code(
            "expr",
            'pl.col("a").cast(pl.Utf8).map_elements('
            'lambda v: hashlib.sha256(v.encode("utf-8")).hexdigest(), '
            "return_dtype=pl.Utf8)",
        )
        _validate_polars_code("expr", "pl.lit(datetime.datetime.now())")
        _validate_polars_code(
            "expr", 'pl.when(pl.col("a") > pl.lit(1)).then(pl.lit(1)).otherwise(pl.lit(2))'
        )
        _validate_polars_code("expr", 'pl.concat_str([pl.col("a"), pl.lit("b")])')


class TestStandardizeQuotes:
    """Requoting '...' as "..." must preserve the value, escaping as needed."""

    @pytest.mark.parametrize(
        "token, expected",
        [
            ("'abc'", '"abc"'),
            ("''", '""'),
            ('"abc"', '"abc"'),
            ("'a\"b'", '"a\\"b"'),
            ("'\"'", '"\\""'),
            ("'a\\\"b'", '"a\\\"b"'),  # already-escaped quote stays single-escaped
            ("'a\\nb'", '"a\\nb"'),  # backslash escapes pass through untouched
            ("'a\\\\'", '"a\\\\"'),  # escaped backslash stays escaped
            ("[col]", "[col]"),  # non-string tokens untouched
            ("42", "42"),
            ("'", "'"),  # a lone quote is not a literal
        ],
    )
    def test_requoting(self, token, expected):
        assert standardize_quotes([token]) == [expected]

    @pytest.mark.parametrize(
        "token",
        ["'abc'", "''", "'a\"b'", "'\"'", "'a\\nb'", "'a\\\\'", "'ü日本'"],
    )
    def test_requoting_preserves_the_value(self, token):
        import ast

        assert ast.literal_eval(standardize_quotes([token])[0]) == ast.literal_eval(token)
