"""Tests for early expression-syntax validation and its error messages.

These cover the validation that runs on the raw expression string before the
preprocessor rewrites conditional keywords into parenthesized syntax, so that
mistakes like a misspelled 'if' produce a clear message instead of a confusing
unbalanced-parentheses error.
"""

import pytest

from polars_expr_transformer import ExpressionSyntaxError, simple_function_to_expr
from polars_expr_transformer.process.expression_validator import (
    validate_expression_syntax,
)


class TestMisplacedConditionalKeywords:
    def test_misspelled_if_reports_then_without_if(self):
        expr = 'f [age] > 30 then "Senior" else "Junior" endif'
        with pytest.raises(ExpressionSyntaxError, match="there is no 'if' before it") as exc_info:
            simple_function_to_expr(expr)
        exc = exc_info.value
        assert exc.position == 13
        assert "Unbalanced" not in str(exc)
        assert "spelled correctly" in str(exc)
        assert expr in str(exc)

    def test_missing_then(self):
        with pytest.raises(ExpressionSyntaxError, match="missing its 'then'"):
            simple_function_to_expr('if [age] > 30 "Senior" else "Junior" endif')

    def test_missing_endif(self):
        with pytest.raises(ExpressionSyntaxError, match="missing its closing 'endif'"):
            simple_function_to_expr('if [age] > 30 then 1 else 2')

    def test_missing_else_branch(self):
        with pytest.raises(ExpressionSyntaxError, match="has no 'else' branch"):
            simple_function_to_expr('if [age] > 30 then 1 endif')

    def test_stray_else(self):
        with pytest.raises(ExpressionSyntaxError, match="there is no 'if' before it"):
            simple_function_to_expr('else 1 endif')

    def test_extra_endif(self):
        with pytest.raises(ExpressionSyntaxError, match="Found 'endif'.*no 'if' before it"):
            simple_function_to_expr('if 1=1 then 1 else 2 endif endif')

    def test_double_then(self):
        with pytest.raises(ExpressionSyntaxError, match="already has a 'then'"):
            simple_function_to_expr('if 1=1 then 1 then 2 else 3 endif')

    def test_elseif_after_else(self):
        with pytest.raises(ExpressionSyntaxError, match="after 'else'"):
            simple_function_to_expr('if 1=1 then 1 else 2 elseif 2=2 then 3 endif')

    def test_second_else(self):
        with pytest.raises(ExpressionSyntaxError, match="second 'else'"):
            simple_function_to_expr('if 1=1 then 1 else 2 else 3 endif')

    def test_if_then_without_else_or_endif(self):
        with pytest.raises(ExpressionSyntaxError, match="missing 'else' and 'endif'"):
            simple_function_to_expr('if 1=1 then 1')

    def test_bare_if_missing_then(self):
        with pytest.raises(ExpressionSyntaxError, match="missing its 'then'"):
            simple_function_to_expr('if 1=1')

    @pytest.mark.parametrize(
        "expr",
        [
            'IF 1=1 then 1 else 2 endif',
            'if 1=1 Then 1 else 2 endif',
            'if 1=1 then 1 else 2 ENDIF',
        ],
    )
    def test_wrong_case_keywords(self, expr):
        with pytest.raises(ExpressionSyntaxError, match="case-sensitive"):
            simple_function_to_expr(expr)


class TestParenthesesCuttingAcrossKeywords:
    def test_paren_opened_after_if(self):
        with pytest.raises(ExpressionSyntaxError, match="inside parentheses opened after"):
            simple_function_to_expr('if (1=1 then 2) else 3 endif')

    def test_paren_closed_before_keyword(self):
        with pytest.raises(ExpressionSyntaxError, match="closed before 'then'"):
            simple_function_to_expr('(if 1=1) then 2 else 3 endif')


class TestUnbalancedParentheses:
    def test_unclosed_paren(self):
        with pytest.raises(ValueError, match="Unbalanced parentheses") as exc_info:
            simple_function_to_expr('((1)')
        exc = exc_info.value
        assert isinstance(exc, ExpressionSyntaxError)
        assert exc.position == 0

    def test_extra_closing_paren(self):
        with pytest.raises(ExpressionSyntaxError, match="has no matching"):
            simple_function_to_expr('(1))')


class TestValidExpressionsDoNotRaise:
    @pytest.mark.parametrize(
        "expr",
        [
            'if [age] > 30 then "Senior" else "Junior" endif',
            'if 1=1 then if 2=2 then "a" else "b" endif else "c" endif',
            'if [a]=1 then 1 elseif [a]=2 then 2 elseif [a]=3 then 3 else 4 endif',
            'if (1=1) then (2) else (3) endif',
            'concat("then", "if")',
            '"if then else endif"',
            "'if then else endif'",
            'concat("(((", [a])',
            'if 1=1 then 1 else 2 endif // stray then ) (',
            'if 1=1 then 1 else 2 endif\n// comment with endif )\n + 1',
            '1 + 2 * 3',
            'uppercase([name])',
        ],
    )
    def test_no_raise(self, expr):
        validate_expression_syntax(expr)

    def test_keywords_in_column_names_are_skipped(self):
        # The validator must not treat [then] (a column reference) as a keyword.
        validate_expression_syntax('[then] = 5')


class TestErrorFormatting:
    def test_caret_snippet_shows_only_offending_line(self):
        expr = '1 + 2\n+ then\n+ 3'
        with pytest.raises(ExpressionSyntaxError) as exc_info:
            validate_expression_syntax(expr)
        lines = str(exc_info.value).split('\n')
        assert '+ then' in lines
        caret_line = lines[lines.index('+ then') + 1]
        assert caret_line == '  ^'

    def test_catchable_as_value_error(self):
        try:
            simple_function_to_expr('f 1=1 then 1 else 2 endif')
        except ValueError:
            pass
        else:
            pytest.fail("expected a ValueError")


class TestSeparatorOutsideFunction:
    def test_top_level_comma(self):
        with pytest.raises(ExpressionSyntaxError, match="outside of a function call"):
            simple_function_to_expr('1, 2')
