import unittest
from polars_expr_transformer.configs.settings import all_split_vals, all_functions
from polars_expr_transformer.process.tokenize import tokenize
from polars_expr_transformer.process.preprocess import preprocess


class TestTokenizer(unittest.TestCase):

    def test_basic_tokenization(self):
        """Test basic tokenization of a simple formula."""
        formula = "a + b * c"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['a', '+', 'b', '*', 'c'])

    def test_string_literals(self):
        """Test tokenization with string literals."""
        formula = "concat('Hello, world!', variable)"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['concat', '(', "'Hello, world!'", ',', 'variable', ')'])

        # Test with double quotes
        formula = 'concat("Hello, world!", variable)'
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['concat', '(', '"Hello, world!"', ',', 'variable', ')'])

    def test_nested_functions(self):
        """Test tokenization of nested function calls."""
        formula = "round(sqrt(a * b), 2)"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['round', '(', 'sqrt', '(', 'a', '*', 'b', ')', ',', '2', ')'])

    def test_logical_operators(self):
        """Test tokenization of logical operators."""
        formula = "a > 0 and b < 10 or c == 5"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['a', '>', '0', 'and', 'b', '<', '10', 'or', 'c', '==', '5'])

    def test_brackets(self):
        """Test tokenization with square brackets (column references)."""
        formula = "[column1] + [column2] * 2"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['[column1]', '+', '[column2]', '*', '2'])

    def test_complex_formula(self):
        """Test tokenization of a complex formula."""
        formula ='$if$((pl.col("a")>10 and pl.col("b")<5) or pl.col("c")=\'value\')$then$(concat(pl.col("a"),\' is \',pl.col("b")))$else$(\'not matched\')$endif$'
        tokens = tokenize(formula)
        expected = ['$if$', '(', '(', 'pl.col', '(', '"a"', ')', '>', '10', 'and', 'pl.col', '(', '"b"', ')', '<', '5',
                    ')', 'or', 'pl.col', '(', '"c"', ')', '=', "'value'", ')', '$then$', '(', 'concat', '(', 'pl.col',
                    '(', '"a"', ')', ',', "' is '", ',', 'pl.col', '(', '"b"', ')', ')', ')', '$else$', '(',
                    "'not matched'", ')', '$endif$']
        self.assertEqual(tokens, expected)

    def test_operators_in_strings(self):
        """Test that operators in string literals are not tokenized."""
        formula = "'a + b * c / d'"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ["'a + b * c / d'"])

        formula = '"a > b and c < d"'
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['"a > b and c < d"'])

    def test_equality_operators(self):
        """Test tokenization of different equality operators."""
        formula = "a == b != c >= d <= e"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['a', '==', 'b', '!=', 'c', '>=', 'd', '<=', 'e'])

    def test_parentheses(self):
        """Test tokenization with parentheses."""
        formula = "(a + b) * (c - d)"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['(', 'a', '+', 'b', ')', '*', '(', 'c', '-', 'd', ')'])

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        formula = "  a  +  b  *  c  "
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['a', '+', 'b', '*', 'c'])

    def test_function_names(self):
        """Test tokenization of function names."""
        formula = "sqrt(abs(sin(a)))"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['sqrt', '(', 'abs', '(', 'sin', '(', 'a', ')', ')', ')'])

    def test_string_with_special_chars(self):
        """Test tokenization of strings with special characters."""
        formula = "'string with (parens) and [brackets] and operators + - * /'"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ["'string with (parens) and [brackets] and operators + - * /'"])

    def test_nested_brackets(self):
        """Test tokenization with nested brackets."""
        formula = "[[nested_column]]"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ["[[nested_column]]"])

    def test_mixed_operators(self):
        """Test tokenization with a mix of different operators."""
        formula = "a + b * c / d - e % f"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['a', '+', 'b', '*', 'c', '/', 'd', '-', 'e', '%', 'f'])

    def test_decimal_numbers(self):
        """Test tokenization of decimal numbers."""
        formula = "1.23 + 4.56 * 7.89"
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['1.23', '+', '4.56', '*', '7.89'])

    def test_complex_logical_expressions(self):
        """Test tokenization of complex logical expressions."""
        formula = "(a > 0 and b < 10) or (c >= 5 and d <= 15)"
        tokens = tokenize(formula)
        expected = ['(', 'a', '>', '0', 'and', 'b', '<', '10', ')', 'or',
                    '(', 'c', '>=', '5', 'and', 'd', '<=', '15', ')']
        self.assertEqual(tokens, expected)

    def test_if_else_statement(self):
        """Test tokenization of if-else statements."""
        formula = '$if$(condition)$then$(action)$else$(other_action)$endif$'
        tokens = tokenize(formula)
        self.assertEqual(tokens, ['$if$', '(', 'condition', ')', '$then$', '(', 'action', ')', '$else$',
                                  '(', 'other_action', ')', '$endif$'])

    def test_empty_string(self):
        """Test tokenization of an empty string."""
        formula = ""
        tokens = tokenize(formula)
        self.assertEqual(tokens, [])
