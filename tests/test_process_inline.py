import unittest
from polars_expr_transformer.process.models import Classifier, Func, IfFunc, ConditionVal
from polars_expr_transformer.configs.settings import operators
from polars_expr_transformer.process.process_inline import (
    reverse_dict,
    inline_to_prefix_formula,
    evaluate_prefix_formula,
    parse_formula,
    flatten_inline_formula,
    resolve_inline_formula,
    parse_inline_functions
)


class TestReverseDict(unittest.TestCase):

    def test_reverse_dict(self):
        """Test reversing a dictionary."""
        original = {'a': 1, 'b': 2, 'c': 3}
        expected = {1: 'a', 2: 'b', 3: 'c'}
        result = reverse_dict(original)
        self.assertEqual(result, expected)

    def test_reverse_dict_with_duplicate_values(self):
        """Test reversing a dictionary with duplicate values."""
        original = {'a': 1, 'b': 2, 'c': 1}
        result = reverse_dict(original)
        # The last occurrence of the value will be kept
        self.assertEqual(result, {1: 'c', 2: 'b'})

    def test_reverse_dict_empty(self):
        """Test reversing an empty dictionary."""
        original = {}
        result = reverse_dict(original)
        self.assertEqual(result, {})


class TestInlineToPrefix(unittest.TestCase):

    def test_inline_to_prefix_simple(self):
        """Test converting a simple inline formula to prefix notation."""
        # a + b
        tokens = [
            Classifier('a'),
            Classifier('+'),
            Classifier('b')
        ]
        # Set up operator precedence manually
        tokens[1].precedence = 1

        result = inline_to_prefix_formula(tokens)

        # Expected: + b a
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].val, '+')
        self.assertEqual(result[1].val, 'b')
        self.assertEqual(result[2].val, 'a')

    def test_inline_to_prefix_with_parentheses(self):
        """Test converting an inline formula with parentheses to prefix notation."""
        # (a + b) * c
        tokens = [
            Classifier('('),
            Classifier('a'),
            Classifier('+'),
            Classifier('b'),
            Classifier(')'),
            Classifier('*'),
            Classifier('c')
        ]
        # Set up operator precedence manually
        tokens[2].precedence = 1
        tokens[5].precedence = 2

        result = inline_to_prefix_formula(tokens)

        # Expected: * c + b a
        expected_vals = ['*', 'c', '+', 'b', 'a']
        for i, val in enumerate(expected_vals):
            self.assertEqual(result[i].val, val)

    def test_inline_to_prefix_complex(self):
        """Test converting a complex inline formula to prefix notation."""
        # a + b * c - d / e
        tokens = [
            Classifier('a'),
            Classifier('+'),
            Classifier('b'),
            Classifier('*'),
            Classifier('c'),
            Classifier('-'),
            Classifier('d'),
            Classifier('/'),
            Classifier('e')
        ]
        # Set up operator precedence manually
        tokens[1].precedence = 1  # +
        tokens[3].precedence = 2  # *
        tokens[5].precedence = 1  # -
        tokens[7].precedence = 2  # /

        result = inline_to_prefix_formula(tokens)

        # The exact output will depend on the implementation
        # Just verify key structural relationships

        # Find positions of tokens in the result
        positions = {token.val: i for i, token in enumerate(result)}

        # Check basic structure and operator precedence
        self.assertIn('+', positions)
        self.assertIn('*', positions)
        self.assertIn('-', positions)
        self.assertIn('/', positions)
        self.assertIn('a', positions)
        self.assertIn('b', positions)
        self.assertIn('c', positions)
        self.assertIn('d', positions)
        self.assertIn('e', positions)

        # Higher precedence operators should be processed before lower precedence ones
        # This should put * and / before + and - in the prefix notation
        self.assertLess(positions['+'], positions['*'])
        self.assertLess(positions['-'], positions['/'])

    def test_inline_to_prefix_with_non_classifier(self):
        """Test handling non-Classifier objects in the formula."""
        func = Func(Classifier('test_func'))
        tokens = [func, Classifier('+'), Classifier('value')]
        tokens[1].precedence = 1  # Set precedence for +

        result = inline_to_prefix_formula(tokens)

        # Non-Classifier objects should be passed through
        self.assertEqual(len(result), 3)
        # Verify the correct ordering in prefix notation
        self.assertEqual(result[0].val, '+')
        # The operands might be in different orders depending on implementation
        operands = [result[1], result[2]]
        self.assertIn(func, operands)
        self.assertTrue(any(op.val == 'value' for op in operands if isinstance(op, Classifier)))


class TestEvaluatePrefixFormula(unittest.TestCase):

    def setUp(self):
        # Save original operators dictionary for cleanup
        self.original_operators = operators.copy()
        # Set up test operators
        operators.clear()
        operators.update({'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'})

    def tearDown(self):
        # Restore original operators
        operators.clear()
        operators.update(self.original_operators)

    def test_evaluate_prefix_with_non_operator(self):
        """Test evaluating a prefix formula with a non-operator."""
        tokens = [Classifier('value')]

        result = evaluate_prefix_formula(tokens)

        # Should return the token unchanged
        self.assertEqual(result.val, 'value')


class TestParseFormula(unittest.TestCase):

    def setUp(self):
        # Save original operators dictionary for cleanup
        self.original_operators = operators.copy()
        # Set up test operators
        operators.clear()
        operators.update({'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'})

    def tearDown(self):
        # Restore original operators
        operators.clear()
        operators.update(self.original_operators)

    def test_parse_formula_with_operators(self):
        """Test parsing a formula with operators."""
        tokens = [
            Classifier('a'),
            Classifier('+'),
            Classifier('b')
        ]

        result = parse_formula(tokens)

        # The '+' operator should be replaced with 'add'
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].val, 'a')
        self.assertEqual(result[1].val, 'add')
        self.assertEqual(result[2].val, 'b')

    def test_parse_formula_with_non_operators(self):
        """Test parsing a formula with non-operator tokens."""
        tokens = [
            Classifier('a'),
            Classifier('b'),
            Classifier('c')
        ]

        result = parse_formula(tokens)

        # The tokens should remain unchanged
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].val, 'a')
        self.assertEqual(result[1].val, 'b')
        self.assertEqual(result[2].val, 'c')

    def test_parse_formula_with_mixed_tokens(self):
        """Test parsing a formula with a mix of operators and non-operators."""
        func = Func(Classifier('test_func'))
        tokens = [
            Classifier('a'),
            Classifier('+'),
            func
        ]

        result = parse_formula(tokens)

        # The '+' operator should be replaced, and the Func object should remain
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].val, 'a')
        self.assertEqual(result[1].val, 'add')
        self.assertEqual(result[2], func)


class TestFlattenInlineFormula(unittest.TestCase):

    def test_flatten_inline_formula_simple(self):
        """Test flattening a simple nested formula."""
        nested = (
            Classifier('func'),
            Classifier('('),
            Classifier('arg'),
            Classifier(')')
        )

        result = flatten_inline_formula(nested)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].val, 'func')
        self.assertEqual(result[1].val, '(')
        self.assertEqual(result[2].val, 'arg')
        self.assertEqual(result[3].val, ')')

    def test_flatten_inline_formula_complex(self):
        """Test flattening a complex nested formula."""
        nested = (
            Classifier('func1'),
            Classifier('('),
            (
                Classifier('func2'),
                Classifier('('),
                Classifier('arg'),
                Classifier(')')
            ),
            Classifier(')')
        )

        result = flatten_inline_formula(nested)

        self.assertEqual(len(result), 7)
        self.assertEqual(result[0].val, 'func1')
        self.assertEqual(result[1].val, '(')
        self.assertEqual(result[2].val, 'func2')
        self.assertEqual(result[3].val, '(')
        self.assertEqual(result[4].val, 'arg')
        self.assertEqual(result[5].val, ')')
        self.assertEqual(result[6].val, ')')

    def test_flatten_inline_formula_empty(self):
        """Test flattening an empty nested formula."""
        nested = ()

        result = flatten_inline_formula(nested)

        self.assertEqual(result, [])


class TestResolveInlineFormula(unittest.TestCase):

    def setUp(self):
        # Save original operators dictionary for cleanup
        self.original_operators = operators.copy()
        # Set up test operators
        operators.clear()
        operators.update({'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'})

    def tearDown(self):
        # Restore original operators
        operators.clear()
        operators.update(self.original_operators)

    def test_resolve_inline_formula_without_operators(self):
        """Test resolving an inline formula without operators."""
        tokens = [
            Classifier('a'),
            Classifier('b'),
            Classifier('c')
        ]
        tokens[0].val_type = 'string'
        tokens[1].val_type = 'string'
        tokens[2].val_type = 'string'

        result = resolve_inline_formula(tokens)

        # Should return the tokens unchanged if there are no operators
        self.assertEqual(result, tokens)

    def test_resolve_inline_formula_with_operators(self):
        """Test resolving a simple inline formula with operators."""
        # a + b
        tokens = [
            Classifier('a'),
            Classifier('+'),
            Classifier('b')
        ]
        tokens[0].val_type = 'string'
        tokens[1].val_type = 'operator'
        tokens[2].val_type = 'string'
        tokens[1].precedence = 1

        result = resolve_inline_formula(tokens)

        # The result should be a flattened representation of add(a, b)
        self.assertTrue(len(result) >= 4)
        # Check that the tokens include 'add', '(', 'a', ',', 'b', ')'
        vals = [token.val for token in result]
        self.assertIn('add', vals)
        self.assertIn('(', vals)
        self.assertIn('a', vals)
        self.assertIn('b', vals)
        self.assertIn(',', vals)
        self.assertIn(')', vals)


# Additional tests for parse_inline_functions would require more complex setup
# and would be implementation-specific. They are omitted here for simplicity.
