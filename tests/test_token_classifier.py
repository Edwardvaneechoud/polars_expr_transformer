import unittest
from typing import List
from polars_expr_transformer.process.models import Classifier
from polars_expr_transformer.process.token_classifier import (
    replace_ambiguity_minus_sign,
    standardize_quotes,
    classify_tokens
)


class TestStandardizeTokens(unittest.TestCase):

    def test_standardize_quotes(self):
        """Test standardizing single quotes to double quotes."""
        # Basic case with single quotes
        tokens = ["'string'", "+", "'another string'"]
        result = standardize_quotes(tokens)
        expected = ['"string"', "+", '"another string"']
        self.assertEqual(result, expected)

        # Mixed quotes
        tokens = ["'single'", "\"double\"", "'mixed'"]
        result = standardize_quotes(tokens)
        expected = ['"single"', "\"double\"", '"mixed"']
        self.assertEqual(result, expected)

        # Non-string tokens should be unchanged
        tokens = ["123", "+", "variable_name", "[column]"]
        result = standardize_quotes(tokens)
        self.assertEqual(result, tokens)

        # Single quote characters that aren't string delimiters
        tokens = ["word", "don't", "isn't"]
        result = standardize_quotes(tokens)
        self.assertEqual(result, tokens)

        # Empty string
        tokens = ["''"]
        result = standardize_quotes(tokens)
        expected = ['""']
        self.assertEqual(result, expected)

        # Edge case: single quotes with content resembling double quotes
        tokens = ["'\"inner\"'"]
        result = standardize_quotes(tokens)
        expected = ['"\"inner\""']
        self.assertEqual(result, expected)

    def test_replace_ambiguity_minus_sign_leading(self):
        """Test replacing leading minus signs."""
        # Leading minus (unary negative)
        tokens = [
            Classifier('-'),
            Classifier('5')
        ]
        result = replace_ambiguity_minus_sign(tokens)
        expected = [
            Classifier('__negative()'),
            Classifier('*'),
            Classifier('5')
        ]
        self.assertEqual([t.val for t in result], [t.val for t in expected])

    def test_replace_ambiguity_minus_sign_after_operator(self):
        """Test replacing minus signs after operators."""
        # Minus after an operator (unary negative)
        tokens = [
            Classifier('10'),
            Classifier('+'),
            Classifier('-'),
            Classifier('5')
        ]
        result = replace_ambiguity_minus_sign(tokens)
        expected = [
            Classifier('10'),
            Classifier('+'),
            Classifier('__negative()'),
            Classifier('*'),
            Classifier('5')
        ]
        self.assertEqual([t.val for t in result], [t.val for t in expected])

        # Minus after multiplication (unary negative)
        tokens = [
            Classifier('10'),
            Classifier('*'),
            Classifier('-'),
            Classifier('5')
        ]
        result = replace_ambiguity_minus_sign(tokens)
        expected = [
            Classifier('10'),
            Classifier('*'),
            Classifier('__negative()'),
            Classifier('*'),
            Classifier('5')
        ]
        self.assertEqual([t.val for t in result], [t.val for t in expected])

    def test_replace_ambiguity_minus_sign_after_number(self):
        """Test replacing minus signs after numbers."""
        # Minus after a number (binary subtraction converted to addition of negative)
        tokens = [
            Classifier('10'),
            Classifier('-'),
            Classifier('5')
        ]
        result = replace_ambiguity_minus_sign(tokens)
        expected = [
            Classifier('10'),
            Classifier('+'),
            Classifier('__negative()'),
            Classifier('*'),
            Classifier('5')
        ]
        self.assertEqual([t.val for t in result], [t.val for t in expected])

    def test_replace_ambiguity_minus_sign_complex(self):
        """Test replacing minus signs in complex expressions."""
        # Complex expression with multiple minus signs
        tokens = [
            Classifier('a'),
            Classifier('+'),
            Classifier('-'),
            Classifier('b'),
            Classifier('*'),
            Classifier('-'),
            Classifier('c')
        ]
        result = replace_ambiguity_minus_sign(tokens)
        expected = [
            Classifier('a'),
            Classifier('+'),
            Classifier('__negative()'),
            Classifier('*'),
            Classifier('b'),
            Classifier('*'),
            Classifier('__negative()'),
            Classifier('*'),
            Classifier('c')
        ]
        self.assertEqual([t.val for t in result], [t.val for t in expected])

    def test_replace_ambiguity_minus_sign_no_minus(self):
        """Test that expressions without minus signs are unchanged."""
        tokens = [
            Classifier('a'),
            Classifier('+'),
            Classifier('b'),
            Classifier('*'),
            Classifier('c')
        ]
        result = replace_ambiguity_minus_sign(tokens)
        self.assertEqual([t.val for t in result], [t.val for t in tokens])

    def test_replace_ambiguity_minus_sign_with_spaces(self):
        """Test that empty tokens representing spaces are handled correctly."""
        tokens = [
            Classifier(''),
            Classifier('a'),
            Classifier(''),
            Classifier('-'),
            Classifier(''),
            Classifier('b')
        ]
        result = replace_ambiguity_minus_sign(tokens)
        expected = [
            Classifier('a'),
            Classifier('__negative()'),
            Classifier('*'),
            Classifier('b')
        ]
        self.assertEqual([t.val for t in result], [t.val for t in expected])

    def test_standardize_tokens(self):
        """Test the complete standardize_tokens function."""
        # Test with mixed tokens including quotes and minus signs
        tokens = ["'string'", "-", "5", "+", "-", "'another'"]
        result = classify_tokens(tokens)

        # Note: The function seems to be commenting out the call to replace_ambiguity_minus_sign
        # so we're only testing the quote standardization and empty token removal here
        expected = [
            Classifier('"string"'),
            Classifier('-'),
            Classifier('5'),
            Classifier('+'),
            Classifier('-'),
            Classifier('"another"')
        ]
        self.assertEqual([t.val for t in result], [t.val for t in expected])

        # Test with empty tokens
        tokens = ["", "a", "", "b", ""]
        result = classify_tokens(tokens)
        expected = [Classifier('a'), Classifier('b')]
        self.assertEqual([t.val for t in result], [t.val for t in expected])

    def test_standardize_tokens_with_minus_enabled(self):
        """Test the standardize_tokens function with ambiguity resolution enabled."""
        # We need to temporarily modify the function to enable minus sign replacement
        # This is a mock test assuming the replace_ambiguity_minus_sign is called
        tokens = ["'string'", "-", "5"]

        # Create a modified version of standardize_tokens that includes minus replacement
        def modified_standardize_tokens(tokens: List[str]) -> List[Classifier]:
            standardized_tokens = standardize_quotes(tokens)
            toks = [Classifier(val) for val in standardized_tokens]
            toks = [t for t in toks if t.val_type != 'empty']
            return replace_ambiguity_minus_sign(toks)

        result = modified_standardize_tokens(tokens)
        expected = [
            Classifier('"string"'),
            Classifier('__negative()'),
            Classifier('*'),
            Classifier('5')
        ]
        self.assertEqual([t.val for t in result], [t.val for t in expected])

