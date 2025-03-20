import unittest
from polars_expr_transformer.process.preprocess import (
    replace_double_spaces, remove_comments, normalize_whitespace,
    add_spaces_around_logical_operators, mark_special_tokens,
    standardize_equality_operators, preserve_logical_operators_with_markers,
    restore_logical_operators, add_additions_outside_of_quotes,
    replace_value_outside_of_quotes, replace_values_outside_of_quotes,
    replace_values, parse_pl_cols, remove_unwanted_characters,
    preprocess
)


class TestPreprocessFunctions(unittest.TestCase):

    def test_replace_double_spaces(self):
        """Test that double spaces are replaced with single spaces."""
        input_str = "hello  world   test    example"
        result = replace_double_spaces(input_str)
        expected = "hello world test example"
        self.assertEqual(result, expected)

        # Test with no double spaces
        input_str = "hello world"
        result = replace_double_spaces(input_str)
        self.assertEqual(result, input_str)

    def test_remove_comments(self):
        """Test that comments are properly removed from strings."""
        # Basic comment removal
        input_str = "code // This is a comment"
        result = remove_comments(input_str)
        expected = "code "
        self.assertEqual(result, expected)

        # Multiple lines with comments
        input_str = "line1 // comment1\nline2 // comment2\nline3"
        result = remove_comments(input_str)
        expected = "line1 \nline2 \nline3"
        self.assertEqual(result, expected)

        # Comment within quotes should be preserved
        input_str = "text with 'string // not a comment' continues"
        result = remove_comments(input_str)
        self.assertEqual(result, input_str)

        input_str = 'text with "string // not a comment" continues'
        result = remove_comments(input_str)
        self.assertEqual(result, input_str)

        # Mixed case with nested quotes
        input_str = "function(param) // comment\nreturn \"string with // preserved\""
        result = remove_comments(input_str)
        expected = "function(param) \nreturn \"string with // preserved\""
        self.assertEqual(result, expected)

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        input_str = "line1\nline2\n\nline3"
        result = normalize_whitespace(input_str)
        expected = "line1 line2 line3"
        self.assertEqual(result, expected)

        # Test with already normalized text
        input_str = "simple text"
        result = normalize_whitespace(input_str)
        self.assertEqual(result, input_str)

        # Test with tabs and multiple spaces
        input_str = "text with\ttabs  and    spaces"
        result = normalize_whitespace(input_str)
        expected = "text with tabs and spaces"
        self.assertEqual(result, expected)

    def test_add_spaces_around_logical_operators(self):
        """Test adding spaces around logical operators."""
        # Basic case
        input_str = "condition1 and condition2 or condition3"
        result = add_spaces_around_logical_operators(input_str)
        expected = "condition1 and condition2 or condition3"
        self.assertEqual(result, expected)

        # Operators in quotes should not be affected
        input_str = "normal and 'quoted and text' or \"more or text\""
        result = add_spaces_around_logical_operators(input_str)
        expected = "normal and 'quoted and text' or \"more or text\""
        self.assertEqual(result, expected)

        # Mixed case
        input_str = "condition1 and 'text'or\"more\"and condition2"
        result = add_spaces_around_logical_operators(input_str)
        expected = "condition1 and 'text' or \"more\" and condition2"
        self.assertEqual(result, expected)

    def test_mark_special_tokens(self):
        """Test marking special tokens."""
        input_str = "if condition then action else other endif"
        result = mark_special_tokens(input_str)
        expected = "$if$( condition )$then$( action )$else$( other )$endif$"
        self.assertEqual(result, expected)

        # Tokens in quotes should not be affected
        input_str = "if condition then 'if then else' endif"
        result = mark_special_tokens(input_str)
        expected = "$if$( condition )$then$( 'if then else' )$endif$"
        self.assertEqual(result, expected)

        # Test with elseif
        input_str = "if cond1 then act1 elseif cond2 then act2 else act3 endif"
        result = mark_special_tokens(input_str)
        expected = "$if$( cond1 )$then$( act1 )$elseif$( cond2 )$then$( act2 )$else$( act3 )$endif$"
        self.assertEqual(result, expected)

    def test_standardize_equality_operators(self):
        """Test standardizing equality operators."""
        input_str = "field1 == value1 and field2 == value2"
        result = standardize_equality_operators(input_str)
        expected = " field1 = value1 and field2 = value2 "
        self.assertEqual(result, expected)

        # Operators in quotes should not be affected
        input_str = "field1 == value1 and 'text == more'"
        result = standardize_equality_operators(input_str)
        expected = " field1 = value1 and 'text == more' "
        self.assertEqual(result, expected)

    def test_preserve_and_restore_logical_operators(self):
        """Test preserving and restoring logical operators."""
        input_str = "condition1 and condition2 or condition3"
        preserved = preserve_logical_operators_with_markers(input_str)
        expected_preserved = "condition1 __and__ condition2 __or__ condition3"
        self.assertEqual(preserved, expected_preserved)

        # Now restore
        restored = restore_logical_operators(preserved)
        expected_restored = "condition1  and  condition2  or  condition3"
        self.assertEqual(restored, expected_restored)

        # Test with operators in quotes
        input_str = "condition1 and 'quoted and string' or condition2"
        preserved = preserve_logical_operators_with_markers(input_str)
        expected_preserved = "condition1 __and__ 'quoted and string' __or__ condition2"
        self.assertEqual(preserved, expected_preserved)

    def test_add_additions_outside_of_quotes(self):
        """Test adding additions outside of quotes."""
        input_str = "if condition then action else other endif"
        result = add_additions_outside_of_quotes(input_str, "#", "if", "then", "else", "endif")
        expected = "#if# condition #then# action #else# other #endif#"
        self.assertEqual(result, expected)

        # Tokens in quotes should not be affected
        input_str = "if condition then 'if then else' endif"
        result = add_additions_outside_of_quotes(input_str, "#", "if", "then", "else", "endif")
        expected = "#if# condition #then# 'if then else' #endif#"
        self.assertEqual(result, expected)

    def test_replace_value_outside_of_quotes(self):
        """Test replacing values outside of quotes."""
        input_str = "field1 === value1 and field2 === value2"
        result = replace_value_outside_of_quotes(input_str, "===", "==")
        expected = " field1 == value1 and field2 == value2 "
        self.assertEqual(result, expected)

        # Values in quotes should not be affected
        input_str = "field1 === value1 and 'field === value'"
        result = replace_value_outside_of_quotes(input_str, "===", "==")
        expected = " field1 == value1 and 'field === value' "
        self.assertEqual(result, expected)

    def test_replace_values_outside_of_quotes(self):
        """Test replacing multiple values outside of quotes."""
        input_str = "a + b - c * d / e"
        replacements = [("+", "ADD"), ("-", "SUB"), ("*", "MUL"), ("/", "DIV")]
        result = replace_values_outside_of_quotes(input_str, replacements)
        expected = "a ADD b SUB c MUL d DIV e"
        self.assertEqual(result, expected)

        # Values in quotes should not be affected
        input_str = "a + b - 'c * d' / e"
        result = replace_values_outside_of_quotes(input_str, replacements)
        expected = "a ADD b SUB 'c * d' DIV e"
        self.assertEqual(result, expected)

    def test_replace_values(self):
        """Test replacing values in a substring."""
        input_str = "var1 + var2 - var3"
        result = replace_values(input_str, "$", "var1", "var3")
        expected = "$var1$ + var2 - $var3$"
        self.assertEqual(result, expected)

        # Only whole words should be replaced
        input_str = "var1 + var12 - var123"
        result = replace_values(input_str, "$", "var1")
        expected = "$var1$ + var12 - var123"
        self.assertEqual(result, expected)

    def test_parse_pl_cols(self):
        """Test parsing Polars column expressions."""
        input_str = "function([column1] + [column2] * 2)"
        result = parse_pl_cols(input_str)
        expected = "function(pl.col(\"column1\") + pl.col(\"column2\") * 2)"
        self.assertEqual(result, expected)

        # Column references in quotes should not be affected
        input_str = "function([column1], '[column2]')"
        result = parse_pl_cols(input_str)
        expected = "function(pl.col(\"column1\"), '[column2]')"
        self.assertEqual(result, expected)

        # Column references with commas should not be transformed
        input_str = "function([column1, column2])"
        result = parse_pl_cols(input_str)
        self.assertEqual(result, input_str)

    def test_remove_unwanted_characters(self):
        """Test removing unwanted characters."""
        input_str = "  function  (  param1  ,  param2  )  "
        result = remove_unwanted_characters(input_str)
        expected = "function(param1,param2)"
        self.assertEqual(result, expected)

        # Special markers should be preserved
        input_str = "condition1 __and__ condition2 __or__ condition3"
        result = remove_unwanted_characters(input_str)
        expected = "condition1__and__condition2__or__condition3"
        self.assertEqual(result, expected)

        # Content in quotes should be preserved
        input_str = "function( 'quoted  string', \"another  string\" )"
        result = remove_unwanted_characters(input_str)
        expected = "function('quoted  string',\"another  string\")"
        self.assertEqual(result, expected)

    def test_preprocess(self):
        """Test the complete preprocessing pipeline."""
        # Test with a complex expression
        input_str = """
        if [col1] == [col2] and length([col3]) > 5 // Check conditions
        then 
            concat([col1], ' ', [col2])  // Concat columns
        else 
            'Not matched' // Default value
        endif
        """
        result = preprocess(input_str)
        # This is a bit tricky to predict exactly due to the multiple transformations,
        # but we can check key aspects of the result
        self.assertIn("$if$", result)
        self.assertIn("pl.col(\"col1\")", result)
        self.assertIn("pl.col(\"col2\")", result)
        self.assertIn("$then$", result)
        self.assertIn("$else$", result)
        self.assertIn("$endif$", result)
        self.assertNotIn("//", result)  # Comments should be removed

        # Check a simpler case
        input_str = "[col1] + [col2] * 2"
        result = preprocess(input_str)
        expected = "pl.col(\"col1\")+pl.col(\"col2\")*2"
        self.assertEqual(result, expected)

        # Test with logical operators
        input_str = "[col1] > 0 and [col2] < 10"
        result = preprocess(input_str)
        self.assertIn("pl.col(\"col1\")>0", result)
        self.assertIn("pl.col(\"col2\")<10", result)
        self.assertIn(" and ", result)  # Space around logical operators should be preserved

#
# # Run the tests
# if __name__ == "__main__":
#     unittest.main()