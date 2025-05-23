import unittest
from unittest.mock import patch, MagicMock
import polars as pl
from polars_expr_transformer.process.models import Func, TempFunc, IfFunc, Classifier
from polars_expr_transformer.process.polars_expr_transformer import (
    finalize_hierarchy,
    build_func,
    simple_function_to_expr
)


class TestFinalizeHierarchy(unittest.TestCase):

    def test_finalize_hierarchy_with_temp_func_one_arg(self):
        """Test finalizing a TempFunc with one argument."""
        # Create a TempFunc with one argument
        func_arg = Func(Classifier("test_func"))
        temp_func = TempFunc()
        temp_func.add_arg(func_arg)

        result = finalize_hierarchy(temp_func)

        # Should return the sole argument
        self.assertEqual(result, func_arg)

    def test_finalize_hierarchy_with_temp_func_multiple_args(self):
        """Test finalizing a TempFunc with multiple arguments."""
        # Create a TempFunc with multiple arguments
        temp_func = TempFunc()
        temp_func.add_arg(Func(Classifier("func1")))
        temp_func.add_arg(Func(Classifier("func2")))

        # Should raise an exception
        with self.assertRaises(Exception):
            finalize_hierarchy(temp_func)

    def test_finalize_hierarchy_with_func(self):
        """Test finalizing a Func."""
        # Create a Func
        func = Func(Classifier("test_func"))

        result = finalize_hierarchy(func)

        # Should return the Func unchanged
        self.assertEqual(result, func)

    def test_finalize_hierarchy_with_if_func(self):
        """Test finalizing an IfFunc."""
        # Create an IfFunc
        if_func = IfFunc(Classifier("$if$"))

        result = finalize_hierarchy(if_func)

        # Should return the IfFunc unchanged
        self.assertEqual(result, if_func)


class TestBuildFunc(unittest.TestCase):

    @patch('polars_expr_transformer.process.polars_expr_transformer.preprocess')
    @patch('polars_expr_transformer.process.polars_expr_transformer.tokenize')
    @patch('polars_expr_transformer.process.polars_expr_transformer.classify_tokens')
    @patch('polars_expr_transformer.process.polars_expr_transformer.build_hierarchy')
    @patch('polars_expr_transformer.process.polars_expr_transformer.parse_inline_functions')
    @patch('polars_expr_transformer.process.polars_expr_transformer.finalize_hierarchy')
    def test_build_func_with_mocks(self, mock_finalize, mock_parse, mock_build,
                                   mock_classify, mock_tokenize, mock_preprocess):
        """Test the build_func function with mocked dependencies."""
        # Setup mocks
        mock_preprocess.return_value = "preprocessed"
        mock_tokenize.return_value = ["token1", "token2"]
        mock_classify.return_value = ["classified1", "classified2"]
        mock_hierarchical = MagicMock()
        mock_build.return_value = mock_hierarchical
        mock_final = MagicMock()
        mock_finalize.return_value = mock_final

        # Call the function
        result = build_func("test_func")

        # Verify the function call sequence
        mock_preprocess.assert_called_once_with("test_func")
        mock_tokenize.assert_called_once_with("preprocessed")
        mock_classify.assert_called_once_with(["token1", "token2"])
        mock_build.assert_called_once_with(["classified1", "classified2"])
        mock_parse.assert_called_once_with(mock_hierarchical)
        mock_finalize.assert_called_once_with(mock_hierarchical)

        # Verify the result
        self.assertEqual(result, mock_final)

    @patch('polars_expr_transformer.process.polars_expr_transformer.preprocess')
    @patch('polars_expr_transformer.process.polars_expr_transformer.tokenize')
    @patch('polars_expr_transformer.process.polars_expr_transformer.classify_tokens')
    @patch('polars_expr_transformer.process.polars_expr_transformer.build_hierarchy')
    @patch('polars_expr_transformer.process.polars_expr_transformer.parse_inline_functions')
    @patch('polars_expr_transformer.process.polars_expr_transformer.finalize_hierarchy')
    @patch('polars_expr_transformer.process.models.Func.get_pl_func')
    def test_build_func_integration(self, mock_get_pl_func, mock_finalize, mock_parse,
                                    mock_build, mock_classify, mock_tokenize, mock_preprocess):
        """Test build_func with minimal mocking to verify integration."""
        # Setup mocks
        mock_preprocess.return_value = "preprocessed"
        mock_tokenize.return_value = ["token1", "token2"]
        mock_classify.return_value = ["classified1", "classified2"]

        # Make get_pl_func return something simple to avoid errors
        mock_get_pl_func.return_value = pl.lit(1)

        # Create a real Func object but with mocked get_pl_func
        func = Func(Classifier("test_func"))
        mock_build.return_value = func
        mock_parse.return_value = func
        mock_finalize.return_value = func

        # Call the function
        result = build_func("test_func")

        # Verify basic interactions
        mock_preprocess.assert_called_once()
        mock_tokenize.assert_called_once()
        mock_classify.assert_called_once()
        mock_build.assert_called_once()
        mock_parse.assert_called_once()
        mock_finalize.assert_called_once()

        # Verify the get_pl_func was called
        mock_get_pl_func.assert_called()

        # Verify the result is a Func
        self.assertIsInstance(result, Func)
        self.assertEqual(result.func_ref.val, "test_func")


class TestSimpleFunctionToExpr(unittest.TestCase):

    @patch('polars_expr_transformer.process.polars_expr_transformer.build_func')
    def test_simple_function_to_expr(self, mock_build_func):
        """Test simple_function_to_expr function."""
        # Setup mock
        mock_func = MagicMock()
        mock_expr = pl.lit("test_result")
        mock_func.get_pl_func.return_value = mock_expr
        mock_build_func.return_value = mock_func

        # Call the function
        result = simple_function_to_expr("test_func")

        # Verify the function calls
        mock_build_func.assert_called_once_with("test_func")
        mock_func.get_pl_func.assert_called_once()

        # We cannot directly compare Polars expressions with assertEqual
        # Instead, let's verify by applying both expressions to a small DataFrame
        # and checking they produce the same result
        df = pl.DataFrame({"dummy": [1]})
        result_value = df.select(result).item()
        expected_value = df.select(mock_expr).item()
        self.assertEqual(result_value, expected_value)


class TestFunctionsToReadableExpr(unittest.TestCase):

    def test_simple_concat_function_to_readable_expr(self):
        f = build_func('concat("a", "b")')
        result_value = f.get_readable_pl_function()
        expected_value = 'concat(pl.lit("a"), pl.lit("b"))'
        self.assertEqual(result_value, expected_value)

    def test_constant_to_readable_expr(self):
        f = build_func('"hello world"')
        result_value = f.get_readable_pl_function()
        expected_value = 'pl.lit("hello world")'
        self.assertEqual(result_value, expected_value)

    def test_simple_if_else_statement_to_readable_expr(self):
        f = build_func('if 1>2 then "a" else "b" endif')
        result_value = f.get_readable_pl_function()
        expected_value = 'pl.when(pl.Expr.gt(pl.lit(1), pl.lit(2))).then(pl.lit("a")).otherwise(pl.lit("b"))'
        self.assertEqual(result_value, expected_value)

    def test_simple_inline_to_readable_pl_expr(self):
        f = build_func('1 + 2')
        result_value = f.get_readable_pl_function()
        expected_value = 'pl.Expr.add(pl.lit(1), pl.lit(2))'
        self.assertEqual(result_value, expected_value)

    def test_complex_inline_to_readable_pl_expr(self):
        f = build_func('1+2*10/(12-1*2)')
        result_value = f.get_readable_pl_function()
        expected_value = 'pl.Expr.add(pl.lit(1), pl.Expr.truediv(pl.Expr.mul(pl.lit(2), pl.lit(10)), pl.Expr.sub(pl.lit(12), pl.Expr.mul(pl.lit(1), pl.lit(2)))))'
        self.assertEqual(result_value, expected_value)

    def test_simple_column_to_readable_expr(self):
        f = build_func("[col_a]")
        result_value = f.get_readable_pl_function()
        expected_value = 'pl.col("col_a")'
        self.assertEqual(result_value, expected_value)

    def test_complex_nested_function(self):
        f = build_func('if 1+2*10/(12-1*2) > 100 then concat("true value", "hello world") else "a" + "b" endif')
        result_value = f.get_readable_pl_function()
        expected_value = 'pl.when(pl.Expr.gt(pl.Expr.add(pl.lit(1), pl.Expr.truediv(pl.Expr.mul(pl.lit(2), pl.lit(10)), pl.Expr.sub(pl.lit(12), pl.Expr.mul(pl.lit(1), pl.lit(2))))), pl.lit(100))).then(concat(pl.lit("true value"), pl.lit("hello world"))).otherwise(pl.Expr.add(pl.lit("a"), pl.lit("b")))'
        self.assertEqual(result_value, expected_value)

    def test_comments_in_readable_function(self):
        f = build_func('"this is a value"//this is a comment')
        result_value = f.get_readable_pl_function()
        expected_value = 'pl.lit("this is a value")'
        self.assertEqual(result_value, expected_value)

    def test_concat(self):
        f = build_func("contains([customer_name], 'John')")
        result_value = f.get_readable_pl_function()
        expected_value = 'contains(pl.col("customer_name"), pl.lit("John"))'
        self.assertEqual(result_value, expected_value)
