import unittest
from unittest.mock import patch, MagicMock
import inspect
from types import NotImplementedType
import polars as pl
from polars_expr_transformer.process.models import (
    get_types_from_func,
    ensure_all_numeric_types_align,
    all_numeric_types,
    allow_expressions,
    allow_non_pl_expressions,
    Classifier,
    Func,
    ConditionVal,
    IfFunc,
    TempFunc
)


class TestUtilityFunctions(unittest.TestCase):

    def test_get_types_from_func(self):
        def sample_func(a: int, b: str, c: float):
            pass

        types = get_types_from_func(sample_func)
        self.assertEqual(types, [int, str, float])

        # Test with a function that has default values
        def func_with_defaults(a: int, b: str = "default", c: float = 1.0):
            pass

        types = get_types_from_func(func_with_defaults)
        self.assertEqual(types, [int, str, float])

        # Test with a function that has no type annotations
        def func_no_annotations(a, b, c):
            pass

        types = get_types_from_func(func_no_annotations)
        self.assertEqual(types, [inspect._empty, inspect._empty, inspect._empty])

    def test_ensure_all_numeric_types_align(self):
        # Test with all integers
        numbers = [1, 2, 3, 4]
        result = ensure_all_numeric_types_align(numbers)
        self.assertEqual(result, [1, 2, 3, 4])
        self.assertTrue(all(isinstance(n, int) for n in result))

        # Test with all floats
        numbers = [1.0, 2.0, 3.0, 4.0]
        result = ensure_all_numeric_types_align(numbers)
        self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(all(isinstance(n, float) for n in result))

        # Test with mixed types (should convert all to float)
        numbers = [1, 2.0, 3, 4.0]
        result = ensure_all_numeric_types_align(numbers)
        self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(all(isinstance(n, float) for n in result))

        # Test with non-numeric types
        with self.assertRaises(Exception):
            ensure_all_numeric_types_align([1, 'a', 3])

    def test_all_numeric_types(self):
        # Test with all numeric types
        self.assertTrue(all_numeric_types([1, 2.0, True]))

        # Test with mixed numeric and non-numeric types
        self.assertFalse(all_numeric_types([1, 'a', 3]))

        # Test with all non-numeric types
        self.assertFalse(all_numeric_types(['a', 'b', 'c']))

        # Test with empty list
        self.assertTrue(all_numeric_types([]))

    def test_allow_expressions(self):
        from polars_expr_transformer.funcs.utils import PlStringType, PlIntType, PlNumericType

        # Test types that allow expressions
        self.assertTrue(allow_expressions(PlStringType))
        self.assertTrue(allow_expressions(PlIntType))
        self.assertTrue(allow_expressions(pl.Expr))
        self.assertTrue(allow_expressions(inspect._empty))
        self.assertTrue(allow_expressions(PlNumericType))

        # Test types that don't allow expressions
        self.assertFalse(allow_expressions(str))
        self.assertFalse(allow_expressions(int))
        self.assertFalse(allow_expressions(float))

    def test_allow_non_pl_expressions(self):
        from polars_expr_transformer.funcs.utils import PlStringType, PlIntType

        # Test types that allow non-polars expressions
        self.assertTrue(allow_non_pl_expressions(str))
        self.assertTrue(allow_non_pl_expressions(int))
        self.assertTrue(allow_non_pl_expressions(float))
        self.assertTrue(allow_non_pl_expressions(bool))
        self.assertTrue(allow_non_pl_expressions(PlIntType))
        self.assertTrue(allow_non_pl_expressions(PlStringType))

        # Test types that don't allow non-polars expressions
        self.assertFalse(allow_non_pl_expressions(pl.Expr))
        self.assertFalse(allow_non_pl_expressions(list))
        self.assertFalse(allow_non_pl_expressions(dict))


class TestClassifier(unittest.TestCase):

    def test_init_and_post_init(self):
        # Test initialization with different value types
        c1 = Classifier("123")
        self.assertEqual(c1.val, "123")
        self.assertEqual(c1.val_type, "number")

        c2 = Classifier("true")
        self.assertEqual(c2.val_type, "boolean")

        c3 = Classifier("+")
        self.assertEqual(c3.val_type, "operator")

        c4 = Classifier("(")
        self.assertEqual(c4.val_type, "prio")

        c5 = Classifier("")
        self.assertEqual(c5.val_type, "empty")

        c6 = Classifier("$if$")
        self.assertEqual(c6.val_type, "case_when")

        c7 = Classifier("-123")
        self.assertEqual(c7.val_type, "number")

        c8 = Classifier("__negative()")
        self.assertEqual(c8.val_type, "special")

        c9 = Classifier("variable")
        self.assertEqual(c9.val_type, "string")

        c10 = Classifier(",")
        self.assertEqual(c10.val_type, "sep")

        # Test with a value that doesn't fit any specific category
        c11 = Classifier("123abc")
        self.assertEqual(c11.val_type, "string")


    def test_get_pl_func(self):
        # Test for boolean values
        c1 = Classifier("true")
        self.assertTrue(c1.get_pl_func())

        c2 = Classifier("false")
        self.assertFalse(c2.get_pl_func())

        # Test for function

        c3 = Classifier("pl.col")
        self.assertEqual(c3.get_pl_func(), pl.col)

        # Test for number
        c4 = Classifier("123")
        self.assertEqual(c4.get_pl_func(), 123)

        # Test for string
        c5 = Classifier('"test"')
        self.assertEqual(c5.get_pl_func(), "test")

        # Test for __negative()

        c6 = Classifier("__negative()")
        assert pl.select(c6.get_pl_func()).equals(pl.select(pl.lit(-1)))

        # Test for unexpected value type
        c7 = Classifier("+")
        with self.assertRaises(Exception):
            c7.get_pl_func()

    def test_equality_and_hash(self):
        c1 = Classifier("test")
        c2 = Classifier("test")
        c3 = Classifier("different")

        # Test equality
        self.assertEqual(c1, c2)
        self.assertEqual(c1, "test")
        self.assertNotEqual(c1, c3)
        self.assertNotEqual(c1, "different")

        # Test hash
        self.assertEqual(hash(c1), hash("test"))


class TestFunc(unittest.TestCase):

    def setUp(self):
        self.classifier = Classifier("concat")
        self.func = Func(self.classifier)

    def test_init(self):
        self.assertEqual(self.func.func_ref, self.classifier)
        self.assertEqual(self.func.args, [])
        self.assertIsNone(self.func.parent)

    def test_add_arg(self):
        self.setUp()
        arg = Classifier("arg")
        self.func.add_arg(arg)
        self.assertEqual(self.func.args, [arg])
        self.assertEqual(arg.parent, self.func)

    def test_get_readable_pl_function(self):
        self.setUp()
        arg1 = Classifier("'a'")
        arg2 = Classifier("'b'")
        self.func.add_arg(arg1)
        self.func.add_arg(arg2)

        self.assertEqual(self.func.get_readable_pl_function(), "concat('a', 'b')")

    @patch('polars_expr_transformer.process.models.funcs')
    def test_get_pl_func_pl_lit(self, mock_funcs):
        mock_funcs.__getitem__.return_value = pl.lit

        # Test with pl.lit and one argument
        func_ref = Classifier("pl.lit")
        func = Func(func_ref)
        arg = MagicMock()
        arg.get_pl_func.return_value = "test"
        func.add_arg(arg)

        result = func.get_pl_func()
        func.add_arg(arg)
        with self.assertRaises(Exception):
            func.get_pl_func()

    @patch('polars_expr_transformer.process.models.funcs')
    @patch('polars_expr_transformer.process.models.all_numeric_types')
    @patch('polars_expr_transformer.process.models.ensure_all_numeric_types_align')
    def test_get_pl_func_non_pl_lit(self, mock_ensure, mock_all_numeric, mock_funcs):
        func_ref = Classifier("test_func")
        func = Func(func_ref)

        # Setup mocks
        arg1 = MagicMock()
        arg1.get_pl_func.return_value = 1
        arg2 = MagicMock()
        arg2.get_pl_func.return_value = 2
        func.add_arg(arg1)
        func.add_arg(arg2)

        mock_all_numeric.return_value = True
        mock_ensure.return_value = [1, 2]
        mock_func = MagicMock()
        mock_func.return_value = "result"
        mock_funcs.__getitem__.return_value = mock_func

        # Test function execution
        result = func.get_pl_func()
        self.assertEqual(result, "result")
        mock_all_numeric.assert_called_once_with([1, 2])
        mock_ensure.assert_called_once_with([1, 2])
        mock_func.assert_called_once_with(1, 2)

    @patch('polars_expr_transformer.process.models.funcs')
    @patch('polars_expr_transformer.process.models.all_numeric_types')
    def test_get_pl_func_mixed_args(self, mock_all_numeric, mock_funcs):
        func_ref = Classifier("test_func")
        func = Func(func_ref)

        # Setup mocks
        arg1 = MagicMock()
        arg1.get_pl_func.return_value = pl.lit(1)
        arg2 = MagicMock()
        arg2.get_pl_func.return_value = 2
        func.add_arg(arg1)
        func.add_arg(arg2)

        mock_all_numeric.return_value = False
        mock_func = MagicMock()
        mock_func.return_value = "result"
        mock_funcs.__getitem__.return_value = mock_func

        # Mock get_types_from_func
        with patch('polars_expr_transformer.process.models.get_types_from_func') as mock_get_types:
            mock_get_types.return_value = [int, int]

            # Mock allow_expressions to always return True
            with patch('polars_expr_transformer.process.models.allow_expressions', return_value=True):
                result = func.get_pl_func()
                self.assertEqual(result, "result")
                mock_func.assert_called_once()

    @patch('polars_expr_transformer.process.models.funcs')
    @patch('polars_expr_transformer.process.models.all_numeric_types')
    @patch('polars_expr_transformer.process.models.logging')
    def test_get_pl_func_not_implemented(self, mock_logging, mock_all_numeric, mock_funcs):
        func_ref = Classifier("test_func")
        func = Func(func_ref)

        # Setup mocks
        arg = MagicMock()
        arg.get_pl_func.return_value = 1
        func.add_arg(arg)

        mock_all_numeric.return_value = True
        mock_func = MagicMock()
        mock_func.return_value = NotImplementedType()
        mock_funcs.__getitem__.return_value = mock_func

        # Test handling of NotImplementedType
        result = func.get_pl_func()
        self.assertFalse(result)
        mock_logging.warning.assert_called_once()


class TestConditionVal(unittest.TestCase):

    def setUp(self):
        self.func_ref = Classifier("test_func")
        self.condition = MagicMock(spec=Func)
        self.val = MagicMock(spec=Func)
        self.condition_val = ConditionVal(self.func_ref, self.condition, self.val)

    def test_init(self):
        self.assertEqual(self.condition_val.func_ref, self.func_ref)
        self.assertEqual(self.condition_val.condition, self.condition)
        self.assertEqual(self.condition_val.val, self.val)
        self.assertIsNone(self.condition_val.parent)

        # Test that parent is correctly set for condition and val
        self.assertEqual(self.condition.parent, self.condition_val)
        self.assertEqual(self.val.parent, self.condition_val)

    @patch('polars_expr_transformer.process.models.pl')
    def test_get_pl_func(self, mock_pl):
        # Setup mocks
        self.condition.get_pl_func.return_value = "condition"
        self.val.get_pl_func.return_value = "value"

        # Mock the pl.when().then() chain
        mock_when = MagicMock()
        mock_pl.when.return_value = mock_when
        mock_then = MagicMock()
        mock_when.then.return_value = mock_then

        # Test get_pl_func
        result = self.condition_val.get_pl_func()

        mock_pl.when.assert_called_once_with("condition")
        mock_when.then.assert_called_once_with("value")
        self.assertEqual(result, mock_then)

    def test_get_pl_condition_and_val(self):
        # Setup mocks
        self.condition.get_pl_func.return_value = "condition"
        self.val.get_pl_func.return_value = "value"

        # Test get_pl_condition
        self.assertEqual(self.condition_val.get_pl_condition(), "condition")

        # Test get_pl_val
        self.assertEqual(self.condition_val.get_pl_val(), "value")


class TestIfFunc(unittest.TestCase):

    def setUp(self):
        self.func_ref = Classifier("$if$")
        self.if_func = IfFunc(self.func_ref)

    def test_init(self):
        self.assertEqual(self.if_func.func_ref, self.func_ref)
        self.assertEqual(self.if_func.conditions, [])
        self.assertIsNone(self.if_func.else_val)
        self.assertIsNone(self.if_func.parent)

    def test_add_condition(self):
        condition = MagicMock(spec=ConditionVal)
        self.if_func.add_condition(condition)

        self.assertEqual(self.if_func.conditions, [condition])
        self.assertEqual(condition.parent, self.if_func)

    def test_add_else_val(self):
        else_val = MagicMock(spec=Func)
        self.if_func.add_else_val(else_val)

        self.assertEqual(self.if_func.else_val, else_val)
        self.assertEqual(else_val.parent, self.if_func)

    @patch('polars_expr_transformer.process.models.pl')
    def test_get_pl_func_single_condition(self, mock_pl):
        # Setup mocks
        condition = MagicMock(spec=ConditionVal)
        condition.get_pl_condition.return_value = "condition"
        condition.get_pl_val.return_value = "value"

        else_val = MagicMock(spec=Func)
        else_val.get_pl_func.return_value = "else_value"

        self.if_func.add_condition(condition)
        self.if_func.add_else_val(else_val)

        # Mock the pl.when().then().otherwise() chain
        mock_when = MagicMock()
        mock_pl.when.return_value = mock_when
        mock_then = MagicMock()
        mock_when.then.return_value = mock_then
        mock_otherwise = MagicMock()
        mock_then.otherwise.return_value = mock_otherwise

        # Test get_pl_func
        result = self.if_func.get_pl_func()

        mock_pl.when.assert_called_once_with("condition")
        mock_when.then.assert_called_once_with("value")
        mock_then.otherwise.assert_called_once_with("else_value")
        self.assertEqual(result, mock_otherwise)

    @patch('polars_expr_transformer.process.models.pl')
    def test_get_pl_func_multiple_conditions(self, mock_pl):
        # Setup mocks
        condition1 = MagicMock(spec=ConditionVal)
        condition1.get_pl_condition.return_value = "condition1"
        condition1.get_pl_val.return_value = "value1"

        condition2 = MagicMock(spec=ConditionVal)
        condition2.get_pl_condition.return_value = "condition2"
        condition2.get_pl_val.return_value = "value2"

        else_val = MagicMock(spec=Func)
        else_val.get_pl_func.return_value = "else_value"

        self.if_func.add_condition(condition1)
        self.if_func.add_condition(condition2)
        self.if_func.add_else_val(else_val)

        # Mock the pl.when().then().when().then().otherwise() chain
        mock_when1 = MagicMock()
        mock_pl.when.return_value = mock_when1
        mock_then1 = MagicMock()
        mock_when1.then.return_value = mock_then1
        mock_when2 = MagicMock()
        mock_then1.when.return_value = mock_when2
        mock_then2 = MagicMock()
        mock_when2.then.return_value = mock_then2
        mock_otherwise = MagicMock()
        mock_then2.otherwise.return_value = mock_otherwise

        # Test get_pl_func
        result = self.if_func.get_pl_func()

        mock_pl.when.assert_called_once_with("condition1")
        mock_when1.then.assert_called_once_with("value1")
        mock_then1.when.assert_called_once_with("condition2")
        mock_when2.then.assert_called_once_with("value2")
        mock_then2.otherwise.assert_called_once_with("else_value")
        self.assertEqual(result, mock_otherwise)

    def test_get_pl_func_no_conditions(self):
        else_val = MagicMock(spec=Func)
        self.if_func.add_else_val(else_val)

        # Test that get_pl_func raises an Exception when there are no conditions
        with self.assertRaises(Exception):
            self.if_func.get_pl_func()


class TestTempFunc(unittest.TestCase):

    def setUp(self):
        self.temp_func = TempFunc()

    def test_init(self):
        self.assertEqual(self.temp_func.args, [])

    def test_add_arg(self):
        arg = MagicMock()
        self.temp_func.add_arg(arg)

        self.assertEqual(self.temp_func.args, [arg])
        self.assertEqual(arg.parent, self.temp_func)

