import unittest
from unittest.mock import patch, MagicMock
from polars_expr_transformer.process.models import Classifier, Func, IfFunc, TempFunc, ConditionVal
from polars_expr_transformer.process.hierarchy_builder import (
    handle_opening_bracket,
    handle_if,
    handle_then,
    handle_else,
    handle_elseif,
    handle_endif,
    handle_closing_bracket,
    handle_function,
    handle_literal,
    build_hierarchy
)


class TestHandleOpeningBracket(unittest.TestCase):

    def test_handle_opening_bracket_with_function(self):
        # Test handling an opening bracket after a function
        current_func = Func(Classifier("test_func"))
        previous_val = Classifier("test_func")
        previous_val.val_type = "function"

        result = handle_opening_bracket(current_func, previous_val)

        # Should return the current function unchanged
        self.assertEqual(result, current_func)

    def test_handle_opening_bracket_with_if(self):
        # Test handling an opening bracket after an if statement
        if_func = IfFunc(Classifier("$if$"))
        previous_val = Classifier("$if$")

        result = handle_opening_bracket(if_func, previous_val)

        # Should add a condition to the if function and return the condition part
        self.assertEqual(len(if_func.conditions), 1)
        self.assertIsNotNone(if_func.else_val)
        self.assertEqual(result, if_func.conditions[0].condition)

    def test_handle_opening_bracket_with_elseif(self):
        # Test handling an opening bracket after an elseif statement
        if_func = IfFunc(Classifier("$elseif$"))
        previous_val = Classifier("$elseif$")

        result = handle_opening_bracket(if_func, previous_val)

        # Should add a condition to the if function and return the condition part
        self.assertEqual(len(if_func.conditions), 1)
        self.assertIsNone(if_func.else_val)  # No else_val for elseif
        self.assertEqual(result, if_func.conditions[0].condition)

    def test_handle_opening_bracket_with_other(self):
        # Test handling an opening bracket in other contexts
        current_func = Func(Classifier("test_func"))
        previous_val = Classifier("value")

        result = handle_opening_bracket(current_func, previous_val)

        # Should add a new argument function and return it
        self.assertEqual(len(current_func.args), 1)
        self.assertEqual(result, current_func.args[0])
        self.assertEqual(result.func_ref.val, "pl.lit")


class TestHandleIf(unittest.TestCase):

    def test_handle_if(self):
        current_func = Func(Classifier("test_func"))
        current_val = Classifier("$if$")

        result = handle_if(current_func, current_val)

        # Should add a new IfFunc and return it
        self.assertEqual(len(current_func.args), 1)
        self.assertIsInstance(current_func.args[0], IfFunc)
        self.assertEqual(result, current_func.args[0])
        self.assertEqual(result.func_ref.val, "$if$")


class TestHandleThen(unittest.TestCase):

    def test_handle_then_in_condition_val(self):
        # Test handling 'then' in a ConditionVal context
        condition = Func(Classifier("pl.lit"))
        val = Func(Classifier("pl.lit"))
        condition_val = ConditionVal(condition=condition, val=val)
        current_val = Classifier("$then$")
        next_val = Classifier("(")

        result, new_pos = handle_then(condition_val, current_val, next_val, 0)

        # Should set the func_ref of condition_val and return the val part
        self.assertEqual(condition_val.func_ref, current_val)
        self.assertEqual(result, val)
        self.assertEqual(new_pos, 1)  # Position incremented for next value

    def test_handle_then_in_condition_val_without_bracket(self):
        # Test handling 'then' in a ConditionVal context without following bracket
        condition = Func(Classifier("pl.lit"))
        val = Func(Classifier("pl.lit"))
        condition_val = ConditionVal(condition=condition, val=val)
        current_val = Classifier("$then$")
        next_val = Classifier("value")  # Not a bracket

        result, new_pos = handle_then(condition_val, current_val, next_val, 0)

        # Should set the func_ref of condition_val and return the val part
        self.assertEqual(condition_val.func_ref, current_val)
        self.assertEqual(result, val)
        self.assertEqual(new_pos, 0)  # Position not incremented


class TestHandleElse(unittest.TestCase):

    def test_handle_else_in_if_func(self):
        # Test handling 'else' in an IfFunc context
        if_func = IfFunc(Classifier("$if$"))
        else_val = Func(Classifier("pl.lit"))
        if_func.add_else_val(else_val)

        current_func = Func(Classifier("test_func"))
        current_func.parent = if_func

        next_val = Classifier("(")

        result, new_pos = handle_else(current_func, next_val, 0)

        # Should return the else_val of the if_func
        self.assertEqual(result, else_val)
        self.assertEqual(new_pos, 1)  # Position incremented for next value

    def test_handle_else_in_if_func_without_bracket(self):
        # Test handling 'else' in an IfFunc context without following bracket
        if_func = IfFunc(Classifier("$if$"))
        else_val = Func(Classifier("pl.lit"))
        if_func.add_else_val(else_val)

        current_func = Func(Classifier("test_func"))
        current_func.parent = if_func

        next_val = Classifier("value")  # Not a bracket

        result, new_pos = handle_else(current_func, next_val, 0)

        # Should return the else_val of the if_func
        self.assertEqual(result, else_val)
        self.assertEqual(new_pos, 0)  # Position not incremented

    def test_handle_else_without_if_func(self):
        # Test handling 'else' without an IfFunc parent
        current_func = Func(Classifier("test_func"))
        current_func.parent = Func(Classifier("parent_func"))  # Not an IfFunc

        next_val = Classifier("(")

        # Should raise an exception
        with self.assertRaises(Exception):
            handle_else(current_func, next_val, 0)


class TestHandleElseIf(unittest.TestCase):

    def test_handle_elseif_in_if_func(self):
        # Test handling 'elseif' in an IfFunc context
        if_func = IfFunc(Classifier("$if$"))

        current_func = Func(Classifier("test_func"))
        current_func.parent = if_func

        result = handle_elseif(current_func)

        # Should return the if_func parent
        self.assertEqual(result, if_func)

    def test_handle_elseif_without_if_func(self):
        # Test handling 'elseif' without an IfFunc parent
        current_func = Func(Classifier("test_func"))
        current_func.parent = Func(Classifier("parent_func"))  # Not an IfFunc

        # Should raise an exception
        with self.assertRaises(Exception):
            handle_elseif(current_func)


class TestHandleEndIf(unittest.TestCase):

    def test_handle_endif_in_if_func(self):
        # Test handling 'endif' in an IfFunc context
        parent_func = Func(Classifier("parent_func"))
        if_func = IfFunc(Classifier("$if$"))
        if_func.parent = parent_func

        result = handle_endif(if_func)

        # Should return the parent of the if_func
        self.assertEqual(result, parent_func)

    def test_handle_endif_without_if_func(self):
        # Test handling 'endif' without being in an IfFunc
        current_func = Func(Classifier("test_func"))

        # Should raise an exception
        with self.assertRaises(Exception):
            handle_endif(current_func)


class TestHandleClosingBracket(unittest.TestCase):

    def test_handle_closing_bracket_with_parent(self):
        # Test handling a closing bracket with a parent function
        parent_func = Func(Classifier("parent_func"))
        current_func = Func(Classifier("test_func"))
        current_func.parent = parent_func

        result, main_func = handle_closing_bracket(current_func, parent_func)

        # Should return the parent function
        self.assertEqual(result, parent_func)
        self.assertEqual(main_func, parent_func)

    def test_handle_closing_bracket_with_main_func(self):
        # Test handling a closing bracket at the main function level
        main_func = Func(Classifier("main_func"))

        result, new_main_func = handle_closing_bracket(main_func, main_func)

        # Should create a new TempFunc and add the main_func as an argument
        self.assertIsInstance(new_main_func, TempFunc)
        self.assertEqual(len(new_main_func.args), 1)
        self.assertEqual(new_main_func.args[0], main_func)
        self.assertEqual(result, new_main_func)

    def test_handle_closing_bracket_without_parent(self):
        # Test handling a closing bracket without a parent function
        current_func = Func(Classifier("test_func"))
        other_func = Func(Classifier("other_func"))

        # Should raise an exception
        with self.assertRaises(Exception):
            handle_closing_bracket(current_func, other_func)


class TestHandleFunction(unittest.TestCase):

    def test_handle_function(self):
        current_func = Func(Classifier("test_func"))
        current_val = Classifier("new_func")

        result = handle_function(current_func, current_val)

        # Should add a new function as an argument and return it
        self.assertEqual(len(current_func.args), 1)
        self.assertEqual(current_func.args[0].func_ref, current_val)
        self.assertEqual(result, current_func.args[0])


class TestHandleLiteral(unittest.TestCase):

    def test_handle_literal(self):
        current_func = Func(Classifier("test_func"))
        current_val = Classifier("literal_value")

        handle_literal(current_func, current_val)

        # Should add the literal value as an argument
        self.assertEqual(len(current_func.args), 1)
        self.assertEqual(current_func.args[0], current_val)


class TestBuildHierarchy(unittest.TestCase):

    def test_build_hierarchy_with_function_first(self):
        # Test building a hierarchy with a function as the first token
        tokens = [
            Classifier("test_func"),
            Classifier("("),
            Classifier("arg1"),
            Classifier(","),
            Classifier("arg2"),
            Classifier(")")
        ]
        tokens[0].val_type = "function"

        result = build_hierarchy(tokens)

        # Should build a function hierarchy
        self.assertEqual(result.func_ref.val, "test_func")
        self.assertEqual(len(result.args), 2)
        self.assertEqual(result.args[0].val, "arg1")
        self.assertEqual(result.args[1].val, "arg2")

    def test_build_hierarchy_with_non_function_first(self):
        # Test building a hierarchy with a non-function as the first token
        tokens = [
            Classifier("value"),
            Classifier("+"),
            Classifier("123")
        ]

        result = build_hierarchy(tokens)

        # Should build a hierarchy with pl.lit as the root
        self.assertEqual(result.func_ref.val, "pl.lit")
        self.assertEqual(len(result.args), 3)
        self.assertEqual(result.args[0].val, "value")
        self.assertEqual(result.args[1].val, "+")
        self.assertEqual(result.args[2].val, "123")

    def test_build_hierarchy_with_if_then_else(self):
        # Test building a hierarchy with if-then-else structure
        tokens = [
            Classifier("$if$"),
            Classifier("("),
            Classifier("condition"),
            Classifier(")"),
            Classifier("$then$"),
            Classifier("("),
            Classifier("then_value"),
            Classifier(")"),
            Classifier("$else$"),
            Classifier("("),
            Classifier("else_value"),
            Classifier(")"),
            Classifier("$endif$")
        ]

        with patch('polars_expr_transformer.process.hierarchy_builder.handle_if',
                   side_effect=handle_if) as mock_handle_if, \
                patch('polars_expr_transformer.process.hierarchy_builder.handle_then',
                      side_effect=handle_then) as mock_handle_then, \
                patch('polars_expr_transformer.process.hierarchy_builder.handle_else',
                      side_effect=handle_else) as mock_handle_else, \
                patch('polars_expr_transformer.process.hierarchy_builder.handle_endif',
                      side_effect=handle_endif) as mock_handle_endif:
            result = build_hierarchy(tokens)

            # Verify that the handler functions were called
            mock_handle_if.assert_called()
            mock_handle_then.assert_called()
            mock_handle_else.assert_called()
            mock_handle_endif.assert_called()

    def test_build_hierarchy_with_negation(self):
        # Test building a hierarchy with negation
        tokens = [
            Classifier("-"),
            Classifier("5")
        ]
        tokens[0].val_type = "operator"

        with patch('polars_expr_transformer.process.hierarchy_builder.handle_function',
                   side_effect=handle_function) as mock_handle_function:
            result = build_hierarchy(tokens)

            # Verify that handle_function was called with a negation classifier
            mock_handle_function.assert_called_with(
                result, Classifier("negation"))

