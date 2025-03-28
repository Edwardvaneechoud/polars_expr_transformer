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
    handle_seperator,
    handle_operator,
    build_hierarchy
)


class TestHandleOpeningBracket(unittest.TestCase):

    def test_handle_opening_bracket(self):
        # Test handling an opening bracket
        current_func = Func(Classifier("test_func"))
        previous_val = Classifier("test_func")

        result = handle_opening_bracket(current_func, previous_val)

        # Should add a new pl.lit function as argument and return it
        self.assertEqual(len(current_func.args), 1)
        self.assertEqual(current_func.args[0].func_ref.val, "pl.lit")
        self.assertEqual(result, current_func.args[0])


class TestHandleIf(unittest.TestCase):

    def test_handle_if_with_bracket(self):
        current_func = Func(Classifier("test_func"))
        current_val = Classifier("$if$")
        next_val = Classifier("(")
        pos = 0

        result, new_pos = handle_if(current_func, current_val, next_val, pos)

        # Should add a new IfFunc as argument, create a condition with condition and val parts,
        # and return the condition part
        self.assertEqual(len(current_func.args), 1)
        self.assertIsInstance(current_func.args[0], IfFunc)
        self.assertEqual(current_func.args[0].func_ref.val, "$if$")
        self.assertEqual(len(current_func.args[0].conditions), 1)
        self.assertIsNotNone(current_func.args[0].else_val)
        self.assertEqual(result, current_func.args[0].conditions[0].condition)
        self.assertEqual(new_pos, 1)  # Position incremented for next value

    def test_handle_if_without_bracket(self):
        current_func = Func(Classifier("test_func"))
        current_val = Classifier("$if$")
        next_val = Classifier("value")  # Not a bracket
        pos = 0

        # Should raise an exception when not followed by opening bracket
        with self.assertRaises(Exception) as context:
            handle_if(current_func, current_val, next_val, pos)

        self.assertTrue('Expected opening bracket' in str(context.exception))


class TestHandleThen(unittest.TestCase):

    def test_handle_then_in_condition_val_with_bracket(self):
        # Test handling 'then' in a ConditionVal context with bracket
        condition = Func(Classifier("pl.lit"))
        val = Func(Classifier("pl.lit"))
        condition_val = ConditionVal(condition=condition, val=val)
        current_val = Classifier("$then$")
        next_val = Classifier("(")
        pos = 0

        result, new_pos = handle_then(condition_val, current_val, next_val, pos)

        # Should set the func_ref of condition_val and return the val part
        self.assertEqual(condition_val.func_ref, current_val)
        self.assertEqual(result, val)
        self.assertEqual(new_pos, 1)  # Position incremented for next value

    def test_handle_then_in_condition_val_without_bracket(self):
        # Test handling 'then' in a ConditionVal context without bracket
        condition = Func(Classifier("pl.lit"))
        val = Func(Classifier("pl.lit"))
        condition_val = ConditionVal(condition=condition, val=val)
        current_val = Classifier("$then$")
        next_val = Classifier("value")  # Not a bracket
        pos = 0

        # Should raise an exception when not followed by opening bracket
        with self.assertRaises(Exception) as context:
            handle_then(condition_val, current_val, next_val, pos)

        self.assertTrue('Expected opening bracket' in str(context.exception))

    def test_handle_then_not_in_condition_val(self):
        # Test handling 'then' not in a ConditionVal context
        current_func = Func(Classifier("test_func"))
        current_val = Classifier("$then$")
        next_val = Classifier("(")
        pos = 0

        # Should raise an exception when not in a ConditionVal
        with self.assertRaises(Exception) as context:
            handle_then(current_func, current_val, next_val, pos)

        self.assertTrue('Expected to be in a condition val' in str(context.exception))


class TestHandleElse(unittest.TestCase):

    def test_handle_else_in_if_func_with_bracket(self):
        # Test handling 'else' in an IfFunc context with bracket
        if_func = IfFunc(Classifier("$if$"))
        else_val = Func(Classifier("pl.lit"))
        if_func.add_else_val(else_val)

        current_func = Func(Classifier("test_func"))
        current_func.parent = if_func

        next_val = Classifier("(")
        pos = 0

        result, new_pos = handle_else(current_func, next_val, pos)

        # Should return the else_val of the if_func and increment position
        self.assertEqual(result, else_val)
        self.assertEqual(new_pos, 1)  # Position incremented for next value

    def test_handle_else_without_if_func(self):
        # Test handling 'else' without an IfFunc parent
        current_func = Func(Classifier("test_func"))
        current_func.parent = Func(Classifier("parent_func"))  # Not an IfFunc

        next_val = Classifier("(")
        pos = 0

        # Should raise an exception
        with self.assertRaises(Exception) as context:
            handle_else(current_func, next_val, pos)

        self.assertTrue('Expected if' in str(context.exception))


class TestHandleElseIf(unittest.TestCase):

    def test_handle_elseif_in_if_func_with_bracket(self):
        # Test handling 'elseif' in an IfFunc context with bracket
        if_func = IfFunc(Classifier("$if$"))

        current_func = Func(Classifier("test_func"))
        current_func.parent = if_func

        current_val = Classifier("$elseif$")
        next_val = Classifier("(")
        pos = 0

        result, new_pos = handle_elseif(current_func, current_val, next_val, pos)

        # Should add a new condition to if_func and return the condition part
        self.assertEqual(len(if_func.conditions), 1)
        self.assertEqual(result, if_func.conditions[0].condition)
        self.assertEqual(new_pos, 1)  # Position incremented for next value

    def test_handle_elseif_in_if_func_without_bracket(self):
        # Test handling 'elseif' in an IfFunc context without bracket
        if_func = IfFunc(Classifier("$if$"))

        current_func = Func(Classifier("test_func"))
        current_func.parent = if_func

        current_val = Classifier("$elseif$")
        next_val = Classifier("value")  # Not a bracket
        pos = 0

        result, new_pos = handle_elseif(current_func, current_val, next_val, pos)

        # Should add a new condition to if_func and return the condition part without incrementing position
        self.assertEqual(len(if_func.conditions), 1)
        self.assertEqual(result, if_func.conditions[0].condition)
        self.assertEqual(new_pos, 0)  # Position not incremented

    def test_handle_elseif_without_if_func(self):
        # Test handling 'elseif' without an IfFunc parent
        current_func = Func(Classifier("test_func"))
        current_func.parent = Func(Classifier("parent_func"))  # Not an IfFunc

        current_val = Classifier("$elseif$")
        next_val = Classifier("(")
        pos = 0

        # Should raise an exception
        with self.assertRaises(Exception) as context:
            handle_elseif(current_func, current_val, next_val, pos)

        self.assertTrue('Expected if' in str(context.exception))


class TestHandleEndIf(unittest.TestCase):

    def test_handle_endif_in_if_func(self):
        # Test handling 'endif' in an IfFunc
        parent_func = Func(Classifier("parent_func"))
        if_func = IfFunc(Classifier("$if$"))
        if_func.parent = parent_func

        result = handle_endif(if_func)

        # Should return the parent of the if_func
        self.assertEqual(result, parent_func)

    def test_handle_endif_not_in_if_func(self):
        # Test handling 'endif' not in an IfFunc
        current_func = Func(Classifier("test_func"))

        # Should raise an exception
        with self.assertRaises(Exception) as context:
            handle_endif(current_func)

        self.assertTrue('Expected if' in str(context.exception))


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
        with self.assertRaises(Exception) as context:
            handle_closing_bracket(current_func, other_func)

        self.assertTrue('Expected parent' in str(context.exception))


class TestHandleFunction(unittest.TestCase):

    def test_handle_function_with_bracket(self):
        # Test handling a function with bracket
        current_func = Func(Classifier("test_func"))
        current_val = Classifier("new_func")
        next_val = Classifier("(")
        pos = 0

        result, new_pos = handle_function(current_func, current_val, next_val, pos)

        # Should add a new function as an argument and return it with incremented position
        self.assertEqual(len(current_func.args), 1)
        self.assertEqual(current_func.args[0].func_ref, current_val)
        self.assertEqual(result.parent, current_func.args[0])
        self.assertEqual(new_pos, 1)  # Position incremented for next value

    def test_handle_function_without_bracket(self):
        # Test handling a function without bracket (not negation)
        current_func = Func(Classifier("test_func"))
        current_val = Classifier("new_func")
        next_val = Classifier("value")  # Not a bracket
        pos = 0

        # Should raise an exception
        with self.assertRaises(Exception) as context:
            handle_function(current_func, current_val, next_val, pos)

        self.assertTrue('Expected opening bracket' in str(context.exception))

    def test_handle_function_negation(self):
        # Test handling a negation function without bracket
        current_func = Func(Classifier("test_func"))
        current_val = Classifier("negation")
        next_val = Classifier("value")  # Not a bracket
        pos = 0

        result, new_pos = handle_function(current_func, current_val, next_val, pos)

        # Should add a new function as an argument and return it without incrementing position
        self.assertEqual(len(current_func.args), 1)
        self.assertEqual(current_func.args[0].func_ref, current_val)
        self.assertEqual(result.parent, current_func.args[0])
        self.assertEqual(new_pos, 0)  # Position not incremented


class TestHandleLiteral(unittest.TestCase):

    def test_handle_literal(self):
        # Test handling a literal
        current_func = Func(Classifier("test_func"))
        current_val = Classifier("literal_value")

        handle_literal(current_func, current_val)

        # Should add the literal value as an argument
        self.assertEqual(len(current_func.args), 1)
        self.assertEqual(current_func.args[0], current_val)


class TestHandleSeparator(unittest.TestCase):

    def test_handle_separator(self):
        # Test handling a separator
        parent_func = Func(Classifier("parent_func"))
        current_func = TempFunc()
        current_func.parent = parent_func

        # Before adding any arguments to parent_func
        self.assertEqual(len(parent_func.args), 0)

        result = handle_seperator(current_func)

        # Should add a new TempFunc as argument to parent_func
        self.assertEqual(len(parent_func.args), 1)
        self.assertIsInstance(parent_func.args[0], TempFunc)

        # Should return the newly created TempFunc
        self.assertEqual(result, parent_func.args[0])
        self.assertNotEqual(id(result), id(current_func))


class TestHandleOperator(unittest.TestCase):

    def test_handle_operator(self):
        # Test handling an operator
        parent_func = Func(Classifier("parent_func"))
        current_func = Func(Classifier("test_func"))
        current_func.parent = parent_func
        current_val = Classifier("+")

        result = handle_operator(current_func, current_val)

        # Should add the operator to the parent and return the parent
        self.assertEqual(len(parent_func.args), 1)
        self.assertEqual(parent_func.args[0], current_val)
        self.assertEqual(result, parent_func)


class TestBuildHierarchy(unittest.TestCase):

    def test_build_hierarchy_with_function_first(self):
        # Test building a hierarchy with a function as the first token
        tokens = [
            Classifier("test_func"),
            Classifier("("),
            Classifier("arg1"),
            Classifier(")"),
        ]
        tokens[0].val_type = 'function'
        result = build_hierarchy(tokens)
        # result = main_result.args[0]
        # Should build a hierarchy with pl.lit as the root
        self.assertEqual(result.func_ref.val, "pl.lit")
        self.assertGreaterEqual(len(result.args), 1)

        # Check the nested function
        test_func = None
        for arg in result.args:
            if isinstance(arg, Func) and arg.func_ref.val == "test_func":
                test_func = arg
                break

        self.assertIsNotNone(test_func)
        self.assertEqual(len(test_func.args), 1)
        self.assertEqual(test_func.args[0].args[0].val, "arg1")

    def test_build_hierarchy_with_negation(self):
        # Test building a hierarchy with negation
        tokens = [
            Classifier("-", val_type="operator"),
            Classifier("5", val_type="number")
        ]

        result = build_hierarchy(tokens)

        # Should build a hierarchy with pl.lit as the root
        self.assertEqual(result.func_ref.val, "pl.lit")

        # Check for negation function
        self.assertGreaterEqual(len(result.args), 1)
        negation_func = None
        for arg in result.args:
            if isinstance(arg, Func) and arg.func_ref.val == "negation":
                negation_func = arg
                break

        self.assertIsNotNone(negation_func)
        self.assertEqual(len(negation_func.args), 1)
        self.assertEqual(negation_func.args[0].args[0].val, "5")

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

        result = build_hierarchy(tokens)

        # Should build a hierarchy with pl.lit as the root
        self.assertEqual(result.func_ref.val, "pl.lit")

        # Check for if function
        self.assertGreaterEqual(len(result.args), 1)
        if_func = None
        for arg in result.args:
            if isinstance(arg, IfFunc) and arg.func_ref.val == "$if$":
                if_func = arg
                break

        self.assertIsNotNone(if_func)
        self.assertEqual(len(if_func.conditions), 1)
        # Check condition
        self.assertEqual(if_func.conditions[0].condition.func_ref.val, "pl.lit")
        self.assertEqual(len(if_func.conditions[0].condition.args), 1)
        self.assertEqual(if_func.conditions[0].condition.args[0].val, "condition")
        # Check then value
        self.assertEqual(if_func.conditions[0].val.func_ref.val, "pl.lit")
        self.assertEqual(len(if_func.conditions[0].val.args), 1)
        self.assertEqual(if_func.conditions[0].val.args[0].val, "then_value")
        # Check else value
        self.assertEqual(if_func.else_val.func_ref.val, "pl.lit")
        self.assertEqual(len(if_func.else_val.args), 1)
        self.assertEqual(if_func.else_val.args[0].val, "else_value")

    def test_build_hierarchy_with_operators(self):
        # Test building a hierarchy with operators
        tokens = [
            Classifier("a"),
            Classifier("+", val_type="operator"),
            Classifier("b")
        ]

        result = build_hierarchy(tokens)

        # Should build a hierarchy with pl.lit as the root
        self.assertEqual(result.func_ref.val, "pl.lit")

        # Check arguments
        self.assertEqual(len(result.args), 3)
        self.assertEqual(result.args[0].val, "a")
        self.assertEqual(result.args[1].val, "+")
        self.assertEqual(result.args[2].val, "b")

    def test_build_hierarchy_with_negative_literal(self):
        # Test building a hierarchy with a special negative literal
        tokens = [
            Classifier("__negative()")
        ]

        result = build_hierarchy(tokens)

        # Should build a hierarchy with pl.lit as the root
        self.assertEqual(result.func_ref.val, "pl.lit")

        # Check arguments (should be -1)
        self.assertEqual(len(result.args), 1)
        self.assertEqual(result.args[0].val, "-1")
