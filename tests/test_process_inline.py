import unittest
from unittest.mock import patch, MagicMock
import pytest
from typing import List, Union, Any

from polars_expr_transformer.configs.settings import operators
from polars_expr_transformer.process.models import IfFunc, Classifier, Func, TempFunc
from polars_expr_transformer.process.hierarchy_builder import build_hierarchy
from polars_expr_transformer.process.process_inline import parse_inline_functions, build_operator_tree


class TestParseInlineFunctions(unittest.TestCase):

    def setUp(self):
        # Common test objects
        self.plus_op = Classifier('+', val_type='operator')
        self.minus_op = Classifier('-', val_type='operator')
        self.mult_op = Classifier('*', val_type='operator')
        self.div_op = Classifier('/', val_type='operator')
        self.and_op = Classifier('and', val_type='operator')
        self.or_op = Classifier('or', val_type='operator')
        self.gt_op = Classifier('>', val_type='operator')
        self.lt_op = Classifier('<', val_type='operator')
        self.eq_op = Classifier('==', val_type='operator')

        self.lit_func = Classifier("pl.lit", val_type='function')
        self.add_func = Classifier("pl.add", val_type='function')
        self.sub_func = Classifier("pl.sub", val_type='function')
        self.mul_func = Classifier("pl.mul", val_type='function')
        self.div_func = Classifier("pl.div", val_type='function')

        # Mock operators dictionary
        self.operators_patch = patch('polars_expr_transformer.process.process_inline.operators', {
            '+': 'pl.add',
            '-': 'pl.sub',
            '*': 'pl.mul',
            '/': 'pl.div',
            'and': 'pl.and',
            'or': 'pl.or',
            '>': 'pl.gt',
            '<': 'pl.lt',
            '==': 'pl.eq',
        })
        self.mock_operators = self.operators_patch.start()

    def tearDown(self):
        self.operators_patch.stop()

    def test_parse_simple_tempfunc_with_operators(self):
        """Test parsing a simple TempFunc with a single operator."""
        # Create a simple expression: a + b
        a = Classifier('a', val_type='column')
        b = Classifier('b', val_type='column')
        temp_func = TempFunc(args=[a, self.plus_op, b])

        # Expected result: Func(pl.add, [a, b])
        expected = Func(
            func_ref=self.add_func,
            args=[a, b]
        )

        result = parse_inline_functions(temp_func)
        self.assertEqual(type(result), Func)
        self.assertEqual(result.func_ref.val, expected.func_ref.val)
        self.assertEqual(len(result.args), len(expected.args))
        self.assertEqual(result.args[0].val, expected.args[0].val)
        self.assertEqual(result.args[1].val, expected.args[1].val)

    def test_parse_complex_tempfunc_with_operators(self):
        """Test parsing a complex TempFunc with multiple operators and precedence."""
        # Create a complex expression: a + b * c - d
        a = Classifier('a', val_type='column')
        b = Classifier('b', val_type='column')
        c = Classifier('c', val_type='column')
        d = Classifier('d', val_type='column')

        temp_func = TempFunc(args=[a, self.plus_op, b, self.mult_op, c, self.minus_op, d])

        result = parse_inline_functions(temp_func)

        # The result should be a nested structure respecting operator precedence
        self.assertEqual(type(result), Func)
        self.assertEqual(result.func_ref.val, 'pl.sub')  # Outermost operation should be subtraction

        # First arg of subtraction should be an addition operation
        add_operation = result.args[0]
        self.assertEqual(add_operation.func_ref.val, 'pl.add')
        self.assertEqual(add_operation.args[0].val, 'a')

        # Second arg of addition should be a multiplication operation
        mul_operation = add_operation.args[1]
        self.assertEqual(mul_operation.func_ref.val, 'pl.mul')
        self.assertEqual(mul_operation.args[0].val, 'b')
        self.assertEqual(mul_operation.args[1].val, 'c')

        # Second arg of subtraction should be 'd'
        self.assertEqual(result.args[1].val, 'd')

    def test_parse_func_with_operators(self):
        """Test parsing a Func that contains operators in its arguments."""
        # Create a function with operators in its args: some_func(a + b)
        a = Classifier('a', val_type='column')
        b = Classifier('b', val_type='column')

        func = Func(
            func_ref=Classifier('some_func', val_type='function'),
            args=[a, self.plus_op, b]
        )

        result = parse_inline_functions(func)

        # The result should be a Func with an argument that is also a Func
        self.assertEqual(type(result), Func)
        self.assertEqual(result.func_ref.val, 'some_func')
        self.assertEqual(len(result.args), 1)

        # The argument should be an addition operation
        add_operation = result.args[0]
        self.assertEqual(add_operation.func_ref.val, 'pl.add')
        self.assertEqual(add_operation.args[0].val, 'a')
        self.assertEqual(add_operation.args[1].val, 'b')

    def test_parse_iffunc_with_operators(self):
        """Test parsing an IfFunc with operators in conditions and values."""
        # Create an if-else structure: if a > b then c + d else e
        a = Classifier('a', val_type='column')
        b = Classifier('b', val_type='column')
        c = Classifier('c', val_type='column')
        d = Classifier('d', val_type='column')
        e = Classifier('e', val_type='column')

        condition = TempFunc(args=[a, self.gt_op, b])
        value = TempFunc(args=[c, self.plus_op, d])

        if_func = IfFunc(
            conditions=[MagicMock(condition=condition, val=value)],
            else_val=e,
            func_ref=Classifier('if', val_type='function')
        )

        result = parse_inline_functions(if_func)

        # The result should be an IfFunc with processed condition and value
        self.assertEqual(type(result), IfFunc)

        # Check condition
        processed_condition = result.conditions[0].condition
        self.assertEqual(processed_condition.func_ref.val, 'pl.gt')
        self.assertEqual(processed_condition.args[0].val, 'a')
        self.assertEqual(processed_condition.args[1].val, 'b')

        # Check value
        processed_value = result.conditions[0].val
        self.assertEqual(processed_value.func_ref.val, 'pl.add')
        self.assertEqual(processed_value.args[0].val, 'c')
        self.assertEqual(processed_value.args[1].val, 'd')

        # Check else value
        self.assertEqual(result.else_val.val, 'e')

    def test_parse_nested_functions(self):
        """Test parsing nested functions with operators."""
        # Create nested functions: outer_func(inner_func(a + b))
        a = Classifier('a', val_type='column')
        b = Classifier('b', val_type='column')

        inner_func = Func(
            func_ref=Classifier('inner_func', val_type='function'),
            args=[a, self.plus_op, b]
        )

        outer_func = Func(
            func_ref=Classifier('outer_func', val_type='function'),
            args=[inner_func]
        )

        result = parse_inline_functions(outer_func)

        # The result should be a Func with an argument that is also a Func
        self.assertEqual(type(result), Func)
        self.assertEqual(result.func_ref.val, 'outer_func')
        self.assertEqual(len(result.args), 1)

        # The inner function should be processed
        inner_result = result.args[0]
        self.assertEqual(inner_result.func_ref.val, 'inner_func')
        self.assertEqual(len(inner_result.args), 1)

        # The argument of inner function should be an addition operation
        add_operation = inner_result.args[0]
        self.assertEqual(add_operation.func_ref.val, 'pl.add')
        self.assertEqual(add_operation.args[0].val, 'a')
        self.assertEqual(add_operation.args[1].val, 'b')

    def test_operator_precedence(self):
        """Test that operator precedence is correctly handled."""
        # Test expression: a + b * c + d / e
        a = Classifier('a', val_type='column')
        b = Classifier('b', val_type='column')
        c = Classifier('c', val_type='column')
        d = Classifier('d', val_type='column')
        e = Classifier('e', val_type='column')

        temp_func = TempFunc(args=[
            a, self.plus_op, b, self.mult_op, c, self.plus_op, d, self.div_op, e
        ])

        result = parse_inline_functions(temp_func)

        # Expected structure:
        # (a + (b * c)) + (d / e)
        # Outermost operation should be addition
        self.assertEqual(result.func_ref.val, 'pl.add')

        # First argument should be another addition
        first_add = result.args[0]
        self.assertEqual(first_add.func_ref.val, 'pl.add')
        self.assertEqual(first_add.args[0].val, 'a')

        # Second argument of first addition should be multiplication
        mult = first_add.args[1]
        self.assertEqual(mult.func_ref.val, 'pl.mul')
        self.assertEqual(mult.args[0].val, 'b')
        self.assertEqual(mult.args[1].val, 'c')

        # Second argument of outermost addition should be division
        div = result.args[1]
        self.assertEqual(div.func_ref.val, 'pl.div')
        self.assertEqual(div.args[0].val, 'd')
        self.assertEqual(div.args[1].val, 'e')

    def test_empty_tokens(self):
        """Test that build_operator_tree handles empty token lists."""
        result = build_operator_tree([])
        self.assertIsNone(result)

    def test_single_token(self):
        """Test that build_operator_tree handles a list with a single token."""
        a = Classifier('a', val_type='column')
        result = build_operator_tree([a])

        # Should wrap single token in a lit function
        self.assertEqual(type(result), Func)
        self.assertEqual(result.func_ref.val, 'pl.lit')
        self.assertEqual(len(result.args), 1)
        self.assertEqual(result.args[0].val, 'a')

    def test_logical_operators(self):
        """Test parsing expressions with logical operators."""
        # Test expression: a > b and c < d
        a = Classifier('a', val_type='column')
        b = Classifier('b', val_type='column')
        c = Classifier('c', val_type='column')
        d = Classifier('d', val_type='column')

        temp_func = TempFunc(args=[
            a, self.gt_op, b, self.and_op, c, self.lt_op, d
        ])

        result = parse_inline_functions(temp_func)

        # Outermost operation should be 'and'
        self.assertEqual(result.func_ref.val, 'pl.and')

        # First argument should be a > b
        gt_op = result.args[0]
        self.assertEqual(gt_op.func_ref.val, 'pl.gt')
        self.assertEqual(gt_op.args[0].val, 'a')
        self.assertEqual(gt_op.args[1].val, 'b')

        # Second argument should be c < d
        lt_op = result.args[1]
        self.assertEqual(lt_op.func_ref.val, 'pl.lt')
        self.assertEqual(lt_op.args[0].val, 'c')
        self.assertEqual(lt_op.args[1].val, 'd')

    def test_parenthesized_expressions(self):
        """Test parsing expressions with parentheses."""
        # Test expression: (a + b) * c
        a = Classifier('a', val_type='column')
        b = Classifier('b', val_type='column')
        c = Classifier('c', val_type='column')

        # Create parenthesized expression using a lit function
        parenthesized = Func(
            func_ref=self.lit_func,
            args=[a, self.plus_op, b]
        )

        temp_func = TempFunc(args=[
            parenthesized, self.mult_op, c
        ])

        result = parse_inline_functions(temp_func)

        # Outermost operation should be multiplication
        self.assertEqual(result.func_ref.val, 'pl.mul')

        # First argument should be a + b
        add_op = result.args[0]
        self.assertEqual(add_op.func_ref.val, 'pl.add')
        self.assertEqual(add_op.args[0].val, 'a')
        self.assertEqual(add_op.args[1].val, 'b')

        # Second argument should be c
        self.assertEqual(result.args[1].val, 'c')

    def test_infinite_recursion_prevention(self):
        """Test that parse_inline_functions prevents infinite recursion."""
        # Create a self-referential structure
        a = Classifier('a', val_type='column')
        func = Func(
            func_ref=Classifier('some_func', val_type='function'),
            args=[a]
        )

        # Create a circular reference
        func.args.append(func)  # This creates a circular reference

        # Should not cause infinite recursion
        try:
            result = parse_inline_functions(func)
            # If we get here, no infinite recursion occurred
            self.assertTrue(True)
        except RecursionError:
            self.fail("parse_inline_functions entered infinite recursion")

