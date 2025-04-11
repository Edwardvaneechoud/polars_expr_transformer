from typing import List, Union, Any
from polars_expr_transformer.configs.settings import operators, PRECEDENCE
from polars_expr_transformer.process.models import IfFunc, Classifier, Func, TempFunc
from polars_expr_transformer.process.hierarchy_builder import build_hierarchy


def parse_inline_functions(formula: Union[Func, TempFunc, IfFunc]):
    """
    Process a formula containing inline operators and convert them to proper function calls.

    Args:
        formula: The hierarchical formula to parse.
    """
    any_changes = [True]
    processed_objects = set()

    def process_formula(f: Union[Func, IfFunc, TempFunc]):
        """
        Process a single formula structure.

        Args:
            f: The formula to process.

        Returns:
            The processed formula.
        """
        obj_id = id(f)
        if obj_id in processed_objects:
            return f
        processed_objects.add(obj_id)

        if isinstance(f, TempFunc):
            if any(isinstance(arg, Classifier) and arg.val_type == 'operator' for arg in f.args):
                result = build_operator_tree(f.args)
                any_changes[0] = True
                return result

            for i in range(len(f.args)):
                if isinstance(f.args[i], (Func, IfFunc, TempFunc)):
                    f.args[i] = process_formula(f.args[i])
            return f

        # Handle operators in Func
        elif isinstance(f, Func):
            if any(isinstance(arg, Classifier) and arg.val_type == 'operator' for arg in f.args):
                result = build_operator_tree(f.args)
                f.args = [result] if result else f.args
                any_changes[0] = True
                return f

            for i in range(len(f.args)):
                if isinstance(f.args[i], (Func, IfFunc, TempFunc)):
                    f.args[i] = process_formula(f.args[i])
            return f

        # Handle operators in IfFunc
        elif isinstance(f, IfFunc):
            for cond in f.conditions:
                if isinstance(cond.condition, (Func, IfFunc, TempFunc)):
                    cond.condition = process_formula(cond.condition)
                if isinstance(cond.val, (Func, IfFunc, TempFunc)):
                    cond.val = process_formula(cond.val)

            if isinstance(f.else_val, (Func, IfFunc, TempFunc)):
                f.else_val = process_formula(f.else_val)

            return f

        return f

    while any_changes[0]:
        any_changes[0] = False
        processed_objects.clear()
        formula = process_formula(formula)

    return formula

def build_operator_tree(tokens: List[Any]) -> Func:
    """
    Build a tree of function calls from a list of tokens containing operators.

    This function uses a recursive descent parser approach to build the tree
    with correct operator precedence.

    Args:
        tokens: List of tokens potentially containing operators.

    Returns:
        A Func object representing the operator tree.
    """
    if not tokens:
        return None

    tokens = tokens.copy()

    def parse_expression(token_list, min_precedence=0):
        """Recursive helper function to parse expression with operator precedence."""
        # First get the left-hand side
        left = parse_primary(token_list)

        while token_list and is_operator(token_list[0]) and get_precedence(token_list[0]) >= min_precedence:
            op = token_list.pop(0)
            op_precedence = get_precedence(op)

            # The key fix: For operators with same precedence, use > instead of >= to ensure proper left-to-right association
            # This prevents higher precedence operators from being incorrectly nested under lower precedence ones
            next_min_precedence = op_precedence + 1 if op.val in ['and', 'or'] else op_precedence

            right = parse_expression(token_list, next_min_precedence)

            op_func = operators.get(op.val)
            if op_func:
                left = Func(
                    func_ref=Classifier(op_func, val_type='function'),
                    args=[left, right]
                )

        return left

    def parse_primary(token_list):
        """Parse a primary expression (a value or nested expression)."""
        if not token_list:
            return None

        if isinstance(token_list[0], Func) and token_list[0].func_ref.val == 'pl.lit':
            inner_tokens = token_list.pop(0).args
            return parse_expression(inner_tokens)

        return token_list.pop(0)

    def is_operator(token):
        """Check if token is an operator."""
        return isinstance(token, Classifier) and token.val_type == 'operator'

    def get_precedence(token):
        """Get precedence of operator token."""
        if is_operator(token):
            # Fix: Ensure all operators have a precedence, defaulting to high precedence if not specified
            return PRECEDENCE.get(token.val, 10)
        return 0

    result = parse_expression(tokens)

    if not isinstance(result, Func):
        result = Func(
            func_ref=Classifier("pl.lit", val_type='function'),
            args=[result]
        )

    return result