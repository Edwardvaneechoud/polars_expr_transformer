from typing import List, Union, Any
from polars_expr_transformer.configs.settings import operators
from polars_expr_transformer.process.models import IfFunc, Classifier, Func, TempFunc
from polars_expr_transformer.process.hierarchy_builder import build_hierarchy


def parse_inline_functions(formula: Func):
    """
    Process a formula containing inline operators and convert them to proper function calls.

    Args:
        formula: The hierarchical formula to parse.
    """
    # Track whether we need to continue processing
    any_changes = [True]

    def process_formula(f: Union[Func, IfFunc, TempFunc]):
        """
        Process a single formula structure.

        Args:
            f: The formula to process.

        Returns:
            The processed formula.
        """
        # Check if this formula contains operators
        if isinstance(f, Func) and any(isinstance(arg, Classifier) and arg.val_type == 'operator'
                                       for arg in f.args):
            # This formula contains operators and needs processing
            f.args = [build_operator_tree(f.args)]
            any_changes[0] = True
            return f

        # Process nested formulas
        if isinstance(f, Func):
            for i in range(len(f.args)):
                if isinstance(f.args[i], (Func, IfFunc, TempFunc)):
                    f.args[i] = process_formula(f.args[i])
        elif isinstance(f, IfFunc):
            for cond in f.conditions:
                if isinstance(cond.condition, (Func, IfFunc, TempFunc)):
                    cond.condition = process_formula(cond.condition)
                if isinstance(cond.val, (Func, IfFunc, TempFunc)):
                    cond.val = process_formula(cond.val)
            if isinstance(f.else_val, (Func, IfFunc, TempFunc)):
                f.else_val = process_formula(f.else_val)
        elif isinstance(f, TempFunc):
            for i in range(len(f.args)):
                if isinstance(f.args[i], (Func, IfFunc, TempFunc)):
                    f.args[i] = process_formula(f.args[i])

        return f

    # Keep processing until no more changes are made
    while any_changes[0]:
        any_changes[0] = False
        formula = process_formula(formula)

    return formula


def build_operator_tree(tokens: List[Any]) -> Func:
    """
    Build a tree of function calls from a list of tokens containing operators.

    This function handles operator precedence correctly by building the tree bottom-up.

    Args:
        tokens: List of tokens potentially containing operators.

    Returns:
        A Func object representing the operator tree.
    """
    # Define operator precedence (higher number = higher precedence)
    # Make a copy of the tokens to avoid modifying the original
    tokens = tokens.copy()

    # First pass: handle highest precedence operators (multiplication, division)
    i = 0
    while i < len(tokens):
        if isinstance(tokens[i], Classifier) and tokens[i].val_type == 'operator' and tokens[i].precedence == 3:
            # Found a high precedence operator (*, /)
            if i > 0 and i < len(tokens) - 1:
                # Get left and right operands
                left_operand = tokens[i - 1]
                right_operand = tokens[i + 1]

                # Create the function call
                op_func = operators.get(tokens[i].val)
                if op_func:
                    func_call = Func(
                        func_ref=Classifier(op_func),
                        args=[left_operand, right_operand]
                    )

                    # Replace the three tokens with the function call
                    tokens[i - 1:i + 2] = [func_call]

                    # Adjust index to account for the modification
                    i -= 1
        i += 1

    # Second pass: handle lower precedence operators (addition, subtraction, comparison)
    i = 0
    while i < len(tokens):
        if isinstance(tokens[i], Classifier) and tokens[i].val_type == 'operator' and tokens[i].precedence != 3:
            # Found a lower precedence operator (+, -, >, <, etc.)
            if i > 0 and i < len(tokens) - 1:
                # Get left and right operands
                left_operand = tokens[i - 1]
                right_operand = tokens[i + 1]

                # Create the function call
                op_func = operators.get(tokens[i].val)
                if op_func:
                    func_call = Func(
                        func_ref=Classifier(op_func),
                        args=[left_operand, right_operand]
                    )

                    # Replace the three tokens with the function call
                    tokens[i - 1:i + 2] = [func_call]

                    # Adjust index to account for the modification
                    i -= 1
        i += 1

    # At this point, tokens should be reduced to a single Func
    if len(tokens) == 1 and isinstance(tokens[0], Func):
        return tokens[0]

    # If not reduced to a single Func, wrap in a Func
    return Func(
        func_ref=Classifier("pl.lit"),
        args=tokens
    )