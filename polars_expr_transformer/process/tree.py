from typing import Optional, List
from polars_expr_transformer.process.models import Classifier, Func, IfFunc, TempFunc, ConditionVal
from copy import deepcopy


def handle_opening_bracket(current_func: Func, previous_val: Classifier) -> Func:
    """
    Handle the opening bracket in the function hierarchy.

    Args:
        current_func: The current function being processed.
        previous_val: The previous classifier value.

    Returns:
        The updated current function.
    """
    if previous_val and previous_val.val_type == 'function':
        pass
    elif isinstance(current_func, IfFunc) and previous_val.val in ('$if$', '$elseif$'):
        condition = Func(Classifier('pl.lit'))
        val = Func(Classifier('pl.lit'))
        condition_val = ConditionVal(condition=condition, val=val)
        if previous_val.val == '$if$':
            else_val = Func(Classifier('pl.lit'))
            current_func.add_else_val(else_val)
        current_func.add_condition(condition_val)
        current_func = condition
    else:
        new_func = Func(Classifier('pl.lit'))
        current_func.add_arg(new_func)
        current_func = new_func
    return current_func


def handle_if(current_func: Func, current_val: Classifier) -> IfFunc | Func:
    """
    Handle the if condition in the function hierarchy.

    Args:
        current_func: The current function being processed.
        current_val: The current classifier value.

    Returns:
        The updated current function as IfFunc or Func.
    """
    new_func = IfFunc(current_val)
    current_func.add_arg(new_func)
    current_func = new_func
    return current_func


def handle_then(current_func: Func, current_val: Classifier, next_val: Optional[Classifier], pos: int) -> (Func, int):
    """
    Handle the then condition in the function hierarchy.

    Args:
        current_func: The current function being processed.
        current_val: The current classifier value.
        next_val: The next classifier value.
        pos: The current position in the tokens list.

    Returns:
        The updated current function and position.
    """
    if isinstance(current_func, ConditionVal):
        current_func.func_ref = current_val
        current_func = current_func.val
        if next_val and next_val.val == '(':
            pos += 1
    return current_func, pos


def handle_else(current_func: Func, next_val: Optional[Classifier], pos: int) -> (Func, int):
    """
    Handle the else condition in the function hierarchy.

    Args:
        current_func: The current function being processed.
        next_val: The next classifier value.
        pos: The current position in the tokens list.

    Returns:
        The updated current function and position.
    """
    current_func = current_func.parent
    if isinstance(current_func, IfFunc):
        current_func = current_func.else_val
        if next_val and next_val.val == '(':
            pos += 1
    else:
        raise Exception('Expected if')
    return current_func, pos


def handle_elseif(current_func: Func) -> Func | IfFunc:
    """
    Handle the elseif condition in the function hierarchy.

    Args:
        current_func: The current function being processed.

    Returns:
        The updated current function as IfFunc or Func.
    """
    current_func = current_func.parent
    if not isinstance(current_func, IfFunc):
        raise Exception('Expected if')
    return current_func


def handle_endif(current_func: Func) -> Func:
    """
    Handle the endif condition in the function hierarchy.

    Args:
        current_func: The current function being processed.

    Returns:
        The updated current function.
    """
    if isinstance(current_func, IfFunc):
        current_func = current_func.parent
    else:
        raise Exception('Expected if')
    return current_func


def handle_closing_bracket(current_func: Func, main_func: Func) -> (Func, Func):
    """
    Handle the closing bracket in the function hierarchy.

    Args:
        current_func: The current function being processed.
        main_func: The main function being processed.

    Returns:
        The updated current function and main function.
    """
    if current_func.parent is None and current_func == main_func:
        new_main_func = TempFunc()
        new_main_func.add_arg(main_func)
        main_func = current_func = new_main_func
    elif current_func.parent is not None:
        current_func = current_func.parent
    else:
        raise Exception('Expected parent')
    return current_func, main_func


def handle_function(current_func: Func, current_val: Classifier) -> Func:
    """
    Handle a function token in the function hierarchy.

    Args:
        current_func: The current function being processed.
        current_val: The current classifier value.

    Returns:
        The updated current function.
    """
    new_function = Func(current_val)
    current_func.add_arg(new_function)
    current_func = new_function
    return current_func


def handle_literal(current_func: Func, current_val: Classifier):
    """
    Handle a literal token in the function hierarchy.

    Args:
        current_func: The current function being processed.
        current_val: The current classifier value.
    """
    current_func.add_arg(current_val)


def build_hierarchy(tokens: List[Classifier]):
    """
    Build the function hierarchy from a list of tokens.

    Args:
        tokens: A list of Classifier tokens.

    Returns:
        The main function with the built hierarchy.
    """
    # print_classifier(tokens)
    new_tokens = deepcopy(tokens)
    if new_tokens[0].val_type == 'function':
        main_func = Func(new_tokens.pop(0))
    else:
        main_func = Func(Classifier('pl.lit'))
    current_func = main_func
    pos = 0
    while pos < len(new_tokens):
        current_val = new_tokens[pos]
        previous_val = current_func.func_ref if pos < 1 else new_tokens[pos - 1]
        next_val = new_tokens[pos + 1] if len(new_tokens) > pos + 1 else None
        if isinstance(current_val, Classifier):
            if current_val.val == '(':
                current_func = handle_opening_bracket(current_func, previous_val)
            elif current_val.val == '$if$':
                current_func = handle_if(current_func, current_val)
            elif current_val.val == '$then$':
                current_func, pos = handle_then(current_func, current_val, next_val, pos)
            elif current_val.val == '$else$':
                current_func, pos = handle_else(current_func, next_val, pos)
            elif current_val.val == '$elseif$':
                current_func = handle_elseif(current_func)
            elif current_val.val == '$endif$':
                current_func = handle_endif(current_func)
            elif current_val.val == ')':
                if next_val is None:
                    pass
                    break
                current_func, main_func = handle_closing_bracket(current_func, main_func)
            elif current_val.val_type == 'function':
                current_func = handle_function(current_func, current_val)
            elif current_val.val_type in ('string', 'number', 'boolean', 'operator'):
                handle_literal(current_func, current_val)
            elif current_val.val == '__negative()':
                handle_literal(current_func, Classifier('-1'))
        else:
            handle_literal(current_func, current_val)
        pos += 1
    return main_func
