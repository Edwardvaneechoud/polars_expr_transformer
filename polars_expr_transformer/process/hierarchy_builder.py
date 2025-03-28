from typing import Optional, List, Tuple
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

    new_func = Func(Classifier('pl.lit'))
    current_func.add_arg(new_func)
    current_func = new_func
    return current_func


def handle_if(current_func: Func, current_val: Classifier, next_val: Classifier, pos: int) -> Tuple[Func, int]:
    """
    Handle the if condition in the function hierarchy.

    Args:
        current_func: The current function being processed.
        current_val: The current classifier value.
        next_val: The next classifier value.
        pos: The current position in the tokens list.

    Returns:
        The updated current function as IfFunc or Func.
    """
    if_func = IfFunc(current_val)
    current_func.add_arg(if_func)

    if_func.add_else_val(Func(Classifier('pl.lit')))

    if next_val and next_val.val == '(':
        pos += 1
    else:
        raise Exception('Expected opening bracket')
    condition = Func(Classifier('pl.lit'))
    val = Func(Classifier('pl.lit'))
    condition_val = ConditionVal(condition=condition, val=val)
    if_func.add_condition(condition_val)
    return condition_val.condition, pos


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
        else:
            raise Exception('Expected opening bracket')
    # elif isinstance(current_func.parent, ConditionVal):
    #     current_func.parent.func_ref = current_val
    #     current_func = current_func.parent.val
    #     if next_val and next_val.val == '(':
    #         pos += 1
    else:
        raise Exception('Expected to be in a condition val')
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


def handle_elseif(current_func: Func, current_val: Classifier, next_val: Optional[Classifier], pos: int) -> Tuple[Func, int]:
    """
    Handle the elseif condition in the function hierarchy.

    Args:
        current_func: The current function being processed.
        current_val: The current classifier value.
        next_val: The next classifier value.
        pos: The current position in the tokens list.

    Returns:
        The updated current function as IfFunc or Func.
    """
    if not isinstance(current_func.parent, IfFunc):
        raise Exception('Expected if')
    if_func = current_func.parent
    condition = Func(Classifier('pl.lit'))
    val = Func(Classifier('pl.lit'))
    condition_val = ConditionVal(condition=condition, val=val)
    if_func.add_condition(condition_val)
    condition_val.func_ref = current_val
    current_func = condition
    if next_val and next_val.val == '(':
        pos += 1

    return current_func, pos


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
    elif isinstance(current_func, TempFunc):
        parent_func = current_func.parent
        if isinstance(parent_func, Func) and isinstance(parent_func.parent, (TempFunc, Func)):
            current_func = parent_func.parent
    elif current_func.parent is not None:
        current_func = current_func.parent
    else:
        raise Exception('Expected parent')
    return current_func, main_func


def handle_function(current_func: Func, current_val: Classifier, next_val: Classifier, pos: int) -> Tuple[Func, int]:
    """
    Handle a function token in the function hierarchy.

    Args:
        current_func: The current function being processed.
        current_val: The current classifier value.
        next_val: The next classifier value.
        pos: The current position in the tokens list.
    Returns:
        The updated current function.
    """
    new_function = Func(current_val)
    if next_val and next_val.val == '(':
        pos += 1
    elif current_val.val == 'negation':
        pass
    else:
        raise Exception('Expected opening bracket')
    current_func.add_arg(new_function)
    first_arg = TempFunc()
    new_function.add_arg(first_arg)
    return first_arg, pos


def handle_literal(current_func: Func, current_val: Classifier):
    """
    Handle a literal token in the function hierarchy.

    Args:
        current_func: The current function being processed.
        current_val: The current classifier value.
    """
    current_func.add_arg(current_val)


def handle_seperator(current_func: Func) -> Func:
    # find opening of current function
    parent_func = current_func.parent
    if not isinstance(parent_func, Func):
        raise Exception('Expected parent to be a function')

    new_arg = TempFunc()
    parent_func.add_arg(new_arg)
    return new_arg


def handle_operator(current_func: Func, current_val: Classifier):
    current_func.parent.add_arg(current_val)
    return current_func.parent

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
        main_func = Func(Classifier('pl.lit'))
    else:
        main_func = Func(Classifier('pl.lit'))
    current_func = main_func
    pos = 0

    while pos < len(new_tokens):
        current_val = new_tokens[pos]
        previous_val = current_func.func_ref if pos < 1 and not isinstance(current_func, TempFunc) else new_tokens[pos - 1]
        next_val = new_tokens[pos + 1] if len(new_tokens) > pos + 1 else None
        if isinstance(current_val, Classifier):
            if current_val.val == '(':
                current_func = handle_opening_bracket(current_func, previous_val)
            elif current_val.val == '$if$':
                current_func, pos = handle_if(current_func, current_val, next_val, pos)
            elif current_val.val == '$then$':
                current_func, pos = handle_then(current_func, current_val, next_val, pos)
            elif current_val.val == '$else$':
                current_func, pos = handle_else(current_func, next_val, pos)
            elif current_val.val == '$elseif$':
                current_func, pos = handle_elseif(current_func, current_val, next_val, pos)
            elif current_val.val == '$endif$':
                current_func = handle_endif(current_func)
            elif current_val.val_type == 'sep':
                current_func = handle_seperator(current_func)
            elif current_val.val == ')':
                if next_val is None:
                    pass
                    # break
                current_func, main_func = handle_closing_bracket(current_func, main_func)
            elif current_val.val_type == 'function':
                current_func, pos = handle_function(current_func, current_val, next_val, pos)
            elif current_val.val_type in ('string', 'number', 'boolean', 'operator'):
                if (current_val.val_type == 'operator' and
                        current_val.val == '-' and
                        (len(current_func.args) == 0 or previous_val.val_type == 'operator')):
                    current_func, pos = handle_function(current_func, Classifier('negation'), next_val, pos)
                else:
                    handle_literal(current_func, current_val)
            elif current_val.val == '__negative()':
                handle_literal(current_func, Classifier('-1'))
        else:
            handle_literal(current_func, current_val)
        pos += 1

    return main_func
