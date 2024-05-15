from typing import Optional, List
from polars_expr_transformer.process.models import Classifier, Func, IfFunc, TempFunc, ConditionVal
from copy import deepcopy


def handle_opening_bracket(current_func: Func, previous_val: Classifier) -> Func:
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

def handle_if(current_func: Func, current_val: Classifier) -> Func:
    new_func = IfFunc(current_val)
    current_func.add_arg(new_func)
    current_func = new_func
    return current_func

def handle_then(current_func: Func, current_val: Classifier, next_val: Optional[Classifier], pos: int) -> (Func, int):
    if isinstance(current_func, ConditionVal):
        current_func.func_ref = current_val
        current_func = current_func.val
        if next_val and next_val.val == '(':
            pos += 1
    return current_func, pos

def handle_else(current_func: Func, next_val: Optional[Classifier], pos: int) -> (Func, int):
    current_func = current_func.parent
    if isinstance(current_func, IfFunc):
        current_func = current_func.else_val
        if next_val and next_val.val == '(':
            pos += 1
    else:
        raise Exception('Expected if')
    return current_func, pos

def handle_elseif(current_func: Func) -> Func:
    current_func = current_func.parent
    if not isinstance(current_func, IfFunc):
        raise Exception('Expected if')
    return current_func

def handle_endif(current_func: Func) -> Func:
    if isinstance(current_func, IfFunc):
        current_func = current_func.parent
    else:
        raise Exception('Expected if')
    return current_func

def handle_closing_bracket(current_func: Func, main_func: Func) -> (Func, Func):
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
    new_function = Func(current_val)
    current_func.add_arg(new_function)
    current_func = new_function
    return current_func


def handle_literal(current_func: Func, current_val: Classifier):
    current_func.add_arg(current_val)


def build_hierarchy(tokens: List[Classifier]):
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
        previous_val = current_func.func_ref if pos < 1 else new_tokens[pos-1]
        next_val = new_tokens[pos+1] if len(new_tokens) > pos+1 else None
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