from typing import List, Union
from polars_expr_transformer.configs.settings import operators
from polars_expr_transformer.process.models import IfFunc, Classifier, Func
from polars_expr_transformer.process.tree import build_hierarchy


def reverse_dict(d: dict) -> dict:
    """
    Reverse the keys and values in a dictionary.

    Args:
        d: The dictionary to reverse.

    Returns:
        A new dictionary with keys and values swapped.
    """
    return {v: k for k, v in d.items()}


def inline_to_prefix_formula(inline_formula: List[Union[Classifier, IfFunc, Func]]):
    """
    Convert an inline formula to a prefix formula.

    Args:
        inline_formula: The list of tokens in inline formula notation.

    Returns:
        The list of tokens in prefix notation.
    """
    stack = []
    prefix_formula = []
    for i, token in enumerate(inline_formula):
        if not isinstance(token, Classifier):
            prefix_formula.append(token)
            continue
        if token.val == '(':
            stack.append(token)
        elif token.val == ')':
            while len(stack) > 0 and stack[-1].val != '(':
                prefix_formula.append(stack.pop())
            stack.pop()
        elif token.val in operators:
            while len(stack) > 0 and stack[-1].val != '(' and ((token.precedence if token.precedence is not None else 1) <= (stack[-1].precedence if stack[-1].precedence is not None else 0)):
                prefix_formula.append(stack.pop())
            stack.append(token)
        else:
            prefix_formula.append(token)
    while stack:
        prefix_formula.append(stack.pop())
    return prefix_formula[::-1]


def evaluate_prefix_formula(formula_tokens: List[Union[Classifier, IfFunc, Func]]):
    """
    Evaluate a prefix formula to construct the corresponding expression.

    Args:
        formula_tokens: The list of tokens in prefix notation.

    Returns:
        The evaluated expression.
    """
    stack = []
    for token in reversed(formula_tokens):
        if token != '':
            if isinstance(token, Classifier):
                if token in reverse_dict(operators):
                    if len(stack) >= 2:
                        operand1 = stack.pop()
                        operand2 = stack.pop()
                        result = token, Classifier('('), operand2, Classifier(','),  operand1, Classifier(')')
                    elif len(stack) == 1:
                        operand1 = stack.pop()
                        result = token, Classifier('('), operand1, Classifier(')')
                    else:
                        result = token
                    stack.append(result)
                else:
                    stack.append(token)
            else:
                stack.append(token)
    return stack.pop()


def parse_formula(tokens:  List[Union[Classifier, IfFunc, Func]]):
    """
    Parse a list of formula tokens and replace operator tokens with corresponding functions.

    Args:
        tokens: The list of tokens to parse.

    Returns:
        The list of parsed tokens with operators replaced by functions.
    """
    parsed_formula = []
    for val in tokens:
        if isinstance(val, Classifier):
            f = operators.get(val)
            if f is not None:
                parsed_formula.append(Classifier(f))
            else:
                parsed_formula.append(val)
        else:
            parsed_formula.append(val)
    return parsed_formula


def flatten_inline_formula(nested_classifier:  tuple[Classifier]) -> List[Classifier]:
    """
    Flatten a nested structure of classifier tokens into a single list.

    Args:
        nested_classifier: The nested structure of classifier tokens.

    Returns:
        A flat list of classifier tokens.
    """
    flat_result = []

    def flatten_result(vals: List[Classifier]):
        while len(vals)>0:
            current_val = vals.pop(0)
            if isinstance(current_val, (tuple, list)):
                flatten_result(list(current_val))
            else:
                flat_result.append(current_val)

    flatten_result(list(nested_classifier))
    return flat_result


def resolve_inline_formula(inline_formula_tokens: List[Classifier]):
    """
    Resolve an inline formula by converting it to prefix notation and evaluating it.

    Args:
        inline_formula_tokens: The list of tokens in inline formula notation.

    Returns:
        The resolved list of tokens.
    """
    if any(c.val_type == 'operator' for c in inline_formula_tokens):
        prefixed_formula = inline_to_prefix_formula(inline_formula_tokens)
        parsed_prefixed_formula = parse_formula(prefixed_formula)
        evaluated_prefix_formula = evaluate_prefix_formula(parsed_prefixed_formula)
        return flatten_inline_formula(evaluated_prefix_formula)
    return inline_formula_tokens


def parse_inline_functions(_hierarchical_formula: Func):
    """
    Parse inline functions within a hierarchical formula structure.

    Args:
        _hierarchical_formula: The hierarchical formula to parse.
    """
    run = [True]
    def parse_inline_function_worker(_hierarchical_formula: Func):
        if any([a.val_type == 'operator' for a in _hierarchical_formula.args if isinstance(a, Classifier)]):
            prefixed_formula = inline_to_prefix_formula(_hierarchical_formula.args)
            parsed_prefixed_formula = parse_formula(prefixed_formula)
            evaluated_prefix_formula = evaluate_prefix_formula(parsed_prefixed_formula)
            flattened_prefix_formula = flatten_inline_formula(evaluated_prefix_formula)
            _hierarchical_formula.args = [build_hierarchy(flattened_prefix_formula)]
            run[0] = True
        else:
            for arg in _hierarchical_formula.args:
                if isinstance(arg, Func):
                    parse_inline_function_worker(arg)
                elif isinstance(arg, IfFunc):
                    for condition in arg.conditions:
                        parse_inline_function_worker(condition.condition)
                        parse_inline_function_worker(condition.val)
                    parse_inline_function_worker(arg.else_val)
    while run[0]:
        run[0] = False
        parse_inline_function_worker(_hierarchical_formula)
