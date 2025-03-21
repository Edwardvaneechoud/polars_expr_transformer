from polars_expr_transformer.process.models import Func, IfFunc


def post_process_hierarchical_formula(hierarchical_formula: Func):
    args_to_do = [hierarchical_formula]
    seen = set()
    while True:
        arg = args_to_do.pop(0)
        if id(arg) in seen:
            continue
        seen.add(id(arg))
        if isinstance(arg, Func):
            arg.standardize_args()
            for a in arg.args:
                if isinstance(a, Func):
                    args_to_do.append(a)
        elif isinstance(arg, IfFunc):
            for condition in arg.conditions:
                args_to_do.append(condition.condition)
                args_to_do.append(condition.val)
            args_to_do.append(arg.else_val)
        if len(args_to_do) == 0:
            break
    return hierarchical_formula
