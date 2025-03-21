from polars_expr_transformer.process.polars_expr_transformer import build_func


def visualize_function_hierarchy(func_obj, original_expr=None):
    """
    Generate a text-based hierarchy visualization of a function object,
    showing the accurate structure including pl.lit wrappers.

    Args:
        func_obj: The function object (Func, IfFunc, Classifier instance)
        original_expr: Optional string of the original expression

    Returns:
        String containing the formatted hierarchy visualization
    """
    result = []

    # Add header
    if original_expr:
        result.append("# Function Hierarchy: Expression Structure\n")
        result.append(f"This represents the expression: `{original_expr}`\n")

    # Generate the hierarchy tree
    result.append("```")
    tree_lines = _generate_tree(func_obj)
    result.extend(tree_lines)
    result.append("```\n")

    # Add the readable polars function
    if hasattr(func_obj, 'get_readable_pl_function'):
        result.append(f"## Polars Expression\n")
        result.append(f"```python\n{func_obj.get_readable_pl_function()}\n```\n")

    return "\n".join(result)


def _is_pl_lit_wrapper(obj):
    """
    Check if this is a pl.lit wrapper that should be skipped when it contains another expression.
    But only when configured to do so (we're now preserving these structures).

    Args:
        obj: The function object

    Returns:
        bool: True if this is a pl.lit wrapper around another Func
    """
    if not hasattr(obj, '__class__') or obj.__class__.__name__ != 'Func':
        return False

    # Check if it's a pl.lit function
    if not hasattr(obj.func_ref, 'val') or obj.func_ref.val != 'pl.lit':
        return False

    # Check if it has exactly one argument
    if len(obj.args) != 1:
        return False

    # Only unwrap if the argument is a Func (not a Classifier or primitive)
    # return hasattr(obj.args[0], '__class__') and obj.args[0].__class__.__name__ == 'Func'
    return False


def _generate_tree(obj, prefix="", is_last=True, level=0, unwrap_pl_lit=False):
    """
    Recursively generate tree representation of a function object.

    Args:
        obj: The object to visualize
        prefix: Current line prefix for formatting
        is_last: Whether this is the last item in its branch
        level: Current nesting level
        unwrap_pl_lit: Whether to unwrap pl.lit wrappers (default: False)

    Returns:
        List of formatted lines for the tree
    """
    lines = []

    # Handle TempFunc wrapper if present
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'TempFunc':
        if hasattr(obj, 'args') and obj.args:
            return _generate_tree(obj.args[0], prefix, is_last, level, unwrap_pl_lit)
        else:
            lines.append(f"{prefix}└── TempFunc (empty)")
            return lines

    # Root node doesn't need a branch line
    if level > 0:
        branch = "└── " if is_last else "├── "
    else:
        branch = ""

    # Format node based on type
    if obj is None:
        return lines

    # Normal class handling
    if hasattr(obj, '__class__'):
        class_name = obj.__class__.__name__
    else:
        class_name = "Unknown"

    if class_name == "Func":
        # Get function name from classifier
        func_name = obj.func_ref.val if hasattr(obj.func_ref, 'val') else str(obj.func_ref)
        lines.append(f"{prefix}{branch}Func: {func_name}")

        # Next level prefix
        new_prefix = prefix + ("    " if is_last else "│   ")

        # Add arguments
        for i, arg in enumerate(obj.args):
            is_last_arg = (i == len(obj.args) - 1)

            arg_lines = [f"{new_prefix}{'└── ' if is_last_arg else '├── '}Arg {i + 1}"]
            next_prefix = new_prefix + ("    " if is_last_arg else "│   ")

            # Recursively process argument
            sub_lines = _generate_tree(arg, next_prefix, True, level + 2, unwrap_pl_lit)
            arg_lines.extend(sub_lines)
            lines.extend(arg_lines)

    elif class_name == "IfFunc":
        # Format if function node
        lines.append(f"{prefix}{branch}IfFunc")

        # Next level prefix
        new_prefix = prefix + ("    " if is_last else "│   ")

        # Add conditions
        for i, condition_val in enumerate(obj.conditions):
            is_last_cond = (i == len(obj.conditions) - 1) and not obj.else_val

            cond_lines = [f"{new_prefix}{'└── ' if is_last_cond else '├── '}Condition {i + 1}"]
            cond_prefix = new_prefix + ("    " if is_last_cond else "│   ")

            # Add condition expression
            if hasattr(condition_val, 'condition') and condition_val.condition:
                cond_lines.append(f"{cond_prefix}├── Condition Expression")
                cond_expr_prefix = cond_prefix + "│   "
                sub_lines = _generate_tree(condition_val.condition, cond_expr_prefix, True, level + 3, unwrap_pl_lit)
                cond_lines.extend(sub_lines)

            # Add then value
            if hasattr(condition_val, 'val') and condition_val.val:
                cond_lines.append(f"{cond_prefix}└── Then")
                then_prefix = cond_prefix + "    "
                sub_lines = _generate_tree(condition_val.val, then_prefix, True, level + 3, unwrap_pl_lit)
                cond_lines.extend(sub_lines)

            lines.extend(cond_lines)

        # Add else value
        if obj.else_val:
            else_lines = [f"{new_prefix}└── Else"]
            else_prefix = new_prefix + "    "
            sub_lines = _generate_tree(obj.else_val, else_prefix, True, level + 2, unwrap_pl_lit)
            else_lines.extend(sub_lines)
            lines.extend(else_lines)

    elif class_name == "ConditionVal":
        # Format condition value node
        lines.append(f"{prefix}{branch}ConditionVal")

        # Next level prefix
        new_prefix = prefix + ("    " if is_last else "│   ")

        # Add condition
        if hasattr(obj, 'condition') and obj.condition:
            is_last_item = not hasattr(obj, 'val') or obj.val is None
            cond_lines = [f"{new_prefix}{'└── ' if is_last_item else '├── '}Condition"]
            next_prefix = new_prefix + ("    " if is_last_item else "│   ")
            sub_lines = _generate_tree(obj.condition, next_prefix, True, level + 2, unwrap_pl_lit)
            cond_lines.extend(sub_lines)
            lines.extend(cond_lines)

        # Add value
        if hasattr(obj, 'val') and obj.val:
            val_lines = [f"{new_prefix}└── Value"]
            next_prefix = new_prefix + "    "
            sub_lines = _generate_tree(obj.val, next_prefix, True, level + 2, unwrap_pl_lit)
            val_lines.extend(sub_lines)
            lines.extend(val_lines)

    elif class_name == "Classifier":
        # Format classifier node
        # For literals, show their value directly
        val = obj.val if hasattr(obj, 'val') else str(obj)
        val_type = obj.val_type if hasattr(obj, 'val_type') else ""
        type_str = f" ({val_type})" if val_type else ""

        # For literals like numbers, strings, etc. - use a better label
        if val_type in ["number", "string", "boolean"]:
            lines.append(f"{prefix}{branch}Literal: {val}{type_str}")
        else:
            lines.append(f"{prefix}{branch}Classifier: {val}{type_str}")

    else:
        # Handle unknown or primitive value
        display_val = obj.val if hasattr(obj, 'val') else str(obj)
        lines.append(f"{prefix}{branch}Value: {display_val}")

    return lines


def generate_visualization(expr):
    """
    Generate a visualization of the function hierarchy for the given expression.

    Args:
        expr: The expression to visualize

    Returns:
        The visualization string
    """
    # Build the function object from the expression
    func_obj = build_func(expr)

    # Generate visualization
    visualization = visualize_function_hierarchy(func_obj, expr)

    # Print the generated visualization
    print(visualization)

    return visualization
