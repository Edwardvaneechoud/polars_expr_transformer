from polars_expr_transformer.configs.settings import all_split_vals, all_functions


def tokenize(formula: str):
    """
    Tokenize a formula string into components based on specified split values and functions.

    Args:
        formula: The formula string to tokenize.

    Returns:
        A list of tokens extracted from the formula string.
    """
    r = list(formula[::-1])
    output = []
    v = ''
    in_string = False  # Flag to track if we're inside a string literal
    in_brackets = False  # Flag to track if we're inside square brackets
    i = 0
    string_indicator = None
    while i < len(r):
        current_val = r[i]
        # print(i, v[::-1])
        if current_val == string_indicator:
            output.append(v+current_val)
            v = ''
            string_indicator = None
            in_string = False
            i += 1
            continue
        elif current_val in ('"', "'") and string_indicator is None:
            in_string = True  # Toggle the in_string flag
            string_indicator = current_val
        elif current_val in ['[', ']']:
            in_brackets = not in_brackets  # Toggle the in_brackets flag
        elif current_val == '=' and not in_brackets and not in_string:
            if len(r) > i + 1:
                two_character_inline = r[i + 1] in ('<', '>', '=', '!')
                if two_character_inline:
                    current_val += r[i + 1]
                    i += 1
        if not in_string and not in_brackets and current_val[::-1] in all_split_vals:
            if i > 0:
                output.append(v)
            output.append(current_val)
            v = ''
        elif any([vv[::-1] in v+current_val for vv in all_split_vals if len(vv) > 1]) and not in_string:
            splitter = next(vv[::-1] for vv in all_split_vals if len(vv) > 1 and vv[::-1] in v+current_val)
            if splitter:
                # check for longer possiblities
                longer_options = [f for f in all_functions.keys() if (v+current_val)[::-1] in f]
                if len(longer_options)>0:
                    temp_i, temp_v = i, v
                    while temp_i<len(r) and len([f for f in all_functions.keys() if (temp_v+r[temp_i])[::-1] in f])>0:
                        temp_v += r[temp_i]
                        temp_i += 1

                    other_split = next((f for f in all_functions.keys() if temp_v[::-1] == f))
                    next_value = r[temp_i] if temp_i<len(r) else None
                    if next_value in [None, ' '] + list(set(v[0] for v in all_split_vals if len(v)>0)) and other_split is not None:
                        output.append(temp_v)
                        v = ''
                        i = temp_i
                        continue
            for toks in (v+current_val).split(splitter):
                if len(toks) > 0:
                    output.append(toks)
            output.append(splitter)
            v = ''
        else:
            v += current_val
        i += 1

    if v is not None and any([vv[::-1] in v for vv in all_split_vals if len(vv) > 1]):
        splitter = next(vv[::-1] for vv in all_split_vals if len(vv) > 1 and vv[::-1] in v)

        for toks in v.split(splitter):
            if len(toks) > 0:
                output.append(toks)
        output.append(splitter)
    elif v is not None:
        output.append(v)
    output = [''.join(reversed(v)) for v in output]
    output.reverse()
    return output
