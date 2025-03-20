from polars_expr_transformer.configs.settings import all_split_vals, all_functions


def tokenize(formula: str):
    """
    Tokenize a formula string into components based on specified split values and functions.

    Args:
        formula: The formula string to tokenize.

    Returns:
        A list of tokens extracted from the formula string with no leading/trailing spaces
        and no empty tokens.
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

        if current_val == string_indicator:
            v += current_val  # Add the closing quote
            output.append(v)  # Add the complete string literal to output
            v = ''
            string_indicator = None
            in_string = False
            i += 1
            continue
        elif current_val in ('"', "'") and string_indicator is None:
            # We're starting a new string literal
            if v:  # If we have accumulated any non-string content
                stripped_v = v.strip()
                if stripped_v:  # Only add non-empty tokens
                    output.append(stripped_v)  # Strip spaces from the token
                v = ''
            in_string = True
            string_indicator = current_val
            v = current_val  # Start the string with the opening quote
            i += 1
            continue

        # If we're inside a string literal, just accumulate characters
        if in_string:
            v += current_val
            i += 1
            continue

        # Handle brackets
        elif current_val in ['[', ']']:
            in_brackets = not in_brackets  # Toggle the in_brackets flag

        # Handle equality operators
        elif current_val == '=' and not in_brackets:
            if len(r) > i + 1:
                two_character_inline = r[i + 1] in ('<', '>', '=', '!')
                if two_character_inline:
                    current_val += r[i + 1]
                    i += 1

        # Check for logical operators, but ONLY outside of strings
        if not in_string and not in_brackets:
            # Check for ' and ' (reversed)
            if i + 4 < len(r) and r[i:i + 5] == list(' dna '):
                if v:
                    stripped_v = v.strip()
                    if stripped_v:  # Only add non-empty tokens
                        output.append(stripped_v)  # Strip spaces
                output.append('dna')
                v = ''
                i += 5
                continue

            # Check for ' or ' (reversed)
            if i + 3 < len(r) and r[i:i + 4] == list(' ro '):
                if v:
                    stripped_v = v.strip()
                    if stripped_v:  # Only add non-empty tokens
                        output.append(stripped_v)  # Strip spaces
                output.append('ro')
                v = ''
                i += 4
                continue

        # Handle normal split values (only outside strings)
        if not in_string and not in_brackets and current_val[::-1] in all_split_vals:
            if v:
                stripped_v = v.strip()
                if stripped_v:  # Only add non-empty tokens
                    output.append(stripped_v)  # Strip spaces from the token
            output.append(current_val)
            v = ''
        elif not in_string and any([vv[::-1] in v + current_val for vv in all_split_vals if len(vv) > 1]):
            splitter = next((vv[::-1] for vv in all_split_vals if len(vv) > 1 and vv[::-1] in v + current_val), None)
            if splitter:
                # check for longer possibilities
                longer_options = [f for f in all_functions.keys() if (v + current_val)[::-1] in f]
                if len(longer_options) > 0:
                    temp_i, temp_v = i, v
                    while temp_i < len(r) and len(
                            [f for f in all_functions.keys() if (temp_v + r[temp_i])[::-1] in f]) > 0:
                        temp_v += r[temp_i]
                        temp_i += 1

                    other_split = next((f for f in all_functions.keys() if temp_v[::-1] == f), None)
                    next_value = r[temp_i] if temp_i < len(r) else None
                    if next_value in [None, ' '] + list(
                            set(v[0] for v in all_split_vals if len(v) > 0)) and other_split is not None:
                        stripped_temp_v = temp_v.strip()
                        if stripped_temp_v:  # Only add non-empty tokens
                            output.append(stripped_temp_v)  # Strip spaces
                        v = ''
                        i = temp_i
                        continue

                for toks in (v + current_val).split(splitter):
                    stripped_toks = toks.strip()
                    if stripped_toks:  # Only add non-empty tokens
                        output.append(stripped_toks)  # Strip spaces
                output.append(splitter)
                v = ''
            else:
                v += current_val
        else:
            v += current_val
        i += 1

    # Process any remaining content
    if v:
        if not in_string and any([vv[::-1] in v for vv in all_split_vals if len(vv) > 1]):
            splitter = next((vv[::-1] for vv in all_split_vals if len(vv) > 1 and vv[::-1] in v), None)
            if splitter:
                for toks in v.split(splitter):
                    if len(toks.strip()) > 0:  # Only add non-empty tokens
                        output.append(toks.strip())  # Strip spaces
                output.append(splitter)
        else:
            stripped_v = v.strip()
            if stripped_v:  # Only add non-empty tokens
                output.append(stripped_v)  # Strip spaces

    # Reverse the characters in each token and reverse the order of tokens
    # Don't strip string literals or tokens that represent column references
    final_output = []
    for v in output:
        token = ''.join(reversed(v))
        # Only strip spaces if the token is not a string literal or column reference
        if not (token.startswith('"') and token.endswith('"')) and \
                not (token.startswith("'") and token.endswith("'")) and \
                not (token.startswith('[') and token.endswith(']')):
            token = token.strip()
        # Filter out empty tokens
        if token:
            final_output.append(token)

    final_output.reverse()

    return final_output