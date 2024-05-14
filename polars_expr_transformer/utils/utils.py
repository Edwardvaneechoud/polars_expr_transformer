import re
from typing import Tuple, Dict, List
from copy import deepcopy



def find_last_occurrence(ls: List):
    for i in range(len(ls) - 1, -1, -1):
        if ls[i] == '(':
            return i

def reverse_dict(d: Dict) -> Dict:
    return {v: k for k, v in d.items()}


def contains_outside_of_quotes(func_string, *args):
    parts = re.split(r"""("[^"]*"|'[^']*')""", ' ' + func_string + ' ')
    vals = '~~~'.join((v.replace('(', '~~~').replace(')', '~~~').replace(' ', '~~~') for v in parts[::2]))
    return all(f'~~~{arg}~~~' in vals for arg in args)



def find_min_pos(fs: str, characters: Tuple):
    if any(character in fs for character in characters):
        return min(fs.find(c) for c in characters if c in fs)
    else:
        return 0


def test_if_position_is_in_quotes(func_string: str, pos: int):
    in_quotes = False
    for p, v in enumerate(func_string):
        if v in ('"', "'"):
            in_quotes = not in_quotes
            if p == pos:
                raise Exception
        if pos == p:
            return in_quotes


def test_if_func_ref(possible_func_ref: str) -> bool:
    return possible_func_ref.startswith('__f__') and possible_func_ref.endswith('__f__')


def contains_special_funcs(input_str: str) -> bool:
    if any(v in input_str for v in set(operators)):
        in_quotes = False
        data = []
        for w in input_str:
            if w in ('"', "'"):
                in_quotes = not in_quotes
            if w in operators and not in_quotes:
                return True
            data.append(w)
    return False


def replace_commas(input_str: str) -> str:
    in_quotes = False
    data = []
    for w in input_str:
        if w in ('"', "'"):
            in_quotes = not in_quotes
        if w == ',' and in_quotes:
            w = '%^%'
        data.append(w)
    return ''.join(data)


def get_last_value_of_generator(gen):
    last = 0
    for last in gen:  # This will end with 'last' being the last value or 0 if the generator is empty
        pass
    return last


def combine_functions_in_tokens(func_tokens: List[str]):
    first_close = next((i for i, t in enumerate(func_tokens) if t == ')'), 0)
    if first_close > 0:
        first_open = get_last_value_of_generator(i for i, t in enumerate(func_tokens[:first_close]) if t == '(')
        if first_open == 0:
            return func_tokens
        func_tokens = func_tokens[:first_open - 1] + [
            ''.join(func_tokens[first_open - 1:first_close + 1])] + func_tokens[first_close + 1:]
        return combine_functions_in_tokens(func_tokens)
    return func_tokens


def select_largest_non_overlapping(groups):
    # Sort by start point, then by reverse end point to get largest first
    sorted_groups = sorted(groups, key=lambda x: (x[0], -x[1]))
    result = []
    end = -float('inf')
    for group in sorted_groups:
        if group[0] > end:  # Check if non-overlapping
            result.append(group)
            end = group[1]  # Update last known end
    return result


def find_deepest_function(func_string: str) -> Tuple[Tuple[int, int], str]:
    def find_func_range(func_string: str):
        func_string_tokens = tokenize(func_string)
        has_first_close = ')' in func_string_tokens
        if has_first_close:
            first_close = func_string_tokens.index(')')
            last_open = find_last_occurrence(func_string_tokens[:first_close])
            len_of_func = len(''.join(func_string_tokens[last_open - 1:first_close + 1]))
            ref_start = len(''.join(func_string_tokens[:last_open - 1]))
            first_close = ref_start + len_of_func
            return ref_start, first_close
        else:
            return -1, -1

    def find_inline_func_range(func_string: str, check_for_equals: bool = True):
        # func_string = '-120=-121*0'
        tt = tokenize(func_string, False)
        if '=' in tt:
            tt = ''.join(['%~=~%' if t == '=' else t for t in tt]).split('%~=~%')
            results = [(find_inline_func_range(t, False), len(t) + i + 1) for i, t in enumerate(tt)]
            other_inlines = any(r[0][0] >= 0 and r[0][1] >= 0 for r in results)
            if other_inlines:
                pos = 0
                for r in results:
                    inline_func = r[0][0] >= 0 and r[0][1] >= 0
                    if inline_func:
                        return r[0][0] + pos, r[0][1] + pos
                    pos += r[1]

        func_string_tokens = combine_functions_in_tokens(tokenize(func_string, False))
        func_string_tokens = [t for t in func_string_tokens if t != '']
        inline_funcs = []
        for i, tok in enumerate(func_string_tokens):
            if tok in operators:
                inline_funcs.append(i)

        grouped_inline_funcs = []
        current_group = []
        for v in inline_funcs:
            if len(current_group) == 0:
                current_group = [v, v]
            elif v - current_group[1] > 2:
                grouped_inline_funcs.append(current_group)
                current_group = []
            else:
                current_group[1] = v
        if current_group != []:
            grouped_inline_funcs.append(current_group)
        unique_groups = select_largest_non_overlapping(grouped_inline_funcs)
        if len(unique_groups) > 0:
            first_group = unique_groups[0]
            start = sum(len(v) for v in func_string_tokens[:max(first_group[0] - 1, 0)])
            end = sum([len(v) for v in func_string_tokens[:first_group[-1] + 2]])
            return start, end

        else:
            return -1, -1

    def find_if_range(func_string: str):
        endif_index = func_string.find('$endif$')
        if endif_index != -1:
            matches = [match for match in re.finditer(r'\$if\$', func_string[:endif_index])]
            if len(matches) > 0:
                match = matches[-1]
            else:
                match = None
            if match:
                return match.start(), endif_index + len('$endif$')
        return -1, -1

    def test_inner(t1: Tuple[str, Tuple[int, int]], t2: Tuple[str, Tuple[int, int]]):
        # Calculate the range differences
        diff1 = t1[1][1] - t1[1][0]
        diff2 = t2[1][1] - t2[1][0]

        # If the ranges are non-overlapping and the second range starts after the first one ends,
        # return the first range as it starts first.
        if t1[1][1] < t2[1][0]:
            return t1
        if t2[1][1] < t1[1][0]:
            return t2
        if t1[1][1] < t2[1][1] and t1[1][0] > t2[1][0]:
            return t1
        if t2[1][1] < t1[1][1] and t2[1][0] > t1[1][0]:
            return t2
        if t1[0] == 'inline' and t2[0] in ('if', 'f'):
            return t2
        if t2[0] == 'inline' and t1[0] in ('if', 'f'):
            return t1
        return None

    def test_inners(*args):
        args = [a for a in args if sum(a[1]) > 0]
        if len(args) == 0:
            return None
        elif len(args) == 1:
            return args[0]
        while len(args) > 1:
            arg_1 = args.pop(0)
            arg_2 = args.pop(0)
            args.append(test_inner(arg_1, arg_2))
        return args[0]

    if_range = find_if_range(func_string)
    # print(f'deepest if range: {if_range}')
    func_range = find_func_range(func_string)
    # print(f'deepest func range: {func_range}')
    inline_range = find_inline_func_range(func_string)
    # print(f'deepest inline range: {inline_range}')
    inner_func = test_inners(('if', if_range), ('f', func_range), ('inline', inline_range))
    if inner_func is None:
        return None
    else:
        return inner_func[1], inner_func[0]


def replace_double_minus(toks: List[str]) -> List[str]:
    new_toks = []
    i = 0
    while i < len(toks):
        if i < len(toks) - 1 and toks[i] == '-' and toks[i + 1] == '-':
            new_toks.append('+')
            i += 2  # Skip the next index
        else:
            new_toks.append(toks[i])
            i += 1
    return new_toks


def replace_ambiguity_minus_sign(tokens: List[str]) -> List[str]:
    tokens_no_spaces = [t for t in tokens if t != '']
    # tokens_no_spaces = replace_double_minus(toks=tokens_no_spaces)
    if '-' not in tokens:
        return tokens
    i = 0
    new_tokens = []
    while i < len(tokens_no_spaces):
        current_token = tokens_no_spaces[i]
        if current_token == '-':
            if i == 0 or not tokens_no_spaces[i - 1].isnumeric():
                # new_tokens.append(f'pl.Expr.__neg__({tokens_no_spaces[i+1]})')
                new_tokens.append('__negative()')
                new_tokens.append('*')
                i += 1
                continue
            elif tokens_no_spaces[i - 1].isnumeric():
                new_tokens.append('+')
                new_tokens.append('__negative()')
                new_tokens.append('*')
                i += 1
                continue
        new_tokens.append(tokens_no_spaces[i])
        i += 1
    return new_tokens


def tokenize_after_standardization(formula):
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
            string_indicator = None
            in_string = False
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
        elif current_val == '/' and not in_brackets:
            if len(r) > i + 2:
                three_characters_inline = r[i + 1:i + 3] == '/s/'
                if three_characters_inline:
                    current_val += r[i + 1:i + 3]
                    i += 2
                    in_string = not in_string

        if not in_string and not in_brackets and current_val[::-1] in all_split_vals:
            if i > 0:
                output.append(v)
            output.append(current_val)
            v = ''
        elif any([vv[::-1] in v for vv in all_split_vals if len(vv) > 1]):
            splitter = next(vv[::-1] for vv in all_split_vals if len(vv) > 1 and vv[::-1] in v)

            for toks in v.split(splitter):
                if len(toks) > 0:
                    output.append(toks)
            output.append(splitter)
            v = current_val
        else:
            v += current_val
        i += 1

    if any([vv[::-1] in v for vv in all_split_vals if len(vv) > 1]):
        splitter = next(vv[::-1] for vv in all_split_vals if len(vv) > 1 and vv[::-1] in v)

        for toks in v.split(splitter):
            if len(toks) > 0:
                output.append(toks)
        output.append(splitter)
    else:
        output.append(v)
    output = [''.join(reversed(v)) for v in output]
    output.reverse()
    return output


def transform_inline_formula_to_pl_formula(formula: str, convert_to_pl_cols: bool = False) -> str:
    precedence = {'+': 2, '-': 2, '*': 3, '/': 3, '|': -1, '&': -1}

    def get_val(v):
        if v in operators:
            return operators[v]
        return v

    def inline_to_prefix_formula(inline_formula):
        stack = []
        prefix_formula = []  # Define operator precedence
        for token in inline_formula:
            if token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    prefix_formula.append(stack.pop())
                stack.pop()  # Remove '('
            elif token in operators:
                while stack and stack[-1] != '(' and precedence.get(token, 1) <= precedence.get(stack[-1], 0):
                    prefix_formula.append(stack.pop())
                stack.append(token)
            else:
                prefix_formula.append(token)

        while stack:
            prefix_formula.append(stack.pop())

        return prefix_formula[::-1]  # Reverse the formula to obtain prefix notation

    def evaluate_prefix_formula(formula):
        stack = []
        for token in reversed(formula):
            if token != '':
                if token in reverse_dict(operators):
                    if len(stack) >= 2:
                        operand1 = stack.pop()
                        operand2 = stack.pop()
                        result = token + '(' + operand2 + ',' + operand1 + ')'
                    elif len(stack) == 1:
                        operand1 = stack.pop()
                        result = token + '(' + operand1 + ')'
                    else:
                        result = token
                    stack.append(result)
                else:
                    stack.append(token)
        return stack.pop()

    tokens = tokenize(formula, convert_to_pl_cols)
    tokens = replace_ambiguity_minus_sign(tokens)
    prefix_formula = inline_to_prefix_formula(tokens)
    parsed_formula = [get_val(v) for v in prefix_formula]
    result = evaluate_prefix_formula(parsed_formula)
    return result
