from typing import List
from polars_expr_transformer.process.models import Classifier


def replace_ambiguity_minus_sign(tokens: List[Classifier]) -> List[Classifier]:
    """
    Replace ambiguous minus signs in the list of tokens with a negative function and multiplication sign.

    Args:
        tokens: A list of Classifier tokens.

    Returns:
        A list of Classifier tokens with ambiguous minus signs replaced.
    """
    tokens_no_spaces = [t for t in tokens if t.val != '']
    if '-' not in [t.val for t in tokens]:
        return tokens
    i = 0
    new_tokens = []
    while i < len(tokens_no_spaces):
        current_token = tokens_no_spaces[i]
        if current_token.val == '-':
            if i == 0 or not tokens_no_spaces[i - 1].val.isnumeric():
                new_tokens.append(Classifier('__negative()'))
                new_tokens.append(Classifier('*'))
                i += 1
                continue
            elif tokens_no_spaces[i - 1].val.isnumeric():
                new_tokens.append(Classifier('+'))
                new_tokens.append(Classifier('__negative()'))
                new_tokens.append(Classifier('*'))
                i += 1
                continue
        new_tokens.append(tokens_no_spaces[i])
        i += 1
    return new_tokens


def standardize_quotes(tokens: List[str]):
    """
    Standardize single quotes in the list of tokens to double quotes.

    Args:
        tokens: A list of string tokens.

    Returns:
        A list of string tokens with single quotes standardized to double quotes.
    """
    output_tokens = []
    for tok in tokens:
        if len(tok) > 1 and tok[0] == "'" and tok[-1] == "'":
            new_tok = '"' + tok[1:-1] + '"'
            output_tokens.append(new_tok)
        else:
            output_tokens.append(tok)
    return output_tokens


def standardize_tokens(tokens: List[str]) -> List[Classifier]:
    """
    Standardize the list of tokens by converting them to Classifier objects and replacing ambiguous minus signs.

    Args:
        tokens: A list of string tokens.

    Returns:
        A list of Classifier tokens with standardized quotes and ambiguous minus signs replaced.
    """
    standardized_tokens = standardize_quotes(tokens)
    toks = [Classifier(val) for val in standardized_tokens]
    toks = [t for t in toks if t.val_type != 'empty']
    return replace_ambiguity_minus_sign(toks)
