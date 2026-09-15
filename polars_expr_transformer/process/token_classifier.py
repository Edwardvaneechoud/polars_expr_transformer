from typing import List
from polars_expr_transformer.process.models import Classifier
from polars_expr_transformer.string_literals import requote_double


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
    Standardize quoted string tokens to double-quoted Python literals.

    Quotes inside the body are escaped, so requoting cannot break a literal open
    and turn the rest of a formula into code.

    Args:
        tokens: A list of string tokens.

    Returns:
        A list of string tokens with quotes standardized.
    """
    return [requote_double(tok) for tok in tokens]


def merge_multiword_operators(tokens: List[Classifier]) -> List[Classifier]:
    """
    Merge the two tokens of `not in` into the single operator token it stands for.

    Preprocessing strips every space, so `not in` reaches the tokenizer as `notin` and
    only comes apart because `in` is a split value. Rewriting the pair here keeps the
    tokenizer free of a split value containing `in`, which it would mis-split. A `not`
    used as a function is always followed by `(`, so it is never merged.

    Args:
        tokens: A list of Classifier tokens.

    Returns:
        A list of Classifier tokens with `not in` merged into one operator token.
    """
    merged = []
    i = 0
    while i < len(tokens):
        if (tokens[i].val == 'not' and i + 1 < len(tokens)
                and tokens[i + 1].val == 'in'):
            merged.append(Classifier('not in'))
            i += 2
            continue
        merged.append(tokens[i])
        i += 1
    return merged


def classify_tokens(tokens: List[str]) -> List[Classifier]:
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
    return merge_multiword_operators(toks)
