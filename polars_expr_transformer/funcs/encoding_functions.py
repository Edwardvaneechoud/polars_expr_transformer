"""Encoding expressions: reversible text encodings such as base64 and hex.

Unlike the hashing functions, these are two-way — whatever ``encode`` produces,
``decode`` turns back into the original text.
"""

import polars as pl

from polars_expr_transformer.funcs.utils import as_expr, PlStringType

_ENCODINGS = ("base64", "hex")


def __check_encoding(encoding: str) -> str:
    """Validate an encoding name, raising a message that lists the options."""
    if encoding not in _ENCODINGS:
        raise ValueError(
            f"Unknown encoding: {encoding}\n\n"
            f"Possible options are: {', '.join(_ENCODINGS)}"
        )
    return encoding


def encode(text: PlStringType, encoding: str = "base64") -> pl.Expr:
    """
    Encodes text as base64 or hexadecimal.

    For example, encode([first_name], "base64") would return "Sm9obg==" when [first_name] is "John".

    Parameters:
    - text: The column or text to encode
    - encoding: Which encoding to use, either 'base64' (default) or 'hex'

    Returns:
    - The encoded text
    """
    return as_expr(text).cast(pl.Utf8).str.encode(__check_encoding(encoding))


def decode(text: PlStringType, encoding: str = "base64") -> pl.Expr:
    """
    Decodes base64 or hexadecimal text back into readable text. Text that is not valid for the encoding becomes empty, and the decoded result has to be text itself.

    For example, decode("Sm9obg==", "base64") would return "John".

    Parameters:
    - text: The column or text to decode
    - encoding: Which encoding the text uses, either 'base64' (default) or 'hex'

    Returns:
    - The decoded text
    """
    decoded = as_expr(text).str.decode(__check_encoding(encoding), strict=False)
    return decoded.cast(pl.Utf8)


def base64_encode(text: PlStringType) -> pl.Expr:
    """
    Encodes text as base64.

    For example, base64_encode([first_name]) would return "Sm9obg==" when [first_name] is "John".

    Parameters:
    - text: The column or text to encode

    Returns:
    - The base64 text
    """
    return encode(text, "base64")


def base64_decode(text: PlStringType) -> pl.Expr:
    """
    Decodes base64 text back into readable text. Text that is not valid base64 becomes empty, and the decoded result has to be text itself.

    For example, base64_decode("Sm9obg==") would return "John".

    Parameters:
    - text: The column or text to decode

    Returns:
    - The decoded text
    """
    return decode(text, "base64")


def hex_encode(text: PlStringType) -> pl.Expr:
    """
    Encodes text as hexadecimal.

    For example, hex_encode([first_name]) would return "4a6f686e" when [first_name] is "John".

    Parameters:
    - text: The column or text to encode

    Returns:
    - The hexadecimal text
    """
    return encode(text, "hex")


def hex_decode(text: PlStringType) -> pl.Expr:
    """
    Decodes hexadecimal text back into readable text. Text that is not valid hexadecimal becomes empty, and the decoded result has to be text itself.

    For example, hex_decode("4a6f686e") would return "John".

    Parameters:
    - text: The column or text to decode

    Returns:
    - The decoded text
    """
    return decode(text, "hex")
