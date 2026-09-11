"""Hashing expressions: one-way fingerprints of a value.

The cryptographic digests are computed with Python's ``hashlib`` through
``map_elements``. That is slower than a native Polars operation, but it keeps
the results stable across Polars versions and needs no compiled extension, so
it also works in the browser/Pyodide playground.
"""

import hashlib
from typing import Any

import polars as pl

from polars_expr_transformer.funcs.utils import as_expr, PlStringType


def __digest(text: PlStringType, algorithm: str) -> pl.Expr:
    """Hash every value with the named ``hashlib`` algorithm, as hex text."""
    hasher = getattr(hashlib, algorithm)

    def _hex_digest(value: str) -> str:
        return hasher(value.encode("utf-8")).hexdigest()

    return as_expr(text).cast(pl.Utf8).map_elements(_hex_digest, return_dtype=pl.Utf8)


def hash(value: Any) -> pl.Expr:
    """
    Creates a fast numeric fingerprint of a value, useful for grouping or comparing values without storing them.

    For example, hash([first_name]) would return a number like 10309318109784178017 when [first_name] is "John".

    Note that this is a fast, non-cryptographic hash whose numbers may change between Polars versions. Use sha256 when you need a fingerprint that stays the same over time.

    Parameters:
    - value: The column or value to fingerprint

    Returns:
    - A whole number fingerprint of the value
    """
    return as_expr(value).hash()


def md5(text: PlStringType) -> pl.Expr:
    """
    Creates an MD5 hash of text, written as 32 hexadecimal characters.

    For example, md5([first_name]) would return "61409aa1fd47d4a5332de23cbf59a36f" when [first_name] is "John".

    Note that MD5 is not considered secure for passwords or signatures; use sha256 for those.

    Parameters:
    - text: The column or text to hash

    Returns:
    - The MD5 hash as text
    """
    return __digest(text, "md5")


def sha1(text: PlStringType) -> pl.Expr:
    """
    Creates a SHA-1 hash of text, written as 40 hexadecimal characters.

    For example, sha1([first_name]) would return "5753a498f025464d72e088a9d5d6e872592d5f91" when [first_name] is "John".

    Note that SHA-1 is not considered secure for passwords or signatures; use sha256 for those.

    Parameters:
    - text: The column or text to hash

    Returns:
    - The SHA-1 hash as text
    """
    return __digest(text, "sha1")


def sha256(text: PlStringType) -> pl.Expr:
    """
    Creates a SHA-256 hash of text, written as 64 hexadecimal characters.

    For example, sha256([first_name]) would return "a8cfcd74832004951b4408cdb0a5dbcd8c7e52d43f7fe244bf720582e05241da" when [first_name] is "John".

    Parameters:
    - text: The column or text to hash

    Returns:
    - The SHA-256 hash as text
    """
    return __digest(text, "sha256")


def sha512(text: PlStringType) -> pl.Expr:
    """
    Creates a SHA-512 hash of text, written as 128 hexadecimal characters.

    For example, sha512([first_name]) would return a 128 character hash starting with "41b6d0cd5ddab150" when [first_name] is "John".

    Parameters:
    - text: The column or text to hash

    Returns:
    - The SHA-512 hash as text
    """
    return __digest(text, "sha512")
