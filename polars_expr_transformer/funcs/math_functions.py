import polars as pl
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col, PlNumericType

string_type = pl.Expr | str


def negation(number: PlNumericType) -> pl.Expr:
    """
    Changes the sign of a number (positive becomes negative, negative becomes positive).

    For example, negation(5) would return -5, and negation(-3) would return 3.

    Parameters:
    - number: The number to change the sign of

    Returns:
    - The number with its sign flipped
    """
    if is_polars_expr(number):
        return pl.Expr.neg(number)
    else:
        return pl.lit(number).neg()


def log(number: PlNumericType) -> pl.Expr:
    """
    Calculates the natural logarithm of a number.

    For example, log(2.718) would return approximately 1.

    Parameters:
    - number: The number to calculate the logarithm for

    Returns:
    - The natural logarithm of the number
    """
    return pl.Expr.log(number)


def exp(number: PlNumericType) -> pl.Expr:
    """
    Calculates e raised to the power of a number.

    For example, exp(1) would return approximately 2.718.

    Parameters:
    - number: The power to raise e to

    Returns:
    - The result of e^number
    """
    if is_polars_expr(number):
        return pl.Expr.exp(number)
    else:
        return pl.lit(number).exp()


def sqrt(number: PlNumericType) -> pl.Expr:
    """
    Calculates the square root of a number.

    For example, sqrt(9) would return 3.

    Parameters:
    - number: The number to calculate the square root of

    Returns:
    - The square root of the number
    """
    if is_polars_expr(number):
        return pl.Expr.sqrt(number)
    else:
        return pl.lit(number).sqrt()


def abs(number: PlNumericType) -> pl.Expr:
    """
    Returns the absolute value of a number (removes negative sign).

    For example, abs(-5) would return 5, and abs(5) would still return 5.

    Parameters:
    - number: The number to get the absolute value of

    Returns:
    - The positive version of the number
    """
    if is_polars_expr(number):
        return pl.Expr.abs(number)
    else:
        return pl.lit(number).abs()


def sin(angle: PlNumericType) -> pl.Expr:
    """
    Calculates the sine of an angle (in radians).

    For example, sin(0) would return 0.

    Parameters:
    - angle: The angle in radians

    Returns:
    - The sine of the angle
    """
    if is_polars_expr(angle):
        return pl.Expr.sin(angle)
    else:
        return pl.lit(angle).sin()


def cos(angle: PlNumericType) -> pl.Expr:
    """
    Calculates the cosine of an angle (in radians).

    For example, cos(0) would return 1.

    Parameters:
    - angle: The angle in radians

    Returns:
    - The cosine of the angle
    """
    if is_polars_expr(angle):
        return pl.Expr.cos(angle)
    else:
        return pl.lit(angle).cos()


def tan(angle: PlNumericType) -> pl.Expr:
    """
    Calculates the tangent of an angle (in radians).

    For example, tan(0) would return 0.

    Parameters:
    - angle: The angle in radians

    Returns:
    - The tangent of the angle
    """
    if is_polars_expr(angle):
        return pl.Expr.tan(angle)
    else:
        return pl.lit(angle).tan()


def asin(number: PlNumericType) -> pl.Expr:
    """
    Calculates the arcsine (inverse sine) of a number.

    For example, asin(0) would return 0.

    Parameters:
    - number: The number to calculate the arcsine of (between -1 and 1)

    Returns:
    - The angle in radians whose sine equals the input
    """
    if is_polars_expr(number):
        return pl.Expr.arcsin(number)
    else:
        return pl.lit(number).arcsin()


def acos(number: PlNumericType) -> pl.Expr:
    """
    Calculates the arccosine (inverse cosine) of a number.

    For example, acos(1) would return 0.

    Parameters:
    - number: The number to calculate the arccosine of (between -1 and 1)

    Returns:
    - The angle in radians whose cosine equals the input
    """
    if is_polars_expr(number):
        return pl.Expr.arccos(number)
    else:
        return pl.lit(number).arccos()


def atan(number: PlNumericType) -> pl.Expr:
    """
    Calculates the arctangent (inverse tangent) of a number.

    For example, atan(0) would return 0.

    Parameters:
    - number: The number to calculate the arctangent of

    Returns:
    - The angle in radians whose tangent equals the input
    """
    if is_polars_expr(number):
        return pl.Expr.arctan(number)
    else:
        return pl.lit(number).arctan()


def power(base: PlNumericType, exponent: PlNumericType) -> pl.Expr:
    """
    Raises a number to a power.

    For example, power(2, 3) would return 8 (2^3 = 8).

    Parameters:
    - base: The number to raise
    - exponent: The power to raise the base to

    Returns:
    - The result of base raised to the power of exponent
    """
    b = base if is_polars_expr(base) else pl.lit(base)
    e = exponent if is_polars_expr(exponent) else pl.lit(exponent)
    return b.pow(e)


def pow(base: PlNumericType, exponent: PlNumericType) -> pl.Expr:
    """
    Raises a number to a power (alias for power).

    For example, pow(2, 3) would return 8 (2^3 = 8).

    Parameters:
    - base: The number to raise
    - exponent: The power to raise the base to

    Returns:
    - The result of base raised to the power of exponent
    """
    return power(base, exponent)


def mod(dividend: PlNumericType, divisor: PlNumericType) -> pl.Expr:
    """
    Calculates the remainder of division (modulo).

    For example, mod(10, 3) would return 1 (10 divided by 3 is 3 remainder 1).

    Parameters:
    - dividend: The number to be divided
    - divisor: The number to divide by

    Returns:
    - The remainder of the division
    """
    d = dividend if is_polars_expr(dividend) else pl.lit(dividend)
    div = divisor if is_polars_expr(divisor) else pl.lit(divisor)
    return d.mod(div)


def sign(number: PlNumericType) -> pl.Expr:
    """
    Returns the sign of a number (-1, 0, or 1).

    For example, sign(-5) returns -1, sign(0) returns 0, sign(5) returns 1.

    Parameters:
    - number: The number to check

    Returns:
    - -1 if negative, 0 if zero, 1 if positive
    """
    if is_polars_expr(number):
        return pl.Expr.sign(number)
    else:
        return pl.lit(number).sign()


def log10(number: PlNumericType) -> pl.Expr:
    """
    Calculates the base-10 logarithm of a number.

    For example, log10(100) would return 2.

    Parameters:
    - number: The number to calculate the logarithm for

    Returns:
    - The base-10 logarithm of the number
    """
    if is_polars_expr(number):
        return pl.Expr.log(number, base=10)
    else:
        return pl.lit(number).log(base=10)


def log2(number: PlNumericType) -> pl.Expr:
    """
    Calculates the base-2 logarithm of a number.

    For example, log2(8) would return 3.

    Parameters:
    - number: The number to calculate the logarithm for

    Returns:
    - The base-2 logarithm of the number
    """
    if is_polars_expr(number):
        return pl.Expr.log(number, base=2)
    else:
        return pl.lit(number).log(base=2)


def ceil(number: PlNumericType) -> pl.Expr:
    """
    Rounds a number up to the nearest whole number.

    For example, ceil(4.2) would return 5, and ceil(4.9) would also return 5.

    Parameters:
    - number: The number to round up

    Returns:
    - The rounded up number
    """
    if is_polars_expr(number):
        return pl.Expr.ceil(number)
    else:
        return pl.lit(number).ceil()


def round(number: PlNumericType, decimal_places: int = None) -> pl.Expr:
    """
    Rounds a number to a specified number of decimal places.

    For example, round(3.14159, 2) would return 3.14.

    Parameters:
    - number: The number to round
    - decimal_places: How many decimal places to keep (default is 0)

    Returns:
    - The rounded number
    """
    if is_polars_expr(number):
        return pl.Expr.round(number, decimal_places)
    else:
        return pl.lit(number).round(decimal_places)


def floor(number: PlNumericType) -> pl.Expr:
    """
    Rounds a number down to the nearest whole number.

    For example, floor(4.7) would return 4, and floor(4.2) would also return 4.

    Parameters:
    - number: The number to round down

    Returns:
    - The rounded down number
    """
    if is_polars_expr(number):
        return pl.Expr.floor(number)
    else:
        return pl.lit(number).floor()


def tanh(number: PlNumericType) -> pl.Expr:
    """
    Calculates the hyperbolic tangent of a number.

    The result is always between -1 and 1.

    Parameters:
    - number: The number to calculate the hyperbolic tangent of

    Returns:
    - The hyperbolic tangent of the number
    """
    if is_polars_expr(number):
        return pl.Expr.tanh(number)
    else:
        return pl.lit(number).tanh()


def negative() -> int:
    """
    Returns the value -1.

    This is a simple utility function that always returns negative one.

    Returns:
    - The value -1
    """
    return -1


def random_int(min_value: int = 0, max_value: int = 2):
    """
    Generates a random whole number between two values.

    For example, random_int(1, 6) might return 4 (like rolling a die).

    Parameters:
    - min_value: The smallest possible number to generate (default is 0)
    - max_value: The largest possible number to generate (default is 2)

    Returns:
    - A random number between min_value and max_value
    """
    return pl.int_range(min_value, max_value).sample(n=pl.len(), with_replacement=True)