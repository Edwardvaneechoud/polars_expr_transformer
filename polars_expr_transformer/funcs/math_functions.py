import polars as pl
from polars_expr_transformer.funcs.utils import is_polars_expr, create_fix_col, PlNumericType

string_type = pl.Expr | str


def negation(number: PlNumericType) -> pl.Expr:
    """
    Flips the sign of each value in a column (positive becomes negative, negative becomes positive).

    For example, negation([price]) would return -19.99 when [price] is 19.99.

    Parameters:
    - number: The column or value to flip the sign of

    Returns:
    - The value with its sign flipped
    """
    if is_polars_expr(number):
        return pl.Expr.neg(number)
    else:
        return pl.lit(number).neg()


def log(number: PlNumericType) -> pl.Expr:
    """
    Calculates the natural logarithm (base e) of each value in a column.

    For example, log([price]) would return 0.0 when [price] is 1.0.

    Parameters:
    - number: The column or value to take the natural logarithm of

    Returns:
    - The natural logarithm of the value
    """
    return pl.Expr.log(number)


def exp(number: PlNumericType) -> pl.Expr:
    """
    Calculates e raised to the power of each value in a column.

    For example, exp([discount]) would return 1.0 when [discount] is 0.0.

    Parameters:
    - number: The column or value to use as the exponent of e

    Returns:
    - The result of e raised to the value
    """
    if is_polars_expr(number):
        return pl.Expr.exp(number)
    else:
        return pl.lit(number).exp()


def sqrt(number: PlNumericType) -> pl.Expr:
    """
    Calculates the square root of each value in a column.

    For example, sqrt([price]) would return 3.0 when [price] is 9.0.

    Parameters:
    - number: The column or value to take the square root of

    Returns:
    - The square root of the value
    """
    if is_polars_expr(number):
        return pl.Expr.sqrt(number)
    else:
        return pl.lit(number).sqrt()


def abs(number: PlNumericType) -> pl.Expr:
    """
    Returns the absolute value of each value in a column (removes the negative sign).

    For example, abs([price]) would return 5.0 when [price] is -5.0.

    Parameters:
    - number: The column or value to get the absolute value of

    Returns:
    - The non-negative version of the value
    """
    if is_polars_expr(number):
        return pl.Expr.abs(number)
    else:
        return pl.lit(number).abs()


def sin(angle: PlNumericType) -> pl.Expr:
    """
    Calculates the sine of each value in a column, where values are angles in radians.

    For example, sin([angle]) would return 0.0 when [angle] is 0.

    Parameters:
    - angle: The column or value with angles in radians

    Returns:
    - The sine of the angle
    """
    if is_polars_expr(angle):
        return pl.Expr.sin(angle)
    else:
        return pl.lit(angle).sin()


def cos(angle: PlNumericType) -> pl.Expr:
    """
    Calculates the cosine of each value in a column, where values are angles in radians.

    For example, cos([angle]) would return 1.0 when [angle] is 0.

    Parameters:
    - angle: The column or value with angles in radians

    Returns:
    - The cosine of the angle
    """
    if is_polars_expr(angle):
        return pl.Expr.cos(angle)
    else:
        return pl.lit(angle).cos()


def tan(angle: PlNumericType) -> pl.Expr:
    """
    Calculates the tangent of each value in a column, where values are angles in radians.

    For example, tan([angle]) would return 0.0 when [angle] is 0.

    Parameters:
    - angle: The column or value with angles in radians

    Returns:
    - The tangent of the angle
    """
    if is_polars_expr(angle):
        return pl.Expr.tan(angle)
    else:
        return pl.lit(angle).tan()


def asin(number: PlNumericType) -> pl.Expr:
    """
    Calculates the arcsine (inverse sine) of each value in a column.

    For example, asin([ratio]) would return 0.0 when [ratio] is 0.

    Parameters:
    - number: The column or value to take the arcsine of (values between -1 and 1)

    Returns:
    - The angle in radians whose sine equals the value
    """
    if is_polars_expr(number):
        return pl.Expr.arcsin(number)
    else:
        return pl.lit(number).arcsin()


def acos(number: PlNumericType) -> pl.Expr:
    """
    Calculates the arccosine (inverse cosine) of each value in a column.

    For example, acos([ratio]) would return 0.0 when [ratio] is 1.

    Parameters:
    - number: The column or value to take the arccosine of (values between -1 and 1)

    Returns:
    - The angle in radians whose cosine equals the value
    """
    if is_polars_expr(number):
        return pl.Expr.arccos(number)
    else:
        return pl.lit(number).arccos()


def atan(number: PlNumericType) -> pl.Expr:
    """
    Calculates the arctangent (inverse tangent) of each value in a column.

    For example, atan([ratio]) would return 0.0 when [ratio] is 0.

    Parameters:
    - number: The column or value to take the arctangent of

    Returns:
    - The angle in radians whose tangent equals the value
    """
    if is_polars_expr(number):
        return pl.Expr.arctan(number)
    else:
        return pl.lit(number).arctan()


def power(base: PlNumericType, exponent: PlNumericType) -> pl.Expr:
    """
    Raises each value in a column to a power.

    For example, power([quantity], 2) would return 9 when [quantity] is 3.

    Parameters:
    - base: The column or value to raise
    - exponent: The power to raise the base to

    Returns:
    - The result of base raised to the power of exponent
    """
    b = base if is_polars_expr(base) else pl.lit(base)
    e = exponent if is_polars_expr(exponent) else pl.lit(exponent)
    return b.pow(e)


def pow(base: PlNumericType, exponent: PlNumericType) -> pl.Expr:
    """
    Raises each value in a column to a power (alias for power).

    For example, pow([quantity], 2) would return 16 when [quantity] is 4.

    Parameters:
    - base: The column or value to raise
    - exponent: The power to raise the base to

    Returns:
    - The result of base raised to the power of exponent
    """
    return power(base, exponent)


def mod(dividend: PlNumericType, divisor: PlNumericType) -> pl.Expr:
    """
    Calculates the remainder after dividing each value in a column (modulo).

    For example, mod([quantity], 3) would return 1 when [quantity] is 10.

    Parameters:
    - dividend: The column or value to be divided
    - divisor: The number to divide by

    Returns:
    - The remainder of the division
    """
    d = dividend if is_polars_expr(dividend) else pl.lit(dividend)
    div = divisor if is_polars_expr(divisor) else pl.lit(divisor)
    return d.mod(div)


def sign(number: PlNumericType) -> pl.Expr:
    """
    Returns the sign of each value in a column (-1, 0, or 1).

    For example, sign([quantity]) would return -1 when [quantity] is -2.

    Parameters:
    - number: The column or value to check

    Returns:
    - -1 if negative, 0 if zero, 1 if positive
    """
    if is_polars_expr(number):
        return pl.Expr.sign(number)
    else:
        return pl.lit(number).sign()


def log10(number: PlNumericType) -> pl.Expr:
    """
    Calculates the base-10 logarithm of each value in a column.

    For example, log10([salary]) would return 4.0 when [salary] is 10000.0.

    Parameters:
    - number: The column or value to take the base-10 logarithm of

    Returns:
    - The base-10 logarithm of the value
    """
    if is_polars_expr(number):
        return pl.Expr.log(number, base=10)
    else:
        return pl.lit(number).log(base=10)


def log2(number: PlNumericType) -> pl.Expr:
    """
    Calculates the base-2 logarithm of each value in a column.

    For example, log2([price]) would return 3.0 when [price] is 8.0.

    Parameters:
    - number: The column or value to take the base-2 logarithm of

    Returns:
    - The base-2 logarithm of the value
    """
    if is_polars_expr(number):
        return pl.Expr.log(number, base=2)
    else:
        return pl.lit(number).log(base=2)


def ceil(number: PlNumericType) -> pl.Expr:
    """
    Rounds each value in a column up to the nearest whole number.

    For example, ceil([price]) would return 5.0 when [price] is 4.2.

    Parameters:
    - number: The column or value to round up

    Returns:
    - The value rounded up to a whole number
    """
    if is_polars_expr(number):
        return pl.Expr.ceil(number)
    else:
        return pl.lit(number).ceil()


def round(number: PlNumericType, decimal_places: int = None) -> pl.Expr:
    """
    Rounds each value in a column to a specified number of decimal places.

    For example, round([price], 2) would return 19.99 when [price] is 19.987.

    Parameters:
    - number: The column or value to round
    - decimal_places: How many decimal places to keep (default is 0)

    Returns:
    - The rounded value
    """
    if is_polars_expr(number):
        return pl.Expr.round(number, decimal_places)
    else:
        return pl.lit(number).round(decimal_places)


def floor(number: PlNumericType) -> pl.Expr:
    """
    Rounds each value in a column down to the nearest whole number.

    For example, floor([price]) would return 4.0 when [price] is 4.7.

    Parameters:
    - number: The column or value to round down

    Returns:
    - The value rounded down to a whole number
    """
    if is_polars_expr(number):
        return pl.Expr.floor(number)
    else:
        return pl.lit(number).floor()


def tanh(number: PlNumericType) -> pl.Expr:
    """
    Calculates the hyperbolic tangent of each value in a column; the result is always between -1 and 1.

    For example, tanh([discount]) would return 0.0 when [discount] is 0.0.

    Parameters:
    - number: The column or value to take the hyperbolic tangent of

    Returns:
    - The hyperbolic tangent of the value
    """
    if is_polars_expr(number):
        return pl.Expr.tanh(number)
    else:
        return pl.lit(number).tanh()


def negative() -> int:
    """
    Returns the constant value -1; it takes no arguments.

    For example, negative() would return -1.

    Returns:
    - The value -1
    """
    return -1


def random_int(min_value: int = 0, max_value: int = 2):
    """
    Generates a random whole number for each row, from min_value (inclusive) up to max_value (exclusive).

    For example, random_int(1, 100) would return a random value such as 57.

    Parameters:
    - min_value: The smallest possible number to generate (default is 0)
    - max_value: The upper bound, which is never reached (default is 2)

    Returns:
    - A random whole number for each row in the given range
    """
    return pl.int_range(min_value, max_value).sample(n=pl.len(), with_replacement=True)