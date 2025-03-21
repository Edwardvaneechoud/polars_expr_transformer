from polars_expr_transformer.configs.settings import PRECEDENCE
from typing import TypeAlias, Literal, List, Union, Optional, Any, Callable
from polars_expr_transformer.funcs.utils import PlStringType, PlIntType, PlNumericType
from polars_expr_transformer.configs.settings import operators, funcs
from polars_expr_transformer.configs import logging
from dataclasses import dataclass, field
import polars as pl
from types import NotImplementedType
import inspect


def get_types_from_func(func: Callable):
    """
    Get the types of the parameters of a function.

    Args:
        func: The function to inspect.

    Returns:
        A list of types of the function's parameters.
    """
    return [param.annotation for param in inspect.signature(func).parameters.values()]


def ensure_all_numeric_types_align(numbers: List[Union[int, float]]):
    """
    Ensure all numeric types in the list are aligned to either all int or all float.

    Args:
        numbers: A list of numbers.

    Returns:
        A list of numbers with aligned types.

    Raises:
        Exception: If the numbers are not all of numeric types.
    """
    if not all_numeric_types(numbers):
        raise Exception('Expected all numbers to be of type int')
    if all(isinstance(number, int) for number in numbers):
        return numbers
    if all(isinstance(number, float) for number in numbers):
        return numbers
    return [float(number) for number in numbers]


def all_numeric_types(numbers: List[any]):
    """
    Check if all elements in the list are numeric types.

    Args:
        numbers: A list of elements to check.

    Returns:
        True if all elements are numeric types, False otherwise.
    """
    return all(isinstance(number, (float, int, bool)) for number in numbers)


def allow_expressions(_type):
    """
    Check if a type allows expressions.

    Args:
        _type: The type to check.

    Returns:
        True if the type allows expressions, False otherwise.
    """
    return _type in [PlStringType, PlIntType, pl.Expr, Any, inspect._empty, PlNumericType]


def allow_non_pl_expressions(_type):
    """
    Check if a type allows non-polars expressions.

    Args:
        _type: The type to check.

    Returns:
        True if the type allows non-polars expressions, False otherwise.
    """
    return _type in [str, int, float, bool, PlIntType, PlStringType, Any, inspect._empty]


value_type: TypeAlias = Literal['string', 'number', 'boolean', 'operator', 'function', 'column', 'empty', 'case_when',
'prio', 'sep', 'special']


@dataclass
class Classifier:
    """
    Represents a token or a value in the expression with its type, precedence, and parent function.

    Attributes:
        val (str): The value of the classifier.
        val_type (value_type): The type of the value.
        precedence (int): The precedence of the value in expressions.
        parent (Optional[Union["Classifier", "Func"]]): The parent of this classifier.
    """
    val: str
    val_type: value_type = None
    precedence: int = None
    parent: Optional[Union["Classifier", "Func"]] = field(repr=False, default=None)

    def __post_init__(self):
        self.val_type = self.get_val_type()
        self.precedence = self.get_precedence()

    def get_precedence(self):
        return PRECEDENCE.get(self.val)

    def get_val_type(self) -> value_type:
        if self.val.lower() in ['true', 'false']:
            return 'boolean'
        elif self.val in operators:
            return 'operator'
        elif self.val in ('(', ')'):
            return 'prio'
        elif self.val == '':
            return 'empty'
        elif self.val in funcs:
            return 'function'
        elif self.val in ('$if$', '$then$', '$else$', '$endif$'):
            return 'case_when'
        elif self.val.isdigit():
            return 'number'
        elif self.val.startswith('-') and len(self.val) > 1 and self.val[1:].isdigit():
            return 'number'
        elif self.val == '__negative()':
            return 'special'
        elif self.val.isalpha():
            return 'string'
        elif self.val == ',':
            return 'sep'
        else:
            return 'string'

    def get_pl_func(self):
        if self.val_type == 'boolean':
            return True if self.val.lower() == 'true' else False
        elif self.val_type == 'function':
            return funcs[self.val]
        elif self.val_type in ('number', 'string'):
            return eval(self.val)
        elif self.val == '__negative()':
            return funcs['__negative']()
        else:
            raise Exception('Did not expect to get here')

    def get_repr(self):
        return str(self.val)

    def __eq__(self, other):
        return self.val == other

    def __hash__(self):
        return hash(self.val)

    def get_readable_pl_function(self):
        return self.val


@dataclass
class Func:
    """
    Represents a function in the expression with its reference, arguments, and parent function.

    Attributes:
        func_ref (Union[Classifier, "IfFunc"]): The reference to the function or classifier.
        args (List[Union["Func", Classifier, "IfFunc"]]): The list of arguments for the function.
        parent (Optional["Func"]): The parent function of this function.
    """
    func_ref: Union[Classifier, "IfFunc"]
    args: List[Union["Func", Classifier, "IfFunc"]] = field(default_factory=list)
    parent: Optional["Func"] = field(repr=False, default=None)

    @staticmethod
    def _check_if_standardization_of_args_is_needed(args: List[pl.Expr | Any]) -> bool:
        """
        Check if arguments need standardization based on mixed Polars expression types.

        This method determines whether standardization is needed by checking if the argument
        list contains a mix of Polars expressions and non-Polars expressions. Standardization
        is necessary when some arguments are pl.Expr instances and others are not.

        Args:
            args: A list of arguments that may contain Polars expressions and other types.

        Returns:
            bool: True if standardization is needed (mixed pl.Expr types), False otherwise.
        """
        return any(isinstance(arg, pl.Expr) for arg in args) and any(not isinstance(arg, pl.Expr) for arg in args)

    def get_readable_pl_function(self):
        """
        Generate a human-readable string representation of the Polars function.

        This method creates a string representation of the function call that can be read
        and understood by humans. It handles special cases like 'pl.lit', ensures numeric type
        alignment, and properly formats arguments based on their types.

        Special handling is applied when mixing Polars expressions with non-Polars values,
        where non-Polars values may need to be wrapped with pl.lit() for compatibility.

        Returns:
            str: A string representation of the Polars function call.

        Raises:
            Exception: If 'pl.lit' is used with an incorrect number of arguments.
        """

        if self.func_ref == 'pl.lit':
            if len(self.args) != 1:
                raise Exception('Expected must contain 1 argument not more not less')
            if isinstance(self.args[0].get_pl_func(), pl.expr.Expr):
                return self.args[0].get_readable_pl_function()
        pl_args = [arg.get_pl_func() for arg in self.args]
        if all_numeric_types(pl_args):
            pl_args = ensure_all_numeric_types_align(pl_args)
        func = funcs[self.func_ref.val]
        if self._check_if_standardization_of_args_is_needed(pl_args):
            func_types = get_types_from_func(func)
            standardized_args: List[str] = []
            if len(func_types) == len(pl_args):
                for func_type, pl_arg, arg in zip(func_types, pl_args, self.args):
                    if not isinstance(pl_arg, pl.Expr) and allow_expressions(func_type):
                        standardized_args.append(f'pl.lit({arg.get_readable_pl_function()})')
                    else:
                        standardized_args.append(arg.get_readable_pl_function())
            else:
                standardized_args = [f'pl.lit({arg.get_readable_pl_function()})'
                                     if not isinstance(pl_arg, pl.Expr) else arg.get_readable_pl_function()
                                     for pl_arg, arg in zip(pl_args, self.args)]
        else:
            standardized_args = [arg.get_readable_pl_function() for arg in self.args]
        return f'{self.func_ref.val}({", ".join(standardized_args)})'

    def add_arg(self, arg: Union["Func", Classifier, "IfFunc"]):
        """
        Add an argument to this function and set its parent reference.

        This method appends the provided argument to the function's argument list
        and establishes a parent-child relationship by setting this function as
        the parent of the argument.

        Args:
            arg: The argument to add, which can be a Func, Classifier, or IfFunc instance.
        """
        self.args.append(arg)
        arg.parent = self

    def get_pl_func(self):
        """
        Execute and return the actual Polars function result.

        This method evaluates the function with its arguments, handling special cases
        like 'pl.lit', ensuring numeric type alignment, and standardizing argument types
        when necessary. It applies the function to the processed arguments and returns
        the result.

        The method also includes error handling for NotImplementedType results,
        which can occur with unsupported operations.

        Returns:
            The result of applying the Polars function to the arguments, or False if the
            operation is not implemented.

        Raises:
            Exception: If 'pl.lit' is used with an incorrect number of arguments.
        """
        if self.func_ref == 'pl.lit':
            if len(self.args) != 1:
                raise Exception('Expected must contain 1 argument not more not less')
            if isinstance(self.args[0].get_pl_func(), pl.expr.Expr):
                return self.args[0].get_pl_func()
            return funcs[self.func_ref.val](self.args[0].get_pl_func())
        pl_args = [arg.get_pl_func() for arg in self.args]
        if all_numeric_types(pl_args):
            pl_args = ensure_all_numeric_types_align(pl_args)
        func = funcs[self.func_ref.val]
        if self._check_if_standardization_of_args_is_needed(pl_args):
            func_types = get_types_from_func(func)
            standardized_args = []
            if len(func_types) == len(pl_args):
                for func_type, arg in zip(func_types, pl_args):
                    if not isinstance(arg, pl.Expr) and allow_expressions(func_type):
                        standardized_args.append(pl.lit(arg))
                    else:
                        standardized_args.append(arg)
            else:
                standardized_args = [pl.lit(arg) if not isinstance(arg, pl.Expr) else arg for arg in pl_args]

        else:
            standardized_args = pl_args
        r = func(*standardized_args)

        if isinstance(r, NotImplementedType):
            try:
                logging.warning(f'Not implemented type: {self.get_readable_pl_function()}')
            except Exception as e:
                logging.warning('Not implemented type')
                logging.debug(e)
            return False
        return r


@dataclass
class ConditionVal:
    """
    Represents a condition value used in conditional functions with references to condition and value functions.

    Attributes:
        func_ref (Union[Classifier, "IfFunc", "Func"]): The reference to the function or classifier.
        condition (Func): The condition function.
        val (Func): The value function.
        parent ("IfFunc"): The parent IfFunc of this condition value.
    """
    func_ref: Union[Classifier, "IfFunc", "Func"] = None
    condition: Func = None
    val: Func = None
    parent: "IfFunc" = field(repr=False, default=None)

    def __post_init__(self):
        if self.condition:
            self.condition.parent = self
        if self.val:
            self.val.parent = self

    def get_pl_func(self):
        return pl.when(self.condition.get_pl_func()).then(self.val.get_pl_func())

    def get_pl_condition(self):
        return self.condition.get_pl_func()

    def get_pl_val(self):
        return self.val.get_pl_func()

    def get_readable_pl_function(self) -> str:
        when_str = self.condition.get_readable_pl_function()
        then_str = self.val.get_readable_pl_function()
        return f"pl.when({when_str}).then({then_str})"


@dataclass
class IfFunc:
    """
    Represents an if function with its reference, conditions, else value, and parent function.

    Attributes:
        func_ref (Union[Classifier]): The reference to the classifier function.
        conditions (Optional[List[ConditionVal]]): The list of condition values.
        else_val (Optional[Func]): The else value function.
        parent (Optional[Func]): The parent function of this if function.
    """
    func_ref: Union[Classifier]
    conditions: Optional[List[ConditionVal]] = field(default_factory=list)
    else_val: Optional[Func] = None
    parent: Optional[Func] = field(repr=False, default=None)

    def add_condition(self, condition: ConditionVal):
        self.conditions.append(condition)
        condition.parent = self

    def add_else_val(self, else_val: Func):
        self.else_val = else_val
        else_val.parent = self

    def get_pl_func(self):
        full_expr = None
        if len(self.conditions) == 0:
            raise Exception('Expected at least one condition')
        for condition in self.conditions:
            if full_expr is None:
                full_expr = pl.when(condition.get_pl_condition()).then(condition.get_pl_val())
            else:
                full_expr = full_expr.when(condition.get_pl_condition()).then(condition.get_pl_val())
        return full_expr.otherwise(self.else_val.get_pl_func())

    def get_readable_pl_function(self) -> str:
        full_expr_str: Optional[str] = None
        for condition in self.conditions:
            when_str = condition.condition.get_readable_pl_function()
            then_str = condition.val.get_readable_pl_function()
            if full_expr_str is None:
                full_expr_str = f'pl.when({when_str}).then({then_str})'
            else:
                full_expr_str += f'.when({when_str}).then({then_str})'

        full_expr_str += f'.otherwise({self.else_val.get_readable_pl_function()})'
        return full_expr_str


@dataclass
class TempFunc:
    """
    Represents a temporary function used during parsing with a list of arguments.

    Attributes:
        args (List[Union["Func", Classifier, "IfFunc"]]): The list of arguments for the temporary function.
    """
    args: List[Union["Func", Classifier, "IfFunc"]] = field(default_factory=list)

    def add_arg(self, arg: Union["Func", Classifier, "IfFunc"]):
        self.args.append(arg)
        arg.parent = self

