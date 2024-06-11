from polars_expr_transformer.configs.settings import PRECEDENCE
from typing import TypeAlias, Literal, List, Union, Optional, Any, Callable
from polars_expr_transformer.funcs.utils import PlStringType, PlIntType
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
    return _type in [PlStringType, PlIntType, pl.Expr, Any, inspect._empty]


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

    def get_readable_pl_function(self):
        return f'{self.func_ref.val}({", ".join([arg.get_readable_pl_function() for arg in self.args])})'

    def add_arg(self, arg: Union["Func", Classifier, "IfFunc"]):
        self.args.append(arg)
        arg.parent = self

    def get_pl_func(self):
        if self.func_ref == 'pl.lit':
            if len(self.args) != 1:
                raise Exception('Expected must contain 1 argument not more not less')
            if isinstance(self.args[0].get_pl_func(), pl.expr.Expr):
                return self.args[0].get_pl_func()
            return funcs[self.func_ref.val](self.args[0].get_pl_func())
        args = [arg.get_pl_func() for arg in self.args]
        if all_numeric_types(args):
            args = ensure_all_numeric_types_align(args)
        func = funcs[self.func_ref.val]
        if any(isinstance(arg, pl.Expr) for arg in args) and any(not isinstance(arg, pl.Expr) for arg in args):
            func_types = get_types_from_func(func)
            standardized_args = []
            if len(func_types) == len(args):
                for func_type, arg in zip(func_types, args):
                    if not isinstance(arg, pl.Expr) and allow_expressions(func_type):
                        standardized_args.append(pl.lit(arg))
                    else:
                        standardized_args.append(arg)
            else:
                standardized_args = [pl.lit(arg) if not isinstance(arg, pl.Expr) else arg for arg in args]

        else:
            standardized_args = args
        r = func(*standardized_args)

        if isinstance(r, NotImplementedType):
            try:
                readable_pl_function = self.get_readable_pl_function()
                logging.warning(f'Not implemented type: {self.get_readable_pl_function()}')
            except:
                logging.warning('Not implemented type')
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
