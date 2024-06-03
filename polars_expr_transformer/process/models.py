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
    return [param.annotation for param in inspect.signature(func).parameters.values()]


def ensure_all_numeric_types_align(numbers: List[Union[int, float]]):
    if not all_numeric_types(numbers):
        raise Exception('Expected all numbers to be of type int')
    if all(isinstance(number, int) for number in numbers):
        return numbers
    if all(isinstance(number, float) for number in numbers):
        return numbers
    return [float(number) for number in numbers]


def all_numeric_types(numbers: List[any]):
    return all(isinstance(number, (float, int, bool)) for number in numbers)


def allow_expressions(_type):
    return _type in [PlStringType, PlIntType, pl.Expr, Any, inspect._empty]


def allow_non_pl_expressions(_type):
    return _type in [str, int, float, bool, PlIntType, PlStringType, Any, inspect._empty]


value_type: TypeAlias = Literal['string', 'number', 'boolean', 'operator', 'function', 'column', 'empty', 'case_when',
                                'prio', 'sep', 'special']


@dataclass
class Classifier:
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
            if len(self.args)!=1:
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
        if len(self.conditions)==0:
            raise Exception('Expected at least one condition')
        for condition in self.conditions:
            if full_expr is None:
                full_expr = pl.when(condition.get_pl_condition()).then(condition.get_pl_val())
            else:
                full_expr = full_expr.when(condition.get_pl_condition()).then(condition.get_pl_val())
        return full_expr.otherwise(self.else_val.get_pl_func())


@dataclass
class TempFunc:
    args: List[Union["Func", Classifier, "IfFunc"]] = field(default_factory=list)

    def add_arg(self, arg: Union["Func", Classifier, "IfFunc"]):
        self.args.append(arg)
        arg.parent = self
