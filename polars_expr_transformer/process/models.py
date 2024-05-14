from polars_expr_transformer.configs.settings import PRECEDENCE
from typing import TypeAlias, Literal, List, Union, Optional, Any
from polars_expr_transformer.configs.settings import operators, funcs
from dataclasses import dataclass, field
import polars as pl


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


@dataclass
class Func:
    func_ref: Union[Classifier, "IfFunc"]
    args: List[Union["Func", Classifier, "IfFunc"]] = field(default_factory=list)
    parent: Optional["Func"] = field(repr=False, default=None)

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
        if any(isinstance(arg, pl.Expr) for arg in args) and any(not isinstance(arg, pl.Expr) for arg in args):
            standardized_args = []
            for arg in args:
                if not isinstance(arg, pl.Expr):
                    standardized_args.append(pl.lit(arg))
                else:
                    standardized_args.append(arg)
        else:
            standardized_args = args
        return funcs[self.func_ref.val](*standardized_args)


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
