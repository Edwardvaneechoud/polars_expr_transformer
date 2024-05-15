import os
import importlib.util
import inspect
from polars_expr_transformer.schemas import ExpressionRef, ExpressionsOverview
from typing import List, Optional


_available_expressions: Optional[ExpressionsOverview] = None


def get_formula_scripts():
    try:
        current_file_ref = os.path.abspath(__file__)
        current_file_path = os.path.dirname(current_file_ref)
    except:
        current_file_path = os.path.sep.join(('backend','flowfile','rowFuncs'))
    script_file_path = os.path.join(current_file_path, 'funcs')
    python_files = ((f[:-3], os.path.join(script_file_path,f)) for f in os.listdir(script_file_path) if '.py')
    filtered_files = (f for f in python_files if f[0][:min(2,len(f))] != '__' and f[0] != 'utils')
    return filtered_files


def get_module_members(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module_members = inspect.getmembers(module)
    module_function_names = [ExpressionRef(name=name, doc=obj.__doc__) for name, obj in module_members
                             if inspect.isfunction(obj) and obj.__module__ == module.__name__
                             and name[:min(len(name),2)]!='__']
    return module_function_names


def get_expression_overview() -> List[ExpressionsOverview]:
    global _available_expressions
    if _available_expressions is None:
        scripts = get_formula_scripts()
        module_members = [ExpressionsOverview(expression_type = module_name,
                                              expressions = get_module_members(module_name, path))
                          for module_name, path in scripts]
        _available_expressions = module_members
    return _available_expressions


def get_all_expressions() -> List[str]:
    expression_overview = get_expression_overview()
    return [expression.name for eo in expression_overview for expression in eo.expressions]




