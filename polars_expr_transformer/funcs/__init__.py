import os

from polars_expr_transformer.funcs import logic_functions, string_functions, math_functions, special_funcs, date_functions

all_functions = {}
all_functions.update(logic_functions.__dict__)
all_functions.update(string_functions.__dict__)
all_functions.update(math_functions.__dict__)
all_functions.update(special_funcs.__dict__)
all_functions.update(date_functions.__dict__)
