# polars_expr_transformer

`polars_expr_transformer` is a Python package designed to simplify the transformation of string-based expressions into Polars DataFrame operations. It enables users to write expressive and readable string-based functions that are automatically converted into Polars expressions for data manipulation and analysis.

## Features

- **Preprocess Expressions:** Preprocess string-based functions to prepare them for transformation.
- **Convert to Polars Expressions:** Transform preprocessed string expressions into Polars expressions.
- **Integrate with Polars DataFrames:** Easily apply transformed expressions to Polars DataFrames.

## Installation

You can install the package using pip:
```sh
pip install polars_expr_transformer
```

Usage
Here's a simplified example of how to use polars_expr_transformer:


from polars_expr_transformer.process.polars_expr_transformer import preprocess, simple_function_to_expr
import polars as pl

# Define a string-based function
test_func = """'abcd' in 'adbc'"""

# Preprocess the function
parsed_test_func = preprocess(test_func)

# Create a Polars DataFrame
df = pl.from_dicts([{'a': 'edward', 'b': 'courtney'}, {'a': 'courtney', 'b': 'edward'}])

# Apply the transformed expressions to the DataFrame
print(df.select(simple_function_to_expr("'hallo world'")))
License
This project is licensed under the MIT License. See the LICENSE file for more details.


Contact
For any questions or inquiries, please contact me.
