Polars Expression Transformer
=============================

Polars Expression Transformer is a Python library that provides a simple and intuitive way to transform and manipulate data using Polars expressions. It is designed for users who are familiar with SQL and Tableau and want to leverage the power of Polars for data processing tasks.

Purpose
-------

The purpose of Polars Expression Transformer is to provide a high-level interface for working with Polars expressions. It allows users to write simple string expressions that can be easily translated into Polars expressions, making it easy to perform complex data transformations without having to write Python code.

Target Group
------------

Polars Expression Transformer is ideal for users who:

* Are familiar with SQL or Tableau and want to use Polars for data transformation tasks.
* Want to perform complex data transformations without having to write Python code.
* Need to integrate Polars into an application or tool and want to provide a simple and intuitive interface for users to perform data transformations.

When to Use
-----------

Polars Expression Transformer is particularly useful in the following scenarios:

* When you are not directly exposed to Python, for example in an application.
* When you want to provide a simple and intuitive interface for users to perform complex data transformations without having to write Python code.

When Not to Use
--------------

Polars Expression Transformer may not be the best choice for users who:

* Are already familiar with Polars and are developing in an IDE. In this case, it may be more efficient to write Polars expressions directly.
* Want to have the best performance and all features of Polars. Polars Expression Transformer adds an additional layer on top of Polars, which may result in a performance overhead.
* Need to perform low-level optimizations or custom transformations that are not supported by Polars Expression Transformer

Installation
------------

To install Polars Expression Transformer, you can use pip:
```
pip install polars-expr-transformer
```
Examples
----------------

Let's say you have a Polars DataFrame `df` with columns "names" and "subnames", and you want to create a new column "combined" that concatenates the values in "names" and "subnames" with a space in between.

Without Polars Expression Transformer, you would need to write Python code to accomplish this:
```python
df = df.with_column(pl.col("names") + " " + pl.col("subnames").alias("combined"))
```
With Polars Expression Transformer, you can write a simple string expression instead:
```python
from polars_expr_transformer.process.polars_expr_transformer import simple_function_to_expr

df = df.select(simple_function_to_expr('concat([names], " ", [subnames])').alias("combined"))
```
This makes it easy to perform complex data transformations without having to write Python code.

Using DataFrame and LazyFrame
-----------------------------

Polars Expression Transformer provides `DataFrame` and `LazyFrame` classes that extend the functionality of Polars' native `DataFrame` and `LazyFrame` classes, allowing you to apply simple functions using string expressions.

### DataFrame

The `DataFrame` class allows you to apply simple functions to a DataFrame and store the results in a new column.

Example:

```python
from polars_expr_transformer import DataFrame

df = DataFrame({'names': ['Alice', 'Bob'], 'surnames': ['Smith', 'Jones']})
result = df.apply_expression('concat([names], " ", [surnames])', 'full_name')
print(result)

```
```commandline
output:
shape: (2, 3)
┌───────┬─────────┬────────────┐
│ names ┆ surnames┆ full_name  │
│ ---   ┆ ---     ┆ ---        │
│ str   ┆ str     ┆ str        │
╞═══════╪═════════╪════════════╡
│ Alice ┆ Smith   ┆ Alice Smith│
│ Bob   ┆ Jones   ┆ Bob Jones  │
└───────┴─────────┴────────────┘
```

Built on Polars
--------------

Polars Expression Transformer is built on top of the amazing Polars library. Polars is a blazing fast DataFrame library implemented in Rust and Python. It is designed to be a high-performance alternative to Pandas and other DataFrame libraries. I highly recommend checking out Polars if you are working with large datasets or need to perform complex data transformations quickly.

Acknowledgements
----------------

We would like to thank the Polars team for creating such an amazing library.