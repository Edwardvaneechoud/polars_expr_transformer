from polars_expr_transformer.process.polars_expr_transformer import simple_function_to_expr
from polars.dataframe.frame import DataFrame as PlDataFrame
from polars.lazyframe.frame import LazyFrame as PlLazyFrame
from polars.datatypes.constants import N_INFER_DEFAULT
from polars.type_aliases import FrameInitTypes, SchemaDict, SchemaDefinition, Orientation


class DataFrame(PlDataFrame):

    def __init__(self,
                 data: FrameInitTypes | None = None,
                 schema: SchemaDefinition | None = None,
                 *,
                 schema_overrides: SchemaDict | None = None,
                 strict: bool = True,
                 orient: Orientation | None = None,
                 infer_schema_length: int | None = N_INFER_DEFAULT,
                 nan_to_null: bool = False, ):
        super().__init__(data=data,
                         schema=schema,
                         schema_overrides=schema_overrides,
                         strict=strict,
                         orient=orient,
                         infer_schema_length=infer_schema_length,
                         nan_to_null=nan_to_null)

    def apply_expression(self, expression: str, column_name: str):
        """
        Apply a simple function to the DataFrame in SQL style

        This method allows the user to apply a simple function to the DataFrame and store the result in a new column.

        Parameters
        ----------
        expression : str
            The string representation of the function to apply. The function should be a valid expression that can be interpreted by the `simple_function_to_expr` method.
        column_name : str
            The name of the new column to store the result of the function.

        Returns
        -------
        DataFrame
            A new DataFrame with the result of the function applied.

        Examples
        --------
        Apply a simple concatenation function to the DataFrame:

        >>> df = DataFrame({'names': ['Alice', 'Bob'], 'surnames': ['Smith', 'Jones']})
        >>> result = df.apply_expression('concat([names], " ", [surnames])', 'full_name')
        >>> result
        shape: (2, 3)
        ┌───────┬────────┬────────────┐
        │ names ┆ surnames ┆ full_name │
        │ ---   ┆ ---     ┆ ---        │
        │ str   ┆ str     ┆ str        │
        ╞═══════╪═════════╪════════════╡
        │ Alice ┆ Smith   ┆ Alice Smith│
        │ Bob   ┆ Jones   ┆ Bob Jones  │
        └───────┴────────┴────────────┘

        """
        expr = simple_function_to_expr(expression)
        result = self.with_columns(expr.alias(column_name))
        return result


class LazyFrame(PlLazyFrame):

    def __init__(self,
                 data: FrameInitTypes | None = None,
                 schema: SchemaDefinition | None = None,
                 *,
                 schema_overrides: SchemaDict | None = None,
                 strict: bool = True,
                 orient: Orientation | None = None,
                 infer_schema_length: int | None = N_INFER_DEFAULT,
                 nan_to_null: bool = False, ):
        super().__init__(data=data,
                         schema=schema,
                         schema_overrides=schema_overrides,
                         strict=strict,
                         orient=orient,
                         infer_schema_length=infer_schema_length,
                         nan_to_null=nan_to_null)

    def apply_expression(self, expression: str, column_name: str):
        """
        Apply a simple function to the LazyFrame.

        This method allows the user to apply a simple expression to the DataFrame and store the result in a new column.

        Parameters
        ----------
        expression : str
            The string representation of the function to apply. The function should be a valid expression that can be interpreted by the `simple_function_to_expr` method.
        column_name : str
            The name of the new column to store the result of the function.

        Returns
        -------
        DataFrame
            A new DataFrame with the result of the function applied.

        Examples
        --------
        Apply a simple concatenation function to the DataFrame:

        >>> df = LazyFrame({'names': ['Alice', 'Bob'], 'surnames': ['Smith', 'Jones']})
        >>> result = df.apply_expression('concat([names], " ", [surnames])', 'full_name')
        >>> result.collect()
        shape: (2, 3)
        ┌───────┬────────┬────────────┐
        │ names ┆ surnames ┆ full_name │
        │ ---   ┆ ---     ┆ ---        │
        │ str   ┆ str     ┆ str        │
        ╞═══════╪═════════╪════════════╡
        │ Alice ┆ Smith   ┆ Alice Smith│
        │ Bob   ┆ Jones   ┆ Bob Jones  │
        └───────┴────────┴────────────┘

        """
        expr = simple_function_to_expr(expression)
        result = self.with_columns(expr.alias(column_name))
        return result


LazyFrame.__doc__ = PlLazyFrame.__doc__
DataFrame.__doc__ = PlDataFrame.__doc__
