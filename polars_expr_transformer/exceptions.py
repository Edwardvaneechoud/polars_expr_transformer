"""Custom exceptions for the Polars Expression Transformer."""


class PolarsCodeGenError(Exception):
    """Raised when generated Polars code fails eval-based validation.

    Attributes:
        expression: The original string expression that was transformed.
        generated_code: The Polars Python code string that was generated.
        eval_error: The underlying exception from eval().
    """

    def __init__(self, expression: str, generated_code: str, eval_error: Exception):
        self.expression = expression
        self.generated_code = generated_code
        self.eval_error = eval_error
        super().__init__(
            f"Generated invalid Polars code for expression: {expression!r}\n"
            f"Generated code: {generated_code}\n"
            f"Eval error: {eval_error}"
        )
