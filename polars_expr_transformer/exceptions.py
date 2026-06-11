"""Custom exceptions for the Polars Expression Transformer."""


class ExpressionSyntaxError(ValueError):
    """Raised when a string expression cannot be parsed.

    Subclasses ValueError for backward compatibility with code that catches
    ValueError (e.g. the historical unbalanced-parentheses error).

    Attributes:
        bare_message: The one-line description of the problem.
        expression: The original expression string, if available.
        position: 0-based character index into `expression`, if known.
        hint: Optional suggestion for fixing the problem.
    """

    def __init__(
        self,
        message: str,
        expression: str = None,
        position: int = None,
        hint: str = None,
    ):
        self.bare_message = message
        self.expression = expression
        self.position = position
        self.hint = hint
        super().__init__(self._format())

    def _format(self) -> str:
        parts = [self.bare_message]
        if self.expression is not None and self.position is not None:
            line_start = self.expression.rfind("\n", 0, self.position) + 1
            line_end = self.expression.find("\n", self.position)
            if line_end == -1:
                line_end = len(self.expression)
            parts.append(self.expression[line_start:line_end])
            parts.append(" " * (self.position - line_start) + "^")
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        return "\n".join(parts)


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
