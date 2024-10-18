from pydantic import BaseModel
from typing import Optional, List


class ExpressionRef(BaseModel):
    name: str
    doc: Optional[str] = None


class ExpressionsOverview(BaseModel):
    expression_type: str
    expressions: List[ExpressionRef]
