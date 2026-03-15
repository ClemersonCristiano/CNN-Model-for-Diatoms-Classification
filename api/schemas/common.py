from typing import Generic, TypeVar

from pydantic import BaseModel

_T = TypeVar("_T")


class ApiResponse(BaseModel, Generic[_T]):
    success: bool = True
    message: str
    data: _T
