from typing import List

from pydantic import BaseModel


class DataInput(BaseModel):
    text: List[str]
    target: List[int]
