from pydantic import BaseModel, Field


class RenderImage(BaseModel):
    prompt: str
    clip_skip: int = Field(default=1)
    bs: int = Field(default=1)
    num: int = Field(default=1)
