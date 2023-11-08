import os
from pydantic import BaseModel, Field

from config import ROOT_PATH, MODEL_PATH, OUTPUT_PATH, EMBS_PATH


class SaveTemplate(BaseModel):
    out_dir: str = Field(default=OUTPUT_PATH)
    save_cfg: bool = Field(default=True)
    image_type: str = Field(default='png')
    quality: int = Field(default=95)


class RenderImage(BaseModel):
    pretrained_model: str = Field(default=os.path.join(MODEL_PATH, os.listdir(MODEL_PATH)[0]))
    prompt: str
    clip_skip: int = Field(default=1)
    bs: int = Field(default=1)
    num: int = Field(default=1)
    emb_dir: str = Field(EMBS_PATH)
    save: dict = Field(default=dict(SaveTemplate()))
    task_id: str = ""


# TODO 适配controlnet 图生图
