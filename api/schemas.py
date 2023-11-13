from typing import Optional, List
from pydantic import BaseModel, Field


class InferArgsTemplate(BaseModel):
    width: int = Field(default=512)
    height: int = Field(default=512)
    guidance_scale: float = Field(default=7.5)
    num_inference_steps: int = Field(default=50)
    strength: Optional[float] = Field(default=0.75)


class Img2ImgTemplate(BaseModel):
    type: str = Field(default='i2i')
    image: str
    mask: Optional[str] = Field(description="重绘蒙版图")


class ControlnetTemplate(BaseModel):
    image: str = Field(description="预处理后的图片")
    model: str = Field(description="controlnet模型")


class LoraTemplate(BaseModel):
    type: str = Field(description="unet or TE")
    alpha: float = Field(default=0.8)
    layers: str = Field(default="all")
    name: str = Field(description="lora名称")


class RenderImage(BaseModel):
    pretrained_model: str = Field(description="模型名称")
    prompt: str = Field(description="正向提示词")
    clip_skip: int = Field(default=1)
    bs: int = Field(default=1, description="单批次生图数量")
    num: int = Field(default=1, description="总共生成批次")
    N_repeats: int = Field(default=1)
    controlnet: ControlnetTemplate = Field(default=None, description="额外参数，目前controlnet使用时需要用到")
    condition: Img2ImgTemplate = Field(default=None, description="条件描述，图生图需要添加参数")
    infer_args: dict = Field(default=dict(InferArgsTemplate()))
    lora: List[LoraTemplate] = Field(default=None, description="lora模型列表")

