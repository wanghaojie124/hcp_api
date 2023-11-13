import os
import base64
from typing import Optional, Dict

from orjson import orjson
from pydantic.generics import GenericModel


def orjson_dumps(v, *, default):
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    return orjson.dumps(v, default=default).decode()


class IResponseBase(GenericModel):
    message: str = ""
    meta: Optional[Dict] = {}
    data: None | str | dict
    errcode: int = 0

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps


def create_response(
        data: dict = None,
        message: Optional[str] = "",
        errcode: Optional[int] = 0,
        meta=None,
):
    return IResponseBase(data=data, message=message, errcode=errcode, meta=meta)


class MyList:

    def __init__(self, max_length):
        self.max_length = max_length
        self.data = list()

    def put(self, data):
        if len(self.data) >= self.max_length:
            self.data.append(data)
            self.data.pop(0)
        else:
            self.data.append(data)


def image2base64(image):
    with open(image, 'rb') as f:
        b64encode = base64.b64encode(f.read())
        b64_encode = f"data:image/png;base64,{b64encode.decode()}"
    return b64_encode


def base64_to_image(base64_encod_str: str, save_path: str):
    if "data:image" in base64_encod_str:
        base64_encod_str = base64_encod_str.split(",")[-1]
    img_b64decode = base64.b64decode(base64_encod_str)
    # 保存图片
    with open(save_path, 'wb') as img:
        img.write(img_b64decode)


def list_full_path(directory):
    """
    以绝对路径列出文件夹下所有文件
    :param directory:
    :return:
    """
    return [os.path.join(directory, file) for file in os.listdir(directory)]
