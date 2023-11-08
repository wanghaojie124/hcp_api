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
