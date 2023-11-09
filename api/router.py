import os.path
import uuid

from fastapi import APIRouter

from controller.text2img import engine
from api.schemas import RenderImage
from utils.utils import create_response

api = APIRouter()


@api.post(
    "/text2img",
)
async def text2image(args: RenderImage):
    task_id = uuid.uuid1().hex
    args.task_id = engine.task_id = task_id
    args.save['out_dir'] = os.path.join(args.save['out_dir'], task_id)
    res = engine.add_task(dict(args))
    if res:
        return create_response(data={"task_id": task_id}, message="success")
    else:
        return create_response(data={}, message="当前服务器繁忙，请稍后再试")


@api.get(
    "/progress",
)
async def get_progress(task_id: str):
    progress = engine.current_progress(task_id)
    return create_response(data={"current_progress": progress}, message="success")


@api.get(
    "/cancel",
)
async def cancel(task_id):
    engine.cancel(task_id)
    return create_response(message="success")


# TODO 图片下载
@api.get(
    "/download",
)
async def download(task_id):
    res = engine.get_result(task_id)
    return create_response(data={"images": res}, message="success")
