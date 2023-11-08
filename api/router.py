from fastapi import APIRouter

from controller.text2img import engine
from api.schemas import RenderImage
from utils.utils import create_response

api = APIRouter()


@api.post(
    "/text2img",
)
async def text2image(args: RenderImage):
    engine.add_task(**dict(args))
    task_id = engine.task_id
    print(task_id)
    return create_response(data={"task_id": task_id}, message="success")


@api.get(
    "/progress",
)
async def get_progress():
    progress = engine.current_progress()
    return create_response(data={"current_progress": progress}, message="success")


@api.get(
    "/cancel",
)
async def cancel():
    engine.cancel()
    return create_response(message="success")
