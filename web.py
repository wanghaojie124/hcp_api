from threading import Thread

from fastapi import FastAPI

from controller.text2img import engine
from api.router import api
from logger import init_logging


app = FastAPI()


app.include_router(api, prefix="/api/v1")


@app.on_event('startup')
async def startup():
    t = Thread(target=engine.task_handler)
    t.start()

    init_logging()

