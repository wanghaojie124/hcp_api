import os
from threading import Thread

from fastapi import FastAPI

from controller.text2img import engine
from controller.train import trainer
from api.router import api
from logger import init_logging
from config import train_data_path


app = FastAPI()


app.include_router(api, prefix="/api/v1")


@app.on_event('startup')
async def startup():
    os.makedirs(train_data_path, exist_ok=True)

    t = Thread(target=engine.task_handler)
    t.start()

    t = Thread(target=trainer.task_handler)
    t.start()

    init_logging()

