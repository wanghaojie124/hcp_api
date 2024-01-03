import time

from queue import Queue

from hcpdiff.train_ac import Trainer
from hcpdiff.utils.utils import load_config_with_cli
from config import lora_tem_path, GPU_LOCK
from logger import logger


class TrainerController:

    def __init__(self):
        self.task_queue = Queue(10)
        self.task_id = None

    def run(self, args: dict):
        self.train(args)

    def train(self, cfgs: dict):
        _trainer = Trainer(cfgs)
        _trainer.train()

    def task_handler(self):
        while 1:
            try:
                args = self.task_queue.get()
                GPU_LOCK.acquire()
                self.task_id = args.get("task_id")
                self.run(args)
                time.sleep(0.5)
                GPU_LOCK.release()
            except Exception as e:
                logger.error(e)
            finally:
                GPU_LOCK.release()


trainer = TrainerController()
