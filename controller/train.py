import time

from queue import Queue

from hcpdiff.train_ac import Trainer
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
                logger.info(f"训练任务队列：{self.task_queue.qsize()}")
                self.task_id = args.get("task_id")
                self.run(args)
                time.sleep(0.5)
                GPU_LOCK.release()
                logger.info(f"{self.task_id}训练完成")
            except Exception as e:
                logger.error(e)
            finally:
                GPU_LOCK.release()


trainer = TrainerController()
