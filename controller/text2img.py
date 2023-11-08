import os.path
import time
import uuid
import copy
from queue import Queue
from loguru import logger

from hcpdiff import visualizer
from hcpdiff.utils.utils import load_config_with_cli
from hcpdiff.vis.base_interface import BaseInterface

from config import ROOT_PATH


class ProgressInterface(BaseInterface):
    def __init__(self):
        super().__init__(show_steps=1)
        self.current_step = 0
        self.total_step = 1
        self.img_name_list = []
        self.interrupt = False

    def on_inter_step(self, i, num_steps, t, latents, images):
        if i + 1 < num_steps:  # 最后一步要等图片储存完毕
            self.current_step = i + 1
            self.total_step = num_steps

        return self.interrupt


class Text2Image:

    def __init__(self):
        self.model_path = os.path.join(ROOT_PATH, 'models/hcp/')
        self.base_cfg = {
            "pretrained_model": os.path.join(self.model_path, os.listdir(self.model_path)[0]),
            "emb_dir": os.path.join(ROOT_PATH, 'models/embs/'),
            "save": {
                "out_dir": os.path.join(ROOT_PATH, 'output'),
                "save_cfg": True,
                "image_type": "png",
                "quality": 95,
            }
        }
        self.task_queue = Queue()
        self.current_instance = None
        self.task_id = None
        self.progress_interface = ProgressInterface()

    def add_task(self, **kwargs):
        self.task_queue.put(kwargs)

    def task_handler(self):
        while 1:
            args = self.task_queue.get()
            self.text2img(**args)
            time.sleep(0.5)

    def text2img(self, **kwargs):
        self.task_id = uuid.uuid1().hex
        cfg_dict = copy.deepcopy(self.base_cfg)
        cfg_dict.update(kwargs)
        cfgs = load_config_with_cli(
            os.path.join(ROOT_PATH, "HCP-Diffusion/cfgs/infer/text2img.yaml"),
            args_list=[
                f'{k}={v}' for k, v in cfg_dict.items()
            ]
        )
        self.current_instance = visualizer.Visualizer(cfgs)
        self.current_instance.cfgs.interface.append(self.progress_interface)
        if cfgs.seed is not None:
            seeds = list(range(cfgs.seed, cfgs.seed + cfgs.num * cfgs.bs))
        else:
            seeds = [None] * (cfgs.num * cfgs.bs)
        batch_num = cfgs.get("num")
        for i in range(batch_num):
            prompt = cfgs.prompt[i * cfgs.bs:(i + 1) * cfgs.bs] if isinstance(cfgs.prompt, list) \
                else [cfgs.prompt] * cfgs.bs
            negative_prompt = cfgs.neg_prompt[i * cfgs.bs:(i + 1) * cfgs.bs] if isinstance(cfgs.neg_prompt, list) \
                else [cfgs.neg_prompt] * cfgs.bs
            try:
                self.current_instance.vis_to_dir(
                    prompt=prompt, negative_prompt=negative_prompt,
                    seeds=seeds[i * cfgs.bs:(i + 1) * cfgs.bs], save_cfg=cfgs.save.save_cfg,
                    **cfgs.infer_args)
            except AttributeError:
                logger.info(f"cancel task: {self.task_id}")

    def current_progress(self):
        progress = self.progress_interface.current_step / self.progress_interface.total_step
        return progress

    def cancel(self):
        self.progress_interface.interrupt = True


engine = Text2Image()

if __name__ == '__main__':
    engine.text2img(prompt="1girl", num=1, bs=4)
