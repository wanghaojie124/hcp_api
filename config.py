import os
from threading import Lock


root_path = os.path.abspath(os.path.dirname(__file__))
GPU_LOCK = Lock()

# hcp配置项
model_path = os.path.join(root_path, 'models/hcp/')
embs_path = os.path.join(root_path, 'models/embs/')
controlnet_model_path = os.path.join(root_path, 'models/controlnet/')
lora_model_path = os.path.join(root_path, 'models/lora/')
output_path = os.path.join(root_path, 'output/')
init_image_path = os.path.join(root_path, 'output/init_images/')
for path in [
    model_path, embs_path, controlnet_model_path, output_path, init_image_path, lora_model_path
]:
    os.makedirs(path, exist_ok=True)
t2i_cfg_path = os.path.join(root_path, "cfgs/infer/text2img.yaml")
i2i_cfg_path = os.path.join(root_path, "cfgs/infer/img2img.yaml")
lora_tem_path = os.path.join(root_path, "lora_pt.yaml")
train_data_path = os.path.join(root_path, "train_data")
controlnet_plugin_cfg_path = os.path.join(root_path, "cfgs/plugins/plugin_controlnet.yaml")
save_cfg = {
    'out_dir': 'output/',
    'save_cfg': True,
    'image_type': 'png',
    'quality': 95
}
ex_input = {
    "cond": {
        "_target_": "hcpdiff.data.data_processor.ControlNetProcessor",
        "image": ''  # 处理后的controlnet输入图
    }
}
cnet_merge = {
    "plugin_cfg": controlnet_plugin_cfg_path,
    "group1": {
        "type": "unet",
        "base_model_alpha": 1.0,
        "lora": None,
        "part": None,
        "plugin": {
            "controlnet1": {
                "path": "",  # controlnet模型
                "layers": "all"
            }
        }

    }
}
lora_unet = {
    "type": "unet",
    "base_model_alpha": 1.0,
    "lora": [],
    "part": None
}
lora_te = {
    "type": "TE",
    "base_model_alpha": 1.0,
    "lora": [],
    "part": None
}
