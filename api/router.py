import os.path
import copy
import uuid

from fastapi import APIRouter

from controller.text2img import engine
from api.schemas import RenderImage
from utils.utils import create_response, base64_to_image
from config import embs_path, save_cfg, output_path, model_path, ex_input, cnet_merge, controlnet_model_path,\
    init_image_path, lora_unet, lora_te, lora_model_path

api = APIRouter()


@api.post(
    "/text2img",
)
async def text2image(args: RenderImage):
    data = {}
    merge = {}
    task_id = uuid.uuid1().hex
    data['task_id'] = engine.task_id = task_id
    save_cfg['out_dir'] = os.path.join(output_path, task_id)
    args.pretrained_model = os.path.join(model_path, args.pretrained_model)
    args = dict(args)
    args['save'] = save_cfg
    args['emb_dir'] = embs_path
    controlnet = args.pop('controlnet')
    condition = args.pop('condition')
    lora = args.pop('lora')
    if lora:
        unet = copy.deepcopy(lora_unet)
        unet['lora'] = [
            {
                "path": os.path.join(lora_model_path, i.name),
                "alpha": i.alpha,
                "layers": i.layers
            } for i in lora if i.type == 'unet'
        ]
        te = copy.deepcopy(lora_te)
        te['lora'] = [
            {
                "path": os.path.join(lora_model_path, i.name),
                "alpha": i.alpha,
                "layers": i.layers
            } for i in lora if i.type == 'TE'
        ]
        if unet['lora']:
            merge['unet_lora'] = unet
        if te['lora']:
            merge['te_lora'] = te

    if condition:
        init_image = f"{os.path.join(init_image_path, task_id)}_init.jpg"
        base64_to_image(condition.image, init_image)
        condition.image = init_image
        args['condition'] = dict(condition)
    if controlnet:
        cnet = copy.deepcopy(cnet_merge)
        ex = copy.deepcopy(ex_input)
        controlnet_image = f"{os.path.join(init_image_path, task_id)}_controlnet.jpg"
        base64_to_image(controlnet.image, controlnet_image)
        ex['cond']['image'] = controlnet_image
        cnet['group1']['plugin']['controlnet1']['path'] = os.path.join(controlnet_model_path, controlnet.model)
        data['ex_input'] = ex
        merge['controlnet'] = cnet
    if merge:
        data['merge'] = merge
    data.update(args)
    res = engine.add_task(data)
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
