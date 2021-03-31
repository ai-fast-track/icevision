__all__ = [
    "resnet50_caffe_1x",
    "resnet50_caffe_mstrain_1x",
    "resnet50_caffe_mstrain_3x",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetTridentNetBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="tridentnet", **kwargs)


base_config_path = mmdet_configs_path / "tridentnet"
base_weights_url = "https://download.openmmlab.com/mmdetection/v2.0/tridentnet"

resnet50_caffe_1x = MMDetTridentNetBackboneConfig(
    config_path=base_config_path / "tridentnet_r50_caffe_1x_coco.py",
    weights_url=f"{base_weights_url}/tridentnet_r50_caffe_1x_coco/tridentnet_r50_caffe_1x_coco_20201230_141838-2ec0b530.pth",
)

resnet50_caffe_mstrain_1x = MMDetTridentNetBackboneConfig(
    config_path=base_config_path / "tridentnet_r50_caffe_mstrain_1x_coco.py",
    weights_url=f"{base_weights_url}/tridentnet_r50_caffe_mstrain_1x_coco/tridentnet_r50_caffe_mstrain_1x_coco_20201230_141839-6ce55ccb.pth",
)

resnet50_caffe_mstrain_3x = MMDetTridentNetBackboneConfig(
    config_path=base_config_path / "tridentnet_r50_caffe_mstrain_3x_coco.py",
    weights_url=f"{base_weights_url}/tridentnet_r50_caffe_mstrain_3x_coco/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539-46d227ba.pth",
)
