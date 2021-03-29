__all__ = [
    "resnet50_1x8",
    "resnet50_8x8",
    "resnet101_1x8",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *

MMDetYolactBackboneConfig = partial(MMDetBackboneConfig, model_name="yolact")
base_config_path = mmdet_configs_path / "yolact"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/yolact"

resnet50_1x8 = MMDetYolactBackboneConfig(
    config_path=base_config_path / "yolact_r50_1x8_coco.py",
    weights_url=f"{base_weights_url}/yolact_r50_1x8_coco_20200908-f38d58df.pth",
)

resnet50_8x8 = MMDetYolactBackboneConfig(
    config_path=base_config_path / "yolact_r50_8x8_coco.py",
    weights_url=f"{base_weights_url}/yolact_r50_8x8_coco_20200908-ca34f5db.pth",
)

resnet101_1x8 = MMDetYolactBackboneConfig(
    config_path=base_config_path / "yolact_r101_1x8_coco.py",
    weights_url=f"{base_weights_url}/yolact_r101_1x8_coco_20200908-4cbe9101.pth",
)
