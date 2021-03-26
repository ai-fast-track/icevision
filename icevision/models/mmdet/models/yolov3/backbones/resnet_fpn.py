__all__ = [
    "yolov3_d53_320_273e",
    "yolov3_d53_mstrain_416_273e",
    "yolov3_d53_mstrain_608_273e",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *

MMDetYolov3BackboneConfig = partial(MMDetBackboneConfig, model_name="yolov3")
base_config_path = mmdet_configs_path / "yolo"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/retinanet"

yolov3_d53_320_273e = MMDetYolov3BackboneConfig(
    config_path=base_config_path / "yolov3_d53_320_273e_coco.py",
    weights_url=f"{base_weights_url}/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth",
)

yolov3_d53_mstrain_416_273e = MMDetYolov3BackboneConfig(
    config_path=base_config_path / "yolov3_d53_mstrain-416_273e_coco.py",
    weights_url=f"{base_weights_url}/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth",
)

yolov3_d53_mstrain_608_273e = MMDetYolov3BackboneConfig(
    config_path=base_config_path / "yolov3_d53_mstrain-608_273e_coco.py",
    weights_url=f"{base_weights_url}/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-139f5633.pth",
)
