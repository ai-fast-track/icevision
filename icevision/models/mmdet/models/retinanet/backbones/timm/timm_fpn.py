__all__ = [
    "mobilenetv3_large_100",
]

import icevision
from icevision.imports import *
from icevision.models.mmdet.utils import *
from icevision.models.mmdet.backbones.timm import *

base_config_path = Path(icevision.__file__).parent / "models/mmdet/models/retinanet"


class MMDetTimmRetinanetBackboneConfig(MMDetTimmBackboneConfig):
    def __init__(self, backbone_dict, **kwargs):
        super().__init__(
            model_name="retinanet",
            config_path=base_config_path / "retinanet_no_backbone_fpn.py",
            backbone_dict=backbone_dict,
            **kwargs
        )


mobilenetv3_large_100 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "MobileNetv3Large100",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)
