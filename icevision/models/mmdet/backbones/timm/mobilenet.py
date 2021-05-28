__all__ = [
    "BaseMobileNetV3",
    "MobileNetV3_Large_100",   
]

from icevision.models.mmdet.backbones.timm.common import *
from mmdet.models.builder import BACKBONES

from typing import Optional, Collection
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Tuple, Collection, List

import timm
from timm.models.mobilenetv3 import *
from timm.models.registry import *

import torch.nn as nn
import torch


class BaseMobileNetV3(MMDetTimmBase):
    """
    Base class that implements model freezing and forward methods
    """

    def __init__(
        self,
        model_name: str = None,
        pretrained: bool = True,  # doesn't matter
        out_indices: Collection[int] = (1, 2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):

        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
        )
        self.frozen_stages = frozen_stages
        self.frozen_stem = frozen_stem

    def test_pretrained_weights(self):
        # Get model method from the timm registry by model name
        model_fn = model_entrypoint(self.model_name)
        model = model_fn(pretrained=self.pretrained)
        assert torch.equal(self.model.conv_stem.weight, model.conv_stem.weight)

    def post_init_setup(self):
        self.freeze(
            freeze_stem=self.frozen_stem,
            freeze_blocks=self.frozen_stages,
        )
        # self.test_pretrained_weights()

    def freeze(self, freeze_stem: bool = True, freeze_blocks: int = 1):
        "Optionally freeze the stem and/or Inverted Residual blocks of the model"
        assert 0 <= freeze_blocks <= 7, f"Can freeze 0-7 blocks only"
        m = self.model

        # Stem freezing logic
        if freeze_stem:
            for l in [m.conv_stem, m.bn1]:
                l.eval()
                for param in l.parameters():
                    param.requires_grad = False

        # `freeze_blocks=1` freezes the first block, and so on
        for i, block in enumerate(m.blocks, start=1):
            if i > freeze_blocks:
                break
            else:
                block.eval()
                for param in block.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        "Convert the model to training mode while optionally freezing BatchNorm"
        super(BaseMobileNetV3, self).train(mode)
        self.freeze(
            freeze_stem=self.frozen_stem,
            freeze_blocks=self.frozen_stages,
        )
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


# @BACKBONES.register_module(force=True)
# class MobileNetV2_100(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_mobilenetv2_100(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )

# @BACKBONES.register_module(force=True)
# class MobileNetV2_110D(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_mobilenetv2_110d(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )

# @BACKBONES.register_module(force=True)
# class MobileNetV2_120D(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_mobilenetv2_120d(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )

# @BACKBONES.register_module(force=True)
# class MobileNetV2_140(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_mobilenetv2_140(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )


# @BACKBONES.register_module(force=True)
# class MobileNetV3_Large_075(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_mobilenetv3_large_075(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )


@BACKBONES.register_module(force=True)
class MobileNetV3_Large_100(BaseMobileNetV3):
    def __init__(
        self,
        pretrained: bool = True,  # doesn't matter
        out_indices: Collection[int] = (1, 2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "MobileNetV3 Large with hardcoded `pretrained=True`"
        super().__init__(
            model_name="mobilenetv3_large_100",
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
            frozen_stages=frozen_stages,
            frozen_stem=frozen_stem,
        )
        self.model_name = "mobilenetv3_large_100"
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.frozen_stem = frozen_stem
        self.model = mobilenetv3_large_100(
            pretrained=True, features_only=True, out_indices=out_indices
        )
        self.post_init_setup()


# @BACKBONES.register_module(force=True)
# class MobileNetV3_RW(BaseMobileNetV3):
#     def __init__(
#         self,
#         pretrained: bool = True,  # doesn't matter
#         out_indices: Collection[int] = (1, 2, 3, 4),
#         norm_eval: bool = True,
#         frozen_stages: int = 1,
#         frozen_stem: bool = True,
#     ):
#         "register_module Large with hardcoded `pretrained=True`"
#         super().__init__(
#             model_name="mobilenetv3_rw",
#             pretrained=pretrained,
#             out_indices=out_indices,
#             norm_eval=norm_eval,
#             frozen_stages=frozen_stages,
#             frozen_stem=frozen_stem,
#         )
#         self.model_name = "mobilenetv3_rw"
#         self.norm_eval = norm_eval
#         self.frozen_stages = frozen_stages
#         self.frozen_stem = frozen_stem
#         self.model = mobilenetv3_rw(
#             pretrained=True, features_only=True, out_indices=out_indices
#         )
#         self.post_init_setup()


# @BACKBONES.register_module(force=True)
# class MobileNetV3_Small_075(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_mobilenetv3_small_075(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )


# @BACKBONES.register_module(force=True)
# class MobileNetV3_Small_100(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_mobilenetv3_small_100(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )


# @BACKBONES.register_module(force=True)
# class TF_MobileNetV3_Large_075(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_tf_mobilenetv3_large_075(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )


# @BACKBONES.register_module(force=True)
# class TF_MobileNetV3_Large_100(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_tf_mobilenetv3_large_100(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )


# @BACKBONES.register_module(force=True)
# class TF_MobileNetV3_Large_Minimal_100(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_tf_mobilenetv3_large_minimal_100(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )


# @BACKBONES.register_module(force=True)
# class TF_MobileNetV3_Small_075(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_tf_mobilenetv3_small_075(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )


# @BACKBONES.register_module(force=True)
# class TF_MobileNetV3_Small_100(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_tf_mobilenetv3_small_100(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )


# @BACKBONES.register_module(force=True)
# class TF_MobileNetV3_Small_Minimal_100(MMDetTimmBase):
#     def __init__(self, pretrained=True, out_indices=(1, 2, 3, 4), **kwargs):
#         super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
#         self.model = ice_tf_mobilenetv3_small_minimal_100(
#             pretrained=pretrained,
#             out_indices=out_indices,
#             **kwargs,
#         )