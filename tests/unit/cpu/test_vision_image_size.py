import pytest
import torch
from transformers import DetrImageProcessor, GroundingDinoConfig, SwinConfig, ViTImageProcessor
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoConvEncoder

from optimum.rbln.transformers.models.detr.configuration_detr import RBLNDetrForObjectDetectionConfig
from optimum.rbln.transformers.models.detr.modeling_detr import RBLNDetrForObjectDetection
from optimum.rbln.transformers.models.grounding_dino.configuration_grounding_dino import (
    RBLNGroundingDinoEncoderConfig,
)
from optimum.rbln.transformers.models.grounding_dino.modeling_grounding_dino import _update_spatial_shapes
from optimum.rbln.transformers.models.swin.configuration_swin import RBLNSwinBackboneConfig
from optimum.rbln.transformers.models.swin.modeling_swin import RBLNSwinBackbone


def test_detr_takes_the_image_size_from_the_image_processor():
    rbln_config = RBLNDetrForObjectDetection._update_rbln_config(
        preprocessors=[DetrImageProcessor()], rbln_config=RBLNDetrForObjectDetectionConfig()
    )

    assert rbln_config.image_size == (1333, 1333)


def test_swin_takes_the_image_size_from_the_image_processor():
    rbln_config = RBLNSwinBackbone._update_rbln_config(
        preprocessors=[ViTImageProcessor(size={"height": 224, "width": 384})], rbln_config=RBLNSwinBackboneConfig()
    )

    assert rbln_config.image_size == (224, 384)


@pytest.mark.parametrize("image_size", [(480, 640), (333, 500)])
def test_grounding_dino_spatial_shapes_match_the_backbone(image_size):
    backbone_config = SwinConfig(
        embed_dim=12,
        depths=[1, 1, 2, 1],
        num_heads=[1, 2, 3, 4],
        patch_size=4,
        window_size=7,
        out_features=["stage2", "stage3", "stage4"],
    )
    config = GroundingDinoConfig(backbone_config=backbone_config, num_feature_levels=4)
    rbln_config = _update_spatial_shapes(config, RBLNGroundingDinoEncoderConfig(image_size=image_size))

    with torch.no_grad():
        features = GroundingDinoConvEncoder(config)(torch.zeros(1, 3, *image_size), torch.ones(1, *image_size))
        expected = [list(feature_map.shape[-2:]) for feature_map, _ in features]
        # The extra level comes from a 3x3 stride-2 convolution on the last feature map.
        extra = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)(torch.zeros(1, 1, *expected[-1]))
        expected.append(list(extra.shape[-2:]))

    assert rbln_config.spatial_shapes_list == expected
