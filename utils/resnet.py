# Adapted from https://github.com/pytorch/vision/blob/v0.11.0/torchvision/models/resnet.py

import torch
from torchvision import transforms
from typing import Type, Union, List, Any
from torchvision.models.resnet import _resnet, BasicBlock, Bottleneck, ResNet
import torchvision
import torch.nn as nn
from typing import Callable

def load_resnet50(pretrained: bool = False):
    backbone = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=pretrained, progress=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return backbone, normalize


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNetFeatures(block, layers, **kwargs)
    if pretrained:
        if int(torch.__version__[0]) <= 1:
            from torch.hub import load_state_dict_from_url
            from torchvision.models.resnet import model_urls
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)
        else:
            raise NotImplementedError("Pretrained models not supported in PyTorch 2.0+")
    return model


class ResNetFeatures(ResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)

    def _forward_impl(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }


def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module