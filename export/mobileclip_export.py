import json
import os

import open_clip
import torch
import torch.nn.utils as nn_utils
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch import nn

from mobileclip.modules.common.mobileone import reparameterize_model


def export_mobileclip_to_xnnpack():
    model_name = "MobileCLIP-S2"
    model_clip, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='datacompdr')
    image_encoder = model_clip.visual
    text_encoder = model_clip.text

    with open(f"../mobileclip/configs/{model_name.lower().replace('-', '_')}.json", "r") as f:
        configs = json.load(f)

    # batch_size=1 because processing 1 image at time, channels=3 because RGB has 3 colors channels
    batch_size = 1
    rgb_channels = 3
    image_size = configs["image_cfg"]["image_size"]
    image_inputs = (torch.randn(batch_size, rgb_channels, image_size, image_size),)

    vocab_size = configs["text_cfg"]["vocab_size"]
    context_length = configs["text_cfg"]["context_length"]
    text_inputs = (torch.randint(0, vocab_size, (batch_size, context_length)),)

    export_model(image_encoder, image_inputs, f"{model_name.lower()}_visual")
    export_model(text_encoder, text_inputs, f"{model_name.lower()}_text")


def export_model(model, sample_input, model_name):
    model.eval()
    # Reparameterize to fuse BNs in MobileOne blocks
    model = reparameterize_model(model)
    # All of these needed because XNNPACK export doesn't support BatchNorm
    model = fuse_conv_norm(model)
    # Fuse any remaining sequential Conv + BN
    model = fuse_all_conv_bn(model)
    # Replace standalone BNs with depthwise Convs
    model = replace_standalone_bn(model)

    print("Exporting model...")
    model_exp = torch.export.export(model, sample_input)

    edge = to_edge_transform_and_lower(
        model_exp,
        partitioner=[XnnpackPartitioner(verbose=True)],
        # compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )
    print("Converting to ExecuTorch...")
    exec_prog = edge.to_executorch()

    output_folder = "models"
    os.makedirs(output_folder, exist_ok=True)
    with open(f"{output_folder}/{model_name}.pte", "wb") as f:
        exec_prog.write_to_file(f)

    print(f"✓ Model exported")


def fuse_conv_norm(model):
    """
    Recursively fuse ConvNormAct modules by fusing their conv and bn (BatchNormAct2d).
    BatchNormAct2d is a subclass of BatchNorm2d, so treat it directly as the BN layer.
    """
    for name, child in model.named_children():
        if 'ConvNormAct' in type(child).__name__:  # Check by name since class not imported
            conv = child.conv
            bn_module = child.bn
            if (isinstance(conv, nn.Conv2d) and
                    isinstance(bn_module, nn.BatchNorm2d) and
                    isinstance(bn_module.act, nn.Identity) and
                    isinstance(bn_module.drop, nn.Identity)):
                fused_conv = nn_utils.fusion.fuse_conv_bn_eval(conv, bn_module)
                setattr(model, name, fused_conv)
        else:
            fuse_conv_norm(child)
    return model


def fuse_all_conv_bn(model):
    """
    Recursively fuse all Conv2d + BatchNorm2d pairs in the model.
    This traverses submodules and handles Sequential containers.
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Sequential):
            i = 0
            while i < len(module) - 1:
                if isinstance(module[i], torch.nn.Conv2d) and isinstance(module[i + 1], torch.nn.BatchNorm2d):
                    fused_conv = nn_utils.fusion.fuse_conv_bn_eval(module[i], module[i + 1])
                    module[i] = fused_conv
                    module[i + 1] = torch.nn.Identity()  # Replace BN with Identity to remove it
                    i += 1  # Skip the Identity
                else:
                    i += 1
        # Recurse into submodules if they have children
        if len(list(module.children())) > 0:
            fuse_all_conv_bn(module)
    return model


def replace_standalone_bn(model):
    """
    Recursively replace standalone BatchNorm2d or BatchNormAct2d with depthwise 1x1 Conv2d.
    This applies the fixed affine transform from eval-mode BN stats.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            bn = child
            if hasattr(bn, 'act') and not isinstance(bn.act, nn.Identity):
                continue  # Skip if act is not Identity
            if hasattr(bn, 'drop') and not isinstance(bn.drop, nn.Identity):
                continue  # Skip if drop is not Identity
            with torch.no_grad():
                scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                bias = bn.bias - bn.running_mean * scale
                num_channels = bn.num_features
                conv = nn.Conv2d(num_channels, num_channels, kernel_size=1, groups=num_channels, bias=True)
                conv.weight.data = scale.view(-1, 1, 1, 1)
                conv.bias.data = bias
            setattr(model, name, conv)
        if len(list(child.children())) > 0:
            replace_standalone_bn(child)
    return model


if __name__ == "__main__":
    export_mobileclip_to_xnnpack()
