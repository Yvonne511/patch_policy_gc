import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class resnet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        output_dim: int = 512,  # fixed for resnet18; included for consistency with config
        unit_norm: bool = False,
        n_patches: int = 1,  # fixed for resnet18; included for consistency with config
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.pretrained = pretrained
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unit_norm = unit_norm

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            # flatten all dimensions to batch, then reshape back at the end
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.normalize(x)
        out = self.resnet(x)
        out = self.flatten(out)
        if self.unit_norm:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        # add a patch dim
        out = out.unsqueeze(-2)
        return out


class Resnet18Patches(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        output_dim: int = 256,
        unit_norm: bool = False,
        ckpt_path: Optional[str] = None,
        return_layers: tuple[str] = ("layer3",),
        n_patches: Optional[int] = None,
    ):
        super().__init__()
        # We need to construct the full backbone first to extract features
        base_model = torchvision.models.resnet18(pretrained=pretrained)
        
        # Map requested layers to the node names expected by create_feature_extractor.
        # ResNet18 structure:
        # conv1, bn1, relu, maxpool
        # layer1
        # layer2
        # layer3
        # layer4
        # We assume return_layers contains strings like "layer1", "layer2", etc.
        self.return_layers = list(return_layers)
        
        # create_feature_extractor returns a dict of {node_name: output}
        from torchvision.models.feature_extraction import create_feature_extractor
        self.backbone = create_feature_extractor(base_model, return_nodes=self.return_layers)

        self.pretrained = pretrained
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unit_norm = unit_norm
        self.output_dim = output_dim
        self.latent_ndim = 2 # Assuming 2D feature maps from CNN
        
        # Determine number of patches and setup projections
        self.n_patches = 0
        self.projections = nn.ModuleDict()
        
        # We need to know the channel counts for each layer to set up projections.
        # Run a dummy forward pass to get shapes.
        dummy_input = torch.zeros(1, 3, 224, 224)
        # We can't rely on self.normalize yet? or we can. 
        # Actually create_feature_extractor copies the layers, it doesn't run forward.
        # But we need to know output channels.
        with torch.no_grad():
            features = self.backbone(dummy_input)
             
        for layer_name in self.return_layers:
            feat = features[layer_name]
            # feat shape: [B, C, H, W]
            c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
            self.n_patches += h * w
            
            if c != self.output_dim:
                self.projections[layer_name] = nn.Conv2d(c, self.output_dim, kernel_size=1)
            else:
                self.projections[layer_name] = nn.Identity()

        assert n_patches == self.n_patches

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.load_from_resnet18_state_dict(ckpt.state_dict())

    def load_from_resnet18_state_dict(self, state_dict, strict: bool = False):
        """
        Load weights from the original global-pooling resnet18 encoder checkpoints.
        Updates to map legacy sequential keys to the new named module structure.
        """
        # Original Sequential structure:
        # 0: conv1
        # 1: bn1
        # 2: relu
        # 3: maxpool
        # 4: layer1
        # 5: layer2
        # 6: layer3
        # 7: layer4 (often not used in the old Resnet18Patches but present in full resnet)
        
        idx_map = {
            "0": "conv1",
            "1": "bn1",
            # 2 (relu) and 3 (maxpool) are stateless usually
            "4": "layer1",
            "5": "layer2",
            "6": "layer3",
            "7": "layer4"
        }
        
        new_state = {}
        for k, v in state_dict.items():
            # Old keys might look like "module.resnet.0.weight" or "resnet.4.0.conv1.weight"
            if k.startswith("module.resnet."):
                k = k[len("module.") :]
            if not k.startswith("resnet."):
                continue
            
            # k is now like "resnet.0.weight" or "resnet.6.0.conv1.weight"
            parts = k.split(".")
            # parts[0] is "resnet"
            seq_idx = parts[1]
            
            if seq_idx not in idx_map:
                # e.g. avgpool layer if it existed in that checkpoint?
                continue
                
            module_name = idx_map[seq_idx]
            
            # Construct new key: "backbone.{module_name}.{rest}"
            # e.g. "backbone.conv1.weight" or "backbone.layer3.0.conv1.weight"
            new_key_parts = ["backbone", module_name] + parts[2:]
            new_key = ".".join(new_key_parts)
            new_state[new_key] = v

        missing, unexpected = self.load_state_dict(new_state, strict=strict)
        return missing, unexpected

    def forward(self, x):
        dims = len(x.shape)
        squeeze_batch = dims == 3
        if squeeze_batch:
            x = x.unsqueeze(0)

        leading_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        x = self.normalize(x)
        
        # features is a dict: {layer_name: tensor}
        features = self.backbone(x)
        
        token_list = []
        
        for layer_name in self.return_layers:
            feat = features[layer_name] # (B_flat, C_layer, H, W)
            
            # Project to output_dim
            proj = self.projections[layer_name]
            feat = proj(feat) # (B_flat, output_dim, H, W)
            
            # Flatten to tokens
            tokens = feat.flatten(2).transpose(1, 2) # (B_flat, H*W, output_dim)
            token_list.append(tokens)
            
        # Concatenate tokens
        out = torch.cat(token_list, dim=1) # (B_flat, total_patches, output_dim)

        if self.unit_norm:
            out = F.normalize(out, p=2, dim=-1)

        out = out.reshape(*leading_shape, out.shape[1], out.shape[2])
        if squeeze_batch:
            out = out.squeeze(0)
        return out
