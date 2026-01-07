import torch
import einops
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key, output_dim=None, postprocess=None):
        super().__init__()
        print("Encoder feature_key:", feature_key)
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2:b48308a", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        self.output_dim = self.emb_dim # for compatibility
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

        # TODO: sanity check
        self.postprocess = postprocess
        if postprocess is not None:
            if postprocess == 'avg_pool':
                self.latent_ndim = 1

    def forward(self, x):
        b, v, c, h, w = x.shape
        x = x.reshape(b * v, c, h, w)

        emb = self.base_model.forward_features(x)[self.feature_key]
        emb = einops.rearrange(emb, '(b v) ... -> b v ...', b=b) # b v p e

        if self.postprocess == 'avg_pool':
            emb = torch.mean(emb, dim=(2)) # (b, v, e)

        if self.latent_ndim == 1:
            emb = emb.unsqueeze(2) # dummy patch dim, b v 1 e
        return emb