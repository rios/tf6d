# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .patch_embed import get_patch_embed
import einops as E
from models.croco import CroCoNet  # noqa
from evals.models.utils import fill_default_args, freeze_all_params, tokens_to_output, center_padding

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse(
    "0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo(
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 output="dense",
                 layer=-1,
                 return_multilayer=False,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        # self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_freeze(freeze)
        self.checkpoint_name = "dust3r"
        self.patch_size = self.patch_embed.patch_size[0]
        self.output = output

        num_layers = len(self.enc_blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            # self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            # self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def forward(self, img):
        # encode the two images --> B,S,D
        img = center_padding(img, self.patch_size)
        img_h, img_w = img.shape[-2:]
        out_h, out_w = img_h // self.patch_embed.patch_size[1], img_w // self.patch_embed.patch_size[0]

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape = img.shape[1:]
        # warning! maybe the images have different portrait/landscape orientations
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(img, true_shape=img.shape[-2:])

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        embeds = []
        for i, blk in enumerate(self.enc_blocks):
            x = blk(x, pos)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        # embeds[-1] = self.enc_norm(embeds[-1])

        # if return_all_blocks:
        #     out = []
        #     for blk in self.enc_blocks:
        #         x = blk(x, posvis)
        #         out.append(x)
        #     out[-1] = self.enc_norm(out[-1])
        #     return out, pos, masks
        # else:
        #     for blk in self.enc_blocks:
        #         x = blk(x, posvis)
        #     x = self.enc_norm(x)
        #     return x, pos, masks

        # now apply the transformer encoder and normalization
        # for blk in self.enc_blocks:
        #     x = blk(x, pos)

        # x = self.enc_norm(x)
        # dense_tokens = E.rearrange(x, "b (h w) c -> b c h w", h=out_h, w=out_w)
        # output = dense_tokens.contiguous()

        # return output

        outputs = []
        for i, x_i in enumerate(embeds):
            # ignoring register tokens
            x_i = tokens_to_output(self.output, x_i, None, (out_h, out_w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
