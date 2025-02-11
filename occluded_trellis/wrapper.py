from types import MethodType
from typing import *
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import rembg


def _mask2patches_mask(mask: np.ndarray,
                       patch_size: int,
                       general_tokens_length: int = 5) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Patchify original occlusion mask
    :param mask: np.ndarray of shape HxW, binary, 0 for occluded zone, 1 in other zones
    :param patch_size: patch size of DinoV2
    :param general_tokens_length: length of general tokens which we never mask (class and registers), 5 for DinoV2
    (1 class + 4 registers)
    :return: tuple with:
        boolean flat mask for DinoV2 tokens
        HxW patchified binary mask [0, 1]
    """
    assert list(sorted(np.unique(mask))) == [0, 1], "expected binary mask as input"
    mask_pt = torch.from_numpy(mask).view(1, 1, mask.shape[0], mask.shape[1])
    pooling = torch.nn.MaxPool2d(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    pooled = 1 - pooling(1 - mask_pt)
    patches_mask = pooled.squeeze().view(-1).numpy()
    cls_registers_mask = np.ones(general_tokens_length)
    tokens_mask = np.concatenate([cls_registers_mask, patches_mask]) > 0.5
    mask_resized = np.kron(pooled.squeeze().numpy(), np.ones((patch_size, patch_size)))
    return tokens_mask, mask_resized


def _prepare_tokens_with_masks(self, x, masks=None):
    """
    This method redefines tokenization of the image for DinoV2.
    Instead of masking tokens as in the original code we just drop them.
    """
    B, nc, w, h = x.shape
    x = self.patch_embed(x)

    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.interpolate_pos_encoding(x, w, h)

    if self.register_tokens is not None:
        x = torch.cat(
            (
                x[:, :1],
                self.register_tokens.expand(x.shape[0], -1, -1),
                x[:, 1:],
            ),
            dim=1,
        )

    if masks is not None:
        x = x[:, masks]
    return x


def _preprocess_image(self, input: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    Preprocess the input image and mask.
    """
    # if has alpha channel, use it directly; otherwise, remove background
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if getattr(self, 'rembg_session', None) is None:
            self.rembg_session = rembg.new_session('u2net')
        output = rembg.remove(input, session=self.rembg_session)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    mask = mask.crop(bbox)
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    mask = mask.resize((518, 518), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output, mask


@torch.no_grad()
def _encode_image(self, image: Image.Image, mask: Image.Image) -> torch.Tensor:
    """
    Encode the image.

    Args:
        image (Union[torch.Tensor, list[Image.Image]]): The image to encode

    Returns:
        torch.Tensor: The encoded features.
    """
    mask = np.array(mask)
    mask = (cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 128).astype(np.float32)
    tokens_mask, mask_patched = _mask2patches_mask(mask, patch_size=14)

    image = np.array(image.convert('RGB')).astype(np.float32) / 255
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

    image = self.image_cond_model_transform(image).to(self.device)
    features = self.models['image_cond_model'](image, is_training=True, masks=tokens_mask)['x_prenorm']
    patchtokens = F.layer_norm(features, features.shape[-1:])
    return patchtokens


def _get_cond(self, image: Image.Image, mask: Image.Image) -> dict:
    """
    Get the conditioning information for the model.

    Args:
        image Image.Image: The image prompt.
        mask Image.Image: The mask for image

    Returns:
        dict: The conditioning information
    """
    cond = self.encode_image(image, mask)
    neg_cond = torch.zeros_like(cond)
    return {
        'cond': cond,
        'neg_cond': neg_cond,
    }


@torch.no_grad()
def _run(
        self,
        image: Image.Image,
        mask: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
) -> dict:
    """
    Run the pipeline.

    Args:
        image (Image.Image): The image prompt.
        num_samples (int): The number of samples to generate.
        sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
        slat_sampler_params (dict): Additional parameters for the structured latent sampler.
        preprocess_image (bool): Whether to preprocess the image.
    """
    if preprocess_image:
        image, mask = self.preprocess_image(image, mask)
    cond = self.get_cond(image, mask)
    torch.manual_seed(seed)
    coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
    slat = self.sample_slat(cond, coords, slat_sampler_params)
    return self.decode_slat(slat, formats)


@torch.no_grad()
def _run_multi_image(self, *args, **kwargs):
    raise NotImplementedError("Multi image is not implemented for Occluded version of Trellis")


def apply_occluded_patch(pipe: "TrellisImageTo3DPipeline") -> "TrellisImageTo3DPipeline":
    cond_model = pipe.models["image_cond_model"]
    cond_model.prepare_tokens_with_masks = MethodType(_prepare_tokens_with_masks, cond_model)
    pipe.preprocess_image = MethodType(_preprocess_image, pipe)
    pipe.get_cond = MethodType(_get_cond, pipe)
    pipe.encode_image = MethodType(_encode_image, pipe)
    pipe.run = MethodType(_run, pipe)
    pipe.run_multi_image = MethodType(_run_multi_image, pipe)
    return pipe
