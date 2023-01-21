from __future__ import division

import torch
import torch.nn.functional as F

# general libs
from PIL import Image
import numpy as np
import cv2


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array


def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda()
    else:
        return xs


def To_onehot(mask, num_objects):
    S = np.zeros((1, num_objects + 1, mask.shape[0], mask.shape[1]))
    for o in range(num_objects + 1):
        S[0, o] = (mask == o).astype(np.float32)
    return S


def Dilate_scribble(mask, num_objects):
    new_mask = np.zeros_like(mask)
    for o in range(num_objects + 1):  # include bg scribbles
        bmask = (mask[0, o] > 0.5).astype(np.uint8)
        new_mask[0, o] = cv2.dilate(
            bmask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=2
        )
    return new_mask


def load_frames(path, number_of_frames, size=None):
    frame_list = []
    for i in range(number_of_frames):
        if size:
            frame_list.append(
                np.array(
                    Image.open(f"{path}/{str(i).zfill(5)}.png")
                    .convert("RGB")
                    .resize((size[0], size[1]), Image.BICUBIC),
                    dtype=np.uint8,
                )
            )
    frames = np.stack(frame_list, axis=0)
    return frames


def load_UnDP(path):
    # load dataparallel wrapped model properly
    state_dict = torch.load(path, map_location="cpu")
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def overlay_davis(image, mask, rgb=[255, 0, 0], cscale=2, alpha=0.5):
    """Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    im_overlay = image.copy()

    foreground = (
        im_overlay * alpha
        + np.ones(im_overlay.shape)
        * (1 - alpha)
        * np.array(rgb, dtype=np.uint8)[None, None, :]
    )
    binary_mask = mask == 1
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    countours = binary_dilation(binary_mask) ^ binary_mask
    im_overlay[countours, :] = 0
    return im_overlay.astype(image.dtype)
