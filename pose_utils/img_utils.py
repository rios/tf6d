import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Union, Tuple
from PIL import Image, ImageOps


def make_quadratic_crop_ratio(image, bbox, patch_size=14, final_ratio=0.6):
    # Define the bounding box
    x_left, y_top, width, height = bbox

    # Calculate the longer side of the bounding box
    longer_side = max(width, height)

    # Calculate the final crop size based on the longer side and the final_ratio (0.6)
    crop_size = int(longer_side / final_ratio)

    # Ensure crop_size is divisible by patch_size (14 in this case)
    if crop_size % patch_size != 0:
        crop_size = (crop_size // patch_size) * patch_size + patch_size  # Round up to nearest multiple of patch_size

    # Calculate the center of the bounding box
    center_x = x_left + width / 2
    center_y = y_top + height / 2

    # Calculate the coordinates of the top-left corner of the square crop
    crop_x = int(center_x - crop_size / 2)
    crop_y = int(center_y - crop_size / 2)

    # Check if the crop goes beyond the image boundaries
    if crop_x < 0 or crop_y < 0 or crop_x + crop_size > image.shape[1] or crop_y + crop_size > image.shape[0]:

        # If the crop goes beyond the image boundaries, crop first and add a border using cv2.copyMakeBorder to make the crop quadratic
        crop = image[max(crop_y, 0):min(crop_y + crop_size, image.shape[0]),
               max(crop_x, 0):min(crop_x + crop_size, image.shape[1])]
        border_size = max(crop_size - crop.shape[1], crop_size - crop.shape[0])
        border_size = max(0, border_size)  # Make sure the border size is not negative

        if crop_x < 0 or crop_x + crop_size > image.shape[1]:
            left = border_size // 2
            right = border_size - left
            crop = cv2.copyMakeBorder(crop, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif crop_y < 0 or crop_y + crop_size > image.shape[0]:
            top = border_size // 2
            bottom = border_size - top
            crop = cv2.copyMakeBorder(crop, top, bottom, 0, 0,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            print("Something went wrong during rectifying crop")
            return None

    else:
        # If the crop is within the image boundaries, just crop the image
        crop = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    return crop, crop_y, crop_x



def make_quadratic_crop(image, bbox, patch_size=14):
    # Define the bounding box
    x_left, y_top, width, height = bbox

    # Calculate the size of the square crop based on the longer side
    longer_side = max(width, height)

    # Calculate the center of the bounding box
    center_x = x_left + width / 2
    center_y = y_top + height / 2
    crop_size = min(longer_side, int(max(width / 2, height / 2) * 2))

    # Ensure crop_size is divisible by 14
    if crop_size % patch_size != 0:
        crop_size = (crop_size // patch_size) * patch_size + patch_size  # Round up to the nearest multiple of 14

    # Calculate the coordinates of the top-left corner of the square crop
    crop_x = int(center_x - crop_size / 2)
    crop_y = int(center_y - crop_size / 2)

    # Check if the crop goes beyond the image boundaries
    if crop_x < 0 or crop_y < 0 or crop_x + crop_size > image.shape[1] or crop_y + crop_size > image.shape[0]:

        # If the crop goes beyond the image boundaries, crop first and add a border using cv2.copyMakeBorder to make the crop quadratic
        crop = image[max(crop_y, 0):min(crop_y + crop_size, image.shape[0]),
               max(crop_x, 0):min(crop_x + crop_size, image.shape[1])]
        border_size = max(crop_size - crop.shape[1], crop_size - crop.shape[0])
        border_size = max(0, border_size)  # Make sure the border size is not negative

        if crop_x < 0 or crop_x + crop_size > image.shape[1]:
            left = border_size // 2
            right = border_size - left
            crop = cv2.copyMakeBorder(crop, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # crop = cv2.copyMakeBorder(crop, 0, 0, left, right, cv2.BORDER_REPLICATE)
        elif crop_y < 0 or crop_y + crop_size > image.shape[0]:
            top = border_size // 2
            bottom = border_size - top
            crop = cv2.copyMakeBorder(crop, top, bottom, 0, 0,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # crop = cv2.copyMakeBorder(crop, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
        else:
            print("Something went wrong during rectifying crop")
            return None

    else:
        # If the crop is within the image boundaries, just crop the image
        crop = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    return crop, crop_y, crop_x


def rle_to_mask(rle):
    """Converts a COCOs run-length encoding (RLE) to 3 channel mask with [0,255]

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')

    start = 0
    for i in range(len(counts) - 1):
        start += counts[i]
        end = start + counts[i + 1]
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get('size'), order='F')

    # First, convert True to 255 and False to 0.
    mask = binary_mask * 255

    # # Then, convert the mask to 3-channel.
    # mask_3c = np.dstack([mask]*3)

    return mask


def get_bounding_box_from_mask(mask):
    # Convert to binary mask (0 and 1) if it is not
    mask_binary = np.where(mask > 0, 1, 0)

    # Find min and max rows and columns with a value of 1
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # return top-left and bottom-right corners and width, height
    x_left = cmin
    y_upper = rmin
    w = cmax - cmin + 1
    h = rmax - rmin + 1

    return [x_left, y_upper, w, h]


def warp_to_virtual_camera(image, bbox, S=420, delta=0.6):
    # Define the bounding box (x, y, width, height)
    x_left, y_top, width, height = bbox

    # Calculate the center of the bounding box
    center_x = x_left + width / 2
    center_y = y_top + height / 2

    # Calculate the scaling factor to make the longer side delta*S pixels long
    longer_side = max(width, height)
    scale_factor = (delta * S) / longer_side

    # Compute the transformation matrix (scaling and translation to center)
    M = np.array([
        [scale_factor, 0, S / 2 - scale_factor * center_x],
        [0, scale_factor, S / 2 - scale_factor * center_y]
    ])

    # Warp the image to the new virtual camera view
    warped_image = cv2.warpAffine(image, M, (S, S))

    return warped_image


def preprocess_norm_pad(img: Image.Image, model_type: str, load_size: Union[int, Tuple[int, int]] = None) -> Tuple[
    torch.Tensor, Image.Image, int, Tuple[int, int, int, int]]:
    scale_factor = 1
    mean = (0.485, 0.456, 0.406) if "dino" in model_type else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if "dino" in model_type else (0.5, 0.5, 0.5)

    padding = (0, 0, 0, 0)  # Initialize padding as (left, top, right, bottom)

    if load_size is not None:
        width, height = img.size
        target_width, target_height = load_size if isinstance(load_size, tuple) else (load_size, load_size)

        # Calculate total padding required
        pad_width = max(0, target_width - width)
        pad_height = max(0, target_height - height)

        # Split padding equally between left-right and top-bottom
        padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)

        # Apply padding to the image
        if any(pad > 0 for pad in padding):  # Only pad if padding is needed
            img = ImageOps.expand(img, padding, fill=0)

        # Calculate scale factor (if any scaling is performed)
        scale_factor = img.size[0] / width

    # Preprocess image: convert to tensor and normalize
    prep = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Prepare the image
    prep_img = prep(img)[None, ...]

    return prep_img, img, scale_factor, padding


def preprocess_norm(img: Image.Image, model_type: str='mast3r', load_size: Union[int, Tuple[int, int]] = None) -> Tuple[
    torch.Tensor, Image.Image, int]:
    scale_factor = 1
    mean = (0.485, 0.456, 0.406) if "dino" in model_type else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if "dino" in model_type else (0.5, 0.5, 0.5)

    if load_size is not None:
        width, height = img.size  # img has to be quadratic
        img = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(img)
        # img = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.NEAREST)(img)
        scale_factor = img.size[0] / width

    prep = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    prep_img = prep(img)[None, ...]

    return prep_img, img, scale_factor


def preprocess(img: Image.Image, load_size: Union[int, Tuple[int, int]] = None) -> Tuple[
    torch.Tensor, Image.Image, int]:
    scale_factor = 1
    if load_size is not None:
        width, height = img.size  # img has to be quadratic
        img = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(img)
        # img = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.NEAREST)(img)
        scale_factor = img.size[0] / width

    prep = transforms.Compose([
        transforms.ToTensor(),
    ])

    prep_img = prep(img)

    return prep_img, img, scale_factor


def preprocess_pad(img: Image.Image, load_size: Union[int, Tuple[int, int]] = None) -> Tuple[
    torch.Tensor, Image.Image, int, Tuple[int, int, int, int]]:
    scale_factor = 1

    padding = (0, 0, 0, 0)  # Initialize padding as (left, top, right, bottom)

    if load_size is not None:
        width, height = img.size
        target_width, target_height = load_size if isinstance(load_size, tuple) else (load_size, load_size)

        # Calculate total padding required
        pad_width = max(0, target_width - width)
        pad_height = max(0, target_height - height)

        # Split padding equally between left-right and top-bottom
        padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)

        # Apply padding to the image
        if any(pad > 0 for pad in padding):  # Only pad if padding is needed
            img = ImageOps.expand(img, padding, fill=0)

        # Calculate scale factor (if any scaling is performed)
        scale_factor = img.size[0] / width

    # Preprocess image: convert to tensor and normalize
    prep = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Prepare the image
    prep_img = prep(img)

    return prep_img, img, scale_factor, padding

# Center padding function remains the same
def center_padding(image: Image.Image, patch_size: int) -> Image.Image:
    w, h = image.size  # Get the width and height of the PIL image
    diff_h = h % patch_size
    diff_w = w % patch_size

    # If the image dimensions are already divisible by patch_size, no padding is needed
    if diff_h == 0 and diff_w == 0:
        return image

    # Calculate padding needed to make the height and width divisible by patch_size
    pad_h = patch_size - diff_h if diff_h != 0 else 0
    pad_w = patch_size - diff_w if diff_w != 0 else 0

    # Divide the padding into top/bottom and left/right
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad the image using ImageOps.expand
    padded_image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

    return padded_image
