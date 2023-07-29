import numpy as np
import torch
import math
import torch.nn.functional as F

_ROOT_2 = math.sqrt(2.0)
_ROOT_2_PI = math.sqrt(2.0 * math.pi)


def tensor_normalize_to_range(tensor, min_val, max_val):
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8) * (max_val - min_val) + min_val
    return tensor


'''def get_ic(image_front, image_now, log_add=10, threshold=0.13, clip_range=0.8, auto_threshold=None):
    image_front = np.asarray(image_front, dtype=np.float32)
    image_front = np.log(image_front + log_add)
    image_now = np.asarray(image_now, dtype=np.float32)
    image_now = np.log(image_now + log_add)
    image_change_ = torch.unsqueeze(torch.from_numpy(image_now - image_front), dim=0)
    if auto_threshold is not None:
        image_pil = auto_threshold[0]
        choice = auto_threshold[1]
        if choice == 'image_hsv_s':
            image_pil_hsv = image_pil.convert('HSV')
            image_hsv = np.array(image_pil_hsv)
            image_hsv_s = image_hsv[:, :, 1]
            large_threshold = 5 * threshold
            small_threshold = 0.0 * threshold
            image_threshold = torch.from_numpy(image_hsv_s) / 255 * (large_threshold - small_threshold) + small_threshold
        else:
            assert choice == 'image_gray'
            image_anti_gray = 255 - np.array(image_pil.convert('L'))
            large_threshold = 5 * threshold
            small_threshold = -5 * threshold
            image_threshold = torch.from_numpy(image_anti_gray) / 255 * (large_threshold - small_threshold) + small_threshold
            image_threshold[image_anti_gray < 100] = 0
        mask = (torch.abs(image_change_) <= image_threshold)
    else:
        mask = (torch.abs(image_change_) <= threshold)
    image_change_[mask] = 0
    image_change_smaller_0 = image_change_.detach().clone()
    image_change_[image_change_ < 0] = 0
    image_change_ = torch.clamp(image_change_, 0, clip_range)
    image_change_ = tensor_normalize_to_range(image_change_, min_val=0, max_val=1)
    image_change_smaller_0[image_change_smaller_0 > 0] = 0
    image_change_smaller_0 = torch.clamp(image_change_smaller_0, -clip_range, 0)
    image_change_smaller_0 = tensor_normalize_to_range(image_change_smaller_0, min_val=-1, max_val=0)
    image_change_ += image_change_smaller_0
    return image_change_


def get_image_change_from_pil(pil_image, width, height, data_type, shift_pixel=4,
                              _log_add=None, _threshold=None, _clip_range=None, auto_threshold=None):
    assert data_type in {'day', 'night', 'new_day'}
    if data_type == 'day':
        log_add, threshold, clip_range = 30, 0.1, 0.5  # 30, 0.1, 0.5
    elif data_type == 'night':
        log_add, threshold, clip_range = 1e-5, 0.13, 0.4
    elif data_type == 'new_day':
        log_add, threshold, clip_range = 1e-5, 0, 0.6
    if _log_add is not None:
        log_add = _log_add
    if _threshold is not None:
        threshold = _threshold
    if _clip_range is not None:
        clip_range = _clip_range

    if shift_pixel == 1:
        clip_range = clip_range / 2
    elif shift_pixel == 5:
        clip_range = clip_range * 2

    inputs_gray = np.array(pil_image.convert('L'))  # (480, 640)
    inputs_right = np.concatenate((inputs_gray[:, :shift_pixel], inputs_gray[:, :width-shift_pixel]), axis=1)
    inputs_down = np.concatenate((inputs_gray[:shift_pixel, :], inputs_gray[:height-shift_pixel, :]), axis=0)
    image_change_1 = get_ic(inputs_gray, inputs_right, log_add=log_add, threshold=threshold,
                            clip_range=clip_range, auto_threshold=auto_threshold)
    image_change_2 = get_ic(inputs_gray, inputs_down, log_add=log_add, threshold=threshold,
                            clip_range=clip_range, auto_threshold=auto_threshold)
    image_avg = image_change_1 / 2 + image_change_2 / 2
    return image_avg  # tensor -1 ~ +1'''


def get_ic(image_front, image_now, val_range, threshold, clip_range):
    image_front = np.asarray(image_front, dtype=np.float32)
    image_front = np.log(image_front / 255 * (val_range[1] - val_range[0]) + val_range[0])
    image_now = np.asarray(image_now, dtype=np.float32)
    image_now = np.log(image_now / 255 * (val_range[1] - val_range[0]) + val_range[0])
    image_change_ = torch.unsqueeze(torch.from_numpy(image_now - image_front), dim=0)
    threshold = (np.log(val_range[1]) - np.log(val_range[0])) * threshold
    clip_range = (np.log(val_range[1]) - np.log(val_range[0])) * clip_range
    mask = (torch.abs(image_change_) <= threshold)
    image_change_[mask] = 0
    image_change_smaller_0 = image_change_.detach().clone()
    image_change_[image_change_ < 0] = 0
    image_change_ = torch.clamp(image_change_, 0, clip_range)
    image_change_ = tensor_normalize_to_range(image_change_, min_val=0, max_val=1)
    image_change_smaller_0[image_change_smaller_0 > 0] = 0
    image_change_smaller_0 = torch.clamp(image_change_smaller_0, -clip_range, 0)
    image_change_smaller_0 = tensor_normalize_to_range(image_change_smaller_0, min_val=-1, max_val=0)
    image_change_ += image_change_smaller_0
    return image_change_


def get_image_change_from_pil(pil_image, width, height, data_type=None, shift_pixel=4, val_range=None,
                              _threshold=None, _clip_range=None, auto_threshold=None, shift_direction='rightdown'):
    '''assert data_type in {'day', 'night', 'new_day'}
    if data_type == 'day':
        val_range, threshold, clip_range = (30, 255 + 30), 0.0444, 0.222  # 30, 0.1, 0.5
    elif data_type == 'new_day':
        val_range, threshold, clip_range = (1e-5, 255 + 1e-5), 0, 0.03518  # 1e-5, 0, 0.6
    elif data_type == 'night':
        val_range, threshold, clip_range = (1e-5, 255 + 1e-5), 7.623e-3, 0.023455  # 1e-5, 0.13, 0.4
    if val_range is not None:
        val_range = val_range
    if _threshold is not None:
        threshold = _threshold
    if _clip_range is not None:
        clip_range = _clip_range'''
    val_range, threshold, clip_range = val_range, _threshold, _clip_range
    if auto_threshold is not None:
        raise ValueError('auto_threshold function not implement！')
    inputs_gray = np.array(pil_image.convert('L'))  # (480, 640)

    if shift_direction == 'all':
        inputs_left = np.concatenate((inputs_gray[:, shift_pixel:], inputs_gray[:, width - shift_pixel:]), axis=1)
        inputs_right = np.concatenate((inputs_gray[:, :shift_pixel], inputs_gray[:, :width - shift_pixel]), axis=1)
        inputs_up = np.concatenate((inputs_gray[shift_pixel:, :], inputs_gray[height - shift_pixel:, :]), axis=0)
        inputs_down = np.concatenate((inputs_gray[:shift_pixel, :], inputs_gray[:height - shift_pixel, :]), axis=0)
        image_change_1 = get_ic(inputs_gray, inputs_up, val_range=val_range, threshold=threshold, clip_range=clip_range)
        image_change_2 = get_ic(inputs_gray, inputs_left, val_range=val_range, threshold=threshold, clip_range=clip_range)
        image_change_3 = get_ic(inputs_gray, inputs_down, val_range=val_range, threshold=threshold, clip_range=clip_range)
        image_change_4 = get_ic(inputs_gray, inputs_right, val_range=val_range, threshold=threshold, clip_range=clip_range)
        image_avg = image_change_1 / 4 + image_change_2 / 4 + image_change_3 / 4 + image_change_4 / 4
    else:
        if 'left' in shift_direction:
            inputs_row = np.concatenate((inputs_gray[:, shift_pixel:], inputs_gray[:, width-shift_pixel:]), axis=1)
        else:
            assert 'right' in shift_direction
            inputs_row = np.concatenate((inputs_gray[:, :shift_pixel], inputs_gray[:, :width-shift_pixel]), axis=1)
        if 'up' in shift_direction:
            inputs_col = np.concatenate((inputs_gray[shift_pixel:, :], inputs_gray[height-shift_pixel:, :]), axis=0)
        else:
            assert 'down' in shift_direction
            inputs_col = np.concatenate((inputs_gray[:shift_pixel, :], inputs_gray[:height-shift_pixel, :]), axis=0)
        image_change_1 = get_ic(inputs_gray, inputs_row, val_range=val_range, threshold=threshold, clip_range=clip_range)
        image_change_2 = get_ic(inputs_gray, inputs_col, val_range=val_range, threshold=threshold, clip_range=clip_range)
        image_avg = image_change_1 / 2 + image_change_2 / 2
    return image_avg  # tensor -1 ~ +1


def gaussian_kernels(sigmas, max_sigma):
    """Make Gaussian kernels for Gaussian blur.
    Args:
        sigmas: kernel sigmas as a [N]
        max_sigma: sigma upper limit as a float (this is used to determine
          the size of kernel required to fit all kernels)
    Returns:
        a (N, kernel_width)
    """
    sigmas = sigmas[:, None]
    size = round(max_sigma * 3) * 2 + 1
    x = torch.arange(-size, size + 1)[None, :].float().to(sigmas.device)
    y = torch.exp(-0.5 * x ** 2 / sigmas ** 2)
    return y / (sigmas * _ROOT_2_PI)


def cow_masks(disp, log_sigma_range=[math.log(4), math.log(16)], max_sigma=16, prop_range=[0.25, 1.0]):
    bz, _, ht, wd = disp.shape
    device = disp.device
    p = torch.randn([bz, ]).uniform_(prop_range[0], prop_range[1]).to(device)
    threshold_factors = (torch.erfinv(2 * p - 1) * _ROOT_2)
    sigmas = torch.exp(torch.randn([bz, ]).uniform_(log_sigma_range[0], log_sigma_range[1]))

    # [B, 1, H, W]
    noise = torch.from_numpy(np.random.normal(size=[bz, 1, ht, wd])).float().to(device)
    # [B, kW]
    kernels = gaussian_kernels(sigmas, max_sigma).to(device)
    kW = kernels.shape[1]
    # [B, 1, 1, kW]
    krn_y = kernels[:, None, None, :]
    # [B, 1, kW, 1]
    krn_x = kernels[:, None, :, None]

    # [B, 1, H, W]
    '''smooth_noise = F.conv2d(noise, krn_y, padding='same')
    smooth_noise = F.conv2d(smooth_noise, krn_x, padding='same')'''
    noise = F.pad(noise, pad=((kW - 1) // 2, (kW - 1) // 2, 0, 0), mode='reflect')
    smooth_noise = F.conv2d(noise, krn_y)
    smooth_noise = F.pad(smooth_noise, pad=(0, 0, (kW - 1) // 2, (kW - 1) // 2), mode='reflect')
    smooth_noise = F.conv2d(smooth_noise, krn_x)

    n_std, n_mean = torch.std_mean(smooth_noise, [1, 2, 3], keepdim=True)

    thresholds = threshold_factors[:, None, None, None] * n_std + n_mean
    mask = (smooth_noise <= thresholds)
    return mask