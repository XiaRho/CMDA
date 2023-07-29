# Obtained from: https://github.com/vikolss/DACS

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def strong_transform(param, data=None, target=None, isr_flag=False):
    assert ((data is not None) or (target is not None))
    if data is None and isr_flag:
        data = target
        target = None
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    if isr_flag:
        mean = torch.tensor([[[[127.5]], [[127.5]], [[127.5]]]]).cuda()
        std = torch.tensor([[[[127.5]], [[127.5]], [[127.5]]]]).cuda()
    else:
        mean = param['mean']
        std = param['std']
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=mean,
        std=std,
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], sigma=param['sigma'], data=data, target=target)
    if isr_flag:
        target = data
        data = None
    return data, target


def get_mean_std(img_metas, dev):
    if img_metas is None:
        mean = [torch.as_tensor([123.675, 116.28, 103.53], device=dev)]
        std = [torch.as_tensor([58.395, 57.12, 57.375], device=dev)]
    else:
        mean = [torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
                for i in range(len(img_metas))]
        std = [torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
               for i in range(len(img_metas))]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)  # (-2, 1) --> (0, 255) --> (0, 1)
                data = seq(data)
                renorm_(data, mean, std)  # (0, 1) --> (0, 255) --> (-2, 1)
    return data, target


def gaussian_blur(blur, sigma, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target


def sky_mask_transform(param, isr, label):
    kernel_size = torch.randint(21, 61, size=(1,)).item()
    lambda_erase_expansion = torch.empty(size=(1,)).uniform_(0.1, 0.3).item()
    noise_intensity = torch.empty(size=(1,)).uniform_(0.5, 1.2).item()
    chunk_size = 8
    sky_mask = (label == 10).float()
    if torch.nonzero(sky_mask).size(0) < 10:
        return isr
    if kernel_size % 2 == 0:
        kernel_size += 1

    isr *= (1 - sky_mask)

    sky_mask_expansion = F.max_pool2d(sky_mask, kernel_size=kernel_size, stride=1,
                                      padding=kernel_size // 2)
    sky_mask_weight = F.avg_pool2d(sky_mask, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size // 2)

    sky_mask_weight = sky_mask_weight * torch.logical_not(sky_mask)
    max_val, min_val = torch.max(sky_mask_weight), torch.min(sky_mask_weight)
    sky_mask_weight = (sky_mask_weight - min_val) / (max_val - min_val)
    isr_blur_weight = 1 - torch.clamp(sky_mask_weight + lambda_erase_expansion * (sky_mask_weight != 0), min=0,
                                      max=1)
    # noise_path = self.sky_mask + self.isr_noise_list[torch.randint(0, len(self.isr_noise_list), size=(1,)).item()]
    noise_path = param['noise_root_path'] + param['noise_list'][torch.randint(0, len(param['noise_list']), size=(1,)).item()]
    noise = Image.open(noise_path)
    noise = torch.from_numpy(np.array(noise)).cuda() / 128 - 1

    for i in range(2):
        image_chunks = torch.split(noise, chunk_size, dim=i)
        shuffle_idx = torch.randperm(len(image_chunks))
        image_chunks_shuffled = [image_chunks[i] for i in shuffle_idx]
        noise = torch.cat(image_chunks_shuffled, dim=i)

    isr_aug = isr * isr_blur_weight + noise * sky_mask_expansion * noise_intensity
    isr_aug = torch.clamp(isr_aug, min=-1, max=1)

    return isr_aug


@torch.no_grad()
def seg_label_to_edge_label(seg_label):
    kernel_size = 3
    label_diff = F.pad(seg_label.float(), (1, 1, 1, 1), mode='replicate')
    label_diff = F.avg_pool2d(label_diff, kernel_size=kernel_size, stride=1, padding=0)
    label_diff = label_diff - seg_label
    mask = F.max_pool2d(seg_label.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    label_diff[label_diff != 0] = 1
    label_diff[mask >= 255] = 0
    return label_diff.long().detach()


@torch.no_grad()
def add_noise_on_isr(img_self_res, transform_type='noise+blur'):
    if 'blur' in transform_type:
        if torch.rand(1) < 0.5:  # blur
            raw_size = img_self_res.shape[1:]
            blur_kernel_size = 2
            img_self_res = F.avg_pool2d(img_self_res[None], kernel_size=(blur_kernel_size, blur_kernel_size))
            img_self_res = F.interpolate(img_self_res, size=raw_size, mode='bilinear', align_corners=False)[0]

    if 'noise' in transform_type:
        disappear_mask_threshold = (1.0, 1.5)
        random_mask_threshold = (0.4, 0.6)
        noise_intensity = (0.1, 0.3)
        # mask some pixel
        disappear_mask_threshold = torch.empty(size=(1,)).uniform_(*disappear_mask_threshold).item()
        disappear_mask = torch.abs(torch.randn_like(img_self_res)) < disappear_mask_threshold
        img_self_res = img_self_res * disappear_mask

        # add gauss noise on random pixel
        random_mask_threshold = torch.empty(size=(1,)).uniform_(*random_mask_threshold).item()
        noise_intensity = torch.empty(size=(1,)).uniform_(*noise_intensity).item()
        random_mask = torch.abs(torch.randn_like(img_self_res)) < random_mask_threshold
        img_self_res = img_self_res + torch.randn_like(img_self_res) * noise_intensity * random_mask

        img_self_res = torch.clamp(img_self_res, min=-1, max=1)
    return img_self_res

