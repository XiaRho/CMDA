import torch
import numpy as np
import os
import random
import string
import torch.nn as nn
import zipfile
from PIL import Image

predicts_color_list_dsec = [70, 130, 180, 70, 70, 70, 190, 153, 153, 220, 20, 60, 153, 153, 153, 128, 64, 128,
                            244, 35, 232, 107, 142, 35, 0, 0, 142, 102, 102, 156, 250, 170, 30]
predicts_color_list_cityscapes = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153,
                                  250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
                                  255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]


def dataset_tensor_to_np(tensor, dtype='image'):
    assert dtype in ['image', 'events', '01', 'seg_11', '-1+1', 'seg_19']
    assert len(tensor.shape) == 4
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor_show = np.zeros((tensor.shape[2], 0, 3))
    tensor = tensor.cpu().numpy()
    if dtype == 'image':
        if np.max(tensor) > 1:
            for index in range(tensor.shape[1]):
                tensor[:, index, :, :] = tensor[:, index, :, :] * mean_std[1][index] + mean_std[0][index]
        output = np.uint8(np.transpose(tensor * 255, (0, 2, 3, 1)))  # (0, 1)
    elif dtype == 'events':
        tensor = np.mean(tensor, axis=1, keepdims=True)
        tensor = np.uint8((tensor + 1) / 2 * 255).repeat(3, axis=1)
        output = np.transpose(tensor, (0, 2, 3, 1))
    elif dtype == 'seg_11' or dtype == 'seg_19':
        color_list = predicts_color_list_dsec if dtype == 'seg_11' else predicts_color_list_cityscapes
        tensor_list = []
        for i in range(tensor.shape[0]):
            tensor_color = Image.fromarray(tensor[i, 0].astype(np.uint8)).convert('P')
            tensor_color.putpalette(color_list)
            tensor_color = np.transpose(np.uint8(np.array(tensor_color.convert('RGB'))), (0, 1, 2))
            tensor_list.append(tensor_color[None])
        output = np.concatenate(tensor_list, axis=0)
    elif dtype == '-1+1':
        tensor = np.uint8((tensor + 1) / 2 * 255).repeat(3, axis=1)
        output = np.transpose(tensor, (0, 2, 3, 1))
    else:
        tensor = np.uint8(tensor * 255).repeat(3, axis=1)
        output = np.transpose(tensor, (0, 2, 3, 1))
    for i in range(tensor.shape[0]):
        tensor_show = np.concatenate((tensor_show, output[i]), axis=1)
    return np.uint8(tensor_show)


def get_random_string(length):
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(length))


def lr_poly(base_lr, now_iter_num, total_iter_num, power):
    return base_lr * ((1 - float(now_iter_num) / total_iter_num) ** (power))


def lr_warmup(base_lr, now_iter_num, total_warmup_iter_num):
    return base_lr * (float(now_iter_num) / total_warmup_iter_num)


def adjust_learning_rate(optimizer, now_iter_num, total_iter_num, learning_rate, warm_up_flag=True):
    if now_iter_num < total_iter_num // 20 and warm_up_flag:
        lr = lr_warmup(learning_rate, now_iter_num, total_iter_num // 20)
    else:
        lr = lr_poly(learning_rate, now_iter_num, total_iter_num, power=0.9)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class Rho(nn.Module):
    def __init__(self):
        super(Rho, self).__init__()
        rho_kernels = [
            torch.tensor([[3, -1],
                          [-1, -1]], dtype=torch.float),
            torch.tensor([[-1, 3],
                          [-1, -1]], dtype=torch.float),
            torch.tensor([[-1, -1],
                          [-1, 3]], dtype=torch.float),
            torch.tensor([[-1, -1],
                          [3, -1]], dtype=torch.float)]
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, padding=1, bias=False, padding_mode='reflect')
        with torch.no_grad():
            for i, kernel in enumerate(rho_kernels):
                self.conv1.weight[i].copy_(kernel.unsqueeze(0))

    @torch.no_grad()
    def forward(self, x):
        return self.conv1(x)[:, :, :-1, :-1]


class Diff(nn.Module):
    def __init__(self):
        super(Diff, self).__init__()
        diff_kernels = [
            torch.tensor([[3, -1],
                          [-1, -1]], dtype=torch.float)]
        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, padding=1, bias=False, padding_mode='reflect')
        with torch.no_grad():
            for i, kernel in enumerate(diff_kernels):
                self.conv1.weight[i].copy_(kernel.unsqueeze(0))

    @torch.no_grad()
    def forward(self, x):
        return self.conv1(x)[:, :, :-1, :-1]


def zipdir(path, zip_file_path, not_included_dirs={'dsec_dataset', '[work_dirs]', 'pretrained_model', 'wandb'}):
    zip_file = zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED)
    def a_in_b_set(a, b):
        for item in b:
            if item in a:
                return True
        return False
    for root, dirs, files in os.walk(path):
        if a_in_b_set(root, not_included_dirs):
            continue
        for file in files:
            if '.zip' in file:
                continue
            filepath = os.path.join(root, file)
            zip_file.write(filepath)
    zip_file.close()


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y, weight=None):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        if weight is not None:
            error = error * weight
        loss = torch.mean(error)
        return loss


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


def intersect_and_union(pred_label, label, num_classes, ignore_index, label_map=dict(), reduce_zero_label=False):
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    '''if isinstance(label, str):
        label = torch.from_numpy( mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:'''
    label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label
