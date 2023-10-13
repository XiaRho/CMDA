import os
import time

from PIL import Image
import random

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
from torchvision.transforms.functional import crop
from mmseg.datasets.utils import get_image_change_from_pil, cow_masks
import math

from .builder import DATASETS


def my_defined_crop(image):
    return crop(image, 592, 464, 512, 512)  # top, left, height, width

@DATASETS.register_module()
class CityscapesICDataset(Dataset):
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, dataset_path, image_resize_size=(1024, 512), image_crop_size=(512, 512),
                 image_change_range=1, classes=CLASSES, palette=PALETTE, return_GI_or_IC='image_change',
                 isr_shift_pixel=4, enforce_3_channels=True, outputs={'image', 'label'}, isr_noise=False,
                 isr_cow_mask=False, high_resolution_isr=False, random_flare=None, cs_isr_data_type='day',
                 sky_mask=None, shift_3_channel=False, isr_parms='', shift_type='rightdown'):

        assert image_crop_size[0] <= image_crop_size[0] and image_crop_size[1] <= image_resize_size[1]

        # self.file_path = {'image': [], 'image_change': [], 'label': []}
        self.file_path = {'image': [], 'image_change': [], 'label': [], 'events_night': [], 'events_gan': []}
        city_names_list = os.listdir('{}leftImg8bit/train/'.format(dataset_path))
        for city_name in city_names_list:
            city_path = '{}leftImg8bit/train/{}/'.format(dataset_path, city_name)
            images_list = os.listdir(city_path)
            for image_name in images_list:

                image_path = city_path + image_name
                self.file_path['image'].append(image_path)

                image_change_path = image_path.replace('leftImg8bit', 'leftImg8bit_IC1')[:-8] + '.png'
                self.file_path['image_change'].append(image_change_path)

                events_night_path = image_path.replace('leftImg8bit', 'leftImg8bit_EN1')[:-8] + '.png'
                self.file_path['events_night'].append(events_night_path)

                label_name = image_path.replace('leftImg8bit', 'gtFine')[:-4] + '_labelTrainIds.png'
                self.file_path['label'].append(label_name)

                # events_gan_name = image_path.replace('leftImg8bit', 'leftImg8bit_EventGAN')[:-13] + '.png'
                # self.file_path['events_gan'].append(events_gan_name)
                #
                # events_gan_name = image_path.replace('leftImg8bit', 'leftImg8bit_esim')[:-13] + '.png'
                # self.file_path['events_esim'].append(events_gan_name)

        self.image_resize_size = image_resize_size
        self.image_crop_size = image_crop_size
        self.image_change_range = image_change_range
        self.CLASSES, self.PALETTE = classes, palette
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        image_transform = [standard_transforms.ToTensor(), standard_transforms.Normalize(*self.mean_std)]
        self.image_transform = standard_transforms.Compose(image_transform)
        self.HorizontalFlip = standard_transforms.RandomHorizontalFlip(p=1)
        self.ignore_index = 255
        self.label_map = None
        self.reduce_zero_label = False
        self.return_GI_or_IC = return_GI_or_IC
        self.isr_shift_pixel = isr_shift_pixel
        self.enforce_3_channels = enforce_3_channels
        self.isr_noise = isr_noise
        self.disappear_mask_threshold = (1.0, 1.5)
        self.random_mask_threshold = (0.4, 0.6)
        self.noise_intensity = (0.1, 0.3)
        self.outputs = outputs
        self.isr_cow_mask = isr_cow_mask
        self.high_resolution_isr = high_resolution_isr
        self.random_flare = random_flare
        self.cs_isr_data_type = cs_isr_data_type
        assert self.cs_isr_data_type in {'day', 'new_day'}
        self.sky_mask = sky_mask
        self.shift_3_channel = shift_3_channel
        self.isr_parms = isr_parms
        self.shift_type = shift_type
        assert self.shift_type in {'all', 'random', 'rightdown'}
        if shift_3_channel:
            assert not self.high_resolution_isr
            if self.cs_isr_data_type == 'day':
                self.image_change_parms = [
                    {'val_range': (1, 10), '_threshold': 0.025, '_clip_range': 0.05, 'shift_pixel': 1},
                    {'val_range': (1, 10), '_threshold': 0.030, '_clip_range': 0.20, 'shift_pixel': 3},
                    {'val_range': (1, 10), '_threshold': 0.040, '_clip_range': 0.32, 'shift_pixel': 5}]
            elif self.cs_isr_data_type == 'new_day':
                self.image_change_parms = [
                    {'val_range': (1e-5, 255 + 1e-5), '_threshold': 0, '_clip_range': 0.015, 'shift_pixel': 1},
                    {'val_range': (1e-5, 255 + 1e-5), '_threshold': 0, '_clip_range': 0.040, 'shift_pixel': 3},
                    {'val_range': (1e-5, 255 + 1e-5), '_threshold': 0, '_clip_range': 0.070, 'shift_pixel': 5}]
        else:
            if self.cs_isr_data_type == 'day':
                self.image_change_parms = {'val_range': (1, 10), '_threshold': 0.03, '_clip_range': 0.2, 'shift_pixel': 3}
            elif self.cs_isr_data_type == 'new_day':
                self.image_change_parms = {'val_range': (1e-5, 255 + 1e-5), '_threshold': 0, '_clip_range': 0.040, 'shift_pixel': 3}

        if self.isr_parms != '':
            assert not shift_3_channel
            assert isinstance(self.isr_parms, dict)
            self.image_change_parms = self.isr_parms

        if self.random_flare is not None:
            assert not self.high_resolution_isr
            self.flare_name_list = os.listdir(self.random_flare)
            self.to_tensor = standard_transforms.ToTensor()
            self.color_jitter = standard_transforms.ColorJitter(brightness=(0.8, 3), hue=0.0)
            self.blur_transform = standard_transforms.GaussianBlur(21, sigma=(0.1, 3.0))
            self.to_pil = standard_transforms.ToPILImage()
            self.transform_flare = standard_transforms.Compose([
                standard_transforms.RandomVerticalFlip(),
                # standard_transforms.RandomAffine(degrees=(0, 360), scale=(0.8, 1.5), translate=(300/1440, 300/1440), shear=(-20, 20)),
                standard_transforms.RandomAffine(degrees=(0, 360), scale=(0.2, 0.4), translate=(256/1440, 128/1440), shear=(-20, 20)),
                # standard_transforms.CenterCrop((512, 512)),
                standard_transforms.Lambda(my_defined_crop),
                standard_transforms.RandomHorizontalFlip()])

        if self.sky_mask is not None:
            assert not self.high_resolution_isr
            self.enforce_sky_zero = True
            self.isr_noise_list = os.listdir(self.sky_mask)

        assert self.return_GI_or_IC in ['image_change', 'gray_image', 'ic_wo_cyclegan', 'events_gan', 'events_esim']

    def __len__(self):
        return len(self.file_path['image'])

    def __getitem__(self, idx):
        output = dict()
        flip_flag = True if random.random() < 0.5 else False
        x = random.randint(0, self.image_resize_size[0] - self.image_crop_size[0])
        y = random.randint(0, self.image_resize_size[1] - self.image_crop_size[1])

        if 'image' in self.outputs:
            raw_image = Image.open(self.file_path['image'][idx]).convert('RGB')
            resize_image = raw_image.resize(size=self.image_resize_size, resample=Image.BILINEAR)
            crop_image = resize_image.crop(box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))  # (L, upper, R, lower)
            if flip_flag:
                crop_image = self.HorizontalFlip(crop_image)
            if self.random_flare is not None:
                crop_image = self.flare_transform(crop_image)
            image = self.image_transform(crop_image)
            output['image'] = image

        if 'label' in self.outputs:
            raw_label = Image.open(self.file_path['label'][idx])
            label = raw_label.resize(size=self.image_resize_size, resample=Image.NEAREST)
            label = label.crop(box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))
            if flip_flag:
                label = self.HorizontalFlip(label)
            label = np.asarray(label, dtype=np.float32)
            label = torch.from_numpy(label)
            label = torch.round(label).long()[None]
            output['label'] = label

        if 'img_time_res' in self.outputs:
            if self.return_GI_or_IC in {'image_change', 'ic_wo_cyclegan'}:
                img_time_res = Image.open(self.file_path['image_change'][idx]).convert('L')
                img_time_res = img_time_res.resize(size=self.image_resize_size, resample=Image.BILINEAR)
                img_time_res = img_time_res.crop(box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))
                if flip_flag:
                    img_time_res = self.HorizontalFlip(img_time_res)
                img_time_res = np.asarray(img_time_res, dtype=np.float32)
                img_time_res = (torch.from_numpy(img_time_res)[None] / 255.0 - 0.5) / 0.5
            elif self.return_GI_or_IC == 'events_gan':
                img_time_res = Image.open(self.file_path['events_gan'][idx]).convert('L')
                img_time_res = img_time_res.resize(size=self.image_resize_size, resample=Image.BILINEAR)
                img_time_res = img_time_res.crop(box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))
                if flip_flag:
                    img_time_res = self.HorizontalFlip(img_time_res)
                img_time_res = np.asarray(img_time_res, dtype=np.float32)
                img_time_res = (torch.from_numpy(img_time_res)[None] / 255.0 - 0.5) / 0.5
            elif self.return_GI_or_IC == 'events_esim':
                img_time_res = Image.open(self.file_path['events_esim'][idx]).convert('L')
                img_time_res = img_time_res.crop(box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))
                if flip_flag:
                    img_time_res = self.HorizontalFlip(img_time_res)
                img_time_res = np.asarray(img_time_res, dtype=np.float32)
                img_time_res = (torch.from_numpy(img_time_res)[None] / 255.0 - 0.5) / 0.5
            else:
                img_time_res = Image.open(self.file_path['image'][idx]).convert('L')
                img_time_res = img_time_res.resize(size=self.image_resize_size, resample=Image.BILINEAR)
                img_time_res = img_time_res.crop(box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))
                if flip_flag:
                    img_time_res = self.HorizontalFlip(img_time_res)
                img_time_res = np.asarray(img_time_res, dtype=np.float32)
                img_time_res = (torch.from_numpy(img_time_res)[None] / 255.0 - 0.5) / 0.5
            if self.enforce_3_channels:
                img_time_res = img_time_res.repeat(3, 1, 1)
            output['img_time_res'] = img_time_res

        if 'img_self_res' in self.outputs:
            if self.high_resolution_isr:
                img_self_res = get_image_change_from_pil(raw_image, width=raw_image.size[0],
                                                         height=raw_image.size[1], data_type=self.cs_isr_data_type,
                                                         shift_pixel=self.isr_shift_pixel * 2)
                '''img_self_res = F.interpolate(img_self_res[None], size=(self.image_resize_size[1], self.image_resize_size[0]),
                                             mode='bilinear', align_corners=False)[0]'''
                img_self_res = F.interpolate(img_self_res[None], size=(self.image_resize_size[1], self.image_resize_size[0]),
                                             mode='nearest')[0]
                img_self_res = img_self_res[:, y: y + self.image_crop_size[1], x: x + self.image_crop_size[0]]
                if flip_flag:
                    img_self_res = self.HorizontalFlip(img_self_res)
            else:
                if self.shift_3_channel:
                    img_self_res = []
                    for i in range(3):
                        img_self_res.append(get_image_change_from_pil(crop_image, width=self.image_crop_size[0],
                                                                      height=self.image_crop_size[1],
                                                                      **self.image_change_parms[i]))
                    img_self_res = torch.cat(img_self_res, dim=0)
                else:
                    if self.shift_type == 'random':
                        direct = [['leftdown', 'leftup'], ['rightdown', 'rightup']]
                        this_shift_direction = direct[x % 2][int(flip_flag)]
                    else:
                        this_shift_direction = self.shift_type
                    img_self_res = get_image_change_from_pil(crop_image, width=self.image_crop_size[0],
                                                             height=self.image_crop_size[1],
                                                             shift_direction=this_shift_direction,
                                                             **self.image_change_parms)
                if self.sky_mask is not None:
                    img_self_res = self.sky_mask_transform(img_self_res, label)
            if self.isr_noise:
                if torch.rand(1) < 0.5:  # blur
                    raw_size = img_self_res.shape[1:]
                    blur_kernel_size = 2
                    img_self_res = F.avg_pool2d(img_self_res[None], kernel_size=(blur_kernel_size, blur_kernel_size))
                    img_self_res = F.interpolate(img_self_res, size=raw_size, mode='bilinear', align_corners=False)[0]

                # mask some pixel
                disappear_mask_threshold = random.uniform(*self.disappear_mask_threshold)
                disappear_mask = torch.abs(torch.randn_like(img_self_res)) < disappear_mask_threshold
                img_self_res = img_self_res * disappear_mask

                # add gauss noise on random pixel
                random_mask_threshold = random.uniform(*self.random_mask_threshold)
                noise_intensity = random.uniform(*self.noise_intensity)
                random_mask = torch.abs(torch.randn_like(img_self_res)) < random_mask_threshold
                img_self_res = img_self_res + torch.randn_like(img_self_res) * noise_intensity * random_mask

                img_self_res = torch.clamp(img_self_res, min=-1, max=1)

            if self.isr_cow_mask:
                cow_mask = cow_masks(torch.zeros([1, 1, self.image_crop_size[1], self.image_crop_size[0]]),
                                     prop_range=[0.7, 0.7], log_sigma_range=[math.log(16), math.log(17)]).float()[0]
                img_self_res *= cow_mask

            if self.enforce_3_channels and img_self_res.shape[0] == 1:
                img_self_res = img_self_res.repeat(3, 1, 1)
            output['img_self_res'] = img_self_res

        return output

    def flare_transform(self, crop_image_pil):
        # self.random_flare + random.choice(self.flare_name_list)
        flare_path = self.random_flare + self.flare_name_list[torch.randint(0, len(self.flare_name_list), size=(1,)).item()]
        flare_img = Image.open(flare_path)
        flare_img = self.to_tensor(flare_img)

        # flare_img = remove_background(flare_img)
        flare_img = np.float32(np.array(flare_img))
        _EPS = 1e-7
        rgb_max = np.max(flare_img, (0, 1))
        rgb_min = np.min(flare_img, (0, 1))
        flare_img = (flare_img - rgb_min) * rgb_max / (rgb_max - rgb_min + _EPS)
        flare_img = torch.from_numpy(flare_img)

        flare_img = self.transform_flare(flare_img)
        flare_img = self.color_jitter(flare_img)

        flare_img = self.blur_transform(flare_img)
        # flare_img + flare_DC_offset, np.random.uniform(-0.02, 0.02)
        flare_img = flare_img + torch.empty(size=(1,)).uniform_(-0.02, 0.02).item()
        flare_img = torch.clamp(flare_img, min=0, max=1)

        crop_image_torch = self.to_tensor(crop_image_pil)
        merge_img = crop_image_torch + flare_img
        merge_img = torch.clamp(merge_img, min=0, max=1)
        merge_img_pil = self.to_pil(merge_img)

        return merge_img_pil

    def sky_mask_transform(self, isr, label):
        kernel_size = torch.randint(21, 61, size=(1,)).item()
        lambda_erase_expansion = torch.empty(size=(1,)).uniform_(0.1, 0.3).item()
        noise_intensity = torch.empty(size=(1,)).uniform_(0.5, 1.2).item()
        chunk_size = 8
        sky_mask = (label == 10).float()
        if torch.nonzero(sky_mask).size(0) < 10:
            return isr
        if kernel_size % 2 == 0:
            kernel_size += 1
        if self.enforce_sky_zero:
            isr *= (1 - sky_mask)

        sky_mask_expansion = F.max_pool2d(sky_mask.cuda(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2).cpu()
        sky_mask_weight = F.avg_pool2d(sky_mask.cuda(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2).cpu()

        sky_mask_weight = sky_mask_weight * torch.logical_not(sky_mask)
        max_val, min_val = torch.max(sky_mask_weight), torch.min(sky_mask_weight)
        sky_mask_weight = (sky_mask_weight - min_val) / (max_val - min_val)
        isr_blur_weight = 1 - torch.clamp(sky_mask_weight + lambda_erase_expansion * (sky_mask_weight != 0), min=0, max=1)
        noise_path = self.sky_mask + self.isr_noise_list[torch.randint(0, len(self.isr_noise_list), size=(1,)).item()]
        noise = Image.open(noise_path)
        noise = torch.from_numpy(np.array(noise)) / 128 - 1

        for i in range(2):
            image_chunks = torch.split(noise, chunk_size, dim=i)
            shuffle_idx = torch.randperm(len(image_chunks))
            image_chunks_shuffled = [image_chunks[i] for i in shuffle_idx]
            noise = torch.cat(image_chunks_shuffled, dim=i)

        isr_aug = isr * isr_blur_weight + noise * sky_mask_expansion * noise_intensity
        isr_aug = torch.clamp(isr_aug, min=-1, max=1)

        return isr_aug


if __name__ == '__main__':
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    save_name = None
    dataset = CityscapesICDataset(dataset_path='D:/研究生/Python/HANet/cityscapes/', isr_cow_mask=False,
                                  outputs={'img_self_res', 'image', 'label'}, high_resolution_isr=False,
                                  sky_mask='E:/Dataset/dark_zurich_noise/isr_noise/',
                                  cs_isr_data_type='new_day')  # random_flare='E:/Dataset/Flare7k/Scattering_Flare/Compound_Flare/'
    '''data = dataset[50]
    isr = data['img_self_res'][None].cuda()
    image = data['image'][None].cuda()
    print(isr.shape, torch.max(isr), torch.min(isr))
    print(image.shape, torch.max(image), torch.min(image))

    means = torch.tensor([[[[123.6750]], [[116.2800]], [[103.5300]]]]).cuda()
    stds = torch.tensor([[[[58.3950]], [[57.1200]], [[57.3750]]]]).cuda()
    color_jitter_s = 0.2
    color_jitter_p = 0.2
    blur = 1.0
    color_jitter_random_val = 1.0
    strong_parameters = {'mix': torch.mean(torch.zeros_like(image), dim=1, keepdim=True)[0],
                         'color_jitter': color_jitter_random_val,
                         'color_jitter_s': color_jitter_s, 'color_jitter_p': color_jitter_p,
                         'blur': blur, 'mean': means, 'std': stds}
    from mmseg.models.utils.dacs_transforms import strong_transform, denorm
    denorm_image = denorm(image, means, stds)  # (0~1)
    mixed_img, _ = strong_transform(strong_parameters, data=torch.stack((image[0], image[0])))
    denorm_mixed_image = denorm(mixed_img, means, stds)  # (0~1)

    denorm_image = np.transpose(np.uint8(denorm_image.cpu().numpy() * 255)[0], (1, 2, 0))
    denorm_mixed_image = np.transpose(np.uint8(denorm_mixed_image.cpu().numpy() * 255)[0], (1, 2, 0))

    # isr
    means_isr = torch.tensor([[[[127.5]], [[127.5]], [[127.5]]]]).cuda()
    stds_isr = torch.tensor([[[[127.5]], [[127.5]], [[127.5]]]]).cuda()
    strong_parameters = {'mix': torch.mean(torch.zeros_like(image), dim=1, keepdim=True)[0],
                         'color_jitter': color_jitter_random_val,
                         'color_jitter_s': color_jitter_s, 'color_jitter_p': color_jitter_p,
                         'blur': blur, 'mean': means_isr, 'std': stds_isr}
    denorm_isr = denorm(isr, means_isr, stds_isr)  # (0~1)
    mixed_isr, _ = strong_transform(strong_parameters, data=torch.stack((isr[0], isr[0])))
    mixed_isr = torch.mean(mixed_isr, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    denorm_mixed_isr = denorm(mixed_isr, means_isr, stds_isr)  # (0~1)

    denorm_isr = np.transpose(np.uint8(denorm_isr.cpu().numpy() * 255)[0], (1, 2, 0))
    denorm_mixed_isr = np.transpose(np.uint8(denorm_mixed_isr.cpu().numpy() * 255)[0], (1, 2, 0))

    output_image = Image.fromarray(np.concatenate((denorm_image, denorm_mixed_image, denorm_isr, denorm_mixed_isr), axis=1))
    # output_image = Image.fromarray(isr)
    output_image.show('1')
    if save_name is not None:
        output_image.save(save_name)'''


    def dataset_tensor_to_np(tensor, dtype='image'):
        assert dtype in ['image', 'events', 'seg']
        output = None
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        cityscapes_color_list = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153,
                                 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
                                 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        if dtype == 'image':
            tensor = tensor.cpu().numpy()
            for index in range(tensor.shape[0]):
                tensor[index, :, :] = tensor[index, :, :] * mean_std[1][index] + mean_std[0][index]
            output = np.uint8(np.transpose(tensor * 255, (1, 2, 0)))  # (0, 1)
        elif dtype == 'events':
            tensor = tensor.cpu().numpy()
            tensor = np.uint8((tensor[0] + 1) / 2 * 255)
            tensor = np.expand_dims(tensor, axis=0).repeat(3, axis=0)
            output = np.transpose(tensor, (1, 2, 0))
        elif dtype == 'seg':
            if tensor.shape[0] != 1:
                tensor = torch.argmax(tensor, dim=0)
            else:
                tensor = tensor[0]
            tensor = tensor.cpu().numpy()
            tensor = Image.fromarray(tensor.astype(np.uint8)).convert('P')
            tensor.putpalette(cityscapes_color_list)
            output = np.uint8(np.array(tensor.convert('RGB')))
        return output

    save_path = 'D:\研究生\Python\Dataset\Flare7k\Scattering_Flare'

    for i in range(10):
        data = dataset[i]
        isr = data['img_self_res'][None].cuda()
        image = data['image'][None].cuda()
        label = data['label'][None].cuda()

        image_pil = dataset_tensor_to_np(image[0], dtype='image')
        isr_pil = dataset_tensor_to_np(isr[0], dtype='events')
        label_pil = dataset_tensor_to_np(label[0], dtype='seg')

        output_image = Image.fromarray(np.concatenate((image_pil, isr_pil, label_pil), axis=1))
        save_name = '{}/{}.png'.format(save_path, i)
        output_image.save(save_name)
        print(save_name)
