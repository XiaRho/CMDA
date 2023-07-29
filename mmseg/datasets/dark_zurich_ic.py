import os
from PIL import Image
import random

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F
import mmcv
from .builder import DATASETS
from functools import reduce
from mmseg.core import eval_metrics
from collections import OrderedDict
from prettytable import PrettyTable
from mmcv.utils import print_log
from mmcv.parallel.data_container import DataContainer
from mmseg.datasets.utils import get_image_change_from_pil


@DATASETS.register_module()
class DarkZurichICDataset(Dataset):
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, dataset_path, image_resize_size=(960, 540), image_crop_size=(512, 512),
                 image_resize_size2=None, test_mode=False, split_train=False,
                 dz_isr_data_type='night', shift_pixel=3,
                 enforce_3_channels=True, classes=CLASSES, palette=PALETTE,
                 outputs={'image', 'day_image', 'day_t_isr', 'night_isr', 'night_t_isr', 'label', 'image_deflare', 'night_isr_deflare'},
                 submit_to_website=False, auto_threshold=False, high_resolution_isr=False,
                 isr_parms='', shift_3_channel=False, shift_type='rightdown'):

        assert image_crop_size[0] <= image_crop_size[0] and image_crop_size[1] <= image_resize_size[1]

        self.file_path = {'image': [], 'transferred_isr': [], 'transferred_events': [], 'label': [], 'night_deflare': []}
        train_val_name = 'val' if test_mode else 'train'
        # train_val_name = 'train'
        if submit_to_website:
            train_val_name = 'test'
        sequences_list = os.listdir('{}rgb_anon/{}/night/'.format(dataset_path, train_val_name))
        for sequence in sequences_list:
            sequence_path = '{}rgb_anon/{}/night/{}/'.format(dataset_path, train_val_name, sequence)
            images_list = os.listdir(sequence_path)
            for image_name in images_list:

                image_path = sequence_path + image_name
                self.file_path['image'].append(image_path)

                transferred_isr_path = image_path.replace('night', 'night_t_isr')
                self.file_path['transferred_isr'].append(transferred_isr_path)

                night_deflare_path = image_path.replace('night', 'night_deflare')
                self.file_path['night_deflare'].append(night_deflare_path)

                transferred_events_path = image_path.replace('night', 'night_t_events')
                self.file_path['transferred_events'].append(transferred_events_path)

                label_path = image_path.replace('rgb_anon', 'gt').replace('_gt', '_gt_labelTrainIds')
                self.file_path['label'].append(label_path)
        if not test_mode and ('day_image' in outputs or 'day_t_isr' in outputs):
            self.file_path['day_image'] = list()
            self.file_path['day_t_isr'] = list()
            sequences_day_list = os.listdir('{}rgb_anon/train/day/'.format(dataset_path))
            for sequence in sequences_day_list:
                sequence_path = '{}rgb_anon/train/day/{}/'.format(dataset_path, sequence)
                images_list = os.listdir(sequence_path)
                for image_name in images_list:
                    image_path = sequence_path + image_name
                    self.file_path['day_image'].append(image_path)

                    transferred_isr_path = image_path.replace('day', 'day_t_isr')
                    self.file_path['day_t_isr'].append(transferred_isr_path)
            self.day_length = len(self.file_path['day_image'])

        self.image_resize_size = image_resize_size
        self.image_crop_size = image_crop_size
        self.image_resize_size2 = image_resize_size2
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image_transform = [standard_transforms.ToTensor(), standard_transforms.Normalize(*self.mean_std)]
        self.image_transform = standard_transforms.Compose(image_transform)
        self.isr_events_transform = standard_transforms.Compose([standard_transforms.ToTensor(),
                                                                 standard_transforms.Normalize(*([0.5], [0.5]))])

        self.HorizontalFlip = standard_transforms.RandomHorizontalFlip(p=1)
        self.ignore_index = 255
        self.label_map = None
        self.reduce_zero_label = False
        self.test_mode = test_mode
        self.split_train = split_train
        self.dz_isr_data_type = dz_isr_data_type
        self.shift_pixel = shift_pixel
        self.enforce_3_channels = enforce_3_channels
        self.outputs = outputs
        self.submit_to_website = submit_to_website
        self.auto_threshold = auto_threshold
        self.high_resolution_isr = high_resolution_isr
        self.isr_parms = isr_parms
        self.shift_3_channel = shift_3_channel
        self.shift_type = shift_type
        assert self.shift_type in {'all', 'random', 'rightdown'}
        if shift_3_channel:
            assert not self.high_resolution_isr
            if self.dz_isr_data_type == 'night':
                self.image_change_parms = [
                    {'val_range': (9, 255 + 9), '_threshold': 0.012, '_clip_range': 0.04, 'shift_pixel': 1},
                    {'val_range': (9, 255 + 9), '_threshold': 0.012, '_clip_range': 0.12, 'shift_pixel': 3},
                    {'val_range': (9, 255 + 9), '_threshold': 0.012, '_clip_range': 0.20, 'shift_pixel': 5}]
            elif self.dz_isr_data_type == 'new_night':
                self.image_change_parms = [
                    {'val_range': (500, 1000), '_threshold': 0.015, '_clip_range': 0.05, 'shift_pixel': 1},
                    {'val_range': (500, 1000), '_threshold': 0.02, '_clip_range': 0.12, 'shift_pixel': 3},
                    {'val_range': (500, 1000), '_threshold': 0.025, '_clip_range': 0.2, 'shift_pixel': 5}]
        else:
            if self.dz_isr_data_type == 'night':
                self.image_change_parms = {'val_range': (1, 10 ** 2), '_threshold': 0.04, '_clip_range': 0.2, 'shift_pixel': 3}
            elif self.dz_isr_data_type == 'new_night':
                self.image_change_parms = {'val_range': (500, 1000), '_threshold': 0.02, '_clip_range': 0.12, 'shift_pixel': 3}

        if self.isr_parms != '':
            assert not shift_3_channel
            assert isinstance(self.isr_parms, dict)
            self.image_change_parms = self.isr_parms

        self.CLASSES, self.PALETTE = classes, palette

    def __len__(self):
        return len(self.file_path['image'])

    def __getitem__(self, idx):
        output = dict()
        if not self.test_mode:
            flip_flag = True if random.random() < 0.5 else False
            x = random.randint(0, self.image_resize_size[0] - self.image_crop_size[0])
            y = random.randint(0, self.image_resize_size[1] - self.image_crop_size[1])

        if 'image' in self.outputs:
            raw_image = Image.open(self.file_path['image'][idx]).convert('RGB')
            image = raw_image.resize(size=self.image_resize_size, resample=Image.BILINEAR)
            image_pil = image
            if not self.test_mode:
                image = image_pil.crop(box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))  # (L, upper, R, lower)
                if self.image_resize_size2 is not None:
                    image = image.resize(size=self.image_resize_size2, resample=Image.BILINEAR)
                if flip_flag:
                    image = self.HorizontalFlip(image)
            image = self.image_transform(image)
            output['image'] = image

        if self.test_mode and 'label' in self.outputs and not self.submit_to_website:
            raw_label = Image.open(self.file_path['label'][idx])
            label = raw_label.resize(size=self.image_resize_size, resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.float32)
            label = torch.from_numpy(label)
            label = torch.round(label).long()[None]
            output['label'] = label

        if 'night_t_isr' in self.outputs:
            transferred_isr = Image.open(self.file_path['transferred_isr'][idx]).convert('L')
            # transferred_isr = transferred_isr.resize(size=self.image_resize_size, resample=Image.BILINEAR)
            if not self.test_mode:
                transferred_isr = transferred_isr.crop(
                    box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))  # (L, upper, R, lower)
                if self.image_resize_size2 is not None:
                    transferred_isr = transferred_isr.resize(size=self.image_resize_size2, resample=Image.BILINEAR)
                if flip_flag:
                    transferred_isr = self.HorizontalFlip(transferred_isr)
            transferred_isr = self.isr_events_transform(transferred_isr)
            if self.enforce_3_channels and transferred_isr.shape[0] == 1:
                transferred_isr = transferred_isr.repeat(3, 1, 1)
            output['night_t_isr'] = transferred_isr

        if 'image_deflare' in self.outputs:
            image_deflare = Image.open(self.file_path['night_deflare'][idx]).convert('RGB')
            image_deflare_pil = image_deflare
            if not self.test_mode:
                image_deflare = image_deflare_pil.crop(box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))
                if self.image_resize_size2 is not None:
                    image_deflare = image_deflare.resize(size=self.image_resize_size2, resample=Image.BILINEAR)
                if flip_flag:
                    image_deflare = self.HorizontalFlip(image_deflare)
            image_deflare = self.image_transform(image_deflare)
            output['image_deflare'] = image_deflare

        if 'night_isr_deflare' in self.outputs:
            auto_threshold = (image_deflare_pil, 'image_gray') if self.auto_threshold else None
            if self.shift_3_channel:
                night_isr = []
                for i in range(3):
                    night_isr.append(get_image_change_from_pil(image_deflare_pil, width=self.image_resize_size[0],
                                                               shift_pixel=i + 1,
                                                               height=self.image_resize_size[1],
                                                               data_type='night',
                                                               **self.image_change_parms,
                                                               auto_threshold=auto_threshold))
                night_isr = torch.cat(night_isr, dim=0)
            else:
                night_isr = get_image_change_from_pil(image_deflare_pil, width=self.image_resize_size[0],
                                                      shift_pixel=self.shift_pixel,
                                                      height=self.image_resize_size[1], data_type='night',
                                                      **self.image_change_parms,
                                                      auto_threshold=auto_threshold)
            if not self.test_mode:
                night_isr = night_isr[:, y: y + self.image_crop_size[1], x: x + self.image_crop_size[0]]
                assert self.image_resize_size2 is None
                if flip_flag:
                    night_isr = self.HorizontalFlip(night_isr)
            if self.enforce_3_channels and night_isr.shape[0] == 1:
                night_isr = night_isr.repeat(3, 1, 1)
            output['night_isr_deflare'] = night_isr

        if 'night_isr' in self.outputs:
            # [1, 540, 960]
            auto_threshold = (image_pil, 'image_gray') if self.auto_threshold else None
            if self.high_resolution_isr:
                night_isr = get_image_change_from_pil(raw_image, width=raw_image.size[0],
                                                      shift_pixel=self.shift_pixel * 2,
                                                      height=raw_image.size[1], data_type='night',
                                                      **self.image_change_parms,
                                                      auto_threshold=auto_threshold)
                night_isr = F.interpolate(night_isr[None], size=(self.image_resize_size[1], self.image_resize_size[0]),
                                          mode='nearest')[0]
            else:
                if self.shift_3_channel:
                    night_isr = []
                    for i in range(3):
                        night_isr.append(get_image_change_from_pil(image_pil, width=self.image_resize_size[0],
                                                                   height=self.image_resize_size[1],
                                                                   data_type='night',
                                                                   **self.image_change_parms[i],
                                                                   auto_threshold=auto_threshold))
                    night_isr = torch.cat(night_isr, dim=0)
                else:
                    if self.shift_type == 'random':
                        direct = [['leftdown', 'leftup'], ['rightdown', 'rightup']]
                        this_shift_direction = direct[x % 2][y % 2]
                    else:
                        this_shift_direction = self.shift_type
                    night_isr = get_image_change_from_pil(image_pil, width=self.image_resize_size[0],
                                                          height=self.image_resize_size[1],
                                                          **self.image_change_parms,
                                                          shift_direction=this_shift_direction,
                                                          auto_threshold=auto_threshold)
            if not self.test_mode:
                night_isr = night_isr[:, y: y + self.image_crop_size[1], x: x + self.image_crop_size[0]]
                assert self.image_resize_size2 is None
                if flip_flag:
                    night_isr = self.HorizontalFlip(night_isr)
            if self.enforce_3_channels and night_isr.shape[0] == 1:
                night_isr = night_isr.repeat(3, 1, 1)
            output['night_isr'] = night_isr

        # outputs = {'image', 'day_image', 'day_t_isr', 'night_isr', 'night_t_isr', 'night_events', 'label'}
        if 'day_image' in self.outputs:
            idx_day = random.randint(0, self.day_length - 1)

            day_image = Image.open(self.file_path['day_image'][idx_day]).convert('RGB')
            day_image = day_image.resize(size=self.image_resize_size, resample=Image.BILINEAR)
            day_image_pil = day_image
            day_image = day_image_pil.crop(
                box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))  # (L, upper, R, lower)
            if self.image_resize_size2 is not None:
                day_image = day_image.resize(size=self.image_resize_size2, resample=Image.BILINEAR)
            if flip_flag:
                day_image = self.HorizontalFlip(day_image)
            day_image = self.image_transform(day_image)
            output['day_image'] = day_image

        if 'day_t_isr' in self.outputs:
            day_t_isr = Image.open(self.file_path['day_t_isr'][idx_day]).convert('L')
            day_t_isr = day_t_isr.crop(box=(960, 0, 960 + 960, 0 + 540))
            day_t_isr = day_t_isr.crop(box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))  # (L, upper, R, lower)
            if self.image_resize_size2 is not None:
                day_t_isr = day_t_isr.resize(size=self.image_resize_size2, resample=Image.BILINEAR)
            if flip_flag:
                day_t_isr = self.HorizontalFlip(day_t_isr)
            day_t_isr = self.isr_events_transform(day_t_isr)
            if self.enforce_3_channels and day_t_isr.shape[0] == 1:
                day_t_isr = day_t_isr.repeat(3, 1, 1)
            output['day_t_isr'] = day_t_isr

        '''if 'day_t_isr' in self.outputs:
        elif not self.return_only_image and not self.test_mode:
            transferred_events = Image.open(self.file_path['transferred_events'][idx]).convert('L')
            if self.isr_type == 'raw':
                transferred_events = transferred_events.resize(size=self.image_resize_size, resample=Image.BILINEAR)
            # transferred_events = transferred_events.resize(size=self.image_resize_size, resample=Image.BILINEAR)
            if not self.test_mode:
                if self.isr_type == 'raw':
                    transferred_events = transferred_events.resize(size=self.image_resize_size, resample=Image.BILINEAR)
                transferred_events = transferred_events.crop(box=(x, y, x + self.image_crop_size[0], y + self.image_crop_size[1]))  # (L, upper, R, lower)
                if self.image_resize_size2 is not None:
                    transferred_events = transferred_events.resize(size=self.image_resize_size2, resample=Image.BILINEAR)
                if flip_flag:
                    transferred_events = self.HorizontalFlip(transferred_events)
            transferred_events = self.isr_events_transform(transferred_events)
            if self.enforce_3_channels:
                transferred_events = transferred_events.repeat(3, 1, 1)'''

        if self.test_mode:
            img_metas = dict()
            img_metas['img_norm_cfg'] = dict()
            img_metas['img_norm_cfg']['mean'] = [123.675, 116.28, 103.53]
            img_metas['img_norm_cfg']['std'] = [58.395, 57.12, 57.375]
            img_metas['img_norm_cfg']['to_rgb'] = True

            img_metas['img_shape'] = (self.image_resize_size[1], self.image_resize_size[0])
            img_metas['pad_shape'] = (self.image_resize_size[1], self.image_resize_size[0])
            img_metas['ori_shape'] = (self.image_resize_size[1], self.image_resize_size[0])
            img_metas['ori_filename'] = self.file_path['image'][idx].split('/')[-1]

            img_metas['flip'] = False
            if img_metas['flip']:
                img_metas['flip_direction'] = 'horizontal'
            img_metas = DataContainer(img_metas, cpu_only=True)
            output['img_metas'] = img_metas

        return output

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for idx in range(len(self.file_path['label'])):
            seg_map = self.file_path['label'][idx]
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')
            # gt_seg_map = gt_seg_map[:440, :]
            if gt_seg_map.shape == (1080, 1920):
                gt_seg_map_pil = Image.fromarray(gt_seg_map)
                gt_seg_map_pil = gt_seg_map_pil.resize(size=self.image_resize_size, resample=Image.NEAREST)
                gt_seg_map = np.uint8(np.array(gt_seg_map_pil))  # (960, 540) <'numpy.ndarray'> <'numpy.uint8'>
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def evaluate(self, results, metric='mIoU', logger=None, efficient_test=False, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        '''assert len(results) == len(gt_seg_maps)
        num_classes = 11
        class_names = ('background', 'building', 'fence', 'person', 'pole', 'road',
                       'sidewalk', 'vegetation', 'car', 'wall', 'traffic sign')
        dsec_19_to_11_classes = [[0, 5], [1, 6], [2, 1], [3, 9], [4, 2], [5, 4], [6, 10], [7, 10], [8, 7],
                                 [9, 7], [10, 0], [11, 3], [12, 3], [13, 8], [14, 8], [15, 8], [16, 8],
                                 [17, 8], [18, 8]]
        for i in range(len(results)):
            converted_result = np.copy(results[i])
            converted_gt_seg_map = np.copy(gt_seg_maps[i])
            for old_id, new_id in dsec_19_to_11_classes:
                converted_result[results[i] == old_id] = new_id
                converted_gt_seg_map[gt_seg_maps[i] == old_id] = new_id
            results[i] = converted_result
            gt_seg_maps[i] = converted_gt_seg_map'''

        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results


if __name__ == '__main__':
    save_name = None
    dataset = DarkZurichICDataset(dataset_path='E:/Dataset/dark_zurich/',
                                  outputs={'night_isr', 'image'}, high_resolution_isr=False, test_mode=True)
    isr = dataset[30]['night_isr']
    print(isr.shape, torch.max(isr), torch.min(isr))

    isr = np.uint8((isr.cpu().numpy() + 1) / 2 * 255)[0]
    isr = np.expand_dims(isr, axis=2)
    isr = np.repeat(isr, repeats=3, axis=2)

    # output_image = Image.fromarray(np.concatenate((isr, transferred_isr), axis=1))
    output_image = Image.fromarray(isr)
    output_image.show('1')
    if save_name is not None:
        output_image.save(save_name)