import os
import os.path as osp
import hdf5plugin
import h5py
from collections import OrderedDict
from functools import reduce
from PIL import Image
import random
import functools

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmcv.parallel.data_container import DataContainer
from prettytable import PrettyTable
import torchvision.transforms as standard_transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

from mmseg.core import eval_metrics
from .builder import DATASETS  # from .builder import DATASETS
from mmseg.datasets.utils import get_image_change_from_pil


def events_to_voxel_grid(time, x, y, pol, width, height, num_bins, normalize_flag=False):

    assert x.shape == y.shape == pol.shape == time.shape
    assert x.ndim == 1

    voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float, requires_grad=False)
    C, H, W = voxel_grid.shape

    with torch.no_grad():
        voxel_grid = voxel_grid.to(pol.device)
        voxel_grid = voxel_grid.clone()

        t_norm = time
        t_norm = (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

        x0 = x.int()
        y0 = y.int()
        t0 = t_norm.int()

        value = 2 * pol - 1

        for xlim in [x0, x0 + 1]:
            for ylim in [y0, y0 + 1]:
                for tlim in [t0, t0 + 1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < num_bins)
                    interp_weights = value * (1 - (xlim - x).abs()) * (1 - (ylim - y).abs()) * (
                                1 - (tlim - t_norm).abs())

                    index = H * W * tlim.long() + \
                            W * ylim.long() + \
                            xlim.long()

                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        if normalize_flag:
            mask = torch.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean

    return voxel_grid


def tensor_normalize_to_range(tensor, min_val, max_val):
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8) * (max_val - min_val) + min_val
    return tensor


@torch.no_grad()
def events_norm(events, clip_range=1.0, final_range=1.0, enforce_no_events_zero=False):
    # assert clip_range > 0

    if clip_range == 'auto':
        n_mean = events[events < 0].mean() * 1.5  # tensor(-0.7947)
        p_mean = events[events > 0].mean() * 1.5  # tensor(1.2755)
    else:
        nonzero_ev = (events != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = events.sum() / num_nonzeros
            stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero_ev.float()
            events = mask * (events - mean) / (stddev + 1e-8)
        n_mean = -clip_range
        p_mean = clip_range
    '''mask = torch.nonzero(events, as_tuple=True)
    if mask[0].size()[0] > 0:
        mean = events[mask].mean()
        std = events[mask].std()
        if std > 0:
            events[mask] = (events[mask] - mean) / std
        else:
            events[mask] = events[mask] - mean'''

    if enforce_no_events_zero:
        events_smaller_0 = events.detach().clone()
        events[events < 0] = 0
        # events = torch.clamp(events, 0, clip_range)
        events = torch.clamp(events, 0, p_mean)
        events = tensor_normalize_to_range(events, min_val=0, max_val=final_range)
        events_smaller_0[events_smaller_0 > 0] = 0
        # events_smaller_0 = torch.clamp(events_smaller_0, -clip_range, 0)
        events_smaller_0 = torch.clamp(events_smaller_0, n_mean, 0)

        events_smaller_0 = tensor_normalize_to_range(events_smaller_0, min_val=-final_range, max_val=0)
        events += events_smaller_0
    else:
        events = torch.clamp(events, -clip_range, clip_range) * final_range
        events = events / clip_range * final_range
    return events


@DATASETS.register_module()
class DSECDataset(Dataset):
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, dataset_txt_path, events_num=-1, events_bins=5, events_clip_range=None, crop_size=(400, 400),
                 after_crop_resize_size=(512, 512), image_change_range=1, outputs={'events_vg', 'image'}, output_num=1,
                 classes=CLASSES, palette=PALETTE, isr_shift_pixel=4, test_mode=False, events_bins_5_avg_1=False,
                 isr_parms='', isr_type='real_time', enforce_3_channels=True, shift_type='rightdown'):
        self.dataset_txt_path = dataset_txt_path
        self.events_num = events_num
        self.events_bins = events_bins
        self.events_bins_5_avg_1 = events_bins_5_avg_1
        if self.events_bins_5_avg_1:
            assert events_bins == 1
            self.events_bins = 5
            print('self.events_bins: 5-->avg 1')
        self.events_clip_range = events_clip_range
        self.crop_size = (crop_size[1], crop_size[0]) if 'label' not in outputs else crop_size  # (H, W)-->(W, H)
        self.after_crop_resize_size = (after_crop_resize_size[1], after_crop_resize_size[0]) \
            if 'label' not in outputs else after_crop_resize_size  # (H, W)-->(W, H)
        self.image_change_range = image_change_range
        assert self.image_change_range in {1, 2}
        self.outputs = outputs
        self.output_num = output_num
        self.CLASSES, self.PALETTE = classes, palette

        self.dataset_txt = np.loadtxt(self.dataset_txt_path, dtype=str, encoding='utf-8')
        self.events_height = 480
        self.events_width = 640
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        image_transform = [standard_transforms.ToTensor(), standard_transforms.Normalize(*self.mean_std)]
        self.image_transform = standard_transforms.Compose(image_transform)
        self.totensor_transform = standard_transforms.Compose([standard_transforms.ToTensor()])
        self.HorizontalFlip = standard_transforms.RandomHorizontalFlip(p=1)
        self.rectify_events = True
        self.ignore_index = 255
        self.label_map = None
        self.reduce_zero_label = False
        self.isr_shift_pixel = isr_shift_pixel
        self.isr_type = isr_type
        self.enforce_3_channels = enforce_3_channels
        assert self.isr_type in {'raw', 'denoised', 'real_time'}
        # self.image_change_parms = {'val_range': (9, 255 + 9), '_threshold': 0.012, '_clip_range': 0.12, 'shift_pixel': 3}
        self.image_change_parms = {'val_range': (1, 10 ** 2), '_threshold': 0.04, '_clip_range': 0.2, 'shift_pixel': 3}
        self.isr_parms = isr_parms
        if self.isr_parms != '':
            assert isinstance(self.isr_parms, dict)
            self.image_change_parms = self.isr_parms
        self.shift_type = shift_type
        assert self.shift_type in {'all', 'random', 'rightdown'}

    def __len__(self):
        """Total number of samples of data."""
        return self.dataset_txt.shape[0]

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        image_path = self.dataset_txt[idx][0]
        events_h5_path = image_path.replace('images', 'events')[:-20] + 'events.h5'
        sequence_name = image_path.split('/')[-5]
        output = dict()

        if 'label' not in self.outputs:
            flip_flag = True if random.random() < 0.5 else False
            x = random.randint(0, 640 - self.crop_size[0])  # error 680
            # y = random.randint(0, 440 - self.crop_size[1])
            y = random.randint(0, 480 - self.crop_size[1])

        if 'path' in self.outputs:
            output['path'] = image_path

        now_image_index = int(image_path.split('/')[-1].split('.')[0])
        if 'image' in self.outputs:
            image = Image.open(image_path).convert('RGB')
            _resize_size = (960, 720)  # (720, 540)
            image = image.resize(size=_resize_size, resample=Image.BILINEAR)
            image_x = random.randint(0, _resize_size[0] - 512)  # error 680
            image_y = random.randint(0, _resize_size[1] - 512)
            image = image.crop(box=(image_x, image_y, image_x + 512, image_y + 512))
            if flip_flag:
                image = self.HorizontalFlip(image)
            image = self.image_transform(image)
            output['image'] = image

        if 'warp_image' in self.outputs:
            warp_image_name = image_path.replace('images/left/rectified', 'warp_images')
            warp_image_pil = Image.open(warp_image_name).convert('RGB')
            if 'label' not in self.outputs:  # do Data Augmentation
                warp_image_pil = warp_image_pil.crop(box=(x, y, x + self.crop_size[0], y + self.crop_size[1]))
                if flip_flag:
                    warp_image_pil = self.HorizontalFlip(warp_image_pil)
                warp_image_pil = warp_image_pil.resize(size=self.after_crop_resize_size, resample=Image.BILINEAR)
                warp_image = self.image_transform(warp_image_pil)
            else:  # test mode
                warp_image = self.image_transform(warp_image_pil)[:, :440]
            output['warp_image'] = warp_image

        if 'warp_img_self_res' in self.outputs:
            if self.isr_type in {'raw', 'denoised'}:
                if self.isr_type == 'raw':
                    warp_img_self_res_name = image_path.replace('images/left/rectified', 'warp_raw_img_self_res')
                else:
                    warp_img_self_res_name = image_path.replace('images/left/rectified', 'warp_img_self_res')
                warp_img_self_res_pil = Image.open(warp_img_self_res_name).convert('L')
                warp_img_self_res_pil = warp_img_self_res_pil.crop(box=(x, y, x + self.crop_size[0], y + self.crop_size[1]))
                if flip_flag:
                    warp_img_self_res_pil = self.HorizontalFlip(warp_img_self_res_pil)
                warp_img_self_res_pil = warp_img_self_res_pil.resize(size=self.after_crop_resize_size, resample=Image.BILINEAR)
                warp_img_self_res = self.totensor_transform(warp_img_self_res_pil)
                warp_img_self_res = (warp_img_self_res - 0.5) / 0.5
            else:
                if self.shift_type == 'random':
                    direct = [['leftdown', 'leftup'], ['rightdown', 'rightup']]
                    this_shift_direction = direct[x % 2][y % 2]
                else:
                    this_shift_direction = self.shift_type
                warp_img_self_res = get_image_change_from_pil(warp_image_pil, width=warp_image_pil.size[0],
                                                              height=warp_image_pil.size[1],
                                                              shift_direction=this_shift_direction,
                                                              **self.image_change_parms)
            if self.enforce_3_channels and warp_img_self_res.shape[0] == 1:
                warp_img_self_res = warp_img_self_res.repeat(3, 1, 1)
            output['warp_img_self_res'] = warp_img_self_res

        if '19classes' in self.outputs:
            _19classes_name = '{}19classes/{:06d}.png'.format(image_path.split('images/left/rectified')[0],
                                                              now_image_index)
            _19classes = Image.open(_19classes_name)
            _19classes = np.asarray(_19classes, dtype=np.float32)
            _19classes = torch.from_numpy(_19classes)
            _19classes = torch.round(_19classes).long()
            output['19classes'] = _19classes

        if 'label' in self.outputs:
            label_name = '{}labels/{}_{:06d}_grey_gtFine_labelTrainIds.png'.format(
                image_path.split('images/left/rectified')[0],
                sequence_name, now_image_index)
            label = Image.open(label_name)
            label = np.asarray(label, dtype=np.float32)
            label = torch.from_numpy(label)
            label = torch.round(label).long()
            label = label[:440, :]
            output['label'] = label

        if 'events_vg' in self.outputs:
            self.events_h5 = h5py.File(events_h5_path, 'r')
            if self.rectify_events:
                rectify_map_path = image_path.replace('images', 'events')[:-20] + 'rectify_map.h5'
                rectify_map = h5py.File(rectify_map_path, 'r')
                self.rectify_map = np.asarray(rectify_map['rectify_map'])
            images_to_events_index = np.loadtxt(image_path.split('left/rectified')[0] + 'images_to_events_index.txt',
                                                dtype=str, encoding='utf-8')
            events_vg = torch.zeros((self.output_num, self.events_bins, self.events_height, self.events_width))
            for i in range(self.output_num):
                events_finish_index = int(images_to_events_index[now_image_index - i])
                if self.events_num != -1:
                    events_start_index = events_finish_index - self.events_num + 1
                else:
                    events_start_index = int(images_to_events_index[now_image_index - self.image_change_range - i])
                if events_start_index > events_finish_index:
                    return None
                events_vg[self.output_num - 1 - i, :] = self.get_events_vg(events_finish_index, events_start_index)
            if self.events_bins_5_avg_1:
                events_vg = torch.mean(events_vg, dim=1, keepdim=True)
            if self.output_num == 1:
                events_vg = events_vg[0]

            if 'label' not in self.outputs:  # do Data Augmentation
                events_vg = events_vg[:, y: y + self.crop_size[1], x: x + self.crop_size[0]]
                if flip_flag:
                    events_vg = self.HorizontalFlip(events_vg)
                height_weight = (self.after_crop_resize_size[1], self.after_crop_resize_size[0])
                events_vg = F.interpolate(events_vg[None], size=height_weight, mode='bilinear',
                                          align_corners=False)[0]
            else:  # test mode
                events_vg = events_vg[:, :440, :]
            if self.enforce_3_channels:
                events_vg = events_vg.repeat(3, 1, 1)
            output['events_vg'] = events_vg

        if 'img_metas' in self.outputs:
            output['img_metas'] = dict()
            output['img_metas']['img_norm_cfg'] = dict()
            output['img_metas']['img_norm_cfg']['mean'] = [123.675, 116.28, 103.53]
            output['img_metas']['img_norm_cfg']['std'] = [58.395, 57.12, 57.375]
            output['img_metas']['img_norm_cfg']['to_rgb'] = True

            output['img_metas']['img_shape'] = (440, 640)
            output['img_metas']['pad_shape'] = (440, 640)
            output['img_metas']['ori_shape'] = (440, 640)
            output['img_metas']['ori_filename'] = sequence_name + '_' + image_path.split('/')[-1]

            output['img_metas']['flip'] = False
            if output['img_metas']['flip']:
                output['img_metas']['flip_direction'] = 'horizontal'
            output['img_metas'] = DataContainer(output['img_metas'], cpu_only=True)

        return output

    def get_events_vg(self, events_finish_index, events_start_index):
        events_t = np.asarray(self.events_h5['events/{}'.format('t')][events_start_index: events_finish_index + 1])
        events_x = np.asarray(self.events_h5['events/{}'.format('x')][events_start_index: events_finish_index + 1])
        events_y = np.asarray(self.events_h5['events/{}'.format('y')][events_start_index: events_finish_index + 1])
        events_p = np.asarray(self.events_h5['events/{}'.format('p')][events_start_index: events_finish_index + 1])

        events_t = (events_t - events_t[0]).astype('float32')
        events_t = torch.from_numpy((events_t / events_t[-1]))
        events_p = torch.from_numpy(events_p.astype('float32'))
        if self.rectify_events:
            xy_rect = self.rectify_map[events_y, events_x]
            events_x = xy_rect[:, 0]
            events_y = xy_rect[:, 1]
        events_x = torch.from_numpy(events_x.astype('float32'))
        events_y = torch.from_numpy(events_y.astype('float32'))
        events_vg = events_to_voxel_grid(events_t, events_x, events_y, events_p, self.events_width, self.events_height,
                                         num_bins=self.events_bins, normalize_flag=False)

        if self.events_clip_range is not None:
            events_clip_range = random.uniform(self.events_clip_range[0], self.events_clip_range[1])
        else:
            events_clip_range = (events_finish_index - events_start_index) / 500000 * 1.5
            # events_clip_range = 'auto'
        # print('events_clip_range: {}'.format(events_clip_range))
        events_vg = events_norm(events_vg, clip_range=events_clip_range, final_range=1.0, enforce_no_events_zero=True)
        return events_vg

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for idx in range(self.dataset_txt.shape[0]):
            image_path = self.dataset_txt[idx][0]
            now_image_index = int(image_path.split('/')[-1].split('.')[0])
            sequence_name = image_path.split('/')[-5]
            seg_map = '{}labels/{}_{:06d}_grey_gtFine_labelTrainIds.png'.format(
                image_path.split('images/left/rectified')[0],
                sequence_name, now_image_index)
            #  seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')
            gt_seg_map = gt_seg_map[:440, :]
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
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
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
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
            results,  # np.int64, size: (H, W), range: 0~18
            gt_seg_maps,  # np.uint8, size: (H, W), range: 0~18+255
            num_classes,  # 19
            self.ignore_index,  # 255
            metric,  # ['mIoU']
            label_map=self.label_map,  # None
            reduce_zero_label=self.reduce_zero_label)  # False

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
    events_bins_5_avg_1 = False
    if events_bins_5_avg_1:
        events_bins = 1
        events_clip_range = None  # (1.0, 1.0)
    else:
        events_bins = 1
        events_clip_range = None
    dataset = DSECDataset(dataset_txt_path='D:/研究生/Python/Night/DSEC_dataset/night_test_labels_dataset.txt',
                          outputs={'warp_image', 'events_vg', 'label', 'img_metas'},
                          events_bins=events_bins, events_clip_range=events_clip_range,
                          events_bins_5_avg_1=events_bins_5_avg_1)
    data_0 = dataset[20]
    events_vg = data_0['events_vg']
    events_vg = torch.mean(events_vg, dim=0, keepdim=True)
    events_vg = (events_vg + 1) / 2 * 255
    events_vg = events_vg.repeat(3, 1, 1).numpy()
    events_vg = np.uint8(np.transpose(events_vg, (1, 2, 0)))
    events_vg = Image.fromarray(events_vg)
    events_vg.show('1')





