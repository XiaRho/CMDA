import os
import math
import numpy as np
import hdf5plugin
import h5py
from tqdm import tqdm
import argparse


def create_images_to_events_index(images_timestamps_path, events_h5_path, output_txt_path):
    if os.path.isfile(output_txt_path):
        os.remove(output_txt_path)
    images_to_events_index_list = []
    events_h5 = h5py.File(events_h5_path, 'r')
    events_t = events_h5['events/{}'.format('t')]  # (391111416,)
    t_offset = int(events_h5['t_offset'][()])
    ms_to_idx = np.asarray(events_h5['ms_to_idx'], dtype='int64')  # (59001,)
    images_timestamps = np.loadtxt(images_timestamps_path, dtype='int64')  # (1181,)
    for i in tqdm(range(images_timestamps.shape[0])):
        timestamps_us = images_timestamps[i] - t_offset
        if timestamps_us <= 0 or timestamps_us > events_t[-1]:
            images_to_events_index_list.append(-1)
        else:
            timestamps_ms = math.floor(timestamps_us / 1000)
            timestamps_ms = max(timestamps_ms - 1, 0)
            left_index = ms_to_idx[timestamps_ms]
            '''
            if timestamps_ms + 1 < ms_to_idx.shape[0]:
                right_index = ms_to_idx[timestamps_ms + 1]
            else:
                right_index = events_t.shape[0] - 1
            '''
            right_index = ms_to_idx[timestamps_ms + 2]
            if right_index > events_t.shape[0] - 1:
                right_index = events_t.shape[0] - 1

            if not events_t[left_index] <= timestamps_us <= events_t[right_index]:
                print(events_t[left_index], timestamps_us, events_t[right_index])
                raise ValueError('range error!')
            t_windows = np.asarray(events_t[left_index: right_index + 1], dtype='int64')
            index = np.searchsorted(t_windows, timestamps_us, 'right')
            images_to_events_index_list.append(left_index + index - 1)

    txt_file = open(output_txt_path, 'w', encoding='UTF-8')
    for i in range(images_timestamps.shape[0]):
        txt_file.write(str(images_to_events_index_list[i]) + '\n')
    txt_file.close()


def images_to_events_index_all(dst_path):

    file_list = os.listdir(dst_path)
    file_list.sort()
    for file_name in file_list:
        if 'zurich_city_' not in file_name:
            continue
        print('processing {}...'.format(file_name))
        images_timestamps_path = '{}{}/images/timestamps.txt'.format(dst_path, file_name)
        events_h5_path = '{}{}/events/left/events.h5'.format(dst_path, file_name)
        output_txt_path = '{}{}/images/images_to_events_index.txt'.format(dst_path, file_name)
        create_images_to_events_index(images_timestamps_path, events_h5_path, output_txt_path)


def create_dsec_dataset(dst_path, dataset_txt_path, events_num, image_change_num=1,
                        labels_txt=False, labels_range=None, warp_images_flag=False):

    assert not (labels_txt and labels_range is not None)
    file_list = os.listdir(dst_path)
    file_list.sort()
    if os.path.isfile(dataset_txt_path):
        os.remove(dataset_txt_path)
    dataset_txt = open(dataset_txt_path, 'w', encoding='UTF-8')
    # images_to_events_index_path, events_h5_path, events_num, events_path_txt

    for file_name in file_list:
        if 'zurich_city_' not in file_name:
            continue
        city_name = file_name.split('zurich_city_')[-1]

        if labels_txt:
            if not os.path.isdir('{}{}/labels/'.format(dst_path, file_name)):
                continue
            else:
                labels_index = set()
                labels_name_list = os.listdir('{}{}/labels/'.format(dst_path, file_name))
                for labels_name in labels_name_list:
                    name_index = int(labels_name.split('_')[4])
                    labels_index.add(name_index)

        print('processing {}...'.format(file_name))

        images_to_events_index_path = '{}{}/images/images_to_events_index.txt'.format(dst_path, file_name)
        images_to_events_index = np.loadtxt(images_to_events_index_path, dtype='int64')

        events = h5py.File('{}{}/events/left/events.h5'.format(dst_path, file_name), 'r')
        events_total_num = int(events['events/t'].shape[0])

        images_list_path = '{}{}/images/left/rectified/'.format(dst_path, file_name)
        if not warp_images_flag:
            images_list = os.listdir(images_list_path)
            images_list.sort()

            assert images_to_events_index.shape[0] == len(images_list)
        else:
            images_list = []
            for i in range(images_to_events_index.shape[0]):
                images_list.append('{:06d}.png'.format(i))

        for i, images_name in enumerate(images_list):
            if warp_images_flag:
                if not os.path.isfile('{}{}'.format(images_list_path, images_name).replace('images/left/rectified', 'warp_images')):
                    continue

            if events_num < images_to_events_index[i] and i >= image_change_num:
                images_path = '{}{}'.format(images_list_path, images_name)
                if labels_txt and int(images_name.split('.')[0]) not in labels_index:
                    continue
                if labels_range is not None and city_name in labels_range.keys():
                    if labels_range[city_name][0] <= int(images_name.split('.')[0]) <= labels_range[city_name][1]:
                        continue
                dataset_txt.write(images_path + ' ' + str(images_to_events_index[i]) + '\n')
    dataset_txt.close()


if __name__ == '__main__':
    print('create_dsec_dataset_txt.py')

    # root path of the DSEC_Night dataset
    root_dir = '/path_to_CMDA/CMDA/data/DSEC_Night/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=root_dir)
    opt = parser.parse_args()

    labels_range = {'09_a': (0, 810 - 13), '09_b': (0, 162 - 13), '09_c': (0, 594 - 13),
                    '09_d': (0, 756 - 13), '09_e': (0, 378 - 13)}

    images_to_events_index_all(opt.root_dir)
    create_dsec_dataset(dst_path=opt.root_dir,
                        dataset_txt_path='./night_dataset_warp.txt',
                        events_num=0, labels_txt=False, labels_range=labels_range, image_change_num=2,
                        warp_images_flag=True)

    create_dsec_dataset(dst_path=opt.root_dir,
                        dataset_txt_path='./night_test_dataset_warp.txt',
                        events_num=0, labels_txt=True, labels_range=None, image_change_num=2,
                        warp_images_flag=True)
