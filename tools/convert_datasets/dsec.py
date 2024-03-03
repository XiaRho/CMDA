# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add class stats computation

import argparse
import json
import os
import os.path as osp

import mmcv
import numpy as np
from PIL import Image


def convert_json_to_label(label_file):
    # label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    # json2labelImg(json_file, label_file, 'trainIds')

    pil_label = Image.open(label_file)
    label = np.asarray(pil_label)
    sample_class_stats = {}
    for c in range(19):
        n = int(np.sum(label == c))
        if n > 0:
            sample_class_stats[int(c)] = n
    sample_class_stats['file'] = label_file
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DSEC annotations to TrainIds')
    parser.add_argument('dsec_path', help='DSEC data path')
    parser.add_argument('--gt-dir', default='19classes', type=str)
    parser.add_argument('--dsec_dataset_txt_path', default='./day_dataset_warp.txt', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w', encoding='utf-8') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w', encoding='utf-8') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w', encoding='utf-8') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    dsec_path = args.dsec_path
    out_dir = args.out_dir if args.out_dir else dsec_path
    mmcv.mkdir_or_exist(out_dir)

    dsec_dataset_txt = np.loadtxt(args.dsec_dataset_txt_path, dtype=str, encoding='utf-8')[:, 0]
    dsec_dataset_label_file = set()

    for image_file in dsec_dataset_txt:
        dsec_dataset_label_file.add(image_file.replace('images/left/rectified', '19classes'))

    png_files = []
    city_name_list = os.listdir(dsec_path)
    for city_name in city_name_list:
        if os.path.isdir(osp.join(dsec_path, city_name)):
            gt_dir = osp.join(dsec_path, city_name, args.gt_dir)
            for png in mmcv.scandir(gt_dir, '.png', recursive=True):
                png_file = osp.join(gt_dir, png)
                if png_file in dsec_dataset_label_file:
                # if os.path.isfile(png_file.replace('19classes', 'warp_images')):
                    png_files.append(png_file)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(convert_json_to_label, png_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_json_to_label, png_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()

