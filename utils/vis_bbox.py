import os
import json
import cv2
from tqdm import tqdm

coda_val_29_classes = ['pedestrian', 'cyclist', 'car', 'truck', 'tricycle', 'bus', 'bicycle',
                       'moped', 'motorcycle', 'stroller', 'cart', 'construction_vehicle', 'dog',
                       'barrier', 'bollard', 'sentry_box', 'traffic_cone', 'traffic_island',
                       'traffic_light', 'traffic_sign', 'debris', 'suitcace', 'dustbin',
                       'concrete_block', 'machinery', 'garbage', 'plastic_bag', 'stone', 'misc']
images_path = 'G:/OOD_Detection/Datasets/CODA/test_images/'
save_bbox_path = 'G:\OOD_Detection\Datasets\CODA/0.3_vis_results/'
annotation_file_path = r'D:\研究生\Python\InternImage\detection\work_dirs\dino_4scale_internimage_l_3x_soda_and_once_' \
                       r'0.1x_backbone_lr_all_aug_coda_val\dino_internimage_coda_val_0.3.json'

images_list = os.listdir(images_path)
images_list.sort()

with open(annotation_file_path) as f:
    annotation_file = json.load(f)

# {'image_id': 1, 'category_id': 2, 'bbox': [962.4969482421875, 378.0677795410156, 26.29827880859375, 69.8837890625], 'score': 0.5454616546630859}
start_ann_index = 0
for image_index in tqdm(range(len(images_list))):
    draw_image = cv2.imread(images_path + images_list[image_index])

    new_start_index = start_ann_index
    for now_ann_index in range(start_ann_index, len(annotation_file)):
        # print(annotation_file[now_ann_index]['image_id'], image_index + 1)
        if annotation_file[now_ann_index]['image_id'] == image_index + 1:
            now_instance_class = coda_val_29_classes[annotation_file[now_ann_index]['category_id'] - 1]
            bbox = annotation_file[now_ann_index]['bbox']
            start = (round(bbox[0]), round(bbox[1]))
            end = (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3]))
            color = (0, 255, 0) if annotation_file[now_ann_index]['category_id'] >= 8 else (255, 0, 0)
            cv2.rectangle(draw_image, start, end, color=color, thickness=2)
            cv2.putText(draw_image, now_instance_class, start, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            new_start_index += 1
        else:
            break
    start_ann_index = new_start_index
    print('1', save_bbox_path + images_list[image_index])
    # 中文路径不能 cv2.imwrite
    cv2.imwrite(save_bbox_path + images_list[image_index], draw_image)
