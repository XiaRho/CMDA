import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import random
from mmseg.models.cyclegan import define_G

def tensor_normalize_to_range(tensor, min_val, max_val):
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8) * (max_val - min_val) + min_val
    return tensor


def get_image_change(image_now, image_front):
    image_front = np.asarray(image_front, dtype=np.float32)
    image_front = np.log(image_front + log_add)
    image_now = np.asarray(image_now, dtype=np.float32)
    image_now = np.log(image_now + log_add)
    image_change = torch.unsqueeze(torch.from_numpy(image_now - image_front), dim=0)
    mask = (torch.abs(image_change) <= threshold)
    image_change[mask] = 0
    image_change_smaller_0 = image_change.detach().clone()
    image_change[image_change < 0] = 0
    image_change = torch.clamp(image_change, 0, clip_range)
    image_change = tensor_normalize_to_range(image_change, min_val=0, max_val=1)
    image_change_smaller_0[image_change_smaller_0 > 0] = 0
    image_change_smaller_0 = torch.clamp(image_change_smaller_0, -clip_range, 0)
    image_change_smaller_0 = tensor_normalize_to_range(image_change_smaller_0, min_val=-1, max_val=0)
    image_change += image_change_smaller_0
    image_change = image_change[0, :].numpy()
    image_change = np.uint8(np.around((image_change + 1) / 2 * 255))
    image_change = Image.fromarray(image_change, mode='L')
    return image_change


def create_cityscapes_image_change(src_path, src_path_aux, dst_path):
    assert 'leftImg8bit_sequence' in src_path
    os.makedirs(dst_path) if not os.path.isdir(dst_path) else None

    city_names = os.listdir(src_path)
    city_names.sort()
    for city_name in tqdm(city_names):
        src_city_path_aux = '{}{}/'.format(src_path_aux, city_name)
        src_city_path = '{}{}/'.format(src_path, city_name)
        dst_city_path = '{}{}/'.format(dst_path, city_name)
        os.makedirs(dst_city_path) if not os.path.isdir(dst_city_path) else None
        image_names = os.listdir(src_city_path_aux)
        image_names.sort()
        for image_name in tqdm(image_names, leave=False):
            if image_name != 'aachen_000082_000019_leftImg8bit.png':
                continue
            index = int(image_name.split('_')[2])
            image_front_name = image_name[: -22] + '{:06d}_leftImg8bit.png'.format(index - image_change_range)

            image_change_name = image_name

            if os.path.isfile(dst_city_path + image_change_name):
                continue

            image_front = Image.open(src_city_path + image_front_name).convert('L')
            image_now = Image.open(src_city_path_aux + image_name).convert('L')
            image_change = get_image_change(image_now, image_front)

            image_change.save(dst_city_path + image_change_name)


def create_cityscapes_icd_for_cyclegan(src_path, dst_path, resize_size=(1024, 512),
                                       crop_size=(640, 480), create_num=1000):
    assert crop_size[0] <= resize_size[0] and crop_size[1] <= resize_size[1]
    os.makedirs(dst_path) if not os.path.isdir(dst_path) else None
    city_names = os.listdir(src_path)
    images_path_list = []
    for city_name in city_names:
        images_name_list = os.listdir(src_path + city_name)
        for image_name in images_name_list:
            image_path = '{}{}/{}'.format(src_path, city_name, image_name)
            images_path_list.append(image_path)
    random.shuffle(images_path_list)

    for i, image_path in tqdm(enumerate(images_path_list[:create_num])):
        image = Image.open(image_path).convert('L')
        image = image.resize(size=resize_size, resample=Image.BILINEAR)
        x = random.randint(0, resize_size[0] - crop_size[0])
        y = random.randint(0, resize_size[1] - crop_size[1])
        image = image.crop(box=(x, y, x + crop_size[0], y + crop_size[1]))  # (left, upper, right, lower)
        image_name = '{:04d}.png'.format(i)
        image.save(dst_path + image_name)


def create_cityscapes_id_for_cyclegan(src_path, dst_path, resize_size=(1024, 512),
                                      crop_size=(640, 480), create_num=1000):
    assert crop_size[0] <= resize_size[0] and crop_size[1] <= resize_size[1]
    os.makedirs(dst_path) if not os.path.isdir(dst_path) else None
    city_names = os.listdir(src_path)
    images_path_list = []
    for city_name in city_names:
        images_name_list = os.listdir(src_path + city_name)
        for image_name in images_name_list:
            image_path = '{}{}/{}'.format(src_path, city_name, image_name)
            images_path_list.append(image_path)
    random.shuffle(images_path_list)

    for i, image_path in tqdm(enumerate(images_path_list[:create_num])):
        image = Image.open(image_path).convert('L')
        image = image.resize(size=resize_size, resample=Image.BILINEAR)
        x = random.randint(0, resize_size[0] - crop_size[0])
        y = random.randint(0, resize_size[1] - crop_size[1])
        image = image.crop(box=(x, y, x + crop_size[0], y + crop_size[1]))  # (left, upper, right, lower)
        image_name = '{:04d}.png'.format(i)
        image.save(dst_path + image_name)


def check_cityscapes_icd_file(src_path):
    error_file = []
    city_names = os.listdir(src_path)
    city_names.sort()
    for city_name in tqdm(city_names):
        src_city_path = '{}{}/'.format(src_path, city_name)
        image_names = os.listdir(src_city_path)
        image_names.sort()
        for image_name in image_names:
            try:
                image = Image.open(src_city_path + image_name).convert('L')
            except OSError:
                error_file.append(image_name)
    print(error_file)


def create_cityscapes_en_for_train(src_path, dst_path, pretrained_cyclegan_model):

    cyclegan_model = define_G().cuda()
    cyclegan_model_pth = torch.load(pretrained_cyclegan_model)
    cyclegan_model.load_state_dict(cyclegan_model_pth)
    cyclegan_model.eval()

    os.makedirs(dst_path) if not os.path.isdir(dst_path) else None
    city_names = os.listdir(src_path)
    city_names.sort()
    for city_name in tqdm(city_names):
        src_city_path = '{}{}/'.format(src_path, city_name)
        dst_city_path = '{}{}/'.format(dst_path, city_name)
        os.makedirs(dst_city_path) if not os.path.isdir(dst_city_path) else None
        image_change_names = os.listdir(src_city_path)
        image_change_names.sort()
        for image_change_name in tqdm(image_change_names, leave=False):

            if os.path.isfile(dst_city_path + image_change_name):
                continue
            image_change = Image.open(src_city_path + image_change_name).convert('L')
            image_change = image_change.resize(size=(1024, 512), resample=Image.BILINEAR)
            image_change = np.asarray(image_change, dtype=np.float32)
            image_change = torch.from_numpy(image_change)[None][None].cuda()
            image_change = (image_change / 255 - 0.5) * 2
            # print('image_change: ', torch.max(image_change), torch.min(image_change))
            with torch.no_grad():
                events_night = cyclegan_model(image_change)[0, 0]
            events_night = events_night.cpu().numpy()
            events_night = np.uint8((events_night + 1) / 2 * 255)
            # print('events_night', events_night.shape, np.max(events_night), np.min(events_night))
            events_night = Image.fromarray(events_night).convert('L')
            events_night.save(dst_city_path + image_change_name)


if __name__ == '__main__':
    print('create_cityscapes_image_change.py')

    image_change_range = 1
    log_add = 50
    threshold = 0.1
    clip_range = 0.8
    root_dir = '/home/ubuntu/XRH/city-scapes-script-master/cityscapes_dataset/'

    # create_cityscapes_image_change(src_path=root_dir + 'leftImg8bit_sequence/train/',
    #                                src_path_aux=root_dir + 'leftImg8bit/train/',
    #                                dst_path=root_dir + 'leftImg8bit_IC1/train/')

    # create_cityscapes_icd_for_cyclegan(src_path='D:/研究生/Python/HANet/cityscapes/leftImg8bit_IC1/train/',
    #                                    dst_path='D:/研究生/Python/pytorch-CycleGAN-and-pix2pix-master/datasets/ICD2EN_dataset_cs/trainA/')

    # create_cityscapes_icd_for_cyclegan(src_path='D:/研究生/Python/HANet/cityscapes/leftImg8bit/train/',
    #                                    dst_path='D:/研究生/Python/pytorch-CycleGAN-and-pix2pix-master/datasets/ID2EN_dataset_cs/trainA/')

    # check_cityscapes_icd_file(src_path='D:/研究生/Python/HANet/cityscapes/leftImg8bit_IC1/train/')
    # check_cityscapes_icd_file(src_path=root_dir + 'leftImg8bit_IC1/train/')

    create_cityscapes_en_for_train(src_path='/home/ubuntu/XRH/city-scapes-script-master/cityscapes_dataset/leftImg8bit_IC1/train/',
                                   dst_path='/home/ubuntu/XRH/city-scapes-script-master/cityscapes_dataset/leftImg8bit_EN1/train/',
                                   pretrained_cyclegan_model='/home/ubuntu/XRH/Events_DAFormer/pretrained/cityscapes_ICD_to_dsec_EN.pth')