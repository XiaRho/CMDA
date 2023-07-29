import os

import numpy as np
import yaml
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as nn
import time
from scipy import interpolate

class WrapPixel(object):
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as file:
            parameter = yaml.load(file.read(), Loader=yaml.Loader)
            events_rect_cam_K = parameter['intrinsics']['camRect0']['camera_matrix']
            self.events_rect_cam_K = np.array([[events_rect_cam_K[0], 0, events_rect_cam_K[2]],
                                               [0, events_rect_cam_K[1], events_rect_cam_K[3]],
                                               [0, 0, 1]])
            images_rect_cam_K = parameter['intrinsics']['camRect1']['camera_matrix']
            self.images_rect_cam_K = np.array([[images_rect_cam_K[0], 0, images_rect_cam_K[2]],
                                               [0, images_rect_cam_K[1], images_rect_cam_K[3]],
                                               [0, 0, 1]])
            T_events_to_images = parameter['extrinsics']['T_10']
            self.T_events_to_images = np.array(T_events_to_images)
            events_disparity_to_depth = parameter['disparity_to_depth']['cams_03']
            self.events_disparity_to_depth = np.array(events_disparity_to_depth)
            events_R_rect = parameter['extrinsics']['R_rect0']
            self.events_R_rect = np.array(events_R_rect)
            self.events_R_rect_inv = np.linalg.inv(self.events_R_rect)
            images_R_rect = parameter['extrinsics']['R_rect1']
            self.images_R_rect = np.array(images_R_rect)

    '''def single_pixel_wrap(self, x, y, z, print_flag=False):
        events_world_coord = np.array([[x], [y], [z / 256.0], [1.0]])
        events_world_coord = np.matmul(self.events_disparity_to_depth, events_world_coord)
        events_world_coord = events_world_coord / events_world_coord[-1, 0]
        events_world_coord[:3, :] = np.matmul(self.events_R_rect_inv, events_world_coord[:3, :])
        if print_flag:
            print('events_world_coord_x: {}, events_world_coord_y: {}, events_world_coord_z: {}'.
                  format(events_world_coord[0, 0], events_world_coord[1, 0], events_world_coord[2, 0]))
        images_world_coord = np.matmul(self.T_events_to_images, events_world_coord)
        if print_flag:
            print('images_world_coord_x: {}, images_world_coord_y: {}, images_world_coord_z: {}'.
                  format(images_world_coord[0, 0], images_world_coord[1, 0], images_world_coord[2, 0]))
        images_world_coord[:3, :] = np.matmul(self.images_R_rect, images_world_coord[:3, :])
        images_world_coord = images_world_coord[:3, :]
        images_cam_coord = np.matmul(self.images_rect_cam_K, images_world_coord / images_world_coord[2, 0])
        if print_flag:
            print('images_cam_x: {}, images_cam_y: {}'.format(images_cam_coord[0, 0], images_cam_coord[1, 0]))
        return round(images_cam_coord[0, 0]), round(images_cam_coord[1, 0])'''

    def all_pixel_wrap(self, events_depth):
        rows, cols = np.meshgrid(np.arange(events_depth.shape[0]), np.arange(events_depth.shape[1]), indexing='ij')
        index_form = np.stack([cols, rows, events_depth], axis=-1).reshape(-1, 3)  # [480x640, 3]  (x, y, z)
        index_form = index_form[index_form[:, 2] != 0, :]  # [valid_num, 3]  (x, y, z)
        index_form = np.double(index_form.transpose(1, 0))  # [3, valid_num]
        index_form[2, :] = index_form[2, :] / 256.0
        index_form = np.concatenate((index_form, np.ones((1, index_form.shape[1]))), axis=0)  # [4, valid_num]
        events_world_coord = np.matmul(self.events_disparity_to_depth, index_form)
        events_world_coord = events_world_coord / events_world_coord[-1, :]
        events_world_coord[:3, :] = np.matmul(self.events_R_rect_inv, events_world_coord[:3, :])
        images_world_coord = np.matmul(self.T_events_to_images, events_world_coord)
        images_world_coord[:3, :] = np.matmul(self.images_R_rect, images_world_coord[:3, :])
        images_world_coord = images_world_coord[:3, :]
        images_cam_coord = np.matmul(self.images_rect_cam_K, images_world_coord / images_world_coord[2, :])
        return index_form[:2, :], images_cam_coord[:2, :]  # [2, valid_num] (x, y)

    def complete_outside_corner_point(self, start_point, direction, step):
        for i in range(40):
            delta_x, delta_y = i * step * direction[0], i * step * direction[1]
            x, y = start_point[0] + delta_x, start_point[1] + delta_y
            if not np.isnan(self.interp_x_coord[y * 640 + x][0]):
                # print(i, interp_x_coord[y * 640 + x], interp_y_coord[y * 640 + x])
                first_x = self.interp_x_coord[y * 640 + x]
                first_y = self.interp_y_coord[y * 640 + x]
                next_delta_x = (i + 1) * step * direction[0]
                next_delta_y = (i + 1) * step * direction[1]
                next_x, next_y = start_point[0] + next_delta_x, start_point[1] + next_delta_y
                second_x = self.interp_x_coord[next_y * 640 + next_x]
                second_y = self.interp_y_coord[next_y * 640 + next_x]
                return np.array([first_x - (second_x - first_x) * i, first_y - (second_y - first_y) * i])

    def create_wrap_image_step_1(self, events_depth, raw_image):
        image_tensor = torch.from_numpy(np.transpose(raw_image, (2, 0, 1)))[None].double()  # [1, 3, 1080, 1440]
        before_wrap, after_wrap = self.all_pixel_wrap(events_depth)  # [2, N]
        wrap_grid = np.zeros((2, 640, 480))
        wrap_grid[:, np.int32(before_wrap[0, :]), np.int32(before_wrap[1, :])] = after_wrap[:, :]
        wrap_grid = wrap_grid.transpose(2, 1, 0)
        wrap_grid = torch.from_numpy(wrap_grid)[None]  # torch.Size([1, 480, 640, 2]) (x, y)
        wrap_grid[:, :, :, 0] = wrap_grid[:, :, :, 0] / (1440 / 2) - 1
        wrap_grid[:, :, :, 1] = wrap_grid[:, :, :, 1] / (1080 / 2) - 1
        wrap_grid = torch.clamp(wrap_grid, min=-1, max=1)
        wrap_image = nn.grid_sample(input=image_tensor, grid=wrap_grid)  # torch.Size([1, 3, 480, 640])
        return wrap_image

    def create_wrap_image_step_2(self, events_depth, raw_image):
        image_tensor = torch.from_numpy(np.transpose(raw_image, (2, 0, 1)))[None].double()  # [1, 3, 1080, 1440]
        before_wrap, after_wrap = self.all_pixel_wrap(events_depth)  # [2, N]
        interp_x = interpolate.LinearNDInterpolator(before_wrap.transpose(1, 0), after_wrap.transpose(1, 0)[:, 0])
        interp_y = interpolate.LinearNDInterpolator(before_wrap.transpose(1, 0), after_wrap.transpose(1, 0)[:, 1])
        rows, cols = np.meshgrid(np.arange(events_depth.shape[0]), np.arange(events_depth.shape[1]), indexing='ij')
        index_form = np.stack([cols, rows], axis=-1).reshape(-1, 2)  # [480x640, 2]
        self.interp_x_coord = np.expand_dims(interp_x(index_form), axis=1)  # [N, 1]
        self.interp_y_coord = np.expand_dims(interp_y(index_form), axis=1)  # [N, 1]
        wrap_grid = np.zeros((2, 640, 480))
        interp_x_y_coord = np.concatenate((self.interp_x_coord, self.interp_y_coord), axis=1).transpose(1, 0)  # [2, N]
        wrap_grid[:, np.int32(index_form[:, 0]), np.int32(index_form[:, 1])] = interp_x_y_coord[:, :]
        wrap_grid = wrap_grid.transpose(2, 1, 0)
        wrap_grid = torch.from_numpy(wrap_grid)[None]  # torch.Size([1, 480, 640, 2]) (x, y)
        # Normalize  0~1440->-1~1  0~1080->-1~1
        wrap_grid[:, :, :, 0] = wrap_grid[:, :, :, 0] / (1440 / 2) - 1
        wrap_grid[:, :, :, 1] = wrap_grid[:, :, :, 1] / (1080 / 2) - 1
        wrap_grid = torch.clamp(wrap_grid, min=-1, max=1)
        wrap_image = nn.grid_sample(input=image_tensor, grid=wrap_grid)  # torch.Size([1, 3, 480, 640])
        return wrap_image

    def create_wrap_image(self, events_depth, raw_image, return_wrap_coord=False):
        # before_wrap, after_wrap = now_sequence_wrap.all_pixel_wrap(events_depth)
        before_wrap, after_wrap = self.all_pixel_wrap(events_depth)  # [2, N]

        interp_x = interpolate.LinearNDInterpolator(before_wrap.transpose(1, 0), after_wrap.transpose(1, 0)[:, 0])
        interp_y = interpolate.LinearNDInterpolator(before_wrap.transpose(1, 0), after_wrap.transpose(1, 0)[:, 1])
        rows, cols = np.meshgrid(np.arange(events_depth.shape[0]), np.arange(events_depth.shape[1]), indexing='ij')
        index_form = np.stack([cols, rows], axis=-1).reshape(-1, 2)  # [480x640, 2]
        self.interp_x_coord = np.expand_dims(interp_x(index_form), axis=1)  # [N, 1]
        self.interp_y_coord = np.expand_dims(interp_y(index_form), axis=1)  # [N, 1]

        left_up_coord = self.complete_outside_corner_point(start_point=(0, 0), direction=(1, 1), step=10)
        left_down_coord = self.complete_outside_corner_point(start_point=(0, 479), direction=(1, -1), step=10)
        right_down_coord = self.complete_outside_corner_point(start_point=(639, 479), direction=(-1, -1), step=10)
        right_up_coord = self.complete_outside_corner_point(start_point=(639, 0), direction=(-1, 1), step=10)

        before_wrap = np.concatenate((before_wrap, np.array([[0], [0]]), np.array([[0], [479]]),
                                      np.array([[639], [479]]), np.array([[639], [0]])), axis=1)
        after_wrap = np.concatenate((after_wrap, left_up_coord, left_down_coord, right_down_coord, right_up_coord), axis=1)

        interp_x = interpolate.LinearNDInterpolator(before_wrap.transpose(1, 0), after_wrap.transpose(1, 0)[:, 0])
        interp_y = interpolate.LinearNDInterpolator(before_wrap.transpose(1, 0), after_wrap.transpose(1, 0)[:, 1])

        interp_x_coord = np.expand_dims(interp_x(index_form), axis=1)  # [N, 1]
        interp_y_coord = np.expand_dims(interp_y(index_form), axis=1)  # [N, 1]
        interp_x_y_coord = np.concatenate((interp_x_coord, interp_y_coord), axis=1).transpose(1, 0)  # [2, N]
        wrap_grid = np.zeros((2, 640, 480))
        wrap_grid[:, np.int32(index_form[:, 0]), np.int32(index_form[:, 1])] = interp_x_y_coord[:, :]
        wrap_grid = wrap_grid.transpose(2, 1, 0)
        wrap_grid = torch.from_numpy(wrap_grid)[None]  # torch.Size([1, 480, 640, 2]) (x, y)
        # Normalize  0~1440->-1~1  0~1080->-1~1
        wrap_grid[:, :, :, 0] = wrap_grid[:, :, :, 0] / (1440 / 2) - 1
        wrap_grid[:, :, :, 1] = wrap_grid[:, :, :, 1] / (1080 / 2) - 1
        wrap_grid = torch.clamp(wrap_grid, min=-1, max=1)
        if return_wrap_coord:
            return wrap_grid
        image_tensor = torch.from_numpy(np.transpose(raw_image, (2, 0, 1)))[None].double()  # [1, 3, 1080, 1440]
        wrap_image = nn.grid_sample(input=image_tensor, grid=wrap_grid)  # torch.Size([1, 3, 480, 640])
        return wrap_image

def wrap_sequence_images(save_path, cam_to_cam_path, disparity_event_path, image_path,
                         step=3, save_wrap_coord=False):
    if save_wrap_coord:
        assert step == 3
    os.makedirs(save_path) if not os.path.isdir(save_path) else None
    now_sequence_wrap = WrapPixel(yaml_path=cam_to_cam_path)
    events_depth_names = os.listdir(disparity_event_path)
    events_depth_names.sort()
    for events_depth_name in tqdm(events_depth_names, desc=" sequence", position=1, leave=False):
        if os.path.isfile(save_path + events_depth_name) and (not save_wrap_coord):
            continue
        if os.path.isfile(save_path + events_depth_name.replace('png', 'pth')) and save_wrap_coord:
            continue
        events_depth = Image.open(disparity_event_path + events_depth_name)
        events_depth = np.array(events_depth)
        if not save_wrap_coord:
            image_pil = Image.open(image_path + events_depth_name)
            image_np = np.array(image_pil)
        else:
            image_np = None
        if step == 1:
            wrap_image = now_sequence_wrap.create_wrap_image_step_1(events_depth, image_np)
        elif step == 2:
            wrap_image = now_sequence_wrap.create_wrap_image_step_2(events_depth, image_np)
        elif step == 3:
            wrap_image = now_sequence_wrap.create_wrap_image(events_depth, image_np, save_wrap_coord)
        if save_wrap_coord:
            wrap_image = wrap_image.to(torch.float16)
            torch.save(wrap_image, save_path + events_depth_name.replace('png', 'pth'))
        else:
            wrap_image = np.uint8(wrap_image[0].numpy().transpose(1, 2, 0))
            wrap_image = Image.fromarray(wrap_image)
            wrap_image.save(save_path + events_depth_name)

def wrap_images_from_wrap_coord(wrap_grid_path, raw_image_path, save_path):
    wrap_grid = torch.load(wrap_grid_path)
    wrap_grid = wrap_grid.to(torch.float32)
    raw_image = Image.open(raw_image_path)
    raw_image = np.array(raw_image)
    image_tensor = torch.from_numpy(np.transpose(raw_image, (2, 0, 1)))[None].to(torch.float32)
    wrap_image = nn.grid_sample(input=image_tensor, grid=wrap_grid)
    wrap_image = np.uint8(wrap_image[0].numpy().transpose(1, 2, 0))
    wrap_image = Image.fromarray(wrap_image)
    wrap_image.save(save_path)

# root_path = 'D:/研究生/Python/Night/DSEC_dataset/Day/zurich_city_02_a/'
# save_path = root_path + 'wrap_coord/'
# cam_to_cam_path = root_path + 'cam_to_cam.yaml'
# disparity_event_path = root_path + 'disparity_event/'
# image_path = root_path + 'images/left/rectified/'
#
# wrap_sequence_images(save_path, cam_to_cam_path, disparity_event_path, image_path, save_wrap_coord=True)


if __name__ == '__main__':
    '''
    create wrap_coord
    '''
    # root_path = './'
    # save_path = root_path + 'train_wrap_coord/'
    # disparity_path = root_path + 'train_disparity/'
    # calibration_path = root_path + 'train_calibration/'
    #
    # city_names = os.listdir(disparity_path)
    # city_names.sort()
    # for city_name in tqdm(city_names, desc=" city", position=0):
    #     wrap_sequence_images(save_path='{}{}/'.format(save_path, city_name),
    #                          cam_to_cam_path='{}{}/calibration/cam_to_cam.yaml'.format(calibration_path, city_name),
    #                          disparity_event_path='{}{}/disparity/event/'.format(disparity_path, city_name),
    #                          image_path=None,
    #                          save_wrap_coord=True)

    '''
    wrap_coord to wrap_images
    '''
    # root_path = 'G:/DSEC_Dataset/'
    # save_path = root_path + 'train_wrap_images/'
    # wrap_coord_path = root_path + 'train_wrap_coord/'
    # images_path = root_path + 'train_images/'
    #
    # city_names = os.listdir(wrap_coord_path)
    # city_names.sort()
    # for city_name in tqdm(city_names, desc=" city", position=0):
    #     save_city_path = save_path + city_name + '/'
    #     wrap_coord_city_path = wrap_coord_path + city_name + '/'
    #     images_city_path = images_path + city_name + '/images/left/rectified/'
    #
    #     os.makedirs(save_city_path) if not os.path.isdir(save_city_path) else None
    #     images_names = os.listdir(wrap_coord_city_path)
    #     images_names.sort()
    #     for images_name in tqdm(images_names, desc=" sequence", position=1, leave=False):
    #         if os.path.isfile(save_city_path + images_name):
    #             continue
    #         wrap_images_from_wrap_coord(wrap_grid_path=wrap_coord_city_path + images_name,
    #                                     raw_image_path=images_city_path + images_name.replace('pth', 'png'),
    #                                     save_path=save_city_path + images_name.replace('pth', 'png'))
