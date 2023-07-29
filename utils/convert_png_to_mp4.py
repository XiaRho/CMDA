import cv2
import os
import argparse
from tqdm import tqdm

input_png_path = '../[work_dirs]/[TestSeg_keep]_230505-2235_17-TestSeg_7lc/seg_images'
video_fps = 30
start_index = 0
end_index = 100
output_path = '../[work_dirs]/28_UDA_seg.avi'
merge_results_path = '../../UDA_events_to_edges/work_dirs/230513_1927_5[2]_no-color-aug_no-fd_85de3/preds/'

parser = argparse.ArgumentParser(description='convert_png_to_mp4')
parser.add_argument('--input_png_path', type=str, default=input_png_path)
parser.add_argument('--video_fps', type=int, default=video_fps)
parser.add_argument('--start_index', type=int, default=start_index)
parser.add_argument('--end_index', type=int, default=end_index)
parser.add_argument('--output_path', type=str, default=output_path)
parser.add_argument('--merge_results_path', type=str, default=merge_results_path)
args = parser.parse_args()

# 读取文件夹中的PNG图像序列
png_folder = args.input_png_path
images = os.listdir(png_folder)
images.sort()
images = [img for img in images if img.endswith(".png")]
images.sort()
frame = cv2.imread(os.path.join(png_folder, images[0]))
height, width, layers = frame.shape

if args.merge_results_path != '':
    merge_results = os.listdir(args.merge_results_path)
    merge_results.sort()
    merge_results = [img for img in merge_results if img.endswith(".png")]

# 创建视频编码器
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video = cv2.VideoWriter(args.output_path, fourcc, args.video_fps, (width, height))

# 将PNG图像序列转换为MP4视频
for index, image in enumerate(tqdm(images[args.start_index: args.end_index])):
    img_path = os.path.join(png_folder, image)
    frame = cv2.imread(img_path)

    if args.merge_results_path != '':
        merge_frame = cv2.imread(os.path.join(args.merge_results_path, merge_results[index]))
        frame[:, 5 * 640: 6 * 640, :] = merge_frame

    video.write(frame)

# 释放资源
cv2.destroyAllWindows()
video.release()
