import os
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='', required=True)
    opt = parser.parse_args()

    comman_command = 'python -m tools.test --submit_to_website'  # --submit_to_website --image_isr

    # work_dirs = ['work_dirs/local-basic/230221_1646_cs2dz_image+raw-isr_SharedD_L07_C01_b5_896f6']

    test_output_type = 'image'
    # python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR} --test_output_type ${TEST_OUTPUT_TYPE}

    config_file = opt.work_dir + '/' + opt.work_dir.split('/')[-1].split('[')[0] + '.json'
    checkpoint_file = opt.work_dir + '/iter_40000.pth'
    show_dir = opt.work_dir + '/preds/'
    now_command = '{} {} {} --show-dir {} --test_output_type {} --eval mIoU --opacity 1'.\
        format(comman_command, config_file, checkpoint_file, show_dir, test_output_type)
    print(now_command)
    os.system(now_command)


