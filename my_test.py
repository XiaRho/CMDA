import os
import multiprocessing


def run_command(command):
    os.system(command)


def main(command_list):
    p = multiprocessing.Pool(processes=len(command_list))
    _ = [p.apply_async(func=run_command, args=(i,)) for i in command_list]
    p.close()
    p.join()


if __name__ == '__main__':
    comman_command = 'python -m tools.test'  # --submit_to_website --image_isr

    command_list_front = ['CUDA_VISIBLE_DEVICES=0',
                          'CUDA_VISIBLE_DEVICES=1',
                          'CUDA_VISIBLE_DEVICES=2',
                          'CUDA_VISIBLE_DEVICES=3',
                          'CUDA_VISIBLE_DEVICES=4',
                          'CUDA_VISIBLE_DEVICES=5',
                          'CUDA_VISIBLE_DEVICES=6',
                          'CUDA_VISIBLE_DEVICES=7']

    work_dirs = ['work_dirs/local-basic/230525_1420_cs2dsec_image+events_together_A01B0005C1_esim_b5_55888',
                 ]

    test_output_type = 'image'
    # python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR} --test_output_type ${TEST_OUTPUT_TYPE}

    command_list = []
    for i in range(8):
        if i >= len(command_list_front) or i >= len(work_dirs):
            break
        config_file = work_dirs[i] + '/' + work_dirs[i].split('/')[-1].split('[')[0] + '.json'
        checkpoint_file = work_dirs[i] + '/latest.pth'
        show_dir = work_dirs[i] + '/preds/'
        now_command = '{} {} {} {} --show-dir {} --test_output_type {} --eval mIoU --opacity 1'.\
            format(command_list_front[i], comman_command, config_file, checkpoint_file, show_dir, test_output_type)
        command_list.append(now_command)
        # print(now_command)
    main(command_list)


