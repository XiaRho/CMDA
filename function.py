import argparse
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, choices=['rename_work_dirs', 'convert_pth'], default='rename_work_dirs')
    parser.add_argument('--pth_path', type=str)
    # python function.py --function rename_work_dirs
    # python function.py --function convert_pth --pth_path ./work_dirs/local-basic/230120_2239_cs2dz_image+raw-isr_Attf_NMiTMi_HShareD_b5_b8f48[42.13]/
    args = parser.parse_args()

    if args.function == 'rename_work_dirs':
        root_path = './work_dirs/local-basic/'
        all_work_dirs = os.listdir(root_path)
        all_work_dirs.sort()
        for work_dir in tqdm(all_work_dirs):
            if '[' in work_dir and ']' in work_dir:
                continue
            if not os.path.exists(root_path + work_dir + '/test_results/'):
                continue
            test_results = os.listdir(root_path + work_dir + '/test_results/')
            test_results.sort()
            if test_results[-1][:6] == '40000_':
                final_miou = test_results[-1][6:]
                os.rename(src=root_path + work_dir, dst='{}{}[{}]'.format(root_path, work_dir, final_miou))
                # print(root_path + work_dir, '-->', '{}{}[{}]'.format(root_path, work_dir, final_miou))
    elif args.function == 'convert_pth':
        import torch
        pth_path = args.pth_path + 'iter_40000.pth'
        pth = torch.load(pth_path)  # ['meta', 'state_dict', 'optimizer']
        pth_state_dict = pth['state_dict']
        new_state_dict = dict()
        for key in pth_state_dict.keys():
            if 'ema_model' in key or 'cyclegan' in key:
                continue
            new_state_dict[key] = pth_state_dict[key]
        torch.save(new_state_dict, pth_path.split('.pth')[0] + '_state_dict.pth')
