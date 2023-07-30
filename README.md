## CMDA: Cross-Modality Domain Adaptation for Nighttime Semantic Segmentation

**by Ruihao Xia, Chaoqiang Zhao, Meng Zheng, Ziyan Wu, Qiyu Sun, and Yang Tang**

**[[Arxiv]](https://arxiv.org/abs/xxxx.xxxxx)**
**[[Paper]](https://arxiv.org/pdf/xxxx.xxxxx.pdf)**

:bell: We are happy to announce that CMDA was accepted at **ICCV2023**. :bell:


## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
conda create -n CMDA python=3.7
conda activate CMDA
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Further, please download the pretrained backbone `mit_b5.pth` 
[here](https://drive.google.com/file/d/1TwUh8H9flg-zUHZmq7vu-FtyaSMrf9oq/view?usp=sharing), 
style transfer network `cityscapes_ICD_to_dsec_EN.pth` 
[here](https://drive.google.com/file/d/10ZG_fiCvfnhNNppSdPtQhUL9XPTBSIEF/view?usp=sharing) 
and put them in a folder `pretrained/` within this project.

## Setup Datasets

**Cityscapes:** 

① Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

② Please, download `leftImg8bit_IC1` from [here](https://drive.google.com/file/d/19474kcmbyz8WRBBez29MOINeQT1yMZyZ/view?usp=sharing)
and extract them to `data/cityscapes`.

③ Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS.

```shell
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

**DSEC Night_Semantic:** 

① Please, download the events_left.zip and images_rectified_left.zip in `zurich_city_09_a~09_e` from
[here](https://dsec.ifi.uzh.ch/dsec-datasets/download/) and extract it to `data/DSEC_Night/events` and 
`data/DSEC_Night/images`.

② Please, download the `labels` and `warp_images` folders from
[here](https://drive.google.com/file/d/1oBX44pEKSrD6OZ0bDrU7FVYIZ_3XWi0-/view?usp=sharing) and extract it to `data/DSEC_Night`.

③ Finally, run the following scripts to generate night_dataset_wrap.txt 
and night_test_dataset_wrap.txt for DSEC DataLoader.

```shell
python create_dsec_dataset_txt.py --root_dir /path_to_CMDA/CMDA/data/DSEC_Night/
```

**DarkZurich (Optional):** 

Please, download the Dark_Zurich_train_anon.zip  and Dark_Zurich_val_anon.zip from
[here](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) and extract it
to `data/dark_zurich`.

```shell
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

The data folder structure should look like this:

```none
CMDA
├── ...
├── data
│   ├── cityscapes
│   │   ├── gtFine
│   │   │   ├── ...
│   │   ├── leftImg8bit
│   │   │   ├── ...
│   │   ├── leftImg8bit_IC1
│   │   │   ├── ...
│   │   ├── sample_class_stats_dict.json
│   │   ├── sample_class_stats.json
│   │   ├── samples_with_class.json
│   ├── DSEC_Night
│   │   ├── zurich_city_09_a
│   │   │   ├── events
│   │   │   │   ├── left
│   │   │   │   │   ├── events.h5
│   │   │   │   │   ├── rectify_map.h5
│   │   │   ├── images
│   │   │   │   ├── left
│   │   │   │   │   ├── rectified
│   │   │   │   │   │   ├── xxxxxx.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── exposure_timestamps_left.txt
│   │   │   │   ├── images_to_events_index.txt
│   │   │   │   ├── timestamps.txt
│   │   │   ├── labels
│   │   │   │   ├── zurich_city_09_x_xxxxxx_grey_gtFine_labelTrainIds.png
│   │   │   │   ├── ...
│   │   │   ├── warp_images
│   │   │   │   ├── xxxxxx.png
│   │   │   │   ├── ...
│   │   ├── zurich_city_09_b~e
│   │   │   ├── ...
│   ├── dark_zurich
│   │   ├── gt
│   │   │   ├── ...
│   │   ├── rgb_anon
│   │   │   ├── ...
├── ...
```

## Training

**Cityscapes→DSEC_Night:** 

```shell
python my_run_experiments.py --root_path /path_to_CMDA/CMDA/ --base_config configs/fusion/cs2dsec_image+events_together_b5.py --name cmda_cs2dsec
```

**Cityscapes→DarkZurich (Optional):** 

```shell
python my_run_experiments.py --root_path /path_to_CMDA/CMDA/ --base_config configs/fusion/cs2dz_image+raw-isr_b5.py --name cmda_cs2dz
```

## Testing & Predictions

**Cityscapes→DSEC_Night:** 

Testing & predictions are already done after 
training and do not require additional steps.

**Cityscapes→DarkZurich (Optional):** 

The CMDA checkpoint trained on Cityscapes→DarkZurich can be tested 
on the DarkZurich testset using:

```shell
python my_test.py --work_dir work_dirs/local-basic/230221_1646_cs2dz_image+raw-isr_SharedD_L07_C01_b5_896f6
```

The predictions can be submitted to the public evaluation server of the
respective dataset to obtain the test score.

## Checkpoints

* [CMDA for Cityscapes→DSEC](https://drive.google.com/file/d/1pG3kDClZDGwp1vSTEXmTchkGHmnLQNdP/view?usp=sharing)
* [CMDA for Cityscapes→DarkZurich](https://drive.google.com/file/d/1V9EpoTePjGq33B8MfombxEEcq9a2rBEt/view?usp=sharing)

The checkpoints come with the training logs. Please note that:

* The logs provide the mIoU for 19 classes. For Cityscapes→DSEC, it is
  necessary to convert the mIoU to the 18 valid classes, i.e., the final mIoU
  56.89 should be converted to 56.89*19/18=60.05.

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [DAFormer](https://github.com/lhoyer/DAFormer)
* [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
