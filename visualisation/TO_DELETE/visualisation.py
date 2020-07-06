# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:50:49 2020

@author: quent
"""


import os
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.loaders import get_dataset
from dataio.transformation import get_dataset_transformation

# from models import get_model
from utils import utils
from utils.utils import json_file_to_pyobj


import torchio


"""
Notes:
shift is working
rotation is working
noise is working

rot + shift takes wrong interpolation for mask
skewing introduces weird artifacts, same with scaling
"""


# todo images need to be in form X,Y,Z
def aug_vis(json_filename):
    # Visualisation arguments
    with_mask = True
    len_x = 5  # number of images on x-axis for vis pdf
    len_y = 5  # number of images on y-axis for vis pdf
    nbr_pages = 5
    total_images = len_x * len_y * nbr_pages  # total number of slices that will be augmented

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    # arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset('gsd_pCT')
    ds_path = json_opts.data.data_dir
    ds_transform = get_dataset_transformation('gsd_pCT', opts=json_opts.augmentation,
                                              max_output_channels=json_opts.model.output_nc)

    # Setup channels
    channels = json_opts.data_opts.channels
    if len(channels) != json_opts.model.input_nc \
            or len(channels) != getattr(json_opts.augmentation, 'gsd_pCT').scale_size[-1]:
        raise Exception(
            'Number of data channels must match number of model channels, and patch and scale size dimensions')

    # Setup Data Loader
    split_opts = json_opts.data_split
    train_dataset = ds_class(ds_path, split='train',      
                             transform=ds_transform['train'], preload_data=train_opts.preloadData,
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed, 
                             channels=channels)

    #train_dataset.selection = train_dataset.selection[:(total_images // 96) * 3]

    train_loader = DataLoader(dataset=train_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=True)

    save_dir = os.path.join('visualisation', json_opts.model.experiment_name)
    utils.rm_and_mkdir(save_dir)

    slices = []
    masks = []

    for epoch_iter, (images, labels, indices) in tqdm(enumerate(train_loader, 1),
                                                      total=len(train_loader)):
        if epoch_iter <= total_images:

            images = images.numpy()
            labels = labels.numpy()

            for image_idx in range(images.shape[0]):
                image = np.squeeze(images[image_idx])
                label = np.squeeze(labels[image_idx])
                for slice in range(image.shape[2]):
                    if not np.max(image[..., slice]) <= 0:
                        slices.append(image[..., slice])
                        masks.append(label[..., slice])

    temp = list(zip(slices, masks))

    random.shuffle(temp)

    slices, masks = zip(*temp)
    list_index = 1
    print(len(slices))
    print(len_x * len_y)
    with PdfPages(save_dir + '/augm_img_vis.pdf') as pdf:
        for page in range(nbr_pages):
            plt.figure()
            plt.figtext(.05, .9, str(json_opts.augmentation.gsd_pCT), fontsize=4)
            idx = 1
            for slice in range(len_x * len_y):
                plt.subplot(len_y, len_x, idx)
                print(list_index)
                plt.imshow(slices[list_index], cmap='gray')
                if with_mask:
                    plt.imshow(masks[list_index], cmap='Blues', alpha=0.4)
                plt.axis('off')
                idx += 1
                list_index += 1
            pdf.savefig()
            plt.close()

    list_index = 1
    with PdfPages(save_dir + '/augm_mask_vis.pdf') as pdf:
        for page in range(nbr_pages):
            plt.figure()
            plt.figtext(.05, .9, str(json_opts.augmentation.gsd_pCT), fontsize=4)
            idx = 1
            for slice in range(len_x * len_y):
                plt.subplot(len_y, len_x, idx)
                plt.imshow(masks[list_index], cmap='gray')
                plt.axis('off')
                idx += 1
                list_index += 1
            pdf.savefig()
            plt.close()





if __name__ == '__main__':
    import argparse
    config_paths = [
        'D:/GitHub/StrokeLesionPredict-BIO503/PerfusionCT-Net/configs/quentin_config_test.json',
    ]
    


    aug_vis(config_paths[0])
