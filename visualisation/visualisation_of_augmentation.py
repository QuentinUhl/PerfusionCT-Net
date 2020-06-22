# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:50:49 2020

@author: quentin uhl
"""
PerfusionCT_Net_dir = 'D:/GitHub/StrokeLesionPredict-BIO503/PerfusionCT-Net'

import os
#import random

# If this file is in a visualisation folder
import sys
sys.path.insert(1, PerfusionCT_Net_dir)

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




"""
Notes:
shift is working
rotation is working
noise is working

rot + shift takes wrong interpolation for mask
skewing introduces weird artifacts, same with scaling
"""


# todo images need to be in form X,Y,Z
def visualisation_of_augmentations(arguments):
    
    # Parse input arguments
    if arguments.config[:5] == 'grid_':
        json_filename = PerfusionCT_Net_dir + '/configs/Visualisation/'+str(arguments.config)+'.json'
    else:
        json_filename = arguments.config

    # Dataset option to have no transformation
    normal_json_opts = json_file_to_pyobj(PerfusionCT_Net_dir + '/configs/Visualisation/grid_normal.json')
    #network_debug = arguments.debug
    
    # Visualisation arguments
    with_mask = True
    len_x = 4  # number of images on x-axis for vis pdf
    len_pairs_y = 3
    len_y = 2 * len_pairs_y  # must be even, number of images on y-axis for vis pdf
    # nbr_pages = 10
    # total_images = len_x * len_y * nbr_pages  # total number of slices that will be augmented

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Set experiment name
    exp_name = json_opts.model.experiment_name

    # Setup Dataset and Augmentation
    ds_class = get_dataset('gsd_pCT')
    ds_path = json_opts.data.data_dir
    
    ds_original = get_dataset_transformation('gsd_pCT', opts=normal_json_opts.augmentation,
                                          max_output_channels=json_opts.model.output_nc)
    ds_transform = get_dataset_transformation('gsd_pCT', opts=json_opts.augmentation,
                                              max_output_channels=json_opts.model.output_nc)


    # Setup channels
    channels = json_opts.data_opts.channels
    if len(channels) != json_opts.model.input_nc \
            or len(channels) != getattr(json_opts.augmentation, 'gsd_pCT').scale_size[-1]:
        raise Exception(
            'Number of data channels must match number of model channels, and patch and scale size dimensions')

    # Setup Data Loaders
    split_opts = json_opts.data_split
    
    initial_dataset = ds_class(ds_path, split='train',      
                              transform=ds_original['train'], # no transformation
                              preload_data=train_opts.preloadData,
                              train_size=split_opts.train_size, test_size=split_opts.test_size,
                              valid_size=split_opts.validation_size, split_seed=split_opts.seed, 
                              channels=channels)
    train_dataset = ds_class(ds_path, split='train',      
                             transform=ds_transform['train'], # studied transformation
                             preload_data=train_opts.preloadData,
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed, 
                             channels=channels)
    
    # train_dataset.selection = train_dataset.selection[:(total_images // 96) * 3]

    initial_loader = DataLoader(dataset=initial_dataset, num_workers=0, batch_size=train_opts.batchSize)
    train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=train_opts.batchSize)

    save_dir = os.path.join('visualisation', json_opts.model.experiment_name)
    utils.rm_and_mkdir(save_dir)

    slice_type = ['sagittal','coronal','horizontal']
    
    def correct_images(image_to_correct):
        # return image_to_correct
        return np.rot90(image_to_correct.clip(0))

    for slice_type_index in reversed(range(3)):
    
        slices = []
        masks = []
        patient_numbers=[]
        
        initial_iter = iter(initial_loader)
        
        for epoch_iter, (images, labels, indices) in tqdm(enumerate(train_loader, 1),
                                                          total=len(train_loader)):
            
            (inital_images, inital_labels, inital_indices) = next(initial_iter)
            patient_current_index = 0
            inital_images = inital_images.numpy()
            inital_labels = inital_labels.numpy()
            images = images.numpy()
            labels = labels.numpy()
            print("\n")
            # print("initial shape :",inital_images.shape)
            # print("images shape :",images.shape)
            
            for image_idx in range(images.shape[0]):
                patient_current_index += 1
                inital_image = np.squeeze(inital_images[image_idx])
                inital_label = np.squeeze(inital_labels[image_idx])
                image = np.squeeze(images[image_idx])
                label = np.squeeze(labels[image_idx])
                # Sagittal slices
                if slice_type_index==0:
                    print("Printing Sagittal Slices for patient "+str(patient_current_index))
                    for slice in range(image.shape[0]):
                        # if not np.max(inital_image[slice,:,:]) <= 0:
                        slices.append(correct_images(inital_image[slice,:,:]))
                        masks.append(correct_images(inital_label[slice,:,:]))
                        patient_numbers.append((patient_current_index, slice))
                        slices.append(correct_images(image[slice,:,:]))
                        masks.append(correct_images(label[slice,:,:]))
                        patient_numbers.append((patient_current_index, slice))
                # Coronal slices
                if slice_type_index==1:
                    print("Printing Coronal Slices for patient "+str(patient_current_index))
                    for slice in range(image.shape[1]):
                        # if not np.max(inital_image[:,slice,:]) <= 0:
                        slices.append(correct_images(inital_image[:,slice,:]))
                        masks.append(correct_images(inital_label[:,slice,:]))
                        patient_numbers.append((patient_current_index, slice))
                        slices.append(correct_images(image[:,slice,:]))
                        masks.append(correct_images(label[:,slice,:]))
                        patient_numbers.append((patient_current_index, slice))
                # Horizontal slices
                if slice_type_index==2:
                    print("Printing Horizontal Slices for patient "+str(patient_current_index))
                    for slice in range(image.shape[2]):
                        # if not np.max(inital_image[:,:, slice]) <= 0:
                        # print("initial image max : ",np.max(inital_image[:,:, slice]), " / initial image min :",np.min(inital_image[:,:, slice]))
                        # print("image max :",np.max(image[:,:, slice]), " / image min :",np.min(image[:,:, slice]))
                        slices.append(correct_images(inital_image[:,:, slice]))
                        masks.append(correct_images(inital_label[:,:, slice]))
                        patient_numbers.append((patient_current_index, slice))
                        slices.append(correct_images(image[:,:, slice]))
                        masks.append(correct_images(label[:,:, slice]))
                        patient_numbers.append((patient_current_index, slice))
        
        temp = list(zip(slices, masks))
        slices, masks = zip(*temp)
        
        # Create new order to display images
        tmp_order_by_pairs = np.arange(0, len_x*len_pairs_y).reshape((len_pairs_y,len_x)).transpose().flatten()
        tmp_order = np.zeros(len_x*len_y, dtype=int)
        tmp_order[0::2] = 2*tmp_order_by_pairs
        tmp_order[1::2] = 2*tmp_order_by_pairs +1
        new_order = np.zeros(len_x*len_y)
        for i in range(len_x*len_y):
            new_order[tmp_order[i]] = i
            
        def row_based_idx(new_order, idx):
            return new_order[idx-1]+1
        
        # Compute number of pages required to plot everything
        nbr_pages = len(slices)//(len_x * len_y) + (len(slices)%(len_x * len_y) != 0)
        total_images = len_x * len_y * nbr_pages  # total number of slices that will be augmented
        
        list_index = 0
        with PdfPages(save_dir + '/img_vis_'+exp_name+'_'+slice_type[slice_type_index]+'.pdf') as pdf:
            for page in range(nbr_pages):
                plt.figure()
                plt.figtext(.05, .9, str(json_opts.augmentation.gsd_pCT), fontsize=4)
                idx = 1
                for slice in range(len_x * len_y):
                    if list_index<len(slices):
                        plt.subplot(len_x, len_y, row_based_idx(new_order, idx))
                        plt.imshow(slices[list_index], cmap='gray')
                        if with_mask:
                            plt.imshow(masks[list_index], cmap='Reds', alpha=0.5)
                        plt.axis('off')
                        if idx%2==1:
                            plt.title("P"+str(patient_numbers[list_index][0])+" initial "+str(patient_numbers[list_index][1]), fontsize=5)
                        else:
                            plt.title("P"+str(patient_numbers[list_index][0])+"  "+str(patient_numbers[list_index][1]), fontsize=5)
                        idx += 1
                        list_index += 1
                pdf.savefig()
                plt.close()
    
        list_index = 0
        with PdfPages(save_dir + '/mask_vis_'+exp_name+'_'+slice_type[slice_type_index]+'.pdf') as pdf:
            for page in range(nbr_pages):
                plt.figure()
                plt.figtext(.05, .9, str(json_opts.augmentation.gsd_pCT), fontsize=4)
                idx = 1
                for slice in range(len_x * len_y):
                    if list_index<len(slices):
                        plt.subplot(len_x, len_y, row_based_idx(new_order, idx))
                        plt.imshow(masks[list_index], cmap='gray')
                        plt.axis('off')
                        idx += 1
                        list_index += 1
                pdf.savefig()
                plt.close()
        
        # Print end message
        print("\n"+str(total_images)+" images have been augmented with "+exp_name)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Unet Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    visualisation_of_augmentations(args)