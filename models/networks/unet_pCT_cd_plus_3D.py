import numpy as np
import torch.nn as nn
import torch
from .utils import UnetConv3, UnetUp3_CT, UnetGridGatingSignal3, UnetDsv3
import torch.nn.functional as F
from models.networks_other import init_weights
from models.layers.grid_attention_layer import GridAttentionBlock3D


class unet_pCT_cd_plus_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=2, is_deconv=True, in_channels=4,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True, cd_size=0):
        super(unet_pCT_cd_plus_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.cd_size = cd_size

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)
        
        # add clinical data
        self.fc1 = nn.Linear(self.cd_size,64) #256*6*6*6
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64,512) #256*6*6*6
        self.drop2 = nn.Dropout(p=0.5)
        self.convmed = nn.Conv3d(in_channels=1, out_channels=filters[4], kernel_size=1)
        self.relu = nn.ReLU()
        self.attentionblockmed = MultiAttentionBlock(in_size=filters[4], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        # Recording Activity
        self.med_count = 0

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(n_classes*4, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs, clinical_data):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        
        # Add medical data
        #print("center size : ", center.shape)
        #print("clinical size : ", clinical_data.shape)
        decoded_1 = self.relu(self.fc1(clinical_data.float()))
        decoded_1 = self.drop1(decoded_1)
        decoded_2 = self.relu(self.fc2(decoded_1))
        decoded_2 = self.drop2(decoded_2)
        decoded_clinical_data = decoded_2.view((-1,1,8,8,8))
        #print("decoded clinical size : ", decoded_clinical_data.shape)
        conv_decoded_clinical_data = self.convmed(decoded_clinical_data)
        #TODO Afficher decoded_clinical_data
        if self.med_count%(50*61) == 0:
            np.savez_compressed('/home/quhl/data/decoded_clinical_data_epoch_' + str(self.med_count//(50*144)),
                                decoded_clinical_data=decoded_clinical_data.cpu().detach().numpy())
            print("Saving decoded clinical data")
        self.med_count+=1
        #pregating = center + conv_decoded_clinical_data

        pregating, _ = self.attentionblockmed(center, conv_decoded_clinical_data)

        gating = self.gating(pregating) # or (center+decoded_clinical_data)
        

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

        return final


    @staticmethod
    def apply_argmax_softmax(pred, dim=1):
        if dim is None:
            log_p = F.sigmoid(pred)
        else:
            log_p = F.softmax(pred, dim=dim)

        return log_p


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)


