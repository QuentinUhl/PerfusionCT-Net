import torch.nn as nn
import torch
from .utils import UnetConv3, UnetUp3_CT, UnetGridGatingSignal3, UnetDsv3
import torch.nn.functional as F
from models.networks_other import init_weights
from models.layers.grid_attention_layer import GridAttentionBlock3D


class unet_pCT_cd_multi_down_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=2, is_deconv=True, in_channels=4,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True):
        super(unet_pCT_cd_multi_down_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        self.filters = [64, 128, 256, 512, 1024]
        self.filters = [int(x / self.feature_scale) for x in self.filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, self.filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(self.filters[0], self.filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(self.filters[1], self.filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(self.filters[2], self.filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(self.filters[3], self.filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.gating = UnetGridGatingSignal3(self.filters[4], self.filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)
        
        # add clinical data
        self.fc1 = nn.Linear(21,self.filters[0])
        self.attentionblockmed1 = MultiAttentionBlock(in_size=self.filters[0], gate_size=self.filters[0], inter_size=self.filters[0],
                                                     nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.fc2 = nn.Linear(21,self.filters[1])
        self.attentionblockmed2 = MultiAttentionBlock(in_size=self.filters[1], gate_size=self.filters[1], inter_size=self.filters[1],
                                                     nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.fc3 = nn.Linear(21,self.filters[2])
        self.attentionblockmed3 = MultiAttentionBlock(in_size=self.filters[2], gate_size=self.filters[2], inter_size=self.filters[2],
                                                     nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.fc4 = nn.Linear(21,self.filters[3])
        self.attentionblockmed4 = MultiAttentionBlock(in_size=self.filters[3], gate_size=self.filters[3], inter_size=self.filters[3],
                                                     nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)

        self.fc5a = nn.Linear(21, 64)
        self.fc5b = nn.Linear(64,self.filters[4]) #256*1*1*1
        self.relu = nn.ReLU()
        self.attentionblockmed5 = MultiAttentionBlock(in_size=self.filters[4], gate_size=self.filters[4], inter_size=self.filters[4],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=self.filters[1], gate_size=self.filters[2], inter_size=self.filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=self.filters[2], gate_size=self.filters[3], inter_size=self.filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=self.filters[3], gate_size=self.filters[4], inter_size=self.filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp3_CT(self.filters[4], self.filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(self.filters[3], self.filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(self.filters[2], self.filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(self.filters[1], self.filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=self.filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=self.filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=self.filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=self.filters[0], out_channels=n_classes, kernel_size=1)

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
        cd_fc1 = self.fc1(clinical_data.float())
        cd_fc1 = self.relu(cd_fc1)
        conv1 = self.conv1(inputs)
        print("conv1 size : ", conv1.shape)
        print("cd_fc1 size : ", cd_fc1.shape)
        print("cd_fc1 size : ", cd_fc1.view((-1, self.filters[1], 1, 1, 1)).shape)
        cd_fc1 = cd_fc1.view((-1, self.filters[0], 1, 1, 1)) * conv1
        conv1, _ = self.attentionblockmed1(conv1, cd_fc1)
        maxpool1 = self.maxpool1(conv1)

        cd_fc2 = self.fc2(clinical_data.float())
        cd_fc2 = self.relu(cd_fc2)
        conv2 = self.conv2(maxpool1)
        cd_fc2 = cd_fc2.view((-1, self.filters[1], 1, 1, 1)) * conv2
        conv2, _ = self.attentionblockmed2(conv2, cd_fc2)
        maxpool2 = self.maxpool2(conv2)

        cd_fc3 = self.fc3(clinical_data.float())
        cd_fc3 = self.relu(cd_fc3)
        conv3 = self.conv3(maxpool2)
        cd_fc3 = cd_fc3.view((-1, self.filters[2], 1, 1, 1)) * conv3
        conv3, _ = self.attentionblockmed3(conv3, cd_fc3)
        maxpool3 = self.maxpool3(conv3)

        cd_fc4 = self.fc4(clinical_data.float())
        cd_fc4 = self.relu(cd_fc4)
        conv4 = self.conv4(maxpool3)
        cd_fc4 = cd_fc4.view((-1, self.filters[3], 1, 1, 1)) * conv4
        conv4, _ = self.attentionblockmed4(conv4, cd_fc4)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Add medical data
        # print("center size : ", center.shape)
        # print("clinical size : ", clinical_data.shape)
        cd_fc5a = self.fc5a(clinical_data.float())
        cd_fc5a = self.relu(cd_fc5a)
        cd_fc5b = self.fc5b(cd_fc5a)
        cd_fc5b = self.relu(cd_fc5b)
        decoded_clinical_data = cd_fc5b.view((-1, self.filters[4], 1, 1, 1)) * center
        # print("decoded clinical size : ", decoded_clinical_data.shape)
        pregating, _ = self.attentionblockmed5(center, decoded_clinical_data)
        

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


