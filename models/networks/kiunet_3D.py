import torch.nn as nn
import torch.nn.functional as F
import torch


class kiunet_3d(nn.Module):

    def __init__(self, training):
        super(kiunet_3d, self).__init__()
        self.training = training
        self.start = nn.Conv3d(1, 1, 3, stride=2, padding=1)

        self.encoder1 = nn.Conv3d(1, 16, 3, stride=1,
                                  padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.en1_bn = nn.BatchNorm3d(16)
        self.encoder2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.en2_bn = nn.BatchNorm3d(32)
        self.encoder3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm3d(64)

        self.decoder1 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.de1_bn = nn.BatchNorm3d(32)
        self.decoder2 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm3d(16)
        self.decoder3 = nn.Conv3d(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm3d(8)

        self.decoderf1 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm3d(32)
        self.decoderf2 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm3d(16)
        self.decoderf3 = nn.Conv3d(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm3d(8)

        self.encoderf1 = nn.Conv3d(1, 16, 3, stride=1,
                                   padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.enf1_bn = nn.BatchNorm3d(16)
        self.encoderf2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm3d(32)
        self.encoderf3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm3d(64)

        self.intere1_1 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm3d(16)
        self.intere2_1 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm3d(32)
        self.intere3_1 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm3d(64)

        self.intere1_2 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm3d(16)
        self.intere2_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm3d(32)
        self.intere3_2 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm3d(64)

        self.interd1_1 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm3d(32)
        self.interd2_1 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm3d(16)
        self.interd3_1 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm3d(64)

        self.interd1_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm3d(32)
        self.interd2_2 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm3d(16)
        self.interd3_2 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm3d(64)

        # self.start = nn.Conv3d(1, 1, 3, stride=1, padding=1)
        self.final = nn.Conv3d(8, 1, 1, stride=1, padding=0)
        self.fin = nn.Conv3d(1, 1, 1, stride=1, padding=0)

        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=16, mode='trilinear'),
            nn.Sigmoid()
        )

        # 128*128
        self.map3 = nn.Sequential(
            nn.Conv3d(16, 1, 1, 1),
            nn.Upsample(scale_factor=8, mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64
        self.map2 = nn.Sequential(
            nn.Conv3d(8, 1, 1, 1),
            nn.Upsample(scale_factor=4, mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=32, mode='trilinear'),
            nn.Sigmoid()
        )

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        outx = self.start(x)
        # print(outx.shape)
        out = F.relu(self.en1_bn(F.max_pool3d(self.encoder1(outx), 2, 2)))  # U-Net branch
        out1 = F.relu(self.enf1_bn(
            F.interpolate(self.encoderf1(outx), scale_factor=1, mode='trilinear')))  # Ki-Net branch
        tmp = out
        # print(out.shape,out1.shape)
        out = torch.add(out, F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))), scale_factor=0.5,
                                           mode='trilinear'))  # CRFB
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))), scale_factor=2,
                                             mode='trilinear'))  # CRFB

        u1 = out  # skip conn
        o1 = out1  # skip conn

        out = F.relu(self.en2_bn(F.max_pool3d(self.encoder2(out), 2, 2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1), scale_factor=1, mode='trilinear')))
        tmp = out
        # print(out.shape,out1.shape)
        out = torch.add(out, F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))), scale_factor=0.25,
                                           mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))), scale_factor=4,
                                             mode='trilinear'))

        u2 = out
        o2 = out1
        out = F.pad(out, [0, 0, 0, 0, 0, 1])
        # print(out.shape)
        out = F.relu(self.en3_bn(F.max_pool3d(self.encoder3(out), 2, 2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1), scale_factor=2, mode='trilinear')))
        # print(out.shape,out1.shape)
        tmp = out
        out = torch.add(out,
                        F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))), scale_factor=0.0625,
                                      mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))), scale_factor=16,
                                             mode='trilinear'))

        ### End of encoder block

        ### Start Decoder

        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out), scale_factor=2, mode='trilinear')))  # U-NET
        out1 = F.relu(self.def1_bn(F.max_pool3d(self.decoderf1(out1), 2, 2)))  # Ki-NET
        tmp = out
        # print(out.shape,out1.shape)
        out = torch.add(out, F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))), scale_factor=0.25,
                                           mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))), scale_factor=4,
                                             mode='trilinear'))
        # print(out.shape)
        #output1 = self.map4(out)
        out = torch.add(out, u2)  # skip conn
        out1 = torch.add(out1, o2)  # skip conn

        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out), scale_factor=2, mode='trilinear')))
        out1 = F.relu(self.def2_bn(F.max_pool3d(self.decoderf2(out1), 1, 1)))
        # print(out.shape,out1.shape)
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))), scale_factor=0.5,
                                           mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))), scale_factor=2,
                                             mode='trilinear'))
        #output2 = self.map3(out)
        # print(out1.shape,o1.shape)
        out = torch.add(out, u1)
        out1 = torch.add(out1, o1)

        out = F.sigmoid(self.de3_bn(F.interpolate(self.decoder3(out), scale_factor=2, mode='trilinear')))
        out1 = F.sigmoid(self.def3_bn(F.max_pool3d(self.decoderf3(out1), 1, 1)))
        # print(out.shape,out1.shape)
        #output3 = self.map2(out)

        out = torch.add(out, out1)  # fusion of both branches

        out = F.relu(self.final(out))  # 1*1 conv

        output4 = F.interpolate(self.fin(out), scale_factor=2, mode='trilinear')
        output4 = torch.flatten(output4, start_dim=0, end_dim=0)
        # print(out.shape)
        # print(output4.shape)
        # out = self.soft(out)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        # if self.training is True:
        #     return output1, output2, output3, output4
        # else:

        return output4

    @staticmethod
    def apply_argmax_softmax(pred, dim=1):
        if (dim is None) or (dim==1):
            log_p = F.sigmoid(pred)
        else:
            log_p = F.softmax(pred, dim=dim)

        return log_p