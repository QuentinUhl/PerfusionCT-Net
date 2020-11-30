import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
import utils.utils as util
from .base_model import BaseModel
from .networks import get_network
from .layers.loss import *
from .networks_other import get_scheduler, print_network, benchmark_fp_bp_time
from .utils import segmentation_stats, get_optimizer, get_criterion
from .networks.utils import HookBasedFeatureExtractor


class FeedForwardSegmentation(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain


        # define network input and output pars
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim
        self.output_nc = opts.output_nc
        self.multi_channel_output = opts.multi_channel_output
        self.output_cdim = opts.output_cdim
        if hasattr(opts, 'use_clinical_data'):
            self.use_clinical_data = opts.use_clinical_data
        else:
            self.use_clinical_data = False

        if hasattr(opts, 'cd_size'):
            self.cd_size = opts.cd_size
            print("youpi")
        else:
            self.cd_size = 0
            print("wtf")

        if hasattr(opts, 'use_cuda'):
            self.use_cuda = opts.use_cuda
        else:
            self.use_cuda = True

        # load/define networks
        self.net = get_network(opts.model_type, n_classes=opts.output_cdim*opts.output_nc,
                               in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                               tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                               attention_dsample=opts.attention_dsample,
                               cd_size=self.cd_size)
        if self.use_cuda: self.net = self.net.cuda()

        # load the model if a path is specified or it is in inference mode
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        #if self.isTrain:
        self.criterion = get_criterion(opts)
        # initialize optimizers
        self.schedulers = []
        self.optimizers = []
        self.optimizer_S = get_optimizer(opts, self.net.parameters())
        self.optimizers.append(self.optimizer_S)

        # print the network details
        if kwargs.get('verbose', True):
            print('Network is initialized')
            print_network(self.net)

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input.size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                _input = _input.permute(0,4,1,2,3).contiguous().view(bs[0]*bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.cuda() if self.use_cuda else _input
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
            if self.use_clinical_data and idx == 2:
                self.clinical_data = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                # assert self.input.size() == self.target.size()

    def forward(self, split):
        if split == 'train':
            if self.use_clinical_data:
                self.prediction = self.net(Variable(self.input), self.clinical_data)
            else:
                self.prediction = self.net(Variable(self.input))
            self.pred_seg = None
        elif split == 'test':
            with torch.no_grad():
                if self.use_clinical_data:
                    self.prediction = self.net(Variable(self.input), self.clinical_data)
                else:
                    self.prediction = self.net(Variable(self.input))
                # Apply a softmax and return a segmentation map
                if self.multi_channel_output: 
                    if self.output_nc > 1: # multiclass in multiple channels
                        self.logits = self.net.apply_argmax_softmax(self.prediction, dim=1) # TODO Verify dim = 1 or 2
                        self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1) # give each voxel the class index with max proba
                        print("Warning : Multiclass in multiple channels not implemented yet")
                        print("Help models/feedforward_seg_model")
                    else: # uniclass in multiple channels
                        self.logits = self.net.apply_argmax_softmax(self.prediction, dim=None)
                        self.pred_seg = (self.logits > 0.5).float()
                else:
                    if self.output_nc > 1: # multiclass in a single channel
                        self.logits = self.net.apply_argmax_softmax(self.prediction, dim=1)
                        self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1) # give each voxel the class index with max proba
                    else: # uniclass in a single channel
                        self.logits = self.net.apply_argmax_softmax(self.prediction, dim=None)
                        self.pred_seg = (self.logits > 0.5).float()

    def backward(self):
        self.loss_S = self.criterion(self.prediction, self.target)
        self.loss_S.backward()

    def optimize_parameters(self):
        self.net.train()
        self.forward(split='train')

        self.optimizer_S.zero_grad()
        self.backward()
        self.optimizer_S.step()

    # This function updates the network parameters every "accumulate_iters"
    def optimize_parameters_accumulate_grd(self, iteration):
        accumulate_iters = int(2)
        if iteration == 0: self.optimizer_S.zero_grad()
        self.net.train()
        self.forward(split='train')
        self.backward()

        if iteration % accumulate_iters == 0:
            self.optimizer_S.step()
            self.optimizer_S.zero_grad()

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        self.loss_S = self.criterion(self.prediction, self.target)

    def get_segmentation_stats(self):
        self.seg_scores, self.class_dice_score, self.chan_dice_score, self.overall_dice_score, self.roc_auc_score, self.WBCE_score, self.L1_score, self.Volume_score = segmentation_stats(self.prediction, self.target, self.output_nc, self.output_cdim)
        seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou']),
                     ('Overall_Dice', self.overall_dice_score), ('ROC_AUC', self.roc_auc_score),
                     ('WBCE_score', self.WBCE_score), ('L1_score', self.L1_score), ('Volume_score', self.Volume_score)]
        for class_id in range(self.class_dice_score.size):
            seg_stats.append(('Class_{}_Dice'.format(class_id), self.class_dice_score[class_id]))
        for chan_id in range(self.chan_dice_score.size):
            seg_stats.append(('Channel_{}_Dice'.format(chan_id), self.chan_dice_score[chan_id]))
        return OrderedDict(seg_stats)

    def get_current_errors(self):
        return OrderedDict([('Seg_Loss', self.loss_S.data.item())
                            ])

    def get_current_visuals(self):
        inp_img = util.tensor2im(self.input, 'img')
        target_img = util.tensor2im(self.target, 'lbl')
        seg_img = util.tensor2im(self.pred_seg, 'lbl')
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img), ('target_S', target_img)])

    def get_current_volumes(self):
        def to_volume(a): return a.cpu().float().detach().numpy()
        output = self.prediction if self.pred_seg is None else self.pred_seg
        return OrderedDict([('output', to_volume(output)),
                            ('input', to_volume(self.input)),
                            ('target', to_volume(self.target))])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    # returns the fp/bp times of the model
    def get_fp_bp_time (self, size=None):
        if size is None:
            size = (1, 1, 160, 160, 96)

        inp_array = Variable(torch.zeros(*size)).cuda()
        out_array = Variable(torch.zeros(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp/float(bsize), bp/float(bsize)

    def save(self, network_label, epoch_label):
        self.save_network(self.net, network_label, epoch_label, self.gpu_ids)
        if self.saved_model is not None:
            self.delete_saved_network()
        self.update_saved_model(network_label, epoch_label)

