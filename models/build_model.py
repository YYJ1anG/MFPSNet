import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import conv
from models.auto_feature import  AutoPrior, AutoSR
from .arch_util import pixel_unshuffle
from config_utils.train_args import obtain_train_args


args = obtain_train_args()
cuda = args.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class SRSNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_grow_ch=32, Layers=6, Filter=8, Block=4, Step=3):
        super(SRSNet, self).__init__()
        self.scale = scale
        self.Layers, self.Filter, self.Block, self.Step = Layers, Filter, Block, Step
        self.conv_first = nn.Conv2d(num_in_ch*64, num_feat, 3, 1, 1)
        self.RDB_first = nn.Sequential(*[ResidualDenseBlock(num_feat, num_grow_ch) for _ in range(3)])

        self.feature  = AutoSR()

        self.redis_Conv1 = nn.Conv2d(1536, 1536//3, 1, 1)
        self.redis_Conv2 = nn.Conv2d(1536//3, num_feat, 1, 1)

        self.RDB_last = nn.Sequential(*[ResidualDenseBlock(num_feat, num_grow_ch) for _ in range(3)])
        self.conv_afterRDB = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
    
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x): 
        feat = pixel_unshuffle(x, scale=8)
        feat = self.conv_first(feat)
        nas_in = self.RDB_first(feat)
        print( nas_in.shape)
        nas_out = self.feature(nas_in)  # [batchsize, 24, 16, 16]

        redistill = pixel_unshuffle(nas_out, scale=8) # [batchsize, 1536, 16, 16]
        redistill = self.redis_Conv2(self.lrelu(self.redis_Conv1(redistill))) # [batchsize, 64, 16, 16]
        redistill = redistill + nas_in

        feat_out = self.conv_afterRDB(self.RDB_last(redistill)) # [batchsize, 64, 16, 16]
        feat = feat_out + feat
        
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in ["feature.alphas", "feature.betas"]]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in ["feature.alphas", "feature.betas"]]


class MFPSNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_grow_ch=32, 
                sr_Layers=6, sr_Step=3,                             # SR searching
                FacialPrior_Layers=6, FacialPrior_Step=3,  # facial prior searching
                PriorFuse_Layers=6, PriorFuse_Step=3,       # prior fusing
                parse_channel = 19,
                heatmaps_channel = 71,
                facedict_channel = 256,
                prior_num = 3
                ):
      
        super(MFPSNet, self).__init__()
        self.scale = scale
        self.sr_Layers, self.sr_Step = sr_Layers, sr_Step
        self.FacialPrior_Layers,self.FacialPrior_Step = FacialPrior_Layers, FacialPrior_Step
        self.PriorFuse_Layers, self.PriorFuse_Step = PriorFuse_Layers, PriorFuse_Step

        self.conv_first = nn.Conv2d(num_in_ch*64, num_feat, 3, 1, 1)
        self.RDB_first = nn.Sequential(*[ResidualDenseBlock(num_feat, num_grow_ch) for _ in range(3)])
        # self.autoSR  = AutoSR(self.sr_Layers, num_feat, self.sr_Step)
        

        #nas_baseline
        self.conv_up_sr1 = nn.Conv2d(64, 50, 3, 1, 1)
        self.conv_up_sr2 = nn.Conv2d(50, 38, 3, 1, 1)
        self.conv_up_sr3 = nn.Conv2d(38, 24, 3, 1, 1)
        # nas_parse
        # parse_network_arch = network_layer_to_space(parse_network_path, 1) # network_space[layer][level][sample]  sample:  0: down   1: None   2: Up
        self.parse_feature = AutoPrior()
        self.parse_conv = nn.Conv2d(parse_channel+3, 8, 3, 1, 1)

        #nas_heat

        # heat_network_arch = network_layer_to_space(heat_network_path, 1) # network_space[layer][level][sample]  sample:  0: down   1: None   2: Up
        # self.heat_feature = AutoPrior(heat_network_arch, heat_cell_arch, args=args)
        # self.heat_conv = nn.Conv2d(heatmaps_channel+3, 8, 3, 1, 1)

        #nas_dict

        # dict_network_arch = network_layer_to_space(dict_network_path, 1) # network_space[layer][level][sample]  sample:  0: down   1: None   2: Up
        # self.dict_feature = AutoPrior(dict_network_arch, dict_cell_arch, args=args)
        # self.dict_conv1 = nn.Conv2d(facedict_channel, 174, 3, 1, 1)
        # self.dict_conv2 = nn.Conv2d(174, 90, 3, 1, 1)
        # self.dict_conv3 = nn.Conv2d(90, 8, 3, 1, 1)
        self.conv24 =  nn.Conv2d(48,24, 3, 1, 1)

        self.redis_Conv1 = nn.Conv2d(1536, 1536//3, 1, 1)
        self.redis_Conv2 = nn.Conv2d(1536//3, num_feat, 1, 1)

        self.RDB_last = nn.Sequential(*[ResidualDenseBlock(num_feat, num_grow_ch) for _ in range(3)])
        self.conv_afterRDB = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x, parse_x, heat_x, dict_x): 
      
        feat = pixel_unshuffle(x, scale=8)
        feat = self.conv_first(feat)

        #baseline
        autoSR_in = self.RDB_first(feat)
        autoSR = self.lrelu(self.conv_up_sr1(F.interpolate(autoSR_in, scale_factor=2, mode='nearest')))
        autoSR = self.lrelu(self.conv_up_sr2(F.interpolate(autoSR, scale_factor=2, mode='nearest')))
        autoSR_out = self.lrelu(self.conv_up_sr3(F.interpolate(autoSR, scale_factor=2, mode='nearest')))
        # autoSR_out = self.sr_feature(autoSR_in)

        #parse
        parse_in = self.parse_conv(torch.cat([x, parse_x], dim=1))
        parse_feature = self.parse_feature(parse_in)

        #heatmap
        # heat_in = self.heat_conv(torch.cat([x, heat_x], dim=1))
        # heat_feature = self.heat_feature(heat_in)

        #dict
        # dict_in = self.dict_conv1(dict_x)
        # dict_in = self.dict_conv2(self.lrelu(dict_in))
        # dict_in = self.dict_conv3(self.lrelu(dict_in))
        # dict_feature = self.dict_feature(dict_in)

        step_in = [autoSR_out, parse_feature]
        
        feature_fuse_in = self.conv24(torch.cat(step_in, dim=1)) # concat on dim of channel
        # feature_fuse_out = self.autoPriorFusion(feature_fuse_in) #[batchsize, ?, 128, 128]

        redistill = pixel_unshuffle(feature_fuse_in, scale=8) # [batchsize, ?, 16, 16]
        redistill = self.redis_Conv2(self.lrelu(self.redis_Conv1(redistill))) # [batchsize, 64, 16, 16]
        redistill = redistill + autoSR_in

        feat_out = self.conv_afterRDB(self.RDB_last(redistill)) # [batchsize, 64, 16, 16]
        feat = feat_out + feat
        
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat))) # [batchsize, 3, 128, 128]
        return out

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if "betas" in name or "alphas" in name]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "betas" not in name and "alphas" not in name]
