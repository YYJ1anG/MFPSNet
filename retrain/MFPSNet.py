import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.decoding_formulas import network_layer_to_space
from models.build_model import ResidualDenseBlock
from retrain.build_SRmodule import SRmodule, FPmodule
from models.arch_util import pixel_unshuffle


class SRSNet(nn.Module):
    def __init__(self, args, num_in_ch=3, num_out_ch=3, num_feat=64, num_grow_ch=32):
        super(SRSNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch*64, num_feat, 3, 1, 1)
        self.RDB_first = nn.Sequential(*[ResidualDenseBlock(num_feat, num_grow_ch) for _ in range(3)])

        sr_network_path, sr_cell_arch = np.load(args.sr_net_arch), np.load(args.sr_cell_arch)
        print('Feature network path:{} \n'.format(sr_network_path))
        sr_network_arch = network_layer_to_space(sr_network_path, 0) # network_space[layer][level][sample]  sample:  0: down   1: None   2: Up
        self.feature = SRmodule(sr_network_arch, sr_cell_arch, args=args)

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

        nas_out = self.feature(nas_in)  # [batchsize, 24, 128, 128]

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


class MFPSNet(nn.Module):
    def __init__(self, args, num_in_ch=3, num_out_ch=3, num_feat=64, num_grow_ch=32,  
                parse_channel = 19,
                heatmaps_channel = 71,
                facedict_channel = 256,
                feat_num = 3
                ):
        super(MFPSNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch*64, num_feat, 3, 1, 1)
        self.RDB_first = nn.Sequential(*[ResidualDenseBlock(num_feat, num_grow_ch) for _ in range(3)])

#-------------------------------------------------------nas_baseline-------------------------------------------------------------------------------
        sr_network_arch = np.array([2,1,1,1,0,1])
        sr_cell_arch = np.array([[0,0],[1,0],[2,2],[3,0],[7,1],[8,3]])
        self.sr_feature = SRmodule(sr_network_arch, sr_cell_arch, args=args)
        
        self.upsample_2 = nn.Upsample(size=[16*2, 16*2], mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(size=[16*4, 16*4], mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(size=[16*8, 16*8], mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(64, 50, 3, 1, 1)
        self.conv2 = nn.Conv2d(50, 38, 3, 1, 1)
        self.conv3 = nn.Conv2d(38, 24, 3, 1, 1)

#-------------------------------------------------------nas_parse----------------------------------------------------------------------------------
        parse_network_arch = np.array([1,2,1,0,1,0])
        parse_cell_arch = np.array([[0,1],[1,1],[3,1],[4,3],[5,3],[7,0]])
        self.parse_feature = FPmodule(parse_network_arch, parse_cell_arch, args=args, c_in=8)
        self.parse_conv = nn.Conv2d(parse_channel+3, 8, 3, 1, 1)


#-------------------------------------------------------nas_heat----------------------------------------------------------------------------------
        heat_network_arch = np.array([0,0,0,1,0,1])
        heat_cell_arch = np.array([[0,3],[1,1],[2,2],[3,2],[5,1],[7,2]])
        self.heat_feature = FPmodule(heat_network_arch, heat_cell_arch, args=args, c_in=8)
        self.heat_conv = nn.Conv2d(heatmaps_channel+3, 8, 3, 1, 1)

# #-------------------------------------------------------nas_dict----------------------------------------------------------------------------------
        dict_network_arch = np.array([0,0,1,2,1,0])
        dict_cell_arch = np.array([[0,0],[1,1],[2,3],[4,0],[6,1],[7,0]])
        self.dict_feature = FPmodule(dict_network_arch, dict_cell_arch, args=args, c_in=8)
        self.dict_conv1 = nn.Conv2d(facedict_channel, 174, 3, 1, 1)
        self.dict_conv2 = nn.Conv2d(174, 90, 3, 1, 1)
        self.dict_conv3 = nn.Conv2d(90, 8, 3, 1, 1)

        self.conv8 = nn.Conv2d(24, 8, 3, 1, 1)

# # #-------------------------------------------------------nas_fusion---------------------------------------------------------------------------------
        fution_network_arch = np.array([0,1,2,1,1,2])
        fution_cell_arch = np.array([[0,0],[1,0],[2,2],[4,2],[7,2],[8,3]])
        self.fusion_feature = FPmodule(fution_network_arch, fution_cell_arch, args=args,  c_in=48, isFusion=True)
        
        self.fusionconv1 = nn.Conv2d(72+feat_num*24, 48, 3, 1, 1)
        self.fusionconv2 = nn.Conv2d(48, 24, 3, 1, 1)
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

    def forward(self, x, parse_x, heat_x, dict_x, iter_num):
        output_imgs = []
        base_x = x
        for i in range(iter_num):
                if  i == 0:
                        
                        feat = pixel_unshuffle(base_x, scale=8)
                        feat = self.conv_first(feat)

                        sr_in = self.RDB_first(feat)
                        sr_out = self.sr_feature(sr_in)  # [batchsize, 24, 128, 128]
                        
                        feature_fuse_out = sr_out

                        redistill = pixel_unshuffle(feature_fuse_out, scale=8) # [batchsize, 1536, 16, 16]
                        redistill = self.redis_Conv2(self.lrelu(self.redis_Conv1(redistill))) # [batchsize, 64, 16, 16]
                        redistill = redistill + sr_in

                        feat_out = self.conv_afterRDB(self.RDB_last(redistill)) # [batchsize, 64, 16, 16]
                        feat = feat_out + feat
                        
                        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
                        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
                        feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
                        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
                        output_imgs.append(out)
                        base_x = out
                else:
                                

                        feat = pixel_unshuffle(base_x, scale=8)
                        feat = self.conv_first(feat)

                        sr_in = self.RDB_first(feat)
                        sr_out = self.sr_feature(sr_in)  # [batchsize, 24, 128, 128]

                        parse_in = torch.cat([base_x, parse_x], dim=1)
                        parse_in = self.parse_conv(parse_in)
                        
                        parse_feature = self.parse_feature(parse_in)
                        parse_feature = self.conv8(parse_feature)

                        heat_in = torch.cat([base_x, heat_x], dim=1)
                        heat_in = self.heat_conv(heat_in)
                        
                        heat_feature = self.heat_feature(heat_in)
                        heat_feature = self.conv8(heat_feature)

                        dict_in = self.dict_conv1(dict_x)
                        dict_in = self.dict_conv2(dict_in)
                        dict_in = self.dict_conv3(dict_in)
                        
                        dict_feature = self.dict_feature(dict_in)
                        dict_feature = self.conv8(dict_feature)


                        step_in = [sr_out, parse_feature, heat_feature, dict_feature] 
                        #step_in = [sr_out, parse_feature]
                        

                        feature_fuse_in = torch.cat(step_in, dim=1)
                        
                        
                        feature_fuse_out = self.fusion_feature(feature_fuse_in) #[batchsize, ?, 128, 128]
                        # print("feature_fuse_out shape : {}".format(feature_fuse_out.shape))
                        # feature_fuse_out = self.conv__(feature_fuse_in)
                        feature_fuse_out = self.fusionconv1(feature_fuse_out)
                        feature_fuse_out = self.fusionconv2(feature_fuse_out)

                        redistill = pixel_unshuffle(feature_fuse_out, scale=8) # [batchsize, 1536, 16, 16]
                        redistill = self.redis_Conv2(self.lrelu(self.redis_Conv1(redistill))) # [batchsize, 64, 16, 16]
                        redistill = redistill + sr_in

                        feat_out = self.conv_afterRDB(self.RDB_last(redistill)) # [batchsize, 64, 16, 16]
                        feat = feat_out + feat
                        
                        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
                        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
                        feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
                        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
                        base_x = out
                        output_imgs.append(out)
        return output_imgs


