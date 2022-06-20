import torch
import torch.nn as nn
import torch.nn.functional as F
from models.genotypes_2d import PRIMITIVES
from models.operations_2d import *
import torch.nn.functional as F

class Cell(nn.Module):

    def __init__(self, 
                C_prev, C_prev_prev, 
                C_out,
                downup_sample,
                cell_arch,
                steps = 3,
                block_multiplier = 3,
        ):
        super(Cell, self).__init__()
        self.cell_arch = cell_arch

        self.C_out = C_out
        self.C_prev = C_prev
        self.C_prev_prev = C_prev_prev
        self.downup_sample = downup_sample
        self.pre_preprocess = ConvBR(self.C_prev_prev, self.C_out, 1, 1, 0)
        self.preprocess = ConvBR(self.C_prev, self.C_out, 1, 1, 0)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1: # down
            self.scale = 0.5
        elif downup_sample == 1: # up
            self.scale = 2
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](self.C_out, stride=1)
            self._ops.append(op)

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                                            mode='bilinear', align_corners=True)

        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
        s1 = self.preprocess(s1)

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1) 
        return prev_input, concat_feature




class SRmodule(nn.Module):
    def __init__(self, network_arch, cell_arch, cell=Cell, args=None):
        super(SRmodule, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        # self.network_arch = torch.from_numpy(network_arch)
        self.level = network_arch
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = 3
        self._num_layers = len(network_arch)
        self.C_in = 64

        self.stem0 = ConvBR(self.C_in, self.C_in*3, 3, stride=1, padding=1)

        filter_param_dict = {0: 8, 1: 4, 2: 2, 3: 1}

        for i in range(self._num_layers):
            level = self.level[i]
            prev_level = self.level[i-1]
            prev_prev_level = self.level[i-2]

            if i == 0:
                downup_sample = 3 - level
                _cell = cell(self.C_in * 3, self.C_in, 
                             self.C_in // filter_param_dict[level],
                             downup_sample,
                             self.cell_arch)

            else:
                downup_sample = prev_level - level
                if i == 1:
                    _cell = cell(self.C_in * 3 // filter_param_dict[prev_level], self.C_in * 3, 
                                 self.C_in // filter_param_dict[level],
                                 downup_sample,
                                 self.cell_arch)

                else:
                    _cell = cell(self.C_in * 3 // filter_param_dict[prev_level], self.C_in * 3 // filter_param_dict[prev_prev_level], 
                                 self.C_in // filter_param_dict[level],
                                 downup_sample,
                                self.cell_arch)

            self.cells += [_cell]

        initc = self.C_in*3
        self.last_8 = ConvBR(initc//8 , initc//8,  1, 1, 0, bn=False, relu=False)
        self.last_4 = ConvBR(initc//4 , initc//8,  1, 1, 0) 
        self.last_2 = ConvBR(initc//2 , initc//4,  1, 1, 0)  
        self.last_1 = ConvBR(initc    , initc//2,  1, 1, 0)  
          

    def forward(self, x):
        stem0 = self.stem0(x)
        out = (x, stem0)

        for i in range(self._num_layers):
            out = self.cells[i](out[0], out[1])


        last_output = out[-1]

        h, w = stem0.size()[2], stem0.size()[3]
        upsample_2  = nn.Upsample(size=[h*2, w*2], mode='bilinear', align_corners=True)
        upsample_4 = nn.Upsample(size=[h*4, w*4], mode='bilinear', align_corners=True)
        upsample_8 = nn.Upsample(size=[h*8, w*8], mode='bilinear', align_corners=True)

        if last_output.size()[2] == h:
            fea = self.last_8(upsample_8(self.last_4(upsample_4(self.last_2(upsample_2(self.last_1(last_output))))))) 
            fea = self.last_1(last_output)
        elif last_output.size()[2] == h*2:
            fea = self.last_8(upsample_8(self.last_4(upsample_4(self.last_2(last_output)))))
        elif last_output.size()[2] == h*4:
            fea = self.last_8(upsample_8(self.last_4(last_output)))
        elif last_output.size()[2] == h*8:
            fea = self.last_8(last_output)      

        return fea

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params





class FPmodule(nn.Module):
    def __init__(self, network_arch, cell_arch, cell=Cell, c_in=1, isFusion=False, args=None):
        super(FPmodule, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        # self.network_arch = torch.from_numpy(network_arch)
        self.level = network_arch
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = 3
        self._num_layers = len(network_arch)
        self.C_in = c_in

        self.stem0 = ConvBR(self.C_in, self.C_in*3, 3, stride=1, padding=1)

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}

        for i in range(self._num_layers):
            level = self.level[i]
            prev_level = self.level[i-1]
            prev_prev_level = self.level[i-2]

            if i == 0:
                downup_sample = 0 - level
                _cell = cell(self.C_in * 3, self.C_in, 
                             self.C_in * filter_param_dict[level],
                             downup_sample,
                             self.cell_arch)

            else:
                downup_sample = prev_level - level
                if i == 1:
                    _cell = cell(self.C_in * 3 * filter_param_dict[prev_level], self.C_in * 3, 
                                 self.C_in * filter_param_dict[level],
                                 downup_sample,
                                 self.cell_arch)
                else:
                    _cell = cell(self.C_in * 3 * filter_param_dict[prev_level], self.C_in * 3 * filter_param_dict[prev_prev_level], 
                                 self.C_in  * filter_param_dict[level],
                                 downup_sample,
                                 self.cell_arch)
            self.cells += [_cell]
        initc = self.C_in*3
        self.last_1  = ConvBR(initc, initc, 1, 1, 0, bn=False, relu=False) 
        self.last_2  = ConvBR(initc*2 , initc,    1, 1, 0)  
        self.last_4 = ConvBR(initc*4 , initc*2,  1, 1, 0)  
        self.last_8 = ConvBR(initc*8 , initc*4,  1, 1, 0)  

    def forward(self, x):
        stem0 = self.stem0(x)
        out = (x, stem0)

        for i in range(self._num_layers):
            out = self.cells[i](out[0], out[1])

        last_output = out[-1]

        h, w = stem0.size()[2], stem0.size()[3]
        upsample_2  = nn.Upsample(size=stem0.size()[2:], mode='bilinear', align_corners=True)
        upsample_4 = nn.Upsample(size=[h//2, w//2], mode='bilinear', align_corners=True)
        upsample_8 = nn.Upsample(size=[h//4, w//4], mode='bilinear', align_corners=True)

        if last_output.size()[2] == h:
            fea = self.last_1(last_output)
        elif last_output.size()[2] == h//2:
            fea = self.last_1(upsample_2(self.last_2(last_output)))
        elif last_output.size()[2] == h//4:
            fea = self.last_1(upsample_2(self.last_2(upsample_4(self.last_4(last_output)))))
        elif last_output.size()[2] == h//8:
            fea = self.last_1(upsample_2(self.last_2(upsample_4(self.last_4(upsample_8(self.last_8(last_output)))))))        

        return fea

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params

