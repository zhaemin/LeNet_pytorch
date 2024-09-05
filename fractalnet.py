import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

class Conv_Block(nn.Module):
    def __init__(self, in_features, out_features, dp):
        super(Conv_Block,self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU())
        self.dropout = nn.Dropout2d(dp)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x

class fractal_block(nn.Module):
    def __init__(self, in_features, out_features, c, dp, training, local_dp=0.15, device='cuda:0'):
        super(fractal_block,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.outputs_in_i = []
        self.dp = dp
        self.c = c
        self.max_depth = int(2**(self.c-1))
        self.training = training
        self.local_dp = local_dp
        self.max_pooling = nn.MaxPool2d(2, 2)
        self.device = device
        
    def build_blocks(self, in_features, out_features):
        module_list = [[]for col in range(self.c)]
        for i in range(self.max_depth):
            for c in range(self.c):
                new_block = None
                if i == 0: #all col
                    new_block = Conv_Block(in_features, out_features, self.dp).to(self.device)
                else:
                    if i%2 == 0: #3 col
                        if c == self.c-2:
                            new_block = Conv_Block(out_features, out_features, self.dp).to(self.device)
                    if i%4 == 0: #2 col
                        if c == self.c-3:
                            new_block = Conv_Block(out_features, out_features, self.dp).to(self.device)
                    if  c == self.c-1:
                        new_block = Conv_Block(out_features, out_features, self.dp).to(self.device)
                module_list[c].append(new_block)
        return module_list
        
    def join(self, col_numbers, do_local):
        if do_local:
            mask = np.random.rand(len(col_numbers)) > self.local_dp
            is_all_False = True
            for m in mask:
                if m == True:
                    is_all_False = False
            if is_all_False:
                mask[random.randint(0,len(col_numbers)-1)] = True
            col_numbers = col_numbers * mask
        
        init = True
        for col in col_numbers:
            if col == 0: #local drop-path
                pass
            else:
                if init:
                    stacked = self.outputs_in_i[col-1]
                    init = False
                else:
                    stacked = (stacked+self.outputs_in_i[col-1])/2
        return stacked

    def forward(self, x):
        do_local = False
        testing = False
        
        if self.training:
            local_or_global = random.random()
            if local_or_global > 0.5:
                do_local = True
            else:
                do_local = False
        else: #for test
            testing = True
        fractalblock_list = self.build_blocks(self.in_features, self.out_features)
        
        if do_local or testing:
            self.outputs_in_i = [x for i in range(self.c)]
            for i in range(self.max_depth):
                join_list = []
                for c in range(self.c):
                    pass_layer = nn.ModuleList(fractalblock_list[c])
                    if pass_layer[i] == None:
                        pass
                    else:
                        self.outputs_in_i[c] = pass_layer[i](self.outputs_in_i[c])
                if i == 1:
                    join_list = [self.c, self.c-1]
                    new_outputs = self.join(join_list, do_local)
                if i == 3:
                    join_list = [self.c, self.c-1, self.c-2]
                    new_outputs = self.join(join_list, do_local)
                if i == 5:
                    join_list = [self.c, self.c-1]
                    new_outputs = self.join(join_list, do_local)
                if i == 7:
                    join_list = [self.c, self.c-1, self.c-2, self.c-3]
                    new_outputs = self.join(join_list, do_local)
                for j in join_list:
                    self.outputs_in_i[j-1] = new_outputs
            y = self.outputs_in_i[self.c-1]
        
        else: #global drop-path
            global_col = random.randint(0,self.c-1)
            global_path = nn.Sequential()
            idx = 0
            for layer in fractalblock_list[global_col]:
                if layer != None:
                    global_path.add_module(f'conv_{idx}',layer)
                    idx+=1
            y = global_path(x)
        y = self.max_pooling(y)
        
        return y

class FractalNet(nn.Module):
    def __init__(self, layer_num, training, num_classes):
        super(FractalNet, self).__init__()
        c = int(math.log2(layer_num//5))+1
        
        drop_list = [0, 0.1, 0.2, 0.3, 0.4]
        self.block1 = fractal_block(3, 64, c, drop_list[0], training)
        self.block2 = fractal_block(64, 128, c, drop_list[1], training)
        self.block3 = fractal_block(128, 256, c, drop_list[2], training)
        self.block4 = fractal_block(256, 512, c, drop_list[3], training)
        self.block5 = fractal_block(512, 512, c, drop_list[4], training)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x