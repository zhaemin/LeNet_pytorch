import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock_dense(nn.Module):
    def __init__(self, in_features, k):
        super(BottleneckBlock_dense, self).__init__()
        
        self.batchnorm1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, k*4, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(k*4)
        self.conv2 = nn.Conv2d(k*4, k, 3, padding=1, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        y = self.batchnorm1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv2(y)
        
        return torch.cat([x, y], dim=1)

class Transition_Layer(nn.Module):
    def __init__(self, in_features, theta):
        super(Transition_Layer, self).__init__()
        self.conv = nn.Conv2d(in_features, int(in_features*theta), 1, bias=False)
        self.avg_pool = nn.AvgPool2d(2, stride=2)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.avg_pool(x)
        
        return x

class DenseNet_BC(nn.Module):
    def __init__(self, layer_num, k=32, theta=0.5, num_classes=10):
        super(DenseNet_BC, self).__init__()
        
        if layer_num == 121:
            layers = [6,12,24,16]
        elif layer_num == 169:
            layers = [6,12,32,32]
        elif layer_num == 201:
            layers = [6,12,48,32]
        elif layer_num == 264:
            layers = [6,12,65,48]
            
        self.k = k
        self.theta = theta
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.k*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.k*2),
            nn.ReLU()
        )
        self.curr_channel = self.k*2
        self.dense1 = self.build_blocks(layers[0])
        self.dense2 = self.build_blocks(layers[1])
        self.dense3 = self.build_blocks(layers[2])
        self.dense4 = self.build_blocks(layers[3],True)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(self.curr_channel, num_classes)
        
    def build_blocks(self, layer, DB4=False):
        module_list = []
        for i in range(layer):
            module_list.append(BottleneckBlock_dense(self.curr_channel, self.k))
            self.curr_channel += self.k
            
        module_list.append(nn.Sequential(
            nn.BatchNorm2d(self.curr_channel),
            nn.ReLU()
        ))
        
        if DB4:
            pass
        else:
            module_list.append(Transition_Layer(self.curr_channel, self.theta))
            self.curr_channel = int(self.curr_channel*self.theta)
            
        return nn.Sequential(*module_list)
            
    def forward(self,x):
        x = self.conv(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x