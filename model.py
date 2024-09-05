import torch
import torch.nn as nn
import torch.nn.functional as F

<<<<<<< Updated upstream
# x.view() =>  x를 (? , 16*5*5) 차원으로 변경 ->  원소의 개수가 16*5*5이므로 (1,16*5*5)로 변경된다
# -1이 있는 부분은 자동으로 채운다
=======
import densenet
import fractalnet
>>>>>>> Stashed changes

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)
        
    def forward(self,x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock,self).__init__()
        
        if in_features == out_features:
<<<<<<< Updated upstream
            stride_1 = 1
            self.identity = nn.Identity()
        else:
            stride_1 = 2
=======
            stride = 1
            self.identity = nn.Identity()
        else:
            stride = 2
>>>>>>> Stashed changes
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_features)
                )
            
<<<<<<< Updated upstream
        self.conv1 = nn.Conv2d(in_features, out_features, 3, stride=stride_1, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.batchnorm(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batchnorm(y)
=======
        self.conv1 = nn.Conv2d(in_features, out_features, 3, stride=stride, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.batchnorm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batchnorm2(y)
>>>>>>> Stashed changes
        y += self.identity(x)
        y = self.relu(y)
        
        return y
<<<<<<< Updated upstream
    
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = ResidualBlock(64,64)
        self.conv3_x = ResidualBlock(64,128)
        self.conv4_x = ResidualBlock(128,256)
        self.conv5_x = ResidualBlock(256,512)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.avgpool = nn.AvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
=======

class Preact_ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(Preact_ResidualBlock,self).__init__()
        
        if in_features == out_features:
            stride = 1
            self.identity = nn.Identity()
        else:
            stride = 2
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_features)
                )
        
        self.batchnorm1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, out_features, 3, stride=stride, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y = self.batchnorm1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y += self.identity(x)
        
        return y

class BottleneckBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BottleneckBlock,self).__init__()
        
        if in_features == out_features*4:
            stride = 1
            self.identity = nn.Identity()
        elif in_features == out_features:
            stride = 1
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features*4, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_features*4)
                )
        else:
            stride = 2
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features*4, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_features*4)
                )
        
        self.conv1 = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        self.conv3 = nn.Conv2d(out_features, out_features*4, 1, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(out_features*4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.batchnorm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.batchnorm3(y)
        y += self.identity(x)
        y = self.relu(y)
        
        return y

class Preact_BottleneckBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(Preact_BottleneckBlock,self).__init__()
        
        if in_features == out_features*4:
            stride = 1
            self.identity = nn.Identity()
        elif in_features == out_features:
            stride = 1
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features*4, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_features*4)
                )
        else:
            stride = 2
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features*4, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_features*4)
                )

        self.batchnorm1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_features)
        self.conv3 = nn.Conv2d(out_features, out_features*4, 1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.batchnorm1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batchnorm3(y)
        y = self.relu(y)
        y = self.conv3(y)
        y += self.identity(x)
        
        return y

class Preact_ResNet(nn.Module):
    def __init__(self, layer_num, num_classes):
        super(Preact_ResNet, self).__init__()
        
        self.block_type = 'Residual'
        
        if layer_num == 18:
            layers = [2,2,2,2]
        elif layer_num == 34:
            layers = [3,4,6,3]
        elif layer_num == 50:
            layers = [3,4,6,3]
            self.block_type = 'Bottleneck'
        elif layer_num == 101:
            layers = [3,4,23,3]
            self.block_type = 'Bottleneck'
        elif layer_num == 152:
            layers = [3,8,36,3]
            self.block_type = 'Bottleneck'
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2_x = self.build_blocks(layers[0], 64, True)
        self.conv3_x = self.build_blocks(layers[1], 128)
        self.conv4_x = self.build_blocks(layers[2], 256)
        self.conv5_x = self.build_blocks(layers[3], 512)
        self.avgpool = nn.AvgPool2d(4)
        self.relu = nn.ReLU()
        if self.block_type == 'Bottleneck':
            self.fc = nn.Linear(2048, num_classes)
        else:
            self.fc = nn.Linear(512, num_classes)
            
    def build_blocks(self, layer, x, conv2=False):
        module_list = []
        if self.block_type == 'Residual':
            for i in range(layer):
                if i == 0:
                    if conv2:
                        module_list.append(ResidualBlock(x, x))
                    else: 
                        module_list.append(Preact_ResidualBlock(x//2, x))
                else:
                    module_list.append(Preact_ResidualBlock(x, x))
        elif self.block_type == 'Bottleneck':
            for i in range(layer):
                if i == 0:
                    if conv2:
                        module_list.append(BottleneckBlock(x, x))
                    else:
                        module_list.append(Preact_BottleneckBlock(x*2, x))
                else:
                    module_list.append(Preact_BottleneckBlock(x*4, x))
        return nn.Sequential(*module_list)
            
    def forward(self,x):
        x = self.conv1(x)
>>>>>>> Stashed changes
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
<<<<<<< Updated upstream
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
=======
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def set_net(model, training=True, num_classes=10):
    if model == 'LeNet':
        net = LeNet(num_classes)
    elif model == 'ResNet-18':
        net = Preact_ResNet(18, num_classes)
    elif model == 'ResNet-34':
        net = Preact_ResNet(34, num_classes)
    elif model == 'ResNet-50':
        net = Preact_ResNet(50, num_classes)
    elif model == 'ResNet-101':
        net = Preact_ResNet(101, num_classes)
    elif model == 'ResNet-152':
        net = Preact_ResNet(152, num_classes)
    elif model == 'DenseNet_BC-121':
        net = densenet.DenseNet_BC(121, num_classes=num_classes)
    elif model == 'DenseNet_BC-169':
        net = densenet.DenseNet_BC(169, num_classes=num_classes)
    elif model == 'DenseNet_BC-201':
        net = densenet.DenseNet_BC(201, num_classes=num_classes)
    elif model == 'DenseNet_BC-264':
        net = densenet.DenseNet_BC(264, num_classes=num_classes)
    elif model == 'FractalNet-20':
        net = fractalnet.FractalNet(20, training=training, num_classes=num_classes)
    elif model == 'FractalNet-40':
        net = fractalnet.FractalNet(40, training=training, num_classes=num_classes)
    return net
>>>>>>> Stashed changes
