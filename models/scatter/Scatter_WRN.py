import torch.optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from kymatio import Scattering2D
import kymatio.datasets as scattering_datasets
import torch
import argparse
import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Scattering2dCNN(nn.Module):
    '''
        Simple CNN with 3x3 convs based on VGG
    '''

    def __init__(self, classifier_type='cnn', J=2, N=32, blocks=2, num_classes=10, mode=1, use_avg_pool=False):
        super(Scattering2dCNN, self).__init__()
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.blocks = blocks
        self.use_avg_pool = use_avg_pool
        # new add for wrn
        self.nspace = int(N / (2 ** J))
        if(mode==2):
            self.nfscat = int((1 + 8 * J + 8 * 8 * J * (J - 1) / 2))
        elif(mode==1):
            self.nfscat = int(1 + 8 * J)

        print("image_size:", self.nspace)
        print("nfscat:",self.nfscat) 

        self.ichannels = 256
        self.ichannels2 = 512
        self.inplanes = self.ichannels
        self.build()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def build(self):
        cfg = [256, 256, 256, 'M', 512, 512, 512, 1024, 1024]
        layers = []
        in_channels = self.nfscat * 3
        self.bn = nn.BatchNorm2d(in_channels)
        if self.classifier_type == 'cnn':
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v,
                                       kernel_size=3, padding=1)
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    in_channels = v

            layers += [nn.AdaptiveAvgPool2d(2)]
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(1024*4, self.num_classes)

        elif self.classifier_type == 'WRN':
            self.bn0 = nn.BatchNorm2d(
                in_channels, eps=1e-5, momentum=0.9, affine=False)
            self.conv1 = nn.ConvTranspose2d(
                in_channels, self.ichannels, 5, 4, 3, output_padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.ichannels)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(
                BasicBlock, self.ichannels, 1, stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, self.ichannels2, blocks=self.blocks, stride=2)
            self.layer3 = self._make_layer(
                BasicBlock, self.ichannels2*2, blocks=self.blocks, stride=2)

            self.conv0_2 = nn.Conv2d(
                self.ichannels2*2, self.ichannels2*2, kernel_size=1, padding=0)
            self.conv0_3 = nn.Conv2d(
                self.ichannels2*2, self.ichannels2*2, kernel_size=1, padding=0)
            self.bn0_2 = nn.BatchNorm2d(self.ichannels2*2)
            self.bn0_3 = nn.BatchNorm2d(self.ichannels2*2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.ichannels2*2, self.num_classes)

        elif self.classifier_type == 'mlp':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            if(not self.use_avg_pool):
                in_channels *= self.nspace * self.nspace
            print("channels classifier:", in_channels)
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, self.num_classes))
            self.features = None

        elif self.classifier_type == 'linear':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            if(not self.use_avg_pool):
                in_channels *= self.nspace * self.nspace
            print("channels classifier:", in_channels)
            self.classifier = nn.Linear(in_channels, self.num_classes)
            self.features = None

    def forward(self, x):
        if self.classifier_type == 'WRN':

            x = x.view(x.size(0), 3*self.nfscat, self.nspace, self.nspace)
            x = self.bn0(x)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.relu(self.bn0_2(self.conv0_2(x)))
            x = self.relu(self.bn0_3(self.conv0_3(x)))
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.bn(x.view(-1, 3*self.nfscat, self.nspace, self.nspace))
            if self.features:
                x = self.features(x)
            if self.use_avg_pool:
                x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x


class ScatResNet(nn.Module):
    def __init__(self, J=3, N=224, num_classes=1000, classifier='WRN', mode=1):
        super(ScatResNet, self).__init__()

        self.classifier = classifier

        self.nspace = int(N / (2 ** J))
        if(mode==2):
            self.nfscat = int((1 + 8 * J + 8 * 8 * J * (J - 1) / 2))
        elif(mode==1):
            self.nfscat = int(1 + 8 * J)
        self.ichannels = 256
        self.ichannels2 = 512
        self.inplanes = self.ichannels
        print("J",J)
        print("scattering coef", self.nfscat)
        print("image_size", self.nspace)
        self.bn0 = nn.BatchNorm2d(
            3*self.nfscat, eps=1e-5, momentum=0.9, affine=False)

        if(self.classifier == "WRN"):
            # This line is the original code
            self.conv1 = nn.Conv2d(3*self.nfscat, self.ichannels, kernel_size=3,padding=1)
            # conv3x3_3D(self.nfscat,self.ichannels)
            self.bn1 = nn.BatchNorm2d(self.ichannels)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(BasicBlock, self.ichannels, 2)
            self.layer2 = self._make_layer(BasicBlock, self.ichannels2, 2, stride=2)
            self.avgpool = nn.AvgPool2d(self.nspace//2)

            self.fc = nn.Linear(self.ichannels2, num_classes)
        elif(self.classifier == "mlp"):
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifiernet = nn.Sequential(
                nn.Linear(self.nfscat*3, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, num_classes))
        elif(self.classifier == "linear"):
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifiernet = nn.Linear(self.nfscat*3, num_classes)
        else:
            raise "Unknow configuration (Scatter)"

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if(m.affine):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        #print(x.size(0))
        x = x.view(x.size(0), 3*self.nfscat, self.nspace, self.nspace)
        x = self.bn0(x)
        if(self.classifier == "WRN"):
            x = self.conv1(x)
            x = x.view(x.size(0), self.ichannels, self.nspace, self.nspace)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif(self.classifier == "mlp" or self.classifier == "linear"):
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifiernet(x)
        else:
            raise Exception("No implemented: {}".format(self.classifier))
        return x


def scatresnet6_2(N, J):
    """Constructs a Scatter + ResNet-10 model.
    Args:
        N: is the crop size (normally 224)
        J: scattering scale (normally 3,4,or 5 for imagenet)
    """
    model = ScatResNet(J, N)
    model = model.cuda()

    return model
