import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    """This code is the pytorch version of this network:
    http://torch.ch/blog/2015/07/30/cifar.html"""
    def conv_bn_relu(self, in_planes, out_planes):
        return nn.Sequential(nn.Conv2d(in_planes, out_planes,
                                       kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(out_planes),
                             nn.ReLU(True),
                             )

    def __init__(self, num_classes, big_input=True):
        super(VGG, self).__init__()

        # This option is to make a global average contracting or not
        self.big_input = big_input

        # Main network definition
        self.vgg = nn.Sequential(
            # ConvBNReLU(3,64):add(nn.Dropout(0.3))
            # ConvBNReLU(64,64)
            # vgg:add(MaxPooling(2,2,2,2):ceil())
            self.conv_bn_relu(3, 64),
            nn.Dropout(p=0.3),
            self.conv_bn_relu(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ConvBNReLU(64,128):add(nn.Dropout(0.4))
            # ConvBNReLU(128,128)
            # vgg:add(MaxPooling(2,2,2,2):ceil())
            self.conv_bn_relu(64, 128),
            nn.Dropout(p=0.4),
            self.conv_bn_relu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ConvBNReLU(128,256):add(nn.Dropout(0.4))
            # ConvBNReLU(256,256):add(nn.Dropout(0.4))
            # ConvBNReLU(256,256)
            # vgg:add(MaxPooling(2,2,2,2):ceil())
            self.conv_bn_relu(128, 256),
            nn.Dropout(p=0.4),
            self.conv_bn_relu(256, 256),
            nn.Dropout(p=0.4),
            self.conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ConvBNReLU(256,512):add(nn.Dropout(0.4))
            # ConvBNReLU(512,512):add(nn.Dropout(0.4))
            # ConvBNReLU(512,512)
            # vgg:add(MaxPooling(2,2,2,2):ceil())
            self.conv_bn_relu(256, 512),
            nn.Dropout(p=0.4),
            self.conv_bn_relu(512, 512),
            nn.Dropout(p=0.4),
            self.conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ConvBNReLU(512,512):add(nn.Dropout(0.4))
            # ConvBNReLU(512,512):add(nn.Dropout(0.4))
            # ConvBNReLU(512,512)
            # vgg:add(MaxPooling(2,2,2,2):ceil())
            self.conv_bn_relu(512, 512),
            nn.Dropout(p=0.4),
            self.conv_bn_relu(512, 512),
            nn.Dropout(p=0.4),
            self.conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # vgg:add(nn.View(512))
        # That will be done inside the forward

        # Classifier definition
        self.classifier = nn.Sequential(
            # classifier = nn.Sequential()
            # classifier:add(nn.Dropout(0.5))
            # classifier:add(nn.Linear(512,512))
            # classifier:add(nn.BatchNormalization(512))
            # classifier:add(nn.ReLU(true))
            # classifier:add(nn.Dropout(0.5))
            # classifier:add(nn.Linear(512,10))
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # TODO: We use bias here, does make a problem?
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.vgg(x)
        if(self.big_input):
            # TODO: This is different from: https://github.com/szagoruyko/cifar.torch
            # This for handling bigger inputs than 32x32 images
            out = F.avg_pool2d(out, 7)
        out = out.view(-1, 512)
        out = self.classifier(out)
        return out
