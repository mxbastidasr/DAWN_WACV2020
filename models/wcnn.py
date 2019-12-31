import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Haar wavelet
from .lifting import WaveletHaar2D, LiftingScheme2D, Wavelet2D


class BasicConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(BasicConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))


class BasicSkipBlock(nn.Module):
    def __init__(self, in_planes, stride):
        super(BasicSkipBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False)

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.block1 = BasicConvBlock(in_planes, out_planes, stride=1)
        self.block2 = BasicConvBlock(out_planes, out_planes, stride=2)

    def forward(self, x):
        return self.block2(self.block1(x))


class WCNN(nn.Module):
    """Network definition of "Wavelet Convolutional Neural Networks for Texture Classification"
    in pytorch. Follow closely the network structure describe in Figure 4.
    However, this implementation is able to choose the number of decomposition needed.
    """

    def __init__(self, num_classes, big_input=True, wavelet='haar', levels=4):
        super(WCNN, self).__init__()
        self.big_input = big_input
        if self.big_input and levels != 4:
            raise "4 levels is required with big inputs"

        print("WCNN:")
        print(" - wavelet type:", wavelet)

        # reduce by 2 the dimension
        # gives back (LL, LH, HL, HH)
        if wavelet == "haar":
            self.wavelet1 = WaveletHaar2D()
            self.wavelet2 = WaveletHaar2D()
            self.wavelet3 = WaveletHaar2D()
            self.wavelet4 = WaveletHaar2D()
        elif wavelet == "lifting":
            # TODO: The main problem for the moment is the kernel size
            #       used inside the lifting scheme
            raise "Unsupported lifting scheme for shin"
        elif wavelet == "db2":
            self.wavelet1 = Wavelet2D(3, "db2")  # RGB
            self.wavelet2 = Wavelet2D(3, "db2")  # RGB
            self.wavelet3 = Wavelet2D(3, "db2")  # RGB
            self.wavelet4 = Wavelet2D(3, "db2")  # RGB
        else:
            raise "Unknown wavelet scheme"

        # Save the number of levels and precompute the number of features per levels
        if levels > 4:
            raise "Impossible to go beyong 4 levels in WCNN"
        elif levels < 2:
            raise "Impossible to go below 2 levels in WCNN"
        self.levels = levels
        in_level_1 = 4 * 3
        in_level_2 = in_level_1 + 4 * 3 + 64 * 2
        in_level_3 = in_level_2 + 4 * 3 + 128 * 2
        in_level_4 = in_level_3 + 4 * 3 + 256 * 2
        print("Levels (DEBUG): ")
        print("-", in_level_1)
        print("-", in_level_2)
        print("-", in_level_3)
        print("-", in_level_4)

        # Level 1
        self.block1 = ConvBlock(in_level_1, 64)
        self.block_haar1 = nn.Sequential(
            BasicConvBlock(in_level_1, 64, stride=1)
        )
        self.skip1 = BasicSkipBlock(in_level_1, stride=2)
        self.skip_haar1 = BasicSkipBlock(in_level_1, stride=1)

        # Level 2
        # Skip from previous input, haar skip and conv layers
        if levels >= 2:
            self.block2 = ConvBlock(in_level_2, 128)
            self.skip2 = BasicSkipBlock(in_level_2, stride=2)
            if levels > 2:
                self.block_haar2 = nn.Sequential(
                    BasicConvBlock(in_level_1, 64, stride=1),
                    BasicConvBlock(64, 128, stride=1),
                )
                self.skip_haar2 = BasicSkipBlock(in_level_1, stride=1)

        # Level 3
        if levels >= 3:
            self.block3 = ConvBlock(in_level_3, 256)
            self.skip3 = BasicSkipBlock(in_level_3, stride=2)
            if levels > 3:
                self.block_haar3 = nn.Sequential(
                    BasicConvBlock(in_level_1, 64, stride=1),
                    BasicConvBlock(64, 128, stride=1),
                    BasicConvBlock(128, 256, stride=1),
                )
                self.skip_haar3 = BasicSkipBlock(in_level_1, stride=1)

        # Level 4
        if levels >= 4:
            self.block4 = ConvBlock(in_level_4, 512)
            self.skip4 = BasicSkipBlock(in_level_4, stride=2)

        # Final
        if levels == 1:
            raise "Unsupported"
        elif levels == 2:
            self.in_planes = in_level_2 + 128
            self.size_image = 4
        elif levels == 3:
            self.in_planes = in_level_3 + 256
            self.size_image = 2
        elif levels == 4:
            self.in_planes = in_level_4 + 512
            self.size_image = 1
        print("Final number of features before FC: "+str(self.in_planes))
        self.fc = nn.Linear(self.in_planes, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        # haar multi-levels
        (LL0, LH0, HL0, HH0) = self.wavelet1(x)
        haar_all_0 = torch.cat([LL0, LH0, HL0, HH0], 1)
        (LL1, LH1, HL1, HH1) = self.wavelet2(LL0)
        haar_all_1 = torch.cat([LL1, LH1, HL1, HH1], 1)
        (LL2, LH2, HL2, HH2) = self.wavelet3(LL1)
        haar_all_2 = torch.cat([LL2, LH2, HL2, HH2], 1)
        (LL3, LH3, HL3, HH3) = self.wavelet4(LL2)
        haar_all_3 = torch.cat([LL3, LH3, HL3, HH3], 1)

        x = torch.cat([self.skip1(haar_all_0),
                       self.block1(haar_all_0),
                       self.block_haar1(haar_all_1),
                       self.skip_haar1(haar_all_1)], 1)

        # Level 2
        if self.levels > 2:
            x = torch.cat([self.skip2(x),
                           self.block2(x),
                           self.block_haar2(haar_all_2),
                           self.skip_haar2(haar_all_2)], 1)
        elif self.levels == 2:
            x = torch.cat([self.skip2(x),
                           self.block2(x)], 1)
        # Level 3
        if self.levels > 3:
            x = torch.cat([self.skip3(x),
                           self.block3(x),
                           self.block_haar3(haar_all_3),
                           self.skip_haar3(haar_all_3)], 1)
        elif self.levels == 3:
            x = torch.cat([self.skip3(x),
                           self.block3(x)], 1)

        # Level 4
        if self.levels >= 4:
            x = torch.cat([self.skip4(x),
                           self.block4(x)], 1)

        # Classifier
        if self.levels == 4:
            if self.big_input:
                x = F.avg_pool2d(x, 7)
        else:
            if self.size_image != 1:
                x = F.avg_pool2d(x, self.size_image)
        x = x.view(-1, self.in_planes)
        return self.fc(x)
