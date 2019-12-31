import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNN(nn.Module):
    """This is an implementation of 'Using Filter Banks in CNN'
    The version of the network implemented if T-CNN-3 (texture only)
    Note that there is few changes compared to the original version:
     - LRN are replaced by Batchnorm
     - Adding batch norm at the network end
    It is still possible to use this old version. 
    """

    def conv_bn_relu(self, in_planes, out_planes):
        return nn.Sequential(nn.Conv2d(in_planes, out_planes,
                                       kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(out_planes),
                             nn.ReLU(True),
                             )

    def __init__(self, num_classes, big_input=True, use_original=False):
        super(TCNN, self).__init__()
        if(not big_input):
            raise Exception("IMPOSSIBLE TO USE TCNN without big inputs")

        # TODO: Note that the input of this TNN is slightly less than the original paper
        # TODO: This makes the output 26x26 (compared to 27x27)

        # Main network definition
        if not use_original:
            self.l1 = nn.Sequential(
                nn.Sequential(nn.Conv2d(3, 96,
                                        kernel_size=11, stride=2),
                              nn.BatchNorm2d(96),
                              nn.ReLU(True),
                              ),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )

            self.l2 = nn.Sequential(
                nn.Sequential(nn.Conv2d(96, 256,
                                        kernel_size=5, padding=2),
                              nn.BatchNorm2d(256),
                              nn.ReLU(True),
                              ),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )

            self.l3 = nn.Sequential(nn.Conv2d(256, 384,
                                              kernel_size=3, padding=1),
                                    nn.BatchNorm2d(384),
                                    nn.ReLU(True))

            # Classifier definition
            self.classifier = nn.Sequential(
                nn.Linear(384, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            self.l1 = nn.Sequential(
                nn.Conv2d(3, 96,
                          kernel_size=11, stride=2),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(5)
            )

            self.l2 = nn.Sequential(
                nn.Conv2d(96, 256,
                          kernel_size=5, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(5)
            )

            self.l3 = nn.Sequential(nn.Conv2d(256, 384,
                                              kernel_size=3, padding=1),
                                    nn.ReLU(True))

            # Classifier definition
            self.classifier = nn.Sequential(
                nn.Linear(384, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        out = F.avg_pool2d(x, 26)
        out = out.view(-1, 384)
        out = self.classifier(out)
        return out
