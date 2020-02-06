import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .lifting import LiftingScheme2D, LiftingScheme


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # This disable the conv if compression rate is equal to 1
        self.disable_conv = in_planes == out_planes
        if not self.disable_conv:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(self.bn1(x))
        else:
            return self.conv1(self.relu(self.bn1(x)))

class Haar(nn.Module):
    def __init__(self, in_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(Haar, self).__init__()
        from pytorch_wavelets import DWTForward
        self.wavelet = DWTForward(J=1,mode='zero', wave='db1').cuda()
        self.share_weights = share_weights
        if no_bottleneck:
            # We still want to do a BN and RELU, but we will not perform a conv
            # as the input_plane and output_plare are the same
            self.bootleneck = BottleneckBlock(in_planes * 1, in_planes * 1)            
        else:
            self.bootleneck = BottleneckBlock(in_planes * 4, in_planes * 2)
            
    def forward(self, x):
        LL, H = self.wavelet(x)
        LH = H[0][:,:,0,:,:]
        HL = H[0][:,:,1,:,:]
        HH = H[0][:,:,2,:,:]
                          
        x = LL
        details=torch.cat([LH,HL,HH],1)                          
        r = 0 # No regularisation here
        
        return x, r,details

class LevelDAWN(nn.Module):
    def __init__(self, in_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(LevelDAWN, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            # L2 loss function with mean
            # Note that might not be ideal for the details
            # as it does not favor sparse solution
            #self.loss_details = nn.MSELoss()
            # Potentially better solution as it less sensitive to outliers
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme2D(in_planes, share_weights,
                                       size=lifting_size, kernel_size=kernel_size,
                                       simple_lifting=simple_lifting)
        self.share_weights = share_weights
        if no_bottleneck:
            # We still want to do a BN and RELU, 
            # but we will not perform a conv as the input_plane and output_plare are the same
            # Note that it BN and RELU is to get a more stable training in our case.
            self.bootleneck = BottleneckBlock(in_planes * 1, in_planes * 1)            
        else:
            self.bootleneck = BottleneckBlock(in_planes * 4, in_planes * 2)

    def forward(self, x):
        (c, d, LL, LH, HL, HH) = self.wavelet(x)
        x = LL
        details=torch.cat([LH,HL,HH],1)                          

        r = None
        if(self.regu_approx + self.regu_details != 0.0):
            # Constraint on the details
            if self.regu_details:
                rd = self.regu_details * \
                    d.abs().mean()
                #self.loss_details(d, torch.zeros(d.size()).cuda())
                rd += self.regu_details * \
                    LH.abs().mean()
                #self.loss_details(LH, torch.zeros(LH.size()).cuda())
                rd += self.regu_details * \
                    HH.abs().mean()
                #self.loss_details(HH, torch.zeros(HH.size()).cuda())

            # Constrain on the approximation
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(c.mean(), x.mean(), p=2)
                rc += self.regu_approx * torch.dist(LL.mean(), c.mean(), p=2)
                rc += self.regu_approx * torch.dist(HL.mean(), d.mean(), p=2)

            if self.regu_approx == 0.0:
                # Only the details
                r = rd
            elif self.regu_details == 0.0:
                # Only the approximation
                r = rc
            else:
                # Both
                r = rd + rc

        if self.bootleneck:
            return self.bootleneck(x), r,details
        else:
            return x, r,details

    def image_levels(self, x):
        (c, d, LL, LH, HL, HH) = self.wavelet(x)
        x = torch.cat([LL, LH, HL, HH], 1)

        if self.bootleneck:
            return self.bootleneck(x), (LL, LH, HL, HH)
        else:
            return x, (LL, LH, HL, HH)

class DAWN(nn.Module):
    def __init__(self, num_classes, big_input=True, first_conv=3,
                 number_levels=4,
                 lifting_size=[2, 1], kernel_size=4, no_bootleneck=False,
                 classifier="mode1", share_weights=False, simple_lifting=False,
                 COLOR=True, regu_details=0.01, regu_approx=0.01, haar_wavelet=False):
        super(DAWN, self).__init__()
        self.big_input = big_input
        if COLOR:
            channels = 3
        else:
            channels = 1

        self.initialization = False
        self.nb_channels_in = first_conv

        # First convolution
        if first_conv != 3 and first_conv != 1:
            self.first_conv = True
            # Old parameter that tune the gabor filters
            self.conv1 = nn.Sequential(
                nn.Conv2d(channels, first_conv,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(first_conv),
                nn.ReLU(True),
                nn.Conv2d(first_conv, first_conv,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(first_conv),
                nn.ReLU(True),
            )
        else:
            self.first_conv = False
        if big_input:
            img_size = 224
        else:
            img_size = 32

        print("DAWN:")
        print("- first conv:", first_conv)
        print("- image size:", img_size)
        print("- nb levels :", number_levels)
        print("- levels U/P:", lifting_size)
        print("- channels: ", channels)

        # Construct the levels recursively
        self.levels = nn.ModuleList()

        in_planes = first_conv
        out_planes=first_conv
        for i in range(number_levels):
            bootleneck = True
            if no_bootleneck and i == number_levels - 1:
                bootleneck = False
            if i==0:
                if haar_wavelet:
                    self.levels.add_module(
                        'level_'+str(i),
                        Haar(in_planes,
                                lifting_size, kernel_size,  bootleneck,
                                share_weights, simple_lifting, regu_details, regu_approx)
                    )
                else:
                    self.levels.add_module(
                    'level_'+str(i),
                    LevelDAWN(in_planes,
                            lifting_size, kernel_size,  bootleneck,
                            share_weights, simple_lifting, regu_details, regu_approx)
                    )    
            else:
                self.levels.add_module(
                    'level_'+str(i),
                    LevelDAWN(in_planes,
                            lifting_size, kernel_size,  bootleneck,
                            share_weights, simple_lifting, regu_details, regu_approx)
                )
            in_planes *= 1
            img_size = img_size // 2
             # Here you can change this number if you want compression
            out_planes += in_planes * 3

        if no_bootleneck:
            in_planes *= 1
        self.img_size = img_size
   
        self.num_planes = out_planes

        print("Final channel:", self.num_planes)
        print("Final size   :", self.img_size)

        # Different classifier definition
        # In the original part, mode1 was used for all the results.
        if classifier == "mode1":
            self.fc = nn.Linear(out_planes, num_classes)
        elif classifier == "mode2":
            if in_planes//2 < num_classes:
                raise "Impossible to use mode2 in such scenario, abord"
            # Add one step classifier. Hope that it will improve the results.
            self.fc = nn.Sequential(
                nn.Linear(in_planes, in_planes//2),
                nn.BatchNorm1d(in_planes//2),
                nn.ReLU(True),
                nn.Linear(in_planes//2, num_classes)
            )
        else:
            raise "Unknown classifier"

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def process_levels(self, x):
        """This method is used for visualization proposes"""
        w,h = x.shape[-2:]

        # Choose to make X average
        x = x[:, 0, :, :]
        x = x.repeat(1, self.nb_channels_in, 1, 1)
        x_in = x
        print(x_in[:,0,:,:])


        out = []
        out_down = []
        for l in self.levels:
            w = w // 2
            h = h // 2
            x_down = nn.AdaptiveAvgPool2d((w,h))(x_in)
            x, r, details = l(x)
            out_down += [x_down]
            out += [x]
        return out, out_down

    def forward(self, x):
        if self.initialization:
            # This mode is to train the weights
            # of the lifting scheme only
            # Note that this code was only for debugging proposes
            # This part have not been used inside the paper
            w,h = x.shape[-2:]
            rs = []
            rs_diff = []

            # Choose to make X average
            x = torch.mean(x, 1, True)
            x = x.repeat(1, self.nb_channels_in, 1, 1)
            x_in = x

            # Do all the levels
            for l in self.levels:
                w = w // 2
                h = h // 2
                x_down = nn.AdaptiveAvgPool2d((w,h))(x_in)
                x, r, details = l(x)
                diff = torch.dist(x, x_down, p=2)
                rs += [r]
                rs_diff += [diff]
            return rs_diff, rs
        else:
            if self.first_conv:
                x = self.conv1(x)
            
            # Apply the different levels sequentially
            rs = [] # List of constrains on details and mean
            det = [] # List of averaged pooled details
            
            for l in self.levels:
                x, r,details = l(x)
                # Add the constrain of this level
                rs += [r]
                # Globally avgpool all the details
                det += [self.avgpool(details)]
            # At the last level (only) we GAP the approximation coefficients
            aprox = self.avgpool(x)
            # We add them inside the all GAP detail coefficients
            # Finally we concat all before applying the classifier
            det += [aprox]
            x = torch.cat(det,1)     
            x = x.view(-1, x.size()[1])
            
            return self.fc(x), rs

    def image_levels(self, x):
        """This method is used for visualization proposes"""
        if self.first_conv:
            x = self.conv1(x)

        # Apply the different levels sequentially
        # Extract all information
        images = []
        for l in self.levels:
            x, curr_images = l.image_levels(x)
            images += [(curr_images[0], curr_images[1],
                        curr_images[2], curr_images[3])]
        return images
