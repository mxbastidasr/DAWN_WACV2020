import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# This file contains the lifting scheme implementation
# There is no complete network definition inside this file.
# Note that it also contains other wavelet transformation
# used in WCNN and DAWN networks.

# To change if we do horizontal first inside the LS
HORIZONTAL_FIRST = True

class Splitting(nn.Module):
    def __init__(self, horizontal):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        self.horizontal = horizontal
        if(horizontal):
            self.conv_even = lambda x: x[:, :, :, ::2]
            self.conv_odd = lambda x: x[:, :, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, :, ::2, :]
            self.conv_odd = lambda x: x[:, :, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))


class WaveletHaar(nn.Module):
    def __init__(self, horizontal):
        super(WaveletHaar, self).__init__()
        self.split = Splitting(horizontal)
        self.norm = math.sqrt(2.0)

    def forward(self, x):
        '''Returns the approximation and detail part'''
        (x_even, x_odd) = self.split(x)

        # Haar wavelet definition
        d = (x_odd - x_even) / self.norm
        c = (x_odd + x_even) / self.norm
        return (c, d)


class WaveletHaar2D(nn.Module):
    def __init__(self):
        super(WaveletHaar2D, self).__init__()
        self.horizontal_haar = WaveletHaar(horizontal=True)
        self.vertical_haar = WaveletHaar(horizontal=False)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.horizontal_haar(x)
        (LL, LH) = self.vertical_haar(c)
        (HL, HH) = self.vertical_haar(d)
        return (LL, LH, HL, HH)


class Wavelet(nn.Module):
    """This module extract wavelet coefficient defined in pywt
    and create 2D convolution kernels to be able to use GPU"""

    def _coef_h(self, in_planes, coef):
        """Construct the weight matrix for horizontal 2D convolution.
        The weights are repeated on the diagonal"""
        v = []
        for i in range(in_planes):
            l = []
            for j in range(in_planes):
                if i == j:
                    l.append([[c for c in coef]])
                else:
                    l.append([[0.0 for c in coef]])
            v.append(l)
        return v

    def _coef_v(self, in_planes, coef):
        """Construct the weight matrix for vertical 2D convolution.
        The weights are repeated on the diagonal"""
        v = []
        for i in range(in_planes):
            l = []
            for j in range(in_planes):
                if i == j:
                    l.append([[c] for c in coef])
                else:
                    l.append([[0.0] for c in coef])
            v.append(l)
        return v

    def __init__(self, in_planes, horizontal, name="db2"):
        super(Wavelet, self).__init__()

        # Import wavelet coefficients
        import pywt
        wavelet = pywt.Wavelet(name)
        coef_low = wavelet.dec_lo
        coef_high = wavelet.dec_hi
        # Determine the kernel 2D shape
        nb_coeff = len(coef_low)
        if horizontal:
            kernel_size = (1, nb_coeff)
            stride = (1, 2)
            pad = (nb_coeff // 2, nb_coeff - 1 - nb_coeff // 2, 0, 0)
            weights_low = self._coef_h(in_planes, coef_low)
            weights_high = self._coef_h(in_planes, coef_high)
        else:
            kernel_size = (nb_coeff, 1)
            stride = (2, 1)
            pad = (0, 0, nb_coeff // 2, nb_coeff - 1 - nb_coeff // 2)
            weights_low = self._coef_v(in_planes, coef_low)
            weights_high = self._coef_v(in_planes, coef_high)
        # TODO: Debug prints
        # print("")
        # print("Informations: ")
        # print("- kernel_size: ", kernel_size)
        # print("- stride     : ", stride)
        # print("- pad        : ", pad)
        # print("- low        : ", weights_low)
        # print("- high       : ", weights_high)

        # Create the conv2D
        self.conv_high = nn.Conv2d(
            in_planes, in_planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv_low = nn.Conv2d(
            in_planes, in_planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.padding = nn.ReflectionPad2d(padding=pad)
        # TODO: Debug prints
        # print("- low        : ", self.conv_low.weight)
        # print("- high       : ", self.conv_high.weight)

        # Replace their weights
        self.conv_high.weight = torch.nn.Parameter(
            data=torch.Tensor(weights_high), requires_grad=False)
        self.conv_low.weight = torch.nn.Parameter(
            data=torch.Tensor(weights_low), requires_grad=False)

    def forward(self, x):
        '''Returns the approximation and detail part'''
        x = self.padding(x)
        return (self.conv_low(x), self.conv_high(x))


class Wavelet2D(nn.Module):
    def __init__(self, in_planes, name="db1"):
        super(Wavelet2D, self).__init__()
        self.horizontal_wavelet = Wavelet(
            in_planes, horizontal=True, name=name)
        self.vertical_wavelet = Wavelet(in_planes, horizontal=False, name=name)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.horizontal_wavelet(x)
        (LL, LH) = self.vertical_wavelet(c)
        (HL, HH) = self.vertical_wavelet(d)
        return (LL, LH, HL, HH)


class LiftingScheme(nn.Module):
    def __init__(self, horizontal, in_planes, modified=True, size=[], splitting=True, k_size=4, simple_lifting=False):
        super(LiftingScheme, self).__init__()
        self.modified = modified
        if horizontal:
            kernel_size = (1, k_size)
            pad = (k_size // 2, k_size - 1 - k_size // 2, 0, 0)
        else:
            kernel_size = (k_size, 1)
            pad = (0, 0, k_size // 2, k_size - 1 - k_size // 2)

        self.splitting = splitting
        self.split = Splitting(horizontal)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:            
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
        else:
            size_hidden = 2
            
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ]
            prev_size = size_hidden

            # Final dense
            modules_P += [
                nn.Conv2d(in_planes*prev_size, in_planes,
                          kernel_size=(1, 1), stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.Conv2d(in_planes*prev_size, in_planes,
                          kernel_size=(1, 1), stride=1),
                nn.Tanh()
            ]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c)
            return (c, d)
        else:
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return (c, d)


class LiftingScheme2D(nn.Module):
    def __init__(self, in_planes, share_weights, modified=True, size=[2, 1], kernel_size=4, simple_lifting=False):
        super(LiftingScheme2D, self).__init__()
        self.level1_lf = LiftingScheme(
            horizontal=HORIZONTAL_FIRST, in_planes=in_planes, modified=modified,
            size=size, k_size=kernel_size, simple_lifting=simple_lifting)
        self.share_weights = share_weights
        if share_weights:
            self.level2_1_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)
            self.level2_2_lf = self.level2_1_lf  # Double check this
        else:
            self.level2_1_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)
            self.level2_2_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.level1_lf(x)
        (LL, LH) = self.level2_1_lf(c)
        (HL, HH) = self.level2_2_lf(d)
        return (c, d, LL, LH, HL, HH)


if __name__ == "__main__":
    input = torch.randn(1, 1, 10, 10)
    #m_harr = WaveletLiftingHaar2D()
    m_wavelet = Wavelet2D(1, name="db2")
    print(input)
    print(m_wavelet(input))

    # TODO: Do more experiments with the code
