"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['Scattering']

import warnings
import torch
from utils_scat import cdgmm, Modulus, Periodize, Fft
from torch.nn import ReflectionPad2d as pad_function
import numpy as np
import scipy.fftpack as fft

def filters_bank(M, N, J, L=8):
    filters = {}
    filters['psi'] = []


    offset_unpad = 0
    for j in range(J):
        for theta in range(L):
            psi = {}
            psi['j'] = j
            psi['theta'] = theta
            psi_signal = morlet_2d(M, N, 0.8 * 2**j, (int(L-L/2-1)-theta) * np.pi / L, 3.0 / 4.0 * np.pi /2**j,offset=offset_unpad) # The 5 is here just to match the LUA implementation :)
            psi_signal_fourier = fft.fft2(psi_signal)
            for res in range(j + 1):
                psi_signal_fourier_res = crop_freq(psi_signal_fourier, res)
                psi[res]=torch.FloatTensor(np.stack((np.real(psi_signal_fourier_res), np.imag(psi_signal_fourier_res)), axis=2))
                # Normalization to avoid doing it with the FFT!
                psi[res].div_(M*N// 2**(2*j))
            filters['psi'].append(psi)

    filters['phi'] = {}
    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0, offset=offset_unpad)
    phi_signal_fourier = fft.fft2(phi_signal)
    filters['phi']['j'] = J
    for res in range(J):
        phi_signal_fourier_res = crop_freq(phi_signal_fourier, res)
        filters['phi'][res]=torch.FloatTensor(np.stack((np.real(phi_signal_fourier_res), np.imag(phi_signal_fourier_res)), axis=2))
        filters['phi'][res].div_(M*N // 2 ** (2 * J))

    return filters


def crop_freq(x, res):
    M = x.shape[0]
    N = x.shape[1]

    crop = np.zeros((M // 2 ** res, N // 2 ** res), np.complex64)

    mask = np.ones(x.shape, np.float32)
    len_x = int(M * (1 - 2 ** (-res)))
    start_x = int(M * 2 ** (-res - 1))
    len_y = int(N * (1 - 2 ** (-res)))
    start_y = int(N * 2 ** (-res - 1))
    mask[start_x:start_x + len_x,:] = 0
    mask[:, start_y:start_y + len_y] = 0
    x = np.multiply(x,mask)

    for k in range(int(M / 2 ** res)):
        for l in range(int(N / 2 ** res)):
            for i in range(int(2 ** res)):
                for j in range(int(2 ** res)):
                    crop[k, l] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res)]

    return crop


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=None):
    """ This function generated a morlet"""
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset, fft_shift)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset, fft_shift)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=None):
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab = gab + np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab = gab / norm_factor

    if (fft_shift):
        gab = np.fft.fftshift(gab, axes=(0, 1))
    return gab

class Scattering(object):
    """Scattering module.
    Runs scattering on an input image in NCHW format
    Input args:
        M, N: input image size
        J: number of layers
        pre_pad: if set to True, module expect pre-padded images
        jit: compile kernels on the fly for speed
    """
    def __init__(self, M, N, J, pre_pad=False, jit=True):
        super(Scattering, self).__init__()
        self.M, self.N, self.J = M, N, J
        self.pre_pad = pre_pad
        self.jit = jit
        self.fft = Fft()
        self.modulus = Modulus(jit=jit)
        self.periodize = Periodize(jit=jit)

        self._prepare_padding_size([1, 1, M, N])

        self.padding_module = pad_function(2**J)

        # Create the filters
        filters = filters_bank(self.M_padded, self.N_padded, J)

        self.Psi = filters['psi']
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.padding_module.type(str(_type).split('\'')[1])
        return self

    def cuda(self):
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        return self._type(torch.FloatTensor)

    def _prepare_padding_size(self, s):
        M = s[-2]
        N = s[-1]

        self.M_padded = ((M + 2 ** (self.J))//2**self.J+1)*2**self.J
        self.N_padded = ((N + 2 ** (self.J))//2**self.J+1)*2**self.J

        if self.pre_pad:
            warnings.warn('Make sure you padded the input before to feed it!', RuntimeWarning, stacklevel=2)

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = torch.Size([a for a in s])

    # This function copies and view the real to complex
    def _pad(self, input):
        if(self.pre_pad):
            output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)
            output.narrow(output.ndimension()-1, 0, 1).copy_(input)
        else:
            out_ = self.padding_module(input)
            output = input.new(out_.size(0), out_.size(1), out_.size(2), out_.size(3), 2).fill_(0)
            output.narrow(4, 0, 1).copy_(out_.unsqueeze(4))
        return output

    def _unpad(self, in_):
        return in_[..., 1:-1, 1:-1]

    def forward(self, input):
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if((input.size(-1)!=self.N or input.size(-2)!=self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!'%(self.M,self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        J = self.J
        phi = self.Phi
        psi = self.Psi
        n = 0

        fft = self.fft
        periodize = self.periodize
        modulus = self.modulus
        pad = self._pad
        unpad = self._unpad

        S = input.new(input.size(0),
                      input.size(1),
                      1 + 8*J + 8*8*J*(J - 1) // 2,
                      self.M_padded//(2**J)-2,
                      self.N_padded//(2**J)-2)
        U_r = pad(input)
        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c

        # First low pass filter
        U_1_c = periodize(cdgmm(U_0_c, phi[0], jit=self.jit), k=2**J)

        U_J_r = fft(U_1_c, 'C2R')

        S[..., n, :, :].copy_(unpad(U_J_r))
        n = n + 1

        for n1 in range(len(psi)):
            j1 = psi[n1]['j']
            U_1_c = cdgmm(U_0_c, psi[n1][0], jit=self.jit)
            if(j1 > 0):
                U_1_c = periodize(U_1_c, k=2 ** j1)
            fft(U_1_c, 'C2C', inverse=True, inplace=True)
            U_1_c = fft(modulus(U_1_c), 'C2C')

            # Second low pass filter
            U_2_c = periodize(cdgmm(U_1_c, phi[j1], jit=self.jit), k=2**(J-j1))
            U_J_r = fft(U_2_c, 'C2R')
            S[..., n, :, :].copy_(unpad(U_J_r))
            n = n + 1

            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                if(j1 < j2):
                    U_2_c = periodize(cdgmm(U_1_c, psi[n2][j1], jit=self.jit), k=2 ** (j2-j1))
                    fft(U_2_c, 'C2C', inverse=True, inplace=True)
                    U_2_c = fft(modulus(U_2_c), 'C2C')

                    # Third low pass filter
                    U_2_c = periodize(cdgmm(U_2_c, phi[j2], jit=self.jit), k=2 ** (J-j2))
                    U_J_r = fft(U_2_c, 'C2R')

                    S[..., n, :, :].copy_(unpad(U_J_r))
                    n = n + 1

        return S

    def __call__(self, input):
        return self.forward(input)