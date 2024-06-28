import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AngularPropagation(nn.Module):
    def __init__(self, distance, n_padd, pixel_pitch, lambd_list, size):
        super(AngularPropagation, self).__init__()
        self.size = size
        self.com_pitch = 2 * n_padd + self.size
        self.dx = pixel_pitch
        self.distance_non_dimension = distance / self.dx
        self.lambda_non_dimension = torch.from_numpy(lambd_list / self.dx).unsqueeze(-1).unsqueeze(-1)
        self.k_non_dimension = 2 * torch.pi / self.lambda_non_dimension
        fx = torch.linspace(-self.com_pitch / 2 + 1, self.com_pitch / 2, self.com_pitch)
        fy = fx
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')

        deter_matrix = 1 - (self.lambda_non_dimension ** 2 / self.com_pitch ** 2) * (FX ** 2 + FY ** 2)
        mask = deter_matrix < 0
        deter_matrix = torch.clamp(deter_matrix, min=0)
        deter_matrix = torch.exp(1j * self.distance_non_dimension * self.k_non_dimension * torch.sqrt(deter_matrix))
        deter_matrix[mask] = 0
        deter_matrix = torch.fft.fftshift(deter_matrix, (1,2)).to(torch.complex64)
        self.h = nn.Parameter(deter_matrix, requires_grad=False)

    def forward(self, x, i):
        x = torch.fft.fft2(x)
        x = x * self.h[i]
        x = torch.fft.ifft2(x)
        return x


class ModulationLayer(nn.Module):
    def __init__(self, device, n_padd, size, lambd_list):
        super(ModulationLayer, self).__init__()
        self.padd = n_padd
        self.n_material = 1.72
        self.n_air = 1.00
        self.lambd_mean = 0.8e-3
        self.bit_depth = 8
        self.hmin = 0.25 * self.lambd_mean
        self.hmax = 1.5 * self.lambd_mean
        self.delta_k = nn.Parameter((self.n_material - self.n_air) * 2 * torch.pi / torch.from_numpy(lambd_list), requires_grad=False)
        self.linspace = nn.Parameter(torch.linspace(self.hmin, self.hmax, 2 ** self.bit_depth), requires_grad=False)
        self.height = nn.Parameter(torch.randn(size, size,
                                            requires_grad=True, dtype=torch.float32, device=device))

    def constrain_and_round(self):
        phase = (torch.sin(self.height) + 1) / 2 * (2 ** self.bit_depth - 1)
        phase = torch.round(phase).int()
        phase = self.linspace[phase]
        return phase

    def forward(self, x, i):
        phase = (torch.sin(self.height) + 1) / 2 * (self.hmax - self.hmin) + self.hmin
        phase = F.pad(torch.exp(1j * self.delta_k[i] * phase),
                          pad=(self.padd, self.padd, self.padd, self.padd))
        x = x * phase
        return x


class PoolLayer(nn.Module):
    def __init__(self, margin, total_pad, bin):
        super(PoolLayer, self).__init__()
        self.start, self.end = total_pad + margin, -(total_pad + margin)
        self.pool = nn.AvgPool2d((bin, bin))

    def forward(self, x):
        x = x[:, self.start:self.end, self.start:self.end]
        x = torch.complex(self.pool(x.real), self.pool(x.imag))
        return x


class D2NNmodel(nn.Module):
    def __init__(self, pixel_pitch, distance, n_padd, device, lambd_list, size, margin, input_fov, binning_factor):
        super(D2NNmodel, self).__init__()
        self.bin = binning_factor
        self.input_fov = input_fov
        self.fov_pad = (size - input_fov*binning_factor) // 2
        self.total_pad = self.fov_pad + n_padd
        self.propagation = AngularPropagation(distance, n_padd, pixel_pitch, lambd_list, size)
        self.modulation_1 = ModulationLayer(device, n_padd, size, lambd_list)
        self.modulation_2 = ModulationLayer(device, n_padd, size, lambd_list)
        self.modulation_3 = ModulationLayer(device, n_padd, size, lambd_list)
        self.modulation_4 = ModulationLayer(device, n_padd, size, lambd_list)
        self.modulation_5 = ModulationLayer(device, n_padd, size, lambd_list)
        self.modulation_6 = ModulationLayer(device, n_padd, size, lambd_list)
        self.pool = PoolLayer(margin, self.total_pad, self.bin)

    def featurizer(self, x, i):
        x = x.repeat_interleave(self.bin, dim=1).repeat_interleave(self.bin, dim=2)
        x = F.pad(x, pad=(self.total_pad, self.total_pad, self.total_pad, self.total_pad))
        x = self.propagation(x, i)
        x = self.modulation_1(x, i)
        x = self.propagation(x, i)
        x = self.modulation_2(x, i)
        x = self.propagation(x, i)
        x = self.modulation_3(x, i)
        x = self.propagation(x, i)
        x = self.modulation_4(x, i)
        x = self.propagation(x, i)
        x = self.modulation_5(x, i)
        x = self.propagation(x, i)
        x = self.modulation_6(x, i)
        x = self.propagation(x, i)
        x = self.pool(x)
        return x

    def forward(self, x, i):
        x = self.featurizer(x, i)
        return x
