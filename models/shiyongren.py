import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def vol_apply(x, i):
    if i == 0:
        x = (-231.719219020378 * torch.pow(x, 9)
             + 808.538255174754 * torch.pow(x, 8)
             - 1035.47643433502 * torch.pow(x, 7)
             + 543.873548058793 * torch.pow(x, 6)
             - 36.1882532770725 * torch.pow(x, 5)
             - 76.7131007160318 * torch.pow(x, 4)
             + 34.2911164473991 * torch.pow(x, 3)
             - 6.64858514399416 * torch.pow(x, 2)
             + 1.46164279870724 * x)
    return x


class AngularPropagation(nn.Module):
    def __init__(self, distance, n_padd, pixel_pitch, lambd_list, size):
        super(AngularPropagation, self).__init__()
        self.size = size
        self.com_pitch = 2 * n_padd + self.size
        self.dx = pixel_pitch
        self.distance_non_dimension = distance / self.dx
        self.lambda_non_dimension = (lambd_list / self.dx).unsqueeze(-1).unsqueeze(-1)
        self.k_non_dimension = 2 * torch.pi / self.lambda_non_dimension
        fx = torch.linspace(-self.com_pitch / 2 + 1, self.com_pitch / 2, self.com_pitch)
        fy = fx
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')

        deter_matrix = 1 - (self.lambda_non_dimension ** 2 / self.com_pitch ** 2) * (FX ** 2 + FY ** 2)
        mask = deter_matrix < 0
        deter_matrix = torch.clamp(deter_matrix, min=0)
        deter_matrix = torch.exp(1j * self.distance_non_dimension * self.k_non_dimension * torch.sqrt(deter_matrix))
        deter_matrix[mask] = 0
        deter_matrix = torch.fft.fftshift(deter_matrix, (1, 2))
        self.h = nn.Parameter(deter_matrix, requires_grad=False)

    def forward(self, x, i):
        x = torch.fft.fft2(x)
        x = x * self.h[i]
        x = torch.fft.ifft2(x)
        return x


class ModulationLayer(nn.Module):
    def __init__(self, device, n_padd, size):
        super(ModulationLayer, self).__init__()
        self.padd = n_padd
        self.normalized_voltage = nn.Parameter(torch.randn(size, size,
                                                           requires_grad=True, dtype=torch.float32, device=device))

    def forward(self, x, i):
        phase = torch.sigmoid(self.normalized_voltage)
        phase = vol_apply(phase, i)
        phase = F.pad(torch.exp(1j * 2 * torch.pi * phase),
                      pad=(self.padd, self.padd, self.padd, self.padd))
        x = x * phase
        return x


class DetectionPlane(nn.Module):
    def __init__(self, margin, total_pad):
        super(DetectionPlane, self).__init__()
        self.start, self.end = total_pad + margin, -(total_pad + margin)

    def forward(self, x):
        x = x[:, self.start:self.end, self.start:self.end]
        x = (x * torch.conj(x)).real
        return x


class D2NNmodel(nn.Module):
    def __init__(self, pixel_pitch, distance, n_padd, device, lambd_list, size, margin):
        super(D2NNmodel, self).__init__()
        self.total_pad = n_padd
        self.propagation_1 = AngularPropagation(distance[0], n_padd, pixel_pitch, lambd_list, size)
        self.propagation_2 = AngularPropagation(distance[1], n_padd, pixel_pitch, lambd_list, size)
        self.propagation_3 = AngularPropagation(distance[2], n_padd, pixel_pitch, lambd_list, size)
        self.propagation_4 = AngularPropagation(distance[3], n_padd, pixel_pitch, lambd_list, size)
        self.modulation_1 = ModulationLayer(device, n_padd, size)
        self.modulation_2 = ModulationLayer(device, n_padd, size)
        self.modulation_3 = ModulationLayer(device, n_padd, size)
        self.modulation_4 = ModulationLayer(device, n_padd, size)
        self.detect = DetectionPlane(margin, self.total_pad)

    def featurizer(self, x, i):
        x = F.pad(x, pad=(self.total_pad, self.total_pad, self.total_pad, self.total_pad))
        x = self.modulation_1(x, i)
        x = self.propagation_1(x, i)
        x = self.modulation_2(x, i)
        x = self.propagation_2(x, i)
        x = self.modulation_3(x, i)
        x = self.propagation_3(x, i)
        x = self.modulation_4(x, i)
        x = self.propagation_4(x, i)
        x = self.detect(x)
        return x

    def forward(self, x, i):
        x = self.featurizer(x, i)
        return x
