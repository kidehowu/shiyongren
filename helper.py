import matplotlib.pyplot as plt
import scipy.io
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split


def random_matrices_generation(N, H, W):
    matrices_data = torch.rand((N, H, W))
    phase_matrices = torch.rand((N, H, W)) * 2 * torch.pi
    matrices_data = matrices_data * torch.exp(1j * phase_matrices)
    return matrices_data


def loader_list_generation(n, n_multiplexing, input_fov, batch_size):
    transform_dimension = input_fov ** 2
    tr_dataloaders = []
    te_dataloaders = []
    train_size = int(0.9 * n)
    test_size = n - train_size

    target_transform_matrices = random_matrices_generation(n_multiplexing, transform_dimension, transform_dimension)
    for i in range(n_multiplexing):
        data_matrices = random_matrices_generation(n, input_fov, input_fov)
        label_matrices = data_matrices.view(data_matrices.size(0), -1, 1)
        label_matrices = torch.matmul(target_transform_matrices[i], label_matrices)
        label_matrices = label_matrices.view(label_matrices.size(0), input_fov, -1)
        dataset = TensorDataset(data_matrices, label_matrices)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        tr_dataloaders.append(train_loader)
        te_dataloaders.append(test_loader)
    return target_transform_matrices, tr_dataloaders, te_dataloaders


def plot_cosine_similarity_matrix(data):
    data = data.view(len(data), -1)
    data = data / data.norm(dim=1, keepdim=True)
    data = torch.abs(torch.mm(data, torch.conj(data.t())).real)

    fig = plt.figure()
    plt.imshow(data, cmap=plt.cm.Blues)
    thresh = data.max() / 2
    for i in range(len(data)):
        for j in range(len(data)):
            info = data[j, i]
            plt.text(i, j, format(info, ".2f"),
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")
    plt.title("Cosine Similarity Matrix")
    plt.show()
    return fig


def mse_loss():
    def mse(output, target):
        denomi = torch.norm(output, dim=(1, 2)) ** 2
        nomi = (target * output).sum(dim=(1, 2))
        factor = (nomi / denomi).unsqueeze(-1).unsqueeze(-1)
        x = target - factor * output
        # x = target - output
        x = (x ** 2).sum()
        return x

    return mse


def flat_top_gaussian(x, y, x0, y0, sigma_x, sigma_y, A, flat_height):
    gauss = A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))
    flat_gauss = np.minimum(gauss, flat_height)
    return flat_gauss


def convex_loss():
    x = np.arange(800)
    y = x
    x, y = np.meshgrid(x, y)

    # Parameters
    x0, y0 = 400, 400  # center
    sigma_x, sigma_y = 150, 150  # widths
    A = 1  # amplitude
    flat_height = 0.6  # Height at which to truncate the peak

    # Generate the flat-top Gaussian
    z = torch.from_numpy(flat_top_gaussian(x, y, x0, y0, sigma_x, sigma_y, A, flat_height)).to(torch.float32).to('cuda')

    def convex(output):
        x = -(output * z).sum()
        return x

    return convex


def shiyong_loader(load_dic, batch_size):
    dataloaders = []
    datasets0 = []
    datasets1 = []
    for dic in load_dic:
        mat_data = scipy.io.loadmat(dic['path'])
        a = torch.from_numpy(mat_data[dic['vari_name']]).permute(2, 0, 1, 3)
        a = F.normalize(a, dim=(1, 2)).to(torch.complex64)
        if dic['path'] == 'output_field_lamda_12.mat':
            a = (a * torch.conj(a)).real.to(torch.float32)
            # a[:6, :, :, 1] = torch.cat((a[:6, :, 100:, 1], a[:6, :, :100, 1]), dim=2)
            # a[6:, :, :, 1] = torch.cat((a[6:, :, 300:, 1], a[6:, :, :300, 1]), dim=2)
            # a = F.pad(a, pad=(0, 0, 200, 200, 200, 200))
        if dic['path'] == 'input_field_lamda_12.mat':
            a = torch.cat((a[:6, :, :, :], torch.flip(a[6:, :, :, :], [0])), dim=0)
        datasets0.append(a[:, :, :, 0])
        datasets1.append(a[:, :, :, 1])
    dataset0 = TensorDataset(datasets0[0], datasets0[1])
    dataset1 = TensorDataset(datasets1[0], datasets1[1])
    dataloader = DataLoader(dataset0, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dataloaders.append(dataloader)
    dataloader = DataLoader(dataset1, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dataloaders.append(dataloader)
    return dataloaders
