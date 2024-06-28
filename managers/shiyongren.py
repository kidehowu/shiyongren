import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import sys


def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_lr_fn(start_lr, end_lr, num_iter, step_mode='exp'):
    if step_mode == 'linear':
        factor = (end_lr / start_lr - 1) / num_iter

        def lr_fn(iteration):
            return 1 + iteration * factor
    else:
        factor = (np.log(end_lr) - np.log(start_lr)) / num_iter

        def lr_fn(iteration):
            return np.exp(factor) ** iteration
    return lr_fn


class Manager(object):
    def __init__(self, model, loss_fn, optimizer, n_multiplexing):
        # Here we define the attributes of our class

        # We start by storing the arguments as attributes
        # to use them later
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Let's send the model to the specified device right away
        self.model.to(self.device)

        # These attributes are defined here, but since they are
        # not informed at the moment of creation, we keep them None
        self.train_loaders_list = None
        self.val_loaders_list = None

        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.total_epochs = 0
        self.scheduler = None
        self.is_batch_lr_scheduler = False

        self.visualization = {}
        self.handles = {}
        self.alph = torch.ones(n_multiplexing, device=self.device, requires_grad=False)
        self.n_multiplexing = n_multiplexing

        # Creates the train_step function for our model,
        # loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class
        # attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Creates the val_step function for our model and loss
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loaders_list, val_loaders_list=None):
        # This method allows the user to define which train_loaders_list (and val_loaders_list, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loaders_list = train_loaders_list
        self.val_loaders_list = val_loaders_list

    def _make_train_step_fn(self):
        def perform_train_step_fn(iters_list):
            self.model.train()
            # first_iter = True
            losses = []
            for i, it in enumerate(iters_list):
                x, y = next(it)
                x = x.to(self.device)
                y = y.to(self.device)
                yhat = self.model(x, i)
                losses.append(self.loss_fn(yhat, y))
            loss = torch.stack(losses)
            ref_loss = loss.detach()
            loss = (loss * self.alph).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.alph = torch.clamp(0.1 * (ref_loss - ref_loss.mean()) + self.alph, min=0)
            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(iters_list):
            self.model.eval()
            # first_iter = True
            losses = []
            for i, it in enumerate(iters_list):
                x, y = next(it)
                x = x.to(self.device)
                y = y.to(self.device)
                yhat = self.model(x, i)
                losses.append(self.loss_fn(yhat, y))
            loss = torch.stack(losses)
            loss = (loss * self.alph).mean()
            return loss.item()

        return perform_val_step_fn

    def _run_all_mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and
        # corresponding step function is going to be used
        if validation:
            data_loaders_list = self.val_loaders_list
            step_fn = self.val_step_fn
        else:
            data_loaders_list = self.train_loaders_list
            step_fn = self.train_step_fn

        if data_loaders_list is None:
            return None

        data_loader_iterators = [iter(loader) for loader in data_loaders_list]
        mini_batch_losses = []
        for i in range(len(data_loaders_list[0])):
            mini_batch_loss = step_fn(data_loader_iterators)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def _mini_batch_schedulers(self, frac_epoch):
        if self.scheduler:
            if self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.scheduler.step(self.total_epochs + frac_epoch)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)

    def _epoch_schedulers(self, val_loss):
        if self.scheduler:
            if not self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loaders_list.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def train(self, n_epochs, seed=42):
        # To ensure reproducibility of the training process
        self.set_seed(seed)

        for epoch in range(n_epochs):
            self.total_epochs += 1
            loss = self._run_all_mini_batch(validation=False)
            self.losses.append(loss)

            with torch.no_grad():
                val_loss = self._run_all_mini_batch(validation=True)
                self.val_losses.append(val_loss)

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses,
                      'alpha': self.alph}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename, dummy=None):
        # Loads dictionary
        checkpoint = torch.load(filename)
        if dummy is not None:
            checkpoint2 = torch.load(f'saved_model/dummies/{dummy}.pth')
            checkpoint['model_state_dict']['propagation.h'] = checkpoint2['model_state_dict']['propagation.h']
            checkpoint['model_state_dict']['intensity_sum.gaussians'] = checkpoint2['model_state_dict'][
                'intensity_sum.gaussians']

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.alph = checkpoint['alpha'].to(self.device)

        self.model.train()  # always use TRAIN for resuming training

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None, layer_name='', title=None):
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # Sets title, labels, and removes ticks
            if title is not None:
                ax.set_title('{} #{}'.format(title, j), fontsize=12)
            '''ax.set_ylabel(
                '{}\n{}x{}'.format(layer_name, *np.atleast_2d(image).shape),
                rotation=0, labelpad=40
            )'''
            xlabel1 = '' if y is None else '\nLabel: {}'.format(y[j])
            xlabel2 = '' if yhat is None else '\nPredicted: {}'.format(yhat[j])
            xlabel = '{}{}'.format(xlabel1, xlabel2)
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plots weight as an image
            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='gray',
                vmin=minv,
                vmax=maxv
            )
        return

    @staticmethod
    def _visualize_phases(fig, axs, x, y=None, yhat=None, layer_name='', title=None):
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        lambd_mean = 0.8e-3
        hmin = 0.25 * lambd_mean
        hmax = 1.5 * lambd_mean
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # Sets title, labels, and removes ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Plots weight as an image
            im = ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='viridis',
                vmin=minv,
                vmax=maxv
            )
        cbar_ax = fig.add_axes([0.92, 0.12, 0.01, 0.75])  # Adjust these values as needed
        cbar = fig.colorbar(im, cax=cbar_ax)
        # Set custom tick positions
        cbar.set_ticks([hmin, hmax])
        # Set custom tick labels
        cbar.set_ticklabels(['0.25 $\\lambda$', '1.50 $\\lambda$'])

        return

    def attach_hooks(self, layers_to_hook, hook_fn=None):
        # Clear any previous values
        self.visualization = {}
        # Creates the dictionary to map layer objects to their names
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]}

        if hook_fn is None:
            # Hook function to be attached to the forward pass
            def hook_fn(layer, inputs, outputs):
                # Gets the layer name
                name = layer_names[layer]
                # Detaches outputs
                values = outputs.detach().cpu().numpy()
                # Since the hook function may be called multiple times
                # for example, if we make predictions for multiple mini-batches
                # it concatenates the results
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([self.visualization[name], values])

        for name, layer in modules:
            # If the layer is in our list
            if name in layers_to_hook:
                # Initializes the corresponding key in the dictionary
                self.visualization[name] = None
                # Register the forward hook and keep the handle in another dict
                self.handles[name] = layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        # Loops through all hooks and removes them
        for handle in self.handles.values():
            handle.remove()
        # Clear the dict, as all hooks have been removed
        self.handles = {}

    def visualize_height_mask(self, layers_name, hmin, hmax, bit_depth):
        n_weights = []
        linspace = np.linspace(hmin, hmax, 2 ** bit_depth)
        for name in layers_name:
            layer = getattr(self.model, name)
            weights = layer.height.data.detach().cpu().numpy()
            weights = (np.sin(weights) + 1) / 2 * (2 ** bit_depth - 1)
            weights = np.round(weights).astype(int)
            weights = linspace[weights]
            n_weights.append(weights)

        n_channels = len(layers_name)
        fig, axes = plt.subplots(1, n_channels, figsize=(2 * n_channels + 1, 2))
        Manager._visualize_phases(
            fig,
            axes,
            n_weights,
        )
        return fig

    def get_transform_tensor(self, A):
        input_fov = self.model.input_fov
        x = torch.eye(input_fov ** 2)
        x = x.view((input_fov ** 2, input_fov, input_fov))
        A_hat = []
        with torch.no_grad():
            self.model.eval()
            x = x.to(self.device)
            for i in range(self.n_multiplexing):
                yhat = self.model(x, i)
                yhat = yhat.view(yhat.size(0), -1)
                yhat = torch.t(yhat)
                A_hat.append(yhat)
            A_hat = torch.stack(A_hat).detach().cpu()
        denomi = torch.norm(A_hat, dim=(1, 2)) ** 2
        nomi = (A * torch.conj(A_hat)).sum(dim=(1, 2))
        factor = (nomi / denomi).unsqueeze(-1).unsqueeze(-1)
        A_hat = factor * A_hat
        return A_hat

    def _helper_plot(self, fig, dic):
        for i in range(len(dic['axes'])):
            ax = dic['axes'][i]
            cax = ax.imshow(dic['x'][i], cmap=dic['colormap'], vmin=dic['min'], vmax=dic['max'])
            ax.set_title(dic['title'].format(i + 1))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        return

    def figure_1(self, A):
        Ah = self.get_transform_tensor(A)
        A_abs = torch.abs(A)
        A_ang = torch.angle(A)
        Ah_abs = torch.abs(Ah)
        Ah_ang = torch.angle(Ah)
        dA = torch.abs(A - Ah)
        mean_error = dA.mean()
        fig, axes = plt.subplots(4, 5, figsize=(3 * 5, 3 * 4))
        axes = axes.reshape(4, 5)
        dics = [
            {'x': A_abs, 'axes': axes[:, 0], 'title': '$|A_{}|$', 'colormap': 'gray', 'min': 0, 'max': 1},
            {'x': A_ang, 'axes': axes[:, 1], 'title': '$\\angle A_{}$', 'colormap': 'twilight', 'min': -np.pi,
             'max': np.pi},
            {'x': Ah_abs, 'axes': axes[:, 2], 'title': '$|\\hat A_{}|$', 'colormap': 'gray', 'min': 0, 'max': 1},
            {'x': Ah_ang, 'axes': axes[:, 3], 'title': '$\\angle \\hat A_{}$', 'colormap': 'twilight', 'min': -np.pi,
             'max': np.pi},
            {'x': dA, 'axes': axes[:, 4], 'title': '$|A - \\hat A_{}|$', 'colormap': 'afmhot', 'min': 0,
             'max': torch.max(dA)}
        ]
        for dic in dics:
            self._helper_plot(fig, dic)
        plt.show()
        return fig, mean_error

    def figure_2(self):
        a = iter(self.val_loaders_list[0])
        x, y = next(a)
        with torch.no_grad():
            self.model.eval()
            yhat = x.to(self.device)
            yhat = self.model(yhat, 0)
            yhat = yhat.detach().cpu()
        denomi = torch.norm(yhat, dim=(1, 2)) ** 2
        nomi = (y * torch.conj(yhat)).sum(dim=(1, 2))
        factor = (nomi / denomi).unsqueeze(-1).unsqueeze(-1)
        yh = factor * yhat
        x_abs = torch.abs(x)
        x_ang = torch.angle(x)
        y_abs = torch.abs(y)
        y_ang = torch.angle(y)
        yh_abs = torch.abs(yh)
        yh_ang = torch.angle(yh)
        dy_abs = torch.abs(y - yh)
        dy_ang = torch.abs(torch.angle(y) - torch.angle(yh))
        # mean_error = dy_abs.mean()
        fig, axes = plt.subplots(4, 8, figsize=(3.5 * 8, 3 * 4))
        axes = axes.reshape(4, 8)
        dics = [
            {'x': x_abs, 'axes': axes[:, 0], 'title': '$|x_{}|$', 'colormap': 'gray', 'min': 0,
             'max': torch.max(x_abs)},
            {'x': x_ang, 'axes': axes[:, 1], 'title': '$\\angle x_{}$', 'colormap': 'twilight', 'min': -np.pi,
             'max': np.pi},
            {'x': y_abs, 'axes': axes[:, 2], 'title': '$|y_{}|$', 'colormap': 'gray', 'min': 0, 'max': torch.max(y_abs)},
            {'x': y_ang, 'axes': axes[:, 3], 'title': '$\\angle y_{}$', 'colormap': 'twilight', 'min': -np.pi,
             'max': np.pi},
            {'x': yh_abs, 'axes': axes[:, 4], 'title': '$|\\hat y_{}|$', 'colormap': 'gray', 'min': 0, 'max': torch.max(yh_abs)},
            {'x': yh_ang, 'axes': axes[:, 5], 'title': '$\\angle \\hat y_{}$', 'colormap': 'twilight', 'min': -np.pi,
             'max': np.pi},
            {'x': dy_abs, 'axes': axes[:, 6], 'title': '$|y - \\hat y_{}|$', 'colormap': 'afmhot', 'min': 0,
             'max': torch.max(dy_abs)},
            {'x': dy_ang, 'axes': axes[:, 7], 'title': '$|\\angle y - \\angle \\hat y_{}|$', 'colormap': 'afmhot', 'min': 0,
             'max': np.pi}
        ]
        for dic in dics:
            self._helper_plot(fig, dic)
        plt.show()
        return fig
