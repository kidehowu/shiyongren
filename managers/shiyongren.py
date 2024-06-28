import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR


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
        self.alph = torch.ones(2, device=self.device, requires_grad=False)
        self.n_multiplexing = n_multiplexing
        self.efficiency_1 = []
        self.efficiency_2 = []
        self.y_label1 = torch.arange(0, 12).to(self.device)
        self.y_label2 = torch.arange(12, 24).to(self.device)

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
        x1, y1 = next(iter(self.train_loaders_list[0]))
        x2, y2 = next(iter(self.train_loaders_list[1]))
        self.x1 = x1.to(self.device)
        self.y1 = y1.to(self.device)
        self.x2 = x2.to(self.device)
        self.y2 = y2.to(self.device)

    def _make_train_step_fn(self):
        def perform_train_step_fn():
            self.model.train()
            # first_iter = True
            loss = 0.0
            yhat1 = self.model(self.x1, 0)
            yhat2 = self.model(self.x2, 1)
            loss = loss + self.loss_fn(yhat1, self.y_label1)
            loss = loss + self.loss_fn(yhat2, self.y_label2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(iters_list):
            self.model.eval()
            # first_iter = True
            losses = []
            yhat1 = self.model(self.x1, 0)
            yhat2 = self.model(self.x2, 1)
            losses.append(self.loss_fn(yhat1, self.y1))
            losses.append(self.loss_fn(yhat2, self.y2))
            loss = torch.stack(losses)
            loss = (loss * self.alph).mean()
            return loss.item()

        return perform_val_step_fn

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
        for epoch in range(n_epochs):
            self.total_epochs += 1
            loss = self.train_step_fn()
            self.losses.append(loss)

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

    def plot_efficiency(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.efficiency_1, label='efficiency_1 Loss', c='b')
        plt.plot(self.efficiency_2, label='efficiency_2 Loss', c='r')
        plt.ylim([0, 1])
        plt.xlabel('Epochs')
        plt.ylabel('Efficiency')
        plt.legend()
        plt.tight_layout()
        return fig

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    @staticmethod
    def _visualize_phases(fig, axs, x, y=None, yhat=None, layer_name='', title=None):
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
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
        # Set custom tick labels

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

    def visualize_voltage_mask(self, layers_name):
        n_weights = []
        for name in layers_name:
            layer = getattr(self.model, name)
            weights = layer.normalized_voltage.data.detach().cpu()
            weights = torch.sigmoid(weights).numpy()
            n_weights.append(weights)

        n_channels = len(layers_name)
        fig, axes = plt.subplots(1, n_channels, figsize=(2 * n_channels + 1, 2))
        Manager._visualize_phases(
            fig,
            axes,
            n_weights,
        )
        return fig

    def _helper_plot(self, fig, dic):
        for i in range(len(dic['axes'])):
            ax = dic['axes'][i]
            cax = ax.imshow(dic['x'][i], cmap=dic['colormap'], vmin=dic['min'], vmax=dic['max'])
            ax.set_title(dic['title'].format(i + 1))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        return

    def figure_1(self, outputs):
        x, y = self.x1.cpu(), self.y1.cpu()
        x2, y2 = self.x2.cpu(), self.y2.cpu()
        x_abs = torch.abs(x)
        x_ang = torch.angle(x)

        fig, axes = plt.subplots(12, 6, figsize=(3 * 6, 3 * 12))
        axes = axes.reshape(12, 6)
        dics = [
            {'x': x_abs, 'axes': axes[:, 0], 'title': '$|Input_{}|$', 'colormap': 'jet', 'min': 0,
             'max': torch.max(x_abs)},
            {'x': x_ang, 'axes': axes[:, 1], 'title': '$\\angle Input_{}$', 'colormap': 'twilight', 'min': -np.pi,
             'max': np.pi},
            {'x': outputs[:12, :, :], 'axes': axes[:, 2], 'title': '$\\lambda_1\\ |Output_{}|$', 'colormap': 'gray', 'min': 0,
             'max': None},
            {'x': outputs[12:, :, :], 'axes': axes[:, 3], 'title': '$\\lambda_2\\ |Output_{}|$', 'colormap': 'gray', 'min': 0,
             'max': None},
            {'x': y, 'axes': axes[:, 4], 'title': '$|Target_{}|$', 'colormap': 'gray', 'min': 0,
             'max': torch.max(y)},
            {'x': y2, 'axes': axes[:, 5], 'title': '$|Target_{}|$', 'colormap': 'gray', 'min': 0,
             'max': torch.max(y2)}
        ]
        for dic in dics:
            self._helper_plot(fig, dic)
        plt.show()
        return fig

    def eff(self):
        with torch.no_grad():
            self.model.eval()
            yhat = self.model(self.x1, 0).sum(dim=(1, 2)).mean().detach().cpu()
            print(yhat, '1')
            yhat = self.model(self.x2, 1).sum(dim=(1, 2)).mean().detach().cpu()
            print(yhat, '2')

    def lr_range_test(self, data_loader, end_lr, num_iter=100, step_mode='exp', alpha=0.05, ax=None):
        # Since the test updates both model and optimizer we need to store
        # their initial states to restore them in the end
        previous_states = {'model': deepcopy(self.model.state_dict()),
                           'optimizer': deepcopy(self.optimizer.state_dict())}
        # Retrieves the learning rate set in the optimizer
        start_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        # Builds a custom function and corresponding scheduler
        lr_fn = make_lr_fn(start_lr, end_lr, num_iter)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Variables for tracking results and iterations
        tracking = {'loss': [], 'lr': []}
        iteration = 0

        # If there are more iterations than mini-batches in the data loader,
        # it will have to loop over it more than once
        while (iteration < num_iter):
            # That's the typical mini-batch inner loop
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # Step 1
                yhat = self.model(x_batch, 0)
                # Step 2
                loss = self.loss_fn(yhat, y_batch)
                # Step 3
                loss.backward()

                # Here we keep track of the losses (smoothed)
                # and the learning rates
                tracking['lr'].append(scheduler.get_last_lr()[0])
                if iteration == 0:
                    tracking['loss'].append(loss.item())
                else:
                    prev_loss = tracking['loss'][-1]
                    smoothed_loss = alpha * loss.item() + (1 - alpha) * prev_loss
                    tracking['loss'].append(smoothed_loss)

                iteration += 1
                # Number of iterations reached
                if iteration == num_iter:
                    break

                # Step 4
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

        # Restores the original states
        self.optimizer.load_state_dict(previous_states['optimizer'])
        self.model.load_state_dict(previous_states['model'])

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.get_figure()
        ax.plot(tracking['lr'], tracking['loss'])
        if step_mode == 'exp':
            ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        return tracking, fig

    def get_output(self, layers):
        self.attach_hooks(layers)
        with torch.no_grad():
            self.model.eval()
            yhat1 = self.model(self.x1, 0)
            yhat2 = self.model(self.x2, 1)
        self.model.train()
        self.remove_hooks()
        return self.visualization['detect']