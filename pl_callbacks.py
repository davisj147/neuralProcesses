from pytorch_lightning.callbacks.base import Callback
import wandb

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pl_modules import process_data_to_points, batch_img_to_functional

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel

from torchvision.utils import make_grid



class WandbLogPriorPosteriorSamplePlots(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _visualise_prior_1d(self, pl_module):
        x_target = torch.Tensor(np.linspace(-1, 1, 100))
        x_target = x_target.unsqueeze(1).unsqueeze(0).to(pl_module.device)

        for i in range(64):
            z_sample = torch.randn((1, pl_module.z_dim)).to(pl_module.device)
            mu, _ = pl_module.model.decoder(x_target, z_sample)
            plt.plot(x_target.cpu().numpy()[0], mu.detach().cpu().numpy()[0],
                        c='b', alpha=0.5)
            # plt.xlim(-1, 1)
        
        wandb.log({"prior_samples": plt})

    
    def _visualise_prior_img(self, trainer, pl_module):
        x, y = next(iter(trainer.datamodule.val_dataloader()))
        _, channels, img_h, img_w = x.shape
        xs, _ = batch_img_to_functional(x)
        x_target = xs[0, :, :].unsqueeze(0).to(pl_module.device)

        samples = []
        for i in range(6):
            z_sample = torch.randn((1, pl_module.z_dim)).to(pl_module.device)
            mu, _ = pl_module.model.decoder(x_target, z_sample)
            img_mu = mu.permute(0,2,1).reshape((channels, img_h, img_w)).detach().cpu()
            samples.append(img_mu)
        
        grid = make_grid(samples, nrow=3, pad_value=1.)
        plt.imshow(grid.permute(1, 2, 0).numpy())
            # plt.xlim(-1, 1)

        wandb.log({"prior_samples": plt})
    
    
    def _visualise_posterior_1d(self, trainer, pl_module):
         # Visualize samples from posterior
        # Extract a batch from data_loader
        # Use batch to create random set of context points

        pl_module.eval()

        # x, y = x[(sample_id-1):sample_id], y[(sample_id-1):sample_id]
        if trainer.datamodule.dataset_type == "gpdata":
            rng = np.random.default_rng()
            x, y, l, s, p = trainer.datamodule.val_dataloader().dataset.generate_gp_sample(rng)
            x, y = torch.tensor(x).float().to(pl_module.device).unsqueeze(0), torch.tensor(y).float().to(pl_module.device).unsqueeze(0).unsqueeze(2)
            kernel_type = trainer.datamodule.val_dataloader().dataset.kernel

        else:
            x, y = next(iter(trainer.datamodule.val_dataloader()))
            sample_id = random.randint(1, x.shape[0])
            x, y = x[(sample_id-1):sample_id], y[(sample_id-1):sample_id]
            kernel_type = None

        fig, axs = plt.subplots(2,2, figsize=(18, 6))
        flat_axs = axs.flatten()
        
        # plt.clf()
        for j, n_context in enumerate([4, 8, 16, 64]):

            x_context, y_context, _, _ = process_data_to_points(x, y, n_context)

            # Create a set of target points corresponding to entire [-pi, pi] range
            x_target = torch.Tensor(np.linspace(torch.min(x).item(), torch.max(x).item(), 100))
            x_target = x_target.unsqueeze(1).unsqueeze(0)
            for i in range(32):
                # Neural process returns distribution over y_target
                p_y_pred, _, _ = pl_module.model(x_context.to(pl_module.device),
                                                    y_context.to(pl_module.device),
                                                    x_target.to(pl_module.device), None)
                # Extract mean of distribution
                mu = p_y_pred[0].loc.detach()
                flat_axs[j].plot(x_target.cpu().numpy()[0], mu.cpu().numpy()[0],
                            alpha=0.1, c='b')
                flat_axs[j].plot(x.cpu().numpy()[0], y.cpu().numpy()[0],
                            alpha=0.7, c='r')

            flat_axs[j].scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c='k')
        wandb.log({"posterior_samples": fig})

        if trainer.datamodule.dataset_type == "gpdata":
            fig, axs = plt.subplots(2,2, figsize=(18, 6))
            flat_axs = axs.flatten()

            for j, n_context in enumerate([4, 8, 16, 64]):
                _visualise_with_gp_comparison(flat_axs[j], x, y, l, s, p, kernel_type, pl_module, n_context, n=1)
            wandb.log({"posterior_single_sample": wandb.Image(fig)})

            plt.clf()


    def _visualise_posterior_img(self, trainer, pl_module):
        x, y = next(iter(trainer.datamodule.val_dataloader()))
        _, channels, img_h, img_w = x.shape
        img_id = random.randint(1, x.shape[0])
        imgs = []
        # for n_context in [8, 32, 64, 256, 28*28]:
        for n_context in [8, 32, 64, 256, trainer.datamodule.val_dataloader().dataset.img_size**2]:
            # x_context, y_context, _, _ = process_data_to_points(x[(i-1):i], y[(i-1):i],
            #                                                     pl_module.num_context)
            x_context, y_context, _, _ = process_data_to_points(x[(img_id-1):img_id], y[(img_id-1):img_id],
                                                                n_context)
            # create target points for the full image
            xs, _ = batch_img_to_functional(x)
            x_target = xs[0, :, :].unsqueeze(0)  

            pl_module.eval()

            # imgs = [context_to_img(x_context, y_context, img_h, img_w)]
            imgs.append(context_to_img(x_context, y_context, img_h, img_w))
            for i in range(5):
                # Neural process returns distribution over y_target
                p_y_pred, _, _ = pl_module.model(x_context.to(pl_module.device),
                                                    y_context.to(pl_module.device),
                                                    x_target.to(pl_module.device), None)
                # Extract mean of distribution
                mu = p_y_pred[0].loc.detach()
                img_mu = mu.permute(0,2,1).reshape((channels, img_h, img_w)).detach().cpu()
                imgs.append(img_mu)
        grid = make_grid(imgs, nrow=6, pad_value=1.)
        plt.imshow(grid.permute(1, 2, 0).numpy())
        wandb.log({"posterior_samples": plt})


    def on_validation_epoch_end(self, trainer, pl_module):
        # Visualize samples from trained prior
        # The prior should now encode some information about the shapes of the functions.

        if not trainer.running_sanity_check:
            torch.manual_seed(0)

            if trainer.datamodule.val_dataloader().dataset.is_img:
                self._visualise_prior_img(trainer, pl_module)
                self._visualise_posterior_img(trainer, pl_module)
            else:     
                self._visualise_prior_1d(pl_module)
                self._visualise_posterior_1d(trainer, pl_module)
           
def context_to_img(x_context, y_context, img_h, img_w):
    channels = y_context.shape[-1]
    x_context = x_context.squeeze()
    x_coords = (x_context[:, 0] * img_w).long()
    y_coords = (x_context[:, 1] * img_h).long()
    img = torch.zeros((3, img_h, img_w))
    if channels != 3:  # if mnist, change the background to blue
        img[2][:,:] = 255
    img[:, x_coords, y_coords] = y_context[0, :, :].T

    return img

def _visualise_with_gp_comparison(ax, x, y, l, s, p, kernel_type, pl_module, n_context, n=16):
    x_context, y_context, _, _ = process_data_to_points(x, y, n_context)

    # Create a set of target points corresponding to entire [-pi, pi] range
    x_target = torch.Tensor(np.linspace(torch.min(x).item(), torch.max(x).item(), 100))
    x_target = x_target.unsqueeze(1).unsqueeze(0)
    for i in range(n):
        # Neural process returns distribution over y_target
        p_y_pred, _, _ = pl_module.model(x_context.to(pl_module.device),
                                            y_context.to(pl_module.device),
                                            x_target.to(pl_module.device), None)
        # Extract mean of distribution
        mu = p_y_pred[0].loc.detach().cpu().numpy()[0]
        sigma = p_y_pred[0].scale.detach().cpu().numpy()

        ax.plot(x_target.cpu().numpy()[0], mu,
                    alpha=0.3, c='b')
                    
        ax.fill_between(x_target.cpu().numpy()[0].flatten(),
         mu.flatten() + 2* sigma.flatten(),
         mu.flatten() - 2* sigma.flatten(),
         color='b', alpha=0.5/n)

    if kernel_type == 'rbf':
        kernel = ConstantKernel(s**2, constant_value_bounds = "fixed") * RBF(length_scale=l, length_scale_bounds="fixed")
    elif kernel_type == 'matern':
        kernel = ConstantKernel(s**2, constant_value_bounds = "fixed") * Matern(length_scale=l, length_scale_bounds="fixed")
    else:
        kernel = ConstantKernel(s**2, constant_value_bounds = "fixed") * ExpSineSquared(length_scale=l, length_scale_bounds="fixed", periodicity=p, periodicity_bounds="fixed")
    
    gpr = GaussianProcessRegressor(kernel)
    gpr.fit(x_context[0].cpu().numpy(), y_context[0].cpu().numpy())
    gp_mean, gp_std = gpr.predict(x_target.cpu().numpy()[0], return_std=True)

    ax.plot(x_target.cpu().numpy()[0], gp_mean,
                    alpha=0.7, c='g')
    ax.plot(x_target.cpu().numpy()[0], gp_mean.flatten() + 2*gp_std.flatten(), c='g', linestyle='dashed')#, alpha=0.7,)
    ax.plot(x_target.cpu().numpy()[0], gp_mean.flatten() - 2*gp_std.flatten(), c='g', linestyle='dashed')#, alpha=0.7,)
        

    ax.scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c='k')