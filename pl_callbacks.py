from pytorch_lightning.callbacks.base import Callback
import wandb

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pl_modules import process_data_to_points, batch_img_to_functional

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
        x, y = next(iter(trainer.datamodule.val_dataloader()))
        sample_id = random.randint(1, x.shape[0])
        pl_module.eval()

        x, y = x[(sample_id-1):sample_id], y[(sample_id-1):sample_id]

        fig, axs = plt.subplots(2,2, figsize=(16, 10))
        flat_axs = axs.flatten()
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
                mu = p_y_pred.loc.detach()
                flat_axs[j].plot(x_target.cpu().numpy()[0], mu.cpu().numpy()[0],
                            alpha=0.1, c='b')
                flat_axs[j].plot(x.cpu().numpy()[0], y.cpu().numpy()[0],
                            alpha=0.7, c='r')

            flat_axs[j].scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c='k')

        wandb.log({"posterior_samples": fig})

    def _visualise_posterior_img(self, trainer, pl_module):
        x, y = next(iter(trainer.datamodule.val_dataloader()))
        _, channels, img_h, img_w = x.shape
        img_id = random.randint(1, x.shape[0])
        imgs = []
        for n_context in [8, 32, 64, 256, 28*28]:
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
                mu = p_y_pred.loc.detach()
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
    img = torch.zeros((channels, img_h, img_w))
    img[:, x_coords, y_coords] = y_context[0, :, :].T

    return img
