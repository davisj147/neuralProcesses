from pytorch_lightning.callbacks.base import Callback
import wandb

import torch
import numpy as np
import matplotlib.pyplot as plt
from pl_modules import process_data_to_points


class WandbLogPriorPosteriorSamplePlots(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Visualize samples from trained prior
        # The prior should now encode some information about the shapes of the functions.

        if not trainer.running_sanity_check:
            torch.manual_seed(0)
            x_target = torch.Tensor(np.linspace(-1, 1, 100))
            x_target = x_target.unsqueeze(1).unsqueeze(0).to(pl_module.device)

            for i in range(64):
                z_sample = torch.randn((1, pl_module.z_dim)).to(pl_module.device)
                mu, _ = pl_module.model.decoder(x_target, z_sample)
                plt.plot(x_target.cpu().numpy()[0], mu.detach().cpu().numpy()[0],
                         c='b', alpha=0.5)
                plt.xlim(-1, 1)

            wandb.log({"prior_samples": plt})

            # Visualize samples from posterior
            # Extract a batch from data_loader
            # Use batch to create random set of context points
            x, y = next(iter(trainer.datamodule.val_dataloader()))
            x_context, y_context, _, _ = process_data_to_points(x[0:1], y[0:1],
                                                                pl_module.num_context,
                                                                pl_module.num_target + pl_module.num_context)

            # Create a set of target points corresponding to entire [-pi, pi] range
            x_target = torch.Tensor(np.linspace(-1, 1, 100))
            # x_target = torch.Tensor(np.linspace(-pi, pi, 100))
            x_target = x_target.unsqueeze(1).unsqueeze(0)

            pl_module.eval()

            for i in range(64):
                # Neural process returns distribution over y_target
                p_y_pred, _, _ = pl_module.model(x_context.to(pl_module.device),
                                                 y_context.to(pl_module.device),
                                                 x_target.to(pl_module.device), None)
                # Extract mean of distribution
                mu = p_y_pred.loc.detach()
                plt.plot(x_target.cpu().numpy()[0], mu.cpu().numpy()[0],
                         alpha=0.05, c='b')

            plt.scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c='k')

            wandb.log({"posterior_samples": plt})
