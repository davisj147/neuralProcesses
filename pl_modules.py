import pytorch_lightning as pl
from torch.distributions import kl_divergence

from np import NeuralProcess
import torch
from np import SimpleNP
from random import randint
import numpy as np


class PLNeuralProcess(pl.LightningModule):
    def __init__(self, x_dim, y_dim, lr=1e-3, num_context=8, num_target=16, r_dim=50, z_dim=50,
                 h_dim=50, h_dim_enc=[50, 50], h_dim_dec=[50, 50, 50]):
        super(PLNeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_context = num_context
        self.num_target = num_target
        self.n_context_range = (3, num_context)
        self.n_target_range = (num_context, num_context+num_target)
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.h_dim_enc = h_dim_enc
        self.h_dim_dec = h_dim_dec

        self.lr = lr
        self.save_hyperparameters()
        self.model = self._build_model()

    # def forward(self, x):
    #     x = self.model(x)
    #     return x

    # def predict(self, images, threshold=None):
    #     self.eval()
    #     with torch.no_grad():
    #         pass

    def _build_model(self):
        neuralprocess = SimpleNP(x_dim=self.x_dim,
                                 y_dim=self.y_dim,
                                 r_dim=self.r_dim,
                                 z_dim=self.z_dim,
                                 h_dim=self.h_dim,
                                 h_dims_dec=self.h_dim_dec,
                                 h_dims_enc=self.h_dim_enc)
        return neuralprocess

    def training_step(self, batch, batch_idx):
        X, y = batch

        n_context = randint(*self.n_context_range)
        n_target = randint(*self.n_target_range)

        x_context, y_context, x_target, y_target = process_data_to_points(X, y, n_context,
                                                                          n_target)
        dist_y, dist_context, dist_target = self.model(x_context, y_context,
                                                       x_target, y_target)

        loss = self._loss(dist_y, y_target, dist_context, dist_target)

        return {'loss': loss}

    def training_step_end(self, outputs):
        self.log('training_loss', outputs['loss'], on_epoch=True, on_step=False,
                 prog_bar=True)
        self.log('learning_rate', self.lr, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': outputs['loss']}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def validation_step_end(self, outputs):
        self.log('validation_loss', outputs['loss'], on_epoch=True, on_step=False,
                 prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=self.lr,
                                     weight_decay=0)

        return [optimizer]

    @staticmethod
    def _loss(dist_y, y_target, dist_context, dist_target):
        # assumes the first dimension (0) corresponds to batch element
        # total log probability of ys averaged over the batch
        ll = dist_y.log_prob(y_target).mean(dim=0).sum()
        kl = kl_divergence(dist_target, dist_context).mean(dim=0).sum()

        return -1 * ll + kl


def process_data_to_points(X_train, y_train, n_context, n_total=None):
    is_img = len(X_train.size()) > 3

    if is_img:
        # for now all images in batch will have the same points selected because I can't figure out gather
        xs, ys = batch_img_to_functional(X_train)

    else:
        xs, ys = X_train, y_train

    _, n_points, _ = xs.size()
    n_total = n_total if n_total else n_points

    rng = np.random.default_rng()

    permutation = rng.permutation(n_points)

    x_context = xs[:, permutation[:n_context], :].float()
    y_context = ys[:, permutation[:n_context], :].float()
    x_target = xs[:, permutation[:n_total], :].float()
    y_target = ys[:, permutation[:n_total], :].float()

    return x_context, y_context, x_target, y_target


def batch_img_to_functional(batch_imgs):
    n_batch, channels, img_w, img_h = batch_imgs.size()
    n_points = img_w * img_h

    # ugly way to make an array of indices
    locations = torch.ones((img_w, img_h)).nonzero(as_tuple=False).float()

    # normalise to [0, 1]
    locations[:, 0] = locations[:, 0] / float(img_w)  # might have accidentally switched h and w
    locations[:, 1] = locations[:, 1] / float(img_h)

    xs = locations.repeat(n_batch, 1, 1)
    ys = batch_imgs.view((n_batch, n_points, channels))

    return xs, ys
