import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pl_modules import PLNeuralProcess
from pl_datasets import NPDataModule
from pl_callbacks import WandbLogPriorPosteriorSamplePlots
from typing import List
import torch
from datasets import GPData, SineData
import matplotlib.pyplot as plt
from torchvision import datasets
from pl_callbacks import context_to_img
from torchvision.utils import make_grid
import argparse

parser = argparse.ArgumentParser(description='Train video action recognition models.')
parser.add_argument('--dataset_type', type=str, default='celeb',
                    choices=['sine', 'gpdata', 'mnist', 'celeb'], help='Dataset name')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of CPU cores to load data on')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Data batch size')
parser.add_argument('--num_context', type=int, default=256,
                    help='Number of Context datapoints')
parser.add_argument('--num_target', type=int, default=512,
                    help='Number of Target datapoints (Context + Target by convention)')
parser.add_argument('--r_dim', type=int, default=256,
                    help='Dimension of encoder representation of context points')
parser.add_argument('--z_dim', type=int, default=256,
                    help='Dimension of sampled latent variable')
parser.add_argument('--h_dim', type=int, default=256,
                    help='Dimension of hidden layer in encoding to gaussian mean/variance network')
parser.add_argument('--h_dim_enc', type=int, nargs='*', default=[512, 512],
                    help='Dimension(s) of hidden layer(s) in encoder')
parser.add_argument('--h_dim_dec', type=int, nargs='*', default=[512, 512, 512],
                    help='Dimension(s) of hidden layer(s) in decoder')
parser.add_argument('--max_epochs', type=int, default=5000,
                    help='Maximum number of training epochs')
parser.add_argument('--cpu', action='store_true',
                    help='Whether to train on the CPU. If ommited, will train on a GPU')
parser.add_argument('--lr', type=float, default=4e-5, #1e-3,
                    help='Initial learning rate')
parser.add_argument('--tune-lr', action='store_true',
                    help='Whether to automatically tune the learning rate prior to training')
parser.add_argument('--gp-kernel', type=str, default='rbf',
                    choices=['rbf', 'matern', 'periodic'],
                    help='kernel to use for the gaussian process dataset generation')
parser.add_argument('--n-samples', type=int, default=2000,
                    help='number of samples for sine or gp dataset generation')
parser.add_argument('--n-points', type=int, default=100,
                    help='number of points per sample samples for sine or gp dataset generation')
parser.add_argument('--lengthscale-range', type=float, nargs='*', default=[0.5, 0.6],
                    help='Range for lengthscales when generating GP data')
parser.add_argument('--sigma-range', type=float, nargs='*', default=[.1, .1],
                    help='Range for sigma (output variance) when generating GP data')
parser.add_argument('--period-range', type=float, nargs='*', default=[1., 1.],
                    help='Range for periods when generating GP data')

args = parser.parse_args()

from np import NeuralProcess
import torch
from np import SimpleNP
from random import randint
import numpy as np

def process_data_to_points_half(X_train, y_train, n_context, n_total=None):
    is_img = len(X_train.size()) > 3

    if is_img:
        # for now all images in batch will have the same points selected because I can't figure out gather
        xs, ys = batch_img_to_functional(X_train)

    else:
        xs, ys = X_train, y_train

    _, n_points, _ = xs.size()
    n_total = n_total if n_total else n_points

    rng = np.random.default_rng()

    permutation = rng.permutation(n_points)  # random
    p = np.arange(n_points)                  # aligned

    
    #x_context = xs[:, p[:n_context], :].float()   # top half
    #y_context = ys[:, p[:n_context], :].float()   # top half
    x_context = xs[:, p[-n_context:], :].float()  # bottom half
    y_context = ys[:, p[-n_context:], :].float()  # bottom half

    x_target = xs[:, permutation[:n_total], :].float()
    y_target = ys[:, permutation[:n_total], :].float()

    return x_context, y_context, x_target, y_target

if __name__ == '__main__':
    # ------------------------
    # 1 SETUP DATA MODULES
    # ------------------------
    dm = NPDataModule(dataset_type=args.dataset_type,
                      num_workers=args.num_workers,
                      batch_size=args.batch_size,
                      kernel=args.gp_kernel,
                      num_samples=args.n_samples,
                      num_points=args.n_points,
                      lengthscale_range=args.lengthscale_range,
                      sigma_range=args.sigma_range,
                      period_range=args.period_range)

    # ------------------------
    # 2 INIT LIGHTNING MODEL
    # ------------------------
    """
    model = PLNeuralProcess(x_dim=dm.x_dim,
                            y_dim=dm.y_dim,
                            lr=args.lr,
                            num_context=args.num_context,
                            num_target=args.num_target,
                            r_dim=args.r_dim,
                            z_dim=args.z_dim,
                            h_dim=args.h_dim,
                            h_dim_enc=args.h_dim_enc,
                            h_dim_dec=args.h_dim_dec)
    """
    pl_module = PLNeuralProcess.load_from_checkpoint('/homes/yk384/MLMI4/exp/wandb/run-20210328_101110-pm44hdz7/files/mlmi4-np-celeb/pm44hdz7/checkpoints/epoch=122-step=387192.ckpt')
    
    from pl_modules import batch_img_to_functional
    import random
    x, y = next(iter(dm.train_dataloader()))
    _, channels, img_h, img_w = x.shape
    img_id = random.randint(1, x.shape[0])
    imgs = []
    if channels == 3:  # celeb
        list_n_context = [int(32*32*0.20), int(32*32/2), 32*32]

    for n_context in list_n_context:
        x_context, y_context, _, _ = process_data_to_points_half(x[(img_id - 1):img_id],
                                                            y[(img_id - 1):img_id],
                                                            n_context)
        # create target points for the full image
        xs, _ = batch_img_to_functional(x)
        x_target = xs[0, :, :].unsqueeze(0)
        pl_module.eval()
        # imgs = [context_to_img(x_context, y_context, img_h, img_w)]
        img = context_to_img(x_context, y_context, img_h, img_w)
        if channels != 3:
            img = img.expand((3, img_h, img_w))
        imgs.append(img)
        for i in range(3):
            # Neural process returns distribution over y_target
            pre_p_y, _, _ = pl_module.model(x_context.to(pl_module.device),
                                            y_context.to(pl_module.device),
                                            x_target.to(pl_module.device), None)
            # Extract mean of distribution
            mu = pre_p_y.loc.detach()
            img_mu = mu.permute(0, 2, 1).reshape((channels, img_h, img_w)).detach().cpu()
            if channels != 3:
                img_mu = img_mu.expand((3, img_h, img_w))
            imgs.append(img_mu)
    grid = make_grid(imgs, nrow=len(list_n_context) , pad_value=1.)
    fig = plt.figure()
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.savefig("image.jpg")
    #plt.show()
    
    """
    #model = wandb.restore('model-best.h5', run_path="vanpelt/my-project/a1b2c3d")
    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    cbs = [#EarlyStopping(monitor='validation_loss', verbose=True, patience=5, mode='min'),
           WandbLogPriorPosteriorSamplePlots()]

    trainer = pl.Trainer(gpus=0 if args.cpu or not torch.cuda.is_available() else 1,
                         max_epochs=args.max_epochs,
                         checkpoint_callback=ModelCheckpoint(
                             monitor='training_loss',
                             filename='checkpoint_neural_process-{epoch:02d}-{training_loss:.2f}'),
                         logger=WandbLogger(project=f'AdvancedML-{args.dataset_type}',
                                            log_model=True),
                         auto_lr_find=args.tune_lr,
                         callbacks=cbs)

    trainer = pl.Trainer(gpus=0 if args.cpu or not torch.cuda.is_available() else 1,
                         max_epochs=args.max_epochs,
                         resume_from_checkpoint='../exp/wandb/run-20210322_211421-1p3rtu4q/files/mlmi4-np-celeb/1p3rtu4q/checkpoints/checkpoint_neural_process-epoch=126-training_loss=-1759.02.ckpt',
                         logger=WandbLogger(project=f'AdvancedML-{args.dataset_type}',
                                            log_model=True),
                         auto_lr_find=args.tune_lr,
                         callbacks=cbs)

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    if args.tune_lr:
        trainer.tune(model, datamodule=dm)

    trainer.fit(model, datamodule=dm)
    print(f'Best model saved! Run PLNeuralProcess.load_from_checkpoint("{trainer.checkpoint_callback.best_model_path}") to re-load best model')

    """
