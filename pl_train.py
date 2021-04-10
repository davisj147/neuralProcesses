import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pl_modules import PLNeuralProcess
from pl_datasets import NPDataModule
from pl_callbacks import WandbLogPriorPosteriorSamplePlots
from typing import List
import torch

import argparse

parser = argparse.ArgumentParser(description='Train Neural Process models on a given dataset.')
parser.add_argument('--dataset_type', type=str, default='sine',
                    choices=['sine', 'gpdata', 'mnist', 'celeb'], help='Dataset name')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of CPU cores to load data on')
parser.add_argument('--batch_size', type=int, default=250,
                    help='Data batch size')
parser.add_argument('--num_context', type=int, default=40,
                    help='Number of Context datapoints')
parser.add_argument('--fix_n_context_and_target_points', action='store_true',
                    help='Whether to always select num_context and num_target points during training. Otherwise, select randomly from (3, num_context) and (1, num_target)')
parser.add_argument('--num_target', type=int, default=60,
                    help='Number of Target datapoints (Context + Target by convention)')
parser.add_argument('--r_dim', type=int, default=50,
                    help='Dimension of encoder representation of context points')
parser.add_argument('--z_dim', type=int, default=50,
                    help='Dimension of sampled latent variable')
parser.add_argument('--h_dim', type=int, default=50,
                    help='Dimension of hidden layer in encoding to gaussian mean/variance network')
parser.add_argument('--h_dim_enc', type=int, nargs='*', default=[50, 50],
                    help='Dimension(s) of hidden layer(s) in encoder')
parser.add_argument('--h_dim_dec', type=int, nargs='*', default=[50, 50, 50],
                    help='Dimension(s) of hidden layer(s) in decoder')
parser.add_argument('--max_epochs', type=int, default=1000,
                    help='Maximum number of training epochs')
parser.add_argument('--cpu', action='store_true',
                    help='Whether to train on the CPU. If ommited, will train on a GPU')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='Initial learning rate')
parser.add_argument('--tune-lr', action='store_true',
                    help='Whether to automatically tune the learning rate prior to training')
parser.add_argument('--gp-kernel', type=str, default='rbf',
                    choices=['rbf', 'matern', 'periodic'],
                    help='kernel to use for the gaussian process dataset generation')
parser.add_argument('--n-samples', type=int, default=1000,
                    help='number of samples for sine or gp dataset generation')
parser.add_argument('--n-points', type=int, default=100,
                    help='number of points per sample samples for sine or gp dataset generation')
parser.add_argument('--n-repeat', type=int, default=1,
                    help='Number of samples per batch during training')
parser.add_argument('--training_type', type=str, default='VI', choices=['VI', 'MLE'],
                    help='Training type- variational inference and MLE')
parser.add_argument('--lengthscale-range', type=float, nargs='*', default=[0.25, 0.3],
                    help='Range for lengthscales when generating GP data')
parser.add_argument('--sigma-range', type=float, nargs='*', default=[0.7, 1.],
                    help='Range for sigma (output variance) when generating GP data')
parser.add_argument('--period-range', type=float, nargs='*', default=[1., 1.],
                    help='Range for periods when generating GP data')
parser.add_argument('--check_val_every_n_epoch', type=int, default=100,
                    help='How often to run validation loop and run logging plots')

args = parser.parse_args()

if __name__ == '__main__':
    # ------------------------
    # 1 SETUP DATA MODULES
    # ------------------------
    dm = NPDataModule(dataset_type=args.dataset_type,
                      num_workers=0, ## Running really slow when > 0
                      # path_to_data='/home/jack/Dropbox/MLMI/AdvancedML/AML_neural_processes/data',
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
    model = PLNeuralProcess(x_dim=dm.x_dim,
                            y_dim=dm.y_dim,
                            lr=args.lr,
                            num_context=args.num_context,
                            num_target=args.num_target,
                            fix_n_context_and_target_points=args.fix_n_context_and_target_points,
                            r_dim=args.r_dim,
                            z_dim=args.z_dim,
                            h_dim=args.h_dim,
                            h_dim_enc=args.h_dim_enc,
                            h_dim_dec=args.h_dim_dec,
                            n_repeat=args.n_repeat,
                            training_type=args.training_type)

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    cbs = [#EarlyStopping(monitor='validation_loss', verbose=True, patience=5, mode='min'),
           WandbLogPriorPosteriorSamplePlots()]
    trainer = pl.Trainer(gpus=0 if args.cpu or not torch.cuda.is_available() else 1,
                         max_epochs=args.max_epochs,
                         checkpoint_callback=True, num_sanity_val_steps=0,
                         # ModelCheckpoint(
                         #     monitor='training_loss',
                         #     filename='neural_process-{epoch:02d}-{validation_loss:.2f}'),
                         logger=WandbLogger(project=f'AdvancedML-{args.dataset_type}',
                                            log_model=True),
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         auto_lr_find=args.tune_lr,
                         callbacks=cbs,
                         profiler='simple')

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    if args.tune_lr:
        trainer.tune(model, datamodule=dm)

    trainer.fit(model, datamodule=dm)
    print(f'Best model saved! Run PLNeuralProcess.load_from_checkpoint("{trainer.checkpoint_callback.best_model_path}") to re-load best model')

