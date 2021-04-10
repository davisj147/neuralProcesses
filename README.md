# Paper Replication Project: Neural Processes

This Repository contains code replicating the implementation of [Neural Processes](https://arxiv.org/pdf/1807.01622.pdf), as described in the original paper. The work was produced as part of course activities for the MPhil in Machine Learning and Machine Intelligence at Cambridge University.

### Usage

A usage example is provided in the notebook `example-1d-lightning.ipynb`

The main model is customisable from the command line and compatible with synthetically genenrated 1D dataset (sine waves and GP samples from GPs using specified kernels) and image datasets (MNIST, with data expected to be stored at `../data/MNIST` and CelebA, with data assumend to be stored at `../data/celebA/img_align_celeba`)

*** 

The `pl_train.py` script (main training entrypoint) can be configured using command line arguments:

    optional arguments:
    -h, --help            show this help message and exit
    --dataset_type {sine,gpdata,mnist,celeb}
                            Dataset name
    --num_workers NUM_WORKERS
                            Number of CPU cores to load data on
    --batch_size BATCH_SIZE
                            Data batch size
    --num_context NUM_CONTEXT
                            Number of Context datapoints
    --fix_n_context_and_target_points
                            Whether to always select num_context and num_target points during training. Otherwise, select randomly from (3, num_context) and (1, num_target)
    --num_target NUM_TARGET
                            Number of Target datapoints (Context + Target by convention)
    --r_dim R_DIM         Dimension of encoder representation of context points
    --z_dim Z_DIM         Dimension of sampled latent variable
    --h_dim H_DIM         Dimension of hidden layer in encoding to gaussian mean/variance network
    --h_dim_enc [H_DIM_ENC [H_DIM_ENC ...]]
                            Dimension(s) of hidden layer(s) in encoder
    --h_dim_dec [H_DIM_DEC [H_DIM_DEC ...]]
                            Dimension(s) of hidden layer(s) in decoder
    --max_epochs MAX_EPOCHS
                            Maximum number of training epochs
    --cpu                 Whether to train on the CPU. If ommited, will train on a GPU
    --lr LR               Initial learning rate
    --tune-lr             Whether to automatically tune the learning rate prior to training
    --gp-kernel {rbf,matern,periodic}
                            kernel to use for the gaussian process dataset generation
    --n-samples N_SAMPLES
                            number of samples for sine or gp dataset generation
    --n-points N_POINTS   number of points per sample samples for sine or gp dataset generation
    --n-repeat N_REPEAT   Number of samples per batch during training
    --training_type {VI,MLE}
                            Training type- variational inference and MLE
    --lengthscale-range [LENGTHSCALE_RANGE [LENGTHSCALE_RANGE ...]]
                            Range for lengthscales when generating GP data
    --sigma-range [SIGMA_RANGE [SIGMA_RANGE ...]]
                            Range for sigma (output variance) when generating GP data
    --period-range [PERIOD_RANGE [PERIOD_RANGE ...]]
                            Range for periods when generating GP data
    --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                            How often to run validation loop and run logging plots
