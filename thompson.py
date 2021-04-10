"""
Mostly unorganised file with functions that can be used to run Thompson sampling experiments
"""
from pl_modules import *
from pl_datasets import *
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel

def random_search(x, y, reps_allowed=True):
    min_id = np.argmin(y)

    if reps_allowed:
        rand_id = -1
        n = 0
        while not rand_id == min_id:
            n += 1
            rand_id = random.randint(0, x.shape[0] -1)
        return n

    else:
        idxs = list(range(x.shape[0]))
        random.shuffle(idxs)
        return idxs.index(min_id) + 1   


def np_thompson(np_x, np_y, x, y, model, reps_allowed=True, return_all_data=False):
    
    min_id = np.argmin(y)
    x_at_min = x[min_id]
    samples = []
    
    sample_min = random.randint(0, x.shape[0]-1)
    next_x = x[sample_min]
    point_ids = [sample_min]
    while not (sample_min == min_id):
        # print(next_x)
        x_context = np_x[:, point_ids, :]
        y_context = np_y[:, point_ids, :]

        p_y_pred, _, _ = model.model(x_context.to(model.device),
                                y_context.to(model.device),
                                np_x.to(model.device), None)

        sample = p_y_pred.loc.detach().cpu().numpy().flatten()
        sample_min = np.argmin(sample)

        if not reps_allowed:
            c=0
            while sample_min in point_ids:
                c += 1
                p_y_pred, _, _ = model.model(x_context.to(model.device),
                                    y_context.to(model.device),
                                    np_x.to(model.device), None)
                sample =  p_y_pred.loc.detach().cpu().numpy().flatten()
                sample_min = np.argmin(sample)
                if c > 100:
                    return  -1
        
        next_x = x[sample_min]
        if return_all_data:
            samples.append(sample)

        point_ids.append(sample_min)

        if len(point_ids) > 700:
            return  -1

    if return_all_data:
        return len(point_ids), samples, point_ids


def gp_thompson(x, y, kernel_type, s, l, p, reps_allowed=True):
    
    if kernel_type == 'rbf':
        kernel = ConstantKernel(s**2, constant_value_bounds = "fixed") * RBF(length_scale=l, length_scale_bounds="fixed")
    elif kernel_type == 'matern':
        kernel = ConstantKernel(s**2, constant_value_bounds = "fixed") * Matern(length_scale=l, length_scale_bounds="fixed")
    else:
        kernel = ConstantKernel(s**2, constant_value_bounds = "fixed") * ExpSineSquared(length_scale=l, length_scale_bounds="fixed", periodicity=p,                                         periodicity_bounds="fixed")
    
    min_id = np.argmin(y)
    x_at_min = x[min_id]
    
    sample_min = random.randint(0, x.shape[0]-1)
    next_x = x[sample_min]
    gpr = GaussianProcessRegressor(kernel)
    point_ids = [sample_min]

    while not (sample_min == min_id):
        xs = x[point_ids].reshape((-1, 1))
        ys = y[point_ids].reshape((-1, 1))

        gpr.fit(xs, ys)
        sample = gpr.sample_y(x.reshape((-1, 1)))
        sample_min = np.argmin(sample)

        if not reps_allowed:
            c=0
            while sample_min in point_ids:
                c += 1
                sample = gpr.sample_y(x.reshape((-1, 1)))
                sample_min = np.argmin(sample)
                if c > 100:
                    return  -1

        next_x = x[sample_min]

        point_ids.append(sample_min)

        if len(point_ids) > 700:
            return  -1

    return xs.shape[0]


def plot_thompson_process(samples, point_ids, x, y):
    fig, axs = plt.subplots(1, len(samples), figsize = (4*(len(samples)), 3))

    for i in range(len(samples)):
        axs[i].plot(x, y, color='#264653', linestyle='dashed')
        axs[i].plot(x, samples[i], '#2a9d8f')
        if i > 0:
            axs[i].scatter(x[point_ids[:i]], y[point_ids[:i]], c='#264653')
            axs[i].scatter([x[point_ids[i]]], [y[point_ids[i]]], c='#e76f51')
        else:
            axs[i].scatter([x[point_ids[i]]], [y[point_ids[i]]], c='#264653')
        
        axs[i].set_ylim((-2, 2))
        
        axs[i].scatter([x[point_ids[i+1]]], [samples[i][point_ids[i+1]]- 0.05] , c='#e76f51', marker='^')
    fig.tight_layout()


def run_loop(dm, model, allow_reps, rng, n):
    model.eval()
    np_vals = []
    gp_vals = []
    rand_vals = []
    for i in range(n):

        print('iteration ', i) 

        x, y, l, s, p = dm.val_dataloader().dataset.generate_gp_sample(rng)
        np_x, np_y = torch.tensor(x).float().to(model.device).unsqueeze(0), torch.tensor(y).float().to(model.device).unsqueeze(0).unsqueeze(2)
        kernel_type = dm.val_dataloader().dataset.kernel
        
        np_val = -1
        restarts = -1
        while np_val < 0:
            restarts += 1
            np_val = np_thompson(np_x, np_y, x.flatten(), y, model, allow_reps)
        print('NP: ', np_val, ' restarts ', restarts)

        gp_val = -1
        restarts = -1
        while gp_val < 0: 
            restarts += 1
            gp_val = gp_thompson(x.flatten(), y.flatten(), 'rbf', s, l, p, allow_reps)
        print('GP: ', gp_val, ' restarts ', restarts)

        rand_val = random_search(x.flatten(), y, allow_reps)
        print(rand_val)

        np_vals.append(np_val)
        gp_vals.append(gp_val)
        rand_vals.append(rand_val)

    return np.array(np_vals), np.array(gp_vals), np.array(rand_vals)


dm = NPDataModule(dataset_type='gpdata',
                    num_workers=4,
                    batch_size=16,
                    kernel='rbf',
                    num_samples=100,
                    num_points=300,
                    lengthscale_range=(0.25, 0.5),
                    sigma_range=(1,1),
                    )
rng = np.random.default_rng()

model_checkpoint = './1200_gp_rbf.ckpt'

model = PLNeuralProcess.load_from_checkpoint(model_checkpoint)

npv, gpv, ranv = run_loop(dm, model, allow_reps, rng, n)

np.savez(f'thomp_{n}_{allow_reps}.npz', np_vals=npv, gp_vals = gpv, rand_vals = ranv)

print('Random mean steps ', np.mean(ranv))
print('GP mean steps ', np.mean(gpv))
print('NP mean steps ', np.mean(npv) )

print('Random mean steps (normalised) ', np.mean(ranv)/np.mean(ranv))
print('GP mean steps (normalised) ', np.mean(gpv)/np.mean(ranv))
print('NP mean steps (normalised) ', np.mean(npv)/np.mean(ranv))