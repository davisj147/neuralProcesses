import torch
import random
from random import randint
from torch import nn
import numpy as np
from torch.distributions.kl import kl_divergence
from tqdm import tqdm

def train(
    device,
    neural_process,
    optimizer,
    print_freq,
    epochs,
    data_loader,
    n_context_range,
    n_target_range=None
):
    update_count = 0
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(data_loader, unit="batch") as tepoch:
            neural_process.train()
            for X, y in tepoch:

                optimizer.zero_grad()

                n_context = randint(*n_context_range)
                n_target = randint(*n_target_range) if n_target_range else None

                x_context, y_context, x_target, y_target = process_data_to_points(X, y, n_context, n_target)
                
                x_context, y_context = x_context.to(device), y_context.to(device)
                x_target, y_target = x_target.to(device), y_target.to(device)

                dist_y, dist_context, dist_target = neural_process(x_context.float(), y_context.float(), x_target.float(), y_target.float())
                
                loss = np_loss(dist_y, y_target, dist_context, dist_target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                update_count += 1

                tepoch.set_postfix(loss=loss.item())

        losses.append(epoch_loss/len(data_loader))
        print(f'\nEpoch {epoch}: average loss per batch {losses[epoch]}\n')

    pass

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

    x_context = xs[:, permutation[:n_context], :]
    y_context = ys[:, permutation[:n_context], :]
    x_target  = xs[:, permutation[:n_total],   :]
    y_target  = ys[:, permutation[:n_total],   :]
    
    return x_context, y_context, x_target, y_target

def batch_img_to_functional(batch_imgs):
    n_batch, channels, img_w, img_h = batch_imgs.size()
    n_points = img_w * img_h

    # ugly way to make an array of indices
    locations = torch.ones((img_w, img_h)).nonzero(as_tuple=False).float()

    # normalise to [0, 1]
    locations[:, 0] = locations[:, 0] / float(img_w) # might have accidentally switched h and w
    locations[:, 1] = locations[:, 1] / float(img_h) 

    xs = locations.repeat(n_batch, 1, 1)
    ys = batch_imgs.view((n_batch, n_points, channels))

    return xs, ys

def np_loss(dist_y, y_target, dist_context, dist_target):
    # assumes the first dimension (0) corresponds to batch element

    # total log probability of ys averaged over the batch
    ll  = dist_y[0].log_prob(y_target).mean(dim=0).sum()
    kl = kl_divergence(dist_target, dist_context).mean(dim=0).sum()
    
    return -1*ll + kl
