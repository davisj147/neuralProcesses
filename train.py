import torch
import random
from torch import nn
from torch.distributions.kl import kl_divergence
from tqdm import tqdm

def train(
    device,
    neural_process,
    optimizer,
    print_freq,
    epochs,
    data_loader
):
    update_count = 0
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(data_loader, unit="batch") as tepoch:
            neural_process.train()
            for X_train, y_train in tepoch:
                
                # <stuff here>
                
                loss = np_loss()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                update_count += 1

                tepoch.set_postfix(loss=loss.item())

        losses.append(epoch_loss/len(data_loader))
        print(f'Epoch {epoch}: average loss per batch {losses[epoch]}\n')



    pass

def np_loss(dist_y, y_target, dist_context, dist_target):
    # assumes the first dimension (0) corresponds to batch element

    # total log probability of ys averaged over the batch
    ll  = dist_y.log_prob(y_target).mean(dim=0).sum()
    kl = kl_divergence(dist_target, dist_context).mean(dim=0).sum()
    
    return -1*ll + kl
