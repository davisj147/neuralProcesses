import torch
from torch import nn
from torch.distributions import Normal

class LinearEncoder(nn.Module):
    """
    Class to map sets X (batch x points x x_dim) and Y (batch x points x y_dim)
    to representations R (batch x points x r_dim)
    The dimension for points is being maintained for (hopefully) maintaining 
    inter-operability with other types of encoder structures
    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dims : List[int]
        Dimensions of hidden layers. Should be non-empty

    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, h_dims, r_dim):
        super(LinearEncoder, self).__init__()

        layers = [nn.Linear(x_dim + y_dim, h_dims[0]), nn.ReLU(inplace=True)]
        for i in range(1, len(h_dims)):
            layers.append(nn.Linear(h_dims[i-1], h_dims[i]))
            layers.append(nn.ReLU(inplace=True))

        # seems like the methods paper implies there's a relu here too, in fact everywhere
        layers.append(nn.Linear(h_dims[-1], r_dim))

        self.to_r = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, points, x_dim)

        y : torch.Tensor
            Shape (batch_size, points, y_dim)
        """

        batch_size, n_points, _ = x.size()

        net_input = torch.cat((
            x.view(batch_size * n_points, -1), 
            y.view(batch_size * n_points, -1)
            ), dim=1)

        rs = self.to_r(net_input)

        return rs.view(batch_size, n_points, -1)
        

class LinearDecoder(nn.Module):
    """
    Class to map sets X (batch x points x x_dim) and Z (batch x z_dim) to 
    predicted means and log variances
    The dimension for points is being maintained for (hopefully) maintaining 
    inter-operability with other types of encoder structures
    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of z values.

    h_dims : List[int]
        Dimensions of hidden layers. Should be non-empty

    y_dim : int
        Dimension of output y.
    """
    def __init__(self, x_dim, z_dim, h_dims, y_dim):
        super(LinearDecoder, self).__init__()

        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + z_dim, h_dims[0]), nn.ReLU(inplace=True)]
        for i in range(1, len(h_dims)):
            layers.append(nn.Linear(h_dims[i-1], h_dims[i]))
            layers.append(nn.ReLU(inplace=True))

        self.to_hidden = nn.Sequential(*layers)

        # seems like the best practices paper might be doing this in one, but likely doesn't matter
        self.to_mu = nn.Linear(h_dims[-1], y_dim)
        self.to_sigma = nn.Linear(h_dims[-1], y_dim)

    def forward(self, x, z):
        batch_size, n_points, _ = x.size()

        # repeat zs in order to be able to concatenate appropriate samples with inputs
        z = z.unsqueeze(1).repeat(1, n_points, 1)

        net_input = torch.cat((
            x.view(batch_size * n_points, -1), 
            z.view(batch_size * n_points, -1)
            ), dim=1)

        hidden_rep = self.to_hidden(net_input)
        means = self.to_mu(hidden_rep).view(batch_size, n_points, self.y_dim)

        sigmas = self.to_sigma(hidden_rep).view(batch_size, n_points, self.y_dim)
        # this as per Emp. evaluation of NP objectives
        # can't seem to find out what the NP paper did exactly
        sigmas = 0.1 + 0.9 * nn.functional.softplus(sigmas)

        return means, sigmas



class MeanCombiner(nn.Module):
    """
    Class to reduce the R representations by averaging them 
    accross context points for every batch entry
    """
    def __init__(self):
        super(MeanCombiner, self).__init__()

    def forward(self, rs):
        return torch.mean(rs, dim=1)


class LinearRToDist(nn.Module):
    def __init__(self, r_dim, h_dim, z_dim):
        super(LinearRToDist, self).__init__()

        self.to_hidden = nn.Sequential(nn.Linear(r_dim, h_dim), nn.ReLU(inplace=True))
        self.to_mu = nn.Linear(h_dim, z_dim)
        self.to_sigma = nn.Linear(h_dim, z_dim)

    def forward(self, r):
        hidden_rep = self.to_hidden(r)
        means = self.to_mu(hidden_rep)
        sigmas = self.to_sigma(hidden_rep)
        # this as per Emp. evaluation of NP objectives
        # can't seem to find out what the NP paper did exactly
        sigmas = 0.1 + 0.9 * torch.sigmoid(sigmas)

        return means, sigmas


class NeuralProcess(nn.Module):
    # should be a generic np class allowing for different encoders/decoders and combination methods
    def __init__(self, encoder, decoder, combiner, r_to_dist_encoder):
        super(NeuralProcess, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.combiner = combiner
        self.r_to_z_dist = r_to_dist_encoder

    def encode_to_z_params(self, x, y):
        # also ahould have any needed reshaping etc
        raw_rs = self.encoder(x, y)
        combined_r = self.combiner(raw_rs)
        z_mu, z_sigma = self.r_to_z_dist(combined_r)
        return z_mu, z_sigma

    def forward(self, x_context, y_context, x_target, y_target):
        # always encode the context
        z_context_mu, z_context_sigma = self.encode_to_z_params(x_context, y_context)
        dist_context = Normal(z_context_mu, z_context_sigma)
        
        if self.training:
            # in training for VI actually also encode the target (which the context is a subset of)
            # and use the z sampled from the more informed approximated posterior
            z_target_mu, z_target_sigma = self.encode_to_z_params(x_target, y_target)
            dist_target = Normal(z_target_mu, z_target_sigma)
            z_sample = dist_target.rsample()

        else:
            # in testing we do not care about the returned context distributions so can just use a dummy
            # however we do need to sample from the context-encoding-parametrised z
            dist_target=None
            z_sample = dist_context.rsample()

        y_mu, y_sigma = self.decoder(x_target, z_sample)

        dist_y = Normal(y_mu, y_sigma)

        return dist_y, dist_context, dist_target


# not a good implementtion but ok for testing
class SimpleNP(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim=16, z_dim=16, h_dims_enc = [64, 64], h_dims_dec=[64,64], h_dim = 128):
        super(SimpleNP, self).__init__()

        encoder = LinearEncoder(x_dim, y_dim, h_dims_enc, r_dim)
        decoder = LinearDecoder(x_dim, z_dim, h_dims_dec, y_dim)
        combiner = MeanCombiner()
        r_to_dist_encoder = LinearRToDist(r_dim, h_dim, z_dim)

        self.np = NeuralProcess(encoder, decoder, combiner, r_to_dist_encoder)

    def forward(self, xc, yc, xt, yt):
        return self.np(xc, yc, xt, yt)
