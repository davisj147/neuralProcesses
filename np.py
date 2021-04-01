import torch
from torch import nn
from torch.distributions import Normal

class LinearEncoder(nn.Module):
    """
    Class to map sets X (batch x points x x_dim) and Y (batch x points x y_dim)
    to representations R (batch x points x r_dim) using a MLP
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
        # the deepmind repo, however, does not use a ReLU for the last layer 
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
            x.reshape(batch_size * n_points, -1),
            y.reshape(batch_size * n_points, -1)
            ), dim=1)

        rs = self.to_r(net_input)

        return rs.reshape(batch_size, n_points, -1)
        

class LinearDecoder(nn.Module):
    """
    Class to map sets X (batch x points x x_dim) and Z (batch x z_dim) to 
    predicted means and log variances using a MLP
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

        # the original tf notebook is doing this by running a bunch of sequential
        # linear layers, with the last one not using activation, and then 
        # splitting the final layer's output in half
        # this seems to be functionally equivalent to using a sequence of linear layer + activation
        # followed by two separate no-activation linear layers for mu and sigma
        # I find this implementation easier to interpret so keeping it this way
        self.to_mu = nn.Linear(h_dims[-1], y_dim)
        self.to_sigma = nn.Linear(h_dims[-1], y_dim)

    def forward(self, x, z):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)
        """

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
        # sigmas = 0.1 + 0.9 * nn.functional.softplus(sigmas)
        sigmas = 0.001 + 0.999 * nn.functional.softplus(sigmas)

        return means, sigmas



class MeanCombiner(nn.Module):
    """
    Class to reduce the R representations by averaging them 
    accross context points for every batch entry
    # def forward(self, xc, yc, xt, yt):
    #     return self.np(xc, yc, xt, yt)
    """
    def __init__(self):
        super(MeanCombiner, self).__init__()

    def forward(self, rs):
        """
        Parameters
        ----------
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        return torch.mean(rs, dim=1)


class LinearRToDist(nn.Module):
    """
    Class to map aggregated summary results R to the parameters mu_z and sigma_z mu_z and sigma_z
    for the latent variable distribution
    Parameters
    ----------
    r_dim : int
        Dimension of r values.

    h_dim : int
    Dimension of the hidden layer.

    z_dim : int
        Dimension of z (latent) values.
    """
    def __init__(self, r_dim, h_dim, z_dim):
        super(LinearRToDist, self).__init__()

        self.to_hidden = nn.Sequential(nn.Linear(r_dim, h_dim), nn.ReLU(inplace=True))
        # as per the original tf notebook, these two do not use activation dunctions
        self.to_mu = nn.Linear(h_dim, z_dim)
        self.to_sigma = nn.Linear(h_dim, z_dim)

    def forward(self, r):
        """
        Parameters
        ----------
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden_rep = self.to_hidden(r)
        means = self.to_mu(hidden_rep)

        sigmas = self.to_sigma(hidden_rep)

        # this as per Emp. evaluation of NP objectives and the tf notebook
        # can't seem to find out what the NP paper did exactly
        # sigmas = 0.1 + 0.9 * torch.sigmoid(sigmas)
        sigmas = 0.001 + 0.999 * nn.functional.softplus(sigmas)

        return means, sigmas


class NeuralProcess(nn.Module):
    """
    Class representing a neural process, utilising the latent path exclusively. 
    Capable of processing context and target sets of the form (n_batch x n_points x n_features), 
    with n_features standing in for the input/output point dimensionality.
    
    Parameters
    ----------
    encoder : nn.Module
    module mapping x, y (shape n_batch x n_points x n_features) to summary representation
    r (shape n_batch x n_points x r_dim)

    decoder : nn.Module
    module mapping x (shape n_batch x n_points x n_features), z (shape n_batch x z_dim) to 
    output distribution parameters y_mean and y_sigma (shapes n_batch x n_points x y_dim)

    combiner : nn.Module
    module performing a permutation invariant combination of r (shape n_batch x points x r_dim)
    for each batch entry to produce summaries (shape n_batch x r_dim)

    r_to_dist_encoder : nn.Module
    module mapping summaries r (shape n_batch x r_dim) to latent variable distribution parameters
    z_mu and z_sigma (shapes n_batch x z_dim)
    """


    # should be a generic np class allowing for different encoders/decoders and combination methods
    def __init__(self, encoder, decoder, combiner, r_to_dist_encoder, n_repeat=1, training_type='VI'):
        super(NeuralProcess, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.combiner = combiner
        self.r_to_z_dist = r_to_dist_encoder
        self.n_repeat = n_repeat
        self.training_type = training_type

    def encode_to_z_params(self, x, y):
        # also should have any needed reshaping etc
        raw_rs = self.encoder(x, y)
        combined_r = self.combiner(raw_rs)
        z_mu, z_sigma = self.r_to_z_dist(combined_r)
        return z_mu, z_sigma

    def forward(self, x_context, y_context, x_target, y_target):
        # always encode the context
        z_context_mu, z_context_sigma = self.encode_to_z_params(x_context, y_context)
        dist_context = Normal(z_context_mu, z_context_sigma)
        
        if self.training:
            if self.training_type == 'VI':
                # in training for VI actually also encode the target (which the context is a subset of)
                # and use the z sampled from the more informed approximated posterior
                z_target_mu, z_target_sigma = self.encode_to_z_params(x_target, y_target)
                dist_target = Normal(z_target_mu, z_target_sigma)
                z_sample = [dist_target.rsample() for i in range(self.n_repeat)]
            elif self.training_type == "MLE":
                z_sample = [dist_context.rsample() for i in range(self.n_repeat)]
                dist_target=dist_context


        else:
            print("Evaluating")
            # in testing we do not care about the returned context distributions so can just use a dummy
            # however we do need to sample from the context-encoding-parametrised z

            # TODO Does it make sense to put something else in here? How would we calculate validation loss
            dist_target=dist_context ## Added for validation step - makes KL 0
            z_sample = [dist_context.rsample()]

        y_mu_sigma = [self.decoder(x_target, z_sample_i) for z_sample_i in z_sample]

        dist_y = [Normal(y_mu_i, y_sigma_i) for y_mu_i, y_sigma_i in y_mu_sigma]
 
        return dist_y, dist_context, dist_target



class SimpleNP(NeuralProcess):
    """
    Class to be used as a shortcut for creating a simple neural process 
    with all modules using linear MLPs 
    """
    def __init__(self, x_dim, y_dim, r_dim=16, z_dim=16, h_dims_enc = [64, 64], h_dims_dec=[64,64], h_dim = 128, n_repeat=1, training_type='VI'):

        # to make compatible with tf notebook (https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb)
        # values should be r_dim = 128, z_dim=128, h_dims_enc = [128]*4, h_dims_dec=[128]*2, h_dim=128

        encoder = LinearEncoder(x_dim, y_dim, h_dims_enc, r_dim)
        decoder = LinearDecoder(x_dim, z_dim, h_dims_dec, y_dim)
        combiner = MeanCombiner()
        r_to_dist_encoder = LinearRToDist(r_dim, h_dim, z_dim)

        super().__init__(
            encoder,
            decoder,
            combiner,
            r_to_dist_encoder,
            n_repeat,
            training_type
        )

