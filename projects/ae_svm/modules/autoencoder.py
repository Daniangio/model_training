import torch
import torch.nn as nn
from common.base_module import BaseModule
from common.resnet import ResNetEncoder, ResNetDecoder
from settings import settings
import torch.nn.functional as F


def _init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight, gain=1)
    if type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_normal(m.weight, gain=1)
    if type(m) == nn.BatchNorm2d:
        torch.nn.init.uniform_(m.weight, -1, 1)


class Autoencoder(BaseModule):
    def __init__(self, in_channels=3, encoder_block_sizes=(64, 128, 256, 512), decoder_block_sizes=(16, 32, 64, 2),
                 decoder_depths=(2, 3, 2, 2), **_):
        super(Autoencoder, self).__init__()
        self.encoder_block_sizes = encoder_block_sizes

        # ENCODER
        self.encoder = ResNetEncoder(in_channels, blocks_sizes=encoder_block_sizes)
        # Output channels: 2048 = 512 * 4 = last block size * resnet bottleneck block expansion

        # self.decoder = ResNetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels,
        #                              blocks_sizes=decoder_block_sizes, depths=decoder_depths)
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.blocks[-1].blocks[-1].expanded_channels,
                      self.encoder.blocks[-1].blocks[-1].expanded_channels),
            nn.Linear(self.encoder.blocks[-1].blocks[-1].expanded_channels, 64 * 64 * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded.view(encoded.size(0), -1))
        decoded = decoded.view(-1, 3, 64, 64)
        return decoded, encoded

    def init_weights(self):
        self.apply(_init_weights)

    def get_weights_filename(self):
        return f'autoencoder_{"_".join(str(x) for x in self.encoder_block_sizes)}_v{settings.ae_svm_model_version}.pt'

    def set_preprocessing_params(self, params):
        # params['mean'] = [0.5]
        # params['std'] = [0.5]
        return params


class VAE(BaseModule):
    def __init__(self, input_size: int = 64, channels: int = 3, zdims: int = 100):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.zdims = zdims
        self.fc1 = nn.Linear(input_size * input_size * channels, 4000)
        self.bnfc1 = nn.BatchNorm1d(4000)
        self.fc21 = nn.Linear(4000, zdims)
        self.fc22 = nn.Linear(4000, zdims)
        self.fc3 = nn.Linear(zdims, 4000)
        self.bnfc3 = nn.BatchNorm1d(4000)
        self.fc4 = nn.Linear(4000, input_size * input_size * channels)

    def encode(self, x):
        h1 = F.leaky_relu(self.bnfc1(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        h3 = F.leaky_relu(self.bnfc1(self.fc3(z)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size * self.input_size * self.channels))
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z, mu, logvar

    def init_weights(self):
        self.apply(_init_weights)

    def get_weights_filename(self):
        return f'vae_size{self.input_size}_ch{self.channels}_zdims{self.zdims}_v{settings.ae_svm_model_version}.pt'

    def set_preprocessing_params(self, params):
        params['mean'] = [0.5]
        params['std'] = [0.5]
        return params

    def plot_gradients(self, iteration: int):
        self.plot_layer_gradient(iteration, self.fc4)

    def plot_mean_var(self, mu, logvar, iteration):
        mu = mu.detach().cpu().numpy()
        self.logger.report_surface(
            "mu",
            "batch sample",
            iteration=iteration,
            matrix=mu,
            xaxis="feature",
            yaxis="sample index",
            zaxis="mu value",
        )
        logvar = logvar.detach().cpu().numpy()
        self.logger.report_surface(
            "logvar",
            "batch sample",
            iteration=iteration,
            matrix=logvar,
            xaxis="feature",
            yaxis="sample index",
            zaxis="logvar value",
        )

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, epoch):
        # how well do input x and output recon_x agree?
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size * self.input_size * self.channels))

        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= recon_x.size(0) * self.input_size * self.input_size * self.channels

        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        #print('----', recon_x[0, :5], x.view(-1, self.input_size * self.input_size * self.channels)[0, :5])
        #print(torch.mean(recon_x), torch.var(recon_x), torch.mean(x), torch.var(x))
        #print(BCE, KLD, '+++++')
        return BCE + KLD
