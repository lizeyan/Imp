import pyro
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import Loss, RunningAverage
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.optim import ClippedAdam
import pyro.distributions as dist
from typing import List, Union
from loguru import logger
from torch.utils.data import DataLoader
from visdom import Visdom


class IndependentGaussianStatistic(nn.Module):
    """
    loc, scale = IndependentGaussianStatistic()(x)
    """

    def __init__(self, input_dim, output_dim, hidden_dims: List = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [400, ]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)

        layers = [nn.Linear(input_dim, hidden_dims[0])]
        for in_features, out_features in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        self.shared_model = nn.Sequential(*layers)

        self.loc_model = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
        )
        self.scale_model = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor in shape (batch_size, input_dim)
        :return: (loc, scale)
        """
        assert (len(x.size()) == 2 and x.size()[1] == self.input_dim), f"bad x shape :{x.size()}"
        shared = self.shared_model(x)
        loc = self.loc_model(shared)
        scale = self.scale_model(shared)
        return loc, scale


class VAE(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, device='cpu'):
        super().__init__()
        self.device = device
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.encoder = IndependentGaussianStatistic(x_dim, z_dim)
        self.decoder = IndependentGaussianStatistic(z_dim, x_dim)
        self.to(device)

    def forward(self, x):
        return self.reconstruct(x)

    def generate(self, batch_size=1):
        with pyro.plate("batch_data", batch_size):
            z_loc = torch.zeros(batch_size, self.z_dim, device=self.device)
            z_scale = torch.ones(batch_size, self.z_dim, device=self.device)
            z = pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
            x_loc, _ = self.decoder(z)
            x_loc = torch.sigmoid(x_loc)
            x = pyro.sample('obs', dist.Bernoulli(x_loc).to_event(1))
            return x_loc

    def model(self, x: torch.Tensor):
        """
        :param x: Tensor in shape (batch_size, x_dim)
        :return:
        """
        pyro.module('decoder', self.decoder)
        batch_size = x.size()[0]
        observed = pyro.condition(lambda: self.generate(batch_size), data={'obs': x})()
        return observed

    def guide(self, x):
        pyro.module('encoder', self.encoder)
        batch_size = x.size()[0]
        with pyro.plate("batch_data", batch_size):
            z_loc, z_scale = self.encoder(x)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            return z

    def reconstruct(self, x):
        z = self.guide(x)
        x_z = pyro.condition(self.model, {'latent': z})(x)
        return x_z


def main_test_mnist():
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize
    transform = Compose([ToTensor()])
    train_dataset = MNIST(root="/tmp", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="/tmp", train=False, download=True, transform=transform)
    vae = VAE(x_dim=784, z_dim=50, device='cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n{vae}")
    optimizer = ClippedAdam({"lr": 1e-3})
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    def _update(engine, batch):
        vae.train()
        x, y = batch
        loss = svi.step(x.view(-1, 784).to(vae.device, non_blocking=True))
        return loss / len(x), len(x)

    def _evaluate(engine, batch):
        vae.eval()
        x, y = batch
        elbo = svi.evaluate_loss(x.view(-1, 784).to(vae.device, non_blocking=True))
        return elbo / len(x), len(x)

    trainer = Engine(_update)
    evaluater = Engine(_evaluate)
    train_dataloader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, pin_memory=True, drop_last=True, num_workers=8
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=256, shuffle=True, pin_memory=True, drop_last=True, num_workers=8
    )
    timer = Timer(average=True)
    timer.attach(
        engine=trainer, start=Events.EPOCH_STARTED, pause=Events.ITERATION_COMPLETED,
        resume=Events.ITERATION_STARTED, step=Events.ITERATION_COMPLETED
    )
    loss_metric = RunningAverage(output_transform=lambda outputs: -outputs[0], alpha=1)
    loss_metric.attach(engine=trainer, name="ELBO")
    loss_metric.attach(engine=evaluater, name="ELBO")
    vis = Visdom(server="gpu1.cluster.peidan.me", port=10697, env='Imp-pyro--vae-MNIST')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_loss(engine):
        elbo = engine.state.metrics['ELBO']
        logger.info(
            f"epoch:{engine.state.epoch}, ELBO: {elbo:.2f}, step time: {timer.value():.3f}s"
        )
        vis.line(Y=[elbo], X=[engine.state.epoch], win="Train-ELBO", update='append', opts={"title": "Train-ELBO"})

    def plot_vae_samples(title):
        x = torch.zeros([1, 784]).to(vae.device)
        for i in range(10):
            images = []
            for rr in range(100):
                # get loc from the model
                sample_loc_i = vae.model(x)
                img = sample_loc_i[0].view(1, 28, 28).cpu().data.numpy()
                images.append(img)
            vis.images(images, 10, 2, win=title, opts={'title': title})

    @trainer.on(Events.EPOCH_COMPLETED)
    def generate_samples(engine):
        epoch = engine.state.epoch
        if epoch % 10 == 0:
            logger.info(f"epoch: {epoch}, plot samples")
            plot_vae_samples(f"epoch-{epoch}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def validation(engine):
        epoch = engine.state.epoch
        if epoch % 5 == 0:
            evaluater.run(test_dataloader)
            elbo = evaluater.state.metrics['ELBO']
            logger.info(f"epoch: {epoch}, validation ELBO: {elbo}")
            vis.line(Y=[elbo], X=[engine.state.epoch], win="Validation-ELBO", update='append', opts={'title': "Validation-ELBO"})

    trainer.run(train_dataloader, max_epochs=2500)


if __name__ == '__main__':
    main_test_mnist()
