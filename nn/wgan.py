import os

import numpy as np
import time
from pycrayon import CrayonClient
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm


experiment_name = 'wgan_{}'.format(int(time.time()))
alpha = 0.00005
c = 0.01
m = 64
n_critic = 5
in_features = 2

base_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
)
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results', 'wgan', experiment_name)
n_epochs = 100000


def main():

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    generator = Generator(in_features=in_features)
    discriminator = Discriminator()
    logger = Logger(experiment_name=experiment_name, backup_dir='/tmp')

    optimizer_discriminator = optim.RMSprop(
        params=discriminator.parameters(),
        lr=alpha
    )
    optimizer_generator = optim.RMSprop(
        params=generator.parameters(),
        lr=alpha
    )
    wasserstein_dist_fn = WassersteinDistance()
    generator_loss_fn = GeneratorLoss()

    data_loader = get_data_loader()
    for i in tqdm(range(n_epochs)):
        for i_critic in range(n_critic):
            try:
                real_examples = next(data_loader)[0]
            except StopIteration:
                data_loader = get_data_loader()
                real_examples = next(data_loader)[0]
            real_examples = Variable(real_examples)
            fake_examples = sample_fake_examples(
                generator, n_examples=m, in_features=in_features
            )

            discriminator.zero_grad()
            real_example_preds = discriminator(real_examples)
            fake_example_preds = discriminator(fake_examples)

            wasserstein_dist = wasserstein_dist_fn(
                real=real_example_preds,
                fake=fake_example_preds
            )
            # Pass -1 since we're maximizing the Wasserstein distance
            wasserstein_dist.backward(torch.Tensor([-1]))
            optimizer_discriminator.step()

            for p in discriminator.parameters():
                p.data.clamp_(-c, c)

        generator.zero_grad()
        fake_examples = sample_fake_examples(
            generator, n_examples=m, in_features=in_features
        )
        gen_loss = generator_loss_fn(discriminator(fake_examples))
        gen_loss.backward()
        optimizer_generator.step()

        if i % 10 == 0:
            logger.log('generator_loss', gen_loss.data[0], step=i)
        if i % 500 == 0:
            imgs = sample_fake_examples(
                generator,
                n_examples=16,
                in_features=in_features
            ).data.unsqueeze(1)
            save_image(
                imgs,
                os.path.join(results_dir, 'examples_iter_{}.jpg'.format(i))
            )
            torch.save(generator, os.path.join(results_dir, 'latest_model.torch'))



def get_data_loader():
    return iter(DataLoader(
        dataset=MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=ToTensor()
        ),
        batch_size=m
    ))


def sample_fake_examples(generator, n_examples, in_features):
    z = Variable(torch.Tensor(np.random.randn(n_examples, in_features)))
    generated_examples = generator(z)

    return generated_examples


class Generator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 28*28)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        x = x.view(-1, 28, 28)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)

        return x


class WassersteinDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, fake):
        return (real.sum() - fake.sum()) / fake.size(0)


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
    #
    def forward(self, x):
        return - x.mean()


class Logger:
    def __init__(self, experiment_name, backup_dir, host='localhost', port='8889'):
        self.client = CrayonClient(
            hostname=host,
            port=port
        )
        self.experiment_name = experiment_name
        self.experiment = self.client.create_experiment(self.experiment_name)
        self.backup_dir = backup_dir

    def log(self, key, val, step=-1):
        self.experiment.add_scalar_value(key, val, step=step)

    def backup(self):
        self.experiment.to_zip(
            filename=os.path.join(self.backup_dir, 'logs_backup')
        )


if __name__ == '__main__':
    main()
