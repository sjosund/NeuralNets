import os
import time

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from tqdm import tqdm

experiment_name = 'ladder_{}'.format(int(time.time()))
base_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
)
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results', 'ladder', experiment_name)

batch_size = 100


def main():
    model = LadderNetwork(in_features=28*28, n_classes=10)
    reconstruction_loss = ReconstructionLoss(weights=[1000, 10, 0.1])
    classification_loss = nn.NLLLoss(weight=torch.FloatTensor([1] * 10 + [0]))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_data, val_data = get_data_loaders(n_labeled=50000)

    n_epochs = 10

    for _ in tqdm(range(n_epochs)):
        res = []
        for batch_idx, (X, y) in enumerate(train_data):
            model.train()
            X, y = Variable(X), Variable(y)

            optimizer.zero_grad()
            # if np.random.uniform(0, 1) > 0.5:
            y_reconstruction, y_pred = model(X, has_y=True)
            model.eval()
            y_hidden, _ = model.clean_forward(X)
            model.train()
            loss = reconstruction_loss(
                y_hidden, y_reconstruction
            ) + classification_loss(
                y_pred, y
            )

            res.append(
                sum(
                    y.data.numpy() == np.argmax(y_pred.data.numpy(), axis=1)
                ) / batch_size
            )

            loss.backward()
            optimizer.step()

        print('Classification acc: \t\t{}'.format(np.mean(res)))
        test(model, val_data)


def test(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        _, output = model.clean_forward(data)
        val_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(val_loader.dataset)
    print('\nval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def get_data_loaders(n_labeled):
    train_dataset = MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    val_dataset = MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    non_labeled_idx = torch.LongTensor(
        np.random.choice(
            range(len(train_dataset)),
            len(train_dataset) - n_labeled,
            replace=False
        )
    )
    train_dataset.train_labels[non_labeled_idx] = 10  # No-label class

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, val_loader



class LadderNetwork(nn.Module):
    def __init__(self, in_features, n_classes, std=0.2):
        super().__init__()
        self.std = std
        self.l1 = nn.Linear(in_features=in_features, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=128)

        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=128)

        self.ul1 = nn.Linear(in_features=128, out_features=128)
        self.ul2 = nn.Linear(in_features=128, out_features=128)
        self.ul3 = nn.Linear(in_features=128, out_features=in_features)

        self.ubn1 = nn.BatchNorm1d(num_features=128)
        self.ubn2 = nn.BatchNorm1d(num_features=128)
        self.ubn3 = nn.BatchNorm1d(num_features=in_features)

        self.combinator1 = Combinator(in_features=128)
        self.combinator2 = Combinator(in_features=128)
        self.combinator3 = Combinator(in_features=in_features)

        self.classifier = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x, has_y=False):

        reconstructions = []

        # Encoder
        x = x.view(-1, 28*28)
        z1 = x + Variable(self.std * torch.randn(x.size()))

        x = self.l1(x)
        # x = self.bn1(x)
        z2 = x + Variable(self.std * torch.randn(x.size()))
        x = F.relu(z2)

        x = self.l2(x)
        x = self.bn2(x)
        z3 = x + Variable(self.std * torch.randn(x.size()))
        x = F.relu(z3)

        if has_y:
            y_pred = self._classify(x)
            y_pred = torch.cat([y_pred, Variable(-np.inf * torch.ones(y_pred.size(0), 1))], dim=1)
        else:
            y_pred = None

        # Decoder
        x = self.ul1(x)
        x = self.ubn1(x)
        zr3 = self.combinator1(x, z3)

        x = self.ul2(x)
        x = self.ubn2(x)
        zr2 = self.combinator2(x, z2)

        x = self.ul3(x)
        x = self.ubn3(x)
        zr1 = self.combinator3(x, z1)

        # Need to save and return mean and stds for the reconstruction cost to use.
        return [zr1, zr2, zr3], y_pred

    def clean_forward(self, x):
        x = x.view(-1, 28*28)
        z1 = x

        x = self.l1(x)
        z2 = self.bn1(x)
        x = F.relu(z1)

        x = self.l1(x)
        z3 = self.bn1(x)
        x = F.relu(z2)

        y_pred = self._classify(x)

        return [z1, z2, z3], y_pred

    def _classify(self, x):
        return F.log_softmax(self.classifier(x))


class Combinator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.b0 = nn.Parameter(torch.zeros(batch_size, in_features))
        self.w0z = nn.Parameter(torch.ones(batch_size, in_features))
        self.w0u = nn.Parameter(torch.zeros(batch_size, in_features))
        self.w0zu = nn.Parameter(torch.zeros(batch_size, in_features))
        self.w_sigma = nn.Parameter(torch.ones(batch_size, in_features))
        self.b1 = nn.Parameter(torch.zeros(batch_size, in_features))
        self.w1z = nn.Parameter(torch.ones(batch_size, in_features))
        self.w1u = nn.Parameter(torch.zeros(batch_size, in_features))
        self.w1zu = nn.Parameter(torch.zeros(batch_size, in_features))

    def forward(self, z, u):
        g = self.b0 +\
            self.w0z * z +\
            self.w0u * u +\
            self.w0zu * z * u +\
            self.w_sigma * F.sigmoid(self.b1 +
                                     self.w1z * z +
                                     self.w1u * u +
                                     self.w1zu * z * u)

        return g


class ReconstructionLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, zs, z_hats):
        loss = 0
        for z, z_hat, weight in zip(zs, z_hats, self.weights):
            loss += weight * torch.mean(
                torch.pow(z - z_hat, 2)
            )  # TODO normalize using sample statistics

        return loss


if __name__ == '__main__':
    main()
