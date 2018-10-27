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


# TODO Build two separate data loaders, one for unlabeled and one for labeled.
# Now 100/60000 examples will be labeled, giving too sparse of a signal, leading
# to mode collapse

experiment_name = 'ladder_{}'.format(int(time.time()))
base_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
)
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results', 'ladder', experiment_name)

batch_size = 100
n_epochs = 100


def main():
    model = LadderNetwork(in_features=28*28, n_hidden=[1000, 500, 250, 250, 250], n_classes=10)
    reconstruction_loss = ReconstructionLoss(weights=[1000, 10, 0.1])
    classification_loss = nn.NLLLoss(weight=torch.FloatTensor([1] * 10 + [0]))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    labeled_train_data, unlabeled_train_data, val_data = get_data_loaders(n_labeled=1000)

    for _ in tqdm(range(n_epochs)):
        res = []
        r_losses = []
        c_losses = []
        # for batch_idx, (X, y) in enumerate(train_data):
        labeled_iterator = iter(labeled_train_data)
        unlabeled_iterator = iter(unlabeled_train_data)
        for i in range(len(unlabeled_train_data)):
            X_unlabeled, y_unlabeled = next(unlabeled_iterator)
            try:
                X_labeled, y_labeled = next(labeled_iterator)
            except StopIteration:
                unlabeled_iterator = iter(unlabeled_train_data)
            X = torch.cat([X_labeled, X_unlabeled], dim=0)
            y = torch.cat([y_labeled, y_unlabeled], dim=0)

            model.train()
            X, y = Variable(X), Variable(y)

            optimizer.zero_grad()
            z_clean, z_hats, y_pred = model(X)
            model.train()
            r_loss = reconstruction_loss(
                z_clean, z_hats
            )
            c_loss = classification_loss(
                y_pred, y
            )
            r_losses.append(r_loss.data.numpy())
            c_losses.append(c_loss.data.numpy())
            loss = r_loss + c_loss

            labeled_idx = np.where(y.data.numpy() != 10)[0]
            if len(labeled_idx) > 0:
                res.append(
                    sum(
                        y.data.numpy()[labeled_idx] == np.argmax(y_pred.data.numpy()[labeled_idx], axis=1)
                    ) / len(labeled_idx)
                )

            loss.backward()
            optimizer.step()

        print('Classification acc: \t\t{}'.format(np.mean(res)))
        print('Reconstruction loss: \t\t{}'.format(np.mean(r_losses)))
        print('Classification loss: \t\t{}'.format(np.mean(c_losses)))
        print('Total loss: \t\t{}'.format(np.mean(c_losses) + np.mean(r_losses)))
        test(model, val_data)


def test(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        _, output = model.clean_forward(data)
        val_loss += F.nll_loss(output, target, size_average=False).data[0]  # Sum up batch loss
        pred = output.data.max(1)[1]  # Get the index of the max log-probability
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

    n_classes = 10

    labeled_idx = []
    unlabeled_idx = []
    for class_label in range(n_classes):
        class_idx = np.where(train_dataset.train_labels.numpy() == class_label)[0]
        idx_to_keep_labeled = np.random.choice(
            class_idx,
            int(n_labeled / n_classes),
            replace=False
        )
        labeled_idx.extend(idx_to_keep_labeled)
        idx_to_remove_labels = np.array(
            [i for i in class_idx if i not in idx_to_keep_labeled]
        )
        unlabeled_idx.extend(idx_to_remove_labels)
        train_dataset.train_labels[torch.from_numpy(idx_to_remove_labels)] = 10  # No-label class

    labeled_train_dataset = PartialMNIST(
        X=train_dataset.train_data[torch.from_numpy(np.array(labeled_idx))],
        y=train_dataset.train_labels[torch.from_numpy(np.array(labeled_idx))],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    unlabeled_train_dataset = PartialMNIST(
        X=train_dataset.train_data[torch.from_numpy(np.array(unlabeled_idx))],
        y=train_dataset.train_labels[torch.from_numpy(np.array(unlabeled_idx))],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    labeled_train_loader = torch.utils.data.DataLoader(
        labeled_train_dataset,
        batch_size=batch_size//2,
        shuffle=True
    )
    unlabeled_train_loader = torch.utils.data.DataLoader(
        unlabeled_train_dataset,
        batch_size=batch_size//2,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return labeled_train_loader, unlabeled_train_loader, val_loader


class PartialMNIST(MNIST):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.train_data = X
        self.train_labels = y
        self.transform = transform
        self.target_transform = target_transform
        self.train = True

    def __len__(self):
        return len(self.train_labels)


class LadderNetwork(nn.Module):
    def __init__(self, in_features, n_hidden, n_classes, std=0.2):
        super().__init__()
        self.std = std

        self.in_features_n_hidden_output = [in_features, *n_hidden, n_classes]
        self.n_hidden = n_hidden

        self.encoder_linear_layers = nn.ModuleList([
            nn.Linear(
                in_features=self.in_features_n_hidden_output[i],
                out_features=self.in_features_n_hidden_output[i + 1],
                bias=False
            ) for i in range(len(self.in_features_n_hidden_output) - 1)
        ])

        self.encoder_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(
                num_features=self.in_features_n_hidden_output[i],
                affine=False
            ) for i in range(1, len(self.in_features_n_hidden_output))
        ])

        self.encoder_gammas = nn.ParameterList([
            nn.Parameter(
                data=torch.randn(self.in_features_n_hidden_output[i])
            ) for i in range(1, len(self.in_features_n_hidden_output))
        ])

        self.encoder_betas = nn.ParameterList([
            nn.Parameter(
                data=torch.randn(self.in_features_n_hidden_output[i])
            ) for i in range(1, len(self.in_features_n_hidden_output))
        ])


        self.decoder_linear_layers = nn.ModuleList([
            nn.Linear(
                in_features=self.in_features_n_hidden_output[i],
                out_features=self.in_features_n_hidden_output[i - 1],
                bias=False
            ) for i in range(1, len(self.in_features_n_hidden_output))
        ])

        self.decoder_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(
                num_features=self.in_features_n_hidden_output[i],
                affine=False
            ) for i in range(len(self.in_features_n_hidden_output))
        ])

        self.combinators = nn.ModuleList([
            Combinator(
                in_features=self.in_features_n_hidden_output[i]
            ) for i in range(len(self.in_features_n_hidden_output))
        ])


    def forward(self, x_input):

        z_tildes = []

        # Encoder
        x_input = x_input.view(-1, 28*28)
        x = x_input + Variable(self.std * torch.randn(x_input.size()))
        z_tildes.append(x)

        # Encoder
        for i in range(len(self.encoder_linear_layers)):
            x = self.encoder_linear_layers[i](x)
            x = self.encoder_batch_norms[i](x)
            x = x + Variable(self.std * torch.randn(x.size()))
            z_tildes.append(x)
            x = F.relu(self.encoder_gammas[i].expand_as(x) * (x + self.encoder_betas[i].expand_as(x)))

        # Supervised output
        y_pred = self._classify(x)
        z_tildes.append(y_pred)  # REMOVE!

        # Clean encoder
        mus = []
        sigmas = []
        x = x_input


        zs_clean = []
        zs_clean.append(x)

        mu = torch.mean(x)#, dim=0)
        mus.append(mu)
        sigma = torch.std(x)#, dim=0)
        sigmas.append(sigma)

        for i in range(len(self.encoder_linear_layers)):
            z_pre = self.encoder_linear_layers[i](x)
            mu = torch.mean(z_pre, dim=0)
            mus.append(mu)
            sigma = torch.std(z_pre, dim=0)
            sigmas.append(sigma)
            x = self.encoder_batch_norms[i](z_pre)  # Should it be put into eval mode first?
            zs_clean.append(x)
            x = F.relu(self.encoder_gammas[i].expand_as(x) * (x + self.encoder_betas[i].expand_as(x)))

        # Decoder

        z_hats = [None] * len(zs_clean)
        for i in range(len(self.in_features_n_hidden_output))[::-1]:
            if i == len(self.in_features_n_hidden_output) - 1:
                u = self.decoder_batch_norms[i](y_pred)
            else:
                u_ = self.decoder_linear_layers[i](z_hats[i + 1])
                u = self.decoder_batch_norms[i](u_)
            z_hat = self.combinators[i](z_tildes[i], u)
            z_hat_bn = (z_hat - mus[i].expand_as(z_hat)) / sigmas[i].expand_as(z_hat)
            z_hats[i] = z_hat_bn

        # Need to save and return mean and stds for the reconstruction cost to use.
        y_pred = torch.cat([y_pred, Variable(-999999999 * torch.ones(y_pred.size(0), 1))], dim=1)

        return zs_clean, z_hats, y_pred

    def clean_forward(self, x):
        # TODO Deduplicate encoding part
        x = x.view(-1, 28*28)

        zs_clean = []
        zs_clean.append(x)

        for i in range(len(self.encoder_linear_layers)):
            z_pre = self.encoder_linear_layers[i](x)
            x = self.encoder_batch_norms[i](z_pre)  # Should it be put into eval mode first?
            zs_clean.append(x)
            x = F.relu(
                self.encoder_gammas[i].expand_as(x) * (x + self.encoder_betas[i].expand_as(x))
            )

        y_pred = self._classify(x)

        return zs_clean, y_pred

    def _classify(self, x):
        return F.log_softmax(x)


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
        a = 0
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

        assert len(zs) == len(z_hats), 'Lengths not equal for reconstruction'
        loss = 0
        for z, z_hat, weight in zip(zs, z_hats, self.weights):
            dl = weight * torch.mean(
                torch.pow(z - z_hat, 2)
            ) / z_hat.size(1)
            loss += dl

        return loss


if __name__ == '__main__':
    main()
