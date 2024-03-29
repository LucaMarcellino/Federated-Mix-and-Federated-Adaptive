import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image),torch.tensor(label)

"""
Baseed on the implementation in https://github.com/AshwinRJ/Federated-Learning-PyTorch
"""


class LocalUpdate(object):
    def __init__(self, dataset, idxs, device, local_batch_size, local_epochs, worker_init_fn, generator, args):
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=local_batch_size, shuffle=True, worker_init_fn=worker_init_fn, generator=generator)
        self.device = 'cuda' if device else 'cpu'
        self.local_epochs = local_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def update_weights(self, model, lr):
        model.train()
        epoch_loss = []

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images).to(self.device)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)