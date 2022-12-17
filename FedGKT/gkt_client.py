import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from reproducibility import make_it_reproducible,seed_worker

import numpy as np
import pandas as pd
import utils

g = torch.Generator()

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class GKTClientTrainer(object):
    def __init__(self, client_index, local_training_data, local_test_data, local_sample_number, device, client_model, args):
        self.client_index = client_index

        self.local_training_data = local_training_data
        self.local_test_data = local_test_data

        self.local_sample_number = local_sample_number

        self.args = args

        self.device = device
        self.client_model = client_model

        self.client_model.to(self.device)

        self.optimizer = optim.SGD(self.client_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = utils.KL_Loss(self.args.temperature)

        self.server_logits_dict = dict()

    def get_sample_number(self):
        return self.local_sample_number

    def update_large_model_logits(self, logits):
        self.server_logits_dict = logits
    
    def remove_records(self):
        for idx in self.client_extracted_feauture_dict.keys():
            self.client_extracted_feauture_dict[idx].clear()
            self.client_logits_dict[idx].clear()
            self.client_labels_dict[idx].clear()
            self.server_logits_dict[idx].clear()
        for id in self.client_extracted_feauture_dict_test.keys():
            self.client_extracted_feauture_dict_test[idx].clear()
            self.client_labels_dict_test[idx].clear()
        self.client_extracted_feauture_dict.clear()
        self.client_logits_dict.clear()
        self.client_labels_dict.clear()
        self.server_logits_dict.clear()
        self.client_extracted_feauture_dict_test.clear()
        self.client_labels_dict_test.clear()

    def train(self):
        # key: batch_index; value: extracted_feature_map
        extracted_feature_dict = dict()

        # key: batch_index; value: logits
        logits_dict = dict()

        # key: batch_index; value: label
        labels_dict = dict()

        # for test - key: batch_index; value: extracted_feature_map
        extracted_feature_dict_test = dict()
        labels_dict_test = dict()

        self.client_model.train()
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            trainloader = DataLoader(DatasetSplit(self.local_training_data, self.local_sample_number), batch_size = self.args.local_bs, shuffle=True, num_workers=2, worker_init_fn = seed_worker, generator=g)
            for batch_idx, data in enumerate(trainloader):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                log_probs, _ = self.client_model(images)
                loss_true = self.criterion_CE(log_probs, labels)
                if len(self.server_logits_dict) != 0:
                    large_model_logits = torch.from_numpy(self.server_logits_dict[batch_idx]).to(
                        self.device)
                    loss_kd = self.criterion_KL(log_probs, large_model_logits)
                    loss = loss_true + self.args['alpha'] * loss_kd
                else:
                    loss = loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('client {} - Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.client_index, epoch, batch_idx * len(images), len(trainloader),
                                              100. * batch_idx / len(trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.client_model.eval()

        for batch_idx, data in enumerate(trainloader):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            log_probs, extracted_features = self.client_model(images)

            extracted_feature_dict[batch_idx] = extracted_features.cpu().detach().numpy()
            log_probs = log_probs.cpu().detach().numpy()
            logits_dict[batch_idx] = log_probs
            labels_dict[batch_idx] = labels.cpu().detach().numpy()

        testloader = DataLoader(self.local_test_data, batch_size = self.args.local_bs, shuffle=True, num_workers=2, worker_init_fn = seed_worker, generator=g)
        for batch_idx, data in enumerate(testloader):
            images, labels = data
            test_images, test_labels = images.to(self.device), labels.to(self.device)
            _, extracted_features_test = self.client_model(test_images)
            extracted_feature_dict_test[batch_idx] = extracted_features_test.cpu().detach().numpy()
            labels_dict_test[batch_idx] = test_labels.cpu().detach().numpy()

        return extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test