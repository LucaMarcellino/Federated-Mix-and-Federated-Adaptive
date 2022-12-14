import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch.nn as tn

import numpy as np
import pandas as pd
import random

import utils


torch.manual_seed(0)

g = torch.Generator()
g.manual_seed(0)

np.random.seed(0)


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class GKTClientTrainer(object):
    def __init__(self, model, device,  train_dataset, test_dataset, idxs, client_index, args):
        self.model = model
        self.model_params = self.model.parameters()
        self.device = device
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # Batch size should be divisible by number of GPUs
            self.model = tn.DataParallel(self.model)
            next(model.parameters()).is_cuda
        else:
            self.model.to(self.device)

        self.optimizer = args.optimizer
        self.local_batch_size = args.local_bs
        self.lr = args.lr
        self.local_epochs = args.local_ep
        self.client_index = client_index

        self.trainloader, self.testloader = self.train_test(
            train_dataset, test_dataset, list(idxs))  # get train, valid sets
        
        

        # Set optimizer for the local updates
        if self.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model_params, lr=self.lr, momentum=0.9,
                                             nesterov=True, weight_decay=5e-04)
        elif self.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model_params, lr=self.lr, weight_decay=0.0001,
                                              amsgrad=True)

        self.server_logits_dict = dict()

        self.temperature = args.temperature
        self.alpha = args.partition_alpha

        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = utils.KL_Loss(self.temperature)  # knowledge distillation

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


    # for REPRODUCIBILITY https://pytorch.org/docs/stable/notes/randomness.html
    def seed_worker(args):
        worker_seed = args.seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_test(self, train_dataset, test_dataset, idxs):
        """
        Returns train and test dataloaders for a given dataset
        and user indexes.
        """
        trainloader = DataLoader(DatasetSplit(train_dataset, idxs),
                                 batch_size=self.local_batch_size, shuffle=True, generator=g,
                                 worker_init_fn=self.seed_worker)
        testloader = DataLoader(test_dataset,
                                batch_size=self.local_batch_size, shuffle=False, generator=g,
                                worker_init_fn=self.seed_worker)

        return trainloader, testloader

    # def get_sample_number(self):
    #     return self.local_sample_number

    def train(self):
        # Set mode to train model
        self.model.train()

        epoch_loss = []

        for epoch in range(1, self.local_epochs + 1):
            batch_loss = []

            for batch_idx, data in enumerate(self.trainloader):
                # move tensors to GPU if CUDA is available
                
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                log_probs, _ = self.model(images)
                loss_true = self.criterion_CE(log_probs, labels)

                if len(self.server_logits_dict) != 0:
                    large_model_logits = torch.from_numpy(self.server_logits_dict[batch_idx]).to(self.device)
                    loss_kd = self.criterion_KL(log_probs, large_model_logits)  # knowledge distillation
                    loss = loss_true + self.alpha * loss_kd 
                else:
                    loss = loss_true

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()

                batch_loss.append(loss.item())

            print('client {} - Update Epoch: {} - Loss: {:.6f}'.format(
                self.client_index, epoch, sum(batch_loss) / len(batch_loss)))

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # key: batch_index; value: extracted_feature_map
        extracted_feature_dict = dict()

        # key: batch_index; value: logits
        logits_dict = dict()

        # key: batch_index; value: label
        labels_dict = dict()

        # for test - key: batch_index; value: extracted_feature_map
        extracted_feature_dict_test = dict()
        labels_dict_test = dict()

        # Set mode to evaluation model
        self.model.eval()

        for batch_idx, data in enumerate(self.trainloader):
            
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            log_probs, extracted_features = self.model(images)

            extracted_feature_dict[batch_idx] = extracted_features.cpu().detach().numpy()
            log_probs = log_probs.cpu().detach().numpy()
            logits_dict[batch_idx] = log_probs  
            labels_dict[batch_idx] = labels.cpu().detach().numpy()
            

        for batch_idx, data in enumerate(self.testloader):
            # move tensors to GPU if CUDA is available
            images, labels = data
            test_images = images.to(self.device)
            test_labels = labels.to(self.device)

            _, extracted_features_test = self.model(test_images)

            if self.device != "cuda":
                extracted_feature_dict_test[batch_idx] = extracted_features_test.cpu().detach().numpy()
                labels_dict_test[batch_idx] = test_labels.cpu().detach().numpy()
            else:
                extracted_feature_dict_test[batch_idx] = extracted_features_test
                labels_dict_test[batch_idx] = test_labels

        return extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test
