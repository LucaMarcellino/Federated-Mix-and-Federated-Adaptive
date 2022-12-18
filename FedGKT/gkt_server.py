import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import utils
from reproducibility import make_it_reproducible,seed_worker

g = torch.Generator()

class GKTServerTrainer:

    def __init__(self, client_num, device, server_model, args, seed):
        self.client_num = client_num
        self.device = device
        self.args = args

        self.model_global = server_model
        self.model_global.to(self.device)

        self.model_global.train()

        self.optimizer = optim.SGD(self.model_global.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max')

        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = utils.KL_Loss(self.args.temperature)
        self.best_acc = 0.0

        # key: client_index; value: extracted_feature_dict
        self.client_extracted_feature_dict = dict()

        # key: client_index; value: logits_dict
        self.client_logits_dict = dict()

        # key: client_index; value: labels_dict
        self.client_labels_dict = dict()

        # key: client_index; value: labels_dict
        self.server_logits_dict = dict()

        # for test
        self.client_extracted_feature_dict_test = dict()
        self.client_labels_dict_test = dict()

        self.train_metrics_list = []
        self.test_metrics_list = []

        g.manual_seed(seed)
    
    def add_local_trained_result(self, idx, extracted_feature_dict, logits_dict, labels_dict,
                                 extracted_feature_dict_test, labels_dict_test):

        self.client_extracted_feature_dict[idx] = extracted_feature_dict
        self.client_logits_dict[idx] = logits_dict
        self.client_labels_dict[idx] = labels_dict
        self.client_extracted_feature_dict_test[idx] = extracted_feature_dict_test
        self.client_labels_dict_test[idx] = labels_dict_test

        print(len(self.client_extracted_feature_dict))
    
    def remove_records(self):
        for idx in self.client_extracted_feature_dict.keys():
            self.client_extracted_feature_dict[idx].clear()
            self.client_logits_dict[idx].clear()
            self.client_labels_dict[idx].clear()
            self.server_logits_dict[idx].clear()
        for idx in self.client_extracted_feature_dict.keys():
            self.client_extracted_feature_dict[idx].clear()
            self.client_labels_dict_test[idx].clear()
        self.client_extracted_feature_dict.clear()
        self.client_logits_dict.clear()
        self.client_labels_dict.clear()
        self.server_logits_dict.clear()
        self.client_extracted_feature_dict.clear()
        self.client_labels_dict_test.clear()

    def get_global_logits(self, client_index):
        return self.server_logits_dict[client_index]

    def train(self, round_idx):
        self.train_and_eval(round_idx, self.args.epochs)
        self.scheduler.step(self.best_acc)

    def train_and_eval(self, round_idx, epochs):
        for epoch in range(epochs):
            train_metrics = self.train_large_model_on_the_server(round_idx, epoch)
            self.train_metrics_list.append(train_metrics)
            print({"train/loss": train_metrics['train_loss'],"train/accuracy": train_metrics['train_acc'], "round": round_idx + 1})
            if epoch == epochs - 1:
                test_metrics = self.eval_large_model_on_the_server(round_idx)
                self.test_metrics_list.append(test_metrics)
                
                if test_metrics['test_acc'] >= self.best_acc:
                    self.best_acc= test_metrics['test_acc']
                
                print({"test/loss": test_metrics['test_loss'],"test/accuracy": test_metrics['test_acc'], "round": round_idx + 1})

    def train_large_model_on_the_server(self, round_idx, epoch):
        for key in self.server_logits_dict.keys():
            self.server_logits_dict[key].clear()
        self.server_logits_dict.clear()

        self.model_global.train()

        loss_avg = utils.RunningAverage()
        accTop1_avg = utils.RunningAverage()

        for client_index in self.client_extracted_feature_dict.keys():

            extracted_feature_dict = self.client_extracted_feature_dict[client_index]
            logits_dict = self.client_logits_dict[client_index]
            labels_dict = self.client_labels_dict[client_index]

            s_logits_dict = dict()
            self.server_logits_dict[client_index] = s_logits_dict
            for batch_index in extracted_feature_dict.keys():
                batch_feature_map_x = torch.from_numpy(extracted_feature_dict[batch_index]).to(self.device)
                batch_logits = torch.from_numpy(logits_dict[batch_index]).float().to(self.device)
                batch_labels = torch.from_numpy(labels_dict[batch_index]).long().to(self.device)

                output_batch = self.model_global(batch_feature_map_x)

                loss_true = self.criterion_CE(output_batch, batch_labels).to(self.device)
                loss = loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                metrics = utils.accuracy(output_batch, batch_labels, topk=(1,))
                accTop1_avg.update(metrics[0].item())
                loss_avg.update(loss.item())

                s_logits_dict[batch_index] = output_batch.cpu().detach().numpy()

        train_metrics = {
            "round": round_idx,
            "epoch": epoch,
            'train_loss': loss_avg.value(),
            'train_acc': accTop1_avg.value()}

        return train_metrics

    def eval_large_model_on_the_server(self, idxs_chosen_users):
        self.model_global.eval()

        loss_avg = utils.RunningAverage()
        acc_avg = utils.RunningAverage()

        with torch.no_grad():
            for client_index in self.client_extracted_feature_dict.keys():
                extracted_feature_dict = self.client_extracted_feature_dict[client_index]
                labels_dict = self.client_labels_dict_test[client_index]

                for batch_idx in extracted_feature_dict.keys():
                    batch_feature_map = torch.from_numpy(extracted_feature_dict[batch_idx].cpu().detach().numpy()).to(self.device)
                    batch_labels = torch.from_numpy(labels_dict[batch_idx].cpu().detach().numpy()).long().to(self.device)

                    output_batch = self.model_global(batch_feature_map)
                    loss = self.criterion_CE(output_batch, batch_labels)

                    metrics = utils.accuracy(output_batch, batch_labels, topk=(1,))
                    acc_avg.update(metrics[0].item())
                    loss_avg.update(loss.item())

        test_metrics = {
            "round": idxs_chosen_users,
            'test_loss': loss_avg.value(),
            'test_acc': acc_avg.value()}

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
        print("\n- TEST METRICS: " + metrics_string + "\n")

        return test_metrics

    def get_metrics_lists(self):
        return self.train_metrics_list, self.test_metrics_list
