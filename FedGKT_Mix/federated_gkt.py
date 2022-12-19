import torch
import numpy as np
import pandas as pd

from resnet_gkt import ResNet49, ResNet8
from datasets import get_datasets
from sampling import get_user_groups
from trainers import GKTServerTrainer, GKTClientTrainer
from reproducibility import make_it_reproducible
from options import args_parser

device = 'cuda' if torch.cuda.is_available else 'cpu'

args = args_parser()

seed = 0

client_number = args.num_users
norm_type = args.norm_layer
participation_frac = args.frac
iid = args.iid
unbalanced = args.unequal
ROUNDS = args.communication_rounds

args_server = {
    'temperature': 3,
    'epochs_server': 1,
    'alpha': 0.5
}

args_client={
    'temperature': 3,
    'epochs_client': 1,
    'alpha': 0.5
}

##### metric = []

df_train = pd.DataFrame()
df_test = pd.DataFrame()


make_it_reproducible(seed)
# server, reproducibility demanded to the server
server_model = ResNet49(norm_type)
server_trainer = GKTServerTrainer(client_number, device, server_model, args_server, seed=seed)

# client
client_model = ResNet8(norm_type)
trainset, testset = get_datasets(augmentation=True)
user_groups, _ = get_user_groups(trainset, iid=iid, unbalanced=unbalanced, tot_users=client_number)

clients = []
for client_idx in range(client_number):
    clients.append(GKTClientTrainer(client_idx, trainset, testset,
                                    user_groups[client_idx], device, client_model, args_client))

for round in range(ROUNDS):
    print("Communication round: ", round+1)
    m = max(int(participation_frac*client_number), 1)
    chosen_users = np.random.choice(range(client_number), m, replace=False)
    print(f"Chosen users: {chosen_users}")
    for idx in chosen_users:
        extracted_features_dict, extracted_logits_dict, labels_dict,\
        extracted_features_dict_test, labels_dict_test = clients[idx].train()

        server_trainer.add_local_trained_result(idx, extracted_features_dict, extracted_logits_dict, labels_dict,\
        extracted_features_dict_test, labels_dict_test)

    server_trainer.train(round)

    for idx in chosen_users:
        global_logits = server_trainer.get_global_logits(idx)
        clients[idx].update_large_model_logits(global_logits)
    server_trainer.remove_records()

train_metrics, test_metrics = server_trainer.get_metrics_lists()

train_data = pd.DataFrame(train_metrics)
train_data["norm"] = "BN" if norm_type == "Batch Norm" else "GN"
train_data["independence"] = "iid" if iid else "noniid"
train_data["balancement"] = "unbalanced" if unbalanced else "balanced"
train_data["seed"] = seed

test_data = pd.DataFrame(test_metrics)
test_data["norm"] = "BN" if norm_type == "Batch Norm" else "GN"
test_data["independence"] = "iid" if iid else "noniid"
test_data["balancement"] = "unbalanced" if unbalanced else "balanced"
test_data["seed"] = seed

df_train = pd.concat([df_train, train_data], ignore_index=True)
df_test = pd.concat([df_test, test_data], ignore_index=True)

df_train.to_csv("./results/federated_gkt/fedgkt_train_results.csv", index=False)
df_test.to_csv("./results/federated_gkt/fedgkt_test_results.csv", index=False)