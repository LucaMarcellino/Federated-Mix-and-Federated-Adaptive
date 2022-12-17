import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

from gkt_models import ResNet49, ResNet8
from gkt_server import GKTServerTrainer
from gkt_client import GKTClientTrainer

from options import args_parser
from utils import exp_details,get_datasets
from reproducibility import make_it_reproducible
from sampling import get_user_groups

if __name__ == "__main__":

    #Retrieve Arguments
    args = args_parser()
    exp_details(args)

    #Reproducibility
    make_it_reproducible(args.seed)

    #GPU
    device = "cuda" if int(args.gpu) != 0 else "cpu"

    print(f"DEVICE: {device}")

	#Choosing norm
    if args.num_groups == 0:
        norm_type = "Batch Norm"
    elif args.num_groups > 0:
        norm_type = "Group Norm"

    #Model Declaration
    server_model = ResNet49(norm_type)
    client_model = ResNet8(norm_type)

    #Trainer Declaration
    server_trainer = GKTServerTrainer(args.num_users,device,server_model, args, args.seed)

    #Dataset Retrieval
    train_dataset, test_dataset =  get_datasets(augmentation=True)
    user_groups,_ = get_user_groups(train_dataset, iid = args.iid, unbalanced= args.unequal, tot_users = args.num_users)

    clients = []
    for client_idx in range(args.num_users):
        clients.append(GKTClientTrainer(client_idx, train_dataset, test_dataset,
                                        user_groups[client_idx], device, client_model, args))

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    #TRAINING
    for communication_round in tqdm(range(args.communication_rounds)):

        print(f"\nCommunication Round: {communication_round+1}\n")

        m = max(int(args.frac * args.num_users), 1)
        chosen_users = np.random.choice(range(args.num_users), m, replace=False)

        print(f"Chosen users : {chosen_users}")
        for idx in chosen_users:
            extracted_features_dict, extracted_logits_dict, labels_dict,\
            extracted_features_dict_test, labels_dict_test = clients[idx].train()

            server_trainer.add_local_trained_result(idx, extracted_features_dict, extracted_logits_dict, labels_dict,\
            extracted_features_dict_test, labels_dict_test)

        server_trainer.train(communication_round)

        for idx in chosen_users:
            global_logits = server_trainer.get_global_logits(idx)
            clients[idx].update_large_model_logits(global_logits)
        server_trainer.remove_records()

    train_metrics, test_metrics = server_trainer.get_metrics_lists()
    
    train_data = pd.DataFrame(train_metrics)
    train_data["norm"] = "BN" if norm_type == "Batch Norm" else "GN"
    train_data["independence"] = "iid" if args.iid else "noniid"
    train_data["balancement"] = "unbalanced" if args.unequal else "balanced"
    train_data["seed"] = args.seed

    test_data = pd.DataFrame(test_metrics)
    test_data["norm"] = "BN" if norm_type == "Batch Norm" else "GN"
    test_data["independence"] = "iid" if args.iid else "noniid"
    test_data["balancement"] = "unbalanced" if args.unequal else "balanced"
    test_data["seed"] = args.seed

    df_train = pd.concat([df_train, train_data], ignore_index=True)
    df_test = pd.concat([df_test, test_data], ignore_index=True)

df_train.to_csv("./fedgkt_train_iid:{}_unbalanced:{}_norm:{}.csv".format(args.iid,args.unequal,norm_type), index=False)
df_test.to_csv("./fedgkt_test_iid:{}_unbalanced:{}_norm:{}.csv".format(args.iid,args.unequal,norm_type), index=False)