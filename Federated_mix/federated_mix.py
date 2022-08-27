#Utils and System Libraries
import copy
import time
from tqdm import tqdm

#Math Libraries
import numpy as np
import pandas as pd

#Pytorch
import torch
import torch.nn as nn
from options import args_parser 
from utils import exp_details, get_dataset, average_weights
from update import LocalUpdate, DatasetSplit, test_inference
from mix_models import ResNet50
from torchvision import models

if __name__ == '__main__':
    
    args = args_parser()
    exp_details(args)

    device = "cuda" if args.gpu != None else "cpu"
    
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    global_model = ResNet50(alpha_b = 0.5, alpha_g = 0.5)
    global_model.to(device)

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [],[]
    cv_loss, cv_acc = [], []
    print_every = 20
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users),1)
        idxs_users = np.random.choice(range(args.num_users),m, replace=False)


        for idx in idxs_users:
            local_model = LocalUpdate(args=args,dataset=train_dataset,idxs=user_groups[idx])
            local_resnet = ResNet50(alpha_b = 0.1, alpha_g = 0.9)
            gw = global_model.state_dict()
            local_resnet.load_state_dict(gw,strict = False)
            local_resnet.to(device)
            w, loss = local_model.update_weights(model = local_resnet,global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights)

        global_model.load_state_dict(global_weights, strict = False)

        loss_avg = sum(local_losses)/len(local_losses)
        train_loss.append(loss_avg)

        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args,dataset=train_dataset,idxs=user_groups[idx])
            acc, loss = local_model.inference(model = global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))


        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    
    train_dict = {'Epochs': np.array(range(args.epochs)),'Train Loss': np.mean(np.array(train_loss)), 'Train accuracy': np.array(train_accuracy)}
    train_csv = pd.DataFrame(train_dict)
    train_csv.to_csv(f'FedAVG_C:{args.norm_clients}_S:{args.norm_server}_iid:{args.iid}_lr:{args.lr}_mom:{args.momentum}_epochs:{args.epochs}.csv', index = False)

    



