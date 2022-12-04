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
from models import ResNet50_server,ResNet50_clients
from torchvision import models

if __name__ == '__main__':
    
    args = args_parser()
    exp_details(args)

    device = "cuda" if args.gpu != None else "cpu"
    test_accuracy, test_loss_avg = [], []
    train_loss_avg = []
    val_acc_list, net_list = [],[]
    cv_loss, cv_acc = [], []
    print_every = 20
    val_loss_pre, counter = 0, 0

    loss_fn = torch.nn.CrossEntropyLoss()
    g = torch.Generator()
    
    train_dataset, test_dataset, user_groups = get_dataset(args)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False, num_workers=2, generator=g)

    global_model = ResNet50_server(norm_layer = args.norm_server)
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, global_losses = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users),1)
        idxs_users = np.random.choice(range(args.num_users),m, replace=False)


        for idx in idxs_users:
            local_model = LocalUpdate(args=args,dataset=train_dataset,idxs=user_groups[idx])
            w, loss = local_model.update_weights(model = copy.deepcopy(global_model),global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        train_loss_avg.append(sum(local_losses)/len(local_losses))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        total, correct = 0, 0 
        global_model.eval()
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                yhat = global_model(x)
                _, predicted = torch.max(yhat.data, 1)
                global_losses.append(loss_fn(yhat, y).item())
                total += y.size(0)
                correct += (predicted == y).sum().item()
        test_loss_avg.append(sum(global_losses) / len(global_losses))
        test_accuracy.append(correct / total)
        # print global training loss after every 'i' rounds


        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Avg Test Loss: {:.2f}%".format(sum(global_losses) / len(global_losses)))
        print("|---- Test Accuracy: {:.2f}%".format((correct / total)*100))

    
    train_dict = {'Epochs': np.array(range(args.epochs)),'Train Loss Average' : np.array(train_loss_avg),'Test Loss': np.array(test_loss_avg), 'Test accuracy': np.array(test_accuracy)}
    train_csv = pd.DataFrame(train_dict)
    train_csv.to_csv(f'FedAVG_Norm:{args.norm_clients}_iid:{args.iid}_lr:{args.lr}_mom:{args.momentum}_epochs:{args.epochs}.csv', index = False)


