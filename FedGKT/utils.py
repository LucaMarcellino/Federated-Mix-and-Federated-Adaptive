#Useful libraries
import copy
import torch
import torch.nn as nn
from torchvision import datasets,transforms
from sampling import cifar_iid, cifar_noniid, cifar_noniid_unbalanced
import numpy as np
import torch.nn.functional as F

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : ResNet50 for Server, ResNet8 for Client')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

    if args.gpu != 0 and torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")
    return

'''
def get_dataset(args):
    data_dir = '../data/cifar/'
    
    #Normalize used with mean and stds of Cifar10
    apply_transform = transforms.Compose(
        [transforms.RandomResizedCrop(32),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])])

    
    train_dataset = datasets.CIFAR10(data_dir, train= True,download=True, 
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir,train=False,download=True,
                                  transform=apply_transform)
    
    #sample training
    if args.iid:
        #sample IID user
        user_group,_ = cifar_iid(train_dataset,args.num_users)
    elif args.iid == 0 and args.unequal == 0:
        #sample Non-IID user
        user_group,_ = cifar_noniid(train_dataset,args.num_users)
    elif args.iid == 0 and args.unequal == 1:
        user_group,_ = cifar_noniid_unbalanced(train_dataset, args.num_users)
   
    return train_dataset, test_dataset, user_group
'''

def get_datasets(augmentation=False):
    trainset = datasets.CIFAR10("./data", train=True, download=True)
    testset = datasets.CIFAR10("./data", train=False, download=True)

    cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
    cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std),])
    if augmentation:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    testset.transform = transform

    return trainset, testset


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg



#GKT SECTION

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0
        self.list = []

    def update(self, val):
        self.total += val
        self.steps += 1
        self.list.append(val)

    def value(self):
        return self.total / float(self.steps)

    def get_list(self):
        return (np.array(self.list) / self.steps).tolist()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T**2 * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        return loss


class CE_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)

        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -(self.T**2) * torch.sum(torch.mul(output_batch, teacher_outputs)) / teacher_outputs.size(0)

        return loss