import torch
import matplotlib.pyplot as plt

import Gradient_Attack
from FedAvg.models import ResNet50
from torchvision.models import resnet50
from Centralized_Baseline.utils import datasets

def plot(tensor, filename=None):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")

num_images = 1
local_lr = 1e-2
local_steps = 5
use_updates = True

setup = Gradient_Attack.utils.system_startup()
defs = Gradient_Attack.training_strategy('conservative')

loss_fn, trainloader, validloader =  Gradient_Attack.construct_dataloaders('CIFAR10', defs)

#checkpoint = torch.load("/kaggle/input/fedavg-100pt/fedavg_100.pt")  # not available on github due to size restrictions
model = resnet50(pretrained = True).to(**setup)
#model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dm = torch.as_tensor(datasets.cifar10_mean, **setup)[:, None, None]
ds = torch.as_tensor(datasets.cifar10_std, **setup)[:, None, None]

ground_truth, labels = [], []
idx = 0  # ?
while len(labels) < num_images:
    img, label = validloader.dataset[idx]
    idx += 1
    if label not in labels:
        labels.append(torch.as_tensor((label,), device=setup['device']))
        ground_truth.append(img.to(**setup))
ground_truth = torch.stack(ground_truth)
labels = torch.cat(labels)

print(plot(ground_truth))
print([validloader.dataset.classes[l] for l in labels])


model.zero_grad()
target_loss, _, _ = loss_fn(model(ground_truth), labels)
input_parameters = Gradient_Attack.reconstruction_algorithms.loss_steps(model, ground_truth, labels, 
                                                        lr=local_lr, local_steps=local_steps,
                                                                   use_updates=use_updates)
input_parameters = [p.detach() for p in input_parameters]

config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.01,
              optim='sgd',
              restarts=1,
              max_iterations=2_000,
              total_variation=1e-6,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

rec_machine = Gradient_Attack.FedAvgReconstructor(model, (dm, ds), local_steps, local_lr, config,
                                             use_updates=use_updates)
output, _ = rec_machine.reconstruct(input_parameters, labels, img_shape=(3, 32, 32))

print(plot(output))
