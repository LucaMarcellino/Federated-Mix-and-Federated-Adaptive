import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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

for idx in tqdm([7,15,47]):
    num_images = 1
    local_lr = 1e-2
    local_steps = 5
    use_updates = True
    cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
    cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]

    setup = Gradient_Attack.utils.system_startup()
    defs = Gradient_Attack.training_strategy('conservative')

    loss_fn, trainloader, validloader =  Gradient_Attack.construct_dataloaders('CIFAR10', defs)

    model = resnet50(pretrained = True).to(**setup)
    model.eval()
    dm = torch.as_tensor(cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(cifar10_std, **setup)[:, None, None]
    ground_truth, labels = [], []
    while len(labels) < num_images:
        img, label = validloader.dataset[idx]
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))
    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)

    plot(ground_truth, filename=f"gt_{idx}.png")
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
                max_iterations=8_000,
                total_variation=1e-6,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')

    rec_machine = Gradient_Attack.FedAvgReconstructor(model, (dm, ds), local_steps, local_lr, config,
                                                use_updates=use_updates, num_images=num_images)
    output, _ = rec_machine.reconstruct(input_parameters, labels, img_shape=(3, 32, 32))

    plot(output,filename=f"output_{idx}.png")

