import torch
import numpy as np
import multiprocessing
import torch.nn as nn
from models import SPAModel, FactorizedEncoder
from tqdm import tqdm
from dataloader import VideoDataSet
import time

TRAINING_SET_SIZE = 64


def load_weights(model, path='ViT-B_16.npz'):

    npz = np.load(path)

    model.add_pretrained_weights(npz)



def train(model_num, pretrained=True, batch_size=64, epochs=35, warm_up=2.5):
    print('Creating the model')
    if model_num == 1:
        model = SPAModel()
        model_name = 'SPAModel.pth'
    else:
        model = FactorizedEncoder()
        model_name = 'FactorizedEncoder.pth'

    print('Loading the weights')
    if pretrained:
        load_weights(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    
    workers = 4

    print('Creating the training dataset')
    trainset = VideoDataSet('..', '../data/train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              num_workers=24, pin_memory=True)
     
    print('Creating the validation dataset')
    evalset = VideoDataSet('..', '../data/val.txt')
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size,
                                              num_workers=workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    steps_per_epoch = TRAINING_SET_SIZE // batch_size

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(warm_up * steps_per_epoch))

    loss_fn = nn.CrossEntropyLoss()


    # Training loop
    model.to(device)
    model.train()
    st = time.time()
    print('Starting training')
    for epoch in range(epochs):
        epoch_loss = []
        with tqdm(total=len(trainloader)) as t_bar:
            for i, (images, target) in enumerate(trainloader):
                images, target = images.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss.append(loss.item())
                t_bar.update(1)
                del images, target, loss, output

        et = time.time()
        print(f'Epoch {epoch + 1} loss: {np.mean(epoch_loss)}, accuracy: {eval(model, evalloader, device)}, time: {et - st}')
        st = time.time()

    
    
    torch.save(model.state_dict(), model_name)


def eval(model, evalloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, target in evalloader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            del images, target, predicted, output
    return correct / total