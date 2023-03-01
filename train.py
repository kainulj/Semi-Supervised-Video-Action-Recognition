import torch
import numpy as np
import multiprocessing
import torch.nn as nn
from models import SPAModel, FactorizedEncoder
from tqdm import tqdm
from dataloader import VideoDataSet
import time

TRAINING_SET_SIZE = 27


def load_weights(model, path='ViT-B_16.npz'):

    npz = np.load(path)

    model.add_pretrained_weights(npz)



def train(model_num, pretrained, batch_size, epochs, warm_up, train_path, eval_path):
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
    
    workers = 4

    print('Creating the training dataset')
    trainset = VideoDataSet('..', train_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              num_workers=workers, pin_memory=True)
     
    print('Creating the validation dataset')
    evalset = VideoDataSet('..', eval_path)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size,
                                              num_workers=workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    steps_per_epoch = TRAINING_SET_SIZE // 64

    loss_fn = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    # Number of minibatches per batch
    i_per_batch = 64 // batch_size

    lambda1 = lambda step: schedule(step, int(warm_up * steps_per_epoch), int(steps_per_epoch * epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # Training loop
    model.to(device)
    st = time.time()
    print('Starting training')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        with tqdm(total=TRAINING_SET_SIZE) as t_bar:
            for i, (images, target) in enumerate(trainloader):
                images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = loss_fn(output, target) / i_per_batch


                scaler.scale(loss).backward()
                # Update gradients every i_per_batch minibatch
                if (i + 1) % i_per_batch == 0 or (i+1) == len(trainloader):
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                
                epoch_loss += loss
                
                if (i + 1) % (100 * i_per_batch) == 0:
                    t_bar.update(100 * i_per_batch * batch_size)
        et = time.time()
        accuracy, eval_loss = eval(model, evalloader, device)
        print(f'Epoch {epoch + 1}, accuracy: {accuracy}, evaluation loss: {eval_loss}, time: {et - st}')
        st = time.time()
    
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    
    torch.save(state, model_name)


def eval(model, evalloader, device):
    model.eval()
    correct = 0
    total = 0
    eval_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, target in evalloader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss = loss_fn(output, target)
            eval_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total, eval_loss


def schedule(step, warmup, total):
    # linear warmup
    if step < warmup:
        return step / warmup
    step = (step - warmup) / np.min(1, total - warmup)
    # cosine learnig rate
    return 0.5 + 1/2 * (1 + np.cos(step * np.pi))