import torch
import numpy as np
import multiprocessing
import torch.nn as nn
from models import SPAModel, FactorizedEncoder
from dataloader import FixMatchDataSet, split_l_u, create_video_list
import time
import torch.nn.functional as F
import argparse

TRAINING_SET_SIZE = 81663
NUM_CLASSES = 87


def load_weights(model, path='ViT-B_16.npz'):

    npz = np.load(path)

    model.add_pretrained_weights(npz)


def train(model_num, pretrained, batch_size, num_labeled, mu, lambda_u, threshold, epochs, warm_up, train_path, eval_path, load_checkpoint=False):
    print('Creating the model')
    if model_num == 1:
        model = SPAModel()
        model_name = 'SPAModel.pth'
    else:
        model = FactorizedEncoder()
        model_name = 'FactorizedEncoder_Fixmatch.pth'

    if pretrained:
        print('Loading the weights')
        load_weights(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    workers = 6

    labeled_data, unlabeled_data = split_l_u(train_path, num_labeled, NUM_CLASSES)

    print('Creating the training datasets')
    labeledset = FixMatchDataSet('..', labeled_data, labeled=True)
    labeledloader = torch.utils.data.DataLoader(labeledset, batch_size=batch_size, shuffle=True,
                                              num_workers=2, pin_memory=True)

    unlabeledset = FixMatchDataSet('..', unlabeled_data, labeled=False)
    unlabeledloader = torch.utils.data.DataLoader(unlabeledset, batch_size=batch_size*mu, shuffle=True,
                                              num_workers=4, pin_memory=True)
     
    print('Creating the validation dataset')
    eval_data = create_video_list(eval_path)
    evalset = FixMatchDataSet('..', eval_data, labeled=True, is_train=False)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size,
                                              num_workers=workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005, nesterov=True)

    steps_per_epoch = len(unlabeledloader) // (64 * mu)

    scaler = torch.cuda.amp.GradScaler()

    # Number of minibatches per batch
    i_per_batch = 64 // batch_size

    start_epoch = 0
    if load_checkpoint:
        # Load checkpoint
        print("Loading model checkpoint")
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    # Training loop
    st = time.time()
    print('Starting training')
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        supervised_loss = 0.0
        unsupervised_loss = 0.0
        mask_rate = 0.0
        labeled = iter(labeledloader)
        for i, (weak, strong) in enumerate(unlabeledloader):
            weak, strong = weak.to(device, non_blocking=True), strong.to(device, non_blocking=True)
            try:
                labeled_images, targets = next(labeled)
            except:
                labeled = iter(labeledloader)
                labeled_images, targets = next(labeled)
            
            labeled_images, targets = labeled_images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                labeled_output = model(labeled_images)
                weak_output = model(weak)

                # Create the pseudo-labels, and create the mask
                pseudo_label = torch.softmax(weak_output, dim=-1)
                max_probs, pseudo_target = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(threshold).float()
                del max_probs, pseudo_label

                strong_output = model(strong)

                # Supervised loss
                ls = F.cross_entropy(labeled_output, targets, reduction='mean', label_smoothing=0.3)
                # Unsupervised loss
                lu = (F.cross_entropy(strong_output, pseudo_target, reduction='none', label_smoothing=0.3) * mask).mean()
                loss = (ls + lambda_u * lu) / i_per_batch 


            scaler.scale(loss).backward()
            # Update gradients every i_per_batch minibatch
            if (i + 1) % i_per_batch == 0 or (i+1) == len(unlabeledloader):
                # Unscale the gradients before clipping
                scaler.unscale_(optimizer)
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item()
            supervised_loss += ls.item()
            unsupervised_loss += lu.item()
            mask_rate += mask.sum().item() / (i_per_batch * mu)

        et = time.time()
        accuracy, eval_loss = eval(model, evalloader, device)
        print(f'Epoch {epoch + 1}:')
        print(f'loss: {epoch_loss / steps_per_epoch}, Ls: {supervised_loss / steps_per_epoch}, Lu: {unsupervised_loss / steps_per_epoch}, evaluation loss: {eval_loss}')
        print(f'mask rate: {mask_rate / steps_per_epoch}, accuracy: {accuracy}, time: {et - st}')
        st = time.time()
    
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
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
    curr_step = step - warmup
    max_step = max(1, total - warmup) 
    # cosine learnig rate
    return 0.5 * (1 + np.cos(curr_step / max_step * np.pi))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=int, required=True, help="The model number")
    parser.add_argument("--pretrained", type=bool, default=True, help="Is the model initialized with the pretrained model")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--num_labeled", type=int, default=300, help="Number of labeled videos pre class")
    parser.add_argument("--mu", type=int, default=7, help="Coefficient of unlabeled batch size")
    parser.add_argument("--lambda_u", type=int, default=1, help="Coefficient of unlabeled loss")
    parser.add_argument("--threshold", type=float, required=True, help="Confidence threshold")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--warm_up", type=float, required=True, help="Number of warm up epochs")
    parser.add_argument("--train_path", type=str, default='../train.txt', help="Path to the train data file")
    parser.add_argument("--eval_path", type=str, default='../eval.txt', help="Path to the validation data file")
    parser.add_argument("--load_checkpoint", type=bool, default=False, help="Continue from a checkpoint")
    

    args = parser.parse_args()

    train(args.model, args.pretrained, args.batch_size, args.num_labeled*NUM_CLASSES, args.mu, args.lambda_u, args.threshold, args.epochs, args.warm_up, args.train_path, args.eval_path, args.load_checkpoint)
