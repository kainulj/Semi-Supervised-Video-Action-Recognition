import torch
import multiprocessing
from models import SPAModel, FactorizedEncoder
from dataloader import VideoDataSet
import argparse
from tqdm import tqdm

def test(model_num, modelpath, datapath, batch_size):

    if model_num == 1:
        model = SPAModel()
    else:
        model = FactorizedEncoder()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['state_dict'])

    dataset = VideoDataSet('..', datapath, is_train=False)
    dataloader = torch.utils.data.DataLoader(dataset , batch_size=batch_size,
                                              num_workers=4, pin_memory=True)

    loop = tqdm(dataloader)

    model.to(device)
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for step, (images, target) in enumerate(loop):
            images, target = images.to(device), target.to(device)
            output = model(images)
            _, predicted = output.data.topk(5, dim=1)
            predicted = predicted.t()
            total += target.size(0)
            target = target.view(1, -1).expand_as(predicted)
            correct_top1 += predicted.eq(target)[:1].sum()
            correct_top5 += predicted.eq(target)[:5].sum()
    print(f'Test accuracies: top1:{correct_top1 / total}, top5:{correct_top5 / total}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=int, required=True, help="The model number")
    parser.add_argument("--modelpath", type=str, help="Path to the model file")
    parser.add_argument("--data_path", type=str,help="Path to the data file")
    parser.add_argument("--batch_size", type=int, help="Batch size")

    args = parser.parse_args()

    test(args.model, args.modelpath, args.data_path, args.batch_size)
