# Semi-Supervised Video Action Recognition

Aalto university project for course CS-E4875.

[The report](<Semi-Supervised Video Action Recognition.pdf>)

## The pretrained model ViT-B_16.npz:
Downloaded from: https://github.com/google-research/vision_transformer#available-vit-models
```
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```
Pre-trained with ImageNet-21k
The models.py file contains the methods that add the weights to the pytorch model.


## The models

Only model 2, Factorized Encoder was used in the report. 

```
FactorizedEncoder(
  frame_size=224,      # Video frame size
  t=2,                 # Length of the tubelets
  h=16,                # Height of the tubelets
  w=16,                # Width of the tubelets
  hidden=768,          # Hidden size
  c=3,                 # Number of color channels
  frames=32,           # Number of frames in the video
  num_classes=87       # Number of classes in the dataset
  Ls=12,               # Number of layers in the spatial tranformer
  Lt=4                 # Number of layers in the temporal tranformer
  drop_layer_rate=0.3  # Dropout layer rate
)
```
## The supervised folder contains the training script and dataloader for the supervised learning.

### Train the supervised
```
python3 supervised/train.py --model 2 --batch_size 16 --epochs 25 --warm_up 2.5
```
The batch size is the size of the minibatches, the actual batch size is 64.

## The Fixmatch folder contains the training script and dataloader for the SSL learning

### Train the SSL
```
python3 FixMatch/train.py --model 2 --batch_size 1 --epochs 40 --threshold 0.25
```
The batch size is the size of the minibatches, the actual batch size is 64 labeled videos. Minibatch contains of 1 labeled and 2*7 unlabeled videos.

### Optional parameters
`---num_labeled`, Number of labeled videos per class, default is 300

`---mu`, Coefficient of unlabeled batch size, default is 7

`---lambda_u`, Coefficient of unlabeled loss, default is 1


### Optional parameters used in both scripts
`---pretrained`, Default is True, if True uses the pretrained weights

`---train_path`, path to the file containing the training data, default is '../train.txt'

`---eval_path`, path to the file containing the validation data, default is '../eval.txt'

`---load_checkpoint`, Default is False, if True loads model checkpoint

## The data
 Created with code from https://github.com/IBM/action-recognition-pytorch, the following is copied from there

Each line in train.txt, val.txt and test.txt includes 4 elements and separated by a symbol, e.g. space or semicolon. Four elements (in order) include (1)relative paths to video_x_folder from dataset_dir, (2) starting frame number, usually 1, (3) ending frame number, (4) label id (a numeric number).

```
path/to/video_x_folder 1 300 1
```
