#!/bin/bash

#SBATCH --time=40:00:00
#SBATCH --mem=28000
#SBATCH --gres=gpu:v100:1                    
#SBATCH --cpus-per-task=4            # time and memory requirements
mkdir /tmp/$SLURM_JOB_ID                          # get a directory where you will put your data
cp $WRKDIR/train.tar /tmp/$SLURM_JOB_ID           # copy tarred input files
cp $WRKDIR/validation.tar /tmp/$SLURM_JOB_ID           # copy tarred input files
cp $WRKDIR/train.txt /tmp/$SLURM_JOB_ID           # copy tarred input files
cp $WRKDIR/val.txt /tmp/$SLURM_JOB_ID           # copy tarred input files
cp -R $WRKDIR/supervised /tmp/$SLURM_JOB_ID
cp $WRKDIR/models.py tmp/$SLURM_JOB_ID/supervised  # Copy the models file to the supervised folder
cp $WRKDIR/ViT-B_16.npz tmp/$SLURM_JOB_ID/supervised # Copy the pretrained weights to the supervised folder
cd /tmp/$SLURM_JOB_ID

trap "rm -rf /tmp/$SLURM_JOB_ID; exit" TERM EXIT  # set the trap: when killed or exits abnormally you clean up your stuff
module load anaconda
echo "tar training set"
tar xf train.tar                                  # untar the files
echo "tar validation set"
tar xf validation.tar                                  # untar the files
echo "run the python script"
cd supervised 
srun  python3 train.py --model 2 --batch_size 16 --epochs 25 --warm_up 2.5   # train the model
mv  'FactorizedEncoder.pth' $WRKDIR
