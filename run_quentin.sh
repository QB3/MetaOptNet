#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --exclude=cn-e001,cn-e002,cn-e003

# SBATCH --output=tmp/slurm/slurm-%j.out

#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt

# source ~/venv38bis/bin/activate

module load anaconda/3
conda activate torchopt

python train.py --gpu 0 --save-path "/home/mila/q/quentin.bertrand/MetaOptNet/" --train-shot 15 --head Sparse-SVM --network ProtoNet --dataset miniImageNet --eps 0.1
