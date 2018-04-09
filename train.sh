mkdir ./nmt_small_model
python -m nmt.nmt \
    --src=en --tgt=zh \
    --vocab_prefix=./data/vocab  \
    --train_prefix=./data/train_5k \
    --dev_prefix=./data/dev  \
    --test_prefix=./data/test \
    --out_dir=./nmt_small_model \
    --num_train_steps=10000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

''' attention on GPU
#!/bin/bash
#
#SBATCH --job-name=myNMTGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=80:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
##SBATCH --mail-user=yw2848@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load tensorflow/python3.6/1.5.0

cd /scratch/yw2848/nmt
mkdir ./nmt_attention_model_scaled_luong_50w

python -m nmt.nmt \
    --attention=scaled_luong \
    --src=en --tgt=zh \
    --vocab_prefix=./data/vocab  \
    --train_prefix=./data/train \
    --dev_prefix=./data/dev  \
    --test_prefix=./data/test \
    --out_dir=./nmt_attention_model_scaled_luong_50w \
    --num_train_steps=500000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu \
    --num_gpus=1  \

'''


'''  on GPU
#!/bin/bash
#
#SBATCH --job-name=myNMTGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=80:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
##SBATCH --mail-user=yw2848@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load tensorflow/python3.6/1.5.0

cd /scratch/yw2848/nmt
mkdir ./nmt_model
python -m nmt.nmt \
    --src=en --tgt=zh \
    --vocab_prefix=./data/vocab  \
    --train_prefix=./data/train \
    --dev_prefix=./data/dev  \
    --test_prefix=./data/test \
    --out_dir=./nmt_model \
    --num_train_steps=200000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu \
    --num_gpus=1
'''
