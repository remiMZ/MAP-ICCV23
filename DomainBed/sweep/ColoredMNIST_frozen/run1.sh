#!/bin/bash

#SBATCH -p a40-tmp
#SBATCH -q gpu-test
#SBATCH -w ga40q14
#SBATCH -c 60
#SBATCH --gres=gpu:4
#SBATCH --mem=800G

dataset=ColoredMNIST
launcher=$1
data_dir=$2


python -m domainbed.sweep_map delete_incomplete\
       --datasets ${dataset}\
       --algorithms VREx  \
       --data_dir ${data_dir}\
       --command_launcher ${launcher}\
       --fixed_test_envs 2\
       --steps 5001 \
       --holdout_fraction 0.1\
       --n_hparams 20\
       --n_trials 3\
       --skip_confirmation \
       --hparams "$(<conv2.json)"\
       --output_dir "/storage/wangdonglinLab/zhangmin/Codes/Tradeoff_iid_ood/DomainBed/results_frozen/${dataset}/outputs_conv2"

python -m domainbed.sweep_map launch\
       --datasets ${dataset}\
       --algorithms VREx  \
       --data_dir ${data_dir}\
       --command_launcher ${launcher}\
       --fixed_test_envs 2\
       --steps 5001 \
       --holdout_fraction 0.1\
       --n_hparams 20\
       --n_trials 3\
       --skip_confirmation \
       --hparams "$(<conv2.json)"\
       --output_dir "/storage/wangdonglinLab/zhangmin/Codes/Tradeoff_iid_ood/DomainBed/results_frozen/${dataset}/outputs_conv2"