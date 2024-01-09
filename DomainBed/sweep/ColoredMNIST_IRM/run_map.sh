dataset=ColoredMNIST_IRM
command=$1
launcher=$2
data_dir=$3

python -m domainbed.sweep_map ${command}\
       --datasets ${dataset}\
       --algorithms IRM_MAP VREx_MAP ARM_MAP GroupDRO_MAP CDANN_MAP TRM_MAP IB_ERM_MAP IB_IRM_MAP \
       --data_dir ${data_dir}\
       --command_launcher ${launcher}\
       --fixed_test_envs 2\
       --steps 5001 \
       --holdout_fraction 0.1\
       --n_hparams 20\
       --n_trials 3\
       --hparams "$(<hparams_map.json)"\
       --output_dir "./results_map/${dataset}/outputs_IRM"