dataset=NICO_Mixed
command=$1
launcher=$2
data_dir=$3

python -m domainbed.sweep_map ${command}\
       --datasets ${dataset}\
       --algorithms IRM VREx ARM GroupDRO MMD CDANN  \
       --data_dir ${data_dir}\
       --command_launcher ${launcher}\
       --fixed_test_envs 2\
       --steps 5001 \
       --holdout_fraction 0.1\
       --n_hparams 20\
       --n_trials 3\
       --hparams "$hparams.json"\
       --output_dir "./results_map/${dataset}/outputs"