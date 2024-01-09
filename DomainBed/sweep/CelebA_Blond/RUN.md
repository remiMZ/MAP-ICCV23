#   Run the code for baseline

## Run this code with single gpu
bash run_baseline.sh launch local ./datasets/

## Run this code with multiple gpus
bash run_baseline.sh launch multi_gpu ./datasets/

## Delete the incomplete results
bash run_baseline.sh delete_incomplete local ./datasets/