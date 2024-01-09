## Run the code for each dataset

- run_baseline.sh is the file for baseline methods.
- run_map.sh is the file for our MAP method. 

## ColoredMNIST

- ColoredMNIST_CNN means that the feature encoder uses the CNN.

- ColoredMNIST_IRM means that the feature encoder uses the MLP following the IRM paper.

- ColoredMNIST_table4 means that the running code for various distribution shifts. More results are shown in Table 4 in our paper.

## Hyperparameters set in run_map.sh
- pretrained means that whether a pretrained model is used.
- nonlinear_classifier means the model architecture.
- used_map is use our map
- lr_alpha is the learning rate for adapter layers.
- lr_beta is the learning rate for per-classifier alignment mapping.
- lr2 is the trade-off for loss.
- map_ad_type is the connection type, *i.e.*, "residual" or "serial".
- map_ad_form is the form, *i.e.*, "matrix" or "vector".  
- map_opt is that the parameters need to be updated, *i.e.*, "alpha" or "alpha+beta".  
- map_init is the initialization, *i.e.*, "eye" or "random".
- adapter_steps is the gradient steps and set to 10.

## HM.py

HM.py is the metric of the harmonic mean.