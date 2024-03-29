# Instructions for Preparing the Datasets

## Downloading the Datasets

The download links for the datasets are as follows:

- The MNIST dataset can be downloaded by running the script `MAP/DomainBed/domainbed/scripts/download.py`.  
- The ColoredCOCO dataset can be generated by running the script `MAP/DomainBed/domainbed/scripts/colored_coco.py`. 
- The COCOCPlaces dataset can be generated by running the script `MAP/DomainBed/domainbed/scripts/colored_coco_places.py`.
- The NICO dataset can be downloaded from [here](https://nico.thumedialab.com/).
- The CelebA dataset can be downloaded by running the script `MAP/Wilds/wilds/datasets/celebA_dataset.py`.
- The WILDSCamelyon dataset can be downloaded by runing the script `MAP/Wilds/wilds/datasets/camelyon17_dataset.py`.

Place them under `datasets` and make ssure the directory structure are as follows: 
 
## Directory Structure

Make sure that the directory structure of each dataset is arranged as follows:

**MNIST (ColoredMNIST)**
```
MNIST
└── processed
    ├── training.pt
    └── test.pt
```

**COCO (ColoredCOCO)**
```
ColoredCOCO
├── env_train1
├── env_train2
├── env_test
```

**COCOPlaces**
```
COCOPlaces
├── data_256
├── env_train1
├── env_train2
├── env_test
```

**NICO**
```
NICO
├── animal
├── vehicle
└── mixed_split_corrected
    ├── env_train1.csv
    ├── env_train2.csv
    ├── env_val.csv
    └── env_test.csv
```

**CelebA**
```
celeba
├── img_align_celeba
├── list_eval_partition.csv
├── list_attr_celeba.csv
└── blond_split
    ├── tr_env1_df.csv
    ├── tr_env2_df.csv
    └── te_env_df.csv
```

