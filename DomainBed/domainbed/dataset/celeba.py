import os
import csv
import pandas as pd
import numpy as np

root_dir = './datasets/celeba/'

# Read in train/val/test splits
split_df = pd.read_csv(
    os.path.join(root_dir, 'list_eval_partition.csv'))
split_array = split_df['partition'].values
split_dict = {
    'train': 0,
    'val': 1,
    'test': 2
}

train_image_id = {}
for i_ter, value in enumerate(split_array):
    if value == split_dict['train']:
        image_id = split_df['image_id'][i_ter]
        train_image_id[i_ter]= image_id
        
test_image_id = {}
for i_ter, value in enumerate(split_array):
    if value == split_dict['test'] or value == split_dict['val']:
        image_id = split_df['image_id'][i_ter]
        test_image_id[i_ter]= image_id
        
# Read in attributes
attrs_df = pd.read_csv(
    os.path.join(root_dir, 'list_attr_celeba.csv'))
# Split out filenames and attribute names
data_dir = os.path.join(root_dir, 'img_align_celeba')
filename_array = attrs_df['image_id'].values
filename_array = np.array([os.path.join(data_dir, x) for x in filename_array])
attrs_df = attrs_df.drop(labels='image_id', axis='columns')
attr_names = attrs_df.columns.copy()

# Then cast attributes to numpy array and set them to 0 and 1 (originally, they're -1 and 1)
attrs_df = attrs_df.values
attrs_df[attrs_df == -1] = 0

attr_id_blond = 9
attr_id_male = 20
# for testing env
num_blond_male_te = 362
# 120 (5354)
num_blond_female_te = 120
# 362 (15811)
num_not_blond_male_te = 362
# 120 (18302)
num_not_blond_female_te = 120

# for testing env
tes_file = []

b_m=0
b_f=0
nb_m=0
nb_f=0 
for k, v in test_image_id.items():
    if attrs_df[k][attr_id_blond] == 1 and attrs_df[k][attr_id_male]== 1:
        b_m+=1
        if b_m <= num_blond_male_te:
            print("i1", b_m)
            tes_file.append({'': v, '0': attrs_df[k][0], '1':attrs_df[k][1], '2':attrs_df[k][2], '3':attrs_df[k][3], '4':attrs_df[k][4], '5':attrs_df[k][5], '6':attrs_df[k][6], '7':attrs_df[k][7], '8':attrs_df[k][8], '9':attrs_df[k][9], '10':attrs_df[k][10], '11':attrs_df[k][11], '12':attrs_df[k][12], '13':attrs_df[k][13], '14':attrs_df[k][14], '15':attrs_df[k][15], '16':attrs_df[k][16], '17':attrs_df[k][17], '18':attrs_df[k][18], '19':attrs_df[k][19], '20':attrs_df[k][20], '21':attrs_df[k][21], '22':attrs_df[k][22], '23':attrs_df[k][23], '24':attrs_df[k][24], '25':attrs_df[k][25], '26':attrs_df[k][26], '27':attrs_df[k][27], '28':attrs_df[k][28], '29':attrs_df[k][29], '30':attrs_df[k][30], '31':attrs_df[k][31], '32':attrs_df[k][32], '33':attrs_df[k][33], '34':attrs_df[k][34], '35':attrs_df[k][35], '36':attrs_df[k][36], '37':attrs_df[k][37], '38':attrs_df[k][38], '39':attrs_df[k][39]})
        
    if attrs_df[k][attr_id_blond] == 1 and attrs_df[k][attr_id_male]== 0:
        b_f+=1
        if b_f <= num_blond_female_te:
            print("i2", b_f)
            tes_file.append({'': v, '0': attrs_df[k][0], '1':attrs_df[k][1], '2':attrs_df[k][2], '3':attrs_df[k][3], '4':attrs_df[k][4], '5':attrs_df[k][5], '6':attrs_df[k][6], '7':attrs_df[k][7], '8':attrs_df[k][8], '9':attrs_df[k][9], '10':attrs_df[k][10], '11':attrs_df[k][11], '12':attrs_df[k][12], '13':attrs_df[k][13], '14':attrs_df[k][14], '15':attrs_df[k][15], '16':attrs_df[k][16], '17':attrs_df[k][17], '18':attrs_df[k][18], '19':attrs_df[k][19], '20':attrs_df[k][20], '21':attrs_df[k][21], '22':attrs_df[k][22], '23':attrs_df[k][23], '24':attrs_df[k][24], '25':attrs_df[k][25], '26':attrs_df[k][26], '27':attrs_df[k][27], '28':attrs_df[k][28], '29':attrs_df[k][29], '30':attrs_df[k][30], '31':attrs_df[k][31], '32':attrs_df[k][32], '33':attrs_df[k][33], '34':attrs_df[k][34], '35':attrs_df[k][35], '36':attrs_df[k][36], '37':attrs_df[k][37], '38':attrs_df[k][38], '39':attrs_df[k][39]})
    
    if attrs_df[k][attr_id_blond] == 0 and attrs_df[k][attr_id_male]== 1:
        nb_m +=1
        if nb_m <= num_not_blond_male_te:
            print("i3", nb_m)
            tes_file.append({'': v, '0': attrs_df[k][0], '1':attrs_df[k][1], '2':attrs_df[k][2], '3':attrs_df[k][3], '4':attrs_df[k][4], '5':attrs_df[k][5], '6':attrs_df[k][6], '7':attrs_df[k][7], '8':attrs_df[k][8], '9':attrs_df[k][9], '10':attrs_df[k][10], '11':attrs_df[k][11], '12':attrs_df[k][12], '13':attrs_df[k][13], '14':attrs_df[k][14], '15':attrs_df[k][15], '16':attrs_df[k][16], '17':attrs_df[k][17], '18':attrs_df[k][18], '19':attrs_df[k][19], '20':attrs_df[k][20], '21':attrs_df[k][21], '22':attrs_df[k][22], '23':attrs_df[k][23], '24':attrs_df[k][24], '25':attrs_df[k][25], '26':attrs_df[k][26], '27':attrs_df[k][27], '28':attrs_df[k][28], '29':attrs_df[k][29], '30':attrs_df[k][30], '31':attrs_df[k][31], '32':attrs_df[k][32], '33':attrs_df[k][33], '34':attrs_df[k][34], '35':attrs_df[k][35], '36':attrs_df[k][36], '37':attrs_df[k][37], '38':attrs_df[k][38], '39':attrs_df[k][39]})
    
    if attrs_df[k][attr_id_blond] == 0 and attrs_df[k][attr_id_male]== 0:
        nb_f+=1
        if nb_f <= num_not_blond_female_te:
            print("i4", nb_f)
            tes_file.append({'': v, '0': attrs_df[k][0], '1':attrs_df[k][1], '2':attrs_df[k][2], '3':attrs_df[k][3], '4':attrs_df[k][4], '5':attrs_df[k][5], '6':attrs_df[k][6], '7':attrs_df[k][7], '8':attrs_df[k][8], '9':attrs_df[k][9], '10':attrs_df[k][10], '11':attrs_df[k][11], '12':attrs_df[k][12], '13':attrs_df[k][13], '14':attrs_df[k][14], '15':attrs_df[k][15], '16':attrs_df[k][16], '17':attrs_df[k][17], '18':attrs_df[k][18], '19':attrs_df[k][19], '20':attrs_df[k][20], '21':attrs_df[k][21], '22':attrs_df[k][22], '23':attrs_df[k][23], '24':attrs_df[k][24], '25':attrs_df[k][25], '26':attrs_df[k][26], '27':attrs_df[k][27], '28':attrs_df[k][28], '29':attrs_df[k][29], '30':attrs_df[k][30], '31':attrs_df[k][31], '32':attrs_df[k][32], '33':attrs_df[k][33], '34':attrs_df[k][34], '35':attrs_df[k][35], '36':attrs_df[k][36], '37':attrs_df[k][37], '38':attrs_df[k][38], '39':attrs_df[k][39]})

with open(os.path.join(root_dir, 'te_env.csv'), 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'])
    for file in tes_file:
        f_csv.writerow([file[''], file['0'], file['1'], file['2'], file['3'], file['4'], file['5'], file['6'], file['7'], file['8'], file['9'], file['10'], file['11'], file['12'], file['13'], file['14'], file['15'], file['16'], file['17'], file['18'], file['19'], file['20'], file['21'], file['22'], file['23'], file['24'], file['25'], file['26'], file['27'], file['28'], file['29'], file['30'], file['31'], file['32'], file['33'], file['34'], file['35'], file['36'], file['37'], file['38'], file['39']]) 
    
    
    



