'''
The datasets generated process of coco_palces and colored_coco follows the paper " [Systematic generalization with group invariant predictions](https://github.com/Faruk-Ahmed/predictive_group_invariance?tab=readme-ov-file)".

For "coco" you can create the datasets using the coco_places.py and colored_coco.py, which will require installing the [cocoapi](https://github.com/cocodataset/cocoapi) and download the [Places](http://places2.csail.mit.edu/) dataset.
'''

import os, sys, time, io, requests
import numpy as np
import random
from PIL import Image
from pycocotools.coco import COCO
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import imageio
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ['dog','zebra', 'horse', 'bird', 'cow', 'boat', 'airplane', 'truck', 'train', 'bus']
Animal = ['dog','zebra', 'horse', 'bird', 'cow']
Venicle = ['boat', 'airplane', 'truck', 'train', 'bus']
NUM_CLASSES = len(CLASSES) 

output_dir = './datasets/COCOPlaces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

place_data_dir = './datasets/COCOPlaces'
places_dir = os.path.join(place_data_dir, 'data_256')

biased_places = ['d/desert/sand', 'f/forest/broadleaf']

env_confounder_strength = [0.8, 0.9, 0.1]

biased_place_fnames = {}
for i, target_place in enumerate(biased_places):
    L = [f'{target_place}/{filename}' for filename in os.listdir(os.path.join(places_dir, target_place)) if filename.endswith('.jpg')]    
    random.shuffle(L)
    biased_place_fnames[i] = L
    
    
tr1_i = 400*NUM_CLASSES
tr2_i = 400*NUM_CLASSES
te_i = 200*NUM_CLASSES

coco = COCO('coco/annotations/instances_train2017.json')

tr1_s, tr2_s, te_s = 0, 0, 0
for c, class_name in enumerate(CLASSES):
    if c == 5:
        tr1_s = 0
    catIds = coco.getCatIds(catNms=[CLASSES[c]])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    i = -1
    
    # for env_train1
    tr1_si = 0
    print('Class {} (train) : #images = {}'.format(c, len(images)))
    while tr1_si < tr1_i//NUM_CLASSES:
        i += 1
        # get the image
        im = images[i]
        # get the annoatations
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        # pick largest area object
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']
        if max_ann < 10000: continue;
        # load images 
        try: img_data = requests.get(im['coco_url']).content
        except: time.sleep(10); img_data = requests.get(im['coco_url']).content
        I = np.asarray(Image.open(io.BytesIO(img_data)))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])
        # place coco
        env_name = 'env_train1'
        if class_name in Animal:
            label = 'Animal'
            place1 = biased_place_fnames[1]
            place2 = biased_place_fnames[0]
        else:
            label = 'Venicle'
            place1 = biased_place_fnames[0]
            place2 = biased_place_fnames[1]
        
        _path = os.path.join(output_dir, env_name, label)
        if not os.path.exists(_path):
            os.makedirs(_path)
        # get the place
        if np.random.random() > env_confounder_strength[0]:
            place_path = place1[tr1_si]
        else:
            place_path = place2[tr1_si]
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))

        # that's the one:
        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])

        resized_mask = resize(mask, (64, 64))
        resized_image = resize(I, (64, 64))
        resized_place = resize(place_img, (64, 64))

        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask
        
        image_path = os.path.join(_path, '{}.png'.format(tr1_s))
        imageio.imwrite(image_path, new_im)
            
        tr1_s += 1
        tr1_si += 1
        if tr1_si % 100 == 0: print('>'.format(c), end='')
    print('')
    
    # for env_train2
    tr2_si = 0
    j = -1
    if c == 5:
        tr2_s = 0
    print('Class {} (train) : #images = {}'.format(c, len(images)))
    while tr2_si < tr2_i//NUM_CLASSES:
        i += 1
        j +=1
        # get the image
        im = images[i]
        # get the annoatations
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        # pick largest area object
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']
        if max_ann < 10000: continue;
        # load images 
        try: img_data = requests.get(im['coco_url']).content
        except: time.sleep(10); img_data = requests.get(im['coco_url']).content
        I = np.asarray(Image.open(io.BytesIO(img_data)))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])
        # place coco
        env_name = 'env_train2'
        if class_name in Animal:
            label = 'Animal'
            place1 = biased_place_fnames[1]
            place2 = biased_place_fnames[0]
        else:
            label = 'Venicle'
            place1 = biased_place_fnames[0]
            place2 = biased_place_fnames[1]
        
        _path = os.path.join(output_dir, env_name, label)
        if not os.path.exists(_path):
            os.makedirs(_path)
        # get the place
        if np.random.random() > env_confounder_strength[1]:
            place_path = place1[j+int(env_confounder_strength[0]*tr1_si)]
        else:
            place_path = place2[j+int(env_confounder_strength[0]*tr1_si)]
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))

        # that's the one:
        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])

        resized_mask = resize(mask, (64, 64))
        resized_image = resize(I, (64, 64))
        resized_place = resize(place_img, (64, 64))

        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask
        
        image_path = os.path.join(_path, '{}.png'.format(tr2_s))
        imageio.imwrite(image_path, new_im)
            
        tr2_s += 1
        tr2_si += 1
        if tr2_si % 100 == 0: print('>'.format(c), end='')
    print('')

    # for env_test
    te_si = 0
    j = -1
    if c == 5:
        te_s = 0
    print('Class {} (test) : '.format(c), end=' ')
    while te_si < te_i//NUM_CLASSES:
        i += 1
        j += 1
        # get the image
        im = images[i]
        # get the annoatations
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        # pick largest area object
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']
        if max_ann < 10000: continue;
        # load images 
        try: img_data = requests.get(im['coco_url']).content
        except: time.sleep(10); img_data = requests.get(im['coco_url']).content
        I = np.asarray(Image.open(io.BytesIO(img_data)))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])
        env_name = 'env_test'
        if class_name in Animal:
            label = 'Animal'
            place1 = biased_place_fnames[1]
            place2 = biased_place_fnames[0]
        else:
            label = 'Venicle'
            place1 = biased_place_fnames[0]
            place2 = biased_place_fnames[1]
        
        _path = os.path.join(output_dir, env_name, label)
        if not os.path.exists(_path):
            os.makedirs(_path)
        
        if np.random.random() > env_confounder_strength[2]:
            place_path = place1[te_si+int(env_confounder_strength[0]*(tr1_si))+int(env_confounder_strength[1]*(tr2_si))]
        else:
            place_path = place2[te_si+int(env_confounder_strength[0]*(tr1_si))+int(env_confounder_strength[1]*(tr2_si))]
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))

        # that's the one:
        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        
        resized_mask = resize(mask, (64, 64))
        resized_image = resize(I, (64, 64))
        resized_place = resize(place_img, (64, 64))

        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask
        
        image_path = os.path.join(_path, '{}.png'.format(te_s))
        imageio.imwrite(image_path, new_im)
       
        te_s += 1
        te_si += 1
        if te_si % 100 == 0: print('>'.format(c), end='')
    print('')