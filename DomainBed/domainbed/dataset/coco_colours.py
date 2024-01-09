import os, sys, time, io, requests
import numpy as np
from PIL import Image
sys.path.append('MAP/DomainBed/domainbed/dataset/coco/cocoapi/PythonAPI/')  
from pycocotools.coco import COCO
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import imageio
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ['boat', 'airplane', 'truck', 'dog','zebra', 'horse', 'bird','train','bus','motorcycle']
NUM_CLASSES = len(CLASSES)

output_dir = './datasets/ColoredCOCO'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

env_confounder_strength = [0.8, 0.9, 0.1]

biased_colours = [[0,100,0],[188, 143, 143],[255, 0, 0],[255, 215, 0],[0, 255, 0],[65, 105, 225],[0, 225, 225],[0, 0, 255],[255, 20, 147],[160,160,160]]
biased_colours = np.array(biased_colours)
_D = 2500
def random_different_enough_colour():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biased_colours)**2, 1)) > _D:
            break
    return list(x)
unbiased_colours = np.array([random_different_enough_colour() for _ in range(10)])
def test_colour():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biased_colours)**2, 1)) > _D and np.min(np.sum((x - unbiased_colours)**2, 1)) > _D:
            break
    return x
test_unbiased_colours = np.array([test_colour() for _ in range(10)])

tr1_i = 400*NUM_CLASSES
tr2_i = 400*NUM_CLASSES
te_i = 200*NUM_CLASSES

coco = COCO('coco/annotations/instances_train2017.json')

tr1_s, tr2_s, te_s = 0, 0, 0
for c in range(NUM_CLASSES):
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
        # color coco for env_train1
        env_name = 'env_train1'
        _path = os.path.join(output_dir, env_name, CLASSES[c])
        if not os.path.exists(_path):
            os.makedirs(_path)
        # get the place
        if np.random.random() > env_confounder_strength[0]:
            random_colour = unbiased_colours[np.random.choice(unbiased_colours.shape[0])][None,None,:]
            place_img = 0.75*np.multiply(np.ones((64,64,3),dtype='float32'), random_colour)/255.0
        else:
            place_img = 0.75*np.multiply(np.ones((64,64,3),dtype='float32'), biased_colours[c][None,None,:])/255.0
          
        # that's the one:
        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (64, 64), anti_aliasing=True)

        resized_image = resize(I, (64, 64), anti_aliasing=True)
        resized_place = resize(place_img, (64, 64), anti_aliasing=True)

        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask
        
        image_path = os.path.join(_path, '{}.png'.format(tr1_si))
        imageio.imwrite(image_path, new_im)
            
        tr1_s += 1
        tr1_si += 1
        if tr1_si % 100 == 0:
            print('>'.format(c), end='')
            time.sleep(1)
    print(' ')
    
    # for env_train2
    tr2_si = 0
    print('Class {} (train) : #images = {}'.format(c, len(images)))
    while tr2_si < tr2_i//NUM_CLASSES:
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
    
        env_name = 'env_train2'
        _path = os.path.join(output_dir, env_name, CLASSES[c])
        if not os.path.exists(_path):
            os.makedirs(_path)
        # get the place
        if np.random.random() > env_confounder_strength[1]:
            random_colour = unbiased_colours[np.random.choice(unbiased_colours.shape[0])][None,None,:]
            place_img = 0.75*np.multiply(np.ones((64,64,3),dtype='float32'), random_colour)/255.0
        else:
            place_img = 0.75*np.multiply(np.ones((64,64,3),dtype='float32'), biased_colours[c][None,None,:])/255.0
          
        # that's the one:
        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (64, 64), anti_aliasing=True)

        resized_image = resize(I, (64, 64), anti_aliasing=True)
        resized_place = resize(place_img, (64, 64), anti_aliasing=True)

        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask
        
        image_path = os.path.join(_path, '{}.png'.format(tr2_si))
        imageio.imwrite(image_path, new_im)
            
        tr2_s += 1
        tr2_si += 1
        if tr2_si % 100 == 0:
            print('>'.format(c), end='')
            time.sleep(1)
    print(' ')

    # for env_test
    te_si = 0
    print('Class {} (test) : '.format(c), end=' ')
    while te_si < te_i//NUM_CLASSES:
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

        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (64, 64), anti_aliasing=True)
        resized_image = resize(I, (64, 64), anti_aliasing=True)

        env_name = "env_test"
        _path = os.path.join(output_dir, env_name, CLASSES[c])
        if not os.path.exists(_path):
            os.makedirs(_path)
        # if np.random.random() > env_confounder_strength[2]:
        #     random_colour = unbiased_colours[np.random.choice(unbiased_colours.shape[0])][None,None,:]
        #     place_img = 0.75*np.multiply(np.ones((64,64,3),dtype='float32'), random_colour)/255.0
        # else:
        #     place_img = 0.75*np.multiply(np.ones((64,64,3),dtype='float32'), biased_colours[c][None,None,:])/255.0
        random_colour = test_unbiased_colours[np.random.choice(test_unbiased_colours.shape[0])][None, None, :]
        place_img = 0.75*np.multiply(np.ones((64,64,3), dtype="float32"),random_colour)/255.0

        resized_place = resize(place_img, (64, 64), anti_aliasing=True)
        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask
        image_path = os.path.join(_path, '{}.png'.format(te_si))
        imageio.imwrite(image_path, new_im)
        
        te_s += 1
        te_si += 1
        if te_si % 100 == 0: print('>'.format(c), end='')
    print('')
