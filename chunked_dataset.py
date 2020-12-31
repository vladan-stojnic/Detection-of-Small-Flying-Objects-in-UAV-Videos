from skvideo.io import vread
from skvideo.utils import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.util import img_as_float
import os
import h5py
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt
from util import get_parser

args = get_parser().parse_args()

TYPE = args.type

DATASET_PATH = args.output
SIZE = 256
CHUNK_SIZE = 64

NUM_OF_CHANNELS = 5

if TYPE == 'val' or TYPE == 'test':
    MAX_CLASS = 3200
else:
    MAX_CLASS = 13440

BGSUB_PATH = args.bgsub
MASKS_PATH = args.mask


NUM_OF_CHUNKS = MAX_CLASS*4/CHUNK_SIZE
PROB0 = 0.03
PROB1 = 0.08
PROB2 = 0.25
PROB3 = 0.9

count0 = 0
count1 = 0
count2 = 0
count3 = 0

data_np = np.zeros((CHUNK_SIZE, SIZE, SIZE, NUM_OF_CHANNELS), dtype='float32')
label_np = np.zeros((CHUNK_SIZE, 1), dtype = 'u8')
mask_np = np.zeros((CHUNK_SIZE, SIZE//2, SIZE//2), dtype = 'float32')

hf = h5py.File(DATASET_PATH+'.h5', 'w')
data_dset = hf.create_dataset("data", (NUM_OF_CHUNKS*CHUNK_SIZE, SIZE, SIZE, NUM_OF_CHANNELS), chunks = (CHUNK_SIZE, SIZE, SIZE, NUM_OF_CHANNELS), maxshape=(None, SIZE, SIZE, NUM_OF_CHANNELS), compression="gzip", dtype = 'f', compression_opts=4)
label_dset = hf.create_dataset("label", (NUM_OF_CHUNKS*CHUNK_SIZE, 1), chunks = (CHUNK_SIZE, 1), maxshape=(None, 1), compression="gzip", dtype = 'u8', compression_opts=4)
mask_dset = hf.create_dataset("mask", (NUM_OF_CHUNKS*CHUNK_SIZE, SIZE//2, SIZE//2), chunks = (CHUNK_SIZE, SIZE//2, SIZE//2), maxshape=(None, SIZE//2, SIZE//2), compression="gzip", dtype = 'f', compression_opts=4)

def should_write(objects):
    r = np.random.rand()
    
    if objects == 0:
        if (r < PROB0) and (count0 < MAX_CLASS):
            return True
    elif objects == 1:
        if (r < PROB1) and (count1 < MAX_CLASS):
            return True
    elif objects == 2:
        if (r < PROB2) and (count2 < MAX_CLASS):
            return True
    elif objects == 3:
        if (r < PROB3) and (count3 < MAX_CLASS):
            return True
    
    return False
    
def written(objects):
    global count0
    global count1
    global count2
    global count3
    
    if objects == 0:
        count0 += 1
    elif objects == 1:
        count1 += 1
    elif objects == 2:
        count2 += 1
    elif objects == 3:
        count3 += 1
    
def count_objects(image):
    image = img_as_float(image)
    bw = closing(image > 0.3, square(3))
    label_image = label(bw)
    props = regionprops(label_image)
    
    return len(props)

def extract_data(bgsub_path, mask_path, counter):
    bgsub = rgb2gray(vread(bgsub_path))
    mask = rgb2gray(vread(mask_path))
    num_frames, height, width, depth = bgsub.shape
    
    internal_counter = counter
    
    for i in range(2, num_frames - 2):
        for j in range(height//SIZE):
            for k in range(width//SIZE):
                mask_data = rescale(np.uint8(mask[i, j*SIZE:(j+1)*SIZE, k*SIZE:(k+1)*SIZE, 0]), 0.5, anti_aliasing = False)
                objects = count_objects(mask_data)
                
                write = should_write(objects)
                
                if write:
                    data_np[internal_counter%CHUNK_SIZE, :, :, :] = img_as_float(np.uint8(bgsub[i-2:i+3, j*SIZE:(j+1)*SIZE, k*SIZE:(k+1)*SIZE, 0].transpose((1, 2, 0))))
                    label_np[internal_counter%CHUNK_SIZE, :] = objects
                    mask_np[internal_counter%CHUNK_SIZE, :, :] = mask_data
                    written(objects)
                
                if ((internal_counter+1)%CHUNK_SIZE)==0:
                    data_dset[(internal_counter//CHUNK_SIZE)*CHUNK_SIZE:((internal_counter+1)//CHUNK_SIZE)*CHUNK_SIZE, :, :, :] = data_np
                    label_dset[(internal_counter//CHUNK_SIZE)*CHUNK_SIZE:((internal_counter+1)//CHUNK_SIZE)*CHUNK_SIZE, :] = label_np
                    mask_dset[(internal_counter//CHUNK_SIZE)*CHUNK_SIZE:((internal_counter+1)//CHUNK_SIZE)*CHUNK_SIZE, :, :] = mask_np
         
                if write:
                    internal_counter += 1
                    if count0 >= MAX_CLASS and count1 >= MAX_CLASS and count2 >= MAX_CLASS and count3 >= MAX_CLASS:
                        return False, internal_counter

    return True, internal_counter

rng = np.random.RandomState(42)
vids = np.arange(1, 1001)

vids = rng.permutation(vids)

i = 0
cnt = 0

running = True

if TYPE == 'val':
    div = 250
    offset = 500
elif TYPE == 'test':
    div = 250
    offset = 750
else:
    div = 500
    offset = 0

while running:
    vid = vids[(i%div)+offset]
    running, cnt = extract_data(BGSUB_PATH+'/Out_'+str(vid)+'.MP4',
             MASKS_PATH+'/Mask_' + str(vid)+'.MP4', cnt)
    print(cnt)
    i += 1
    
hf.attrs['elems'] = cnt
             
hf.close()