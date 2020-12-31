import os.path
import os
import numpy as np
import skvideo.io
from skimage.util import img_as_float, img_as_ubyte
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB
from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.measure import ransac
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

inPath = '../Video/Stabilized/' # Path to backgrounds of stabilized videos

outPath = '../Video/Cropped_1024x1024/' # Path where to write cropped videos

numVid = 0

for k, vid in enumerate(os.listdir(inPath)) :
    print(vid)
        
    video = skvideo.io.vread(inPath + vid)

    for i in np.arange(200, np.size(video, 1) - 1024 - 200, 200) :
        for j in np.arange(200, np.size(video, 2) - 1024 - 200, 150) :
            skvideo.io.vwrite(outPath + 'Cropped_' + str(numVid + 1) + '.MP4', 
                              video[:, i : i + 1024, j : j + 1024, :],
                              inputdict={'-r': '25/1'}, 
                              outputdict={'-r': '25/1',
                                          '-pix_fmt': 'yuv420p'})
            numVid += 1
