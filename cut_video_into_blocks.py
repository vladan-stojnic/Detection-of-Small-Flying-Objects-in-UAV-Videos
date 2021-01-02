""" Crops videos into 1024x1024 patches.

CL Args:
  -i Path to directory with input videos.
  -o Path to directory with output videos.
"""

import os
import numpy as np
import skvideo.io
from util import get_parser
import warnings
warnings.filterwarnings('ignore')

args = get_parser().parse_args()

inPath = args.input
outPath = args.output

numVid = 0

for k, vid in enumerate(os.listdir(inPath)):
    print(vid)

    video = skvideo.io.vread(os.path.join(inPath, vid))

    for i in np.arange(200, np.size(video, 1) - 1024 - 200, 200):
        for j in np.arange(200, np.size(video, 2) - 1024 - 200, 150):
            skvideo.io.vwrite(os.path.join(outPath, 'Cropped_' + str(numVid + 1) + '.MP4'),
                              video[:, i:i+1024, j:j+1024, :],
                              inputdict={'-r': '25/1'},
                              outputdict={'-r': '25/1',
                                          '-pix_fmt': 'yuv420p'})
            numVid += 1
