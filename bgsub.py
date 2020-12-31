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
from util import get_parser
import warnings
warnings.filterwarnings('ignore')

def background_subtraction_loop(infile, outfile_bgsub, num_avg):
    """ Simple background subtraction using Gaussian as a model for background,
        rescaling intensity and thresholding.
        Path to the input video: infile
        Path to the video with background subtracted: outfile_bgsub
        Path to the video after thresholding: outfile_mask
        Number of frames for averaging: num_avg
    """
    
    metadata = skvideo.io.ffprobe(infile)
    nb_frames = int(metadata['video']['@nb_frames'])
    width = int(metadata['video']['@width'])
    height = int(metadata['video']['@height'])
    rate = metadata['video']['@r_frame_rate']
    videodata = skvideo.io.FFmpegReader(infile)
    print(outfile_bgsub)
    writer_bgsub = skvideo.io.FFmpegWriter(outfile_bgsub, inputdict={
            '-r': rate}, 
        outputdict={
            '-r': rate,
            '-pix_fmt': 'yuv420p'})
    
    videobuffer = np.zeros((num_avg, height, width))
    
    for k, f in enumerate(videodata.nextFrame()):
        videobuffer[np.mod(k, num_avg)] = rgb2gray(f)
        print (k)
        if np.mod(k, num_avg) == (num_avg-1):
            bgmodel = np.mean(videobuffer, axis=0)
            bgsigma = np.std(videobuffer, axis=0)   
            for fbuf in videobuffer:
                diff = fbuf - bgmodel
                minusbg = np.abs(diff) / (bgsigma + 1e-5)
                minusbg = rescale_intensity(minusbg)
                fgmask = minusbg > 0.75
                
                writer_bgsub.writeFrame(img_as_ubyte(minusbg))
        
        
    videodata.close()
    writer_bgsub.close()
	
args = get_parser().parse_args()
	
infile = args.input
bgsubout = args.output

num_avg = 50 # Number of frames for background estimation

background_subtraction_loop(infile, bgsubout, num_avg)
