""" Performs background subtraction.

CL Args:
  -i Path to input video file.
  -o Path to output video file.
  --num_avg Number of frames used for average.
"""

import numpy as np
import skvideo.io
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from util import get_parser
import warnings
warnings.filterwarnings('ignore')


def background_subtraction_loop(infile, outfile_bgsub, num_avg):
    """ Simple background subtraction using Gaussian as a model for background,
        rescaling intensity and thresholding.
        Path to the input video: infile
        Path to the video with background subtracted: outfile_bgsub
        Number of frames for averaging: num_avg
    """

    metadata = skvideo.io.ffprobe(infile)
    width = int(metadata['video']['@width'])
    height = int(metadata['video']['@height'])
    rate = metadata['video']['@r_frame_rate']
    videodata = skvideo.io.FFmpegReader(infile)
    writer_bgsub = skvideo.io.FFmpegWriter(outfile_bgsub, inputdict={'-r': rate},
                                           outputdict={'-r': rate, '-pix_fmt': 'yuv420p'})

    videobuffer = np.zeros((num_avg, height, width))

    for k, f in enumerate(videodata.nextFrame()):
        videobuffer[np.mod(k, num_avg)] = rgb2gray(f)
        if np.mod(k, num_avg) == (num_avg-1):
            bgmodel = np.mean(videobuffer, axis=0)
            bgsigma = np.std(videobuffer, axis=0)
            for fbuf in videobuffer:
                diff = fbuf - bgmodel
                minusbg = np.abs(diff) / (bgsigma + 1e-5)
                minusbg = rescale_intensity(minusbg)

                writer_bgsub.writeFrame(img_as_ubyte(minusbg))

    videodata.close()
    writer_bgsub.close()


args = get_parser().parse_args()

infile = args.input
bgsub_out = args.output

num_avg = args.num_avg

background_subtraction_loop(infile, bgsub_out, num_avg)
