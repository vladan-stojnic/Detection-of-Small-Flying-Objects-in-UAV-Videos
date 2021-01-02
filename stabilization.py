""" Stablilizes a video using ORB detector.

CL Args:
  -i Path to input video file.
  -o Path to output video file.
"""

import cv2
import numpy as np
import skvideo.io
from util import get_parser
from skimage.util import img_as_ubyte
from skimage.transform import warp, AffineTransform
from skimage.measure import ransac

args = get_parser().parse_args()

INFILE = args.input
OUTFILE = args.output

cap = cv2.VideoCapture(INFILE)

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(n_frames)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(cap.get(cv2.CAP_PROP_FPS))

metadata = skvideo.io.ffprobe(INFILE)
rate = metadata['video']['@r_frame_rate']

writer = skvideo.io.FFmpegWriter(OUTFILE, inputdict={'-r': rate, '-pix_fmt': 'bgr24'},
                                 outputdict={'-r': rate, '-pix_fmt': 'yuv420p'})

_, ref_frame = cap.read()

writer.writeFrame(ref_frame)

ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(2000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

frameA, descA = orb.detectAndCompute(ref_frame[300:h-300, 300:w-300], None)

for k in range(1, n_frames):
    print(k)
    _, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Skip 300px from each border to prevent problems with white tapes on
    # minefield edges.
    frameB, descB = orb.detectAndCompute(frame_gray[300:h-300, 300:w-300], None)
    matches = matcher.match(descA, descB)

    pointsA = np.float32([frameA[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pointsB = np.float32([frameB[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    tform, inliers = ransac((pointsA, pointsB), AffineTransform,
                            min_samples=4, residual_threshold=1,
                            max_trials=100)

    warped_frame = warp(frame, tform, output_shape=frame.shape)
    writer.writeFrame(img_as_ubyte(warped_frame))

writer.close()
cap.release()
