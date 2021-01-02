""" Calculates precision/recall/F1 for specified video with detections and human annotations.

CL Args:
  -i Path to input detections video file.
  -a Path to annotations file.
"""

import numpy as np
import pickle
import cv2
from sklearn.metrics import pairwise_distances
from util import get_parser

THR = 30
DIST_THR = 20

args = get_parser().parse_args()

ANNOT = args.annot
PRED = args.input

with open(ANNOT, 'rb') as f:
    annot = pickle.load(f)

annot = annot[:]
nframes = len(annot)

vs = cv2.VideoCapture(PRED)
frames = []
ret = True
kernel = np.ones((3, 3), np.uint8)

while vs.isOpened() and ret:
    ret, frame = vs.read()
    if ret:
        (width, height) = frame.shape[:2]
        dets = cv2.threshold(frame[..., 0], THR, 255, cv2.THRESH_BINARY)[1]
        dets = cv2.erode(dets, kernel, iterations=1)
        dets = cv2.dilate(dets, kernel, iterations=1)
        cnts = cv2.findContours(dets, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        frames.append([])
        for c in cnts:
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            frames[-1].append((cx, cy))

nframes = len(frames)

tp = 0
fp = 0
fn = 0
for k in range(nframes):
    if len(frames[k]) > 0 and len(annot[k]) > 0:
        dist = pairwise_distances(frames[k], annot[k])
        within_thr = dist < DIST_THR
        within_thr_sum = np.sum(within_thr, axis=0, keepdims=True)
        tp += np.count_nonzero(within_thr_sum)
        fn += np.count_nonzero(within_thr_sum == 0)
        fp += np.count_nonzero(within_thr_sum - 1 > 0)
    elif len(frames[k]) == 0 and len(annot[k]) > 0:
        fn += len(annot[k])
    else:
        fp += len(frames[k])

r = tp / (tp+fn)
p = tp / (tp+fp)
f1 = 2*r*p/(r+p)

print('Recall: {:.2f}'.format(r))
print('Precision: {:.2f}'.format(p))
print('F1: {:.2f}'.format(f1))
