# Copyright (c) 2021 Project Bee4Exp.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Calculates precision/recall/F1 on synthetic test dataset.

CL Args:
  --test_data Path to HDF5 test dataset.
  --model Path to trained model.
  --thr Detection threshold value.
"""

from keras.models import load_model
from generator import HDF5Generator
from skimage.transform import rescale
import numpy as np
from util import get_parser
import cv2
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings("ignore")

args = get_parser().parse_args()

TEST_DATASET = args.test_data

model = load_model(args.model)

test_generator = HDF5Generator(TEST_DATASET, shuffle=False)

kernel = np.ones((3, 3), np.uint8)

tp = 0
fp = 0
fn = 0

DIST_THR = 20

for i in range(200):
    X, y = test_generator.__getitem__(i)

    preds = model.predict(X)

    for j in range(X.shape[0]):
        pp = rescale(preds[j, :, :, 0], 2, anti_aliasing=False)
        pp = pp > args.thr
        pp = pp.astype(np.uint8)
        tt = rescale(y[j, :, :, 0], 2, anti_aliasing=False)
        tt = tt > 0.04
        tt = tt.astype(np.uint8)

        pp_cnt = []
        tt_cnt = []

        dets = cv2.erode(pp, kernel, iterations=1)
        dets = cv2.dilate(pp, kernel, iterations=1)
        cnts = cv2.findContours(dets, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in cnts:
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            pp_cnt.append((cx, cy))

        dets = cv2.erode(tt, kernel, iterations=1)
        dets = cv2.dilate(tt, kernel, iterations=1)
        cnts = cv2.findContours(dets, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in cnts:
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            tt_cnt.append((cx, cy))

        if len(pp_cnt) > 0 and len(tt_cnt) > 0:
            dist = pairwise_distances(pp_cnt, tt_cnt)
            within_thr = dist < DIST_THR
            within_thr_sum = np.sum(within_thr, axis=0, keepdims=True)
            tp += np.count_nonzero(within_thr_sum)
            fn += np.count_nonzero(within_thr_sum == 0)
            fp += np.count_nonzero(within_thr_sum - 1 > 0)
        elif len(pp_cnt) == 0 and len(tt_cnt) > 0:
            fn += len(tt_cnt)
        else:
            fp += len(pp_cnt)

r = tp / (tp+fn)
p = tp / (tp+fp)
f1 = 2*r*p/(r+p)

print('{:.2f}/{:.2f}/{:.2f}'.format(r, p, f1))
