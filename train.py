# Copyright (c) 2021 Project Bee4Exp.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Trains CNN segmentation model using synthetic datasets.

CL Args:
  --train_data Path to HDF5 training dataset.
  --val_data Path to HDF5 validation dataset.
  --model Path to model save file.
"""

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.utils import multi_gpu_model
from nets import SegmentationModel
from generator import HDF5Generator
from util import get_parser
import warnings
warnings.filterwarnings("ignore")

args = get_parser().parse_args()

TRAIN_DATASET = args.train_data
VALIDATION_DATASET = args.val_data

optimizer = Adam(lr=1e-3)

reduce_lr = LearningRateScheduler(lambda epoch, lr: lr/5 if ((epoch + 1) % 20) == 0 else lr, verbose=1)

early_stop = EarlyStopping(monitor='val_loss', patience=10)

checkpoint = ModelCheckpoint(filepath=args.model, period=1, save_best_only=True, mode='min')

model = SegmentationModel(input_shape=(256, 256, 5))

try:
    model = multi_gpu_model(model)
except Exception:
    pass

model.compile(optimizer=optimizer, loss='mse')

train_generator = HDF5Generator(TRAIN_DATASET, batch_size=64)

validation_generator = HDF5Generator(VALIDATION_DATASET, shuffle=False, batch_size=64)

model.fit_generator(train_generator,
                    epochs=500,
                    verbose=1,
                    callbacks=[reduce_lr, early_stop, checkpoint],
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    shuffle=False)
