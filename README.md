# Detection of Small Flying Objects in UAV Videos

This repository contains the code used in implementation of the paper Vladan Stojnić, Vladimir Risojević, Mario Muštra, Vedran Jovanović, Janja Filipi, Nikola Kezić, and Zdenka Babić, "Detection of Small Flying Objects in UAV Videos", submitted to Remote Sensing.

Dataset with used videos can be obtained at https://doi.org/10.5281/zenodo.4400650 .

Code was implemented using Python 3.6. To run the code please create Anaconda environment using dependancies defined in *bee4exp.yml*.

Main parts of our code are implemented in following python scripts.

## Stabilization

Script stabilization.py implements the code for video stabilization. To run the script use:

```
python stabilization.py -i INPUT_VIDEO_PATH -o OUTPUT_VIDEO_PATH
```

## Generation of synthetic honeybees

Script add_bees_to_video.py implements the code for creating videos with synthetic honeybees. To run the script use:

```
python add_bees_to_video.py -i INPUT_VIDEOS_DIR_PATH -o OUTPUT_VIDEOS_DIR_PATH --mask MASK_VIDEOS_DIR_PATH --bee_mean BEE_MEAN_VALUE [--num_synthetic_videos NUM_OF_VIDEOS]
```

## Background subtraction

Script bgsub.py implements the code for background subtraction. To run the script use:

```
python bgsub.py -i INPUT_VIDEO_PATH -o OUTPUT_VIDEO_PATH [--num_avg NUM_OF_FRAMES_FOR_AVERAGE]
```

## HDF5 Dataset creation

Script chunked_dataset.py implements the code for creation of HDF5 datasets. It can create train, val and test dataset. To run the script use:

```
python chunked_dataset.py -i INPUT_VIDEOS_DIR_PATH --mask MASK_VIDEOS_DIR_PATH -o OUTPUT_DATASET_PATH --type {train, val, test}
```

## Training

Script train.py implements the code for training of segmentation model. To run the script use:

```
python train.py --train_data TRAIN_DATASET_PATH --val_data VAL_DATASET_PATH --model MODEL_PATH
```

## Detection

Script detection.py implements the code for honeybee detection using trained model. To run the script use:

```
python detection.py -i INPUT_VIDEO_PATH -o DETECTION_VIDEO_PATH --model MODEL_PATH --heat_map DETECTION_HEAT_MAP_PATH
```

## Performance

Script synthetic_test.py implements the code for calculating precision/recall/F1 on synthetic test dataset. To run the script use:

```
python synthetic_test.py --test_data TEST_DATASET_PATH --model MODEL_PATH [--thr DETECTION_THRESHOLD]
```

Script perf.py implements the code for calculating precision/recall/F1 on detections with human labels. To run the script use:

```
python perf.py -i DETECTION_VIDEO_PATH -a ANNOTATIONS_FILE
```
