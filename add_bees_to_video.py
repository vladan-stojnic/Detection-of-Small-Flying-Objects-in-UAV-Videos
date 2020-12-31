import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rotate
from skimage import exposure
from random import randint
import cv2
import skvideo.io
import os
from scipy.ndimage.filters import gaussian_filter

inPath = "../Video/Cropped_1024x1024/" # Path to background videos
outPath = "../Video/BeesAdded_0.30_large_1024x1024/Video/" # Path to videos with synthetic bees
maskPath = "../Video/BeesAdded_0.30_large_1024x1024/Mask/" # Path to videos that contain ground truth masks

gaussian_mean = 0.3 # Gaussian mean for synthetic bees

def gaussian_kernel(size = (9, 9), std = (2, 2)) :
    
    out = np.zeros(size)
    
    out[size[0] // 2, size[1] // 2] = 1
    
    out = gaussian_filter(out, std)

    out /= out.max()

    out = (gaussian_mean + 0.07*np.random.randn(*size))*(out > 0.5).astype(np.float)

    return out

kernelSize = np.array([9, 9])
kernelSizeHalf = 4
minBee = 5
maxBee = 15
maxMovement = 20
stdIntens = 2
stdAngles = 30
minIntens = 3
maxIntens = maxMovement * np.sqrt(2)
maxAngles = 60

inputs = os.listdir(inPath)

numVid = 10000

for k in range(numVid) :
    print(k + 1)
    
    inName = inPath + inputs[randint(0, len(inputs) - 1)]
    print(inName)

    outName = outPath + "Out_" + str(k + 1) + '.MP4'
    outMask = maskPath + "Mask_" + str(k + 1) + '.MP4'

    inVideo = skvideo.io.vread(inName)
    outVideo = np.zeros_like(inVideo)
    maskVideo = np.zeros_like(inVideo)

    frame1 = inVideo[0]

    frameSize = frame1.shape[0:2]

    numBee = randint(minBee, maxBee)

    curLocX = np.random.randint(kernelSizeHalf, frameSize[0] - kernelSizeHalf, numBee)
    curLocY = np.random.randint(kernelSizeHalf, frameSize[1] - kernelSizeHalf, numBee)

    boja = np.random.uniform(0, 5)
    
    frame = np.zeros(frameSize)

    frame[curLocX, curLocY] = 1.0
    kernel = gaussian_kernel(kernelSize, (1.5, 2.5))
    
    frame = cv2.filter2D(frame, -1, kernel)
    alpha = np.expand_dims(frame, axis = 2)
    
    
    outFrame = (frame1 * (1 - alpha)) + boja * alpha
    
    
    outVideo[0] = outFrame
    
    if alpha.max() == 0:
        maskVideo[0] = alpha*255
    else:
        maskVideo[0] = (alpha / alpha.max()) * 255
    
    curMovX = np.random.randint(-maxMovement, maxMovement, numBee)
    curMovY = np.random.randint(-maxMovement, maxMovement, numBee)
    
    intens = np.clip(np.linalg.norm(np.vstack((curMovX, curMovY)), axis = 0), minIntens, maxIntens)
    angles = np.arctan2(curMovX, curMovY) / np.pi * 180

    numFrame = 1
    while(numFrame < inVideo.shape[0]) :
        frame1 = inVideo[numFrame]
        
        intens = np.clip(np.random.normal(intens, stdIntens), minIntens, maxIntens)
        angles += np.clip(np.random.normal(0, stdAngles), -maxAngles, maxAngles)
        
        curLocX += np.round(intens * np.sin(np.pi * angles / 180)).astype('int64') 
        curLocY += np.round(intens * np.cos(np.pi * angles / 180)).astype('int64')
        
        frame = np.zeros(frameSize)
        
        bee_width = np.random.uniform(2, 2.5, numBee)
        bee_height = np.random.uniform(1, 1.5, numBee)


        for j in range(numBee) :
            if not(curLocX[j] >= frameSize[0]-kernelSizeHalf or curLocX[j] < kernelSizeHalf or curLocY[j] >= frameSize[1]-kernelSizeHalf or curLocY[j] < kernelSizeHalf) : 
                
                
                pom = gaussian_kernel(kernelSize, (bee_height[j], bee_width[j]))
                pom = rotate(pom, -angles[j])

                frame[curLocX[j]-kernelSizeHalf:curLocX[j]+kernelSizeHalf+1, 
                      curLocY[j]-kernelSizeHalf:curLocY[j]+kernelSizeHalf+1] = pom
                
        alpha = np.expand_dims(frame, axis = 2)
        
        outFrame = (frame1 * (1 - alpha)) + boja * alpha
    
        outVideo[numFrame] = outFrame
    
        if alpha.max() == 0:
            maskVideo[numFrame] = alpha*255
        else:
            maskVideo[numFrame] = (alpha / alpha.max()) * 255
        numFrame += 1


    skvideo.io.vwrite(outName, outVideo)
    skvideo.io.vwrite(outMask, maskVideo)
