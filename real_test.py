from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from nets import BeyondCountingModel
from generator import HDF5Generator
from sklearn.metrics import confusion_matrix
from skimage.transform import rescale
from skimage.io import imsave
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
from skimage.color import rgb2gray
from skimage.util import img_as_float, img_as_ubyte
from keras.utils import plot_model
from keras.utils import multi_gpu_model
from util import get_parser
from scipy.ndimage.measurements import label
import warnings
warnings.filterwarnings("ignore")

def heat_map(frame, m, n) :
    
    i, j = frame.shape
    
    HM = np.zeros((i // m, j // n))
    
    for ii in np.arange(0, i, m) :
        for jj in np.arange(0, j, n) :
            pom = frame[ii : ii + m, jj : jj + n]
            
            labeled_array, num_labels = label(pom)
            
            HM[ii // m, jj // n] = num_labels
             
    return HM

skala = 64
m = 64
n = 64

args = get_parser().parse_args()

INPUT_FILE = args.input
outfile = args.output
heat = args.heat


metadata = skvideo.io.ffprobe(INPUT_FILE)
rate = metadata['video']['@r_frame_rate']
width = int(metadata['video']['@width'])
height = int(metadata['video']['@height'])
videodata = skvideo.io.FFmpegReader(INPUT_FILE)

writer = skvideo.io.FFmpegWriter(outfile, inputdict={
            '-r': rate}, 
        outputdict={
            '-r': rate,
            '-pix_fmt': 'yuv420p'})

model = BeyondCountingModel(input_shape = (height, width, 5))
model = multi_gpu_model(model)
model.load_weights('../Model/model_0.25_0.50_newbee_bigger_bees_bgsub2.hdf5')

X = np.zeros((1, height, width, 5))

X[0, :, :, 0] = rgb2gray(next(videodata.nextFrame()))
X[0, :, :, 0] -= 0.5
X[0, :, :, 1] = rgb2gray(next(videodata.nextFrame()))
X[0, :, :, 1] -= 0.5
X[0, :, :, 2] = rgb2gray(next(videodata.nextFrame()))
X[0, :, :, 2] -= 0.5
X[0, :, :, 3] = rgb2gray(next(videodata.nextFrame()))
X[0, :, :, 3] -= 0.5

writer.writeFrame(img_as_ubyte(np.zeros((height, width))))
writer.writeFrame(img_as_ubyte(np.zeros((height, width))))

try:
    for k, f in enumerate(videodata.nextFrame()):
        print(k)
        X[0, :, :, 4] = rgb2gray(f)
        X[0, :, :, 4] -= 0.5
        
        pred = model.predict(X)
        ff = rescale(pred[0, :, :, 0], 2, anti_aliasing = False)
    
        writer.writeFrame(img_as_ubyte(ff))
    
        frame_1 = ff[0 : (height // skala) * skala, 0 : (width // skala) * skala] > 0.15
        
        HM = heat_map(frame_1, m, n)
        
        if (k == 0) :
            mapa = HM
        else :
            mapa = mapa + HM

        X[0, :, :, 0] = X[0, :, :, 1].copy()
        X[0, :, :, 1] = X[0, :, :, 2].copy()
        X[0, :, :, 2] = X[0, :, :, 3].copy()
        X[0, :, :, 3] = X[0, :, :, 4].copy()
except RuntimeError:
    a = 0

plt.figure()    
plt.imshow(mapa, cmap = 'hot')
plt.colorbar()
plt.title("Heatmap ulaznog videa na prozoru " + str(m) + "x" + str(n) + ".")
plt.savefig(heat + str(m) + 'x' + str(n) + '.pdf', bbox_inches='tight')

np.save(heat + str(m) + 'x' + str(n), mapa)

videodata.close()
writer.close()
