import cv2
import numpy as np
import pickle
import argparse
from os.path import isfile
from skimage.io import imsave

def on_buttonup(event, x, y, flags, param):
    global annot, cnt, nframes, frame
    if event == cv2.EVENT_LBUTTONUP:
        annot[cnt].append((x, y))
    elif event == cv2.EVENT_RBUTTONUP:
        if len(annot[cnt]) > 0:
            dist = [(p[0] - x)**2 + (p[1] - y)**2 for p in annot[cnt]]
            idx = np.argmin(dist)
            if dist[idx] < 200:
                del annot[cnt][idx]
    show_frame(frame, annot, cnt, nframes)
     
def show_frame(frame, annot, cnt, nframes):
    image = frame.copy()
    cv2.putText(image, 'frame: {:d}/{:d}'.format(cnt, nframes-1), 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255), 2)
    draw_annotations(image)
    cv2.imshow('frame', image)
 
def draw_annotations(img):
    global annot, cnt
    for (x, y) in annot[cnt]:
        cv2.circle(img, (x, y), 20, (0, 0, 255), 2)

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', help='Input video file',
                    type=str, required=True)
parser.add_argument('-a', '--annot', help='Annotations file',
                    type=str, required=True)
args =  parser.parse_args()

print("""
      Komande:
          x - forward frame
          z - backward frame
          s - save anotations
          q - quit
          left click - annotate bee
          right click  - delete annotation
          """)

vs = cv2.VideoCapture(args.video)
frames = []
ret = True
while vs.isOpened() and ret:
    ret, frame = vs.read()
    if ret:
        frames.append(frame)

#imsave('jedan_frejm.jpg', frames[0])

nframes = len(frames)
if isfile(args.annot):
    with open(args.annot, 'rb') as f:
        annot = pickle.load(f)
else:
    annot = [[] for i in range(nframes)]

cnt = 0
cv2.namedWindow('frame', cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('frame', on_buttonup)

while True:
    frame = frames[cnt]
    show_frame(frame, annot, cnt, nframes)

    key = cv2.waitKey(0) & 0xff
    if key == ord('q'):
        break
    elif key == ord('s'):
        with open(args.annot, 'wb') as f:
            pickle.dump(annot, f, pickle.HIGHEST_PROTOCOL)
    elif key == ord('z') and cnt > 0:
        cnt -= 1
    elif key == ord('x') and cnt < nframes-1:
        cnt += 1        
    
vs.release()
cv2.destroyAllWindows()

with open(args.annot, 'wb') as f:
    pickle.dump(annot, f, pickle.HIGHEST_PROTOCOL)
    
