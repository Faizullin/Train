import os
import argparse
import cv2
import numpy as np
import sys
sys.path.insert(0,'..')
import config
from src.pyimagesearch import *



# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--input', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--skip_frames', help='skip_frames of the video file',
                   type=int, default=20)

args = parser.parse_args();
min_conf_threshold = float(args.threshold)
import dlib
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter 


CWD_PATH = os.getcwd();
VIDEO_PATH = os.path.join(CWD_PATH,args.input)
PATH_TO_CKPT = '../'+config.PATHS['modelGraph']
PATH_TO_LABELS = '../'+config.PATHS['modelLabels']
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

maxCount = config.maxCount;

del config;

class VideoOper():
    def __init__(self,skip_frames=None):
        if skip_frames is None:
            self.skip_frames = args.skip_frames;
        else:
            self.skip_frames = int(self.skip_frames)
        self.totalFrames = 0;
    def skip(self):
        return self.totalFrames%self.skip_frames!=0
    
closed = False
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
 
ct = CentroidTracker(maxDisappeared=70)
trackers = []
trackableObjects = {}

video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
_,frame = video.read()
if frame is None:
    sys.exit()
H,W=frame.shape[:2];

print("START")
countF = VideoOper()
count=0;oldCount=0;
totalDown = 0;totalUp = 0;


#fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#writer = cv2.VideoWriter('ready.mp4', fourcc, 30,(W, H), True)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (W,H))
while(video.isOpened()):
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    rects = []
    if not countF.skip():
        
        print("Recog")
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                object_name = labels[int(classes[i])]
                if object_name!='person':continue;
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                tracker = dlib.correlation_tracker()
                print(xmin, ymin, xmax, ymax)
                rect = dlib.rectangle(xmin, ymin, xmax, ymax)
                tracker.start_track(frame_rgb, rect)
                trackers.append(tracker)
                rects.append((xmin , ymin, xmax, ymax))
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    else:
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"
            # update the tracker and grab the updated position
            tracker.update(frame_rgb)
            pos = tracker.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))
    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            print('to is None')
            to = TrackableObject(objectID, centroid)
        else:
            xx = [c[0] for c in to.centroids]
            #direction = centroid[1] - np.mean(y)
            direction = centroid[0] - np.mean(xx)
            to.centroids.append(centroid)
            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                #  if direction < 0 and centroid[1] < H // 2:
                if direction < 0 and centroid[0] < W // 2:
                    totalUp += 1
                    to.counted = True
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                elif direction > 0 and centroid[0] > W // 2:
                    totalDown += 1
                    to.counted = True
                print("If to",direction,totalDown,totalUp)
        trackableObjects[objectID] = to
        text = "ID {}".format(objectID)
        print(text)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    cv2.line(frame, (W// 2, 0), (W//2, H), (0, 255, 255), 2)
    count = totalDown-totalUp;
    if(count!=oldCount):
        if count < 0: count =0;
        elif count >= maxCount:
            count = maxCount
            closed = True
        elif closed and count<maxCount:
            closed = False
        oldCount = count;
    info = [
        ("OUT", totalUp),
        ("IN", totalDown),
        ('INSIDE',count)
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if writer is not None:
        writer.write(frame)
    cv2.imshow('Object detector', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
    countF.totalFrames += 1;
if writer is not None:
    writer.release()
video.release()
cv2.destroyAllWindows()
print("END")
