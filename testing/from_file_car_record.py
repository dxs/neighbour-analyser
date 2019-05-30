import cv2 as cv
import argparse
import sys
import numpy as np 
import os.path 

#set constants
FRONT_CAMERA = 1
BACK_CAMERA = 0
i = 0
confThreshold = 0.5 #Confidence threshold
nmsThreshold = 0.4 #Non-maximum suppression threshold
inpWidth = 416   #Width of network's input image
inpHeight = 416 #Height of network's input image

# Load names of classes
classesFile = 'coco.names'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# LOAD MODEL AND CLASSES
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = 'yolov3-tiny.cfg' # Network configuration
modelWeights = 'yolov3-tiny.weights' #Pre-trained network's weights

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] -1] for i in net.getUnconnectedOutLayers()]

# Process inputs
outputFile = 'cars.avi'

cap = cv.VideoCapture('video_in_car.mp4')

# Get the video writer initialized to save the output video when needed
video_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

counter_seen_car_ago = 0

while cv.waitKey(1) < 0:
    # Get frame from the video
    hasFrame, frame = cap.read()

    if not hasFrame:
        print('Done processing !!!')
        cv.waitKey(3000)
        break

    if counter_seen_car_ago > 0:
        counter_seen_car_ago = counter_seen_car_ago-1
        video_writer.write(frame.astype(np.uint8))
        continue
    
    # Create a 4D blob from a frame
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                if(classes[classId] == 'car'):
                    video_writer.write(frame.astype(np.uint8))
                    i = i + 1
                    counter_seen_car_ago = 10
                    print('save img {0}'.format(i))
    
