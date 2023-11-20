import sys
import os
import glob
import threading
import time
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from queue import Queue
import csv

from djitellopy import Tello
from coloredObjectExtractor import ColoredObjectExtractor

default_command_delay_time = 0.1 #7
picture_first_frame_delay_time = 4.0
totalNumberFramesProcessed = 0
processedImages = []
processedResults = []

flights = [
    # 0
    # 0
    {'name':'Offline', 'requiresDrone':False, 'unprocessed':'Panorama-full-clockwise-360', 'initiallyAcquiredAt':'1700463092'},
    #{'name':'Offline', 'requiresDrone':False, 'unprocessed':'Stationary', 'initiallyAcquiredAt':'1700285669'},
    #{'name':'Offline', 'requiresDrone':False, 'unprocessed':'Speed75-Up-Back-Back-Forward', 'initiallyAcquiredAt':'1699923999'},
    #{'name':'Offline', 'requiresDrone':False, 'unprocessed':'Speed100-Up-CW-Back-Back-CW-Forward-CCW', 'initiallyAcquiredAt':'1700199542'},
    # 1
    {'name':'Stationary', 'requiresDrone':True, 'flight_segments':[
        {'frameGrabDelay': 1, 'frameGrabInterval': 2, 'numFrameGrabIntervals':4    , 'durationLimit': 10 } ] },
    # 2    
    {'name':'Back-Forward-Back', 'requiresDrone':True, 'flight_segments':[
        {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 8},
        {'action':'back 20', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
        {'action':'forward 80', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':5,'durationLimit': 8},
        {'action':'back 60', 'frameGrabDelay': -1, 'durationLimit': 8}, 
        {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] },
    # 3
   {'name':'Panorama-full-counter-clockwise', 'requiresDrone':True, 'flight_segments':[
       {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 8},
       {'action':'ccw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'ccw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'ccw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'ccw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'ccw 40', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] },
    # 4
   {'name':'Panorama-full-clockwise', 'requiresDrone':True, 'flight_segments':[
       {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 10},
       {'action':'down 45', 'frameGrabDelay': -1, 'durationLimit': 10},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},

       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},

       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 24', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},

       {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] },       
    # 5
   {'name':'Panorama-half-counter-clockwise', 'requiresDrone':True, 'flight_segments':[
       {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 10},
       {'action':'cw 90', 'frameGrabDelay': -1, 'durationLimit': 8},
       {'action':'ccw 60', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'ccw 60', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'ccw 60', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 90', 'frameGrabDelay': -1, 'durationLimit': 8},
       {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] },
     #6  
    {'name':'Panorama-half-clockwise', 'requiresDrone':True, 'flight_segments':[
       {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 10},
       {'action':'ccw 90', 'frameGrabDelay': -1, 'durationLimit': 8},
       {'action':'cw 60', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 60', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 60', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'ccw 90', 'frameGrabDelay': -1, 'durationLimit': 8},
       {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] },
    # 7   
    {'name':'Speed85-Up-Back-Back-Forward', 'requiresDrone':True, 'flight_segments':[
        {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 8},
        {'action':'speed 85', 'frameGrabDelay': -1, 'durationLimit': 8},
        {'action':'up 50','frameGrabDelay': -1, 'durationLimit': 8},
        {'action':'back 400', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':4,'durationLimit': 10},
        {'action':'back 200', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 10},
        {'action':'forward 160', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':10,'durationLimit': 10},
        {'action':'cw 90', 'frameGrabDelay': -1, 'durationLimit': 8},
        {'action':'ccw 60', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
        {'action':'ccw 60', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
        {'action':'ccw 60', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
        {'action':'cw 90', 'frameGrabDelay': -1, 'durationLimit': 8},
        {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] },
    # 8   
    {'name':'Speed100-Up-CW-Back-Back-CW-Forward-CCW', 'requiresDrone':True, 'flight_segments':[
        {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 15},
        {'action':'speed 90', 'frameGrabDelay': -1, 'durationLimit': 10},
        #{'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':4,'durationLimit': 8},
        #{'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':4,'durationLimit': 8},
        #{'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':4,'durationLimit': 8},
        #{'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':4,'durationLimit': 8},
        #{'action':'cw 40', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':4,'durationLimit': 8},
        {'action':'up 50','frameGrabDelay': -1, 'durationLimit': 20},
        {'action':'cw 30', 'frameGrabDelay': -1, 'durationLimit': 20},
        {'action':'back 240','frameGrabDelay': 22, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 20},
        {'action':'back 240','frameGrabDelay': 22, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 20},
        #{'action':'back 140', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':6,'durationLimit': 10},
        #{'action':'cw 15', 'frameGrabDelay': 0.1, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 10},
        #{'frameGrabDelay': 0.5, 'frameGrabInterval': 0.5, 'numFrameGrabIntervals':4, 'durationLimit': 3 },
        {'action':'down 20','frameGrabDelay': 22, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 20},
        {'action':'cw 60', 'frameGrabDelay':-1, 'durationLimit': 10},
        {'action':'forward 224', 'frameGrabDelay':-1, 'durationLimit': 20},
        {'action':'forward 230', 'frameGrabDelay':-1, 'durationLimit': 20},
        {'action':'ccw 135', 'frameGrabDelay':12, 'durationLimit': 10},
        #{'frameGrabDelay': 0.5, 'frameGrabInterval': 0.5, 'numFrameGrabIntervals':4, 'durationLimit': 3 },
        #{'action':'forward 150', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 20},
         #'frameGrabDelay': 0, 'frameGrabInterval': 0.5, 'numFrameGrabIntervals':4,'durationLimit': 10},
        #{'action':'cw 75', 'frameGrabDelay': 0.1, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 10},
        #{'action':'down 95','frameGrabDelay': -1, 'durationLimit': 10},
        #{'action':'forward 240', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':3,'durationLimit': 20},
        #{'action':'down 40','frameGrabDelay':  0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 10},
        #{'action':'forward 60', 'frameGrabDelay': 0, 'frameGrabInterval': 0.5, 'numFrameGrabIntervals':4,'durationLimit': 10},
        #{'action':'ccw 45', 'frameGrabDelay': 0.1, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 10},
        #{'action':'ccw 45', 'frameGrabDelay': 0.1, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 10},
        #{'action':'ccw 45', 'frameGrabDelay': 0.1, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 10},
        #{'action':'ccw 30', 'frameGrabDelay': -1, 'durationLimit': 10},
        #{'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1, 'durationLimit': 2 },
        {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 10} ] },
    #9    
    {'name':'Panorama-full-clockwise-45-90-90-90-45', 'requiresDrone':True, 'flight_segments':[
        {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 10},
        #{'action':'down 45', 'frameGrabDelay': -1, 'durationLimit': 10},
        {'action':'cw 45', 'frameGrabDelay': 10, 'extraHoverTime':0.5, 'durationLimit': 8},
        {'action':'cw 45', 'frameGrabDelay': 10, 'extraHoverTime':0.5, 'durationLimit': 8},
        {'action':'cw 45', 'frameGrabDelay': 10, 'extraHoverTime':0.5, 'durationLimit': 8},
        {'action':'cw 45', 'frameGrabDelay': 10, 'extraHoverTime':0.5, 'durationLimit': 8},

        {'action':'cw 45', 'frameGrabDelay': 10, 'extraHoverTime':0.5, 'durationLimit': 8},
        {'action':'cw 45', 'frameGrabDelay': 10, 'extraHoverTime':0.5, 'durationLimit': 8},
        {'action':'cw 45', 'frameGrabDelay': 10, 'extraHoverTime':0.5, 'durationLimit': 8},
        {'action':'cw 45', 'frameGrabDelay': 10, 'extraHoverTime':0.5, 'durationLimit': 8},
        #{'action':'cw 45', 'frameGrabDelay': -1, 'durationLimit': 8},
        {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] },
    #10
    {'name':'Panorama-horizontal-round-trip', 'requiresDrone':True, 'flight_segments':[
        {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 16}, #aquire one image after taking off completes
        #{'action':'down 45', 'frameGrabDelay': -1, 'durationLimit': 10},
        {'action':'left 200', 'frameGrabDelay': -1, 'durationLimit': 10},
        {'action':'right 380', 'frameGrabDelay':0, 'frameGrabInterval': .5, 'numFrameGrabIntervals':12,'durationLimit': 12},
        #{'action':'right 61', 'frameGrabDelay': 12, 'durationLimit': 10},
        #{'action':'right 61', 'frameGrabDelay': 12, 'durationLimit': 10},
        #{'action':'cw 90','frameGrabDelay': 12, 'durationLimit': 10},
        {'action':'cw 180','frameGrabDelay': -1, 'durationLimit': 10},
        {'action':'right 380', 'frameGrabDelay':0, 'frameGrabInterval': .5, 'numFrameGrabIntervals':12,'durationLimit': 12},
        #{'action':'cw 90','frameGrabDelay': 12, 'durationLimit': 10},
        #{'action':'right 61', 'frameGrabDelay': 12, 'durationLimit': 10},
        #{'action':'right 61', 'frameGrabDelay': 12, 'durationLimit': 10},
        #{'action':'right 61', 'frameGrabDelay': 12, 'durationLimit': 10},
        #{'action':'right 61', 'frameGrabDelay': 12, 'durationLimit': 10},
        ##{'action':'right 61', 'frameGrabDelay': 12, 'durationLimit': 10},
        ##{'action':'right 61', 'frameGrabDelay': 12, 'durationLimit': 10},
        #{'action':'cw 90','frameGrabDelay': 12, 'durationLimit': 10},

        #{'action':'cw 90','frameGrabDelay': 12, 'durationLimit': 10},
        #{'action':'cw 180','frameGrabDelay': -1, 'durationLimit': 10},
        #{'action':'right 366', 'frameGrabDelay':0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':6,'durationLimit': 8},
        #{'action':'right 61', 'frameGrabDelay': 12, 'durationLimit': 10},
        #{'action':'right 61', 'frameGrabDelay': 12, 'durationLimit': 10},
        #{'action':'right 61', 'frameGrabDelay': -1, 'durationLimit': 10},        
        {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] }, 
    #11   
    {'name':'Panorama-full-clockwise-360', 'requiresDrone':True, 'flight_segments':[

        {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 10},

        #{'action':'down 45', 'frameGrabDelay': -1, 'durationLimit': 10},

        {'action':'cw 360', 'frameGrabDelay': 0, 'frameGrabInterval': 0.184, 'numFrameGrabIntervals':36,'durationLimit': 20},
        {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] }
    ]

colorKeyedObjectsDetectionConfigAndData = {
    'red': {'count': 0, 'min_area':350, 'max_area':1200},
    'darker_blue': {'count': 0, 'min_area':350, 'max_area':1200},
    'lighter_blue': {'count': 0, 'min_area':350, 'max_area':1200},
    'yellow': {'count': 0, 'min_area':400, 'max_area':250000},
    'orange': {'count': 0, 'min_area':400, 'max_area':20000},
    'darker_green': {'count': 0, 'min_area':350, 'max_area':1200},
    'purple': {'count': 0, 'min_area':400, 'max_area':20000},
    'lighter_green': {'count': 0, 'min_area':350, 'max_area':1200},
    'tennis_ball': {'count': 0, 'min_area':20, 'max_area':250},
    #'tennis_ball2': {'count': 0, 'min_area':10, 'max_area':300},
    'pink': {'count': 0, 'min_area':400, 'max_area':200000}  }

def getColorsAndCounts(configAndData):
    colors = [] #list(colorKeyedObjectsDetectionConfigAndData.keys())
    counts = []
    for colorKey in configAndData.keys():
        cad = configAndData[colorKey]
        colors.append(colorKey)
        counts.append(cad['count'])
    return colors, counts

def colorAnalyzeImage(image, show_image=True, saveInputImageToFolder=None, saveAnalyzedImageToFolder=None):
    global colorKeyedObjectsDetectionConfigAndData
    # [colorKey, minArea, x, y, h, w, a, r]
    fields = ['colorKey', 'minArea', 'centerX', 'centerY', 'height', 'width', 'area', 'ratio']
    if saveInputImageToFolder is not None:
        of_path = os.path.join(saveInputImageToFolder, f'ba_{time.time()}.jpg')
        cv2.imwrite(of_path, image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hsv[:, :, 0] = hsv[:, :, 0] * 1.05
    #hsv[:, :, 1] = hsv[:, :, 1] * .95
    #hsv[:, :, 2] = hsv[:, :, 2] * .95
    #image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 

    value = []
    for colorKey in colorKeyedObjectsDetectionConfigAndData.keys():
        cad = colorKeyedObjectsDetectionConfigAndData[colorKey]
        minArea = cad['min_area']
        maxArea = cad['max_area']
        coe = ColoredObjectExtractor(colorKey, min_area=minArea, max_area=maxArea)
        #coe.extract(hsv, image, True)
        objs = coe.extract(hsv, numIterations=2)
        n = cad['count']
        nThisFrame = len(objs)
        if n < nThisFrame:
            cad['count'] = nThisFrame
        for i in range(nThisFrame):
            (cc, e, a, r) = objs[i]
            ((x,y),(h,w),_) = e
            value.append({'colorKey':colorKey,
                          'minArea':minArea,
                          'minArea':maxArea,
                          'centerX':x,
                          'centerY':y,
                          'height':h,
                          'width':w,
                          'area':a,
                          'ratio':r}) #,
                          #'contourColor':cc})
            if True:
                cv2.ellipse(image, e, cc, 5)
                s = "{:.1f}, {:.3f}".format(a,r)
                cv2.putText(image, s, (int(x) - 50, int(y)+40),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, cc, 2, cv2.LINE_AA)

    if saveAnalyzedImageToFolder is not None:
        of_path = os.path.join(saveAnalyzedImageToFolder, f'aa_{time.time()}.jpg')
        cv2.imwrite(of_path, image)
        csv_path = os.path.join(saveAnalyzedImageToFolder, f'aa_{time.time()}.csv')
        with open(csv_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames = fields)
            writer.writeheader()
            writer.writerows(value)

    if show_image:
        cv2.imshow("color analysis result",image)
        cv2.waitKey()
    
    return image

def acquireImageFrames(imageBuffer:Queue, flightNumber:int):
    global flights
    print("Running image frame aquirer...")

    if flights[flightNumber]['requiresDrone']:
        # When connecting the drone to a home/office AP, we would need to find
        # the drone's DHCP assigned IP address and provide it as the host argument.
        # tello = Tello(host='192.168.50.170', retry_count=0)
        tello = Tello(retry_count=1)
        tello.connect() # this is a synchronous/blocking call, no need to wait after this call before sending next command
        responses = tello.get_own_udp_object()['responses'] #no need to wait because this API isn't tied to talking to the drone
        b = tello.get_battery() # this is getting an asynchronously updated drone state; no need to wait after this call
        print("Battery level: ", b)
        
        #b = 1
        if b < 15:
            print("Battery too low!!! Abort!!!")
            imageBuffer.put(None)
            print("Aborted image frame aquirer.")
            return
        tello.streamon() # this is a synchronous/blocking call, no need to wait after this call before sending next command
        tello.emergency()
        frame_reader = tello.get_frame_read()
        #Tello class hard-code a background frame "receiver" at:
        #        self.frame = np.zeros([300, 400, 3], dtype=np.uint8)
        segments = flights[flightNumber]['flight_segments']
        abortFlight = False        
        for s in segments:
            if 'action' in s.keys():
                hasAirAction = True
                tello.send_command_without_return(s['action'])
            else:
                hasAirAction = False

            segmentStartTime = time.time()
            frameGrabDelay = s['frameGrabDelay']
            duration = s['durationLimit']
            grabFrameAtSegmentEnd = False
            if frameGrabDelay == -1:
                # we are not grabing frames while completing this flight action
               pass
            else:
                if frameGrabDelay > duration:
                    # just grab a frame after this flight segment ends.
                     grabFrameAtSegmentEnd = True
                else:
                    frameGrabInterval = s['frameGrabInterval']
                    totalNumIntervals = s['numFrameGrabIntervals']
                    for ni in range(totalNumIntervals):
                        intervalStartTime = time.time()
                        if frameGrabDelay == 0:
                            # we are grabing a frame immediatly after sending the action command
                            # wait a 100ms to make sure at least a couple of frame intervals have
                            # passed from the time the drone receives the command.
                            frameGrabDelay = 0.1
                        time.sleep(frameGrabDelay)
                        img = frame_reader.frame
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #let the processing thread handle the conversion
                        imageBuffer.put(rgb_img)
                        if hasAirAction:
                            if responses:
                                break
                        delta_time = frameGrabInterval - (time.time() - intervalStartTime)
                        if delta_time > 0.01:
                            time.sleep(delta_time)
            if hasAirAction:
                while not responses:
                    delta_time = time.time() - segmentStartTime
                    if  delta_time > duration:
                        print("Aborting command '{}'. Did not receive a response after {} seconds".format(s['action'], delta_time))
                        abortFlight = True
                        break
                    else: # Sleep while waiting for segment to complete
                        moreTime = duration-delta_time
                        if moreTime > 0.1:
                            time.sleep(0.1)
                        else:
                            time.sleep(moreTime)

                if abortFlight:
                    break            
                r = responses.pop(0)
                r = r.decode("utf-8")
                r = r.rstrip("\r\n")
                
                print(s['action'], ' has response of: ', r, time.time())
            if grabFrameAtSegmentEnd:
                extrHoverTime = -1
                if 'extraHoverTime' in s.keys():
                    extrHoverTime = s['extraHoverTime']
                if extrHoverTime > 0.01:
                    time.sleep(extrHoverTime)
                img = frame_reader.frame
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imageBuffer.put(rgb_img)
      
        imageBuffer.put(None)
        if abortFlight:
            tello.land()  # this is a synchronous/blocking call, no need to wait after this call before sending next command
        tello.streamoff() # this is a synchronous/blocking call, no need to wait after this call before sending next command
    else:
        if 'unprocessed' in flights[flightNumber].keys():
            start_folder = os.path.join('../output', flights[flightNumber]['unprocessed'])
            start_folder = os.path.join(start_folder,  flights[flightNumber]['initiallyAcquiredAt'])
            unprocessedOutputFolder = os.path.join(start_folder, 'unprocessed')
            if not os.path.exists(unprocessedOutputFolder):
                print('Unprocessed image folder not found!')
            else:
                fileList = glob.glob(os.path.join(unprocessedOutputFolder, '*.png'))  #jpg
                for f in fileList:
                    rgb_img = cv2.imread(f)
                    imageBuffer.put(rgb_img)
        else:                   
            # acquire image frames from other (streaming) sources???
            rgb_img = cv2.imread("balloon_seen_stationary2.png")
            imageBuffer.put(rgb_img)
        imageBuffer.put(None)
    print("Finished runing image frame aquirer.")

def stitchFisheyeImage(dst_image, src_image, degrees_clock_wise, angel_span, inner_radius):
    srcSizeX = src_image.shape[1]
    srcSizeY = src_image.shape[0]
    dstSizeX = dst_image.shape[1]
    dstSizeY = dst_image.shape[0]
    #print("[Fisheye Stitcht] Source canvas size:", srcSizeY, srcSizeX)
    #print("[Fisheye Stitcht] Destination canvas size:", dstSizeY, dstSizeX)

    lc = np.zeros_like(dst_image, dtype=np.uint8)
    xPos = (dstSizeX//2 - srcSizeX//2)
    yPos = (dstSizeY//2 - srcSizeY - inner_radius)
    if xPos < 0:
        xStart = srcSizeX + xPos
        xPos = 0
    else:
        xStart = 0
    if yPos < 0:
        yStart = srcSizeY + yPos
        yPos = 0
    else:
        yStart = 0
    #print("xPos=", xPos,"yPos=", yPos, "xStart=", xStart,"yStart=", yStart)
    lc[yPos+yStart:yPos+srcSizeY, xPos+xStart:xPos+srcSizeX] = src_image[yStart:, xStart:]
    center = (dstSizeX//2, dstSizeY//2 + inner_radius)  # this is in the form of (x, y), rather than (y, x)
    mask = np.zeros_like(lc, dtype = np.uint8)
    #cv2.ellipse(mask, center, (dstSizeX//2, dstSizeY//2), 0, 258, 282, (255, 255, 255), thickness=cv2.FILLED)
    cv2.ellipse(mask, center, (dstSizeX//2, dstSizeY//2), 0, 270-angel_span/2, 270+angel_span/2, (255, 255, 255), thickness=cv2.FILLED)
    lc = cv2.bitwise_and(lc, mask)
    rotation_matrix = cv2.getRotationMatrix2D(center, degrees_clock_wise, 1)
    #rotation_matrix[0, 2] -= 0 //(1-0.75) * center[0] # x-axis
    #rotation_matrix[1, 2] -= 0 //center[1]
    lc = cv2.warpAffine(lc, rotation_matrix, (dstSizeX, dstSizeY))
    dst_image += lc
    
def processImageFrames(imageBuffer:Queue, fn:int):
    global totalNumberFramesProcessed, processedImages, processedResults
    print('Running image frame processer...')
    # process image frames
    f = flights[fn]
    start_time = int(time.time())
    start_folder = os.path.join('../output', f['name'])
    if not os.path.exists(start_folder):
        os.makedirs(start_folder)
    start_folder = os.path.join(start_folder, f'{start_time}')
    if not os.path.exists(start_folder):
        os.makedirs(start_folder)
    if f['requiresDrone']:
        unprocessedOutputFolder = os.path.join(start_folder,  'unprocessed')
        if not os.path.exists(unprocessedOutputFolder):
            os.makedirs(unprocessedOutputFolder)
    else:
        unprocessedOutputFolder = None
    processedOutputFolder = os.path.join(start_folder, 'processed')
    if not os.path.exists(processedOutputFolder):
        os.makedirs(processedOutputFolder)    

    nRead = 0
    readFrames=[]
    readFrames2 = []
    while True:
        item = imageBuffer.get()
        if item is None:
            if fn == 11:
                angles = 0
                deltaAngle = 360/nRead
                lc = np.zeros((lcRadius*2, lcRadius*2, 3), dtype=np.uint8)
                for frame in readFrames:
                    stitchFisheyeImage(lc, frame, angles, deltaAngle, ySize//4)
                    angles += deltaAngle
                img = colorAnalyzeImage(lc, show_image=False, saveInputImageToFolder=unprocessedOutputFolder, saveAnalyzedImageToFolder=processedOutputFolder)
                totalNumberFramesProcessed += 1
                processedImages.append(img)             
                break
            elif fn == 10:
                stitcher = cv2.Stitcher_create(mode = cv2.Stitcher_PANORAMA) #cv2.Stitcher_SCANS
                #stitcher.setPanoConfidenceThresh(0.0)
                print(len(readFrames))
                print(len(readFrames2))
                (status, stitched) = stitcher.stitch(readFrames[0:4])
                if status == 0:
                    img1 = colorAnalyzeImage(stitched, show_image=False, saveInputImageToFolder=None, saveAnalyzedImageToFolder=processedOutputFolder)
                    totalNumberFramesProcessed += 1
                    processedImages.append(img1)
                    colors, counts = getColorsAndCounts(colorKeyedObjectsDetectionConfigAndData)
                    processedResults.append((colors, counts))
                else:
                    print("Error stitching Front View!")
                (status, stitched) = stitcher.stitch(readFrames2[0:4])
                if status == 0:
                    img2 = colorAnalyzeImage(stitched, show_image=False, saveInputImageToFolder=None, saveAnalyzedImageToFolder=processedOutputFolder)
                    totalNumberFramesProcessed += 1
                    processedImages.append(img2)
                    colors, counts = getColorsAndCounts(colorKeyedObjectsDetectionConfigAndData)
                    processedResults.append((colors, counts))
                else:
                    print("Error stitching Back View!")
                break                  
            elif fn == 0:
                stitcher = cv2.Stitcher_create(mode = cv2.Stitcher_PANORAMA) #cv2.Stitcher_SCANS
                #stitcher.setPanoConfidenceThresh(0.0)
                #(status, stitched) = stitcher.stitch([readFrames[0], readFrames[1],readFrames[2], readFrames[3], readFrames[4]])
                ss = []
                print(len(readFrames))
                for i in range(0, 28, 2):
                    srcs = []
                    for j in range(5):
                        frame = readFrames[i+j]
                        srcs.append(frame)
                    print(len(srcs))
                    (status, stitched) = stitcher.stitch(srcs)
                    if status == 0:
                        stitched = cv2.resize(stitched, dsize=(960, 720), interpolation=cv2.INTER_CUBIC)
                        ss.append(stitched)                    
                ss2 = []
                for i in range (0, len(ss)-4, 2):
                    srcs = []
                    for j in range(4):
                        frame = ss[i+j]
                        srcs.append(frame)
                    (status, stitched) = stitcher.stitch(srcs)
                    if status == 0:
                        stitched = cv2.resize(stitched, dsize=(960*2, 720), interpolation=cv2.INTER_CUBIC)
                        ss2.append(stitched)
                if len(ss2) > 0:        
                    (status, stitched) = stitcher.stitch(ss2)
                    if status == 0:
                        stitched = cv2.resize(stitched, dsize=(960*4, 720), interpolation=cv2.INTER_CUBIC)
                        if status == 0:
                            img = colorAnalyzeImage(stitched, show_image=False, saveInputImageToFolder=unprocessedOutputFolder, saveAnalyzedImageToFolder=processedOutputFolder)
                            totalNumberFramesProcessed += 1
                            processedImages.append(img)                                 
            else:
                break
        if fn == 10:
            print(unprocessedOutputFolder, nRead)
            #if nRead == 0:
            of_path = os.path.join(unprocessedOutputFolder, f'ba_{nRead}.png')
            cv2.imwrite(of_path, item)
            if nRead < 12:
                readFrames.append(item)
            elif nRead > 11:
                readFrames2.append(item)
            nRead += 1
            print("fn is 0, Processed ", nRead, " frame(s) so far...")    
        elif fn == 0 or fn == 4 or fn == 11:
            if fn > 0:
                of_path = os.path.join(unprocessedOutputFolder, f'ba_{nRead}.png')
                cv2.imwrite(of_path, item)
            if nRead == 0:
                xSize = item.shape[1]
                ySize = item.shape[0]
                lcRadius = int(math.sqrt(xSize*xSize + ySize*ySize))
            readFrames.append(item)
            nRead += 1
            print("fn is 0, Processed ", nRead, " frame(s) so far...")
        elif totalNumberFramesProcessed < 100: #12:
            img = colorAnalyzeImage(item, show_image=False, saveInputImageToFolder=unprocessedOutputFolder, saveAnalyzedImageToFolder=processedOutputFolder)
            #rgb_after = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            totalNumberFramesProcessed += 1
            processedImages.append(img)
            colors, counts = getColorsAndCounts(colorKeyedObjectsDetectionConfigAndData)
            processedResults.append((colors, counts))
            print("Processed ", totalNumberFramesProcessed, " frame(s) so far...")

    # all done
    print('Finished running image frame processer.')

def missionTaskBeginner(flight_number=0, show_charts=0) -> int:
    global processedImages, totalNumberFramesProcessed, processedResults
    ib = Queue()
    t1 = threading.Thread(target = processImageFrames, args=(ib, flight_number, ))
    t2 = threading.Thread(target = acquireImageFrames, args=(ib, flight_number, ))
    s = time.time()
    print("[MISSION INFO] -- start time:", s)                    
    t1.start()
    t2.start()
    t2.join()
    ib.put(None)
    t1.join()

    print("Main thread: totalNumberFramesProcessed -- ", totalNumberFramesProcessed)
    e =  time.time()
    print("[MISSION INFO] -- end time:", e)
    print("[MISSION INFO] -- total flight + processing time:", e-s, "seconds.")
    print("[MISSION INFO] -- Frame by frame PROGRESSIVE results recap:")
    frn = 0 
    for pr in processedResults:
        print(frn, pr)
        frn += 1
    if show_charts < 1:
        return 0
    if totalNumberFramesProcessed > 0:

        fig = plt.figure(figsize=(12, 8))

        inner = gridspec.GridSpec(2, 1, hspace=0.2, height_ratios=[1,5])
        ax0 = plt.Subplot(fig, inner[0])
        ax1 = plt.Subplot(fig, inner[1])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax0.set_title("Objects detected per color")
        #ax0.bar(colors, vs, color ='maroon', width = 0.3)
        fig.add_subplot(ax0)
        fig.add_subplot(ax1)


        nrows = int((totalNumberFramesProcessed+1)/2)
        j = 0
        ris = []
        for i in range(nrows):
            img0 = cv2.cvtColor(processedImages[j], cv2.COLOR_BGR2RGB)
            j+=1
            if j<totalNumberFramesProcessed:
                img1 = cv2.cvtColor(processedImages[j], cv2.COLOR_BGR2RGB)
                ri = cv2.hconcat([img0, img1])
                ris.append(ri)
                j+=1
            else:
                ri = img0
                ris =[ri]


        concatedImage = cv2.vconcat(ris)
        
        class Index:
            def __init__(self):
                self.ind = 0
                self.prev(None)

            def next(self, event):
                if self.ind < (totalNumberFramesProcessed-1):

                    self.ind += 1
                    img = cv2.cvtColor(processedImages[self.ind], cv2.COLOR_BGR2RGB)
                    ax1.imshow(img)
                    ax1.set_title(f'Frame #{self.ind}')
                    (colors, vs) = processedResults[self.ind]
                    ax0.clear()
                    ax0.bar(colors, vs, color ='maroon', width = 0.3)
                    ax0.set_yticks(np.arange(min(vs), max(vs)+1, 1))
                    plt.draw()

            def prev(self, event):
                if self.ind > 0:
                    self.ind -= 1
                    img = cv2.cvtColor(processedImages[self.ind], cv2.COLOR_BGR2RGB)
                    ax1.imshow(img)
                    ax1.set_title(f'Frame #{self.ind}')
                    (colors, vs) = processedResults[self.ind]
                    ax0.clear()
                    ax0.bar(colors, vs, color ='maroon', width = 0.3)
                    ax0.set_yticks(np.arange(min(vs), max(vs)+1, 1))
                    plt.draw()
                elif self.ind == 0:
                    self.ind -= 1
                    ax1.imshow(concatedImage)
                    ax1.set_title('All frames')
                    (colors, vs) = getColorsAndCounts(colorKeyedObjectsDetectionConfigAndData)
                    ax0.clear()
                    ax0.bar(colors, vs, color ='maroon', width = 0.3)
                    ax0.set_yticks(np.arange(min(vs), max(vs)+1, 1))
                    plt.draw()

        callback = Index()
        axprev = fig.add_axes([0.125, 0.02, 0.1, 0.075])
        axnext = fig.add_axes([0.8, 0.02, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)


        plt.show()

    return 99

def offlineTask103() -> int:
    rgb_img = cv2.imread("balloon_seen_stationary2.png")
    colorAnalyzeImage(rgb_img, show_image=True)
    return 103

def main(argv):
    showCharts = 0
    if len(argv) < 1:
        print("Unkown flight #. Abort.")
    else:
        if len(argv) >= 2:
            showCharts = int(argv[1])
        flightNumber = int(argv[0])
        print("Taking flight #: ", flightNumber)       
        missionTaskBeginner(flight_number = flightNumber, show_charts = showCharts)

if __name__ == "__main__":
   main(sys.argv[1:])