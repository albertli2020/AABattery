import os
import glob
import threading
import time
import cv2
import numpy as np
from queue import Queue

from djitellopy import Tello

default_command_delay_time = 0.1 #7
picture_first_frame_delay_time = 4.0

class AAImageFrameAcquirer(Tello):

    flights = [
    # 0
    {'name':'Offline', 'requiresDrone':False, 'type':-1, 'unprocessed':'Panorama-full-clockwise-45-90-90-90-45', 'initiallyAcquiredAt':'1700599630'},
    #{'name':'Offline', 'requiresDrone':False, 'unprocessed':'Stationary', 'initiallyAcquiredAt':'1700285669'},
    #{'name':'Offline', 'requiresDrone':False, 'unprocessed':'Speed75-Up-Back-Back-Forward', 'initiallyAcquiredAt':'1699923999'},
    #{'name':'Offline', 'requiresDrone':False, 'unprocessed':'Speed100-Up-CW-Back-Back-CW-Forward-CCW', 'initiallyAcquiredAt':'17001995
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
    {'name':'Speed100-Up-CW-Back-Back-CW-Forward-CCW', 'requiresDrone':True, 'type':0, 'flight_segments':[
        {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 15},
        {'action':'speed 90', 'frameGrabDelay': -1, 'durationLimit': 10},
        #{'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':4,'durationLimit': 8},
        #{'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':4,'durationLimit': 8},
        #{'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':4,'durationLimit': 8},
        #{'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':4,'durationLimit': 8},
        #{'action':'cw 40', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':4,'durationLimit': 8},
        {'action':'up 80','frameGrabDelay': -1, 'durationLimit': 20},
        {'action':'cw 30', 'frameGrabDelay': -1, 'durationLimit': 20},
        {'action':'back 180','frameGrabDelay': 22, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 20},
        {'action':'back 180','frameGrabDelay': 22, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 20},
        #{'action':'back 140', 'frameGrabDelay': 0, 'frameGrabInterval': 0.2, 'numFrameGrabIntervals':6,'durationLimit': 10},
        #{'action':'cw 15', 'frameGrabDelay': 0.1, 'frameGrabInterval': 1, 'numFrameGrabIntervals':2,'durationLimit': 10},
        #{'frameGrabDelay': 0.5, 'frameGrabInterval': 0.5, 'numFrameGrabIntervals':4, 'durationLimit': 3 },
        {'action':'down 20','frameGrabDelay': 22, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 20},
        {'action':'cw 60', 'frameGrabDelay':-1, 'durationLimit': 10},
        {'action':'forward 112', 'frameGrabDelay':-1, 'durationLimit': 20},
        #{'action':'forward 115', 'frameGrabDelay':-1, 'durationLimit': 20},
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
    {'name':'Panorama-full-clockwise-45-90-90-90-45', 'requiresDrone':True, 'type':0, 'flight_segments':[
        {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 20},
        {'action':'up 80','frameGrabDelay': -1, 'durationLimit': 20},
        {'action':'cw 45', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
        {'action':'cw 90', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
        {'action':'cw 90', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
        {'action':'cw 90', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
        
        {'action':'down 110','frameGrabDelay': -1, 'durationLimit': 20},
        {'action':'cw 45', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
        {'action':'cw 45', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
        {'action':'cw 90', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
        {'action':'cw 90', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
        {'action':'cw 90', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
        {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 20} ] },

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
        {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] },
    {'name':'Panorama-full-clockwise-60x6', 'requiresDrone':True, 'type':0, 'flight_segments':[
    {'action':'takeoff', 'frameGrabDelay': -1, 'durationLimit': 20},
    {'action':'up 70','frameGrabDelay': -1, 'durationLimit': 20},
    {'action':'cw 60', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
    {'action':'cw 60', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
    {'action':'cw 60', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
    {'action':'cw 60', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
    {'action':'cw 60', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
    {'action':'cw 60', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20},
    #{'action':'cw 60', 'frameGrabDelay': 25, 'extraHoverTime':1, 'durationLimit': 20}, 
    {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 20} ] }
    ]
    lowestBatteryLevelToFlyDrone = 25

    def __init__(self, image_buffer:Queue, flight_number:int):
        # When connecting the drone to a home/office AP, we would need to find
        # the drone's DHCP assigned IP address and provide it as the host argument.
        # tello = Tello(host='192.168.50.170', retry_count=0)
        super().__init__(retry_count=1)
        self.currentBatteryLevel = None        
        self.imageBuffer = image_buffer
        self.flightInfo = AAImageFrameAcquirer.flights[flight_number]
        self.t = None


    def start(self):
        t = threading.Thread(target = self.acquire, args=())
        self.t = t
        t.start()

    def end(self):
        if self.t is not None:
            self.t.join()
    
    def acquire(self):
        print("[AAImageFrameAquirer] Starts acquiring frames ...")
        image_buffer = self.imageBuffer

        if self.flightInfo['requiresDrone']:
            self.connect() # this is a synchronous/blocking call, no need to wait after this call before sending next command            
            self.currentBatteryLevel = self.get_battery() # this is getting an asynchronously updated drone state; no need to wait after this call
            print("[AAImageFrameAquirer] Drone battery level: ", self.currentBatteryLevel)
            if self.currentBatteryLevel < AAImageFrameAcquirer.lowestBatteryLevelToFlyDrone:
                print("[AAImageFrameAquirer] Battery too low!!! Aborting the attempt to fly the drone and acquire images!!!")
                image_buffer.put(None)
                return
            
            responses = self.get_own_udp_object()['responses'] #no need to wait because this API isn't tied to talking to the drone
            self.streamon() # this is a synchronous/blocking call, no need to wait after this call before sending next command
            self.emergency()
            frameReader = self.get_frame_read()
            #Tello class hard-code a background frame "receiver" at:
            #        self.frame = np.zeros([300, 400, 3], dtype=np.uint8)
            # but somehow we are geting image frames of shapes [720, 960, 3]...
            segments = self.flightInfo['flight_segments']
            abortFlight = False  
            for s in segments:
                if 'action' in s.keys():
                    hasAirAction = True
                    self.send_command_without_return(s['action'])
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
                    if frameGrabDelay >= duration:
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
                            img = frameReader.frame
                            rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #shouldnt we let the processing thread handle the conversion?
                            image_buffer.put(rgbImage)
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
                            print("[AAImageFrameAquirer] Aborting command '{}'. Did not receive a response after {} seconds".format(s['action'], delta_time))
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
                    print("[AAImageFrameAquirer]", s['action'], ' has response of: ', r, time.time())

                if grabFrameAtSegmentEnd:
                    extraHoverTime = -1
                    if 'extraHoverTime' in s.keys():
                        extraHoverTime = s['extraHoverTime']
                    if extraHoverTime > 0.01:
                        time.sleep(extraHoverTime)
                    img = frameReader.frame
                    rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image_buffer.put(rgbImage)
      
            image_buffer.put(None)
            if abortFlight:
                self.land()  # this is a synchronous/blocking call, no need to wait after this call before sending next command
            self.streamoff() # this is a synchronous/blocking call, no need to wait after this call before sending next command
        else:
            if 'unprocessed' in self.flightInfo.keys():
                startFolder = os.path.join('../output',self.flightInfo['unprocessed'])
                startFolder = os.path.join(startFolder,  self.flightInfo['initiallyAcquiredAt'])
                unprocessedOutputFolder = os.path.join(startFolder, 'unprocessed')
                if not os.path.exists(unprocessedOutputFolder):
                    print('"[AAImageFrameAquirer] Unprocessed image folder not found!')
                else:
                    fileList = glob.glob(os.path.join(unprocessedOutputFolder, '*.png'))  #jpg
                    for f in fileList:
                        rgbImage = cv2.imread(f)
                        image_buffer.put(rgbImage)
            else:                   
                # acquire image frames from other (streaming) sources???
                rgbImage = cv2.imread("balloon_seen_stationary2.png")
                image_buffer.put(rgbImage)
                image_buffer.put(None)
        print("[AAImageFrameAquirer] Finished acquiring image frames.")
