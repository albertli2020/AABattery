import sys
import os
import threading
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

from djitellopy import Tello
from coloredObjectExtractor import ColoredObjectExtractor

default_command_delay_time = 0.1 #7
picture_first_frame_delay_time = 4.0

flights = [
    # 0
    {'name':'Offline', 'requiresDrone':False},
    # 1
    {'name':'Stationary', 'requiresDrone':True, 'flight_segments':[
        {'frameGrabDelay': 1, 'frameGrabInterval': 2, 'numFrameGrabIntervals':3, 'durationLimit': 10 } ] },
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
       {'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 80', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
       {'action':'cw 40', 'frameGrabDelay': 0, 'frameGrabInterval': 1, 'numFrameGrabIntervals':1,'durationLimit': 8},
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
       {'action':'land', 'frameGrabDelay': -1, 'durationLimit': 8} ] } ]
flight_number = 5

colorKeyedObjectsDetectionConfigAndData = {
    'red': {'count': 0, 'min_area':150},
    'blue': {'count': 0, 'min_area':150},
    'light_blue': {'count': 0, 'min_area':150},
    'yellow': {'count': 0, 'min_area':200},
    'orange': {'count': 0, 'min_area':150},
    'green': {'count': 0, 'min_area':150},
    'purple': {'count': 0, 'min_area':150},
    'light_green': {'count': 0, 'min_area':200} }

def colorAnalyzeImage(image, show_image=True, saveInputImageToFolder=None, saveAnalyzedImageToFolder=None):
    global colorKeyedObjectsDetectionConfigAndData
    if saveInputImageToFolder is None:
        pass
    else:
        of_path = os.path.join(saveInputImageToFolder, f'ba_{time.time()}.jpg')
        cv2.imwrite(of_path, image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hsv[:, :, 0] = hsv[:, :, 0] * 1.05
    #hsv[:, :, 1] = hsv[:, :, 1] * .95
    #hsv[:, :, 2] = hsv[:, :, 2] * .95
    #image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 

    for colorKey in colorKeyedObjectsDetectionConfigAndData.keys():
        cad = colorKeyedObjectsDetectionConfigAndData[colorKey]
        coe = ColoredObjectExtractor(colorKey)
        minArea = cad['min_area']
        objs = coe.extract(hsv, minArea, image)
        n = cad['count']
        nThisFrame = len(objs)
        if n < nThisFrame:
            cad['count'] = nThisFrame

    if saveAnalyzedImageToFolder is None:
        pass
    else:
        of_path = os.path.join(saveAnalyzedImageToFolder, f'aa_{time.time()}.jpg')
        cv2.imwrite(of_path, image)

    if show_image:
        cv2.imshow("color analysis result",image)
        cv2.waitKey()
    
    return image

def motionControl(tello:Tello, fn:int) -> int:
    # take a command from command queue and send to drone
    tello.takeoff()
    #time.sleep(default_command_delay_time)
    
    tello.move_up(20)
    #time.sleep(default_command_delay_time)
    h = tello.get_height()
    #time.sleep(default_command_delay_time)
    print("Hovering at height: ", h, "cm.")

    if fn == 0:
        panorama_full_counter_clockwise(tello)
    elif fn == 1:
        panorama_full_clockwise(tello)
    elif fn == 2:
        panorama_half_counter_clockwise(tello)
    elif fn == 3:
        panorama_half_clockwise(tello)
    elif fn == 4:
        panorama_move_forward(tello)
    
    #time.sleep(5)
    tello.land()
    #time.sleep(default_command_delay_time)
    return 0

def recordAndShowFrames(tello:Tello):
    global numObjectsDetectedForColor
    f = flights[flight_number]
    start_time = int(time.time())
    output_folder = os.path.join("../output/processed", f["name"])
    output_folder = os.path.join(f["name"], f'{start_time}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    frame_read = tello.get_frame_read()
    time.sleep(3)
    #h, w, _ = frame_read.frame.shape
    #v = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
    #v = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (w, h))
    #create two subplots
    img = frame_read.frame
    rgb_after = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #fig, ax = plt.subplots()
    #f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    plt.figure(figsize=(12, 9))
    #a0.axis("off")
    #a0.set(title="Waiting for streaming to start...")
    #colors = list(numObjectsDetectedForColor.keys())
    #vs = list(numObjectsDetectedForColor.values())
    #create two image plots
    plt.imshow(rgb_after)
    #a1.bar(vs, colors, color ='maroon', width = 0.4)
    #a1.set(xlabel="Max No. of colored objects", ylabel="Colors", title ="Objects detected per color")
    plt.ion()
    for fn in range(16):
        start_time = time.time()
        img = frame_read.frame
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #rgb_before = rgb_img.copy()
        #v.write(rgb_img)
        img = colorAnalyzeImage(rgb_img, show_image=False, saveAnalyzedImageToFolder=output_folder)
        rgb_after = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_after)
    
        #b = tello.get_battery()
        #plt.title(f'Frame Number: {fn+1}, battery level: {b}')
        plt.title(f'Frame Number: {fn+1}')
        #print("Battery level: ", b)
        delta_time = (1/5.0) - (time.time()-start_time)
        if delta_time > 0:
            plt.pause(delta_time)

    tello.streamoff()
    time.sleep(default_command_delay_time)

    plt.ioff() # due to infinite loop, this gets never called.
    plt.show()

    plt.figure(figsize=(12, 9))
    colors = list(numObjectsDetectedForColor.keys())
    vs = list(numObjectsDetectedForColor.values())
    plt.barh(colors, vs, color ='maroon', height = 0.4)
    plt.xlabel("Max No. of colored objects")
    plt.ylabel("Colors")
    plt.title("Objects detected per color")
    plt.show()



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
        #tello.emergency()
        #b = 1
        if b < 15:
            print("Battery too low!!! Abort!!!")
            imageBuffer.put(None)
            print("Aborted image frame aquirer.")
            return
        tello.streamon() # this is a synchronous/blocking call, no need to wait after this call before sending next command
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
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        imageBuffer.put(rgb_img)
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
                print(s['action'], ' has response of: ', r)
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
        # acquire image frames from other (streaming) sources???
        rgb_img = cv2.imread("balloon_seen_stationary2.png")
        imageBuffer.put(rgb_img)
        imageBuffer.put(None)
    print("Finished runing image frame aquirer.")

def processImageFrames(imageBuffer:Queue, fn:int):
    print('Running image frame processer...')
    # process image frames
    f = flights[flight_number]
    start_time = int(time.time())
    start_folder = os.path.join("../output", f['name'])
    if not os.path.exists(start_folder):
        os.makedirs(start_folder)
    start_folder = os.path.join(start_folder, f'{start_time}')
    if not os.path.exists(start_folder):
        os.makedirs(start_folder)
   
    unprocessedOutputFolder = os.path.join(start_folder,  'unprocessed')
    processedOutputFolder = os.path.join(start_folder, 'processed')
    if not os.path.exists(unprocessedOutputFolder):
        os.makedirs(unprocessedOutputFolder)
    if not os.path.exists(processedOutputFolder):
        os.makedirs(processedOutputFolder)    

    nFrames = 0
    while True:
        item = imageBuffer.get()
        if item is None:
            break
        img = colorAnalyzeImage(item, show_image=False, saveInputImageToFolder=unprocessedOutputFolder, saveAnalyzedImageToFolder=processedOutputFolder)
        #rgb_after = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # report
        nFrames += 1
        print("Processed ", nFrames, " frame(s) so far...")
    # all done
    print('Finished running image frame processer.')

def missionTaskBeginner() -> int:
    global flight_number
    ib = Queue()
    t1 = threading.Thread(target = processImageFrames, args=(ib, flight_number, ))
    t2 = threading.Thread(target = acquireImageFrames, args=(ib, flight_number, ))                        
    t1.start()
    t2.start()
    t2.join()
    ib.put(None)
    t1.join()

    return 99

def stationaryTask9() -> int:
    tello = Tello()
    tello.connect(retry_count=1) # this is a synchronous/blocking call, no need to wait after this call before sending next command
    # tello.set_video_direction(Tello.CAMERA_DOWNWARD)
    recordAndShowFrames(tello)
    return 9

def offlineTask103() -> int:
    rgb_img = cv2.imread("balloon_seen_stationary2.png")
    colorAnalyzeImage(rgb_img, show_image=True)
    return 103

def main(argv):
    if len(argv) < 1:
        tid = -1
        print("Unkown TID.")
    else:
        tid = int(argv[0])
        print("TID:", tid)
    
    if tid == 9:
        stationaryTask9()
    elif tid == 99:
        missionTaskBeginner()
    elif tid == 101:
        pass #offlineTask101()
    elif tid == 103:
        offlineTask103()
    elif tid == 104:
        pass #offlineTask104()    

if __name__ == "__main__":
   main(sys.argv[1:])